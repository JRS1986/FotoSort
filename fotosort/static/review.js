"use strict";

const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="review-token"]').content;
const pageSize = 48;
const filterIds = ["day", "subject", "scene", "moment"];
const panels = ["reference", "candidate"];

let data;
let active;
let reference;
let noteFor; // the photo whose saved note the note field currently shows
let page = 0;
let busy = false;
let zoom = 0;
let center = { x: 0.5, y: 0.5 };
let fitZoomFactor = null;
let renderedPage = "";

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const byId = (id) => data?.photos.find((p) => p.id === id);
const subject = (p) => p.judge_label || p.label || "Unlabelled";
const imageURL = (id, size) => `/image/${id}?size=${size}&token=${encodeURIComponent(token)}`;
const needsAttention = (p) => p.status !== "available" || p.review_status !== "available";

function exposure(p) {
  if (p.shortlisted === "1") return "Seen by judge";
  if (p.shortlisted === "0") return "Not shown to judge";
  return "Judge exposure unknown";
}

function badge(p) {
  if (p.manual_decision) return `Manual ${p.manual_decision}`;
  return p.selected ? "Suggested pick" : "Alternative";
}

function notice(text, error = false) {
  $("notice").textContent = text;
  $("notice").classList.toggle("error", error);
}

function element(tag, className, text) {
  const e = document.createElement(tag);
  e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function request(route, payload) {
  const options = { headers: { "X-FotoSort-Token": token } };
  if (payload) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify({
      ...payload,
      revision: data.revision,
      report_version: data.report_version,
    });
  }
  const response = await fetch(route, options);
  const result = await response.json();
  if (!response.ok) throw Error(result.error || "Request failed");
  return result;
}

function options(id, values, all) {
  const select = $(id);
  const current = select.value;
  select.replaceChildren(new Option(all, ""));
  [...new Set(values.filter(Boolean))].sort().forEach((v) => select.add(new Option(v, v)));
  if ([...select.options].some((o) => o.value === current)) select.value = current;
}

function filters() {
  options("day", data.photos.map((p) => p.day), "All days");
  options("subject", data.photos.map(subject), "All subjects");
  const scoped = data.photos.filter(
    (p) => (!$("day").value || p.day === $("day").value) && (!$("subject").value || subject(p) === $("subject").value),
  );
  options("scene", scoped.map((p) => p.scene), "All scenes");
  const inScene = scoped.filter((p) => !$("scene").value || p.scene === $("scene").value);
  options("moment", inScene.map((p) => p.moment), "All moments");
}

function filtered() {
  const shown = {
    all: () => true,
    kept: (p) => p.selected,
    unseen: (p) => p.shortlisted === "0",
    manual: (p) => !!p.manual_decision,
    attention: needsAttention,
  }[$("show").value];
  return data.photos.filter(
    (p) =>
      (!$("day").value || p.day === $("day").value) &&
      (!$("subject").value || subject(p) === $("subject").value) &&
      (!$("scene").value || p.scene === $("scene").value) &&
      (!$("moment").value || p.moment === $("moment").value) &&
      shown(p),
  );
}

function setData(result) {
  data = result;
  data.photos.sort(
    (a, b) => (a.datetime || a.day).localeCompare(b.datetime || b.day) || a.relative_file.localeCompare(b.relative_file),
  );
  $("collection").textContent = data.collection;
  $("kept").textContent = data.photos.filter((p) => p.selected).length;
  $("manual").textContent = data.photos.filter((p) => p.manual_decision).length;
  if (!byId(reference)) reference = data.photos.find((p) => p.selected)?.id || data.photos[0]?.id;
  if (!byId(active)) active = data.photos.find((p) => p.id !== reference)?.id || reference;
  noteFor = undefined; // saved notes may have changed: show the stored note again
  filters();
  render();
}

// The note field always belongs to the active frame. Whenever the active frame changes, whether by a
// click, a filter or a reload, the field shows that frame's saved note, so a note typed for one photo
// is never saved on another and an existing note is never replaced by an empty field.
function syncNote() {
  if (noteFor === active) return;
  noteFor = active;
  $("note").value = byId(active)?.manual_reason || "";
}

function controls() {
  const p = byId(active);
  const r = byId(reference);
  const available = p?.status === "available";
  for (const id of ["keep", "reject", "pin"]) $(id).disabled = busy || !available;
  $("clear").disabled = busy || !p || !(p.manual_decision || p.clearable);
  for (const id of ["replace", "prefer", "alternatives"]) {
    $(id).disabled = busy || !available || r?.status !== "available" || active === reference;
  }
  $("undo").disabled = busy || !data?.can_undo;
  $("export").disabled = busy || !data;
  $("reload").disabled = busy;
}

function panel(which, p) {
  const image = $(`${which}-image`);
  const error = $(`${which}-error`);
  $(`${which}-name`).textContent = p?.relative_file || "No frame selected";
  $(`${which}-badge`).textContent = p ? badge(p) : "";
  $(`${which}-meta`).textContent = p ? `${subject(p)} · ${p.datetime || p.day || "Unknown date"} · ${exposure(p)}` : "";
  $(`${which}-reason`).textContent = p
    ? [
        p.manual_reason,
        p.source_note,
        needsAttention(p) ? p.review_status : "",
        `Automatic: ${p.auto_decision || p.judge_reason || "No reason recorded"}`,
      ]
        .filter(Boolean)
        .join(" · ")
    : "";
  if (!p) {
    image.removeAttribute("src");
    error.textContent = "No frame selected";
    return;
  }
  const url = imageURL(p.id, zoom ? "full" : "preview");
  if (image.getAttribute("src") !== url) {
    error.textContent = "Loading image…";
    image.style.visibility = "hidden";
    image.alt = `${which === "reference" ? "Reference" : "Candidate"}: ${p.relative_file}`;
    image.src = url;
  }
}

function arrange() {
  panels.forEach((which) => {
    const img = $(`${which}-image`);
    const box = $(`${which}-viewport`);
    if (!img.naturalWidth) return;
    const scale = zoom || Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight);
    const w = img.naturalWidth * scale;
    const h = img.naturalHeight * scale;
    img.style.width = `${w}px`;
    img.style.height = `${h}px`;
    img.style.left = `${box.clientWidth / 2 - w * center.x}px`;
    img.style.top = `${box.clientHeight / 2 - h * center.y}px`;
  });
  $("fit").classList.toggle("active", !zoom);
  $("native").classList.toggle("active", zoom === 1);
}

function thumbnailStatus(p) {
  if (p.status !== "available") return p.status;
  const judged = p.shortlisted === "1" ? "Judged" : p.shortlisted === "0" ? "Not judged" : "Exposure unknown";
  return `${p.manual_decision ? "Manual · " : ""}${judged}`;
}

function createThumbnail(p) {
  const button = element("button", "thumb");
  button.dataset.id = p.id;
  const img = element("img", "");
  img.loading = "lazy";
  img.alt = "";
  img.src = imageURL(p.id, "thumb");
  img.addEventListener("error", () => {
    img.alt = "Image unavailable";
  });
  button.append(
    img,
    element("span", "thumb-name", p.relative_file.split("/").pop()),
    element("span", "thumb-status"),
    element("span", "pick-indicator", "✓"),
  );
  button.addEventListener("click", () => choose(p.id));
  return button;
}

function updateThumbnail(button, p) {
  button.classList.toggle("current", p.id === active);
  button.classList.toggle("pinned", p.id === reference);
  button.classList.toggle("unavailable", p.status !== "available");
  button.setAttribute("aria-label", `${p.relative_file}, ${badge(p)}, ${exposure(p).toLowerCase()}`);
  button.setAttribute("aria-pressed", String(p.id === active));
  button.querySelector(".thumb-status").textContent = thumbnailStatus(p);
  button.querySelector(".pick-indicator").hidden = !p.selected;
}

// Thumbnails are created once per page of results and then only updated, so selecting a frame or
// saving a choice does not request every image again.
function renderFilmstrip(photos) {
  const grid = $("filmstrip");
  const visible = photos.slice(page * pageSize, (page + 1) * pageSize);
  const key = visible.map((p) => p.id).join(",");
  if (key !== renderedPage) {
    grid.replaceChildren(...visible.map(createThumbnail));
    renderedPage = key;
  }
  [...grid.children].forEach((button, index) => updateThumbnail(button, visible[index]));
}

function render() {
  const photos = filtered();
  page = clamp(page, 0, Math.max(0, Math.ceil(photos.length / pageSize) - 1));
  if (!photos.some((p) => p.id === active)) active = photos[0]?.id;
  syncNote();
  panel("reference", byId(reference));
  panel("candidate", byId(active));
  renderFilmstrip(photos);
  const first = page * pageSize + 1;
  const last = Math.min((page + 1) * pageSize, photos.length);
  $("count").textContent = `${photos.length} frames`;
  $("empty").hidden = !!photos.length;
  $("page").textContent = photos.length ? `${first}–${last} / ${photos.length}` : "0 frames";
  $("previous").disabled = page === 0;
  $("next").disabled = (page + 1) * pageSize >= photos.length;
  controls();
  arrange();
}

function choose(id) {
  active = id;
  render();
}

function navigate(delta) {
  const items = filtered();
  const index = items.findIndex((p) => p.id === active);
  const next = items[clamp(index + delta, 0, items.length - 1)];
  if (!next) return;
  page = Math.floor(items.indexOf(next) / pageSize);
  choose(next.id);
}

async function submit(work) {
  if (busy || !data) return;
  busy = true;
  controls();
  try {
    await work();
  } catch (error) {
    notice(`${error.message}. Use Reload to refresh the page.`, true);
  } finally {
    busy = false;
    controls();
  }
}

function change(payload, message) {
  return submit(async () => {
    setData(await request("/api/change", { ...payload, note: $("note").value }));
    notice(`${message} · Saved revision ${data.revision}`);
  });
}

async function reload() {
  try {
    setData(await request("/api/state"));
    const unseen = data.photos.filter((p) => p.shortlisted === "0").length;
    notice(`Review ready · ${unseen} frames were not shown to the judge. Choices save automatically.`);
  } catch (error) {
    notice(error.message, true);
  }
}

function pairName() {
  const ids = [active, reference].sort().map((id) => id.slice(0, 16));
  return `pair-${ids.join("-")}`;
}

$("keep").onclick = () => change({ changes: { [active]: "keep" } }, "Frame kept");
$("reject").onclick = () => change({ changes: { [active]: "reject" } }, "Frame rejected");
$("clear").onclick = () => change({ changes: { [active]: "clear" } }, "Automatic recommendation restored");
$("pin").onclick = () => {
  reference = active;
  render();
};
$("replace").onclick = () =>
  change(
    { changes: { [reference]: "reject", [active]: "keep" }, preference: [active, reference] },
    "Pick replaced and preference recorded",
  );
$("prefer").onclick = () =>
  change({ changes: {}, preference: [active, reference] }, "Preference recorded; picks unchanged");
$("alternatives").onclick = () =>
  change(
    { changes: {}, alternatives: [pairName(), [reference, active]] },
    "Acceptable alternatives recorded; picks unchanged",
  );
$("undo").onclick = () =>
  submit(async () => {
    setData(await request("/api/undo", {}));
    notice(`Last review edit undone · Revision ${data.revision}`);
  });
$("export").onclick = () =>
  submit(async () => {
    const result = await request("/api/export", {});
    notice(`Exported ${result.output}. Apply this report with --apply-report --report ${result.output} --copy.`);
  });
$("reload").onclick = reload;

function applyFilters() {
  page = 0;
  filters();
  render();
}

[...filterIds, "show"].forEach((id) => {
  $(id).onchange = applyFilters;
});
$("reset-filters").onclick = () => {
  filterIds.forEach((id) => {
    $(id).value = "";
  });
  $("show").value = "all";
  applyFilters();
};
$("previous").onclick = () => {
  page--;
  render();
};
$("next").onclick = () => {
  page++;
  render();
};

function setZoom(value) {
  fitZoomFactor = null;
  zoom = value;
  if (!zoom) center = { x: 0.5, y: 0.5 };
  panel("reference", byId(reference));
  panel("candidate", byId(active));
  arrange();
}

function stepZoom(factor) {
  if (!zoom) {
    // Load the full image first; its size decides the zoom level once it arrives.
    setZoom(1);
    fitZoomFactor = factor;
    return;
  }
  setZoom(clamp(zoom * factor, 0.05, 4));
}

$("fit").onclick = () => setZoom(0);
$("native").onclick = () => setZoom(1);
$("zoom-in").onclick = () => stepZoom(1.4);
$("zoom-out").onclick = () => stepZoom(1 / 1.4);

const panKeys = { ArrowLeft: [-0.03, 0], ArrowRight: [0.03, 0], ArrowUp: [0, -0.03], ArrowDown: [0, 0.03] };

panels.forEach((which) => {
  const img = $(`${which}-image`);
  const viewport = $(`${which}-viewport`);
  let drag;
  img.onload = () => {
    if (which === "candidate" && fitZoomFactor) {
      const fit = Math.min(viewport.clientWidth / img.naturalWidth, viewport.clientHeight / img.naturalHeight);
      zoom = clamp(fit * fitZoomFactor, 0.05, 4);
      fitZoomFactor = null;
    }
    img.style.visibility = "visible";
    $(`${which}-error`).textContent = "";
    arrange();
  };
  img.onerror = () => {
    img.style.visibility = "hidden";
    $(`${which}-error`).textContent =
      "Image unavailable or changed. Reload, or rerun analysis if the source changed.";
  };
  viewport.onpointerdown = (event) => {
    if (!zoom) return;
    drag = { x: event.clientX, y: event.clientY, c: { ...center } };
    viewport.setPointerCapture(event.pointerId);
  };
  viewport.onpointermove = (event) => {
    if (!drag) return;
    center = {
      x: clamp(drag.c.x - (event.clientX - drag.x) / (img.naturalWidth * zoom), 0, 1),
      y: clamp(drag.c.y - (event.clientY - drag.y) / (img.naturalHeight * zoom), 0, 1),
    };
    arrange();
  };
  viewport.onpointerup = viewport.onpointercancel = () => {
    drag = null;
  };
  viewport.onkeydown = (event) => {
    if (!zoom || !panKeys[event.key]) return;
    event.preventDefault();
    event.stopPropagation();
    const [dx, dy] = panKeys[event.key];
    center = { x: clamp(center.x + dx, 0, 1), y: clamp(center.y + dy, 0, 1) };
    arrange();
  };
});

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
  if (!data || busy || event.ctrlKey || event.metaKey || event.altKey || typing) return;
  const key = event.key.toLowerCase();
  const buttons = { k: "keep", x: "reject", c: "clear", p: "pin", z: "undo" };
  if (buttons[key] && !$(buttons[key]).disabled) {
    event.preventDefault();
    $(buttons[key]).click();
  }
  if (key === "arrowleft" || key === "arrowright") {
    event.preventDefault();
    navigate(key === "arrowleft" ? -1 : 1);
  }
});

window.addEventListener("resize", arrange);
reload();
