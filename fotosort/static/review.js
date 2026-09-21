"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="review-token"]').content;
let data, active, reference, page = 0, busy = false, zoom = 0, center = {x: .5, y: .5};
let fitZoomFactor = null;
const pageSize = 48;
const exposure = p => p.shortlisted === "1" ? "Seen by judge" : p.shortlisted === "0" ? "Not shown to judge" : "Judge exposure unknown";
const imageURL = (id, size) => `/image/${id}?size=${size}&token=${encodeURIComponent(token)}`;
const notice = (text, error = false) => { $("notice").textContent = text; $("notice").classList.toggle("error", error); };
const byId = id => data?.photos.find(p => p.id === id);
const subject = p => p.judge_label || p.label || "Unlabelled";
const badge = p => p.manual_decision ? `Manual ${p.manual_decision}` : p.selected ? "Suggested pick" : "Alternative";
function element(tag, className, text) { const e = document.createElement(tag); e.className = className; if(text !== undefined) e.textContent = text; return e; }
async function request(route, payload) {
  const options = {headers: {"X-FotoSort-Token": token}};
  if(payload) { options.method = "POST"; options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify({...payload, revision: data.revision, report_version: data.report_version}); }
  const response = await fetch(route, options); const result = await response.json();
  if(!response.ok) throw Error(result.error || "Request failed");
  return result;
}
function options(id, values, all) {
  const current = $(id).value; $(id).replaceChildren(new Option(all, ""));
  [...new Set(values.filter(Boolean))].sort().forEach(v => $(id).add(new Option(v, v)));
  if([...$(id).options].some(o => o.value === current)) $(id).value = current;
}
function filters() {
  options("day", data.photos.map(p => p.day), "All days");
  options("subject", data.photos.map(subject), "All subjects");
  const scoped = data.photos.filter(p => (!$("day").value || p.day === $("day").value) && (!$("subject").value || subject(p) === $("subject").value));
  options("scene", scoped.map(p => p.scene), "All scenes");
  options("moment", scoped.filter(p => !$("scene").value || p.scene === $("scene").value).map(p => p.moment), "All moments");
}
function filtered() {
  return data.photos.filter(p => (!$("day").value || p.day === $("day").value) && (!$("subject").value || subject(p) === $("subject").value) && (!$("scene").value || p.scene === $("scene").value) && (!$("moment").value || p.moment === $("moment").value) && ({all:true, kept:p.selected, unseen:p.shortlisted === "0", manual:!!p.manual_decision, attention:p.status !== "available" || p.review_status === "stale decision"}[$("show").value]));
}
function setData(result) {
  data = result; data.photos.sort((a,b) => (a.datetime || a.day).localeCompare(b.datetime || b.day) || a.relative_file.localeCompare(b.relative_file));
  $("collection").textContent = data.collection;
  $("kept").textContent = data.photos.filter(p => p.selected).length;
  $("manual").textContent = data.photos.filter(p => p.manual_decision).length;
  if(!byId(reference)) reference = data.photos.find(p => p.selected)?.id || data.photos[0]?.id;
  if(!byId(active)) active = data.photos.find(p => p.id !== reference)?.id || reference;
  filters(); render();
}
function controls() {
  const p = byId(active), r = byId(reference), available = p?.status === "available";
  for(const id of ["keep", "reject", "pin"]) $(id).disabled = busy || !available;
  $("clear").disabled = busy || !p || (!p.manual_decision && p.review_status !== "stale decision");
  for(const id of ["replace", "prefer", "alternatives"]) $(id).disabled = busy || !available || r?.status !== "available" || active === reference;
  $("undo").disabled = busy || !data?.can_undo;
  $("export").disabled = busy || !data;
  $("reload").disabled = busy;
}
function panel(which, p) {
  const image = $(`${which}-image`), error = $(`${which}-error`);
  $(`${which}-name`).textContent = p?.relative_file || "No frame selected";
  $(`${which}-badge`).textContent = p ? badge(p) : "";
  $(`${which}-meta`).textContent = p ? `${subject(p)} · ${p.datetime || p.day || "Unknown date"} · ${exposure(p)}` : "";
  $(`${which}-reason`).textContent = p ? [p.manual_reason, p.source_note, p.status !== "available" ? p.status : "", `Automatic: ${p.auto_decision || p.judge_reason || "No reason recorded"}`].filter(Boolean).join(" · ") : "";
  if(!p) { image.removeAttribute("src"); error.textContent = "No frame selected"; return; }
  const size = zoom ? "full" : "preview", url = imageURL(p.id, size);
  if(image.getAttribute("src") !== url) { error.textContent = "Loading image…"; image.style.visibility = "hidden"; image.alt = `${which === "reference" ? "Reference" : "Candidate"}: ${p.relative_file}`; image.src = url; }
}
function arrange() {
  ["reference", "candidate"].forEach(which => {
    const img = $(`${which}-image`), box = $(`${which}-viewport`);
    if(!img.naturalWidth) return;
    const scale = zoom || Math.min(box.clientWidth/img.naturalWidth, box.clientHeight/img.naturalHeight);
    const w = img.naturalWidth*scale, h = img.naturalHeight*scale;
    img.style.width = `${w}px`; img.style.height = `${h}px`;
    img.style.left = `${box.clientWidth/2 - w*center.x}px`; img.style.top = `${box.clientHeight/2 - h*center.y}px`;
  });
  $("fit").classList.toggle("active", !zoom); $("native").classList.toggle("active", zoom === 1);
}
function render() {
  const photos = filtered(); page = Math.min(page, Math.max(0, Math.ceil(photos.length/pageSize)-1));
  if(!photos.some(p => p.id === active)) active = photos[0]?.id;
  panel("reference", byId(reference)); panel("candidate", byId(active));
  const grid = $("filmstrip"); grid.replaceChildren();
  photos.slice(page*pageSize, (page+1)*pageSize).forEach(p => {
    const button = element("button", `thumb${p.id === active ? " current" : ""}${p.id === reference ? " pinned" : ""}${p.status !== "available" ? " unavailable" : ""}`);
    button.setAttribute("aria-label", `${p.relative_file}, ${badge(p)}, ${exposure(p).toLowerCase()}`);
    button.setAttribute("aria-pressed", String(p.id === active));
    const img = element("img", ""); img.loading = "lazy"; img.alt = ""; img.src = imageURL(p.id, "thumb");
    img.addEventListener("error", () => { img.alt = "Image unavailable"; });
    button.append(img, element("span", "thumb-name", p.relative_file.split("/").pop()), element("span", "thumb-status", p.status !== "available" ? p.status : `${p.manual_decision ? "Manual · " : ""}${p.shortlisted === "1" ? "Judged" : p.shortlisted === "0" ? "Not judged" : "Exposure unknown"}`));
    if(p.selected) button.append(element("span", "pick-indicator", "✓"));
    button.addEventListener("click", () => choose(p.id)); grid.append(button);
  });
  $("count").textContent = `${photos.length} frames`; $("empty").hidden = !!photos.length;
  $("page").textContent = photos.length ? `${page*pageSize+1}–${Math.min((page+1)*pageSize,photos.length)} / ${photos.length}` : "0 frames";
  $("previous").disabled = page === 0; $("next").disabled = (page+1)*pageSize >= photos.length;
  controls(); arrange();
}
function choose(id) { active = id; $("note").value = byId(id)?.manual_reason || ""; render(); }
function navigate(delta) { const items = filtered(), index = items.findIndex(p => p.id === active); const next = items[Math.max(0,Math.min(items.length-1,index+delta))]; if(next) { page = Math.floor(items.indexOf(next)/pageSize); choose(next.id); } }
async function change(payload, message) {
  if(busy || !data) return; busy = true; controls();
  try { setData(await request("/api/change", {...payload, note: $("note").value})); notice(`${message} · Saved revision ${data.revision}`); }
  catch(error) { notice(`${error.message}. Use Reload to refresh the page.`, true); }
  finally { busy = false; controls(); }
}
async function reload() { try { setData(await request("/api/state")); notice(`Review ready · ${data.photos.filter(p=>p.shortlisted === "0").length} frames were not shown to the judge. Choices save automatically.`); } catch(error) { notice(error.message, true); } }
$("keep").onclick = () => change({changes:{[active]:"keep"}}, "Frame kept");
$("reject").onclick = () => change({changes:{[active]:"reject"}}, "Frame rejected");
$("clear").onclick = () => change({changes:{[active]:"clear"}}, "Automatic recommendation restored");
$("pin").onclick = () => { reference = active; render(); };
$("replace").onclick = () => change({changes:{[reference]:"reject",[active]:"keep"}, preference:[active,reference]}, "Pick replaced and preference recorded");
$("prefer").onclick = () => change({changes:{}, preference:[active,reference]}, "Preference recorded; picks unchanged");
$("alternatives").onclick = () => change({changes:{}, alternatives:[`pair-${[active,reference].sort().map(x=>x.slice(0,16)).join("-")}`, [reference,active]]}, "Acceptable alternatives recorded; picks unchanged");
$("undo").onclick = async () => { if(busy) return; busy = true; controls(); try { setData(await request("/api/undo",{})); notice(`Last review edit undone · Revision ${data.revision}`); } catch(error) { notice(error.message,true); } finally { busy = false; controls(); } };
$("export").onclick = async () => { busy = true; controls(); try { const result = await request("/api/export",{}); notice(`Exported ${result.output}. Apply this report with --apply-report --report ${result.output} --copy.`); } catch(error) { notice(error.message,true); } finally { busy = false; controls(); } };
$("reload").onclick = reload;
["day","subject","scene","moment","show"].forEach(id => $(id).onchange = () => { page = 0; filters(); render(); });
$("reset-filters").onclick = () => { ["day","subject","scene","moment"].forEach(id=>$(id).value=""); $("show").value="all"; page=0; filters(); render(); };
$("previous").onclick = () => { page--; render(); }; $("next").onclick = () => { page++; render(); };
function setZoom(value) { fitZoomFactor = null; zoom = value; if(!zoom) center={x:.5,y:.5}; panel("reference",byId(reference)); panel("candidate",byId(active)); arrange(); }
$("fit").onclick = () => setZoom(0); $("native").onclick = () => setZoom(1);
function stepZoom(factor) {
  if(!zoom) { setZoom(1); fitZoomFactor = factor; return; }
  setZoom(Math.min(4,Math.max(.05,zoom*factor)));
}
$("zoom-in").onclick = () => stepZoom(1.4); $("zoom-out").onclick = () => stepZoom(1/1.4);
["reference","candidate"].forEach(which => {
  const img=$(`${which}-image`), viewport=$(`${which}-viewport`); let drag;
  img.onload = () => { if(which === "candidate" && fitZoomFactor) { zoom = Math.min(4,Math.max(.05,Math.min(viewport.clientWidth/img.naturalWidth,viewport.clientHeight/img.naturalHeight)*fitZoomFactor)); fitZoomFactor = null; } img.style.visibility="visible"; $(`${which}-error`).textContent=""; arrange(); };
  img.onerror = () => { img.style.visibility="hidden"; $(`${which}-error`).textContent="Image unavailable or changed. Reload, or rerun analysis if the source changed."; };
  viewport.onpointerdown = event => { if(!zoom) return; drag={x:event.clientX,y:event.clientY,c:{...center}}; viewport.setPointerCapture(event.pointerId); };
  viewport.onpointermove = event => { if(!drag) return; center={x:Math.max(0,Math.min(1,drag.c.x-(event.clientX-drag.x)/(img.naturalWidth*zoom))),y:Math.max(0,Math.min(1,drag.c.y-(event.clientY-drag.y)/(img.naturalHeight*zoom)))}; arrange(); };
  viewport.onpointerup = viewport.onpointercancel = () => { drag=null; };
  viewport.onkeydown = event => { if(!zoom || !["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key)) return; event.preventDefault(); event.stopPropagation(); if(event.key==="ArrowLeft") center.x-=.03; if(event.key==="ArrowRight") center.x+=.03; if(event.key==="ArrowUp") center.y-=.03; if(event.key==="ArrowDown") center.y+=.03; center.x=Math.max(0,Math.min(1,center.x)); center.y=Math.max(0,Math.min(1,center.y)); arrange(); };
});
document.addEventListener("keydown", event => {
  if(!data || busy || event.ctrlKey || event.metaKey || event.altKey || ["INPUT","TEXTAREA","SELECT"].includes(event.target.tagName)) return;
  const key=event.key.toLowerCase(); const buttons={k:"keep",x:"reject",c:"clear",p:"pin",z:"undo"};
  if(buttons[key] && !$(buttons[key]).disabled) { event.preventDefault(); $(buttons[key]).click(); }
  if(key==="arrowleft" || key==="arrowright") { event.preventDefault(); navigate(key==="arrowleft"?-1:1); }
});
window.addEventListener("resize",arrange);
reload();
