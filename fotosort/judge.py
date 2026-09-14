"""Optional final pass: a vision model looks at each group's shortlist and picks
the keepers the way a photo editor would.

Providers: "openai" (OPENAI_API_KEY, default model gpt-5.6-sol, OpenAI's
strongest vision model as of September 2026) or "anthropic"
(ANTHROPIC_API_KEY, default model claude-opus-5). A key can also be read from
a file with --key-file (a .env line like OPENAI_API_KEY=..., or a YAML line
like openai_api_key: ...)."""
from __future__ import annotations

import base64
import io
import json
import os
import re
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from fotosort.award import AWARD_PROMPT, AWARD_SYSTEM, CURATION_STAGE, NOMINATION_STAGE, AwardAssessment
from fotosort.editing import PRESETS
from fotosort.quality import load_small, open_full, subject_region

DEFAULT_MODELS = {"openai": "gpt-5.6-sol", "anthropic": "claude-opus-5"}
KEY_NAMES = {"openai": ("OPENAI_API_KEY", "openai_api_key"), "anthropic": ("ANTHROPIC_API_KEY", "anthropic_api_key")}


def read_key(provider: str, key_file: str | None) -> str | None:
    """Key from --key-file (first matching `NAME=value` / `name: value` line) or the environment."""
    names = KEY_NAMES[provider]
    if key_file:
        for line in Path(key_file).expanduser().read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*[:=]\s*(.+?)\s*$", line)
            if m and m.group(1).lower() in {n.lower() for n in names}:
                return m.group(2).strip().strip("'\"")
        raise SystemExit(f"No {names[0]} in {key_file}")
    return os.environ.get(names[0])

SYSTEM = (
    "You are a professional photo editor culling a travel and wildlife shoot for a family album. "
    "You judge photos on: subject clearly visible, sharp and not cut off by the frame; good exposure "
    "with faces not lost in shadow; pleasing composition and light; and a genuine moment. "
    "Near-identical frames from a burst count as one photo: pick only the best of them. "
    "Look for exceptional behaviour, interaction, expression, gesture, light and storytelling. "
    "A distinctive moment can outweigh a small technical flaw. Do not confuse background texture "
    "with subject focus, or reject intentional silhouettes, environmental compositions or a rear view "
    "that tells a compelling story. Subject labels are fallible hints, not restrictions on what to keep. "
    "Consider thirds or golden-ratio placement, central or symmetric compositions, leading lines, "
    "negative space, colour harmony and separation. Treat these as alternatives, never requirements; "
    "originality, narrative and emotional impact can outweigh geometric conformity. "
    "You answer with JSON only."
)

PROMPT = (
    "These {n} photos were taken on {day} and show: {subject}. Where a photo has a '100% crop' after it, "
    "that is the same photo's main subject at full resolution: use it to judge focus and sharpness "
    "when an eye or face is visible. Detail crops show only portions of the subject; use the overview "
    "to understand the complete composition. "
    "Choose up to {k} keepers for a family travel album: technically good, as different from each "
    "other as possible (different pose, composition, moment or individual), and including atmospheric "
    "scenery or environmental shots, not only close-ups. Preserve exceptional action, interactions, "
    "expressions and unusual light even if conventional beauty scores would miss them. "
    "Use the full budget of {k} when there are "
    "enough decent, distinct frames; drop a frame only when it is clearly worse than another keeper, "
    "unintentionally soft, or is a near-duplicate. A small subject in an effective environmental "
    "composition can be a keeper. Unless every frame is unusable, "
    "always keep at least the single best one. "
    "For each keeper also say whether developing its RAW file would clearly improve it "
    "(edit_benefit: low / medium / high, with a few words why: e.g. blown sky to recover, "
    "shadows to lift, noise to clean, or nothing to gain), and choose the one preset from "
    "this list that fits it best: " + ", ".join(PRESETS) + ". "
    'Reply with JSON only, in this shape: {{"picks": [{{"photo": 3, "reason": "short reason", '
    '"edit_benefit": "high", "edit_why": "few words", "preset": "Color Pop"}}]}}'
)

FINAL_PROMPT = (
    "These {n} photos are the finalists from a larger set taken on {day} showing: {subject}. Each one "
    "was already judged good in its own batch. Now choose up to {k} for the album: the strongest, "
    "and as different from each other as possible. Where a photo has a '100% crop' after it, use it to "
    "judge sharpness. The budget is an upper bound: keep only one of near-identical moments, but "
    "preserve distinct exceptional behaviour, expression, interaction and light. "
    "For each keeper also give edit_benefit (low / medium / high: would developing the "
    "RAW clearly improve it), edit_why (few words) and the best preset from: " + ", ".join(PRESETS) + ". "
    'Reply with JSON only: {{"picks": [{{"photo": 3, "reason": "short reason", '
    '"edit_benefit": "high", "edit_why": "few words", "preset": "Color Pop"}}]}}'
)

RETRY_PROMPT = (
    "You kept none of these {n} photos of {subject} from {day}. That is acceptable only if every frame is "
    "unusable. Otherwise pick exactly the single best one, even if imperfect, and say what it does well. "
    'Reply with JSON only: {{"picks": [{{"photo": 3, "reason": "short reason"}}]}}'
)


SUBJECT_PROMPT = (
    "Name the main subject of each of the {n} photos above with a short label of one to four words: the "
    "species or type of animal (for example 'cape fur seal', 'bottlenose dolphin', 'humpback whale', "
    "'kelp gull', 'shark'), or 'people', 'landscape', 'boat', 'fish', or 'spray, no animal' when nothing "
    "is there. Where one of these labels fits, use it exactly: {vocab}. Do not guess a rarer species than "
    "the photo supports. Reply with JSON only: {{\"subjects\": [{{\"photo\": 1, \"subject\": \"...\"}}, ...]}}."
)


@dataclass
class Verdict:
    picks: list[str]            # candidate ids in preference order
    reasons: dict[str, str]     # id -> reason
    error: str | None = None    # set when the judge could not run; caller falls back
    edits: dict[str, dict] | None = None  # id -> {"benefit": low|medium|high, "why": str, "preset": str}
    awards: dict[str, AwardAssessment] | None = None


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(path: Path, max_side: int = 1024) -> str:
    return _to_b64(load_small(str(path), max_side))


def _native_window(img: Image.Image, cx: float, cy: float, side: int) -> Image.Image:
    """A true pixel-for-pixel window, shifted inside the image at its borders."""
    w, h = img.size
    cw, ch = min(side, w), min(side, h)
    left = max(0, min(w - cw, round(cx - cw / 2)))
    top = max(0, min(h - ch, round(cy - ch / 2)))
    return img.crop((left, top, left + cw, top + ch)).convert("RGB")


def subject_crop(path: Path, box: tuple[float, float, float, float], side: int = 512) -> str:
    """A native-resolution central subject detail; never resize the pixels."""
    from PIL import ImageOps

    with open_full(str(path)) as original:
        img = ImageOps.exif_transpose(original)
        return _to_b64(_native_window(img, (box[0] + box[2]) * img.width / 2,
                                     (box[1] + box[3]) * img.height / 2, side))


def subject_crops(path: Path, box: tuple, side: int = 512) -> list[tuple[str, str]]:
    """Subject overview plus native details covering centre and possible head positions.

    There is no eye detector: label each window honestly so the judge only
    assesses eyes actually visible in it. RAW and JPEG share oriented decoding.
    """
    from PIL import ImageOps

    with open_full(str(path)) as original:
        img = ImageOps.exif_transpose(original)
        overview = subject_region(img, box).convert("RGB")
        overview.thumbnail((side, side))
        out = [("subject overview (resized)", _to_b64(overview))]
        w, h = img.size
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        cx = (x1 + x2) / 2
        positions = [("centre", cx, (y1 + y2) / 2)]
        if x2 - x1 > side:
            positions += [("upper left", x1 + side / 2, y1 + min(side, y2 - y1) / 2),
                          ("upper right", x2 - side / 2, y1 + min(side, y2 - y1) / 2)]
        elif y2 - y1 > side:
            positions += [("upper", cx, y1 + side / 2)]
        for label, x, y in positions:
            out.append((f"100% subject detail, {label}", _to_b64(_native_window(img, x, y, side))))
        return out


def parse_verdict(text: str, ids: list[str]) -> Verdict:
    """Extract {"picks": [{"photo": n, "reason": ...}]} from the model's reply (1-based)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return Verdict([], {}, error=f"no JSON in reply: {text[:120]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Verdict([], {}, error=f"bad JSON: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("picks"), list):
        return Verdict([], {}, error="JSON reply must contain a picks list")
    picks, reasons, edits, awards = [], {}, {}, {}
    for item in data.get("picks", []):
        try:
            n = int(item.get("photo") if isinstance(item, dict) else item)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(ids) and ids[n - 1] not in picks:
            picks.append(ids[n - 1])
            reasons[ids[n - 1]] = str(item.get("reason", "")) if isinstance(item, dict) else ""
            if isinstance(item, dict):
                assessment = AwardAssessment.parse(item)
                if assessment is not None:
                    awards[ids[n - 1]] = assessment
                benefit = str(item.get("edit_benefit", "")).lower().strip()
                preset = str(item.get("preset", "")).strip()
                preset = next((p for p in PRESETS if p.lower() == preset.lower()), "")
                if benefit in {"low", "medium", "high"} or preset:
                    edits[ids[n - 1]] = {"benefit": benefit if benefit in {"low", "medium", "high"} else "",
                                         "why": str(item.get("edit_why", "")).strip(), "preset": preset}
    return Verdict(picks, reasons, edits=edits or None, awards=awards or None)


def parse_subjects(text: str, ids: list[str]) -> dict[str, str]:
    """{path: subject} from the model's JSON; malformed entries are skipped."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {}
    out = {}
    for item in data.get("subjects", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        n, subject = item.get("photo"), item.get("subject")
        if isinstance(n, int) and 1 <= n <= len(ids) and isinstance(subject, str) and subject.strip():
            out[ids[n - 1]] = " ".join(subject.lower().split())[:60]
    return out


class Judge:
    def __init__(self, provider: str = "openai", model: str | None = None, key_file: str | None = None,
                 detail: str = "high", hint: str = "", base_url: str | None = None, crops: str = "single",
                 mode: str = "standard"):
        if provider not in DEFAULT_MODELS:
            raise SystemExit(f"Unknown judge provider {provider!r}; use openai or anthropic")
        self.provider = provider
        if mode not in {"standard", "award-roll"}:
            raise ValueError("mode must be standard or award-roll")
        self.mode = mode
        self.award_final = False
        self.system = AWARD_SYSTEM if mode == "award-roll" else SYSTEM
        # Nominations can contain 30 structured species/moment assessments.
        self.max_output_tokens = 10000 if mode == "award-roll" else 4000
        self.base_url = base_url  # an OpenAI-compatible server (Ollama, LM Studio, vLLM, MLX): local and free
        if base_url and provider != "openai":
            raise SystemExit("--judge-base-url works with the openai provider (OpenAI-compatible servers)")
        if base_url and not model:
            raise SystemExit("--judge-base-url needs --judge-model, e.g. --judge-model qwen3-vl:8b")
        self.model = model or DEFAULT_MODELS[provider]
        if crops not in {"single", "multi", "none"}:
            raise ValueError("crops must be single, multi or none")
        self.detail = detail
        self.crops = crops
        self.hint = hint.strip()  # extra guidance for this shoot, appended to every prompt
        self.usage = {"input": 0, "output": 0}
        self.sources: dict[str, Path] = {}
        self._images = OrderedDict()
        key = read_key(provider, key_file)
        if not key and base_url:
            key = "local"  # local servers ignore it, the SDK insists on one
        if not key:
            raise SystemExit(f"--judge needs an API key: set {KEY_NAMES[provider][0]} or pass --key-file")
        if provider == "openai":
            import openai  # slow import, keep local

            self.client = openai.OpenAI(api_key=key, base_url=base_url) if base_url else openai.OpenAI(api_key=key)
        else:
            import anthropic

            self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def judge(self, paths: list[Path], subject: str, day: str, k: int,
              boxes: dict[str, tuple | None] | None = None, final: bool = False) -> Verdict:
        ids, images = [], []
        for p in paths:
            source = self.sources.get(str(p), p)
            box = (boxes or {}).get(str(p))
            try:
                st = source.stat()
                key = (str(source), st.st_mtime_ns, st.st_size, tuple(box) if box is not None else None,
                       self.detail, self.crops)
                if key in self._images:
                    frame, crops = self._images.pop(key)
                else:
                    frame = _jpeg_b64(source, 1024 if self.detail == "high" else 512)
                    crops = []
                    if box is not None and self.crops != "none":
                        # The overview still reaches the judge if full decoding fails.
                        with suppress(Exception):
                            crops = (subject_crops(source, box) if self.crops == "multi" else
                                     [("100% subject detail, centre", subject_crop(source, box))])
                self._images[key] = (frame, crops)
                while len(self._images) > 128:
                    self._images.popitem(last=False)
            except OSError:
                continue
            ids.append(str(p))
            images.append((frame, crops))
        if not ids:
            return Verdict([], {}, error="all files vanished")
        template = FINAL_PROMPT if final else PROMPT
        prompt = template.format(n=len(ids), day=day, subject=subject, k=k)
        if self.mode == "award-roll":
            stage = CURATION_STAGE.format(target=k) if self.award_final else NOMINATION_STAGE
            prompt = AWARD_PROMPT.format(n=len(ids), k=k, stage=stage)
        if self.hint:
            prompt = f"About this shoot: {self.hint}\n\n{prompt}"
        verdict = self._judge(images, ids, prompt)
        if self.mode != "award-roll" and not final and not verdict.error and not verdict.picks and len(paths) >= 3:
            second = self._judge(images, ids, RETRY_PROMPT.format(n=len(ids), day=day, subject=subject))
            if not second.error and second.picks:
                best = second.picks[0]
                verdict = Verdict([best], {best: "(second look) " + second.reasons.get(best, "")})
        return verdict

    def name_subjects(self, paths: list[Path], vocab: list[str], chunk: int = 12) -> dict[str, str]:
        """Ask the model what each photo shows, in batches; returns {path: label}.
        Batches that fail are skipped (the caller keeps the local label)."""
        out: dict[str, str] = {}
        for start in range(0, len(paths), chunk):
            batch, ids, images = paths[start:start + chunk], [], []
            for p in batch:
                source = self.sources.get(str(p), p)
                try:
                    images.append((_jpeg_b64(source, 1024 if self.detail == "high" else 512), []))
                except OSError:
                    continue
                ids.append(str(p))
            if not ids:
                continue
            prompt = SUBJECT_PROMPT.format(n=len(ids), vocab="; ".join(vocab))
            if self.hint:
                prompt = f"About this shoot: {self.hint}\n\n{prompt}"
            try:
                ask = self._ask_openai if self.provider == "openai" else self._ask_anthropic
                text = ask(images, prompt)
            except Exception as e:  # noqa: BLE001 - a failed batch keeps its local labels
                print(f"Subject naming failed for {len(ids)} photos: {type(e).__name__}: {e}")
                continue
            out.update(parse_subjects(text or "", ids))
        return out

    def _judge(self, images, ids, prompt) -> Verdict:
        try:
            ask = self._ask_openai if self.provider == "openai" else self._ask_anthropic
            text = ask(images, prompt)
        except Exception as e:  # auth, network, rate limit after SDK retries
            return Verdict([], {}, error=f"{type(e).__name__}: {e}")
        if text is None:
            return Verdict([], {}, error="model declined to judge this group")
        verdict = parse_verdict(text, ids)
        if self.mode == "award-roll" and not verdict.error and any(
            i not in (verdict.awards or {}) for i in verdict.picks
        ):
            return Verdict([], {}, error="award-roll reply lacks valid wildlife/species/moment evidence")
        return verdict

    def _ask_openai(self, images, prompt: str) -> str:
        content = []
        for i, (b64, crops) in enumerate(images, 1):
            content.append({"type": "text", "text": f"Photo {i}"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": self.detail}})
            for label, crop in crops or []:
                content.append({"type": "text", "text": f"Photo {i}, {label}"})
                content.append({"type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{crop}", "detail": "low"}})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": content}]
        if self.base_url:
            # local servers: older parameter name, and not all of them accept response_format
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=self.max_output_tokens,
                    response_format={"type": "json_object"})
            except Exception as e:  # noqa: BLE001 - retry once without the JSON constraint
                if "response_format" not in str(e) and "json" not in str(e).lower():
                    raise
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=self.max_output_tokens)
        else:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, response_format={"type": "json_object"},
                max_completion_tokens=self.max_output_tokens)
        if resp.usage:
            self.usage["input"] += resp.usage.prompt_tokens
            self.usage["output"] += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""

    def _ask_anthropic(self, images, prompt: str) -> str | None:
        content = []
        for i, (b64, crops) in enumerate(images, 1):
            content.append({"type": "text", "text": f"Photo {i}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
            for label, crop in crops or []:
                content.append({"type": "text", "text": f"Photo {i}, {label}"})
                content.append({"type": "image",
                                "source": {"type": "base64", "media_type": "image/jpeg", "data": crop}})
        content.append({"type": "text", "text": prompt})
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=self.system,
            messages=[{"role": "user", "content": content}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage["input"] += resp.usage.input_tokens
        self.usage["output"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return None
        return "".join(b.text for b in resp.content if b.type == "text")
