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
from dataclasses import dataclass
from pathlib import Path

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

from PIL import Image

from fotosort.quality import load_small

SYSTEM = (
    "You are a professional photo editor culling a travel and wildlife shoot for a family album. "
    "You judge photos on: subject clearly visible, sharp and not cut off by the frame; good exposure "
    "with faces not lost in shadow; pleasing composition and light; and a genuine moment. "
    "Near-identical frames from a burst count as one photo: pick only the best of them. "
    "You answer with JSON only."
)

PROMPT = (
    "These {n} photos were taken on {day} and show: {subject}. Where a photo has a '100% crop' after it, "
    "that is the same photo's main subject at full resolution: use it to judge focus and sharpness "
    "(the eye or face must be crisp), and to check that the subject is not looking away. "
    "Choose up to {k} keepers for a family travel album: technically good, as different from each "
    "other as possible (different pose, composition, moment or individual), and including atmospheric "
    "scenery or environmental shots, not only close-ups. Prefer animals that show their face or eye; a "
    "rear view or an animal looking away only if nothing better exists. Use the full budget of {k} when there are "
    "enough decent, distinct frames; drop a frame only when it is clearly worse than another keeper, "
    "soft, shows a tiny or cut-off subject, or is a near-duplicate. Unless every frame is unusable, "
    "always keep at least the single best one. "
    'Reply with JSON only, in this shape: {{"picks": [{{"photo": 3, "reason": "short reason"}}]}}'
)

FINAL_PROMPT = (
    "These {n} photos are the finalists from a larger set taken on {day} showing: {subject}. Each one "
    "was already judged good in its own batch. Now choose the final {k} for the album: the strongest, "
    "and as different from each other as possible. Where a photo has a '100% crop' after it, use it to "
    "judge sharpness. "
    'Reply with JSON only: {{"picks": [{{"photo": 3, "reason": "short reason"}}]}}'
)

RETRY_PROMPT = (
    "You kept none of these {n} photos of {subject} from {day}. That is acceptable only if every frame is "
    "unusable. Otherwise pick exactly the single best one, even if imperfect, and say what it does well. "
    'Reply with JSON only: {{"picks": [{{"photo": 3, "reason": "short reason"}}]}}'
)


@dataclass
class Verdict:
    picks: list[str]            # candidate ids in preference order
    reasons: dict[str, str]     # id -> reason
    error: str | None = None    # set when the judge could not run; caller falls back


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(path: Path, max_side: int = 1024) -> str:
    return _to_b64(load_small(str(path), max_side))


def subject_crop(path: Path, box: tuple[float, float, float, float], side: int = 512, margin: float = 0.25) -> str:
    """Native-resolution crop around the subject box (normalised coords), padded
    by `margin` of the box size, capped at `side` px. Shows whether the eye is sharp."""
    from PIL import ImageOps

    img = ImageOps.exif_transpose(Image.open(path))
    w, h = img.size
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
    if (x2 - x1) < side and (y2 - y1) < side:  # small subject: take a `side` window around it
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1 = max(0, cx - side / 2), max(0, cy - side / 2)
        x2, y2 = min(w, x1 + side), min(h, y1 + side)
    crop = img.crop((int(x1), int(y1), int(x2), int(y2))).convert("RGB")
    crop.thumbnail((side, side))
    return _to_b64(crop)


def parse_verdict(text: str, ids: list[str]) -> Verdict:
    """Extract {"picks": [{"photo": n, "reason": ...}]} from the model's reply (1-based)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return Verdict([], {}, error=f"no JSON in reply: {text[:120]!r}")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Verdict([], {}, error=f"bad JSON: {e}")
    picks, reasons = [], {}
    for item in data.get("picks", []):
        try:
            n = int(item.get("photo") if isinstance(item, dict) else item)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= len(ids) and ids[n - 1] not in picks:
            picks.append(ids[n - 1])
            reasons[ids[n - 1]] = str(item.get("reason", "")) if isinstance(item, dict) else ""
    return Verdict(picks, reasons)


class Judge:
    def __init__(self, provider: str = "openai", model: str | None = None, key_file: str | None = None,
                 detail: str = "high", hint: str = ""):
        if provider not in DEFAULT_MODELS:
            raise SystemExit(f"Unknown judge provider {provider!r}; use openai or anthropic")
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.detail = detail  # OpenAI image detail: "high" sees sharpness and faces, "low" is ~10x cheaper
        self.hint = hint.strip()  # extra guidance for this shoot, appended to every prompt
        self.usage = {"input": 0, "output": 0}
        key = read_key(provider, key_file)
        if provider == "openai":
            import openai  # slow import, keep local

            self.client = openai.OpenAI(api_key=key) if key else openai.OpenAI()
        else:
            import anthropic

            self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def judge(self, paths: list[Path], subject: str, day: str, k: int,
              boxes: dict[str, tuple | None] | None = None, final: bool = False) -> Verdict:
        ids, images = [], []
        for p in paths:
            try:
                frame = _jpeg_b64(p, 1024 if self.detail == "high" else 512)
            except OSError:  # deleted while we were running: judge the rest
                continue
            crop = None
            box = (boxes or {}).get(str(p))
            if box is not None:
                try:
                    crop = subject_crop(p, box)
                except Exception:
                    crop = None
            ids.append(str(p))
            images.append((frame, crop))
        if not ids:
            return Verdict([], {}, error="all files vanished")
        template = FINAL_PROMPT if final else PROMPT
        prompt = template.format(n=len(paths), day=day, subject=subject, k=k)
        if self.hint:
            prompt = f"About this shoot: {self.hint}\n\n{prompt}"
        verdict = self._judge(images, ids, prompt)
        if not final and not verdict.error and not verdict.picks and len(paths) >= 3:
            second = self._judge(images, ids, RETRY_PROMPT.format(n=len(paths), day=day, subject=subject))
            if not second.error and second.picks:
                verdict = Verdict(second.picks[:1], {i: "(second look) " + second.reasons.get(i, "") for i in second.picks[:1]})
        return verdict

    def _judge(self, images, ids, prompt) -> Verdict:
        try:
            text = self._ask_openai(images, prompt) if self.provider == "openai" else self._ask_anthropic(images, prompt)
        except Exception as e:  # auth, network, rate limit after SDK retries
            return Verdict([], {}, error=f"{type(e).__name__}: {e}")
        if text is None:
            return Verdict([], {}, error="model declined to judge this group")
        return parse_verdict(text, ids)

    def _ask_openai(self, images, prompt: str) -> str:
        content = []
        for i, (b64, crop) in enumerate(images, 1):
            content.append({"type": "text", "text": f"Photo {i}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": self.detail}})
            if crop:
                content.append({"type": "text", "text": f"Photo {i}, 100% crop of the subject"})
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop}", "detail": "low"}})
        content.append({"type": "text", "text": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_completion_tokens=4000,
        )
        if resp.usage:
            self.usage["input"] += resp.usage.prompt_tokens
            self.usage["output"] += resp.usage.completion_tokens
        return resp.choices[0].message.content or ""

    def _ask_anthropic(self, images, prompt: str) -> str | None:
        content = []
        for i, (b64, crop) in enumerate(images, 1):
            content.append({"type": "text", "text": f"Photo {i}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
            if crop:
                content.append({"type": "text", "text": f"Photo {i}, 100% crop of the subject"})
                content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": crop}})
        content.append({"type": "text", "text": prompt})
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM,
            messages=[{"role": "user", "content": content}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage["input"] += resp.usage.input_tokens
        self.usage["output"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return None
        return "".join(b.text for b in resp.content if b.type == "text")
