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
    "These {n} photos were taken on {day} and show: {subject}. Choose up to {k} keepers. "
    "They must be technically good and as different from each other as possible (different pose, "
    "composition, moment or individual). Choose fewer than {k} if the rest are weak, blurry, "
    "show only a tiny or cut-off subject, or are near-duplicates of a better frame. "
    'Reply with JSON only, in this shape: {{"picks": [{{"photo": 3, "reason": "short reason"}}]}}'
)


@dataclass
class Verdict:
    picks: list[str]            # candidate ids in preference order
    reasons: dict[str, str]     # id -> reason
    error: str | None = None    # set when the judge could not run; caller falls back


def _jpeg_b64(path: Path, max_side: int = 1024) -> str:
    img = load_small(str(path), max_side)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


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
                 detail: str = "high"):
        if provider not in DEFAULT_MODELS:
            raise SystemExit(f"Unknown judge provider {provider!r}; use openai or anthropic")
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.detail = detail  # OpenAI image detail: "high" sees sharpness and faces, "low" is ~10x cheaper
        self.usage = {"input": 0, "output": 0}
        key = read_key(provider, key_file)
        if provider == "openai":
            import openai  # slow import, keep local

            self.client = openai.OpenAI(api_key=key) if key else openai.OpenAI()
        else:
            import anthropic

            self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def judge(self, paths: list[Path], subject: str, day: str, k: int) -> Verdict:
        ids = [str(p) for p in paths]
        images = [_jpeg_b64(p, 1024 if self.detail == "high" else 512) for p in paths]
        prompt = PROMPT.format(n=len(paths), day=day, subject=subject, k=k)
        try:
            text = self._ask_openai(images, prompt) if self.provider == "openai" else self._ask_anthropic(images, prompt)
        except Exception as e:  # auth, network, rate limit after SDK retries
            return Verdict([], {}, error=f"{type(e).__name__}: {e}")
        if text is None:
            return Verdict([], {}, error="model declined to judge this group")
        return parse_verdict(text, ids)

    def _ask_openai(self, images: list[str], prompt: str) -> str:
        content = []
        for i, b64 in enumerate(images, 1):
            content.append({"type": "text", "text": f"Photo {i}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": self.detail}})
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

    def _ask_anthropic(self, images: list[str], prompt: str) -> str | None:
        content = []
        for i, b64 in enumerate(images, 1):
            content.append({"type": "text", "text": f"Photo {i}"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}})
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
