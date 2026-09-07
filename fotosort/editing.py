"""Which picks are worth a RAW edit, and which look to give them.

`edit_benefit` estimates from measurable facts how much a frame would gain
from RAW development: clipped highlights that a RAW can recover, crushed
shadows, a dark or flat exposure, high ISO noise. 0 = the JPEG is fine as it
is, 100 = definitely develop the RAW. The judge, when used, overrides it with
its own view and says why.

`PRESETS` is the fixed menu the judge (or the style mapping) chooses from.
"""
from __future__ import annotations

PRESETS = [
    "Natural",
    "Color Pop",
    "Wildlife Crisp",
    "Golden Hour Warmth",
    "Moody Colors",
    "Low Key Dramatic",
    "High Key",
    "Black & White",
    "Vignette",
    "Portrait Soft",
]

STYLE_TO_PRESET = {
    "wildlife": "Wildlife Crisp",
    "landscape": "Color Pop",
    "golden hour": "Golden Hour Warmth",
    "portrait": "Portrait Soft",
    "night": "Low Key Dramatic",
    "black and white": "Black & White",
    "food": "Color Pop",
    "urban": "Moody Colors",
    "water": "Color Pop",
    "macro": "Wildlife Crisp",
    "general": "Natural",
}


def edit_benefit(clip_high: float, clip_low: float, mean: float, std: float | None = None,
                 iso: int | None = None) -> int:
    """0..100. Clipped highlights weigh most (RAW keeps 1-2 stops the JPEG threw
    away), then crushed shadows, then a dark or flat exposure, then noise."""
    score = 0.0
    score += 45.0 * min(1.0, clip_high / 0.05)     # 5 % blown pixels -> full weight
    score += 30.0 * min(1.0, clip_low / 0.15)      # 15 % crushed pixels -> full weight
    if mean < 80:
        score += 15.0 * min(1.0, (80 - mean) / 40)  # underexposed: pull up cleanly from RAW
    elif mean > 175:
        score += 10.0 * min(1.0, (mean - 175) / 40)
    if std is not None and std < 35:
        score += 10.0 * min(1.0, (35 - std) / 20)   # flat, hazy: contrast and dehaze from RAW
    if iso:
        score += 20.0 * min(1.0, max(0.0, (iso - 1600) / 6400))  # noise reduction pays above ISO 1600
    return int(round(min(100.0, score)))


def benefit_band(score: int) -> str:
    return "high" if score >= 50 else "medium" if score >= 25 else "low"
