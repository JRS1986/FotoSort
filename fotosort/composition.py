"""Local composition and editorial preferences, not predictions of award success.

Geometry uses the detected bounding-box centre, not an inferred eye position.
Editorial contrasts reuse frozen CLIP embeddings. All signals are weak
preferences: unusual, centred and convention-breaking photos remain eligible.
"""
from __future__ import annotations

import math

import numpy as np

GOLDEN_MINOR = (3 - math.sqrt(5)) / 2
GEOMETRY_FIELDS = ("rule_of_thirds", "golden_ratio", "center_composition")

# Paired descriptions keep the subject/category unspecified. These are
# uncalibrated semantic preferences, not classifiers trained on award winners.
EDITORIAL_PROMPTS = {
    "light": (
        "a photograph with expressive light, atmospheric depth and beautiful tonal separation",
        "a photograph with flat uninteresting light and muddy tonal separation",
    ),
    "composition": (
        "a photograph with a deliberate composition, visual balance and a clear focal point",
        "a photograph with an accidental composition and distracting visual clutter",
    ),
    "subject": (
        "a photograph with a clearly readable subject and an effective relationship to its surroundings",
        "a photograph with an obscured subject lost in distracting surroundings",
    ),
    "moment": (
        "a photograph capturing a compelling moment, expression, interaction or visual story",
        "a photograph of an uninteresting moment with little expression or visual story",
    ),
}


def _alignment(x: float, y: float, lines: tuple[float, float], tolerance: float) -> float:
    """Reward a guide line, with the strongest match at its intersections."""
    gx = math.exp(-0.5 * (min(abs(x - line) for line in lines) / tolerance) ** 2)
    gy = math.exp(-0.5 * (min(abs(y - line) for line in lines) / tolerance) ** 2)
    return 0.5 * max(gx, gy) + 0.5 * math.sqrt(gx * gy)


def composition_metrics(box: tuple | None, tolerance: float = 0.08) -> dict[str, float | None]:
    """Placement affinity in 0..1; missing/invalid detections remain unknown.

    Golden-ratio lines are at 0.382 and 0.618 of the frame. This does not fit
    a golden spiral or evaluate a horizon. Thirds, golden-ratio and central
    placement are alternative compositions, so their bonuses do not stack.
    Large boxes give uncertain point locations and get less geometric weight.
    """
    result = dict.fromkeys((*GEOMETRY_FIELDS, "placement_reliability", "composition_score"))
    if box is None:
        return result
    coords = np.asarray(box, dtype=float)
    if (coords.shape != (4,) or not np.isfinite(coords).all()
            or (coords < 0).any() or (coords > 1).any()):
        return result
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1 or tolerance <= 0:
        return result
    x, y = (x1 + x2) / 2, (y1 + y2) / 2
    thirds = _alignment(x, y, (1 / 3, 2 / 3), tolerance)
    golden = _alignment(x, y, (GOLDEN_MINOR, 1 - GOLDEN_MINOR), tolerance)
    central = math.exp(-0.5 * (((x - 0.5) / tolerance) ** 2 + ((y - 0.5) / (2 * tolerance)) ** 2))
    # A whole-frame box says little about where the viewer's attention lies.
    reliability = float(np.clip((1 - max(x2 - x1, y2 - y1)) / 0.4, 0, 1))
    result.update(rule_of_thirds=thirds, golden_ratio=golden, center_composition=central,
                  placement_reliability=reliability,
                  composition_score=reliability * max(thirds, golden, central))
    return result


def editorial_scores(embeddings: np.ndarray, text_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    """Positive-minus-negative cosine similarity, one margin per editorial aspect.

    Both inputs must be L2-normalized. Margins are not probabilities or ratings;
    compare them within a subject/day group before using a small bounded bonus.
    """
    if len(text_embeddings) != 2 * len(EDITORIAL_PROMPTS):
        raise ValueError("Expected a positive and negative text embedding per editorial aspect")
    margins = embeddings @ (text_embeddings[::2] - text_embeddings[1::2]).T
    scores = {name: margins[:, i] for i, name in enumerate(EDITORIAL_PROMPTS)}
    scores["editorial_score"] = margins.mean(axis=1)
    return scores
