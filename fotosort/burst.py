"""General local expression/interaction cues; no trained taste or species rules."""
from __future__ import annotations

import numpy as np

from fotosort.quality import subject_embedding_crop

# Keep these broad and separate. In particular, mouth opening is not a proxy
# for a good expression, and motion is not a requirement for a good photograph.
BURST_PROMPTS = {
    "expression": (
        "a photograph with a distinctive, expressive face or engaging gaze",
        "a photograph with an indistinct face and an ordinary neutral expression",
    ),
    "interaction": (
        "a photograph capturing a meaningful interaction or relationship between subjects",
        "a photograph of isolated subjects with no visible interaction or relationship",
    ),
    "gesture": (
        "a photograph capturing a distinctive pose, gesture or behaviour",
        "a photograph capturing an ordinary, unexpressive pose",
    ),
}
MAX_SUBJECT_BOXES = 6


def valid_boxes(boxes):
    result = []
    for box in boxes:
        values = np.asarray(box, dtype=float)
        if (values.shape == (4,) and np.isfinite(values).all() and (values >= 0).all()
                and (values <= 1).all() and values[2] > values[0] and values[3] > values[1]):
            result.append(tuple(float(v) for v in values))
    return sorted(set(result), key=lambda b: (-(b[2] - b[0]) * (b[3] - b[1]), b))[:MAX_SUBJECT_BOXES]


def interaction_box(box, boxes):
    """Include secondary detections and space around the primary subject.

    Expansion also offers a view of nearby subjects the detector missed. It
    does not imply that all animals, faces or small subjects were detected.
    """
    boxes = valid_boxes([*boxes, *([box] if box is not None else [])])
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    dx, dy = (x2 - x1) * .25, (y2 - y1) * .25
    return max(0., x1 - dx), max(0., y1 - dy), min(1., x2 + dx), min(1., y2 + dy)


def local_burst_views(image, box, boxes):
    """At most two additional local views, with no duplicate full-frame view."""
    context = interaction_box(box, boxes)
    views = {}
    for name, region in (("subject", box), ("interaction", context)):
        if region is None or (region[2] - region[0]) * (region[3] - region[1]) >= .9:
            continue
        if name == "interaction" and box is not None and np.allclose(region, box):
            continue
        crop = subject_embedding_crop(image, region)
        if crop is not None:
            views[name] = crop
    return views


def burst_scores(whole, subject, interaction, texts):
    """Uncalibrated paired CLIP margins, not expression probabilities.

    Take the strongest available view of each aspect. These scores only
    nominate alternatives inside bursts; they never change the overall score.
    """
    if len(texts) != 2 * len(BURST_PROMPTS):
        raise ValueError("Expected paired text embeddings for each burst aspect")
    views = np.stack([v for v in (whole, subject, interaction) if v is not None])
    margins = views @ (texts[::2] - texts[1::2]).T
    return {name: float(np.max(margins[:, i])) for i, name in enumerate(BURST_PROMPTS)}
