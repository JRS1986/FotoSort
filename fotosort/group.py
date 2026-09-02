"""Scene (burst) clustering: consecutive shots close in time and appearance."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def cluster_scenes(
    times: Sequence[Optional[float]],
    emb: np.ndarray,
    gap_s: float = 120.0,
    sim_thresh: float = 0.85,
) -> list[int]:
    """Assign a scene id to each item. Items must be sorted by time.

    A new scene starts when the time gap to the previous shot exceeds `gap_s`
    or the cosine similarity to the running scene centroid drops below
    `sim_thresh`. Embeddings must be L2-normalised."""
    ids: list[int] = []
    scene = -1
    centroid: Optional[np.ndarray] = None
    count = 0
    prev_t: Optional[float] = None
    for t, e in zip(times, emb):
        new = centroid is None
        if not new and t is not None and prev_t is not None and t - prev_t > gap_s:
            new = True
        if not new:
            c = centroid / (np.linalg.norm(centroid) + 1e-9)
            if float(np.dot(c, e)) < sim_thresh:
                new = True
        if new:
            scene += 1
            centroid = e.astype(np.float32).copy()
            count = 1
        else:
            centroid = (centroid * count + e) / (count + 1)
            count += 1
        ids.append(scene)
        if t is not None:
            prev_t = t
    return ids
