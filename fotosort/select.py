"""Pick the best, most varied shots from one bucket (e.g. one animal on one day)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    id: str
    score: float
    scene: int
    emb: np.ndarray  # L2-normalised


def select_bucket(cands: list[Candidate], k_max: int, k_min: int, dup_thresh: float) -> list[str]:
    """Greedy diverse selection.

    1. Take the best shot of every scene, strongest scenes first.
    2. Fill up to `k_max` with the next-best shots that are not near-duplicates
       (cosine similarity >= `dup_thresh`) of anything already picked.
    3. If fewer than `k_min` were chosen, top up by score ignoring duplicates.
    """
    if not cands or k_max <= 0:
        return []
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    picked: list[Candidate] = []
    seen_scenes: set[int] = set()

    def is_dup(c: Candidate) -> bool:
        return any(float(np.dot(c.emb, p.emb)) >= dup_thresh for p in picked)

    for c in ranked:  # pass 1: best of each scene
        if len(picked) >= k_max:
            break
        if c.scene not in seen_scenes and not is_dup(c):
            picked.append(c)
            seen_scenes.add(c.scene)
    for c in ranked:  # pass 2: fill with non-duplicates
        if len(picked) >= k_max:
            break
        if c not in picked and not is_dup(c):
            picked.append(c)
    for c in ranked:  # pass 3: guarantee k_min
        if len(picked) >= min(k_min, k_max):
            break
        if c not in picked:
            picked.append(c)
    return [c.id for c in picked]
