"""Pick the best, most varied shots from one bucket (e.g. one animal on one day)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    id: str
    score: float
    scene: int
    emb: np.ndarray  # L2-normalised CLIP embedding
    sig: np.ndarray | None = None  # zero-mean unit-norm thumbnail (see quality.signature)


def select_bucket(cands: list[Candidate], k_max: int, dup_thresh: float, dup_pixel: float = 0.93) -> list[str]:
    """Greedy diverse selection.

    1. Take the best shot of every scene, strongest scenes first.
    2. Fill up to `k_max` with the next-best shots that are not near-duplicates
       of anything already picked. Two frames are near-duplicates when their
       CLIP cosine similarity >= `dup_thresh` or their thumbnail correlation
       >= `dup_pixel`. A bucket with only duplicate frames yields one pick.
    """
    if not cands or k_max <= 0:
        return []
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    picked: list[Candidate] = []
    seen_scenes: set[int] = set()

    def is_dup(c: Candidate) -> bool:
        for p in picked:
            if float(np.dot(c.emb, p.emb)) >= dup_thresh:
                return True
            if c.sig is not None and p.sig is not None and float(np.dot(c.sig, p.sig)) >= dup_pixel:
                return True
        return False

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
    return [c.id for c in picked]
