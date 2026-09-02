"""Pick the best, most varied shots for one day across all subject buckets."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    id: str
    score: float
    bucket: str
    emb: np.ndarray  # L2-normalised CLIP embedding
    sig: np.ndarray | None = None  # zero-mean unit-norm thumbnail (see quality.signature)


def is_duplicate(c: Candidate, picked: list[Candidate], dup_sim: float, dup_pixel: float) -> bool:
    for p in picked:
        if float(np.dot(c.emb, p.emb)) >= dup_sim:
            return True
        if c.sig is not None and p.sig is not None and float(np.dot(c.sig, p.sig)) >= dup_pixel:
            return True
    return False


def select_day(
    cands: list[Candidate],
    budgets: dict[str, int],
    dup_sim: float,
    dup_pixel: float,
    min_score: float = -0.5,
) -> list[str]:
    """Greedy selection over a whole day.

    Candidates are visited best-first. One is taken when its score is at least
    `min_score`, its bucket still has budget, and it is not a near-duplicate
    (CLIP cosine >= `dup_sim` or thumbnail correlation >= `dup_pixel`) of any
    photo already taken that day, in any bucket. So a weak photo is never
    picked just because its subject has free slots, and the same colony shot
    labelled "seagull" and "cormorant" cannot get in twice.
    """
    picked: list[Candidate] = []
    count: dict[str, int] = defaultdict(int)
    for c in sorted(cands, key=lambda c: c.score, reverse=True):
        if c.score < min_score:
            break
        if count[c.bucket] >= budgets.get(c.bucket, 0):
            continue
        if is_duplicate(c, picked, dup_sim, dup_pixel):
            continue
        picked.append(c)
        count[c.bucket] += 1
    return [c.id for c in picked]
