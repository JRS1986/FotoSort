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


@dataclass
class ScenedCandidate(Candidate):
    scene: int = -1


def build_shortlist(cands: list[ScenedCandidate], cap: int, pixel_dup: float = 0.97) -> list[str]:
    """What the judge gets to see for one bucket: the best frame of every scene
    (burst) first, then the next-best frames by score, up to `cap`. Only
    pixel-identical frames (thumbnail correlation >= `pixel_dup`) are dropped;
    deciding between similar-but-different moments is the judge's job."""
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    kept: list[ScenedCandidate] = []
    for c in ranked:
        if any(c.sig is not None and p.sig is not None and float(np.dot(c.sig, p.sig)) >= pixel_dup for p in kept):
            continue
        kept.append(c)
    seen_scenes: set[int] = set()
    out: list[ScenedCandidate] = []
    for c in kept:  # pass 1: one per scene
        if c.scene not in seen_scenes:
            out.append(c)
            seen_scenes.add(c.scene)
    for c in kept:  # pass 2: fill by score
        if c not in out:
            out.append(c)
    return [c.id for c in out[:cap]]


def chunk_for_tournament(cands: list[ScenedCandidate], chunk: int = 24,
                         pixel_dup: float = 0.97) -> tuple[list[list[str]], dict[str, str]]:
    """Split one bucket into judge-sized chunks so every distinct frame is seen.
    Pixel-identical frames are folded onto their best-scoring twin first; the
    rest are ordered by scene, then by score, and cut into chunks of at most
    `chunk` frames, balanced so the last chunk is not a stub.
    Returns (chunks of ids, {dropped id: id of the twin that stands for it})."""
    ranked = sorted(cands, key=lambda c: c.score, reverse=True)
    kept: list[ScenedCandidate] = []
    twins: dict[str, str] = {}
    for c in ranked:
        twin = next((p for p in kept if c.sig is not None and p.sig is not None
                     and float(np.dot(c.sig, p.sig)) >= pixel_dup), None)
        if twin is not None:
            twins[c.id] = twin.id
            continue
        kept.append(c)
    kept.sort(key=lambda c: (c.scene, -c.score))
    if not kept:
        return [], twins
    n_chunks = max(1, -(-len(kept) // chunk))
    size = -(-len(kept) // n_chunks)
    return [[c.id for c in kept[i:i + size]] for i in range(0, len(kept), size)], twins
