"""Scene (burst) clustering: consecutive shots close in time and appearance."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cluster_scenes(
    times: Sequence[float | None],
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
    centroid: np.ndarray | None = None
    count = 0
    prev_t: float | None = None
    for t, e in zip(times, emb, strict=True):
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


def visual_similarity(a: np.ndarray, b: np.ndarray,
                      subject_a: np.ndarray | None = None, subject_b: np.ndarray | None = None) -> float:
    """Either a composition change or a subject change can distinguish a moment."""
    similarity = float(np.dot(a, b))
    if subject_a is not None and subject_b is not None:
        similarity = min(similarity, float(np.dot(subject_a, subject_b)))
    return float(np.clip(similarity, -1, 1))


def cluster_moments(times, scenes, embeddings, subject_embeddings, gap_s=8.0, sim_thresh=0.92):
    """Finer, consecutive groups inside scenes, without changing scene budgets.

    Compare with the running moment centroid to resist slow visual drift.
    A missing subject detection is unknown, not a visual change. Returns moment
    IDs and each moment's largest incoming visual change (shared by its frames).
    Inputs are time ordered and embeddings are normalized; memory is O(N + D).
    """
    ids, changes = [], []
    whole = subject = None
    prev_time = prev_scene = None
    count = subject_count = 0
    moment = -1
    incoming = {}
    for t, scene, vec, crop in zip(times, scenes, embeddings, subject_embeddings, strict=True):
        same_scene = whole is not None and scene == prev_scene
        similarity = visual_similarity(vec, whole / (np.linalg.norm(whole) + 1e-9), crop,
                                       subject / (np.linalg.norm(subject) + 1e-9)
                                       if subject is not None else None) if same_scene else 1.0
        gap = t is not None and prev_time is not None and t - prev_time > gap_s
        if not same_scene or gap or similarity < sim_thresh:
            moment += 1
            whole, count = vec.copy(), 1
            subject, subject_count = (crop.copy(), 1) if crop is not None else (None, 0)
            incoming[moment] = (1 - similarity) / 2 if same_scene else 0.0
        else:
            whole = (whole * count + vec) / (count + 1)
            count += 1
            if crop is not None:
                subject = (subject * subject_count + crop) / (subject_count + 1) if subject_count else crop.copy()
                subject_count += 1
        ids.append(moment)
        prev_scene, prev_time = scene, t
    changes = [incoming[m] for m in ids]
    return ids, changes


def cluster_bursts(times, scenes, embeddings, gap_s=3.0, span_s=12.0, sim_thresh=0.75):
    """Continuous capture sequences, retaining subtle expression changes.

    Only whole-view changes split a sequence visually; subject pose changes
    belong among its alternatives. Missing timestamps are not guessed from
    filenames: those frames receive no temporal burst assignment.
    """
    ids = []
    burst = -1
    start = previous = previous_scene = centroid = None
    count = 0
    for t, scene, vector in zip(times, scenes, embeddings, strict=True):
        if t is None or not np.isfinite(t):
            ids.append(-1)
            start = previous = previous_scene = centroid = None
            continue
        new = (previous is None or scene != previous_scene or t < previous
               or t - previous > gap_s or t - start >= span_s)
        if not new and vector is not None and centroid is not None:
            new = float(np.dot(vector, centroid / (np.linalg.norm(centroid) + 1e-9))) < sim_thresh
        if new:
            burst += 1
            start, count = t, 0
            centroid = None
        if vector is not None:
            centroid = vector.copy() if centroid is None else (centroid * count + vector) / (count + 1)
            count += 1
        ids.append(burst)
        previous, previous_scene = t, scene
    return ids
