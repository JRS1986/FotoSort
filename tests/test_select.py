import numpy as np

from fotosort.select import select_bucket, Candidate


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cands():
    # two bursts: scene 0 has three near-duplicates, scene 1 has one shot
    return [
        Candidate(id="a1", score=0.9, scene=0, emb=_unit([1, 0, 0])),
        Candidate(id="a2", score=0.8, scene=0, emb=_unit([1, 0.01, 0])),
        Candidate(id="a3", score=0.7, scene=0, emb=_unit([1, 0, 0.01])),
        Candidate(id="b1", score=0.5, scene=1, emb=_unit([0, 1, 0])),
    ]


def test_best_from_each_scene_comes_first():
    picks = select_bucket(_cands(), k_max=2, dup_thresh=0.98)
    assert picks == ["a1", "b1"]


def test_near_duplicates_are_skipped_when_filling():
    picks = select_bucket(_cands(), k_max=4, dup_thresh=0.98)
    # a2/a3 are near-duplicates of a1 (cos > 0.98), so only 2 picks
    assert picks == ["a1", "b1"]


def test_k_max_caps_selection():
    cands = [Candidate(id=f"x{i}", score=1 - i * 0.1, scene=i, emb=_unit(np.eye(6)[i])) for i in range(6)]
    picks = select_bucket(cands, k_max=3, dup_thresh=0.98)
    assert picks == ["x0", "x1", "x2"]


def test_all_duplicates_yield_one_pick():
    cands = _cands()[:3]
    assert select_bucket(cands, k_max=5, dup_thresh=0.98) == ["a1"]


def test_pixel_signature_catches_semantically_different_duplicates():
    # CLIP says different (sim 0.0) but the thumbnails are the same framing
    sig = _unit(np.arange(16, dtype=np.float32) - 7.5)
    cands = [
        Candidate(id="p1", score=0.9, scene=0, emb=_unit([1, 0, 0]), sig=sig),
        Candidate(id="p2", score=0.8, scene=1, emb=_unit([0, 1, 0]), sig=sig),
        Candidate(id="p3", score=0.7, scene=2, emb=_unit([0, 0, 1]), sig=_unit(np.random.default_rng(1).normal(size=16))),
    ]
    assert select_bucket(cands, k_max=3, dup_thresh=0.98, dup_pixel=0.93) == ["p1", "p3"]


def test_empty_bucket():
    assert select_bucket([], k_max=3, dup_thresh=0.98) == []
