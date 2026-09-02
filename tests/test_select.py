import numpy as np

from fotosort.select import select_day, Candidate


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _cands():
    return [
        Candidate(id="a1", score=0.9, bucket="zebra", emb=_unit([1, 0, 0])),
        Candidate(id="a2", score=0.8, bucket="zebra", emb=_unit([1, 0.01, 0])),   # near-dup of a1
        Candidate(id="a3", score=0.7, bucket="zebra", emb=_unit([0, 1, 0])),
        Candidate(id="b1", score=0.5, bucket="gull", emb=_unit([0, 0, 1])),
        Candidate(id="b2", score=0.4, bucket="cormorant", emb=_unit([0, 0.01, 1])),  # same shot, other label
        Candidate(id="c1", score=-1.2, bucket="shark", emb=_unit([1, 1, 1])),
    ]


def test_best_first_with_bucket_budgets():
    picks = select_day(_cands(), {"zebra": 2, "gull": 1, "cormorant": 1, "shark": 2}, 0.95, 0.9)
    assert picks == ["a1", "a3", "b1"]


def test_duplicates_are_rejected_across_buckets():
    picks = select_day(_cands(), {"zebra": 5, "gull": 5, "cormorant": 5, "shark": 5}, 0.95, 0.9)
    assert "b2" not in picks and "a2" not in picks


def test_min_score_floor_blocks_weak_photos_even_with_free_slots():
    picks = select_day(_cands(), {"zebra": 5, "gull": 5, "cormorant": 5, "shark": 5}, 0.95, 0.9, min_score=-0.5)
    assert "c1" not in picks
    picks = select_day(_cands(), {"zebra": 5, "gull": 5, "cormorant": 5, "shark": 5}, 0.95, 0.9, min_score=-5)
    assert "c1" in picks


def test_pixel_signature_catches_semantically_different_duplicates():
    sig = _unit(np.arange(16, dtype=np.float32) - 7.5)
    cands = [
        Candidate(id="p1", score=0.9, bucket="x", emb=_unit([1, 0, 0]), sig=sig),
        Candidate(id="p2", score=0.8, bucket="x", emb=_unit([0, 1, 0]), sig=sig),
        Candidate(id="p3", score=0.7, bucket="x", emb=_unit([0, 0, 1]), sig=_unit(np.random.default_rng(1).normal(size=16))),
    ]
    assert select_day(cands, {"x": 3}, 0.95, 0.9) == ["p1", "p3"]


def test_empty():
    assert select_day([], {}, 0.95, 0.9) == []
