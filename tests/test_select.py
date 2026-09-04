import numpy as np

from fotosort.selection import Candidate, select_day


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
        Candidate(id="p3", score=0.7, bucket="x", emb=_unit([0, 0, 1]),
                  sig=_unit(np.random.default_rng(1).normal(size=16))),
    ]
    assert select_day(cands, {"x": 3}, 0.95, 0.9) == ["p1", "p3"]


def test_empty():
    assert select_day([], {}, 0.95, 0.9) == []


def test_shortlist_covers_every_scene_before_filling_by_score():
    from fotosort.selection import ScenedCandidate, build_shortlist
    sig_a = _unit(np.arange(16, dtype=np.float32))

    def rnd(seed):
        return _unit(np.random.default_rng(seed).normal(size=16))

    cands = [
        ScenedCandidate(id="s0a", score=0.9, bucket="x", emb=_unit([1, 0]), sig=sig_a, scene=0),
        ScenedCandidate(id="s0b", score=0.8, bucket="x", emb=_unit([1, 0]), sig=sig_a, scene=0),  # pixel twin of s0a
        ScenedCandidate(id="s0c", score=0.7, bucket="x", emb=_unit([1, 0.1]), sig=rnd(2), scene=0),
        ScenedCandidate(id="s1a", score=0.1, bucket="x", emb=_unit([1, 0.05]), sig=rnd(3), scene=1),
        ScenedCandidate(id="s2a", score=-0.9, bucket="x", emb=_unit([0, 1]), sig=rnd(4), scene=2),
    ]
    # scene coverage first (s0a, s1a, s2a), then fill (s0c); pixel twin s0b never
    assert build_shortlist(cands, cap=4) == ["s0a", "s1a", "s2a", "s0c"]
    assert build_shortlist(cands, cap=2) == ["s0a", "s1a"]


def test_tournament_chunks_see_every_frame_once_and_balance_sizes():
    from fotosort.selection import ScenedCandidate, chunk_for_tournament
    rng = np.random.default_rng(5)
    cands = [ScenedCandidate(id=f"f{i}", score=rng.normal(), bucket="x", emb=_unit([1, 0]),
                             sig=_unit(rng.normal(size=16)), scene=i // 7) for i in range(50)]
    chunks, twins = chunk_for_tournament(cands, chunk=24)
    flat = [i for ch in chunks for i in ch]
    assert sorted(flat) == sorted(c.id for c in cands) and twins == {}   # everything shown exactly once
    assert len(chunks) == 3 and max(map(len, chunks)) - min(map(len, chunks)) <= 1
    assert chunk_for_tournament([], 24) == ([], {})
    # a pixel twin is folded onto its better-scoring sibling and reported as such
    cands[1].sig = cands[0].sig
    lo, hi = sorted([cands[0], cands[1]], key=lambda c: c.score)
    chunks, twins = chunk_for_tournament(cands, chunk=24)
    assert twins == {lo.id: hi.id} and lo.id not in [i for ch in chunks for i in ch]
