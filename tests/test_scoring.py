from pathlib import Path

from fotosort.cli import Photo, subject_term


def _p(subject, edge=False):
    return Photo(path=Path("x.jpg"), key="k", subject=subject, edge=edge)


def test_subject_bonus_grows_with_size_and_caps():
    assert subject_term(_p(0.0), 0.4) == 0.0
    assert subject_term(_p(0.003), 0.4) == 0.0          # whale blow far away
    assert 0 < subject_term(_p(0.03), 0.4) < subject_term(_p(0.10), 0.4)
    assert abs(subject_term(_p(0.35), 0.4) - 0.6) < 1e-9 and subject_term(_p(0.35), 0.4) == subject_term(_p(0.9), 0.4)


def test_small_subject_at_frame_edge_is_penalised_but_large_one_is_not():
    assert subject_term(_p(0.04, edge=True), 0.4) < 0
    assert subject_term(_p(0.6, edge=True), 0.4) > 0
