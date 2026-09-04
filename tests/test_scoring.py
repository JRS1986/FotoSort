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


def test_taste_bonus_rewards_similar_frames(tmp_path, capsys):
    import numpy as np
    from PIL import Image

    from fotosort.cli import taste_bonus
    Image.new("RGB", (64, 64), (200, 30, 30)).save(tmp_path / "fav.jpg")

    class FakeEmb:
        def prepare(self, img): return img
        def embed_batch(self, imgs): return np.array([[1.0, 0.0]], dtype=np.float32)

    like = Photo(path=Path("a.jpg"), key="a", emb=np.array([1.0, 0.0], dtype=np.float32), score=0.0)
    unlike = Photo(path=Path("b.jpg"), key="b", emb=np.array([0.0, 1.0], dtype=np.float32), score=0.0)
    taste_bonus([like, unlike], str(tmp_path), 0.5, FakeEmb())
    assert abs(like.score - 0.5) < 1e-6 and unlike.score == 0.0
