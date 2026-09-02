import numpy as np

from fotosort.group import cluster_scenes


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_time_gap_starts_new_scene():
    emb = np.stack([_unit([1, 0]), _unit([1, 0]), _unit([1, 0])])
    times = [0.0, 10.0, 1000.0]
    assert cluster_scenes(times, emb, gap_s=120, sim_thresh=0.5) == [0, 0, 1]


def test_dissimilar_embedding_starts_new_scene():
    emb = np.stack([_unit([1, 0]), _unit([1, 0.05]), _unit([0, 1])])
    times = [0.0, 1.0, 2.0]
    assert cluster_scenes(times, emb, gap_s=120, sim_thresh=0.8) == [0, 0, 1]


def test_missing_time_never_breaks_on_gap():
    emb = np.stack([_unit([1, 0]), _unit([1, 0])])
    assert cluster_scenes([None, None], emb, gap_s=1, sim_thresh=0.5) == [0, 0]
