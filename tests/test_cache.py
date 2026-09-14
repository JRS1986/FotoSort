from collections import Counter

import numpy as np
import pytest
from numpy.lib.npyio import NpzFile

from fotosort import cache


def entries(n=20):
    return {str(i): dict(emb=np.ones(768), sig=np.ones(576), dino=np.ones(384), person=0,
                         subject=0.2, edge=False, sharpness=100, clip_low=0, clip_high=0, mean=120,
                         digest=f"digest{i}", box=(0.1, 0.2, 0.4, 0.8), subject_sharpness=45)
            for i in range(n)}


def test_cache_reads_each_feature_array_once_and_shares_backing_storage(tmp_path, monkeypatch):
    cache.save(tmp_path, entries(), "detector")
    reads = Counter()
    original = NpzFile.__getitem__

    def read(self, key):
        reads[key] += 1
        return original(self, key)

    monkeypatch.setattr(NpzFile, "__getitem__", read)
    loaded = cache.load(tmp_path, "detector")
    assert len(loaded) == 20
    for key in ("emb", "sig", "dino"):
        assert reads[key] == 1
        assert len({id(c[key].base) for c in loaded.values()}) == 1
    assert loaded["0"]["digest"] == "digest0"
    assert loaded["0"]["box"] == pytest.approx((0.1, 0.2, 0.4, 0.8))
    assert loaded["0"]["subject_sharpness"] == 45


def test_no_subject_round_trip_and_detector_invalidation(tmp_path):
    data = entries(1)
    data["0"].update(box=None, subject_sharpness=None)
    cache.save(tmp_path, data, "detector-a")
    loaded = cache.load(tmp_path, "detector-a")
    assert loaded["0"]["box"] is None and loaded["0"]["subject_sharpness"] is None
    assert cache.load(tmp_path, "detector-b") == {}


def test_optional_subject_and_detail_features_round_trip(tmp_path):
    data = entries(2)
    data['0'].update(subject_dino=np.ones(384), subject_dino_checked=True,
                     detail_focus=123., detail_focus_checked=True)
    data['1'].update(subject_dino=None, subject_dino_checked=True,
                     detail_focus=None, detail_focus_checked=True)
    cache.save(tmp_path, data, 'detector')
    loaded = cache.load(tmp_path, 'detector')
    assert np.array_equal(loaded['0']['subject_dino'], data['0']['subject_dino'])
    assert loaded['0']['detail_focus'] == 123 and loaded['0']['detail_focus_checked']
    assert loaded['1']['subject_dino'] is None and loaded['1']['subject_dino_checked']
    assert loaded['1']['detail_focus'] is None and loaded['1']['detail_focus_checked']


def test_interrupted_save_preserves_last_cache(tmp_path, monkeypatch):
    cache.save(tmp_path, entries(1))
    original = (tmp_path / cache.CACHE_NAME).read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez", fail)
    with pytest.raises(OSError):
        cache.save(tmp_path, entries(2))
    assert (tmp_path / cache.CACHE_NAME).read_bytes() == original
    assert not list(tmp_path.glob(".fotosort-cache-*"))


def test_external_dxo_cache_keys_and_subsecond_updates(tmp_path):
    import os

    root = tmp_path / "photos"
    root.mkdir()
    source = tmp_path / "dxo.dng"
    source.write_bytes(b"first")
    os.utime(source, ns=(1_000_000_001, 1_000_000_001))
    first = cache.cache_key(source, root)
    source.write_bytes(b"other")
    os.utime(source, ns=(1_000_000_002, 1_000_000_002))
    assert cache.cache_key(source, root) != first
