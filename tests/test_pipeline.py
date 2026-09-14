"""Exercise scan through enhanced export and cache reuse with model stand-ins."""
import csv
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fotosort.cache import CACHE_NAME
from fotosort.cli import main
from fotosort.judge import Verdict


@pytest.mark.parametrize("use_judge,use_bursts", [(False, False), (True, False), (True, True)])
def test_pipeline_keeps_originals_records_decisions_and_reuses_features(tmp_path, monkeypatch, use_judge, use_bursts):
    counts = {"embeddings": 0, "detector": 0, "judged": 0, "dino": 0}

    class Embedder:
        device = "cpu"

        def prepare(self, image):
            return np.asarray(image).mean(axis=(0, 1)).astype(np.float32)

        def embed_batch(self, tensors):
            counts["embeddings"] += len(tensors)
            array = np.stack(tensors)
            return array / np.linalg.norm(array, axis=1, keepdims=True)

        def aesthetic(self, embeddings):
            return embeddings[:, 0] * 5

        def text_embeddings(self, labels, template=None):
            return np.tile([1., 0., 0.], (len(labels), 1)).astype(np.float32)

    class Dino:
        def __init__(self, device):
            pass

        def embed_batch(self, images):
            counts["dino"] += len(images)
            return np.tile([1., 0., 0.], (len(images), 1)).astype(np.float32)

    class Detector:
        def __init__(self, device, weights):
            pass

        def detect(self, images):
            counts["detector"] += len(images)
            return [dict(person=0., subject=0.25, edge=False, box=(0.25, 0.25, 0.75, 0.75)) for _ in images]

    class FakeJudge:
        model = "test-judge"
        usage = {"input": 0, "output": 0}

        def __init__(self, *args):
            pass

        def judge(self, paths, label, day, k, boxes, final=False):
            counts["judged"] += len(paths)
            assert all(boxes[str(p)] is not None for p in paths)
            return Verdict([str(paths[0])], {str(paths[0]): "exceptional moment"})

    module = SimpleNamespace(Embedder=Embedder, DinoEmbedder=Dino, SubjectDetector=Detector,
                             classify=lambda emb, texts, labels: (["bird"] * len(emb), np.ones(len(emb))))
    monkeypatch.setitem(sys.modules, "fotosort.embed", module)
    monkeypatch.setattr("fotosort.judge.Judge", FakeJudge)
    y, x = np.mgrid[:128, :128]
    for i in range(3):
        rgb = np.stack([((x // 8 + y // 8) % 2) * 150 + 30 + i * 10,
                        x + 30, y + 40], axis=-1).astype(np.uint8)
        exif = Image.Exif()
        exif.get_ifd(0x8769)[36867] = f"2026:09:01 10:00:0{i}"
        Image.fromarray(rgb).save(tmp_path / f"{i}.jpg", exif=exif.tobytes())
    (tmp_path / "bad.jpg").write_bytes(b"invalid image")
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.jpg")}
    args = [str(tmp_path), "--copy", "--enhance", "--enhance-style", "general", "--recursive"]
    if use_judge:
        args.append("--judge")
    if use_bursts:
        args.append("--burst-alternatives")
    assert main(args) == 0
    with (tmp_path / "fotosort_report.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4 and any(r["reject"] == "unreadable" for r in rows)
    assert all(r["decision"] and r["subject_sharpness"] for r in rows if r["reject"] != "unreadable")
    assert any(r["selected"] == "1" for r in rows)
    assert list((tmp_path / "Highlights").glob("*.jpg"))
    expected_clip = 9 if use_bursts else 3  # whole frame plus two local burst crops
    expected_dino = 9 if use_bursts else 6  # whole, subject, optional interaction view
    assert counts["embeddings"] == expected_clip and counts["detector"] == 3
    assert counts["dino"] == expected_dino
    assert main(args) == 0
    assert counts["embeddings"] == expected_clip and counts["detector"] == 3
    assert counts["dino"] == expected_dino  # no re-extraction, including local burst views
    # The previous cache format can be enriched without running whole-frame
    # image models or the detector again.
    with np.load(tmp_path / CACHE_NAME, allow_pickle=False) as archive:
        legacy = {k: archive[k] for k in archive.files if not k.startswith(('subject_dino', 'detail_focus'))}
    legacy['version'] = np.array(6)
    np.savez(tmp_path / CACHE_NAME, **legacy)
    assert main(args) == 0
    assert counts["embeddings"] == expected_clip and counts["detector"] == 3
    assert counts["dino"] == expected_dino + 3  # only three missing subject views
    # A real v7 cache has no secondary boxes or burst views. Upgrade those
    # features once while retaining full-frame CLIP and DINO extraction.
    with np.load(tmp_path / CACHE_NAME, allow_pickle=False) as archive:
        legacy = {k: archive[k] for k in archive.files
                  if not k.startswith(('subject_boxes', 'subject_clip', 'interaction_', 'burst_features'))}
    legacy['version'] = np.array(7)
    np.savez(tmp_path / CACHE_NAME, **legacy)
    assert main(args) == 0
    assert counts["embeddings"] == expected_clip + (6 if use_bursts else 0)
    assert counts["detector"] == (6 if use_bursts else 3)
    assert counts["dino"] == expected_dino + 3 + (3 if use_bursts else 0)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.jpg")} == originals
    if use_judge:
        assert counts["judged"] >= 6
