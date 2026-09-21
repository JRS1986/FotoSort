"""Exercise report application and analysis/review races without image models."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from fotosort import cli
from fotosort.review_data import Collection, fingerprint, write_rows


def seed(root, name="a.jpg", selected="0"):
    path = root / name
    if path.suffix == ".jpg":
        Image.new("RGB", (20, 20), (1, 2, 3)).save(path)
    else:
        path.write_bytes(b"original RAW content")
    row = dict(file=str(path), relative_file=name, selected=selected, photo_sha256=fingerprint(path),
               edit_benefit="low", preset="Natural")
    write_rows(root / "fotosort_report.csv", list(row), [row])
    return path


def stub_models(monkeypatch, during=None):
    def features(photos, root, args):
        for p in photos:
            p.emb = np.array([1., 0.])
        if during:
            during()
        return SimpleNamespace(text_embeddings=lambda labels: np.array([[1., 0.]]))

    monkeypatch.setattr(cli, "compute_features", features)
    monkeypatch.setattr(cli, "load_labels", lambda _: ["bird"])
    for name in ("assign_scenes_and_labels", "assign_moments", "refine_close_focus", "editing_advice"):
        monkeypatch.setattr(cli, name, lambda *args: None)

    def scoring(photos, args):
        for p in photos:
            p.score, p.reject = 1.0, "blurry"

    monkeypatch.setattr(cli, "score_and_reject", scoring)


@pytest.mark.parametrize("move", [False, True])
def test_raw_cull_refuses_changed_unselected_raw_before_any_output(tmp_path, move):
    path = seed(tmp_path, "a.nef")
    path.write_bytes(b"a DIFFERENT RAW now occupies this filename")
    args = cli.parse_args([str(tmp_path), "--apply-report", "--raw-cull"] + (["--raw-cull-move"] if move else []))
    with pytest.raises(SystemExit, match="changed"):
        cli.apply_report([cli.Photo(path, "a")], tmp_path, args)
    assert path.exists() and path.read_bytes() == b"a DIFFERENT RAW now occupies this filename"
    assert not (tmp_path / "_DELETE_ME_raw").exists()
    assert not (tmp_path / "raw_cull.txt").exists()


def test_raw_cull_still_moves_unchanged_unselected_raw(tmp_path):
    path = seed(tmp_path, "a.nef")
    original = path.read_bytes()
    args = cli.parse_args([str(tmp_path), "--apply-report", "--raw-cull", "--raw-cull-move"])
    assert cli.apply_report([cli.Photo(path, "a")], tmp_path, args) == 0
    assert not path.exists()
    assert (tmp_path / "_DELETE_ME_raw" / "a.nef").read_bytes() == original


@pytest.mark.parametrize("initial,latest,selected", [(None, "keep", "1"), ("keep", "reject", "0"),
                                                    ("keep", "clear", "0")])
def test_analysis_applies_decision_saved_while_features_run(tmp_path, monkeypatch, initial, latest, selected):
    seed(tmp_path)
    collection = Collection(tmp_path)
    identifier = next(iter(collection.entries))
    if initial:
        collection.change(0, {identifier: initial})
    revision = collection.store.load()["revision"]
    stub_models(monkeypatch, lambda: collection.change(revision, {identifier: latest}, "saved during analysis"))
    assert cli.main([str(tmp_path)]) == 0
    with collection.report.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["selected"] == selected
    assert row["manual_decision"] == (latest if latest != "clear" else "")
    assert row["review_revision"] == str(revision + 1)


def test_late_review_edit_aborts_before_report_or_file_operations(tmp_path, monkeypatch, capsys):
    seed(tmp_path)
    collection = Collection(tmp_path)
    identifier = next(iter(collection.entries))
    before = collection.report.read_bytes()
    stub_models(monkeypatch)
    monkeypatch.setattr(cli, "editing_advice", lambda *args: collection.change(0, {identifier: "keep"}))
    for name in ("write_sidecars", "raw_cull", "move_picks"):
        monkeypatch.setattr(cli, name, lambda *args: pytest.fail("file operation after a concurrent review edit"))
    assert cli.main([str(tmp_path), "--copy", "--xmp", "all", "--raw-cull", "--raw-cull-move"]) == 2
    assert collection.report.read_bytes() == before
    assert collection.store.load()["revision"] == 1
    assert "Review changed" in capsys.readouterr().err


def test_forced_include_does_not_reclassify_a_blur_reject_as_automatic(tmp_path, monkeypatch):
    seed(tmp_path)
    stub_models(monkeypatch)
    assert cli.main([str(tmp_path)]) == 0
    report = tmp_path / "fotosort_report.csv"
    with report.open() as stream:
        before = next(csv.DictReader(stream))
    assert before["auto_selected"] == "0"
    assert cli.main([str(tmp_path), "--include", "a.jpg"]) == 0
    with report.open() as stream:
        after = next(csv.DictReader(stream))
    assert after["selected"] == "1" and after["reject"] == ""
    assert after["auto_selected"] == "0"
    assert after["auto_decision"] == before["auto_decision"] == "Rejected: blurry"
