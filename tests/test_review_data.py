import csv
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from fotosort.cli import Photo, apply_report, parse_args, write_report
from fotosort.decisions import main
from fotosort.review_data import Collection, Conflict, ReviewError, ReviewStore, apply_manual_decisions, fingerprint


def collection(root, names=("a.jpg", "b.jpg")):
    photos = []
    for i, name in enumerate(names):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 16), (40 + i * 50, 100, 80)).save(path)
        photos.append(Photo(path, name, selected=i == 0,
                            decision="Selected by score" if i == 0 else "Group budget filled",
                            label="bird", edit_band="low", preset="Natural"))
    write_report(photos, root / "fotosort_report.csv", root)
    return Collection(root), photos


def test_decisions_survive_rerun_and_clear_restores_new_baseline(tmp_path):
    c, photos = collection(tmp_path)
    a, b = list(c.entries)
    c.change(0, {a: "reject", b: "keep"}, "Better expression")
    before = {p.path: fingerprint(p.path) for p in photos}
    apply_manual_decisions(photos, tmp_path)
    assert [p.selected for p in photos] == [False, True]
    assert [p.auto_selected for p in photos] == [True, False]
    write_report(photos, c.report, tmp_path)
    rows = Collection(tmp_path).view()
    assert [r["selected"] for r in rows] == [False, True]
    assert rows[1]["manual_reason"] == "Better expression"
    refreshed = Collection(tmp_path)
    refreshed.change(1, {b: "clear"})
    assert not refreshed.view()[1]["selected"]
    assert before == {p.path: fingerprint(p.path) for p in photos}


def test_whole_folder_move_keeps_identity_duplicate_basenames_stay_distinct(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    c, _ = collection(old, ("one/IMG.jpg", "two/IMG.jpg"))
    first, second = list(c.entries)
    c.change(0, {second: "keep"})
    new = tmp_path / "renamed"
    old.rename(new)
    moved = Collection(new)
    assert list(moved.entries) == [first, second]
    assert moved.view()[0]["manual_decision"] == ""
    assert moved.view()[1]["manual_decision"] == "keep"


def test_changed_and_missing_files_do_not_inherit_decisions(tmp_path):
    c, photos = collection(tmp_path)
    a, b = list(c.entries)
    c.change(0, {a: "reject", b: "keep"})
    photos[0].path.write_bytes(b"changed")
    photos[1].path.unlink()
    changed = Collection(tmp_path)
    rows = changed.view()
    assert [r["status"] for r in rows] == ["changed", "missing"]
    assert all(not r["manual_decision"] for r in rows)
    with pytest.raises(Conflict):
        changed.change(1, {rows[0]["id"]: "keep"})
    changed.change(1, {rows[0]["id"]: "clear"})
    assert a not in changed.store.load()["decisions"]


def test_stale_sessions_cannot_overwrite_and_batch_edits_are_atomic(tmp_path):
    c, _ = collection(tmp_path)
    a, b = list(c.entries)
    c.change(0, {a: "reject", b: "keep"}, preference=[b, a])
    with pytest.raises(Conflict):
        c.change(0, {a: "keep"})
    with pytest.raises(ReviewError):
        c.change(1, {a: "keep", "unknown": "reject"})
    assert c.store.load()["revision"] == 1
    assert c.store.load()["decisions"][a]["choice"] == "reject"
    undone = c.undo(1)
    assert undone["revision"] == 2 and not undone["decisions"] and not undone["preferences"]


def test_concurrent_edits_have_one_winner(tmp_path):
    c, _ = collection(tmp_path)
    a = next(iter(c.entries))

    def change(choice):
        try:
            Collection(tmp_path).change(0, {a: choice})
            return "saved"
        except Conflict:
            return "conflict"
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(change, ["keep", "reject"])) == ["conflict", "saved"]
    assert c.store.load()["revision"] == 1


def test_failed_atomic_write_leaves_previous_review_intact(tmp_path, monkeypatch):
    c, _ = collection(tmp_path)
    a = next(iter(c.entries))
    c.change(0, {a: "keep"})
    previous = c.store.path.read_bytes()

    def fail(*args):
        raise OSError("interrupted replace")
    monkeypatch.setattr("fotosort.review_data.os.replace", fail)
    with pytest.raises(OSError):
        c.change(1, {a: "reject"})
    assert c.store.path.read_bytes() == previous
    assert not list(tmp_path.glob("..fotosort_review.json-*"))


def test_export_roundtrip_and_changed_source_refused_before_copy(tmp_path):
    c, photos = collection(tmp_path)
    a, b = list(c.entries)
    c.change(0, {a: "reject", b: "keep"})
    output = c.export("reviewed.csv", 1)
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert [r["selected"] for r in rows] == ["0", "1"]
    args = parse_args([str(tmp_path), "--apply-report", "--report", "reviewed.csv", "--copy"])
    assert apply_report(photos, tmp_path, args) == 0
    assert sorted(p.name for p in (tmp_path / "Highlights").iterdir()) == ["b.jpg"]
    (tmp_path / "a.jpg").write_bytes(b"new content")
    fresh = [Photo(tmp_path / "a.jpg", "a"), Photo(tmp_path / "b.jpg", "b")]
    with pytest.raises(SystemExit, match="changed"):
        apply_report(fresh, tmp_path, args)


def test_report_changes_and_escaping_paths_are_rejected(tmp_path):
    c, _ = collection(tmp_path)
    a = next(iter(c.entries))
    c.report.write_text(c.report.read_text() + "\n")
    with pytest.raises(Conflict, match="Report changed"):
        c.change(0, {a: "keep"})
    with pytest.raises(ReviewError):
        Collection(tmp_path, "../outside.csv")
    (tmp_path / "escape").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ReviewError):
        c.export("escape/out.csv", 0)


def test_legacy_csv_import_and_feedback_cli(tmp_path, monkeypatch):
    c, _ = collection(tmp_path)
    # Importing a legacy CSV is explicit: even its zero rows become decisions.
    c.report.write_text("file,relative_file,selected\na.jpg,a.jpg,0\nb.jpg,b.jpg,1\n")
    monkeypatch.setitem(__import__("sys").modules, "fotosort.embed", None)
    assert main([str(tmp_path), "import"]) == 0
    assert main([str(tmp_path), "prefer", "b.jpg", "a.jpg"]) == 0
    assert main([str(tmp_path), "alternatives", "one moment", "a.jpg", "b.jpg"]) == 0
    assert main([str(tmp_path), "export"]) == 0
    state = ReviewStore(tmp_path).load()
    assert len(state["preferences"]) == 1 and len(state["alternatives"]["one moment"]) == 2


def test_award_manual_choices_remain_separate(tmp_path):
    c, photos = collection(tmp_path)
    _, b = list(c.entries)
    c.change(0, {b: "keep"}, "Personal addition")
    photos[0].award_rank = 1
    apply_manual_decisions(photos, tmp_path)
    assert photos[1].selected and photos[1].auto_selected is False
    assert photos[1].award_rank == 0 and photos[1].decision == "Manual keep: Personal addition"


def test_malformed_store_and_duplicate_report_identity_refused(tmp_path):
    c, _ = collection(tmp_path)
    c.store.path.write_text('{"version":99}')
    with pytest.raises(ReviewError, match="unsupported"):
        c.store.load()
    c.report.write_text("file,relative_file,selected\na.jpg,a.jpg,1\nb.jpg,a.jpg,0\n")
    with pytest.raises(ReviewError, match="Ambiguous"):
        Collection(tmp_path)


def test_external_processed_source_is_not_followed(tmp_path):
    c, _ = collection(tmp_path)
    c.report.write_text("file,relative_file,selected,source\na.jpg,a.jpg,0,/outside/DxO/a.jpg\n")
    entry = next(iter(Collection(tmp_path).entries.values()))
    assert entry["source"] == tmp_path / "a.jpg"
    assert "showing original" in entry["source_note"]
