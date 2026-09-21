import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from fotosort.cli import Photo, apply_report, parse_args, write_report
from fotosort.decisions import main
from fotosort.review_data import (
    Collection,
    Conflict,
    ReviewError,
    ReviewStore,
    apply_manual_decisions,
    fingerprint,
    relative_name,
)


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
    # A changed frame that is not a pick is never copied, so it does not block the picks.
    (tmp_path / "a.jpg").write_bytes(b"new content")
    fresh = [Photo(tmp_path / "a.jpg", "a"), Photo(tmp_path / "b.jpg", "b")]
    assert apply_report(fresh, tmp_path, args) == 0
    (tmp_path / "b.jpg").write_bytes(b"new content")
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


def test_colon_in_names_is_an_ordinary_posix_character(tmp_path):
    # Finder stores "Safari 5/12" as "Safari 5:12"; only a bare Windows drive is refused.
    c, _ = collection(tmp_path, ("Safari 5:12/IMG:1.jpg", "b.jpg"))
    assert [e["relative_file"] for e in c.entries.values()] == ["Safari 5:12/IMG:1.jpg", "b.jpg"]
    assert all(row["photo_id"] for row in c.rows)
    with pytest.raises(ReviewError):
        relative_name("C:/Users/photo.jpg")


def test_cleared_override_exports_the_automatic_explanation(tmp_path):
    c, photos = collection(tmp_path)
    a, _ = list(c.entries)
    c.change(0, {a: "reject"}, "Soft eyes")
    apply_manual_decisions(photos, tmp_path)
    write_report(photos, c.report, tmp_path)
    rerun = Collection(tmp_path)
    assert rerun.rows[0]["decision"] == "Manual reject: Soft eyes"
    rerun.change(1, {a: "clear"})
    with rerun.export("reviewed.csv", 2).open() as stream:
        row = next(csv.DictReader(stream))
    assert (row["selected"], row["manual_decision"], row["decision"]) == ("1", "", "Selected by score")


def test_manual_reason_keeps_its_own_punctuation(tmp_path):
    c, _ = collection(tmp_path)
    a, _ = list(c.entries)
    c.change(0, {a: "keep"}, "Compare with frame 12:")
    with c.export("reviewed.csv", 1).open() as stream:
        assert next(csv.DictReader(stream))["decision"] == "Manual keep: Compare with frame 12:"


def test_missing_pick_stays_visible_without_blocking_export(tmp_path):
    c, _ = collection(tmp_path)
    (tmp_path / "a.jpg").unlink()
    reopened = Collection(tmp_path)
    with reopened.export("reviewed.csv", 0).open() as stream:
        rows = list(csv.DictReader(stream))
    assert [(r["selected"], r["review_status"]) for r in rows] == [("1", "missing"), ("0", "available")]
    with pytest.raises(Conflict, match="a.jpg"):
        reopened.change(0, {next(iter(reopened.entries)): "reject"})


def test_export_refuses_to_replace_the_open_report(tmp_path):
    c, _ = collection(tmp_path)
    before = c.report.read_bytes()
    with pytest.raises(ReviewError, match="overwrite the report"):
        c.export("fotosort_report.csv", 0)
    assert c.report.read_bytes() == before


def test_photo_moved_inside_the_collection_is_reported_not_transferred(tmp_path):
    c, photos = collection(tmp_path, ("sub/a.jpg", "b.jpg"))
    a, _ = list(c.entries)
    c.change(0, {a: "reject"})
    (tmp_path / "other").mkdir()
    (tmp_path / "sub/a.jpg").rename(tmp_path / "other/a.jpg")
    moved = [Photo(tmp_path / "other/a.jpg", "a", selected=True, digest=fingerprint(tmp_path / "other/a.jpg")),
             Photo(tmp_path / "b.jpg", "b", digest=fingerprint(tmp_path / "b.jpg"))]
    warnings = apply_manual_decisions(moved, tmp_path)
    assert moved[0].selected and moved[0].manual_decision == ""
    assert moved[0].review_status == "decision recorded for sub/a.jpg; not transferred"
    assert any("sub/a.jpg" in w for w in warnings) and any("no longer in this collection" in w for w in warnings)
    write_report(moved, c.report, tmp_path)
    assert Collection(tmp_path).view()[0]["review_status"] == "decision recorded for sub/a.jpg; not transferred"


def test_undo_history_stores_changes_not_copies_of_the_state(tmp_path):
    names = tuple(f"{i:03}.jpg" for i in range(40))
    c, _ = collection(tmp_path, names)
    ids = list(c.entries)
    state = c.change(0, dict.fromkeys(ids, "keep"))
    baseline = len(c.store.path.read_bytes())
    for identifier in ids[:30]:
        state = c.change(state["revision"], {identifier: "reject"})
    assert len(c.store.path.read_bytes()) < 3 * baseline
    assert all(set(entry) == {"delta"} for entry in state["history"])
    state = c.undo(state["revision"])
    assert state["decisions"][ids[29]]["choice"] == "keep"
    for _ in range(30):
        state = c.undo(state["revision"])
    assert state["decisions"] == {}
    with pytest.raises(ReviewError, match="Nothing to undo"):
        c.undo(state["revision"])


def test_earlier_snapshot_history_still_undoes(tmp_path):
    c, _ = collection(tmp_path)
    a, _ = list(c.entries)
    state = c.change(0, {a: "keep"})
    state["history"] = [dict(decisions={}, preferences=[], alternatives={})]
    c.store.path.write_text(json.dumps(state))
    assert c.undo(1)["decisions"] == {}


def test_reversed_preference_replaces_the_earlier_verdict(tmp_path):
    c, _ = collection(tmp_path)
    a, b = list(c.entries)
    c.change(0, {}, preference=[a, b])
    state = c.change(1, {}, preference=[b, a])
    assert [(p["winner"], p["loser"]) for p in state["preferences"]] == [(b, a)]


def test_lazy_open_hashes_only_the_photos_a_decision_relies_on(tmp_path, monkeypatch):
    collection(tmp_path)
    hashed = []
    real = fingerprint
    monkeypatch.setattr("fotosort.review_data.fingerprint", lambda path: hashed.append(path.name) or real(path))
    lazy = Collection(tmp_path, verify_sources=False)
    assert hashed == ["fotosort_report.csv"]
    lazy.change(0, {list(lazy.entries)[1]: "keep"})
    assert hashed == ["fotosort_report.csv", "b.jpg"]
    (tmp_path / "a.jpg").write_bytes(b"new content")
    with pytest.raises(Conflict):
        lazy.change(1, {list(lazy.entries)[0]: "keep"})


def test_import_is_recorded_as_an_import_not_an_explicit_review(tmp_path):
    c, _ = collection(tmp_path)
    assert main([str(tmp_path), "import"]) == 0
    assert {d["origin"] for d in c.store.load()["decisions"].values()} == {"csv import"}


def test_forced_include_is_not_recorded_as_an_automatic_pick(tmp_path):
    _, photos = collection(tmp_path)
    forced = photos[1]
    forced.auto_selected, forced.auto_decision = forced.selected, forced.decision
    forced.selected, forced.decision = True, "Forced by --include"
    apply_manual_decisions(photos, tmp_path)
    write_report(photos, tmp_path / "fotosort_report.csv", tmp_path)
    row = Collection(tmp_path).rows[1]
    assert (row["selected"], row["auto_selected"], row["auto_decision"]) == ("1", "0", "Group budget filled")


def test_written_reports_follow_the_umask_not_the_temporary_file_mode(tmp_path):
    c, _ = collection(tmp_path)
    umask = os.umask(0)
    os.umask(umask)
    assert c.report.stat().st_mode & 0o777 == 0o666 & ~umask
    c.report.chmod(0o640)
    c.export("reviewed.csv", 0)
    write_report([], c.report, tmp_path)
    assert c.report.stat().st_mode & 0o777 == 0o640


def test_invalid_review_record_stops_the_run_before_analysis(tmp_path, monkeypatch, capsys):
    from fotosort import cli

    Image.new("RGB", (24, 16)).save(tmp_path / "a.jpg")
    (tmp_path / ".fotosort_review.json").write_text("{}")
    monkeypatch.setattr(cli, "compute_features", lambda *args: pytest.fail("analysis started"))
    assert cli.main([str(tmp_path)]) == 2
    assert "Invalid or unsupported review file" in capsys.readouterr().err


def test_lazy_verification_records_changed_and_missing_status(tmp_path):
    collection(tmp_path)
    lazy = Collection(tmp_path, verify_sources=False)
    a, b = lazy.entries
    (tmp_path / "a.jpg").write_bytes(b"changed contents")
    (tmp_path / "b.jpg").unlink()
    for identifier in (a, b):
        with pytest.raises(Conflict):
            lazy.verify(identifier)
    assert [r["status"] for r in lazy.view()] == ["changed", "missing"]
    assert lazy.entries[a]["stamp"] is not None
    assert lazy.entries[b]["stamp"] is None


def test_verified_photo_restored_after_missing_is_verified_again(tmp_path):
    c, _ = collection(tmp_path)
    identifier = next(iter(c.entries))
    path = tmp_path / "a.jpg"
    backup = tmp_path / "a.backup"
    path.rename(backup)
    with pytest.raises(Conflict):
        c.verify(identifier)
    backup.rename(path)  # identical size/mtime to the previously verified photo
    assert c.verify(identifier)["status"] == "available"
