import copy
import json
import sys

import pytest

from fotosort.__main__ import main
from fotosort.evaluate import evaluate, export_feedback
from fotosort.review_data import Collection, ReviewError, fingerprint, write_rows


@pytest.fixture
def fixture(tmp_path):
    photos = [dict(id=name, file=f"{name}.jpg") for name in "abcd"]
    rows = [dict(file=p["file"], relative_file=p["file"], selected=str(int(p["id"] in "bc")),
                 auto_selected=str(int(p["id"] in "bc")), shortlisted=str(int(p["id"] in "abc"))) for p in photos]
    report = tmp_path / "baseline.csv"
    write_rows(report, list(rows[0]), rows)
    candidate = copy.deepcopy(rows)
    for row in candidate:
        row["selected"] = row["auto_selected"] = str(int(row["file"] == "a.jpg"))
    write_rows(tmp_path / "candidate.csv", list(rows[0]), candidate)
    run = dict(report="baseline.csv", report_sha256=fingerprint(report),
               budget=dict(candidate_limit=3, request_limit=1),
               config={"private-path": "/PRIVATE/config"}, models={"local": "/PRIVATE/model"},
               usage=dict(requests=1, input_tokens=123, runtime_seconds=99), review_seconds=30, manual_replacements=1)
    shoot = dict(id="/PRIVATE/shoot", split="diagnostic", photos=photos, favourites=["a"],
                 rejected=["c"], acceptable=[["a", "b"], ["c", "d"]],
                 preferences=[dict(winner="a", loser="b")],
                 runs=dict(baseline=run, candidate={**run, "report": "candidate.csv",
                     "report_sha256": fingerprint(tmp_path / "candidate.csv")}))
    path = tmp_path / "evaluation.json"
    document = dict(version=1, shoots=[shoot])
    path.write_text(json.dumps(document))
    return path, document, rows


def save(fixture):
    path, document, _ = fixture
    path.write_text(json.dumps(document))
    return path


def test_exact_and_moment_recall_are_distinct_and_private(fixture, monkeypatch):
    monkeypatch.setitem(sys.modules, "fotosort.embed", None)
    monkeypatch.setitem(sys.modules, "fotosort.judge", None)
    path, _, _ = fixture
    result = evaluate(path, "baseline")
    shoot = result["shoots"][0]
    assert shoot["metrics"]["shortlist_exact"] == dict(hits=1, total=1, unknown=0, recall=1.0)
    assert shoot["metrics"]["selection_exact"]["recall"] == 0
    assert shoot["metrics"]["selection_moments"]["recall"] == 1
    assert shoot["pairwise"]["other_only"] == 1
    assert shoot["candidate_count"] == 3 and shoot["selected_count"] == 2
    assert shoot["usage"]["requests"] == 1
    assert shoot["review_seconds"] == 30 != shoot["usage"]["runtime_seconds"]
    assert shoot["manual_replacements"] == 1
    assert len(shoot["corrections_required"]["add"]) == 1
    serialized = json.dumps(result)
    assert "/PRIVATE" not in serialized and "a.jpg" not in serialized and "baseline.csv" not in serialized
    assert result == evaluate(path, "baseline")


def test_unknown_exposure_and_missing_rows_do_not_become_negative_labels(fixture):
    path, doc, rows = fixture
    shoot = doc["shoots"][0]
    shoot["favourites"] = ["a", "d"]
    rows[0]["shortlisted"] = ""
    write_rows(path.parent / "baseline.csv", list(rows[0]), rows[:3])
    shoot["runs"]["baseline"].pop("report_sha256")
    result = evaluate(save(fixture), "baseline")["shoots"][0]
    assert result["metrics"]["shortlist_exact"] == dict(hits=0, total=0, unknown=2, recall=None)
    assert result["metrics"]["selection_exact"] == dict(hits=0, total=1, unknown=1, recall=0)
    assert result["coverage"]["unevaluated"] == 1
    assert result["coverage"]["shortlist_unknown"] == 2
    assert result["metrics"]["selection_moments"]["recall"] == 1  # a known acceptable hit is enough


def test_missing_changed_and_incomplete_are_separate(fixture):
    path, doc, _ = fixture
    shoot = doc["shoots"][0]
    root = path.parent / "photos"
    root.mkdir()
    for name in "abc":
        (root / f"{name}.jpg").write_bytes(name.encode())
    shoot["root"] = "photos"
    shoot["photos"][1]["sha256"] = "0" * 64
    shoot["runs"]["baseline"]["judgments"] = {"a": "incomplete", "c": "failed"}
    result = evaluate(save(fixture), "baseline")["shoots"][0]
    assert result["coverage"]["missing_files"] == result["coverage"]["changed_files"] == 1
    assert result["coverage"]["incomplete_judgments"] == result["coverage"]["failed_judgments"] == 1
    assert result["metrics"]["selection_exact"]["unknown"] == 1
    assert result["metrics"]["selection_exact"]["recall"] is None
    assert result["metrics"]["shortlist_exact"]["recall"] == 1


def test_empty_annotations_and_absent_measurements_are_unknown(fixture):
    _, doc, _ = fixture
    shoot = doc["shoots"][0]
    for key in ("favourites", "acceptable", "preferences"):
        shoot.pop(key)
    for key in ("usage", "review_seconds", "manual_replacements"):
        shoot["runs"]["baseline"].pop(key)
    result = evaluate(save(fixture), "baseline")
    row = result["shoots"][0]
    assert all(m["recall"] is None for m in row["metrics"].values())
    assert row["review_seconds"] is row["manual_replacements"] is None
    assert row["usage"] == {}
    assert result["splits"]["diagnostic"]["review_seconds"] is None


def test_manual_overrides_never_improve_automated_recall(fixture):
    path, doc, rows = fixture
    rows[0]["selected"] = "1"
    rows[0]["manual_decision"] = "keep"
    write_rows(path.parent / "baseline.csv", list(rows[0]), rows)
    doc["shoots"][0]["runs"]["baseline"].pop("report_sha256")
    result = evaluate(save(fixture), "baseline")["shoots"][0]
    assert result["metrics"]["selection_exact"]["recall"] == 0
    for row in rows:
        row.pop("auto_selected")
    write_rows(path.parent / "baseline.csv", list(rows[0]), rows)
    with pytest.raises(ReviewError, match="auto_selected"):
        evaluate(path, "baseline")


def test_duplicate_picks_and_pairwise_outcomes(fixture):
    path, doc, rows = fixture
    shoot = doc["shoots"][0]
    shoot["acceptable"] = [["a", "b", "c"]]
    shoot["preferences"] += [dict(winner="b", loser="c"), dict(winner="b", loser="c")]
    result = evaluate(save(fixture), "baseline")["shoots"][0]
    assert result["redundant_picks"]["excess_in_annotated_moments"] == 1
    assert result["pairwise"] == dict(preferred_only=0, other_only=1, both=1, neither=0, unknown=0)


def test_compare_frozen_reports_and_equal_budgets(fixture):
    path, _, _ = fixture
    result = evaluate(path, "candidate", "baseline")
    assert result["comparison"][0]["recall_delta"]["selection_exact"] == 1
    assert result["comparison"][0]["recall_delta"]["shortlist_exact"] == 0
    assert result == evaluate(path, "candidate", "baseline")
    report = path.parent / "baseline.csv"
    report.write_text(report.read_text() + "\n")
    with pytest.raises(ReviewError, match="Frozen report"):
        evaluate(path, "candidate", "baseline")


@pytest.mark.parametrize("change", ["budget", "cohort", "hash", "content"])
def test_invalid_comparisons_are_refused(fixture, change):
    path, doc, rows = fixture
    shoot = doc["shoots"][0]
    if change == "budget":
        shoot["runs"]["candidate"]["budget"] = dict(candidate_limit=8, request_limit=1)
    elif change == "hash":
        shoot["runs"]["baseline"].pop("report_sha256")
    elif change == "content":
        shoot["photos"][0]["sha256"] = "a" * 64
        rows[0]["photo_sha256"] = "b" * 64
        write_rows(path.parent / "candidate.csv", list(rows[0]), rows)
        shoot["runs"]["candidate"].pop("report_sha256")
    else:
        write_rows(path.parent / "candidate.csv", list(rows[0]), rows[:3])
        shoot["runs"]["candidate"].pop("report_sha256")
    with pytest.raises(ReviewError):
        evaluate(save(fixture), "candidate", "baseline")


def test_coverage_changes_do_not_produce_comparable_delta(fixture):
    _, doc, _ = fixture
    doc["shoots"][0]["runs"]["candidate"]["judgments"] = {"a": "incomplete"}
    result = evaluate(save(fixture), "candidate", "baseline")
    assert result["comparison"][0]["recall_delta"]["selection_exact"] is None


def test_splits_aggregate_counts_without_mixing_validation(fixture):
    _, doc, _ = fixture
    second = copy.deepcopy(doc["shoots"][0])
    second.update(id="development", split="development", favourites=["b", "c"], rejected=[])
    third = copy.deepcopy(second)
    third.update(id="heldout", split="heldout", favourites=["a", "c"])
    doc["shoots"] += [second, third]
    result = evaluate(save(fixture), "baseline")
    assert set(result["splits"]) == {"development", "heldout", "diagnostic"}
    assert result["splits"]["development"]["metrics"]["selection_exact"]["recall"] == 1
    assert result["splits"]["heldout"]["metrics"]["selection_exact"]["recall"] == .5


def test_feedback_roundtrip_excludes_unreviewed_negatives(tmp_path):
    for name in "abc":
        (tmp_path / f"{name}.jpg").write_bytes(name.encode())
    rows = [dict(file=f"{name}.jpg", relative_file=f"{name}.jpg", selected=str(int(name == "a")), shortlisted="1")
            for name in "abc"]
    write_rows(tmp_path / "fotosort_report.csv", list(rows[0]), rows)
    collection = Collection(tmp_path)
    a, b, c = collection.entries
    collection.change(0, {a: "reject", b: "keep"}, preference=[b, a], alternatives=["moment", [b, c]])
    manifest = tmp_path / "evaluation.json"
    document = export_feedback(tmp_path, "fotosort_report.csv", manifest, "heldout", 75, 1)
    shoot = document["shoots"][0]
    assert shoot["favourites"] == [b] and shoot["rejected"] == [a]
    assert c not in shoot["rejected"]
    assert "note" not in json.dumps(document)
    result = evaluate(manifest, "baseline")["shoots"][0]
    assert result["metrics"]["selection_exact"]["recall"] == 0
    assert len(result["corrections_required"]["remove"]) == 1
    (tmp_path / "b.jpg").write_bytes(b"changed")
    with pytest.raises(ReviewError):
        export_feedback(tmp_path, "fotosort_report.csv", manifest, "diagnostic")


def test_cli_writes_summary_and_protects_inputs(fixture, capsys):
    path, _, _ = fixture
    output = path.parent / "summary.json"
    assert main(["evaluate", "run", str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["mode"] == "saved_reports"
    for target in (path, path.parent / "baseline.csv"):
        before = target.read_bytes()
        with pytest.raises(SystemExit) as exc:
            main(["evaluate", "run", str(path), "--output", str(target)])
        assert exc.value.code == 2 and target.read_bytes() == before
    assert "must not overwrite" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [None, [], {"version": 7}, {"version": 1, "shoots": []}])
def test_malformed_manifests_fail_cleanly(tmp_path, bad):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ReviewError):
        evaluate(path, "baseline")


def test_comparison_rejects_changed_report_content_without_manifest_hashes(fixture):
    path, doc, rows = fixture
    rows[0]["photo_sha256"] = "c" * 64
    write_rows(path.parent / "candidate.csv", list(rows[0]), rows)
    doc["shoots"][0]["runs"]["candidate"].pop("report_sha256")
    with pytest.raises(ReviewError, match="content identities"):
        evaluate(save(fixture), "candidate", "baseline")


def test_actual_usage_above_declared_budget_cannot_be_compared(fixture):
    _, doc, _ = fixture
    doc["shoots"][0]["runs"]["candidate"]["usage"] = dict(requests=2)
    path = save(fixture)
    assert evaluate(path, "candidate")["shoots"][0]["budget_compliance"]["requests"] is False
    with pytest.raises(ReviewError, match="exceeds"):
        evaluate(path, "candidate", "baseline")


def test_review_duration_belongs_to_the_measured_run(fixture):
    _, doc, _ = fixture
    for key in ("review_seconds", "manual_replacements"):
        doc["shoots"][0]["runs"]["candidate"].pop(key)
    result = evaluate(save(fixture), "candidate", "baseline")
    assert result["shoots"][0]["review_seconds"] is None
    assert result["baseline"]["shoots"][0]["review_seconds"] == 30


def test_feedback_preserves_old_identity_for_changed_unreviewed_source(tmp_path):
    from fotosort.review_data import photo_id

    source = tmp_path / "a.jpg"
    source.write_bytes(b"original")
    original_hash = fingerprint(source)
    rows = [dict(file="a.jpg", relative_file="a.jpg", photo_sha256=original_hash,
                 photo_id=photo_id("a.jpg", original_hash), selected="0")]
    write_rows(tmp_path / "fotosort_report.csv", list(rows[0]), rows)
    source.write_bytes(b"changed")
    manifest = tmp_path / "evaluation.json"
    export_feedback(tmp_path, "fotosort_report.csv", manifest, "diagnostic")
    result = evaluate(manifest, "baseline")["shoots"][0]
    assert result["coverage"]["changed_files"] == 1
