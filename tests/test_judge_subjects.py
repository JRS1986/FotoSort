import csv
from pathlib import Path

from fotosort.cli import Photo, layout_dir, name_pick_subjects, update_report_column
from fotosort.judge import parse_subjects


def test_parse_subjects_maps_numbers_to_paths_and_skips_junk():
    ids = ["/a/1.jpg", "/a/2.jpg", "/a/3.jpg"]
    text = ('{"subjects": [{"photo": 1, "subject": "  Cape Fur Seal "}, {"photo": 9, "subject": "x"}, '
            '{"photo": 3, "subject": ""}, "junk", {"photo": 2, "subject": "kelp gull"}]}')
    assert parse_subjects(text, ids) == {"/a/1.jpg": "cape fur seal", "/a/2.jpg": "kelp gull"}


def test_parse_subjects_tolerates_prose_around_json():
    assert parse_subjects('Sure: {"subjects": [{"photo": 1, "subject": "shark"}]} done', ["p"]) == {"p": "shark"}
    assert parse_subjects("not json", ["p"]) == {}


class _FakeJudge:
    sources = {}

    def name_subjects(self, paths, vocab, chunk=12):
        return {str(p): "cape fur seal" for p in paths[:1]}


def test_judge_label_wins_in_species_layout(capsys):
    a = Photo(path=Path("/x/a.jpg"), key="a", label="dolphin with a curved dorsal fin", selected=True)
    b = Photo(path=Path("/x/b.jpg"), key="b", label="people", selected=True)
    name_pick_subjects([a, b], _FakeJudge(), ["seal"])
    assert a.judge_label == "cape fur seal" and b.judge_label == ""
    assert layout_dir(Path("/out"), a, "species") == Path("/out/cape fur seal")
    assert layout_dir(Path("/out"), b, "species") == Path("/out/people")
    assert layout_dir(Path("/out"), a, "flat") == Path("/out")
    assert "1 differ" in capsys.readouterr().out


def test_update_report_column_adds_column_and_keeps_rows(tmp_path):
    report = tmp_path / "r.csv"
    report.write_text("file,label,selected\n/x/a.jpg,seal,1\n/x/b.jpg,gull,0\n")
    update_report_column(report, "judge_label", {"/x/a.jpg": "cape fur seal"})
    with open(report, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["judge_label"] for r in rows] == ["cape fur seal", ""]
    assert rows[1]["label"] == "gull" and not report.with_suffix(".tmp").exists()
