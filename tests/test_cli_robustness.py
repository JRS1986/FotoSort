from pathlib import Path

from fotosort.cli import Photo, _decode


def test_decode_reports_error_instead_of_raising(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    ph, sh, err, tensor, small = _decode(Photo(path=bad, key="k"), lambda img: img)
    assert sh is None and tensor is None and isinstance(err, Exception)
    ph, sh, err, tensor, small = _decode(Photo(path=tmp_path / "missing.jpg", key="k"), lambda img: img)
    assert sh is None and isinstance(err, FileNotFoundError)


def test_enhanced_output_of_a_raw_is_a_jpeg():
    from fotosort.cli import _jpg_name
    assert _jpg_name(Path("x/P123.ORF")) == Path("x/P123.jpg")
    assert _jpg_name(Path("x/P123.JPG")) == Path("x/P123.JPG")


def test_layout_dir_variants():
    from datetime import datetime

    from fotosort.cli import layout_dir
    ph = Photo(path=Path("x.jpg"), key="k", label="rock hyrax (dassie)", time=datetime(2026, 9, 1))
    out = Path("/out")
    assert layout_dir(out, ph, "flat") == out
    assert layout_dir(out, ph, "species") == out / "rock hyrax (dassie)"
    assert layout_dir(out, ph, "day") == out / "2026-09-01"
    assert layout_dir(out, ph, "day-species") == out / "2026-09-01" / "rock hyrax (dassie)"
    ph.label = "a/b:c"
    assert layout_dir(out, ph, "species") == out / "a_b_c"


def test_forced_includes_resolve_stems_names_and_files(tmp_path, capsys):
    from fotosort.cli import forced_includes
    photos = [Photo(path=Path("d/P9051472.JPG"), key="a", reject="blurry"), Photo(path=Path("d/P9050886.JPG"), key="b")]
    got = forced_includes(photos, "p9051472,P9050886.JPG,nope")
    assert [p.path.stem for p in got] == ["P9051472", "P9050886"] and got[0].reject == ""
    assert "nope" in capsys.readouterr().out
    (tmp_path / "list.txt").write_text("P9050886\n")
    assert [p.path.stem for p in forced_includes(photos, f"@{tmp_path / 'list.txt'}")] == ["P9050886"]


def test_apply_report_survives_a_folder_rename(tmp_path):
    from PIL import Image

    from fotosort.cli import apply_report, parse_args
    old = tmp_path / "old"
    old.mkdir()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(old / "IMG_1.jpg")
    (old / "fotosort_report.csv").write_text(
        "file,label,selected,person,edit_benefit,preset\n" f"{old / 'IMG_1.jpg'},zebra,1,0,low,Natural\n")
    new = tmp_path / "new"
    old.rename(new)
    args = parse_args([str(new), "--apply-report"])
    photos = [Photo(path=new / "IMG_1.jpg", key="k")]
    assert apply_report(photos, new, args) == 0
    assert photos[0].selected and photos[0].label == "zebra"
