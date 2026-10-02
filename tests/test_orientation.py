import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from fotosort import cli
from fotosort.enhance import enhance_file, enhancement_report, main
from fotosort.orientation import crop_rotation_corners, load_orienter


def _photo(path, orientation=1):
    pixels = np.zeros((40, 80, 3), dtype=np.uint8)
    pixels[:20, :40] = (230, 20, 40)
    pixels[20:, :40] = (20, 210, 40)
    pixels[:, 40:] = (20, 40, 220)
    exif = Image.Exif()
    exif[274] = orientation
    exif[271] = "Test camera"
    exif[256], exif[257] = 80, 40
    details = exif.get_ifd(0x8769)
    details[36867] = "2026:10:02 12:00:00"
    details[40962], details[40963] = 80, 40
    Image.fromarray(pixels).save(path, exif=exif, quality=100, subsampling=0, icc_profile=b"test ICC")
    return path.read_bytes()


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_is_applied_before_prediction_and_normalized_only_in_copy(
        tmp_path, monkeypatch, rightwayup_stub, orientation):
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    original = _photo(src, orientation)
    model = rightwayup_stub(tier="max", abstain="strict")
    monkeypatch.setattr("fotosort.enhance.enhance", lambda image, *args: image.copy())
    details = enhance_file(src, dst, "general", 1, orienter=model)
    with Image.open(src) as image:
        oriented = ImageOps.exif_transpose(image)
        expected = oriented.transpose(Image.Transpose.ROTATE_90)
        np.testing.assert_array_equal(model.seen[0], np.array(oriented))
    with Image.open(dst) as output:
        assert output.size == expected.size
        assert np.abs(np.array(output).astype(float) - np.array(expected)).mean() < 6
        exif = output.getexif()
        assert exif[274] == 1 and exif[271] == "Test camera"
        assert (exif[256], exif[257]) == output.size
        assert exif.get_ifd(0x8769)[36867] == "2026:10:02 12:00:00"
        assert (exif.get_ifd(0x8769)[40962], exif.get_ifd(0x8769)[40963]) == output.size
        assert output.info["icc_profile"] == b"test ICC"
    assert src.read_bytes() == original
    assert details["orientation_correction_ccw"] == 90


@pytest.mark.parametrize("angle,snap,abstain,correction,status", [
    (89, 90, False, 90, "corrected"), (179, 90, False, 180, "corrected"),
    (269, 90, False, 270, "corrected"), (359, 90, False, 0, "upright"),
    (12.5, 0, False, 12.5, "corrected"), (90, 90, True, 0, "abstained"),
])
def test_report_matches_actual_correction(tmp_path, rightwayup_stub, angle, snap, abstain, correction, status):
    src, dst = tmp_path / "in.jpg", tmp_path / "out.jpg"
    _photo(src)
    model = rightwayup_stub(tier="fast", abstain="strict")
    model.angle, model.abstain = angle, abstain
    details = enhance_file(src, dst, "general", 1, orienter=model, orientation_snap=snap)
    assert details["orientation_angle_cw"] == angle
    assert details["orientation_correction_ccw"] == correction
    assert details["orientation_status"] == status
    assert details["orientation_confidence"] == 0.9
    assert details["orientation_abstain"] == int(abstain)
    assert details["orientation_tier"] == "fast"
    assert len(model.seen) == len(model.corrections) == 1
    with Image.open(dst) as image:
        if snap == 0:
            assert image.width < 80 and image.height < 40
            assert abs(image.width / image.height - 2) < 0.1
            assert details["orientation_crop_fraction"] > 0
        else:
            assert image.size == ((40, 80) if correction in (90, 270) else (80, 40))


@pytest.mark.parametrize("size", [(120, 80), (81, 121), (300, 40), (40, 300)])
@pytest.mark.parametrize("angle", [0.1, 12.5, 44.9, 45, 89.9, 90, 102.5, 179.9, 180, 270, 315, 359.9])
def test_continuous_crop_has_no_empty_corners_and_preserves_upright_aspect(size, angle):
    original = Image.new("RGB", size, "white")
    rotated = original.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    cropped = crop_rotation_corners(rotated, size, angle)
    assert np.array(cropped).min() == 255
    width, height = size if round(angle / 90) % 2 == 0 else size[::-1]
    assert abs(cropped.width - cropped.height * width / height) <= 1 + width / height
    if angle % 90 == 0:
        assert cropped.size == (width, height)


@pytest.mark.parametrize("flags,strength", [(["--no-orientation"], 1), ([], 0)])
def test_disabled_paths_never_import_rightwayup(tmp_path, monkeypatch, flags, strength):
    monkeypatch.setitem(sys.modules, "rightwayup", None)
    args = cli.parse_args([str(tmp_path), *flags])
    assert load_orienter(args, strength) is None


def test_model_setup_failure_precedes_move(tmp_path, monkeypatch):
    src = tmp_path / "in.jpg"
    original = _photo(src)
    monkeypatch.setitem(sys.modules, "rightwayup", None)
    args = cli.parse_args([str(tmp_path), "--move", "--enhance"])
    with pytest.raises(SystemExit, match="RightWayUp is required"):
        cli.move_picks([cli.Photo(src, "k")], tmp_path, args, None)
    assert src.read_bytes() == original
    assert not (tmp_path / "Highlights").exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_source_aliases_cannot_be_overwritten(tmp_path, rightwayup_stub, alias):
    src = tmp_path / "in.jpg"
    original = _photo(src)
    dst = src if alias == "same" else tmp_path / "alias.jpg"
    if alias == "symlink":
        dst.symlink_to(src)
    elif alias == "hardlink":
        dst.hardlink_to(src)
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        enhance_file(src, dst, "general", 1, orienter=rightwayup_stub("max", "strict"))
    assert src.read_bytes() == original


def test_raw_decode_and_zero_strength(tmp_path, monkeypatch, rightwayup_stub):
    src, dst = tmp_path / "in.nef", tmp_path / "out.jpg"
    src.write_bytes(b"RAW original")
    monkeypatch.setattr("fotosort.quality.open_full", lambda _: Image.new("RGB", (80, 40), "red"))
    model = rightwayup_stub("max", "strict")
    assert enhance_file(src, dst, "general", 1, orienter=model)["orientation_correction_ccw"] == 90
    assert Image.open(dst).size == (40, 80)
    details = enhance_file(src, dst, "general", 0, orienter=model)
    assert details["orientation_status"] == "zero_strength"
    assert Image.open(dst).size == (80, 40) and len(model.seen) == 1
    assert src.read_bytes() == b"RAW original"


@pytest.mark.parametrize("flags,expected", [([], "corrected"), (["--no-orientation"], "disabled"),
                                           (["--strength", "0"], "zero_strength")])
def test_standalone_reuses_one_model_and_writes_report(tmp_path, rightwayup_stub, flags, expected):
    originals = {name: _photo(tmp_path / name) for name in ("a.jpg", "b.jpg")}
    assert main([str(tmp_path), "--style", "general", *flags]) == 0
    with (tmp_path / "Enhanced" / "fotosort_enhancement.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(row["orientation_status"] == expected for row in rows)
    if expected == "corrected":
        assert len(rightwayup_stub.instances) == 1
        assert len(rightwayup_stub.instances[0].seen) == 2
    else:
        assert not rightwayup_stub.instances
    assert all((tmp_path / name).read_bytes() == contents for name, contents in originals.items())


def test_standalone_never_replaces_existing_photos_in_output_directory(tmp_path, rightwayup_stub):
    src = tmp_path / "in.jpg"
    original = _photo(src)
    for _ in range(2):
        assert main([str(src), "--out", str(tmp_path), "--style", "general"]) == 0
    assert src.read_bytes() == original
    assert (tmp_path / "in_1.jpg").exists() and (tmp_path / "in_2.jpg").exists()


@pytest.mark.parametrize("operation", ["--copy", "--move"])
def test_apply_report_enhancement_leaves_selection_report_unchanged(tmp_path, rightwayup_stub, operation):
    src = tmp_path / "in.jpg"
    original = _photo(src)
    report = tmp_path / "fotosort_report.csv"
    report.write_text(f"file,selected,edit_benefit,preset\n{src},1,low,Natural\n")
    before = report.read_bytes()
    assert cli.main([str(tmp_path), "--apply-report", operation, "--enhance",
                     "--enhance-style", "general", "--orientation-tier", "fast"]) == 0
    with (tmp_path / "Highlights" / "fotosort_enhancement.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert row["orientation_tier"] == "fast" and row["orientation_status"] == "corrected"
    assert Path(row["source"]).read_bytes() == original
    assert Path(row["output"]).is_file()
    assert src.exists() == (operation == "--copy")
    assert report.read_bytes() == before


@pytest.mark.parametrize("flags", [["--enhance"], ["--copy"]])
def test_dry_run_and_plain_copy_never_load_model(tmp_path, monkeypatch, flags):
    src = tmp_path / "in.jpg"
    _photo(src)
    (tmp_path / "fotosort_report.csv").write_text(
        f"file,selected,edit_benefit,preset\n{src},1,low,Natural\n")
    monkeypatch.setitem(sys.modules, "rightwayup", None)
    assert cli.main([str(tmp_path), "--apply-report", *flags, "--enhance-style", "general"]) == 0
    assert not list(tmp_path.rglob("fotosort_enhancement*.csv"))


def test_export_reports_preserve_previous_runs_and_completed_rows_after_failure(tmp_path):
    for _ in range(2):
        with pytest.raises(RuntimeError), enhancement_report(tmp_path) as record:
            record("in.jpg", "out.jpg", "general", 1, {"orientation_status": "disabled"})
            raise RuntimeError("later image failed")
    reports = list(tmp_path.glob("fotosort_enhancement*.csv"))
    assert len(reports) == 2
    for path in reports:
        with path.open() as stream:
            assert len(list(csv.DictReader(stream))) == 1
