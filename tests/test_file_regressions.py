import base64
import csv
import io

import numpy as np
import pytest
from PIL import Image

from fotosort.cli import Photo, apply_report, attach_dxo, parse_args, raw_cull, write_report
from fotosort.judge import Judge, _native_window, subject_crop, subject_crops
from fotosort.scan import find_images


def test_native_detail_preserves_pixels_and_clamps_to_image_edges():
    data = np.random.default_rng(2).integers(0, 256, (800, 1200, 3), dtype=np.uint8)
    img = Image.fromarray(data)
    assert np.array_equal(np.asarray(_native_window(img, 600, 400, 512)), data[144:656, 344:856])
    assert np.array_equal(np.asarray(_native_window(img, 1200, 0, 512)), data[:512, -512:])


def test_subject_crops_use_raw_loader_and_send_overview_and_native_windows(tmp_path, monkeypatch):
    calls = []

    def load(path):
        calls.append(path)
        return Image.new("RGB", (4000, 3000), "red")

    monkeypatch.setattr("fotosort.judge.open_full", load)
    path = tmp_path / "a.ORF"
    box = (0.2, 0.2, 0.8, 0.8)
    encoded = subject_crop(path, box)
    assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (512, 512)
    crops = subject_crops(path, box)
    assert len(crops) == 4 and "overview" in crops[0][0]
    for name, encoded in crops[1:]:
        assert "100%" in name
        assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (512, 512)
    assert calls == [str(path), str(path)]


def test_judge_uses_processed_source_and_reuses_encoded_images(tmp_path, monkeypatch):
    from fotosort.cli import select_photos

    original, processed = tmp_path / "original.jpg", tmp_path / "processed.jpg"
    Image.new("RGB", (64, 64), "red").save(original)
    Image.new("RGB", (64, 64), "blue").save(processed)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    judge = Judge()
    seen = []

    def ask(images, prompt):
        seen.append(images)
        return '{"picks":[{"photo":1,"reason":"sharp"}]}'

    judge._ask_openai = ask
    photo = Photo(original, "k", source=processed, label="bird", emb=np.array([1., 0.]), score=1)
    args = parse_args([str(tmp_path), "--judge"])
    select_photos([photo], args, judge)
    assert photo.selected
    image = Image.open(io.BytesIO(base64.b64decode(seen[0][0][0])))
    assert np.asarray(image)[..., 2].mean() > 200 and np.asarray(image)[..., 0].mean() < 10
    # A second round should not decode again.
    monkeypatch.setattr("fotosort.judge._jpeg_b64", lambda *a: pytest.fail("decoded a cached frame"))
    judge.judge([original], "bird", "today", 1)
    assert len(seen) == 2


def test_raw_jpeg_in_subfolder_is_one_capture_and_outputs_are_excluded(tmp_path):
    for folder in ("RAW", "DxO", "Enhanced", "_DELETE_ME_raw"):
        (tmp_path / folder).mkdir()
    (tmp_path / "IMG_1.JPG").touch()
    (tmp_path / "RAW" / "img_1.ORF").touch()
    (tmp_path / "RAW" / "IMG_2.ORF").touch()
    for folder in ("DxO", "Enhanced", "_DELETE_ME_raw"):
        (tmp_path / folder / "x.dng").touch()
    assert [str(p.relative_to(tmp_path)) for p in find_images(tmp_path, True, set())] == [
        "IMG_1.JPG", "RAW/IMG_2.ORF"]


def test_raw_keep_overrides_cull_before_any_files_are_moved(tmp_path):
    (tmp_path / "RAW").mkdir()
    raw = tmp_path / "RAW" / "IMG_1.ORF"
    raw.write_bytes(b"placeholder RAW")
    jpeg = tmp_path / "IMG_1.jpg"
    jpeg.touch()
    photos = [Photo(jpeg, "jpeg", selected=True, edit_band="high"), Photo(raw, "raw")]
    args = parse_args([str(tmp_path), "--raw-cull", "--raw-cull-move"])
    raw_cull(photos, tmp_path, args)
    assert raw.exists()
    assert str(raw.resolve()) in (tmp_path / "raw_keep.txt").read_text()
    assert not (tmp_path / "raw_cull.txt").read_text().strip()


def test_dxo_attaches_raw_sibling_of_jpeg_from_external_output_folder(tmp_path):
    root, output = tmp_path / "photos", tmp_path / "processed"
    root.mkdir()
    output.mkdir()
    jpeg, raw, twin = root / "IMG.jpg", root / "IMG.ORF", output / "IMG-DxO.dng"
    for p in (jpeg, raw, twin):
        p.touch()
    photo = Photo(jpeg, "old")
    args = parse_args([str(root), "--dxo", "all", "--dxo-wait", "0", "--dxo-dir", str(output)])
    attach_dxo([photo], root, args, only=None)
    assert photo.source == twin
    assert "../processed/IMG-DxO.dng" in photo.key


@pytest.mark.parametrize("legacy", [False, True])
def test_report_matches_duplicate_filenames_after_folder_move(tmp_path, legacy):
    old = tmp_path / "old"
    photos = []
    for i in (1, 2):
        folder = old / f"day{i}"
        folder.mkdir(parents=True)
        path = folder / "IMG_1.jpg"
        path.touch()
        photos.append(Photo(path, str(i), selected=i == 1, label="bird", edit_band="low", preset="Natural"))
    report = old / "fotosort_report.csv"
    write_report(photos, report, old)
    if legacy:
        with report.open() as stream:
            rows = list(csv.DictReader(stream))
        cols = [c for c in rows[0] if c != "relative_file"]
        with report.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    new = tmp_path / "new"
    old.rename(new)
    photos = [Photo(p, str(p)) for p in find_images(new, True, set())]
    args = parse_args([str(new), "--apply-report"])
    assert apply_report(photos, new, args) == 0
    assert [p.path.parent.name for p in photos if p.selected] == ["day1"]


def test_report_refuses_ambiguous_basename_before_copying(tmp_path):
    photos = []
    for sub in ("a", "b"):
        folder = tmp_path / sub
        folder.mkdir()
        path = folder / "IMG.jpg"
        path.touch()
        photos.append(Photo(path, sub))
    (tmp_path / "fotosort_report.csv").write_text("file,selected\n/old/IMG.jpg,1\n")
    args = parse_args([str(tmp_path), "--apply-report", "--copy"])
    with pytest.raises(SystemExit, match="Ambiguous"):
        apply_report(photos, tmp_path, args)
    assert not (tmp_path / "Highlights").exists()


def test_report_records_subject_focus_and_decision(tmp_path):
    photo = Photo(tmp_path / "IMG.jpg", "k", subject_sharpness=123.456, decision="Group budget filled")
    report = tmp_path / "report.csv"
    write_report([photo], report, tmp_path)
    with report.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["relative_file"] == "IMG.jpg"
    assert row["subject_sharpness"] == "123.46"
    assert row["decision"] == "Group budget filled"


@pytest.mark.parametrize("mode,extra_images", [("single", 1), ("multi", 4), ("none", 0)])
def test_judge_payload_respects_crop_cost_setting(tmp_path, monkeypatch, mode, extra_images):
    path = tmp_path / "IMG.jpg"
    path.touch()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    sizes = []

    def overview(path, max_side):
        sizes.append(max_side)
        return "full-image"

    monkeypatch.setattr("fotosort.judge._jpeg_b64", overview)
    monkeypatch.setattr("fotosort.judge.subject_crop", lambda *args: "native-detail")
    monkeypatch.setattr("fotosort.judge.subject_crops", lambda *args: [(f"detail{i}", "image") for i in range(4)])
    judge = Judge(crops=mode)
    seen = []

    def ask(images, prompt):
        seen.append(images)
        return '{"picks":[{"photo":1,"reason":"good"}]}'

    judge._ask_openai = ask
    boxes = {str(path): (0.1, 0.1, 0.9, 0.9)}
    assert judge.judge([path], "bird", "today", 1, boxes).picks == [str(path)]
    assert len(seen[0][0][1]) == extra_images
    assert sizes == [1024]
    # Final comparisons reuse the same bounded payload, not extra crop windows.
    judge.judge([path], "bird", "today", 1, boxes, final=True)
    assert len(seen[1][0][1]) == extra_images and sizes == [1024]
