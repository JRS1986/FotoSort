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
