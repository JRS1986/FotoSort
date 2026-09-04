
from fotosort.cli import Photo, _decode


def test_decode_reports_error_instead_of_raising(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not a jpeg")
    ph, sh, err, tensor, small = _decode(Photo(path=bad, key="k"), lambda img: img)
    assert sh is None and tensor is None and isinstance(err, Exception)
    ph, sh, err, tensor, small = _decode(Photo(path=tmp_path / "missing.jpg", key="k"), lambda img: img)
    assert sh is None and isinstance(err, FileNotFoundError)
