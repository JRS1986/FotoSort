
from fotosort.dxo import find_twin, wait_for_twins


def test_find_twin_matches_pureraw_naming_templates(tmp_path):
    raw = tmp_path / "_8311254.ORF"
    raw.write_bytes(b"raw")
    assert find_twin(raw) is None                       # no DxO folder yet
    dxo = tmp_path / "DxO"
    dxo.mkdir()
    (dxo / "20260831-_8311254.dng").write_bytes(b"x")   # date-prefixed template (PureRAW 6)
    (dxo / "20260831-_8311254.jpg").write_bytes(b"x")   # a JPEG export too: DNG wins
    (dxo / "20260831-_8311255.dng").write_bytes(b"x")   # another frame
    assert find_twin(raw).name == "20260831-_8311254.dng"
    other = tmp_path / "P9031048.ORF"
    (dxo / "P9031048-DxO_DeepPRIME.dng").write_bytes(b"x")  # suffix template (PureRAW 3/4)
    assert find_twin(other).name == "P9031048-DxO_DeepPRIME.dng"
    assert find_twin(tmp_path / "missing.ORF") is None
    assert find_twin(raw, dxo_dir=tmp_path / "elsewhere") is None


def test_wait_for_twins_returns_what_exists_and_gives_up_on_timeout(tmp_path):
    a, b = tmp_path / "a.ORF", tmp_path / "b.ORF"
    a.write_bytes(b"1")
    b.write_bytes(b"1")
    (tmp_path / "DxO").mkdir()
    (tmp_path / "DxO" / "a.dng").write_bytes(b"processed")
    done = wait_for_twins([a, b], None, timeout_s=0.5, poll_s=0.1, log=lambda *_: None)
    assert list(done) == [a] and done[a].name == "a.dng"
