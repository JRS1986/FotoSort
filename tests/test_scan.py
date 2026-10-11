from pathlib import Path

import pytest

from fotosort import cli
from fotosort.scan import find_images, is_raw


def test_raw_with_jpeg_twin_is_skipped_but_lone_raw_is_kept(tmp_path):
    for name in ["a.JPG", "a.ORF", "b.orf", "c.jpg", "d.CR3", ".hidden.jpg", "notes.txt"]:
        (tmp_path / name).write_bytes(b"x")
    names = [p.name for p in find_images(tmp_path, recursive=False, exclude_dirs=set())]
    assert names == ["a.JPG", "b.orf", "c.jpg", "d.CR3"]
    jpeg_only = find_images(tmp_path, recursive=False, exclude_dirs=set(), include_raw=False)
    assert [p.name for p in jpeg_only] == ["a.JPG", "c.jpg"]


def test_recursive_scan_excludes_output_and_hidden_dirs(tmp_path):
    for d in ["sub", "Highlights", ".cache"]:
        (tmp_path / d).mkdir()
    for d, n in [("sub", "x.jpg"), ("Highlights", "y.jpg"), (".cache", "z.jpg")]:
        (tmp_path / d / n).write_bytes(b"x")
    found = find_images(tmp_path, recursive=True, exclude_dirs={"Highlights"})
    assert [p.name for p in found] == ["x.jpg"]
    assert is_raw(Path("q.NEF")) and not is_raw(Path("q.jpg"))


def test_exclude_dir_skips_named_folders_at_any_depth(tmp_path, monkeypatch):
    # A trip folder with earlier per-chapter outputs that must not be re-analysed.
    for rel in ["day1/a.jpg", "day1/Selection/giraffe/a.jpg", "day2/b.jpg",
                "day2/Selection_Enhanced/b.jpg", "_DELETE_ME/day1/a.jpg", "day2/Selected/c.jpg"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    seen = {}

    def scan(root, recursive, exclude_dirs, include_raw=True):
        seen["exclude"] = exclude_dirs
        return find_images(root, recursive, exclude_dirs, include_raw)

    monkeypatch.setattr(cli, "find_images", scan)
    monkeypatch.setattr(cli, "Photo", lambda **kw: pytest.fail("stop after the scan"))
    with pytest.raises(pytest.fail.Exception):
        cli.main([str(tmp_path), "--recursive", "--exclude-dir", "Selection",
                  "--exclude-dir", "Selection_Enhanced", "--exclude-dir", "_DELETE_ME"])
    assert seen["exclude"] == {"Highlights", "Selection", "Selection_Enhanced", "_DELETE_ME"}
    found = find_images(tmp_path, True, seen["exclude"])
    assert [str(p.relative_to(tmp_path)) for p in found] == ["day1/a.jpg", "day2/Selected/c.jpg", "day2/b.jpg"]


@pytest.mark.parametrize("value", ["", ".", "..", "Selection/giraffe"])
def test_exclude_dir_accepts_folder_names_only(tmp_path, value):
    with pytest.raises(SystemExit):
        cli.parse_args([str(tmp_path), "--exclude-dir", value])
