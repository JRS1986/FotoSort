from pathlib import Path

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
