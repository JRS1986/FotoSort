import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from fotosort.review import ReviewServer
from fotosort.review_data import Collection, write_rows


@pytest.fixture
def server(tmp_path):
    rows = []
    for i in range(3):
        name = f"{i}.jpg"
        Image.new("RGB", (1000, 700), (i * 60, 120, 80)).save(tmp_path / name)
        rows.append(dict(file=name, relative_file=name, selected=str(int(i == 0)), shortlisted=str(int(i == 0))))
    write_rows(tmp_path / "fotosort_report.csv", list(rows[0]), rows)
    app = ReviewServer(Collection(tmp_path))
    worker = threading.Thread(target=app.serve_forever, daemon=True)
    worker.start()
    try:
        yield app
    finally:
        app.shutdown()
        app.server_close()
        worker.join(timeout=5)


def request(server, route, body=None, headers=None, token=True):
    supplied = {"X-FotoSort-Token": server.token} if token else {}
    supplied.update(headers or {})
    if body is not None:
        supplied["Content-Type"] = "application/json"
    req = Request(server.origin + route, data=json.dumps(body).encode() if body is not None else None, headers=supplied)
    with urlopen(req, timeout=5) as response:
        raw = response.read()
        return (json.loads(raw) if response.headers.get_content_type() == "application/json" else raw), response.headers


def test_state_includes_excluded_frames_and_no_absolute_paths(server, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "fotosort.embed", None)
    data, _ = request(server, "/api/state")
    assert len(data["photos"]) == 3 and sum(p["shortlisted"] != "1" for p in data["photos"]) == 2
    assert str(server.collection.root) not in json.dumps(data)
    page, headers = request(server, "/")
    assert b"Choose your keeper" in page and b"__TOKEN__" not in page
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
    assert headers["Referrer-Policy"] == "no-referrer"
    assert b"reload" in request(server, "/assets/review.js")[0]


@pytest.mark.parametrize("headers,token", [({}, False), ({"Origin": "https://example.com"}, True),
                                           ({"Host": "evil.example"}, True)])
def test_unauthorized_and_cross_origin_requests_cannot_read_or_write(server, headers, token):
    for body in (None, {"revision": 0}):
        with pytest.raises(HTTPError) as exc:
            request(server, "/api/change" if body else "/api/state", body, headers, token)
        assert exc.value.code == 403
    assert server.collection.store.load()["revision"] == 0


def test_atomic_replace_undo_and_export(server):
    state, _ = request(server, "/api/state")
    first, second = (p["id"] for p in state["photos"][:2])
    body = dict(revision=0, report_version=state["report_version"],
                changes={first: "reject", second: "keep"}, preference=[second, first])
    updated, _ = request(server, "/api/change", body)
    assert [p["selected"] for p in updated["photos"]] == [False, True, False]
    assert updated["preference_count"] == 1 and updated["can_undo"]
    with pytest.raises(HTTPError) as exc:
        request(server, "/api/change", body)
    assert exc.value.code == 409
    exported, _ = request(server, "/api/export", dict(revision=1, report_version=state["report_version"]))
    assert exported["output"] == "fotosort_reviewed.csv"
    undone, _ = request(server, "/api/undo", dict(revision=1, report_version=state["report_version"]))
    assert [p["selected"] for p in undone["photos"]] == [True, False, False]
    assert undone["preference_count"] == 0


def test_report_refresh_rejects_old_browser_snapshot(server):
    old, _ = request(server, "/api/state")
    report = server.collection.report
    report.write_text(report.read_text() + "\n")
    current, _ = request(server, "/api/state")
    assert current["report_version"] != old["report_version"]
    with pytest.raises(HTTPError) as exc:
        request(server, "/api/change", dict(revision=0, report_version=old["report_version"], changes={}))
    assert exc.value.code == 409


def test_images_are_lazy_sized_and_byte_cache_bounded(server):
    assert not server.images
    identifier = next(iter(server.collection.entries))
    thumb, _ = request(server, f"/image/{identifier}?size=thumb")
    assert max(Image.open(io.BytesIO(thumb)).size) == 320
    full, _ = request(server, f"/image/{identifier}?size=full")
    assert Image.open(io.BytesIO(full)).size == (1000, 700)
    assert server.cache_bytes <= server.cache_limit
    server.images.clear()
    server.cache_bytes = 0
    server.cache_limit = 10
    request(server, f"/image/{identifier}?size=full")
    assert not server.images and server.cache_bytes == 0


def test_paths_and_replaced_symlinks_cannot_escape_collection(server, tmp_path):
    state, _ = request(server, "/api/state")
    with pytest.raises(HTTPError) as exc:
        request(server, "/api/export", dict(revision=0, report_version=state["report_version"], output="../out.csv"))
    assert exc.value.code == 400 and not (tmp_path.parent / "out.csv").exists()
    with pytest.raises(HTTPError) as exc:
        request(server, "/assets/../../etc/passwd")
    assert exc.value.code == 404
    identifier, entry = next(iter(server.collection.entries.items()))
    entry["path"].unlink()
    entry["path"].symlink_to(tmp_path.parent / "outside.jpg")
    with pytest.raises(HTTPError) as exc:
        request(server, f"/image/{identifier}")
    assert exc.value.code == 400


def test_missing_image_and_malformed_edits_are_visible_errors(server):
    identifier, entry = next(iter(server.collection.entries.items()))
    entry["path"].unlink()
    with pytest.raises(HTTPError) as exc:
        request(server, f"/image/{identifier}")
    assert exc.value.code == 409
    state, _ = request(server, "/api/state")
    with pytest.raises(HTTPError) as exc:
        request(server, "/api/change", dict(revision=0, report_version=state["report_version"], changes="invalid"))
    assert exc.value.code == 400
    assert server.collection.store.load()["revision"] == 0


def test_full_raw_uses_shared_decoder_and_orients_native_pixels(server, monkeypatch):
    identifier, entry = next(iter(server.collection.entries.items()))
    raw = server.collection.root / "original.ORF"
    raw.write_bytes(b"fixture raw")
    entry["source"] = raw
    calls = []

    def decode(path):
        calls.append(path)
        image = Image.new("RGB", (900, 600), "red")
        image.getexif()[274] = 6
        return image
    monkeypatch.setattr("fotosort.quality.open_full", decode)
    full, _ = request(server, f"/image/{identifier}?size=full")
    assert Image.open(io.BytesIO(full)).size == (600, 900)
    assert calls == [str(raw)]


def test_reload_marks_missing_manual_choice_stale_and_allows_clear(server):
    state, _ = request(server, "/api/state")
    identifier = state["photos"][0]["id"]
    server.collection.change(0, {identifier: "keep"})
    server.collection.entries[identifier]["path"].unlink()
    updated, _ = request(server, "/api/state")
    assert updated["photos"][0]["status"] == "missing"
    assert updated["photos"][0]["review_status"] == "stale decision"
    server.collection.change(1, {identifier: "clear"})
    assert not server.collection.store.load()["decisions"]
