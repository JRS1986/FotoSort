"""A loopback-only visual reviewer for an existing report. No model calls."""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import secrets
import threading
import webbrowser
from collections import OrderedDict
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fotosort.review_data import Collection, Conflict, ReviewError, fingerprint, inside, stamp

JPEG_SUFFIXES = {".jpg", ".jpeg"}


class ReviewServer(ThreadingHTTPServer):
    # Connections are read on their own threads so an idle or slow client cannot stall the reviewer;
    # `lock` still handles one request at a time, which bounds concurrent full-resolution decoding.
    daemon_threads = True

    def __init__(self, collection: Collection, port: int = 0):
        self.collection = collection
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.images: OrderedDict[tuple, bytes] = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = 32 * 1024 * 1024
        super().__init__(("127.0.0.1", port), ReviewHandler)

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def url(self):
        return f"{self.origin}/?token={self.token}"

    def snapshot(self):
        try:
            self.collection.ensure_current()
        except Conflict:
            relative = self.collection.report.relative_to(self.collection.root).as_posix()
            self.collection = Collection(self.collection.root, relative, verify_sources=False)
            self.images.clear()
            self.cache_bytes = 0
        for entry in self.collection.entries.values():
            # Photos that were hashed are re-read only after their size or modification time moved.
            # Photos not hashed yet are hashed by `verify` when they are first shown or decided.
            try:
                current = stamp(inside(self.collection.root, entry["relative_file"]))
                if entry["stamp"] is None:
                    if entry["status"] == "missing":
                        entry["status"] = "available"
                elif entry["stamp"] != current:
                    digest = fingerprint(entry["path"])
                    expected = entry["row"].get("photo_sha256") or entry["sha256"]
                    entry["status"] = "available" if digest == expected == entry["sha256"] else "changed"
                    entry["stamp"] = current
            except (OSError, ReviewError):
                entry["status"] = "missing"
                entry["stamp"] = None  # a returning file must be verified again, even with the old mtime
        state = self.collection.store.load()
        return dict(collection=self.collection.root.name, revision=state["revision"],
                    report_version=self.collection.report_digest, photos=self.collection.view(state),
                    can_undo=bool(state["history"]), preference_count=len(state["preferences"]))

    def image(self, identifier: str, kind: str) -> bytes:
        from PIL import ImageOps

        from fotosort.quality import load_small, open_full

        if kind not in {"thumb", "preview", "full"}:
            raise ReviewError("Unknown image size")
        self.collection.ensure_current()
        entry = self.collection.verify(identifier)
        source = entry["source"].resolve()
        if not source.is_relative_to(self.collection.root):
            raise ReviewError("Image source is outside the collection")
        if kind == "full" and source.suffix.lower() in JPEG_SUFFIXES:
            # The file itself is the most faithful pixel view; browsers apply its EXIF orientation.
            return source.read_bytes()
        key = (identifier, kind, str(source), *stamp(source))
        if key in self.images:
            self.images.move_to_end(key)
            return self.images[key]
        if kind == "full":
            image = open_full(str(source))
            image = ImageOps.exif_transpose(image).convert("RGB")
        else:
            image = load_small(str(source), max_side=320 if kind == "thumb" else 1400)
        with image:
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=90 if kind == "full" else 82)
        encoded = output.getvalue()
        if len(encoded) <= self.cache_limit:
            while self.images and self.cache_bytes + len(encoded) > self.cache_limit:
                self.cache_bytes -= len(self.images.popitem(last=False)[1])
            self.images[key] = encoded
            self.cache_bytes += len(encoded)
        return encoded


def client_message(exc: Exception) -> str:
    """Error text for the browser. Operating-system errors name absolute paths, so only their reason is sent."""
    if isinstance(exc, OSError):
        return exc.strerror or "File could not be read"
    if isinstance(exc, ReviewError):
        return str(exc)
    return "Invalid request"


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer
    timeout = 30  # seconds a connection may stay silent before its thread lets go of it

    def log_message(self, *args):
        pass  # URLs contain the session token; filenames and tokens stay out of logs.

    def reply(self, status: int, data, content_type="application/json"):
        if content_type == "application/json":
            data = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "img-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self, query):
        expected_host = self.server.origin.removeprefix("http://")
        token = self.headers.get("X-FotoSort-Token", "") or query.get("token", [""])[0]
        origin = self.headers.get("Origin")
        return (self.headers.get("Host") == expected_host and (not origin or origin == self.server.origin)
                and token.isascii() and secrets.compare_digest(token, self.server.token))

    def respond(self, work):
        """Run one request under the server lock, then answer outside it so a slow reader holds nothing."""
        try:
            with self.server.lock:
                answer = work()
        except Conflict as exc:
            answer = 409, {"error": str(exc)}
        except (OSError, ReviewError, ValueError, KeyError, TypeError, ImportError, RecursionError) as exc:
            answer = 400, {"error": client_message(exc)}
        except Exception:  # decoder errors (LibRaw, decompression bombs) must still answer the browser
            answer = 500, {"error": "This photo could not be decoded"}
        self.reply(*answer)

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if not self.authorized(query):
            return self.reply(403, {"error": "Open the session URL printed by fotosort review"})
        self.respond(lambda: self.get(parsed.path, query))

    def get(self, path: str, query: dict) -> tuple:
        if path == "/":
            page = files("fotosort").joinpath("static/review.html").read_text(encoding="utf-8")
            page = page.replace("__TOKEN__", html.escape(self.server.token))
            return 200, page.encode(), "text/html; charset=utf-8"
        if path in {"/assets/review.css", "/assets/review.js"}:
            name = path.rsplit("/", 1)[1]
            data = files("fotosort").joinpath(f"static/{name}").read_bytes()
            mime = "text/css" if name.endswith("css") else "text/javascript"
            return 200, data, mime + "; charset=utf-8"
        if path == "/api/state":
            return 200, self.server.snapshot()
        if path.startswith("/image/"):
            data = self.server.image(path.removeprefix("/image/"), query.get("size", ["thumb"])[0])
            return 200, data, "image/jpeg"
        return 404, {"error": "Not found"}

    def do_POST(self):
        if not self.authorized(parse_qs(urlsplit(self.path).query)):
            return self.reply(403, {"error": "Invalid session or origin"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 65536 or self.headers.get_content_type() != "application/json":
                raise ReviewError("Expected a JSON request of at most 64 KiB")
            body = json.loads(self.rfile.read(length))  # read before taking the lock: the client may be slow
            if not isinstance(body, dict):
                raise ReviewError("Expected a JSON object")
        except (OSError, ValueError, RecursionError) as exc:
            return self.reply(400, {"error": client_message(exc)})
        self.respond(lambda: self.post(urlsplit(self.path).path, body))

    def post(self, route: str, body: dict) -> tuple:
        collection = self.server.collection
        collection.ensure_current()
        if body["report_version"] != collection.report_digest:
            raise Conflict("This page uses an older report; reload before editing")
        revision = body["revision"]
        if route == "/api/change":
            collection.change(revision, body.get("changes", {}), body.get("note", ""),
                              preference=body.get("preference"), alternatives=body.get("alternatives"))
        elif route == "/api/undo":
            collection.undo(revision)
        elif route == "/api/export":
            output = collection.export(body.get("output") or default_export(collection), revision)
            return 200, {"output": output.relative_to(collection.root).as_posix()}
        else:
            return 404, {"error": "Not found"}
        return 200, self.server.snapshot()


def default_export(collection: Collection) -> str:
    """The reviewed CSV is often reviewed again; its next export gets a new name instead of replacing it."""
    name, number = "fotosort_reviewed.csv", 2
    while (collection.root / name).exists() and (collection.root / name).samefile(collection.report):
        name, number = f"fotosort_reviewed_{number}.csv", number + 1
    return name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="fotosort review", description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--report", default="fotosort_report.csv")
    parser.add_argument("--port", type=int, default=0, help="Loopback port (default: choose an available port)")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL without opening a browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        # Photos are hashed when they are first shown or decided, not all at once before the page opens.
        collection = Collection(args.folder, args.report, verify_sources=False)
        with ReviewServer(collection, args.port) as server:
            print(f"Local reviewer: {server.url}\nPress Ctrl+C to stop. Photos stay on this computer.", flush=True)
            if not args.no_browser:
                webbrowser.open(server.url)
            with suppress(KeyboardInterrupt):
                server.serve_forever()
        return 0
    except (OSError, ValueError, csv.Error) as exc:  # ValueError covers ReviewError and undecodable CSVs
        parser.exit(2, f"Review error: {exc}\n")
