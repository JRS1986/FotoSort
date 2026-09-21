"""A loopback-only visual reviewer for an existing report. No model calls."""
from __future__ import annotations

import argparse
import html
import io
import json
import secrets
import webbrowser
from collections import OrderedDict
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fotosort.review_data import Collection, Conflict, ReviewError, fingerprint, inside


class ReviewServer(HTTPServer):
    # One request at a time bounds concurrent full-resolution decoding.
    def __init__(self, collection: Collection, port: int = 0):
        self.collection = collection
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
            self.collection = Collection(self.collection.root, relative)
            self.images.clear()
            self.cache_bytes = 0
        for entry in self.collection.entries.values():
            try:
                path = inside(self.collection.root, entry["relative_file"])
                stat = path.stat()
                stamp = (stat.st_size, stat.st_mtime_ns)
                if entry.get("_stat") != stamp:
                    digest = fingerprint(path)
                    expected = entry["row"].get("photo_sha256") or entry["sha256"]
                    entry["status"] = "available" if digest == expected == entry["sha256"] else "changed"
                    entry["_stat"] = stamp
            except (OSError, ReviewError):
                entry["status"] = "missing"
                entry["_stat"] = None
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
        stat = source.stat()
        key = (identifier, kind, str(source), stat.st_size, stat.st_mtime_ns)
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


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

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

    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if not self.authorized(query):
            return self.reply(403, {"error": "Open the session URL printed by fotosort review"})
        try:
            if parsed.path == "/":
                page = files("fotosort").joinpath("static/review.html").read_text()
                page = page.replace("__TOKEN__", html.escape(self.server.token))
                return self.reply(200, page.encode(), "text/html; charset=utf-8")
            if parsed.path in {"/assets/review.css", "/assets/review.js"}:
                name = parsed.path.rsplit("/", 1)[1]
                data = files("fotosort").joinpath(f"static/{name}").read_bytes()
                mime = "text/css" if name.endswith("css") else "text/javascript"
                return self.reply(200, data, mime + "; charset=utf-8")
            if parsed.path == "/api/state":
                return self.reply(200, self.server.snapshot())
            if parsed.path.startswith("/image/"):
                data = self.server.image(parsed.path.removeprefix("/image/"), query.get("size", ["thumb"])[0])
                return self.reply(200, data, "image/jpeg")
            self.reply(404, {"error": "Not found"})
        except Conflict as exc:
            self.reply(409, {"error": str(exc)})
        except (OSError, ReviewError, ValueError, ImportError) as exc:
            self.reply(400, {"error": str(exc)})

    def do_POST(self):
        if not self.authorized(parse_qs(urlsplit(self.path).query)):
            return self.reply(403, {"error": "Invalid session or origin"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 65536 or self.headers.get_content_type() != "application/json":
                raise ReviewError("Expected a JSON request of at most 64 KiB")
            body = json.loads(self.rfile.read(length))
            collection = self.server.collection
            collection.ensure_current()
            if body["report_version"] != collection.report_digest:
                raise Conflict("This page uses an older report; reload before editing")
            revision = body["revision"]
            route = urlsplit(self.path).path
            if route == "/api/change":
                collection.change(revision, body.get("changes", {}), body.get("note", ""),
                                  preference=body.get("preference"), alternatives=body.get("alternatives"))
            elif route == "/api/undo":
                collection.undo(revision)
            elif route == "/api/export":
                output = collection.export(body.get("output", "fotosort_reviewed.csv"), revision)
                return self.reply(200, {"output": output.relative_to(collection.root).as_posix()})
            else:
                return self.reply(404, {"error": "Not found"})
            self.reply(200, self.server.snapshot())
        except Conflict as exc:
            self.reply(409, {"error": str(exc)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.reply(400, {"error": str(exc)})


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
        with ReviewServer(Collection(args.folder, args.report), args.port) as server:
            print(f"Local reviewer: {server.url}\nPress Ctrl+C to stop. Photos stay on this computer.", flush=True)
            if not args.no_browser:
                webbrowser.open(server.url)
            with suppress(KeyboardInterrupt):
                server.serve_forever()
        return 0
    except (OSError, ReviewError) as exc:
        parser.exit(2, f"Review error: {exc}\n")
