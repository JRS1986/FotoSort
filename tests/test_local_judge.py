"""The local-judge path against a tiny OpenAI-compatible mock server: verifies
the request shape (chat/completions, model, images) and the fallback when the
server rejects response_format."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

from fotosort.judge import Judge


class MockServer(BaseHTTPRequestHandler):
    seen: list[dict] = []
    reject_json_mode = True

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockServer.seen.append({"path": self.path, "body": body})
        if MockServer.reject_json_mode and "response_format" in body:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "response_format is not supported"}}')
            return
        reply = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant",
                                          "content": '{"picks": [{"photo": 2, "reason": "sharper"}]}'}}],
                 "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_local_endpoint_request_shape_and_json_mode_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    server = HTTPServer(("127.0.0.1", 0), MockServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        for i in (1, 2):
            Image.new("RGB", (64, 64), (i * 40, 20, 30)).save(tmp_path / f"{i}.jpg")
        j = Judge("openai", model="qwen3-vl:8b", base_url=f"http://127.0.0.1:{server.server_port}/v1")
        j.client = j.client.with_options(max_retries=0)
        v = j.judge([tmp_path / "1.jpg", tmp_path / "2.jpg"], "whale", "2026-09-03", 1)
        assert v.error is None and v.picks == [str(tmp_path / "2.jpg")] and v.reasons[v.picks[0]] == "sharper"
        assert [s["path"] for s in MockServer.seen] == ["/v1/chat/completions", "/v1/chat/completions"]
        first, second = (s["body"] for s in MockServer.seen)
        assert "response_format" in first and "response_format" not in second  # fell back after the 400
        assert second["model"] == "qwen3-vl:8b" and "max_tokens" in second
        images = [c for c in second["messages"][1]["content"] if c["type"] == "image_url"]
        assert len(images) == 2 and images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert j.usage["input"] == 10
    finally:
        server.shutdown()
