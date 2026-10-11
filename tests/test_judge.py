import json
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import openai
import pytest
from PIL import Image

from fotosort.cli import parse_args
from fotosort.judge import Judge, parse_verdict


@pytest.mark.parametrize("model_flags,expected_model", [
    ([], "gpt-6.1-sol"),
    (["--judge-model", "gpt-5.6-sol"], "gpt-5.6-sol"),
    (["--judge-model", "custom-vision-model"], "custom-vision-model"),
])
def test_cli_openai_model_reaches_pick_and_subject_requests(monkeypatch, tmp_path, model_flags, expected_model):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path)
    requests = []

    def respond(body):
        requests.append(body)
        result = ({"picks": [{"photo": 1, "reason": "sharp"}]} if len(requests) == 1 else
                  {"subjects": [{"photo": 1, "subject": "elephant"}]})
        return {
            "id": "test", "object": "chat.completion", "created": 0, "model": expected_model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": json.dumps(result)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(respond(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            # Redirect the SDK, not Judge.base_url: exercise the cloud payload.
            monkeypatch.setattr(openai, "OpenAI", partial(
                openai.OpenAI, base_url=f"http://127.0.0.1:{server.server_port}/v1", max_retries=0))
            args = parse_args([str(tmp_path), "--judge", *model_flags])
            judge = Judge(args.judge_provider, args.judge_model, detail=args.judge_detail)
            with judge.client:
                verdict = judge.judge([path], "wildlife", "2026-10-11", 1)
                assert verdict.error is None
                assert verdict.picks == [str(path)] and verdict.reasons[str(path)] == "sharp"
                assert judge.name_subjects([path], ["elephant"]) == {str(path): "elephant"}
                assert judge.usage == {"input": 20, "output": 10}
        finally:
            server.shutdown()
            thread.join()

    assert len(requests) == 2
    for body in requests:
        assert body["model"] == expected_model
        assert body["response_format"] == {"type": "json_object"}
        assert body["max_completion_tokens"] == 4000 and "max_tokens" not in body
        images = [part["image_url"] for part in body["messages"][1]["content"] if part["type"] == "image_url"]
        assert len(images) == 1 and images[0]["detail"] == "high"
        assert images[0]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("model_flags,expected_model", [
    ([], "claude-opus-5"),
    (["--judge-model", "custom-anthropic-model"], "custom-anthropic-model"),
])
def test_cli_anthropic_model_routing_is_unchanged(monkeypatch, model_flags, expected_model):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                               stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"picks": []}')])

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **kwargs: client))
    args = parse_args([".", "--judge", "--judge-provider", "anthropic", *model_flags])
    judge = Judge(args.judge_provider, args.judge_model)
    verdict = judge._judge([("image", [])], ["photo"], "Choose a photo; reply with JSON.")
    assert verdict.error is None and verdict.picks == []
    assert len(requests) == 1 and requests[0]["model"] == expected_model
    assert judge.usage == {"input": 10, "output": 5}


def test_parse_verdict_maps_1_based_photos_to_ids():
    ids = ["a", "b", "c"]
    v = parse_verdict('Sure. {"picks": [{"photo": 3, "reason": "sharp"}, {"photo": 1, "reason": "moment"}]}', ids)
    assert v.picks == ["c", "a"] and v.reasons["c"] == "sharp" and v.error is None


def test_parse_verdict_ignores_out_of_range_and_duplicates():
    v = parse_verdict('{"picks": [{"photo": 9}, {"photo": 2}, {"photo": 2}]}', ["a", "b"])
    assert v.picks == ["b"]


def test_parse_verdict_reports_garbage():
    assert parse_verdict("I cannot decide", ["a"]).error


def test_read_key_from_env_and_yaml_files(tmp_path, monkeypatch):
    from fotosort.judge import read_key
    (tmp_path / "s.yaml").write_text("other: 1\nopenai_api_key: sk-test-123\n")
    (tmp_path / ".env").write_text("export OPENAI_API_KEY=\"sk-env-456\"\n")
    assert read_key("openai", str(tmp_path / "s.yaml")) == "sk-test-123"
    assert read_key("openai", str(tmp_path / ".env")) == "sk-env-456"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    assert read_key("openai", None) == "sk-from-env"


def test_hint_is_prepended_to_the_prompt(monkeypatch, tmp_path):
    from PIL import Image

    from fotosort.judge import Judge
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "a.jpg")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    j = Judge("openai", hint="Whale trip: blows are not keepers.")
    seen = {}

    def fake(images, prompt):
        seen["prompt"] = prompt
        return '{"picks": [{"photo": 1, "reason": "ok"}]}'
    j._ask_openai = fake
    v = j.judge([tmp_path / "a.jpg"], "whale", "2026-09-03", 1)
    assert seen["prompt"].startswith("About this shoot: Whale trip") and v.picks


def test_judge_skips_files_deleted_meanwhile(monkeypatch, tmp_path):
    from PIL import Image

    from fotosort.judge import Judge
    Image.new("RGB", (64, 64), (10, 20, 30)).save(tmp_path / "a.jpg")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    j = Judge("openai")
    j._ask_openai = lambda images, prompt: '{"picks": [{"photo": 1, "reason": "ok"}]}'
    v = j.judge([tmp_path / "gone.jpg", tmp_path / "a.jpg"], "x", "d", 1)
    assert v.picks == [str(tmp_path / "a.jpg")] and v.error is None
    assert j.judge([tmp_path / "gone.jpg"], "x", "d", 1).error


def test_judge_needs_a_key(monkeypatch):
    import pytest

    from fotosort.judge import Judge
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        Judge("openai")


def test_local_judge_needs_no_key_but_needs_a_model(monkeypatch):
    import pytest

    from fotosort.judge import Judge
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        Judge("openai", base_url="http://localhost:11434/v1")
    j = Judge("openai", model="qwen3-vl:8b", base_url="http://localhost:11434/v1")
    assert j.base_url and str(j.client.base_url).startswith("http://localhost:11434")
