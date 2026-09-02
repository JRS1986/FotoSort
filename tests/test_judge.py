from fotosort.judge import parse_verdict


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
