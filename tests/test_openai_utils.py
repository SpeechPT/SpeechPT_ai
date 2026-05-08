from pathlib import Path

from speechpt.openai_utils import resolve_openai_api_key


def test_resolve_openai_api_key_from_env(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "key"
    key_file.write_text("OPENAI_API_KEY=sk-file", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    assert resolve_openai_api_key({"api_key_file": str(key_file)}) == "sk-env"


def test_resolve_openai_api_key_from_key_file_assignment(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "key"
    key_file.write_text("OPENAI_API_KEY=sk-file", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert resolve_openai_api_key({"api_key_file": str(key_file)}) == "sk-file"


def test_resolve_openai_api_key_from_json_file(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "key"
    key_file.write_text('{"api_key": "sk-json"}', encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert resolve_openai_api_key({"api_key_file": str(key_file)}) == "sk-json"


def test_resolve_openai_api_key_finds_key_embedded_in_file(monkeypatch, tmp_path: Path):
    key_file = tmp_path / "key"
    key_file.write_text("tracking_id key_xxx\nsecret: sk-proj-example_123", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert resolve_openai_api_key({"api_key_file": str(key_file)}) == "sk-proj-example_123"
