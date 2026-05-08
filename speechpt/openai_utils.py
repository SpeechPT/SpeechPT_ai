"""OpenAI API key resolution helpers."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]+")


def _extract_key_from_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in ("OPENAI_API_KEY", "openai_api_key", "api_key"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().strip('"').strip("'")

    match = _OPENAI_KEY_RE.search(stripped)
    if match:
        return match.group(0)

    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            if name.strip() in {"OPENAI_API_KEY", "openai_api_key", "api_key"}:
                return value.strip().strip('"').strip("'")
        if line.startswith("sk-"):
            return line
    return stripped.strip('"').strip("'")


def resolve_openai_api_key(config: Dict[str, Any] | None = None, default_key_file: str | Path = "key") -> str:
    """Resolve an OpenAI API key without logging or exposing the secret."""
    cfg = config or {}
    explicit = str(cfg.get("api_key") or "").strip()
    if explicit:
        return explicit

    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    key_file = Path(str(cfg.get("api_key_file") or default_key_file))
    if key_file.exists() and key_file.is_file():
        return _extract_key_from_text(key_file.read_text(encoding="utf-8", errors="ignore"))
    return ""
