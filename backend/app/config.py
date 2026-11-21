from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env from repo root or backend directory if present (does not override real env vars)
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
for _env_path in (_REPO_ROOT / ".env", _BACKEND_DIR / ".env"):
    if _env_path.exists():
        load_dotenv(_env_path, override=False)


@dataclass(frozen=True)
class KeyEntry:
    """Single API key configuration inside the ordered key pool."""

    key_id: str
    api_key: str
    model: str


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "on"}


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    """Application settings loaded strictly from environment variables."""

    llm_api_keys: List[str] = field(default_factory=list)
    llm_model: str = "gemini/gemini-3.8-flash"
    llm_models: List[str] = field(default_factory=list)
    database_url: str = "sqlite:///./data/app.db"
    allowed_origins: List[str] = field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    enable_failure_simulation: bool = False
    cache_enabled: bool = True
    cache_type: str = "local"  # "local" (in-memory) or "disk"
    disk_cache_dir: str = "/data/litellm_cache"
    default_system_prompt: str = "You are a helpful, concise AI assistant."
    llm_timeout_seconds: float = 30.0

    @property
    def key_pool(self) -> list[KeyEntry]:
        """Return ordered key entries with deterministic ids: key_1, key_2, ..."""
        entries: list[KeyEntry] = []
        for idx, api_key in enumerate(self.llm_api_keys, start=1):
            model = (
                self.llm_models[idx - 1]
                if idx - 1 < len(self.llm_models) and self.llm_models[idx - 1]
                else self.llm_model
            )
            entries.append(KeyEntry(key_id=f"key_{idx}", api_key=api_key, model=model))
        return entries

    @classmethod
    def from_env(cls) -> "Settings":
        keys = _parse_csv(os.getenv("LLM_API_KEYS"))
        default_model = os.getenv("LLM_MODEL", "gemini/gemini-3.8-flash").strip() or "gemini/gemini-3.8-flash"
        per_key_models = _parse_csv(os.getenv("LLM_MODELS"))
        database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip() or "sqlite:///./data/app.db"
        origins = _parse_csv(os.getenv("ALLOWED_ORIGINS")) or [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
        enable_sim = _parse_bool(os.getenv("ENABLE_FAILURE_SIMULATION"), default=False)
        cache_enabled = _parse_bool(os.getenv("CACHE_ENABLED"), default=True)
        cache_type = (os.getenv("CACHE_TYPE", "local").strip().lower() or "local")
        if cache_type not in {"local", "disk"}:
            cache_type = "local"
        disk_cache_dir = os.getenv("DISK_CACHE_DIR", "/data/litellm_cache").strip() or "/data/litellm_cache"
        default_prompt = (
            os.getenv("DEFAULT_SYSTEM_PROMPT", "You are a helpful, concise AI assistant.").strip()
            or "You are a helpful, concise AI assistant."
        )
        try:
            timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError:
            timeout = 30.0

        return cls(
            llm_api_keys=keys,
            llm_model=default_model,
            llm_models=per_key_models,
            database_url=database_url,
            allowed_origins=origins,
            enable_failure_simulation=enable_sim,
            cache_enabled=cache_enabled,
            cache_type=cache_type,
            disk_cache_dir=disk_cache_dir,
            default_system_prompt=default_prompt,
            llm_timeout_seconds=timeout,
        )
