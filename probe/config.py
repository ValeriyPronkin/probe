"""The only module that reads the environment.

Everything else takes a `Config`. One place to look for what the process needs,
one place that fails loudly when it is missing, and no module that behaves
differently depending on where it was imported from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Mode = Literal["live", "replay"]

ROOT = Path(__file__).resolve().parent.parent
CASSETTE_DIR = ROOT / "fixtures" / "cassettes"
LIVE_DIR = ROOT / "fixtures" / "transcripts"
"""Live sessions: a person's own words, so git ignores the whole folder."""
LIVE_CASSETTE_DIR = LIVE_DIR / "cassettes"
"""A live session's model responses, kept beside its transcript rather than with
the personas' cassettes: they are recorded from the same person's words."""


@dataclass(frozen=True)
class Config:
    mode: Mode
    db_path: str
    record: bool
    """In live mode, write every provider response to a cassette for later replay."""
    openrouter_api_key: str = ""
    deepseek_api_key: str = ""
    cassette_dir: Path = CASSETTE_DIR

    def require_key(self, provider: str) -> str:
        key, variable = {
            "openrouter": (self.openrouter_api_key, "OPENROUTER_API_KEY"),
            "deepseek": (self.deepseek_api_key, "DEEPSEEK_API_KEY"),
        }[provider]
        if not key:
            raise RuntimeError(
                f"PROBE_MODE=live needs {variable} in .env — "
                "or run with PROBE_MODE=replay to use recorded cassettes"
            )
        return key


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader. Values already in the environment win, so a shell
    override does not get silently replaced by the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load(env_file: Path | None = None) -> Config:
    _load_dotenv(env_file or ROOT / ".env")
    mode = os.environ.get("PROBE_MODE", "replay")
    if mode not in ("live", "replay"):
        raise ValueError(f"PROBE_MODE must be 'live' or 'replay', got {mode!r}")
    return Config(
        mode=mode,
        db_path=os.environ.get("PROBE_DB", "probe.db"),
        record=os.environ.get("PROBE_RECORD", "") == "1",
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    )


def replay_only(cassette_dir: Path = CASSETTE_DIR) -> Config:
    """What tests and the demo use. A test that reaches a provider is not a test."""
    return Config(mode="replay", db_path=":memory:", record=False, cassette_dir=cassette_dir)
