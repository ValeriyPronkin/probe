"""One provider client, used by every narrow component.

Replay is not a testing convenience bolted on afterwards — it is how the
instrument stays reproducible. A cassette is keyed by everything that can change
an answer (model, prompt version, exact messages), so a cassette hit proves the
request was identical, and a miss is a loud error rather than a silent live call.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from probe.config import CASSETTE_DIR, Config
from probe.models import (
    APP_HEADERS,
    DEEPSEEK_BASE_URL,
    OPENROUTER_BASE_URL,
    PROMPT_VERSION,
    ModelRole,
)


EMPTY_RETRIES = 2


class CassetteMissing(RuntimeError):
    """Replay mode was asked for a response that was never recorded."""


@dataclass(frozen=True)
class Request:
    role: ModelRole
    system: str
    user: str
    schema_name: str
    schema: dict[str, Any]

    def key(self) -> str:
        blob = json.dumps(
            {
                "model": self.role.model,
                "provider": self.role.provider,
                "temperature": self.role.temperature,
                "prompt_version": PROMPT_VERSION,
                "system": self.system,
                "user": self.user,
                "schema": self.schema,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:24]


def cassette_path(request: Request, directory: Path | None = None) -> Path:
    return (directory or CASSETTE_DIR) / f"{request.role.name}-{request.key()}.json"


def complete_json(
    request: Request, config: Config, cassette_dir: Path | None = None
) -> dict[str, Any]:
    path = cassette_path(request, cassette_dir or config.cassette_dir)

    if config.mode == "replay" or (config.reuse_recorded and path.exists()):
        if not path.exists():
            raise CassetteMissing(
                f"no cassette for {request.role.name} at {path.name}.\n"
                f"Record it with PROBE_MODE=live PROBE_RECORD=1, or check whether the "
                f"prompt changed — a changed prompt is a different key by design."
            )
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    response, served_model, system_sent = _call_provider(request, config)
    if config.record:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "role": request.role.name,
                    "model": request.role.model,
                    "provider": request.role.provider,
                    "served_model": served_model,
                    "prompt_version": PROMPT_VERSION,
                    "system": request.system,
                    "system_sent": system_sent,
                    "user": request.user,
                    "response": response,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return response


def schema_in_system(request: Request) -> str:
    """DeepSeek offers JSON mode but not strict json_schema, and GigaChat's
    json_schema is a beta that answered the observer schema with a 500, so the
    contract goes into the instruction instead. Nothing is lost for the instrument: the answer
    is validated after the fact either way, which is what counts invented ids."""
    schema = json.dumps(request.schema, ensure_ascii=False, indent=2)
    return (
        f"{request.system}\n\n"
        f"Ответь одним JSON-объектом, строго по этой JSON-схеме, без текста вокруг:\n{schema}"
    )


def _call_provider(request: Request, config: Config) -> tuple[dict[str, Any], str, str]:
    """Returns the parsed answer, the model that actually served it, and the
    system text actually sent — both differ from the request on some providers."""
    provider = request.role.provider
    if provider == "gigachat":
        return _call_gigachat(request, config)

    from openai import OpenAI

    if provider == "deepseek":
        client = OpenAI(base_url=DEEPSEEK_BASE_URL, api_key=config.require_key(provider))
        system = schema_in_system(request)
        response_format: dict[str, Any] = {"type": "json_object"}
    else:
        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=config.require_key(provider),
            default_headers=APP_HEADERS,
        )
        system = request.system
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "strict": True,
                "schema": request.schema,
            },
        }
    # DeepSeek's JSON mode occasionally returns empty content. The request is
    # unchanged on a retry, so retrying cannot alter what is measured; a
    # non-empty answer that is not JSON is still an error, below.
    for _ in range(EMPTY_RETRIES + 1):
        completion = client.chat.completions.create(
            model=request.role.model,
            temperature=request.role.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": request.user},
            ],
            response_format=response_format,
        )
        content = completion.choices[0].message.content or ""
        if content.strip():
            break
    try:
        return json.loads(content), completion.model, system
    except json.JSONDecodeError as exc:
        # Structured output was requested; malformed JSON is a provider problem
        # worth surfacing, not something to repair by guessing.
        raise RuntimeError(f"{request.role.name} returned non-JSON: {content[:200]}") from exc


GIGACHAT_TIMEOUT = 300
"""Seconds. A single observer turn took about two minutes on 2026-10-19."""

GIGACHAT_SERVER_RETRIES = (15, 60, 180)
"""Pauses in seconds before each retry after a 5xx or a network error. On 2026-10-19
the API returned 500 on a fraction of identical requests, let one hang past the
timeout and dropped one TLS connection. The request is unchanged on a retry, so
retrying cannot alter what is measured."""


def _call_gigachat(request: Request, config: Config) -> tuple[dict[str, Any], str, str]:
    """Through the official SDK, which refreshes the 30-minute token itself."""
    import time

    import httpx
    from gigachat import GigaChat
    from gigachat.exceptions import ServerError
    from gigachat.models import Chat, Messages, MessagesRole

    system = schema_in_system(request)
    chat = Chat(
        model=request.role.model,
        temperature=request.role.temperature,
        messages=[
            Messages(role=MessagesRole.SYSTEM, content=system),
            Messages(role=MessagesRole.USER, content=request.user),
        ],
    )
    with GigaChat(
        credentials=config.require_key("gigachat"),
        scope="GIGACHAT_API_PERS",
        ca_bundle_file=config.gigachat_ca_bundle,
        timeout=GIGACHAT_TIMEOUT,
    ) as client:
        for _ in range(EMPTY_RETRIES + 1):
            for pause in (*GIGACHAT_SERVER_RETRIES, None):
                try:
                    completion = client.chat(chat)
                    break
                except (ServerError, httpx.TransportError):
                    if pause is None:
                        raise
                    time.sleep(pause)
            content = completion.choices[0].message.content or ""
            if content.strip():
                break
    try:
        return json.loads(content), completion.model, system
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{request.role.name} returned non-JSON: {content[:200]}") from exc
