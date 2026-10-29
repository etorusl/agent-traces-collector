from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx

from .accounting import TokenAccounting
from .config import ModelConfig
from .util import WallTimeout, now, time_limit


class ModelFailure(RuntimeError):
    pass


def server_identity(models: dict[str, Any]) -> list[dict[str, Any]]:
    # ModelCard.created and permission IDs are generated anew by some vLLM versions.
    fields = ("id", "root", "parent", "max_model_len", "owned_by")
    return sorted(
        ({k: m.get(k) for k in fields} for m in models.get("data", [])), key=lambda m: m["id"]
    )


class ModelClient:
    def __init__(self, config: ModelConfig, accounting: TokenAccounting):
        self.config, self.accounting = config, accounting
        key = os.environ.get(config.api_key_env, "")
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {key}"} if key else {},
            follow_redirects=False,
        )

    def close(self) -> None:
        self.client.close()

    def models(self) -> dict[str, Any]:
        with time_limit(self.config.request_timeout_seconds):
            response = self.client.get("models", timeout=self.config.request_timeout_seconds)
            response.raise_for_status()
            return response.json()

    def verify_accounting(self) -> dict[str, Any] | None:
        cfg = self.accounting.config
        if not cfg.token_id_format:
            return None
        markers = {
            "gemma4": ("<|channel>thought\n", "<channel|>"),
            "qwen3": ("<think>", "</think>"),
        }[cfg.token_id_format]
        expected = (cfg.reasoning_start_token_ids, cfg.reasoning_end_token_ids)
        root = self.config.base_url.rstrip("/").removesuffix("/v1")
        result = {}
        for marker, ids in zip(markers, expected, strict=True):
            with time_limit(self.config.request_timeout_seconds):
                response = self.client.post(
                    root + "/tokenize",
                    json={
                        "model": self.config.name,
                        "prompt": marker,
                        "add_special_tokens": False,
                    },
                    timeout=self.config.request_timeout_seconds,
                )
                response.raise_for_status()
                actual = response.json()["tokens"]
            if actual != ids:
                raise ValueError(f"Server token marker mismatch for {marker!r}: {actual} != {ids}")
            result[marker] = actual
        return {
            "format": cfg.token_id_format,
            "markers": result,
            "reasoning_definition": "generated IDs strictly inside thinking markers",
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int,
        deadline: float,
        record: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        cfg = self.config
        payload = {
            "model": cfg.name,
            "messages": messages,
            "stream": False,
            "n": 1,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "seed": cfg.seed,
            "max_tokens": min(cfg.max_tokens, max_tokens),
            **cfg.extra_body,
        }
        if self.accounting.config.token_id_format:
            payload.update(return_token_ids=True, skip_special_tokens=False)
        for attempt in range(cfg.retries + 1):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise WallTimeout("Task deadline before model request")
            started = time.perf_counter()
            event: dict[str, Any] = {
                "request_id": uuid.uuid4().hex,
                "started_at": now(),
                "retry_index": attempt,
                "request": payload,
                "accounting": {},
                "response": None,
            }
            retryable = False
            record(
                {
                    "event_type": "llm_started",
                    "request_id": event["request_id"],
                    "request": payload,
                    "started_at": event["started_at"],
                }
            )
            try:
                timeout = min(cfg.request_timeout_seconds, remaining)
                with time_limit(timeout):
                    response = self.client.post("chat/completions", json=payload, timeout=timeout)
                    event["http_status"] = response.status_code
                    if response.status_code != 200:
                        event["error"] = f"HTTP {response.status_code}: {response.text[:2000]}"
                        retryable = response.status_code in {429, 500, 502, 503, 504}
                    else:
                        raw = response.json()
                        event["response"] = raw
                        event["accounting"] = self.accounting.measure(raw)
                        if raw.get("model", cfg.name) != cfg.name:
                            raise ValueError("Response model ID differs from configured model")
                        expected = cfg.server_metadata.get("system_fingerprint")
                        if expected and raw.get("system_fingerprint") != expected:
                            raise ValueError(
                                "Response system_fingerprint differs from configuration"
                            )
                        choices = raw.get("choices") or []
                        if len(choices) != 1 or not isinstance(choices[0].get("message"), dict):
                            event["error"] = "Expected exactly one Chat Completions choice"
            except (httpx.HTTPError, ValueError, KeyError, TypeError, WallTimeout) as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                retryable = isinstance(exc, (httpx.TransportError, WallTimeout))
            except BaseException as exc:
                event["error"] = f"Interrupted: {type(exc).__name__}"
                raise
            finally:
                event["elapsed_seconds"] = time.perf_counter() - started
                # A failed request can have consumed unobserved server-side generation.
                event["generation_unknown"] = event["response"] is None
                event["event_type"] = "llm_finished"
                record(event)
            if "error" not in event:
                return event["response"]
            if not retryable or attempt == cfg.retries:
                raise ModelFailure(event["error"])
            backoff = cfg.retry_backoff_seconds * 2**attempt
            if time.perf_counter() + backoff >= deadline:
                raise WallTimeout("Task deadline during retry backoff")
            time.sleep(backoff)
        raise AssertionError("Unreachable")
