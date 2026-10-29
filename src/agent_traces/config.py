from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    name: str = "UNCONFIGURED"
    api_key_env: str = "VLLM_API_KEY"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 4096
    seed: int = 42
    request_timeout_seconds: float = 180
    retries: int = 0
    retry_backoff_seconds: float = 2
    extra_body: dict[str, Any] = field(default_factory=dict)
    # Record the actual served checkpoint, quantization, template and parser here.
    server_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 40
    max_generated_tokens: int = 60000
    max_task_seconds: float = 1800
    tool_timeout_seconds: float = 60
    setup_timeout_seconds: float = 180
    evaluation_timeout_seconds: float = 120
    max_observation_chars: int = 20000
    max_protocol_errors: int = 3


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "gaia"
    split: str = "validation"
    gaia_root: str = "data/gaia"
    gaia_year: str = "2023"
    appworld_root: str = "data/appworld"
    search_provider: str = "none"
    search_api_key_env: str = "BRAVE_API_KEY"


@dataclass(frozen=True)
class AccountingConfig:
    # Only enable when thinking is explicitly disabled on the server/template.
    reasoning_mode: str = "auto"  # auto | disabled
    tokenizer: str = ""
    tokenizer_revision: str = ""
    # Optional exact span counting from actual generated IDs; markers are verified on the server.
    token_id_format: str = ""  # "" | gemma4 | qwen3
    reasoning_start_token_ids: list[int] = field(default_factory=list)
    reasoning_end_token_ids: list[int] = field(default_factory=list)
    reasoning_implicit_end_token_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    accounting: AccountingConfig = field(default_factory=AccountingConfig)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        m, a, b = self.model, self.agent, self.benchmark
        u = urlsplit(m.base_url)
        if u.scheme not in {"http", "https"} or not u.netloc:
            raise ValueError("model.base_url must be an HTTP(S) URL")
        if u.username or u.password or u.query or u.fragment:
            raise ValueError("Do not put credentials, query parameters or fragments in base_url")
        if not m.name or m.name == "UNCONFIGURED":
            raise ValueError("Set model.name to the exact served model ID")
        for value in (
            m.max_tokens,
            m.request_timeout_seconds,
            a.max_steps,
            a.max_generated_tokens,
            a.max_task_seconds,
            a.tool_timeout_seconds,
            a.setup_timeout_seconds,
            a.evaluation_timeout_seconds,
            a.max_observation_chars,
            a.max_protocol_errors,
        ):
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("All budgets and timeouts must be positive")
        if m.retries < 0 or m.retry_backoff_seconds < 0:
            raise ValueError("Retry settings cannot be negative")
        reserved = {
            "model",
            "messages",
            "stream",
            "n",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "seed",
            "tools",
            "tool_choice",
            "stop",
        }
        if reserved.intersection(m.extra_body):
            raise ValueError("extra_body overrides a controlled request field")
        if re.search(r"(?i)(api[_-]?key|authorization|password|secret)", str(m.extra_body)):
            raise ValueError("Secrets belong in environment variables, not extra_body")
        if b.name not in {"appworld", "gaia"}:
            raise ValueError("benchmark.name must be appworld or gaia")
        splits = {
            "gaia": {"validation", "test"},
            "appworld": {"train", "dev", "test_normal", "test_challenge"},
        }
        if b.split not in splits[b.name]:
            raise ValueError(f"Invalid {b.name} split: {b.split}")
        if b.search_provider not in {"none", "brave"}:
            raise ValueError("search_provider must be none or brave")
        if self.accounting.reasoning_mode not in {"auto", "disabled"}:
            raise ValueError("reasoning_mode must be auto or disabled")
        ac = self.accounting
        if ac.token_id_format not in {"", "gemma4", "qwen3"}:
            raise ValueError("token_id_format must be empty, gemma4 or qwen3")
        if ac.token_id_format:
            if not ac.reasoning_start_token_ids or not ac.reasoning_end_token_ids:
                raise ValueError("Exact token ID accounting requires both marker sequences")
            if any(
                not isinstance(x, int) or x < 0
                for x in ac.reasoning_start_token_ids + ac.reasoning_end_token_ids
            ):
                raise ValueError("Marker IDs must be nonnegative integers")
            if (
                m.extra_body.get("return_token_ids", True) is not True
                or m.extra_body.get("skip_special_tokens", False) is not False
            ):
                raise ValueError(
                    "Token ID accounting requires token IDs and preserved special tokens"
                )


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if re.search(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", expanded):
            raise ValueError(f"Unresolved environment variable in configuration: {value}")
        return expanded
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as f:
        data = _expand(tomllib.load(f))
    types = {
        "model": ModelConfig,
        "agent": AgentConfig,
        "benchmark": BenchmarkConfig,
        "accounting": AccountingConfig,
    }
    if set(data) - types.keys():
        raise ValueError(f"Unknown configuration sections: {set(data) - types.keys()}")
    result = Config(**{k: types[k](**v) for k, v in data.items()})
    result.validate()
    return result
