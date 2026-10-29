from __future__ import annotations

from typing import Any

from .config import AccountingConfig


def split_reasoning(message: dict[str, Any]) -> tuple[str | None, str, str]:
    """Support new/old vLLM fields and Qwen's raw thinking markup."""
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise ValueError("This agent expects text Chat Completions content")
    for field in ("reasoning", "reasoning_content"):
        value = message.get(field)
        if isinstance(value, str) and value:
            return value, content, field
    if "</think>" in content:
        before, after = content.split("</think>", 1)
        return before.removeprefix("<think>"), after.lstrip(), "think_markup"
    if content.startswith("<think>"):
        return content[len("<think>") :], "", "unfinished_think_markup"
    # Empty parser fields do not prove that thinking is disabled.
    return None, content, "unavailable"


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class TokenAccounting:
    def __init__(self, config: AccountingConfig):
        self.config = config
        self.tokenizer = None
        if config.tokenizer:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer,
                revision=config.tokenizer_revision or None,
                trust_remote_code=False,
            )

    def measure(self, response: dict[str, Any]) -> dict[str, Any]:
        usage = response.get("usage") or {}
        choices = response.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        reasoning, content, source = split_reasoning(message)
        prompt = _count(usage.get("prompt_tokens"))
        completion = _count(usage.get("completion_tokens"))
        details = usage.get("completion_tokens_details") or {}
        reasoning_count = _count(details.get("reasoning_tokens"))
        warnings = []
        reasoning_source = "server_usage" if reasoning_count is not None else "unavailable"
        if reasoning_count == 0 and reasoning and reasoning.strip():
            # Some serving versions return a default 0 even when thinking is present.
            warnings.append("server_reported_zero_reasoning_with_nonempty_reasoning")
            reasoning_count, reasoning_source = None, "unavailable"
        if reasoning_count is not None and completion is not None and reasoning_count > completion:
            warnings.append("reasoning_exceeds_completion")
            reasoning_count, reasoning_source = None, "unavailable"
        if self.config.reasoning_mode == "disabled" and reasoning_count is None:
            if reasoning and reasoning.strip():
                warnings.append("reasoning_present_but_configured_disabled")
            else:
                reasoning_count, reasoning_source = 0, "configured_disabled"
        non_reasoning = (
            completion - reasoning_count
            if completion is not None and reasoning_count is not None
            else None
        )
        estimates: dict[str, int | None] = {
            "reasoning_text_tokens": None,
            "response_text_tokens": None,
        }
        if self.tokenizer is not None:
            estimates["reasoning_text_tokens"] = (
                len(self.tokenizer.encode(reasoning, add_special_tokens=False))
                if reasoning is not None
                else None
            )
            estimates["response_text_tokens"] = len(
                self.tokenizer.encode(content, add_special_tokens=False)
            )
        return {
            "prompt_tokens": prompt,
            "generated_tokens": completion,
            "reasoning_tokens": reasoning_count,
            "non_reasoning_tokens": non_reasoning,
            "server_total_tokens": _count(usage.get("total_tokens")),
            "cached_prompt_tokens": _count(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            ),
            "generated_source": "server_usage" if completion is not None else "unavailable",
            "reasoning_source": reasoning_source,
            "reasoning_text_source": source,
            "text_token_estimates": estimates,
            "warnings": warnings,
        }


def aggregate(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Never turn a partial sum into a complete target."""
    fields = ("prompt_tokens", "generated_tokens", "reasoning_tokens", "non_reasoning_tokens")
    out: dict[str, Any] = {"llm_attempts": len(calls)}
    for field in fields:
        values = [c.get("accounting", {}).get(field) for c in calls]
        known = [v for v in values if v is not None]
        out[f"observed_{field}"] = sum(known)
        out[field] = sum(known) if len(known) == len(calls) else None
        out[f"{field}_coverage"] = {"known_calls": len(known), "all_calls": len(calls)}
    out["usage_complete"] = all(
        c.get("accounting", {}).get("generated_tokens") is not None for c in calls
    )
    out["reasoning_complete"] = all(
        c.get("accounting", {}).get("reasoning_tokens") is not None for c in calls
    )
    out["llm_seconds"] = sum(c.get("elapsed_seconds", 0) for c in calls)
    return out
