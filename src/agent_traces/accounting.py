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
    if content.startswith("<|channel>thought\n"):
        text = content.removeprefix("<|channel>thought\n")
        if "<channel|>" in text:
            reasoning, answer = text.split("<channel|>", 1)
            return reasoning, answer.lstrip(), "gemma4_markup"
        return text, "", "unfinished_gemma4_markup"
    if "<channel|>" in content and not content.lstrip().startswith(("{", "```")):
        reasoning, answer = content.split("<channel|>", 1)
        return reasoning, answer.lstrip(), "gemma4_prompt_open_markup"
    if "</think>" in content:
        before, after = content.split("</think>", 1)
        return before.removeprefix("<think>"), after.lstrip(), "think_markup"
    if content.startswith("<think>"):
        return content[len("<think>") :], "", "unfinished_think_markup"
    # Empty parser fields do not prove that thinking is disabled.
    return None, content, "unavailable"


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def count_reasoning_ids(
    output: list[int],
    prompt: list[int],
    start: list[int],
    end: list[int],
) -> tuple[int, int]:
    """Count actual IDs strictly inside thinking markers, excluding structural markers.

    A marker already in the prompt is not charged as generated work. Handles a
    truncated/unclosed reasoning span and multiple complete spans.
    """
    last_start = max(
        (i for i in range(len(prompt)) if prompt[i : i + len(start)] == start), default=-1
    )
    last_end = max((i for i in range(len(prompt)) if prompt[i : i + len(end)] == end), default=-1)
    thinking = last_start > last_end
    reasoning = overhead = i = 0
    while i < len(output):
        if output[i : i + len(start)] == start:
            overhead += len(start)
            thinking = True
            i += len(start)
        elif output[i : i + len(end)] == end:
            overhead += len(end)
            thinking = False
            i += len(end)
        else:
            reasoning += int(thinking)
            i += 1
    return reasoning, overhead


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
        token_ids = choices[0].get("token_ids") if choices else None
        ids_valid = isinstance(token_ids, list) and all(_count(i) is not None for i in token_ids)
        generated_source = "server_usage" if completion is not None else "unavailable"
        if completion is None and ids_valid and self.config.token_id_format:
            completion = len(token_ids)
            generated_source = "output_token_ids"
        reasoning_source = "server_usage" if reasoning_count is not None else "unavailable"
        overhead = None
        if self.config.token_id_format:
            if not ids_valid or len(token_ids) != completion:
                warnings.append("missing_or_incomplete_output_token_ids")
                reasoning_count, reasoning_source = None, "unavailable"
            else:
                prompt_ids = response.get("prompt_token_ids")
                if not isinstance(prompt_ids, list):
                    warnings.append("missing_prompt_token_ids")
                    reasoning_count, reasoning_source = None, "unavailable"
                else:
                    reasoning_count, overhead = count_reasoning_ids(
                        token_ids,
                        prompt_ids,
                        self.config.reasoning_start_token_ids,
                        self.config.reasoning_end_token_ids,
                    )
                    reasoning_source = "output_token_ids"
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
            "reasoning_overhead_tokens": overhead,
            "response_tokens": non_reasoning - overhead
            if non_reasoning is not None and overhead is not None
            else None,
            "server_total_tokens": _count(usage.get("total_tokens")),
            "cached_prompt_tokens": _count(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            ),
            "generated_source": generated_source,
            "reasoning_source": reasoning_source,
            "reasoning_text_source": source,
            "text_token_estimates": estimates,
            "warnings": warnings,
        }


def aggregate(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Never turn a partial sum into a complete target."""
    fields = (
        "prompt_tokens",
        "generated_tokens",
        "reasoning_tokens",
        "non_reasoning_tokens",
        "reasoning_overhead_tokens",
        "response_tokens",
    )
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
    out["text_token_estimates"] = {}
    for field in ("reasoning_text_tokens", "response_text_tokens"):
        values = [c.get("accounting", {}).get("text_token_estimates", {}).get(field) for c in calls]
        known = [v for v in values if v is not None]
        out["text_token_estimates"][field] = {
            "total": sum(known) if calls and len(known) == len(calls) else None,
            "observed": sum(known),
            "known_calls": len(known),
            "all_calls": len(calls),
        }
    return out
