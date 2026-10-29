from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .store import read_summaries
from .util import digest


def export_rows(
    root: Path,
    output: Path,
    include_unsolved: bool = False,
    require_reasoning: bool = False,
) -> dict[str, int]:
    """Allowlist predictor features; all other fields are targets or provenance."""
    if root.resolve() == output.resolve() or output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Put exports outside the run directory")
    stats = {
        "written": 0,
        "excluded_synthetic": 0,
        "excluded_unsolved": 0,
        "excluded_incomplete_usage": 0,
        "excluded_reasoning": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=output.parent, prefix=".export-")
    try:
        with os.fdopen(fd, "w") as f:
            for row in read_summaries(root):
                if row.get("synthetic"):
                    stats["excluded_synthetic"] += 1
                    continue
                if not include_unsolved and (row["success"] is not True or row["censored"]):
                    stats["excluded_unsolved"] += 1
                    continue
                m = row["metrics"]
                if not include_unsolved and not m["usage_complete"]:
                    stats["excluded_incomplete_usage"] += 1
                    continue
                if require_reasoning and not m["reasoning_complete"]:
                    stats["excluded_reasoning"] += 1
                    continue
                # This is the sole feature schema. Do not append attachments/levels/tool outputs.
                features = {"user_request": row["user_request"]}
                # Group repeats, attempts and identical requests together during later splitting.
                group = (
                    row["task_id"].rsplit("_", 1)[0]
                    if row["benchmark"] == "appworld"
                    else row["task_id"]
                )
                record = {
                    "features": features,
                    "targets": {
                        "generated_tokens": m["generated_tokens"],
                        "reasoning_tokens": m["reasoning_tokens"],
                        "non_reasoning_tokens": m["non_reasoning_tokens"],
                        "response_tokens": m["response_tokens"],
                        "reasoning_overhead_tokens": m["reasoning_overhead_tokens"],
                        "prompt_tokens": m["prompt_tokens"],
                        "agent_seconds": row["agent_seconds"],
                        "reasoning_text_tokens_estimate": m["text_token_estimates"][
                            "reasoning_text_tokens"
                        ]["total"],
                        "response_text_tokens_estimate": m["text_token_estimates"][
                            "response_text_tokens"
                        ]["total"],
                    },
                    "provenance": {
                        "experiment_id": row["experiment_id"],
                        "attempt_id": row["attempt_id"],
                        "benchmark": row["benchmark"],
                        "split": row["split"],
                        "task_id": row["task_id"],
                        "repeat": row["repeat"],
                        "task_group": f"{row['benchmark']}:{group}",
                        "request_sha256": digest(row["user_request"]),
                        "status": row["status"],
                        "success": row["success"],
                        "censored": row["censored"],
                        "usage_complete": m["usage_complete"],
                        "reasoning_complete": m["reasoning_complete"],
                        "target_semantics": "work_until_termination"
                        if include_unsolved
                        else "work_until_verified_success",
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stats["written"] += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return stats


def describe_run(root: Path) -> dict[str, Any]:
    rows = read_summaries(root)
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    solved = [r for r in rows if r["success"] is True and not r["censored"]]
    output: dict[str, Any] = {
        "attempts": len(rows),
        "verified_solved": len(solved),
        "unscored": sum(r["success"] is None for r in rows),
        "censored": sum(r["censored"] for r in rows),
        "statuses": statuses,
        "complete_usage": sum(r["metrics"]["usage_complete"] for r in rows),
        "complete_reasoning": sum(r["metrics"]["reasoning_complete"] for r in rows),
    }
    for name in ("generated_tokens", "reasoning_tokens", "non_reasoning_tokens", "response_tokens"):
        values = sorted(r["metrics"][name] for r in solved if r["metrics"][name] is not None)
        output[name] = {
            "solved_count_with_measurement": len(values),
            "mean": sum(values) / len(values) if values else None,
            "min": values[0] if values else None,
            "max": values[-1] if values else None,
        }
    return output
