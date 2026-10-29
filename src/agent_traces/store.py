from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .accounting import aggregate
from .util import atomic_json, digest, file_digest, now


def runtime_info() -> dict[str, Any]:
    import platform
    import sys

    source = Path(__file__).parent
    files = {p.name: file_digest(p) for p in sorted(source.glob("*.py"))}
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": dict(sorted(packages.items())),
        "source_sha256": digest(files),
    }


def git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--porcelain")}


class TraceWriter:
    def __init__(self, directory: Path, secrets: list[str]):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.fd = os.open(directory / "events.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(self.fd, "w")
        self.secrets = [s for s in secrets if s and len(s) >= 6]
        self.sequence = 0

    def clean(self, value: Any) -> Any:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        for secret in self.secrets:
            # Replace JSON-escaped versions too, without invalidating JSON syntax.
            escaped = json.dumps(secret, ensure_ascii=False)[1:-1]
            text = text.replace(escaped, "[REDACTED]")
        return json.loads(text)

    def emit(self, event_type: str, **data: Any) -> None:
        self.sequence += 1
        row = {"sequence": self.sequence, "time": now(), "event_type": event_type, **data}
        text = json.dumps(self.clean(row), ensure_ascii=False, allow_nan=False)
        self.file.write(text + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        self.file.close()


def read_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    lines = path.read_text().splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"Corrupt trace: {path}:{i + 1}") from None
            # SIGKILL/power loss can leave only the last line partially written.
    return rows


class RunStore:
    def __init__(self, root: Path, identity: dict[str, Any], resume: bool = False):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = (self.root / ".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = self.root / "manifest.json"
            experiment_id = digest(identity)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if not resume:
                    raise ValueError("Run already exists; use --resume or a new --output directory")
                if manifest["experiment_id"] != experiment_id:
                    raise ValueError(
                        "Resume rejected: configuration, code, runtime or task set changed"
                    )
                self.manifest = manifest
            else:
                self.manifest = {
                    "schema_version": 1,
                    "experiment_id": experiment_id,
                    "created_at": now(),
                    "identity": identity,
                    "git": git_info(),
                }
                atomic_json(manifest_path, self.manifest)
            (self.root / "attempts").mkdir(exist_ok=True)
            self.recover()
        except BaseException:
            self.lock.close()
            raise

    def summaries(self) -> list[dict[str, Any]]:
        return read_summaries(self.root)

    def recover(self) -> None:
        for start in (self.root / "attempts").glob("*/started.json"):
            summary_path = start.parent / "summary.json"
            if summary_path.exists():
                continue
            summary = json.loads(start.read_text())
            events = read_events(start.parent / "events.jsonl")
            terminal = [e for e in events if e["event_type"] == "attempt_finished"]
            if terminal:
                atomic_json(summary_path, terminal[-1]["summary"])
                continue
            calls = [e for e in events if e["event_type"] == "llm_finished"]
            finished = {e["request_id"] for e in calls}
            pending = [
                e
                for e in events
                if e["event_type"] == "llm_started" and e["request_id"] not in finished
            ]
            calls.extend({"accounting": {}} for _ in pending)
            summary.update(
                {
                    "status": "interrupted",
                    "success": None,
                    "censored": True,
                    "error": "Recovered unfinished attempt; generation/time may be unobserved",
                    "ended_at": now(),
                    "agent_seconds": None,
                    "evaluation_seconds": None,
                    "setup_seconds": None,
                    "end_to_end_seconds": None,
                    "tool_seconds": None,
                    "final_answer": None,
                    "metrics": aggregate(calls),
                }
            )
            atomic_json(summary_path, summary)

    def new_attempt(self, initial: dict, secrets: list[str]) -> tuple[TraceWriter, dict]:
        attempt_id = uuid.uuid4().hex
        summary = {
            "schema_version": 1,
            "experiment_id": self.manifest["experiment_id"],
            "attempt_id": attempt_id,
            "started_at": now(),
            **initial,
        }
        writer = TraceWriter(self.root / "attempts" / attempt_id, secrets)
        summary = writer.clean(summary)
        atomic_json(writer.directory / "started.json", summary)
        writer.emit("attempt_started", summary=summary)
        return writer, summary

    def finish(self, writer: TraceWriter, summary: dict) -> None:
        summary.update(writer.clean(summary))
        writer.emit("attempt_finished", summary=summary)
        atomic_json(writer.directory / "summary.json", summary)
        writer.close()

    def close(self) -> None:
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def read_summaries(root: Path) -> list[dict[str, Any]]:
    return sorted(
        (json.loads(p.read_text()) for p in root.glob("attempts/*/summary.json")),
        key=lambda row: (row["started_at"], row["attempt_id"]),
    )
