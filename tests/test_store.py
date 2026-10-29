import json

import pytest

from agent_traces.store import RunStore, TraceWriter, read_summaries


def test_resume_rejects_changed_experiment_and_concurrent_writers(tmp_path):
    root = tmp_path / "run"
    store = RunStore(root, {"config": "A"})
    with pytest.raises(BlockingIOError):
        RunStore(root, {"config": "A"}, resume=True)
    store.close()
    with pytest.raises(ValueError, match="Resume rejected"):
        RunStore(root, {"config": "B"}, resume=True)
    resumed = RunStore(root, {"config": "A"}, resume=True)
    resumed.close()


def test_crash_recovery_records_inflight_request_as_unknown(tmp_path):
    root = tmp_path / "run"
    store = RunStore(root, {"config": "A"})
    writer, _ = store.new_attempt(
        {"task_id": "fixture", "repeat": 0, "user_request": "2+2?", "benchmark": "gaia"}, []
    )
    writer.emit("llm_started", request_id="not-finished")
    writer.close()
    # Simulate a partially written final JSONL record after a hard crash.
    with (writer.directory / "events.jsonl").open("a") as f:
        f.write('{"event_type":')
    store.close()
    resumed = RunStore(root, {"config": "A"}, resume=True)
    result = read_summaries(root)[0]
    assert result["status"] == "interrupted"
    assert result["censored"] is True
    assert result["metrics"]["generated_tokens"] is None
    assert result["agent_seconds"] is None
    resumed.close()


def test_secret_redaction_preserves_json_syntax(tmp_path):
    secret = 'secret-"quoted"-key'
    writer = TraceWriter(tmp_path / "trace", [secret])
    writer.emit("example", error=f"credential: {secret}")
    writer.close()
    result = json.loads((writer.directory / "events.jsonl").read_text())
    assert result["error"] == "credential: [REDACTED]"


def test_terminal_event_recovers_complete_summary_after_crash(tmp_path):
    store = RunStore(tmp_path / "run", {"config": "A"})
    writer, initial = store.new_attempt({"task_id": "fixture", "repeat": 0}, [])
    terminal = {**initial, "status": "finished", "success": True, "censored": False}
    writer.emit("attempt_finished", summary=terminal)
    writer.close()
    store.close()
    resumed = RunStore(tmp_path / "run", {"config": "A"}, resume=True)
    assert resumed.summaries()[0] == terminal
    resumed.close()


def test_summary_error_is_also_redacted(tmp_path):
    store = RunStore(tmp_path / "run", {"config": "A"})
    writer, summary = store.new_attempt({"task_id": "fixture", "repeat": 0}, ["secret-api-key"])
    summary["error"] = "server echoed secret-api-key"
    store.finish(writer, summary)
    assert "secret-api-key" not in (writer.directory / "summary.json").read_text()
    assert summary["error"] == "server echoed [REDACTED]"
    store.close()
