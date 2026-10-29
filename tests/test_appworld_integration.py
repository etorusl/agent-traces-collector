import json
import os
import time
from pathlib import Path

import httpx
import pytest

from agent_traces.accounting import TokenAccounting
from agent_traces.agent import collect_task
from agent_traces.benchmarks import TaskSpec
from agent_traces.config import AccountingConfig, BenchmarkConfig, Config, ModelConfig
from agent_traces.model import ModelClient
from agent_traces.store import RunStore, read_events
from agent_traces.worker import list_appworld


@pytest.mark.skipif(
    not os.environ.get("APPWORLD_TEST_ROOT"),
    reason="Set APPWORLD_TEST_ROOT to downloaded official data",
)
def test_official_appworld_exec_and_evaluator_in_real_isolated_process(tmp_path):
    pytest.importorskip("appworld")
    root = str(Path(os.environ["APPWORLD_TEST_ROOT"]).resolve())
    info = list_appworld(root, "dev")
    task_id = info["ids"][0]
    config = Config(
        model=ModelConfig(name="fixture-model"),
        benchmark=BenchmarkConfig(name="appworld", split="dev", appworld_root=root),
    )
    store = RunStore(tmp_path / "run", {"config": config.as_dict()})
    model = ModelClient(config.model, TokenAccounting(AccountingConfig()))
    model.client.close()
    action = json.dumps(
        {
            "tool": "python",
            "arguments": {
                "code": "print(apis.api_docs.show_app_descriptions()); "
                "apis.supervisor.complete_task()",
            },
        }
    )
    model.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": action}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 30, "completion_tokens": 20},
                },
            )
        ),
        base_url="http://fixture/v1/",
    )
    start = time.perf_counter()
    try:
        result = collect_task(TaskSpec("appworld", "dev", task_id), 0, config, model, store)
        assert time.perf_counter() - start > 0
        assert result["status"] == "finished", result
        # This deliberately does not solve the task. The official evaluator must reject it.
        assert result["success"] is False, result
        assert result["agent_seconds"] > 0
        assert result["evaluation_seconds"] > 0
        events = read_events(store.root / "attempts" / result["attempt_id"] / "events.jsonl")
        assert any(e["event_type"] == "evaluation_finished" for e in events)
        assert result["user_request"]
    finally:
        model.close()
        store.close()
