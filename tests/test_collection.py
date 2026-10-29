import dataclasses
import json
import time

import httpx
import pytest

from agent_traces.accounting import TokenAccounting
from agent_traces.agent import collect_task
from agent_traces.benchmarks import TaskSpec
from agent_traces.config import AccountingConfig, AgentConfig, Config, ModelConfig
from agent_traces.export import describe_run, export_rows
from agent_traces.model import ModelClient
from agent_traces.store import RunStore, read_events


def make_response(content, completion=10, reasoning=6, finish_reason="stop"):
    return {
        "id": "fixture-response",
        "model": "fixture-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": "fixture reasoning",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": completion,
            "total_tokens": 30 + completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


@pytest.fixture
def experiment(tmp_path):
    config = Config(
        model=ModelConfig(name="fixture-model"),
        agent=AgentConfig(
            max_steps=5, max_task_seconds=15, tool_timeout_seconds=2, setup_timeout_seconds=15
        ),
    )
    store = RunStore(tmp_path / "run", {"config": config.as_dict()})
    model = ModelClient(config.model, TokenAccounting(AccountingConfig()))
    model.client.close()
    yield config, store, model
    model.close()
    store.close()


def test_multistep_real_python_worker_totals_and_no_feature_leakage(experiment, tmp_path):
    config, store, model = experiment
    attachment = tmp_path / "secret.txt"
    attachment.write_text("SECRET_ATTACHMENT_CONTENT")
    task = TaskSpec(
        "gaia", "validation", "fixture-task", "Calculate 2 + 2.", "4", attachments=[str(attachment)]
    )
    requests = []

    def transport(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            content = json.dumps(
                {
                    "tool": "python",
                    "arguments": {
                        "code": "x = 2 + 2; print(open(attachments[0]).read()); print(x)",
                    },
                }
            )
        else:
            content = json.dumps({"final_answer": "4"})
        return httpx.Response(200, json=make_response(content))

    model.client = httpx.Client(
        transport=httpx.MockTransport(transport), base_url="http://fixture/v1/"
    )
    result = collect_task(task, 0, config, model, store)
    assert result["success"] is True
    assert result["status"] == "finished"
    assert result["censored"] is False
    assert result["metrics"]["generated_tokens"] == 20
    assert result["metrics"]["reasoning_tokens"] == 12
    assert result["metrics"]["non_reasoning_tokens"] == 8
    assert describe_run(store.root)["generated_tokens"]["all_attempts_total"] == 20
    assert len(requests) == 2
    assert "SECRET_ATTACHMENT_CONTENT" in requests[1]["messages"][-1]["content"]
    assert "SECRET_ATTACHMENT_CONTENT" not in str(requests[0])
    output = tmp_path / "export.jsonl"
    assert export_rows(store.root, output)["written"] == 1
    row = json.loads(output.read_text())
    assert row["features"] == {"user_request": task.user_request}
    assert "SECRET_ATTACHMENT_CONTENT" not in output.read_text()
    assert "attachments/" not in output.read_text()


def test_token_budget_and_wrong_answers_are_not_solved_targets(experiment, tmp_path):
    config, store, model = experiment
    config = dataclasses.replace(
        config,
        agent=dataclasses.replace(
            config.agent,
            max_generated_tokens=10,
        ),
    )
    model.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=make_response('{"tool":"python","arguments":{"code":"print(1)"}}'),
            )
        ),
        base_url="http://fixture/v1/",
    )
    task = TaskSpec("gaia", "validation", "fixture", "Compute 2+2", "4")
    result = collect_task(task, 0, config, model, store)
    assert result["status"] == "token_budget"
    assert result["censored"] is True
    assert result["success"] is False
    assert export_rows(store.root, tmp_path / "solved.jsonl")["written"] == 0
    assert (
        export_rows(store.root, tmp_path / "diagnostic.jsonl", include_unsolved=True)["written"]
        == 1
    )


def test_retry_consumption_is_explicitly_unknown(experiment):
    config, store, model = experiment
    model.config = dataclasses.replace(model.config, retries=1, retry_backoff_seconds=0)
    attempts = []

    def transport(request):
        attempts.append(request)
        if len(attempts) == 1:
            raise httpx.ReadTimeout("fixture timeout")
        return httpx.Response(200, json=make_response('{"final_answer":"4"}'))

    model.client = httpx.Client(
        transport=httpx.MockTransport(transport), base_url="http://fixture/v1/"
    )
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert result["success"] is True
    assert result["metrics"]["generated_tokens"] is None
    assert result["metrics"]["observed_generated_tokens"] == 10
    assert result["metrics"]["llm_attempts"] == 2
    assert result["metrics"]["usage_complete"] is False
    assert describe_run(store.root)["generated_tokens"]["all_attempts_total"] is None


def test_python_timeout_terminates_worker_and_saves_trace(experiment):
    config, store, model = experiment
    config = dataclasses.replace(
        config,
        agent=dataclasses.replace(
            config.agent,
            tool_timeout_seconds=0.15,
        ),
    )
    model.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=make_response('{"tool":"python","arguments":{"code":"while True: pass"}}'),
            )
        ),
        base_url="http://fixture/v1/",
    )
    start = time.perf_counter()
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert time.perf_counter() - start < 10
    assert result["status"] == "tool_timeout"
    assert result["censored"] is True
    events = read_events(store.root / "attempts" / result["attempt_id"] / "events.jsonl")
    assert events[-1]["event_type"] == "attempt_finished"
    assert any(e["event_type"] == "tool_finished" and "error" in e for e in events)


def test_hard_model_deadline_saves_unknown_generation(experiment):
    config, store, model = experiment
    model.config = dataclasses.replace(model.config, request_timeout_seconds=0.1)

    def transport(request):
        time.sleep(3)
        return httpx.Response(200, json=make_response('{"final_answer":"4"}'))

    model.client = httpx.Client(
        transport=httpx.MockTransport(transport), base_url="http://fixture/v1/"
    )
    start = time.perf_counter()
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert time.perf_counter() - start < 2
    assert result["status"] == "llm_error"
    assert result["metrics"]["generated_tokens"] is None


def test_model_length_is_censored_even_if_json_is_valid(experiment):
    config, store, model = experiment
    model.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=make_response('{"final_answer":"4"}', finish_reason="length"),
            )
        ),
        base_url="http://fixture/v1/",
    )
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert result["status"] == "model_length"
    assert result["censored"] is True


def test_unknown_usage_stops_instead_of_ignoring_token_budget(experiment):
    config, store, model = experiment
    raw = make_response('{"final_answer":"4"}')
    del raw["usage"]
    model.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json=raw,
            )
        ),
        base_url="http://fixture/v1/",
    )
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert result["status"] == "missing_usage"
    assert result["metrics"]["generated_tokens"] is None


def test_protocol_error_tokens_are_included(experiment):
    config, store, model = experiment
    count = 0

    def transport(request):
        nonlocal count
        count += 1
        return httpx.Response(
            200,
            json=make_response(
                "invalid JSON" if count == 1 else '{"final_answer":"4"}',
            ),
        )

    model.client = httpx.Client(
        transport=httpx.MockTransport(transport), base_url="http://fixture/v1/"
    )
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert result["success"] is True
    assert result["protocol_errors"] == 1
    assert result["metrics"]["generated_tokens"] == 20


def test_gemma_profile_uses_native_prompt_and_preserves_native_history(experiment):
    config, store, model = experiment
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, family="gemma4"))
    requests = []
    native = '<|tool_call>call:python{code:<|"|>print(2+2)<|"|>}<tool_call|>'

    def transport(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=make_response(native if len(requests) == 1 else "4"))

    model.client = httpx.Client(
        transport=httpx.MockTransport(transport), base_url="http://fixture/v1/"
    )
    result = collect_task(
        TaskSpec("gaia", "validation", "fixture", "2+2?", "4"), 0, config, model, store
    )
    assert result["success"] is True
    assert "Use Gemma native tool-call syntax" in requests[0]["messages"][0]["content"]
    assistant = requests[1]["messages"][-2]
    assert assistant["content"] is None
    assert assistant["reasoning"] == "fixture reasoning"
    assert assistant["tool_calls"][0]["function"]["name"] == "python"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"code": "print(2+2)"}
    assert requests[1]["messages"][-1]["role"] == "tool"
