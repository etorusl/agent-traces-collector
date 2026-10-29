import pytest

from agent_traces.accounting import split_reasoning
from agent_traces.agent import parse_action


def test_gemma_native_python_preserves_arbitrary_code():
    code = "x = {'a': [1, 2]}; print(x['a'])\nprint('quoted \\\" string')"
    text = '<|tool_call>call:python{code:<|"|>' + code + '<|"|>}<tool_call|>'
    assert parse_action(text) == {"tool": "python", "arguments": {"code": code}}


def test_gemma_multiple_typed_args():
    assert parse_action(
        '<|tool_call>call:read_file{path:<|"|>attachments/file.txt<|"|>,offset:10,limit:20}'
        "<tool_call|>",
    ) == {
        "tool": "read_file",
        "arguments": {
            "path": "attachments/file.txt",
            "offset": 10,
            "limit": 20,
        },
    }


@pytest.mark.parametrize(
    "text",
    [
        '<|tool_call>call:python{code:<|"|>unclosed}<tool_call|>',
        '<|tool_call>call:python{code:<|"|>print(1)<|"|>}<tool_call|>trailing',
        '<|tool_call>call:python{code:<|"|>a<|"|>,code:<|"|>b<|"|>}<tool_call|>',
    ],
)
def test_malformed_native_calls_fail_closed(text):
    with pytest.raises(ValueError):
        parse_action(text)


def test_duplicate_thinking_channels_do_not_leak_into_action():
    text = (
        "<|channel>thought\nreasoning<channel|><|channel>thought\n<channel|>"
        '<|tool_call>call:python{code:<|"|>print(1)<|"|>}<tool_call|>'
    )
    reasoning, content, _ = split_reasoning({"content": text})
    assert reasoning.strip() == "reasoning"
    assert parse_action(content)["arguments"]["code"] == "print(1)"


def test_gemma_tool_marker_can_end_thinking_implicitly():
    text = (
        '<|channel>thought\nreasoning<|tool_call>call:python{code:<|"|>print(1)<|"|>}<tool_call|>'
    )
    reasoning, content, _ = split_reasoning({"content": text})
    assert reasoning == "reasoning"
    assert parse_action(content)["tool"] == "python"
