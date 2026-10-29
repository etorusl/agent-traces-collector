import pytest

from agent_traces.accounting import TokenAccounting, aggregate, count_reasoning_ids, split_reasoning
from agent_traces.config import AccountingConfig


def response(usage=None, **message):
    return {"choices": [{"message": {"content": "4", **message}}], "usage": usage or {}}


def test_server_usage_reasoning_is_part_of_completion():
    measured = TokenAccounting(AccountingConfig()).measure(
        response(
            {
                "prompt_tokens": 20,
                "completion_tokens": 13,
                "total_tokens": 33,
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
            reasoning="think",
        )
    )
    totals = aggregate([{"accounting": measured}, {"accounting": measured}])
    assert totals["generated_tokens"] == 26
    assert totals["reasoning_tokens"] == 24
    assert totals["non_reasoning_tokens"] == 2
    assert totals["prompt_tokens"] == 40


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_new_and_legacy_reasoning_fields(field):
    assert split_reasoning({"content": "4", field: "2+2"}) == ("2+2", "4", field)


def test_empty_new_field_does_not_hide_legacy():
    assert split_reasoning({"reasoning": "", "reasoning_content": "x", "content": "4"})[0] == "x"


@pytest.mark.parametrize("text", ["<think>x</think>4", "x</think>4"])
def test_raw_think_markup(text):
    assert split_reasoning({"content": text})[:2] == ("x", "4")


def test_unfinished_reasoning_cannot_become_final_answer():
    assert split_reasoning({"content": "<think>still thinking"})[:2] == ("still thinking", "")


def test_missing_reasoning_is_unknown_and_not_completion():
    measured = TokenAccounting(AccountingConfig()).measure(response({"completion_tokens": 13}))
    assert measured["generated_tokens"] == 13
    assert measured["reasoning_tokens"] is None
    assert measured["non_reasoning_tokens"] is None


def test_disabled_reasoning_is_explicit():
    measured = TokenAccounting(AccountingConfig(reasoning_mode="disabled")).measure(
        response({"completion_tokens": 1}),
    )
    assert measured["reasoning_tokens"] == 0
    assert measured["reasoning_source"] == "configured_disabled"


def test_contradictory_server_default_zero_is_not_trusted():
    measured = TokenAccounting(AccountingConfig()).measure(
        response(
            {"completion_tokens": 13, "completion_tokens_details": {"reasoning_tokens": 0}},
            reasoning="nonempty",
        )
    )
    assert measured["reasoning_tokens"] is None
    assert measured["warnings"]


def test_reasoning_cannot_exceed_generation():
    measured = TokenAccounting(AccountingConfig()).measure(
        response(
            {"completion_tokens": 2, "completion_tokens_details": {"reasoning_tokens": 8}},
        )
    )
    assert measured["reasoning_tokens"] is None


def test_timeout_does_not_make_partial_sum_an_exact_target():
    totals = aggregate([{"accounting": {"generated_tokens": 10}}, {"accounting": {}}])
    assert totals["observed_generated_tokens"] == 10
    assert totals["generated_tokens"] is None
    assert totals["usage_complete"] is False


def test_tokenizer_estimates_stay_separate_from_server_counts():
    class ToyTokenizer:
        def encode(self, text, add_special_tokens):
            return list(text)

    meter = TokenAccounting(AccountingConfig())
    meter.tokenizer = ToyTokenizer()
    measured = meter.measure(response({"completion_tokens": 100}, reasoning="abc"))
    assert measured["reasoning_tokens"] is None
    assert measured["generated_tokens"] == 100
    assert measured["text_token_estimates"] == {
        "reasoning_text_tokens": 3,
        "response_text_tokens": 1,
    }


def test_gemma4_raw_thinking_is_removed_before_json_action():
    reasoning, content, source = split_reasoning(
        {
            "content": '<|channel>thought\nthink here<channel|>{"final_answer":"4"}',
        }
    )
    assert reasoning == "think here"
    assert content == '{"final_answer":"4"}'
    assert source == "gemma4_markup"


@pytest.mark.parametrize(
    ("output", "prompt", "expected"),
    [
        ([100, 45518, 107, 10, 20, 101, 30, 106], [], (2, 4)),
        ([10, 20, 101, 30, 106], [100, 45518, 107], (2, 1)),
        ([100, 45518, 107, 10, 20], [], (2, 3)),
        ([30, 106], [100, 45518, 107, 101], (0, 0)),
    ],
)
def test_exact_output_span_counting(output, prompt, expected):
    assert count_reasoning_ids(output, prompt, [100, 45518, 107], [101]) == expected


def test_gemma4_exact_counts_include_separate_structural_overhead():
    config = AccountingConfig(
        token_id_format="gemma4",
        reasoning_start_token_ids=[100, 45518, 107],
        reasoning_end_token_ids=[101],
    )
    raw = response({"completion_tokens": 8}, content="<|channel>thought\nreal tokens<channel|>4")
    raw["choices"][0]["token_ids"] = [100, 45518, 107, 10, 20, 101, 30, 106]
    raw["prompt_token_ids"] = []
    measured = TokenAccounting(config).measure(raw)
    assert measured["generated_tokens"] == 8
    assert measured["reasoning_tokens"] == 2
    assert measured["reasoning_source"] == "output_token_ids"
    assert measured["reasoning_overhead_tokens"] == 4
    assert measured["response_tokens"] == 2
    assert measured["non_reasoning_tokens"] == 6
    raw["choices"][0]["token_ids"].pop()
    assert TokenAccounting(config).measure(raw)["reasoning_tokens"] is None


def test_native_tool_boundary_is_charged_as_response_not_reasoning():
    assert count_reasoning_ids(
        [100, 45518, 107, 10, 48, 30, 106], [], [100, 45518, 107], [101], [48]
    ) == (1, 3)
