import dataclasses

import pytest

from agent_traces.benchmarks import gaia_score, load_gaia
from agent_traces.config import BenchmarkConfig


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [
        ("$1,000", "1000", True),
        ("25%", "25", True),
        ("0.25", "25", False),
        ("Sea gull!", "seagull", True),
        ("A.B, 2", "A.B;2", True),
        ("AB, 2", "A.B;2", False),
        ("b,a", "a,b", False),
        ("answer: 4", "4", False),
    ],
)
def test_official_gaia_normalization(answer, reference, expected):
    assert gaia_score(answer, reference) is expected


@pytest.fixture
def gaia_root(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    folder = tmp_path / "2023" / "validation"
    folder.mkdir(parents=True)
    (folder / "input.txt").write_text("ATTACHMENT_CONTENT")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "fixture-task",
                    "Question": "Calculate 2 + 2.",
                    "Level": "1",
                    "Final answer": "4",
                    "file_name": "input.txt",
                    "file_path": "2023/validation/input.txt",
                    "Annotator Metadata": {"Steps": "GROUND_TRUTH_STEPS_MUST_NOT_LEAK"},
                }
            ]
        ),
        folder / "metadata.parquet",
    )
    return tmp_path


def test_actual_parquet_schema_and_reference_allowlist(gaia_root):
    tasks = load_gaia(BenchmarkConfig(gaia_root=str(gaia_root)))
    assert len(tasks) == 1
    task = tasks[0]
    assert task.user_request == "Calculate 2 + 2."
    assert task.reference_answer == "4"
    spec = task.agent_spec()
    assert "reference_answer" not in spec
    assert "provenance" not in spec
    assert "Annotator Metadata" not in str(dataclasses.asdict(task))
    assert len(task.provenance["attachment_sha256"][0]) == 64


def test_missing_attachment_is_a_load_error(gaia_root):
    (gaia_root / "2023" / "validation" / "input.txt").unlink()
    with pytest.raises(FileNotFoundError, match="attachment"):
        load_gaia(BenchmarkConfig(gaia_root=str(gaia_root)))


def test_gaia_test_reference_is_always_unscored(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    folder = tmp_path / "2023" / "test"
    folder.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "fixture-test-task",
                    "Question": "A hidden-answer question",
                    "Level": "3",
                    "Final answer": "?",
                    "file_name": "",
                    "file_path": "",
                }
            ]
        ),
        folder / "metadata.parquet",
    )
    tasks = load_gaia(BenchmarkConfig(gaia_root=str(tmp_path), split="test"))
    assert tasks[0].reference_answer is None
