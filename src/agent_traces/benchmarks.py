from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .config import BenchmarkConfig
from .util import atomic_json, file_digest, now


@dataclass
class TaskSpec:
    benchmark: str
    split: str
    task_id: str
    user_request: str = ""
    reference_answer: str | None = None
    attachments: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def agent_spec(self) -> dict[str, Any]:
        # Explicit allowlist: never send reference answers/annotator metadata to a worker.
        return {
            k: asdict(self)[k]
            for k in ("benchmark", "split", "task_id", "user_request", "attachments")
        }


def prepare_gaia(root: str, revision: str = "main") -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    repo = "gaia-benchmark/GAIA"
    info = HfApi().dataset_info(repo, revision=revision)
    path = snapshot_download(repo, repo_type="dataset", revision=info.sha, local_dir=root)
    source = {"repo_id": repo, "revision": info.sha, "downloaded_at": now()}
    atomic_json(Path(path) / ".collector-source.json", source)
    return {"path": str(Path(path).resolve()), **source}


def load_gaia(config: BenchmarkConfig) -> list[TaskSpec]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install the gaia extra: uv pip install -e '.[gaia]'") from exc
    root = Path(config.gaia_root).resolve()
    folder = root / config.gaia_year / config.split
    metadata = folder / "metadata.parquet"
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing {metadata}. Run trace-collector prepare-gaia first.")
    source_path = root / ".collector-source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    common = {
        "metadata_sha256": file_digest(metadata),
        "dataset_revision": source.get("revision"),
        "year": config.gaia_year,
    }
    tasks = []
    seen = set()
    for row in pq.read_table(metadata).to_pylist():
        task_id, question = str(row["task_id"]), row["Question"]
        if task_id in seen or not isinstance(question, str) or not question.strip():
            raise ValueError("GAIA metadata has duplicate IDs or an empty/non-text Question")
        seen.add(task_id)
        attachments = []
        if row.get("file_name") or row.get("file_path"):
            relative = row.get("file_path") or str(
                PurePosixPath(config.gaia_year) / config.split / row["file_name"]
            )
            p = PurePosixPath(relative)
            if p.is_absolute() or ".." in p.parts or "\\" in relative:
                raise ValueError(f"Invalid GAIA attachment path: {relative}")
            attachment = root.joinpath(*p.parts)
            if not attachment.is_file():
                raise FileNotFoundError(f"Missing GAIA attachment: {attachment}")
            attachments.append(str(attachment.resolve()))
        answer = row.get("Final answer")
        # Test answers are private; do not score placeholder strings as ground truth.
        reference = (
            str(answer) if config.split == "validation" and answer not in (None, "", "?") else None
        )
        tasks.append(
            TaskSpec(
                "gaia",
                config.split,
                task_id,
                question,
                reference,
                attachments,
                {
                    **common,
                    "level": row.get("Level"),
                    "attachment_sha256": [file_digest(Path(p)) for p in attachments],
                },
            )
        )
    return tasks


def gaia_score(answer: str, reference: str) -> bool:
    """Reimplement the official leaderboard's number/list/string normalization.

    Source: https://huggingface.co/spaces/gaia-benchmark/leaderboard/blob/main/scorer.py
    Ordered lists preserve punctuation; scalar strings discard ASCII punctuation.
    """
    import re
    import string

    def number(s: str) -> float | None:
        try:
            return float(s)
        except ValueError:
            return None

    def normalized(s: str, punct: bool) -> str:
        text = "".join(s.lower().split())
        return text.translate(str.maketrans("", "", string.punctuation)) if punct else text

    def equal(a: str, b: str, punct: bool) -> bool:
        target = number(b)
        if target is None:
            return normalized(a, punct) == normalized(b, punct)
        cleaned = a.translate(str.maketrans("", "", "$%,"))
        return number(cleaned) == target

    if number(reference) is not None:
        return equal(answer, reference, True)
    if "," in reference or ";" in reference:
        a, b = re.split("[,;]", answer), re.split("[,;]", reference)
        return len(a) == len(b) and all(equal(x, y, False) for x, y in zip(a, b, strict=True))
    return equal(answer, reference, True)
