from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .accounting import TokenAccounting
from .agent import collect_task
from .benchmarks import TaskSpec, load_gaia, prepare_gaia
from .config import load_config
from .export import describe_run, export_rows
from .model import ModelClient, server_identity
from .store import RunStore, runtime_info
from .util import digest, file_digest
from .worker import list_appworld


def _tasks(config) -> list[TaskSpec]:
    b = config.benchmark
    if b.name == "gaia":
        return load_gaia(b)
    info = list_appworld(b.appworld_root, b.split, config.agent.setup_timeout_seconds)
    tasks = []
    root = Path(b.appworld_root).resolve()
    for task_id in info["ids"]:
        task_dir = root / "data" / "tasks" / task_id
        specs = json.loads((task_dir / "specs.json").read_text())
        hashes = {
            str(p.relative_to(task_dir)): file_digest(p)
            for p in sorted(task_dir.rglob("*"))
            if p.is_file()
        }
        tasks.append(
            TaskSpec(
                "appworld",
                b.split,
                task_id,
                specs["instruction"],
                provenance={
                    "split_files": info["files"],
                    "task_files_sha256": digest(hashes),
                    "appworld_version": importlib.metadata.version("appworld"),
                },
            )
        )
    # Base databases affect all tasks, and must not change underneath a resumed experiment.
    base = root / "data"
    base_hash = digest(
        {
            str(p.relative_to(base)): file_digest(p)
            for name in ("base_dbs", "api_docs")
            for p in sorted((base / name).rglob("*"))
            if p.is_file()
        }
    )
    for task in tasks:
        task.provenance["shared_data_sha256"] = base_hash
    return tasks


def run(args) -> None:
    config = load_config(args.config)
    if args.limit <= 0 or args.repeats <= 0:
        raise ValueError("--limit and --repeats must be positive")
    if config.benchmark.search_provider == "brave" and not os.environ.get(
        config.benchmark.search_api_key_env,
    ):
        raise ValueError("Configured Brave Search key is missing")
    tasks = _tasks(config)
    if args.task_id:
        wanted = set(args.task_id)
        available = {t.task_id for t in tasks}
        if wanted - available:
            raise ValueError(f"Unknown task IDs: {sorted(wanted - available)}")
        tasks = [t for t in tasks if t.task_id in wanted]
    if not args.all:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("No tasks selected")
    identity = {
        "config": config.as_dict(),
        "runtime": runtime_info(),
        "tasks": [digest(dataclasses.asdict(t)) for t in tasks],
        "repeats": args.repeats,
    }
    store = RunStore(Path(args.output), identity, resume=args.resume)
    model = None
    try:
        model = ModelClient(config.model, TokenAccounting(config.accounting))
        # Save server identity and require the configured served ID to exist.
        models = model.models()
        markers = model.verify_accounting()
        ids = {m["id"] for m in models.get("data", [])}
        if config.model.name not in ids:
            raise ValueError(f"Served model {config.model.name!r} not in /models: {sorted(ids)}")
        server_path = store.root / "server.json"
        if server_path.exists():
            previous = json.loads(server_path.read_text())
            if server_identity(previous) != server_identity(models):
                raise ValueError("Server /models changed; use a new experiment directory")
        else:
            from .util import atomic_json

            atomic_json(server_path, models)
        if markers:
            from .util import atomic_json

            atomic_json(store.root / "token_markers.json", markers)
        previous = {(s["task_id"], s["repeat"]): s for s in store.summaries()}
        for task in tasks:
            for repeat in range(args.repeats):
                old = previous.get((task.task_id, repeat))
                if (
                    old
                    and old["status"] != "interrupted"
                    and not (args.retry_failed and old["success"] is not True)
                ):
                    print(f"skip {task.task_id} repeat={repeat}", flush=True)
                    continue
                print(f"start {config.benchmark.name}/{task.task_id} repeat={repeat}", flush=True)
                result = collect_task(task, repeat, config, model, store)
                metrics = result["metrics"]
                print(
                    json.dumps(
                        {
                            "task_id": task.task_id,
                            "repeat": repeat,
                            "status": result["status"],
                            "success": result["success"],
                            "generated_tokens": metrics["generated_tokens"],
                            "reasoning_tokens": metrics["reasoning_tokens"],
                            "non_reasoning_tokens": metrics["non_reasoning_tokens"],
                            "agent_seconds": result["agent_seconds"],
                            "error": result["error"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        print(json.dumps(describe_run(store.root), indent=2, ensure_ascii=False))
    finally:
        if model:
            model.close()
        store.close()


def doctor(args) -> None:
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "benchmark": config.benchmark.name,
                "model": config.model.name,
                "api_key_present": bool(os.environ.get(config.model.api_key_env)),
                "search_provider": config.benchmark.search_provider,
            },
            indent=2,
        )
    )
    tasks = _tasks(config)
    print(f"Dataset ready: {len(tasks)} tasks")
    accounting = TokenAccounting(config.accounting)
    model = ModelClient(config.model, accounting)
    try:
        print(json.dumps({"server_models": model.models()}, indent=2))
        markers = model.verify_accounting()
        if markers:
            print(json.dumps({"verified_token_markers": markers}, indent=2))
        if args.probe:
            events = []
            model.complete(
                [{"role": "user", "content": "What is 2+2? Answer briefly."}],
                min(config.model.max_tokens, 256),
                time.perf_counter() + config.model.request_timeout_seconds,
                events.append,
            )
            event = next(e for e in reversed(events) if e["event_type"] == "llm_finished")
            print(
                json.dumps(
                    {
                        "probe_accounting": event["accounting"],
                        "raw_usage": event["response"].get("usage"),
                        "message_fields": list(event["response"]["choices"][0]["message"]),
                    },
                    indent=2,
                )
            )
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local agent traces with remote vLLM")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-gaia", help="Download the official gated HF dataset")
    p.add_argument("--root", default="data/gaia")
    p.add_argument("--revision", default="main")
    p = commands.add_parser("prepare-appworld", help="Install and download official AppWorld")
    p.add_argument("--root", default="data/appworld")
    p = commands.add_parser("doctor", help="Check dataset and endpoint; optional billable probe")
    p.add_argument("--config", required=True)
    p.add_argument("--probe", action="store_true")
    p = commands.add_parser("run", help="Run real tasks, sequentially, with durable traces")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--all", action="store_true", help="Select all tasks instead of --limit")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--task-id", action="append")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p = commands.add_parser("stats")
    p.add_argument("run", type=Path)
    p = commands.add_parser("export", help="Export only raw user text as predictor features")
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--include-unsolved",
        action="store_true",
        help="Diagnostic export including censored/failed/unscored attempts",
    )
    p.add_argument("--require-reasoning", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "prepare-gaia":
            print(json.dumps(prepare_gaia(args.root, args.revision), indent=2))
        elif args.command == "prepare-appworld":
            if (Path(args.root) / "data").exists():
                raise ValueError("AppWorld data already exists; reuse it or choose a new --root")
            subprocess.run([sys.executable, "-m", "appworld.cli", "install"], check=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "appworld.cli",
                    "download",
                    "data",
                    "--root",
                    str(Path(args.root).resolve()),
                ],
                check=True,
            )
        elif args.command == "run":
            run(args)
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "stats":
            print(json.dumps(describe_run(args.run), indent=2, ensure_ascii=False))
        elif args.command == "export":
            print(
                json.dumps(
                    export_rows(
                        args.run, args.output, args.include_unsolved, args.require_reasoning
                    ),
                    indent=2,
                )
            )
    except KeyboardInterrupt:
        print("Interrupted; completed events are saved. Resume with --resume.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
