from __future__ import annotations

import multiprocessing as mp
import os
import signal
import traceback
from pathlib import Path
from typing import Any

from .tools import GaiaTools
from .util import file_digest


class AppWorldBackend:
    def __init__(self, task: dict, config: dict, workspace: str):
        os.environ["APPWORLD_ROOT"] = str(Path(config["benchmark"]["appworld_root"]).resolve())
        from appworld import AppWorld

        self.world = AppWorld(
            task_id=task["task_id"],
            experiment_name=f"collector-{Path(workspace).parent.name}",
            timeout_seconds=int(config["agent"]["tool_timeout_seconds"]),
            max_interactions=config["agent"]["max_steps"],
        )

    def setup(self) -> dict[str, Any]:
        return {
            "user_request": self.world.task.instruction,
            "supervisor": dict(self.world.task.supervisor),
            "app_descriptions": self.world.task.app_descriptions,
        }

    def execute(self, name: str, args: dict) -> str:
        if name != "python":
            raise ValueError("AppWorld supports only the python tool")
        return self.world.execute(args["code"])

    def completed(self) -> bool:
        return self.world.task_completed()

    def evaluate(self) -> dict[str, Any]:
        tracker = self.world.evaluate()
        return {"success": bool(tracker.success), "report": tracker.to_dict()}

    def close(self) -> None:
        self.world.close()


def _entry(pipe, task: dict, config: dict, workspace: str) -> None:
    # A separate session lets the parent terminate Python-created descendants as well.
    os.setsid()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    backend = None
    try:
        # Keys used by the collector are not needed by model-generated Python.
        for name in {
            config["model"]["api_key_env"],
            "OPENAI_API_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        }:
            os.environ.pop(name, None)
        if task["benchmark"] == "appworld":
            backend = AppWorldBackend(task, config, workspace)
        else:
            backend = GaiaTools(task, config, workspace)
        pipe.send({"ok": True, "value": backend.setup()})
        while True:
            operation, args = pipe.recv()
            if operation == "close":
                break
            try:
                if operation == "execute":
                    value = backend.execute(*args)
                    complete = backend.completed() if task["benchmark"] == "appworld" else False
                    output = str(value)
                    value = {
                        "output": output[:200000],
                        "output_chars": len(output),
                        "output_truncated": len(output) > 200000,
                        "task_completed": complete,
                    }
                elif operation == "evaluate" and task["benchmark"] == "appworld":
                    value = backend.evaluate()
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                pipe.send({"ok": True, "value": value})
            except Exception as exc:
                pipe.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            pipe.send({"ok": False, "error": traceback.format_exc(limit=10)})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if backend is not None:
            backend.close()
        pipe.close()


class WorkerTimeout(TimeoutError):
    pass


class BackendProcess:
    def __init__(self, task: dict, config: dict, workspace: Path):
        ctx = mp.get_context("spawn")
        self.pipe, child = ctx.Pipe()
        self.process = ctx.Process(target=_entry, args=(child, task, config, str(workspace)))
        self.process.start()
        child.close()

    def receive(self, timeout: float) -> Any:
        if timeout <= 0 or not self.pipe.poll(timeout):
            self.kill()
            raise WorkerTimeout("Benchmark worker timed out and was terminated")
        try:
            response = self.pipe.recv()
        except EOFError as exc:
            raise RuntimeError("Benchmark worker exited unexpectedly") from exc
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["value"]

    def call(self, operation: str, args: Any, timeout: float) -> Any:
        self.pipe.send((operation, args))
        return self.receive(timeout)

    def kill(self) -> None:
        if self.process.is_alive():
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            # Also handles timeout before the child has called setsid().
            if self.process.is_alive():
                self.process.kill()
        self.process.join(timeout=5)

    def close(self) -> None:
        try:
            if self.process.is_alive():
                self.pipe.send(("close", None))
                self.process.join(timeout=2)
        except (BrokenPipeError, OSError):
            pass
        finally:
            self.kill()
            self.pipe.close()


def _list_appworld(pipe, root: str, split: str) -> None:
    try:
        os.environ["APPWORLD_ROOT"] = root
        from appworld import load_task_ids

        task_ids = load_task_ids(split)
        base = Path(root)
        # The split manifest and version are reproducibility metadata, never agent inputs.
        files = [base / "data" / "datasets" / f"{split}.txt", base / "data" / "version.txt"]
        pipe.send(
            {
                "ids": task_ids,
                "files": {str(p.relative_to(base)): file_digest(p) for p in files if p.is_file()},
            }
        )
    except BaseException:
        pipe.send({"error": traceback.format_exc(limit=10)})
    finally:
        pipe.close()


def list_appworld(root: str, split: str, timeout: float = 180) -> dict:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_list_appworld, args=(child, str(Path(root).resolve()), split))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise WorkerTimeout("AppWorld task discovery timed out")
        value = parent.recv()
        if "error" in value:
            raise RuntimeError(value["error"])
        return value
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()
