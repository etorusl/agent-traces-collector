"""Real local HTTP + CLI subprocess + Parquet + a real Python tool process.

Fixtures are synthetic and live only under pytest's temporary directory.
This is a protocol test, not a benchmark/model quality measurement.
"""

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def test_full_cli_run_resume_stats_and_export(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    folder = tmp_path / "gaia" / "2023" / "validation"
    folder.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "synthetic-http-fixture",
                    "Question": "What is 2 + 2?",
                    "Level": "1",
                    "Final answer": "4",
                    "file_name": "",
                    "file_path": "",
                    "Annotator Metadata": {"Steps": "PRIVATE_REFERENCE_METADATA"},
                }
            ]
        ),
        folder / "metadata.parquet",
    )
    posted = []
    model_queries = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, payload):
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            assert self.path == "/v1/models"
            model_queries.append(1)
            self.send_json(
                {
                    "data": [
                        {
                            "id": "fixture-model",
                            "created": len(model_queries),
                            "permission": [{"id": str(len(model_queries))}],
                        }
                    ]
                }
            )

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posted.append(payload)
            assert payload["n"] == 1
            assert payload["stream"] is False
            assert "PRIVATE_REFERENCE_METADATA" not in str(payload)
            action = (
                {"tool": "python", "arguments": {"code": "answer = 2 + 2; print(answer)"}}
                if len(posted) == 1
                else {"final_answer": "4"}
            )
            self.send_json(
                {
                    "model": "fixture-model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "fixture thought",
                                "content": json.dumps(action),
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 12,
                        "total_tokens": 32,
                        "completion_tokens_details": {"reasoning_tokens": 7},
                    },
                }
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = tmp_path / "config.toml"
    config.write_text(
        f'[model]\nname="fixture-model"\nbase_url="http://127.0.0.1:{server.server_port}/v1"\n'
        f'[benchmark]\nname="gaia"\nsplit="validation"\ngaia_root="{folder.parents[1]}"\n',
    )
    run = tmp_path / "run"

    def cli(*args):
        result = subprocess.run(
            [sys.executable, "-m", "agent_traces", *map(str, args)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    try:
        cli("run", "--config", config, "--output", run)
        assert len(posted) == 2
        assert "skip synthetic-http-fixture" in cli(
            "run",
            "--config",
            config,
            "--output",
            run,
            "--resume",
        )
        assert len(posted) == 2, "Resume must not run completed tasks again"
        stats = json.loads(cli("stats", run))
        assert stats["verified_solved"] == 1
        output = tmp_path / "export.jsonl"
        assert json.loads(cli("export", run, "--output", output))["written"] == 1
        row = json.loads(output.read_text())
        assert row["features"] == {"user_request": "What is 2 + 2?"}
        assert row["targets"]["generated_tokens"] == 24
        assert row["targets"]["reasoning_tokens"] == 14
        assert row["targets"]["non_reasoning_tokens"] == 10
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
