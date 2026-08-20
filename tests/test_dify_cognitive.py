import json

from inferdoctor.cli import main
from inferdoctor.core.dify import DifyConfig
from inferdoctor.core.dify_cognitive import (
    capture_dify_cognitive_trace,
)


class FakeCognitiveClient:
    def __init__(
        self,
        base_url,
        api_key=None,
        timeout=30.0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def get_info(self):
        return {
            "name": "Synthetic App",
            "mode": "advanced-chat",
        }

    def run_chat_stream(
        self,
        query,
        *,
        user,
        show_answer=False,
        capture_trace_events=False,
    ):
        assert query == "hello"
        assert show_answer is False
        assert capture_trace_events is True

        return {
            "completion_status": "completed",
            "total_latency_seconds": 1.25,
            "trace_event_capture": (
                "safe_metadata"
            ),
            "trace_events": [
                {
                    "event": "node_finished",
                    "data": {
                        "node_id": "intent",
                        "node_type": (
                            "question-classifier"
                        ),
                        "status": "succeeded",
                        "index": 1,
                        "input_keys": [
                            "query"
                        ],
                        "output_keys": [
                            "class"
                        ],
                    },
                },
                {
                    "event": "node_finished",
                    "data": {
                        "node_id": "llm",
                        "node_type": "llm",
                        "status": "succeeded",
                        "index": 2,
                    },
                },
            ],
        }


def config():
    return DifyConfig(
        app_base_url=(
            "http://127.0.0.1:5001/v1"
        ),
        app_api_key="secret",
    )


def test_capture_reuses_dify_stream():
    result = (
        capture_dify_cognitive_trace(
            config(),
            query="hello",
            client_factory=(
                FakeCognitiveClient
            ),
        )
    )

    assert (
        result["trace"][
            "source_system"
        ]
        == "dify"
    )

    assert (
        result["analysis"][
            "execution_status"
        ]
        == "PASS"
    )

    assert (
        result["analysis"][
            "semantic_status"
        ]
        == "NOT_EVALUATED"
    )

    assert (
        result["analysis"][
            "first_broken_layer"
        ]
        is None
    )


def test_capture_does_not_retain_query():
    result = (
        capture_dify_cognitive_trace(
            config(),
            query="hello",
            client_factory=(
                FakeCognitiveClient
            ),
        )
    )

    dumped = json.dumps(
        result,
        ensure_ascii=False,
    )

    assert '"hello"' not in dumped


def test_dify_trace_cli_writes_capture(
    tmp_path,
    monkeypatch,
):
    output = (
        tmp_path
        / "cognitive.json"
    )

    monkeypatch.setenv(
        "DIFY_APP_API_KEY",
        "secret",
    )

    monkeypatch.setattr(
        "inferdoctor.cli.capture_dify_cognitive_trace",
        lambda config, query: (
            capture_dify_cognitive_trace(
                config,
                query=query,
                client_factory=(
                    FakeCognitiveClient
                ),
            )
        ),
    )

    code = main([
        "dify",
        "trace",
        "capture",
        "--query",
        "hello",
        "--output",
        str(output),
    ])

    assert code == 0
    assert output.exists()

    result = json.loads(
        output.read_text(
            encoding="utf-8"
        )
    )

    assert (
        result["analysis"][
            "semantic_status"
        ]
        == "NOT_EVALUATED"
    )

    assert (
        result["analysis"][
            "first_broken_layer"
        ]
        is None
    )
