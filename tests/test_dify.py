import json
from pathlib import Path

import pytest

from inferdoctor.cli import main
from inferdoctor.core.dify import (
    DIFY_PERF_SCHEMA_VERSION,
    DifyAPIClient,
    DifyConfig,
    export_dify_template,
    interpret_dify_events,
    optimize_dify,
    parse_sse_lines,
    run_dify_check,
    run_dify_knowledge_check,
    run_dify_perf,
    run_dify_smoke,
    summarize_dify_trace_events,
    validate_dify_kit,
)
from inferdoctor.core.perf_baseline import baseline_from_report
from inferdoctor.core.perf_compare import compare_performance


class FakeChatClient:
    def __init__(self, base_url, api_key=None, timeout=30.0):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout

    def get_info(self):
        return {"name": "Demo", "mode": "advanced-chat", "description": "demo"}

    def run_chat_stream(self, query, *, user, show_answer=False):
        return {
            "event_count": 3,
            "first_event_index": 0,
            "first_visible_text_index": 1,
            "ttft_seconds": 0.8,
            "total_latency_seconds": 2.4,
            "node_event_count": 1,
            "workflow_event_count": 1,
            "completion_status": "completed",
            "errors": [],
            "answer_preview": "ok" if show_answer else None,
            "answer_retained": show_answer,
        }


class FakeWorkflowClient(FakeChatClient):
    def get_info(self):
        return {"name": "Workflow", "mode": "workflow"}

    def run_workflow_stream(self, query, *, user, show_answer=False):
        return {
            "event_count": 4,
            "first_event_index": 0,
            "first_visible_text_index": 2,
            "ttft_seconds": 1.2,
            "total_latency_seconds": 3.0,
            "node_event_count": 2,
            "workflow_event_count": 2,
            "completion_status": "completed",
            "errors": [],
        }


class FakeKnowledgeClient(FakeChatClient):
    def retrieve_chunks(self, dataset_id, query):
        return {"records": [{"score": 0.91, "segment": {"content": "private text suppressed"}}]}


def _config():
    return DifyConfig(
        app_base_url="http://127.0.0.1:5001/v1",
        app_api_key="secret-app-key",
        knowledge_base_url="http://127.0.0.1:5001/v1",
        knowledge_api_key="secret-knowledge-key",
        dataset_id="dataset-id",
    )


def test_dify_template_export_and_validate(tmp_path):
    output = tmp_path / "kit"
    written = export_dify_template("local-private-rag", output)

    result = validate_dify_kit(output)

    assert len(written) >= 9
    assert result["status"] == "WARN"
    assert result["readiness_score"] > 80
    assert (output / "dify_app.yaml").exists()
    dsl = (output / "dify_app.yaml").read_text(encoding="utf-8")
    assert "MODEL_NAME_PLACEHOLDER" not in dsl
    assert "provider: ''" in dsl
    assert "dataset_ids: []" in dsl


def test_sse_parser_handles_dify_chat_and_workflow_events():
    events = parse_sse_lines(
        [
            b": keepalive\n",
            b"event: workflow_started\n",
            b'data: {\"event\":\"workflow_started\"}\n',
            b"\n",
            b"event: message\n",
            b'data: {\"answer\":\"hello\"}\n',
            b"\n",
            b"event: message_end\n",
            b"data: {}\n",
            b"\n",
        ]
    )
    metrics = interpret_dify_events(events)

    assert len(events) == 3
    assert metrics["first_visible_text_index"] == 1
    assert metrics["completion_status"] == "completed"
    assert metrics["workflow_event_count"] == 1


def test_dify_check_uses_info_api_with_redacted_endpoint():
    result = run_dify_check(_config(), client_factory=FakeChatClient)

    assert result["status"] == "PASS"
    assert result["app_mode"] == "advanced-chat"
    assert "smoke" in result["supported_operations"]


def test_dify_check_without_key_does_not_call_network():
    result = run_dify_check(DifyConfig(app_base_url="http://127.0.0.1:5001/v1"))

    assert result["status"] == "WARN"
    assert result["authenticated"] is False
    assert "DIFY_APP_API_KEY" in result["warnings"][-1]


def test_dify_smoke_supports_chatflow_and_workflow():
    chat = run_dify_smoke(_config(), client_factory=FakeChatClient)
    workflow = run_dify_smoke(_config(), client_factory=FakeWorkflowClient)

    assert chat["status"] == "PASS"
    assert chat["app_mode"] == "advanced-chat"
    assert workflow["status"] == "PASS"
    assert workflow["app_mode"] == "workflow"


def test_dify_perf_report_works_with_baseline_and_compare():
    report = run_dify_perf(_config(), runs=2, warmup=1, client_factory=FakeChatClient)
    better = dict(report)
    better["metrics"] = dict(report["metrics"], ttft_seconds=0.4)
    better["metrics"]["aggregate"] = dict(report["metrics"]["aggregate"], ttft_median=0.4)

    baseline = baseline_from_report(report, name="before")
    candidate = baseline_from_report(better, name="after")
    comparison = compare_performance(baseline, candidate)

    assert report["schema_version"] == DIFY_PERF_SCHEMA_VERSION
    assert baseline["source_schema_version"] == DIFY_PERF_SCHEMA_VERSION
    assert comparison["metric_changes"]["ttft_seconds"]["direction"] == "improvement"


def test_dify_optimize_uses_report_and_kit(tmp_path):
    output = tmp_path / "kit"
    export_dify_template("local-private-rag", output)
    report = run_dify_perf(_config(), runs=1, client_factory=FakeChatClient)
    report_path = tmp_path / "perf.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    plan = optimize_dify(report_path=str(report_path), kit_path=str(output), retrieval_ms=900)

    assert plan["schema_version"].endswith(".v1")
    assert any(item["evidence"] in {"Observed", "Strongly indicated"} for item in plan["recommendations"])


def test_dify_knowledge_check_suppresses_content_by_default():
    result = run_dify_knowledge_check(_config(), client_factory=FakeKnowledgeClient)

    assert result["status"] == "PASS"
    assert result["result_count"] == 1
    assert result["content_preview"] is None
    assert result["content_retained"] is False


def test_dify_cli_template_flow(tmp_path, capsys):
    output = tmp_path / "kit"

    assert main(["dify", "template", "list"]) == 0
    assert "local-private-rag" in capsys.readouterr().out
    assert main(["dify", "template", "export", "local-private-rag", "--output", str(output)]) == 0
    assert main(["dify", "validate", str(output)]) == 0
    assert main(["dify", "smoke", "--kit", str(output), "--dry-run"]) == 0


def test_dify_cli_help_pages(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["dify", "--help"])
    assert exc.value.code == 0
    assert "Dify" in capsys.readouterr().out


def test_dify_closed_loop_golden_path(tmp_path):
    kit = tmp_path / "kit"
    export_dify_template("local-private-rag", kit)
    validation = validate_dify_kit(kit)
    smoke = run_dify_smoke(_config(), kit_path=str(kit), dry_run=True)
    before = run_dify_perf(_config(), runs=1, client_factory=FakeChatClient)
    after = dict(before)
    after["metrics"] = dict(before["metrics"], ttft_seconds=0.35, total_latency_seconds=1.8)
    after["metrics"]["aggregate"] = dict(before["metrics"]["aggregate"], ttft_median=0.35, total_latency_median=1.8)
    after_path = tmp_path / "after.json"
    after_path.write_text(json.dumps(after), encoding="utf-8")
    before_baseline = baseline_from_report(before, name="before")
    after_baseline = baseline_from_report(after, name="after")
    comparison = compare_performance(before_baseline, after_baseline)
    plan = optimize_dify(report_path=str(after_path), kit_path=str(kit), retrieval_ms=500)

    assert validation["status"] == "WARN"
    assert smoke["status"] == "PASS"
    assert before["source_type"] == "dify"
    assert comparison["verdict"] == "improvement"
    assert plan["recommendations"]


def test_dify_validate_standalone_dsl_does_not_scan_siblings(tmp_path):
    target = tmp_path / "target.yml"
    sibling = tmp_path / "sibling-secret.txt"
    venv_file = tmp_path / "sibling-venv" / "lib" / "python3.12" / "site-packages" / "fake.py"
    binary = tmp_path / "sibling-binary.bin"
    target.write_text("app:\n  mode: advanced-chat\nworkflow:\n  graph:\n    nodes: []\n    edges: []\n", encoding="utf-8")
    sibling.write_text("DIFY_APP_API_KEY=sk-should-not-be-scanned-1234567890\n", encoding="utf-8")
    venv_file.parent.mkdir(parents=True)
    venv_file.write_text("API_TOKEN=sk-venv-should-not-be-scanned-1234567890\n", encoding="utf-8")
    binary.write_bytes(b"\x00\x01token=sk-binary-should-not-be-scanned-1234567890")

    result = validate_dify_kit(target)
    rendered = json.dumps(result)

    assert "sibling-secret" not in rendered
    assert "sibling-venv" not in rendered
    assert "sibling-binary" not in rendered


def test_dify_validate_detects_secret_in_target_dsl(tmp_path):
    target = tmp_path / "target.yml"
    target.write_text("app:\n  mode: advanced-chat\nx: DIFY_APP_API_KEY=sk-target-secret-1234567890\n", encoding="utf-8")

    result = validate_dify_kit(target)

    assert any("secret scan" == check["item"] for check in result["checks"])
    assert any("target.yml" in check["detail"] for check in result["checks"])


def test_dify_validate_kit_skips_external_symlink_and_binary(tmp_path):
    kit = tmp_path / "kit"
    export_dify_template("local-private-rag", kit)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("DIFY_APP_API_KEY=sk-outside-secret-1234567890\n", encoding="utf-8")
    (kit / "sample_docs" / "outside-link.txt").symlink_to(outside)
    (kit / "sample_docs" / "binary.bin").write_bytes(b"\x00\x01DIFY_APP_API_KEY=sk-binary-secret-1234567890")

    result = validate_dify_kit(kit)
    rendered = json.dumps(result)

    assert "outside-secret" not in rendered
    assert "binary.bin" not in rendered
    assert result["validation_level"] == "current_dify_structural_compatibility_validated"


def test_dify_validate_rejects_path_traversal_manifest(tmp_path):
    kit = tmp_path / "kit"
    export_dify_template("local-private-rag", kit)
    manifest = kit / "manifest.yaml"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\nextra_file: ../secret.txt\n", encoding="utf-8")

    result = validate_dify_kit(manifest)

    assert result["status"] == "FAIL"
    assert any(check["item"] == "manifest paths" for check in result["checks"])


def test_old_conceptual_dsl_fails_current_structural_validation(tmp_path):
    old = tmp_path / "old.yml"
    old.write_text(
        "app:\n  mode: advanced-chat\nworkflow:\n  graph:\n    nodes:\n"
        "    - id: start\n      type: start\n    edges: []\n",
        encoding="utf-8",
    )

    result = validate_dify_kit(old)

    assert result["status"] == "FAIL"
    assert result["validation_level"] == "yaml_parsed"
    assert any(check["item"] == "top-level kind" and check["status"] == "FAIL" for check in result["checks"])


def test_sanitized_current_dify_fixture_validates():
    fixture = Path("tests/fixtures/dify/current_cloud_chatflow_structure.yml")
    result = validate_dify_kit(fixture)

    assert result["status"] == "WARN"
    assert result["validation_level"] == "current_dify_structural_compatibility_validated"
    assert any("Knowledge dataset is unresolved" in item for item in result["warnings"])


def test_dify_trace_event_summary_suppresses_private_content():
    events = [
        {
            "event": "node_finished",
            "json": {
                "event": "node_finished",
                "task_id": "task-1",
                "workflow_run_id": "run-1",
                "data": {
                    "node_id": "node-1",
                    "node_type": "question-classifier",
                    "title": "SECRET NODE TITLE",
                    "index": 1,
                    "predecessor_node_id": "start",
                    "inputs": {
                        "query": "SECRET USER QUERY",
                    },
                    "process_data": {
                        "classification": "SECRET CLASS",
                    },
                    "outputs": {
                        "result": "SECRET OUTPUT",
                    },
                    "status": "succeeded",
                    "elapsed_time": 0.25,
                    "inputs_truncated": False,
                    "process_data_truncated": False,
                    "outputs_truncated": False,
                },
            },
        },
        {
            "event": "agent_thought",
            "json": {
                "event": "agent_thought",
                "task_id": "task-1",
                "position": 1,
                "thought": "SECRET THOUGHT",
                "observation": "SECRET OBSERVATION",
                "tool": "crm_lookup",
                "tool_input": "SECRET TOOL INPUT",
            },
        },
        {
            "event": "message_end",
            "json": {
                "event": "message_end",
                "task_id": "task-1",
                "id": "message-1",
                "metadata": {
                    "retriever_resources": [
                        {
                            "position": 1,
                            "dataset_id": "dataset-1",
                            "document_id": "document-1",
                            "document_name": "SECRET DOCUMENT NAME",
                            "segment_id": "segment-1",
                            "score": 0.93,
                            "content": "SECRET RETRIEVED CONTENT",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    },
                },
            },
        },
    ]

    result = summarize_dify_trace_events(
        events
    )

    dumped = json.dumps(
        result,
        ensure_ascii=False,
    )

    for secret in (
        "SECRET NODE TITLE",
        "SECRET USER QUERY",
        "SECRET CLASS",
        "SECRET OUTPUT",
        "SECRET THOUGHT",
        "SECRET OBSERVATION",
        "SECRET TOOL INPUT",
        "SECRET DOCUMENT NAME",
        "SECRET RETRIEVED CONTENT",
    ):
        assert secret not in dumped

    node = result[0]

    assert (
        node["data"]["node_type"]
        == "question-classifier"
    )

    assert node["data"]["input_keys"] == [
        "query"
    ]

    assert node["data"]["output_keys"] == [
        "result"
    ]

    assert node["data"]["title_sha256"]

    agent = result[1]

    assert agent["tool"] == "crm_lookup"
    assert (
        agent["thought_present"]
        is True
    )
    assert (
        agent["tool_input_present"]
        is True
    )

    resource = result[2][
        "retriever_resources"
    ][0]

    assert (
        resource["document_id"]
        == "document-1"
    )
    assert (
        resource["segment_id"]
        == "segment-1"
    )
    assert resource["score"] == 0.93

    assert resource[
        "content_sha256"
    ]

    assert (
        resource[
            "document_name_sha256"
        ]
    )


def test_dify_stream_wrappers_forward_trace_capture_flag():
    class CaptureClient(
        DifyAPIClient
    ):
        def stream_request(
            self,
            path,
            payload,
            *,
            capture_trace_events=False,
        ):
            return {
                "path": path,
                "capture": (
                    capture_trace_events
                ),
                "errors": [],
                "answer_preview": None,
                "answer_retained": False,
            }

    client = CaptureClient(
        "http://127.0.0.1:5001/v1"
    )

    chat = client.run_chat_stream(
        "hello",
        user="u",
        capture_trace_events=True,
    )

    workflow = client.run_workflow_stream(
        "hello",
        user="u",
        capture_trace_events=True,
    )

    assert chat["capture"] is True
    assert workflow["capture"] is True

    assert (
        chat["path"]
        == "/chat-messages"
    )

    assert (
        workflow["path"]
        == "/workflows/run"
    )

