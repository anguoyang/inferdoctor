import json

from inferdoctor.cli import main
from inferdoctor.core.dify_reliability import (
    BoundedCommandRunner,
    CommandResult,
    diagnose_root_causes,
    extract_log_signatures,
    make_observation,
    parse_compose_services,
    render_reliability_report,
    run_dify_connectivity_check,
    run_dify_evidence_collect,
    run_dify_evidence_explain,
    run_dify_selfhost_inspect,
    run_dify_selfhost_preflight,
    service_inventory,
    validate_evidence,
)


COMPOSE = """services:
  api:
    image: langgenius/dify-api:1.0.0
    environment:
      - DB_HOST=db
      - REDIS_HOST=redis
    ports:
      - "5001:5001"
  worker:
    image: langgenius/dify-api:1.0.0
  plugin-daemon:
    image: langgenius/dify-plugin-daemon:1.0.0
  sandbox:
    image: langgenius/dify-sandbox:1.0.0
  ssrf-proxy:
    image: langgenius/dify-ssrf-proxy:1.0.0
  db:
    image: postgres:15
  redis:
    image: redis:7
  qdrant:
    image: qdrant/qdrant:v1
"""


class FakeRunner(BoundedCommandRunner):
    def __init__(self, *, docker_ok=True, logs=""):
        super().__init__()
        self.docker_ok = docker_ok
        self.logs = logs
        self.calls = []

    def run(self, args, *, timeout=5.0, max_bytes=65536):
        self.calls.append(tuple(args))
        if args[:2] == ["docker", "version"]:
            if self.docker_ok:
                return CommandResult(tuple(args), 0, '{"Server":{"Version":"test"}}', "", "ok")
            return CommandResult(tuple(args), 1, "", "Cannot connect to the Docker daemon", "daemon_unavailable")
        if args[:3] == ["docker", "compose", "version"]:
            return CommandResult(tuple(args), 0, "2.29.0", "", "ok")
        if "ps" in args:
            return CommandResult(tuple(args), 0, json.dumps([{"Service": "api", "State": "running", "Health": "healthy"}]), "", "ok")
        if "logs" in args:
            return CommandResult(tuple(args), 0, self.logs, "", "ok")
        if "exec" in args:
            return CommandResult(tuple(args), 1, "", "connection refused", "non_zero_exit")
        return CommandResult(tuple(args), 0, "", "", "ok")


def _compose(tmp_path):
    path = tmp_path / "docker-compose.yaml"
    path.write_text(COMPOSE, encoding="utf-8")
    return path


def test_evidence_schema_redacts_secret_values():
    obs = make_observation(
        component="api",
        component_role="api",
        layer="logs",
        source_type="test",
        source_reference="http://user:secret@example.test?api_key=hidden",
        check_name="redaction",
        status="warn",
        summary="Bearer abcdef123456",
        sanitized_detail="token=abcdef123456",
    )

    validate_evidence(obs)
    text = json.dumps(obs)
    assert "abcdef123456" not in text
    assert "hidden" not in text
    assert "REDACTED" in text


def test_bounded_runner_rejects_unapproved_executable():
    runner = BoundedCommandRunner(allowed={"docker"})
    result = runner.run(["sh", "-c", "echo unsafe"])

    assert result.category == "not_allowed"


def test_compose_role_detection(tmp_path):
    compose = _compose(tmp_path)
    services, warnings = parse_compose_services(compose)
    inventory, _ = service_inventory(compose)

    roles = {item["role"] for item in inventory}
    assert not warnings
    assert len(services) >= 8
    assert {"api", "worker", "plugin_daemon", "sandbox", "ssrf_proxy", "postgres", "redis", "vector_store"} <= roles


def test_preflight_uses_fake_docker_and_reports_roles(tmp_path):
    compose = _compose(tmp_path)
    report = run_dify_selfhost_preflight(compose_file=str(compose), runner=FakeRunner())

    assert report["schema_version"].endswith(".preflight.v1")
    assert report["status"] in {"READY", "ATTENTION"}
    assert any(item["component_role"] == "plugin_daemon" for item in report["findings"])
    assert any(item["component_role"] == "ssrf_proxy" for item in report["findings"])


def test_preflight_handles_no_docker_daemon(tmp_path):
    compose = _compose(tmp_path)
    report = run_dify_selfhost_preflight(compose_file=str(compose), runner=FakeRunner(docker_ok=False))

    assert report["status"] == "ATTENTION"
    assert any("Docker CLI or daemon is unavailable" in item["summary"] for item in report["findings"])


def test_inspect_extracts_plugin_and_ssrf_signatures(tmp_path):
    compose = _compose(tmp_path)
    logs = "plugin daemon restart loop\nSSRF proxy rejected private address\n"
    report = run_dify_selfhost_inspect(compose_file=str(compose), details=True, runner=FakeRunner(logs=logs))

    signatures = {item["error_signature"] for item in report["findings"] if item.get("error_signature")}
    assert "plugin_daemon_failure" in signatures
    assert "ssrf_rejection" in signatures


def test_connectivity_rejects_private_endpoint_without_opt_in(tmp_path):
    compose = _compose(tmp_path)
    report = run_dify_connectivity_check(endpoint="http://192.168.1.20:8000/v1", compose_file=str(compose))

    assert report["status"] == "FAIL"
    assert "requires --allow-non-local" in json.dumps(report)


def test_root_cause_patterns_prioritize_plugin_daemon():
    evidence = [
        make_observation(component="plugin_daemon", component_role="plugin_daemon", layer="logs", source_type="test", source_reference="logs", check_name="signature", status="warn", summary="plugin not found", sanitized_detail="plugin not found", error_signature="plugin_daemon_failure"),
        make_observation(component="provider", component_role="provider", layer="dify_provider", source_type="test", source_reference="provider", check_name="invoke", status="fail", summary="Provider failed downstream"),
    ]

    candidates = diagnose_root_causes(evidence)
    assert candidates[0]["candidate"].startswith("Plugin Daemon")


def test_evidence_collect_and_explain(tmp_path):
    compose = _compose(tmp_path)
    bundle = run_dify_evidence_collect(compose_file=str(compose), details=True, runner=FakeRunner(logs="worker indexing error"))
    explanation = run_dify_evidence_explain(bundle)

    assert bundle["schema_version"].endswith(".bundle.v1")
    assert explanation["evidence_count"] == len(bundle["evidence"])
    assert explanation["root_cause_candidates"]


def test_report_formats_are_stable(tmp_path):
    compose = _compose(tmp_path)
    report = run_dify_selfhost_preflight(compose_file=str(compose), runner=FakeRunner())

    assert "Dify Self-Host Preflight" in render_reliability_report(report)
    assert render_reliability_report(report, "json").startswith("{")
    assert render_reliability_report(report, "markdown").startswith("# Dify Self-Host Preflight")


def test_dify_reliability_cli_smoke(tmp_path, capsys):
    compose = _compose(tmp_path)
    assert main(["dify", "selfhost", "preflight", "--compose-file", str(compose)]) == 0
    assert "Dify Self-Host Preflight" in capsys.readouterr().out

    bundle = tmp_path / "bundle.json"
    assert main(["dify", "evidence", "collect", "--compose-file", str(compose), "--format", "json", "--output", str(bundle)]) == 0
    assert bundle.exists()

    assert main(["dify", "evidence", "explain", str(bundle), "--format", "markdown"]) == 0
    assert "Dify Evidence Explanation" in capsys.readouterr().out


def test_dify_cli_help_pages(capsys):
    for args in (
        ["dify", "selfhost", "--help"],
        ["dify", "selfhost", "preflight", "--help"],
        ["dify", "selfhost", "inspect", "--help"],
        ["dify", "connectivity", "check", "--help"],
        ["dify", "evidence", "collect", "--help"],
        ["dify", "evidence", "explain", "--help"],
    ):
        try:
            main(args)
        except SystemExit as exc:
            assert exc.code == 0
            assert "usage:" in capsys.readouterr().out


def _obs(component, role, layer, status, summary, detail="", signature=None):
    return make_observation(
        component=component,
        component_role=role,
        layer=layer,
        source_type="fixture",
        source_reference=component,
        check_name="golden",
        status=status,
        summary=summary,
        sanitized_detail=detail,
        error_signature=signature,
    )


def test_issue_golden_path_host_loopback_problem():
    evidence = [
        _obs("endpoint", "chat", "host", "pass", "Host TCP connection succeeded."),
        _obs("api", "container_direct", "container", "warn", "Container-direct model probe failed.", "connection refused"),
    ]
    candidates = diagnose_root_causes(evidence)
    assert candidates[0]["candidate"] == "Docker-network or container addressing problem"


def test_issue_golden_path_ssrf_mediated_failure():
    evidence = [
        _obs("endpoint", "chat", "host", "pass", "Host HTTP route returned 200."),
        _obs("api", "container_direct", "container", "pass", "Container-direct model probe succeeded."),
        _obs("ssrf_proxy", "ssrf_proxy", "logs", "warn", "SSRF proxy rejected private address", "blocked address", "ssrf_rejection"),
    ]
    candidates = diagnose_root_causes(evidence)
    assert candidates[0]["candidate"] == "Dify SSRF Proxy or security policy rejection"


def test_issue_golden_path_worker_rag_downstream_symptom():
    evidence = [
        _obs("worker", "worker", "logs", "warn", "Worker indexing queue error", "retrieval empty after indexing failure"),
        _obs("workflow", "rag", "dify_provider", "fail", "Workflow retrieval was empty"),
    ]
    candidates = diagnose_root_causes(evidence)
    assert any(item["candidate"] == "Worker or indexing pipeline problem" for item in candidates)


def test_issue_golden_path_version_drift():
    evidence = [
        _obs("api", "api", "logs", "warn", "migration error after upgrade", "version mismatch between api and worker", "migration_error"),
    ]
    candidates = diagnose_root_causes(evidence)
    assert any(item["candidate"] == "Upgrade or version drift risk" for item in candidates)


def test_issue_golden_path_no_docker_daemon(tmp_path):
    compose = _compose(tmp_path)
    report = run_dify_selfhost_preflight(compose_file=str(compose), runner=FakeRunner(docker_ok=False))
    assert report["status"] == "ATTENTION"
    assert "Docker CLI or daemon is unavailable" in render_reliability_report(report)
