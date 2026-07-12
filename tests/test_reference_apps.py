from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUSTOMER_SERVICE = ROOT / "examples" / "reference_apps" / "customer_service"
LOCAL_DOC_QA = ROOT / "examples" / "reference_apps" / "local_doc_qa"


def _run_script(app_dir: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=app_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )


def _run(app_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run_script(app_dir, "app.py", *args)


def test_customer_service_reference_app_safe_modes():
    help_result = _run(CUSTOMER_SERVICE, "--help")
    assert help_result.returncode == 0
    assert "--dry-run" in help_result.stdout

    config_result = _run(CUSTOMER_SERVICE, "--check-config")
    assert config_result.returncode == 0
    assert "Endpoint:" in config_result.stdout
    assert "No endpoint call was made." in config_result.stdout

    dry_result = _run(CUSTOMER_SERVICE, "--dry-run")
    assert dry_result.returncode == 0
    assert "Dry run" in dry_result.stdout
    assert "No endpoint call was made." in dry_result.stdout


def test_customer_service_reference_app_docs_include_performance_loop():
    readme = (CUSTOMER_SERVICE / "README.md").read_text(encoding="utf-8")
    config = (CUSTOMER_SERVICE / "config.yaml").read_text(encoding="utf-8")
    env_example = (CUSTOMER_SERVICE / ".env.example").read_text(encoding="utf-8")

    assert "inferdoctor perf baseline create" in readme
    assert "inferdoctor perf compare before.json after.json" in readme
    assert "inferdoctor optimize plan" in readme
    assert "--allow-non-local" in readme
    assert "streaming: true" in config
    assert "LOCAL_AI_STREAMING=true" in env_example
    assert (CUSTOMER_SERVICE / "instructions.md").exists()



def test_local_doc_qa_reference_app_safe_modes():
    ingest_help = _run_script(LOCAL_DOC_QA, "ingest.py", "--help")
    assert ingest_help.returncode == 0
    assert "--dry-run" in ingest_help.stdout

    ingest_dry = _run_script(LOCAL_DOC_QA, "ingest.py", "--dry-run")
    assert ingest_dry.returncode == 0
    assert "no endpoint call was made" in ingest_dry.stdout.lower()

    query_help = _run_script(LOCAL_DOC_QA, "query.py", "--help")
    assert query_help.returncode == 0
    assert "--check-config" in query_help.stdout

    query_config = _run_script(LOCAL_DOC_QA, "query.py", "--check-config")
    assert query_config.returncode == 0
    assert "No endpoint call was made" in query_config.stdout

    query_dry = _run_script(LOCAL_DOC_QA, "query.py", "What is InferDoctor?", "--dry-run")
    assert query_dry.returncode == 0
    assert "Retrieval complete" in query_dry.stdout
    assert "no endpoint call was made" in query_dry.stdout.lower()


def test_local_doc_qa_reference_app_docs_include_performance_loop():
    readme = (LOCAL_DOC_QA / "README.md").read_text(encoding="utf-8")
    config = (LOCAL_DOC_QA / "config.yaml").read_text(encoding="utf-8")

    assert "inferdoctor optimize rag" in readme
    assert "inferdoctor perf baseline create" in readme
    assert "inferdoctor perf compare before.json after.json" in readme
    assert "--allow-non-local" in readme
    assert "top_k: 4" in config
    assert "context_budget" in config
