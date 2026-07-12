from __future__ import annotations

import json

from inferdoctor.cli import main
from inferdoctor.core.perf import PerfResult, perf_result_to_dict
from inferdoctor.core.perf_baseline import baseline_from_report, default_baseline_dir, list_baselines


def _sample_report(endpoint: str = "http://user:secret@127.0.0.1:8000/v1?api_key=hidden") -> dict:
    return perf_result_to_dict(
        PerfResult(
            mode="streaming",
            endpoint=endpoint,
            model="local-model",
            reachable=True,
            openai_compatible="yes",
            streaming_supported="confirmed",
            ttft_seconds=0.42,
            total_latency_seconds=1.7,
            generation_duration_seconds=1.2,
            rough_tokens_per_second=18.5,
            output_tokens_exact=22,
            output_token_source="usage",
            tps_quality="exact",
            successful_runs=2,
            failed_runs=0,
            aggregate_metrics={"ttft_median": 0.42, "total_latency_median": 1.7},
            metric_quality={"tokens": "exact", "tps": "exact"},
            user_experience="Responsive for interactive use",
            confidence="medium",
            warnings=["single-machine smoke test"],
        )
    )


def test_baseline_from_report_redacts_endpoint_credentials():
    baseline = baseline_from_report(_sample_report(), name="before", runtime="vllm")

    assert baseline["schema_version"] == "inferdoctor.perf.baseline.v1"
    assert baseline["name"] == "before"
    assert baseline["runtime"] == "vllm"
    assert baseline["endpoint"] == "http://127.0.0.1:8000/v1?api_key=REDACTED"
    assert "secret" not in json.dumps(baseline)
    assert baseline["metrics"]["ttft_seconds"] == 0.42
    assert baseline["readiness_category"] == "Responsive for interactive use"


def test_perf_baseline_cli_create_show_list_and_delete(tmp_path, monkeypatch, capsys):
    baseline_dir = tmp_path / "baselines"
    monkeypatch.setenv("INFERDOCTOR_BASELINE_DIR", str(baseline_dir))
    report_path = tmp_path / "perf.json"
    report_path.write_text(json.dumps(_sample_report()), encoding="utf-8")

    exit_code = main(["perf", "baseline", "create", "--report", str(report_path), "--name", "before", "--runtime", "vllm"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "InferDoctor Performance Baseline" in output
    assert "before" in output
    assert default_baseline_dir() == baseline_dir
    saved = baseline_dir / "before.json"
    assert saved.exists()
    assert "secret" not in saved.read_text(encoding="utf-8")

    exit_code = main(["perf", "baseline", "list"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "before" in output
    assert list_baselines() == [saved]

    exported = tmp_path / "baseline-export.json"
    exit_code = main(["perf", "baseline", "show", "before", "--format", "json", "--output", str(exported)])

    assert exit_code == 0
    exported_data = json.loads(exported.read_text(encoding="utf-8"))
    assert exported_data["schema_version"] == "inferdoctor.perf.baseline.v1"
    assert exported_data["endpoint"] == "http://127.0.0.1:8000/v1?api_key=REDACTED"

    exit_code = main(["perf", "baseline", "delete", "before"])
    err = capsys.readouterr().err

    assert exit_code == 2
    assert "without --yes" in err
    assert saved.exists()

    exit_code = main(["perf", "baseline", "delete", "before", "--yes"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Deleted performance baseline" in output
    assert not saved.exists()


def test_perf_baseline_cli_accepts_explicit_output_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFERDOCTOR_BASELINE_DIR", str(tmp_path / "baselines"))
    report_path = tmp_path / "perf.json"
    report_path.write_text(json.dumps(_sample_report("http://127.0.0.1:8000/v1")), encoding="utf-8")
    output_path = tmp_path / "shared-baseline.json"

    exit_code = main([
        "perf",
        "baseline",
        "create",
        "--report",
        str(report_path),
        "--name",
        "shared",
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    assert output_path.exists()
    assert "shared-baseline.json" in capsys.readouterr().out



def test_baseline_create_rejects_non_report_perf_documents(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFERDOCTOR_BASELINE_DIR", str(tmp_path / "baselines"))
    compare_path = tmp_path / "comparison.json"
    compare_path.write_text(json.dumps({"schema_version": "inferdoctor.perf.compare.v1"}), encoding="utf-8")

    exit_code = main(["perf", "baseline", "create", "--report", str(compare_path), "--name", "bad"])

    assert exit_code == 2
    assert "endpoint or streaming performance report" in capsys.readouterr().err


def test_baseline_show_reports_unsupported_schema_without_traceback(tmp_path, monkeypatch, capsys):
    baseline_dir = tmp_path / "baselines"
    baseline_dir.mkdir()
    monkeypatch.setenv("INFERDOCTOR_BASELINE_DIR", str(baseline_dir))
    (baseline_dir / "old.json").write_text(json.dumps({"schema_version": "inferdoctor.perf.baseline.v999"}), encoding="utf-8")

    exit_code = main(["perf", "baseline", "show", "old"])

    assert exit_code == 2
    assert "Unsupported InferDoctor baseline schema version" in capsys.readouterr().err
