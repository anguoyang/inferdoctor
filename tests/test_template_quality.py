from __future__ import annotations

from pathlib import Path

from inferdoctor.core.templates import create_template_project


CREATABLE_TEMPLATES = {
    "customer-service": [
        "README.md",
        "app.py",
        "requirements.txt",
        ".env.example",
        "config.yaml",
        "prompts/system_prompt.md",
        "data/faq.md",
        "troubleshooting.md",
    ],
    "restaurant-ordering": [
        "README.md",
        "app.py",
        "requirements.txt",
        ".env.example",
        "config.yaml",
        "prompts/system_prompt.md",
        "data/menu.yaml",
        "data/policies.md",
        "examples/sample_orders.md",
        "troubleshooting.md",
    ],
    "local-doc-qa": [
        "README.md",
        "ingest.py",
        "query.py",
        "requirements.txt",
        ".env.example",
        "config.yaml",
        "docs/sample.md",
        "troubleshooting.md",
    ],
}

FORBIDDEN_CONTENT = [
    "sk-",
    "BEGIN OPENSSH PRIVATE KEY",
    "api_key:",
    "password:",
    "/home/",
    "/Users/",
    "NIGHTLY_",
    "CODEX",
    "AI agent",
    "task prompt",
    "transcript",
]


def _all_text(root: Path) -> str:
    chunks = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".example"}:
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def test_generated_templates_have_required_files(tmp_path):
    for template, required_files in CREATABLE_TEMPLATES.items():
        output = tmp_path / template
        create_template_project(template, str(output))

        for relative in required_files:
            assert (output / relative).exists(), f"{template} missing {relative}"


def test_generated_template_readmes_explain_endpoint_and_validation(tmp_path):
    for template in CREATABLE_TEMPLATES:
        output = tmp_path / template
        create_template_project(template, str(output))
        readme = (output / "README.md").read_text(encoding="utf-8")

        assert "LOCAL_AI_BASE_URL" in readme
        assert "OpenAI-compatible" in readme
        assert "inferdoctor template validate ." in readme
        assert "inferdoctor template smoke-test ." in readme
        assert "inferdoctor perf baseline create" in readme
        assert "inferdoctor perf compare before.json after.json" in readme
        assert "inferdoctor optimize plan --report before.json" in readme
        assert "--allow-non-local" in readme
        assert "Expected File Tree" in readme


def test_generated_templates_include_safe_smoke_commands(tmp_path):
    chat_templates = ["customer-service", "restaurant-ordering"]
    for template in chat_templates:
        output = tmp_path / template
        create_template_project(template, str(output))
        app = (output / "app.py").read_text(encoding="utf-8")
        readme = (output / "README.md").read_text(encoding="utf-8")

        assert "--dry-run" in app
        assert "--check-config" in app
        assert "python app.py --dry-run" in readme
        assert "No endpoint call was made" in app

    output = tmp_path / "local-doc-qa"
    create_template_project("local-doc-qa", str(output))
    query = (output / "query.py").read_text(encoding="utf-8")
    readme = (output / "README.md").read_text(encoding="utf-8")
    assert "--dry-run" in query
    assert "--check-config" in query
    assert "python query.py --dry-run" in readme


def test_generated_templates_do_not_contain_obvious_secrets_or_private_paths(tmp_path):
    for template in CREATABLE_TEMPLATES:
        output = tmp_path / template
        create_template_project(template, str(output))
        text = _all_text(output)

        for forbidden in FORBIDDEN_CONTENT:
            assert forbidden not in text


def test_generated_templates_are_streaming_first_by_default(tmp_path):
    for template in CREATABLE_TEMPLATES:
        output = tmp_path / template
        create_template_project(template, str(output))
        config = (output / "config.yaml").read_text(encoding="utf-8")
        env_example = (output / ".env.example").read_text(encoding="utf-8")
        readme = (output / "README.md").read_text(encoding="utf-8")

        assert "streaming: true" in config
        assert "LOCAL_AI_STREAMING=true" in env_example
        assert "TTFT" in readme

    doc_config = (tmp_path / "local-doc-qa" / "config.yaml").read_text(encoding="utf-8")
    assert "top_k: 4" in doc_config
    assert "context_budget" in doc_config


def test_generated_templates_expose_live_endpoint_controls_explicitly(tmp_path):
    for template in CREATABLE_TEMPLATES:
        output = tmp_path / template
        create_template_project(template, str(output))
        source_name = "app.py" if (output / "app.py").exists() else "query.py"
        source = (output / source_name).read_text(encoding="utf-8")
        readme = (output / "README.md").read_text(encoding="utf-8")

        assert "--check-endpoint" in source
        assert "--warmup" in source
        assert "--check-endpoint" in readme
        assert "live" in readme

    query = (tmp_path / "local-doc-qa" / "query.py").read_text(encoding="utf-8")
    assert "--generate" in query
    assert "Retrieval complete" in query
