from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import List

from inferdoctor.i18n import t


@dataclass(frozen=True)
class TemplateValidationItem:
    name: str
    status: str
    summary: str
    fix: str = ""


@dataclass(frozen=True)
class TemplateValidationReport:
    path: Path
    template_type: str
    items: List[TemplateValidationItem]

    @property
    def status(self) -> str:
        if any(item.status == "FAIL" for item in self.items):
            return "FAIL"
        if any(item.status == "WARN" for item in self.items):
            return "WARN"
        return "PASS"

    @property
    def readiness_score(self) -> int:
        return _readiness_score([item.status for item in self.items])


@dataclass(frozen=True)
class TemplateSmokeItem:
    name: str
    status: str
    command: List[str]
    summary: str
    output: str = ""
    fix: str = ""


@dataclass(frozen=True)
class TemplateSmokeReport:
    path: Path
    template_type: str
    items: List[TemplateSmokeItem]

    @property
    def status(self) -> str:
        if any(item.status == "FAIL" for item in self.items):
            return "FAIL"
        if any(item.status == "WARN" for item in self.items):
            return "WARN"
        return "PASS"

    @property
    def readiness_score(self) -> int:
        return _readiness_score([item.status for item in self.items])


def _readiness_score(statuses: List[str]) -> int:
    if not statuses:
        return 0
    weights = {"PASS": 100, "WARN": 65, "FAIL": 0}
    return round(sum(weights.get(status, 50) for status in statuses) / len(statuses))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _detect_template_type(root: Path) -> str:
    readme = _read_text(root / "README.md").lower()
    config = _read_text(root / "config.yaml").lower()
    combined = readme + "\n" + config
    if (root / "data" / "faq.md").exists() or "customer service" in combined:
        return "customer-service"
    if (root / "data" / "menu.yaml").exists() or "restaurant ordering" in combined:
        return "restaurant-ordering"
    if ((root / "query.py").exists() and (root / "docs").exists()) or "local document q&a" in combined or "retrieval: keyword" in config:
        return "local-doc-qa"
    return "unknown"


def _has_entrypoint(root: Path) -> bool:
    return any((root / relative).exists() for relative in ("app.py", "query.py", "ingest.py"))


def _has_sample_data(root: Path) -> bool:
    return any(
        (root / relative).exists()
        for relative in (
            "data/faq.md",
            "data/menu.yaml",
            "data/policies.md",
            "docs/sample.md",
        )
    )


def _contains_endpoint_config(root: Path) -> bool:
    config = _read_text(root / "config.yaml")
    env_example = _read_text(root / ".env.example")
    return any(
        marker in (config + "\n" + env_example)
        for marker in ("LOCAL_AI_BASE_URL", "endpoint:", "/v1")
    )


SECRET_VALUE_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?([a-z0-9_./+\-]{12,})"
)
BEARER_RE = re.compile(r"(?i)bearer\s+[a-z0-9_./+\-]{12,}")


def _secret_hits(root: Path) -> list[str]:
    suspicious_names = (".env", "config.yaml", ".env.example")
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "__pycache__":
            continue
        if path.suffix not in {".py", ".md", ".yaml", ".yml", ".txt", ".example"} and path.name not in suspicious_names:
            continue
        text = _read_text(path)
        if SECRET_VALUE_RE.search(text) or BEARER_RE.search(text):
            hits.append(str(path.relative_to(root)))
    return sorted(set(hits))


def validate_template_project(path: str) -> TemplateValidationReport:
    root = Path(path).expanduser()
    items: list[TemplateValidationItem] = []
    if not root.exists():
        return TemplateValidationReport(
            path=root,
            template_type="missing",
            items=[TemplateValidationItem("project directory", "FAIL", "Directory does not exist.", "Create a template project first.")],
        )
    if not root.is_dir():
        return TemplateValidationReport(
            path=root,
            template_type="invalid",
            items=[TemplateValidationItem("project directory", "FAIL", "Path is not a directory.", "Pass the generated template directory.")],
        )

    template_type = _detect_template_type(root)
    required_common = ["README.md", "requirements.txt", "config.yaml", ".env.example"]
    for relative in required_common:
        exists = (root / relative).exists()
        items.append(
            TemplateValidationItem(
                relative,
                "PASS" if exists else "FAIL",
                "Found {0}.".format(relative) if exists else "Missing {0}.".format(relative),
                "Regenerate the template or add {0}.".format(relative),
            )
        )

    if _contains_endpoint_config(root):
        items.append(TemplateValidationItem("endpoint config", "PASS", "Local endpoint configuration is present."))
    else:
        items.append(TemplateValidationItem("endpoint config", "FAIL", "No local endpoint configuration found.", "Add LOCAL_AI_BASE_URL to .env or endpoint: to config.yaml."))

    if template_type in {"customer-service", "restaurant-ordering"}:
        has_app = (root / "app.py").exists()
        items.append(TemplateValidationItem("app.py", "PASS" if has_app else "FAIL", "Found CLI app." if has_app else "Missing app.py.", "Regenerate the template or restore app.py."))
        has_prompt = (root / "prompts" / "system_prompt.md").exists()
        items.append(TemplateValidationItem("system prompt", "PASS" if has_prompt else "WARN", "Found system prompt." if has_prompt else "No system prompt file found.", "Add prompts/system_prompt.md for clearer assistant behavior."))
    elif template_type == "local-doc-qa":
        for relative in ("ingest.py", "query.py", "docs/sample.md"):
            exists = (root / relative).exists()
            items.append(TemplateValidationItem(relative, "PASS" if exists else "FAIL", "Found {0}.".format(relative) if exists else "Missing {0}.".format(relative), "Regenerate the local-doc-qa template."))
    else:
        items.append(TemplateValidationItem("template type", "WARN", "Template type was not recognized.", "Use a project generated by inferdoctor template create."))
        has_entrypoint = _has_entrypoint(root)
        items.append(TemplateValidationItem("app entrypoint", "PASS" if has_entrypoint else "FAIL", "Found an app/query entrypoint." if has_entrypoint else "No app.py, query.py, or ingest.py found.", "Add an app entrypoint or regenerate a supported template."))
        has_sample_data = _has_sample_data(root)
        items.append(TemplateValidationItem("sample data", "PASS" if has_sample_data else "WARN", "Found sample data." if has_sample_data else "No sample data files found.", "Add sample data under data/ or docs/ so users can test the app."))

    if template_type == "customer-service":
        exists = (root / "data" / "faq.md").exists()
        items.append(TemplateValidationItem("sample data", "PASS" if exists else "FAIL", "Found data/faq.md." if exists else "Missing data/faq.md.", "Add FAQ content under data/faq.md."))
    if template_type == "restaurant-ordering":
        exists = (root / "data" / "menu.yaml").exists() and (root / "data" / "policies.md").exists()
        items.append(TemplateValidationItem("sample data", "PASS" if exists else "FAIL", "Found menu and policy data." if exists else "Missing menu or policy data.", "Add data/menu.yaml and data/policies.md."))

    secret_hits = _secret_hits(root)
    if secret_hits:
        items.append(TemplateValidationItem("secret scan", "WARN", "Suspicious secret-like text found in: {0}".format(", ".join(secret_hits[:5])), "Remove real keys/tokens before sharing or committing."))
    else:
        items.append(TemplateValidationItem("secret scan", "PASS", "No obvious secrets found in text files."))

    return TemplateValidationReport(path=root, template_type=template_type, items=items)


def _smoke_commands(template_type: str) -> list[tuple[str, list[str]]]:
    python = sys.executable
    if template_type in {"customer-service", "restaurant-ordering"}:
        return [
            ("app help", [python, "app.py", "--help"]),
            ("app dry-run", [python, "app.py", "--dry-run"]),
            ("app config", [python, "app.py", "--check-config"]),
        ]
    if template_type == "local-doc-qa":
        return [
            ("ingest help", [python, "ingest.py", "--help"]),
            ("query help", [python, "query.py", "--help"]),
            ("query dry-run", [python, "query.py", "--dry-run"]),
            ("query config", [python, "query.py", "--check-config"]),
        ]
    return []


def _command_display(command: list[str]) -> str:
    parts = ["python" if index == 0 else part for index, part in enumerate(command)]
    return " ".join(parts)


def smoke_test_template_project(path: str, timeout: float = 5.0) -> TemplateSmokeReport:
    root = Path(path).expanduser()
    if not root.exists():
        return TemplateSmokeReport(
            path=root,
            template_type="missing",
            items=[TemplateSmokeItem("project directory", "FAIL", [], "Directory does not exist.", fix="Create a template project first.")],
        )
    if not root.is_dir():
        return TemplateSmokeReport(
            path=root,
            template_type="invalid",
            items=[TemplateSmokeItem("project directory", "FAIL", [], "Path is not a directory.", fix="Pass the generated template directory.")],
        )
    template_type = _detect_template_type(root)
    commands = _smoke_commands(template_type)
    if not commands:
        return TemplateSmokeReport(
            path=root,
            template_type=template_type,
            items=[TemplateSmokeItem("template type", "FAIL", [], "Template type is not supported for smoke tests.", fix="Use customer-service, restaurant-ordering, or local-doc-qa.")],
        )

    items: list[TemplateSmokeItem] = []
    for name, command in commands:
        entrypoint = root / command[1]
        if not entrypoint.exists():
            items.append(
                TemplateSmokeItem(
                    name=name,
                    status="FAIL",
                    command=command,
                    summary="Missing {0}.".format(command[1]),
                    fix="Regenerate the template or restore {0}.".format(command[1]),
                )
            )
            continue
        try:
            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            items.append(
                TemplateSmokeItem(
                    name=name,
                    status="FAIL",
                    command=command,
                    summary="Command timed out after {0:g}s.".format(timeout),
                    fix="Check for blocking input or long-running endpoint calls.",
                )
            )
            continue
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if completed.returncode == 0:
            items.append(
                TemplateSmokeItem(
                    name=name,
                    status="PASS",
                    command=command,
                    summary="Command completed without contacting an endpoint.",
                    output=output[:600],
                )
            )
        else:
            items.append(
                TemplateSmokeItem(
                    name=name,
                    status="FAIL",
                    command=command,
                    summary="Command exited with code {0}.".format(completed.returncode),
                    output=output[:600],
                    fix="Run the command manually from the template directory and inspect the error.",
                )
            )
    return TemplateSmokeReport(path=root, template_type=template_type, items=items)


def render_template_smoke_test(report: TemplateSmokeReport, language: str = "auto") -> str:
    lines = [
        t("template_validation.smoke_title", language),
        "=" * 57,
        "Path: {0}".format(report.path),
        "Detected template: {0}".format(report.template_type),
        "Overall status: {0}".format(report.status),
        "Project Readiness: {0} / 100 (heuristic)".format(report.readiness_score),
        "",
        t("template_validation.checks", language),
    ]
    for item in report.items:
        command = _command_display(item.command) if item.command else "n/a"
        lines.append("  {0:<16} {1:<4} {2}".format(item.name, item.status, item.summary))
        lines.append("    Command: {0}".format(command))
    fixes = [item for item in report.items if item.status in {"WARN", "FAIL"} and item.fix]
    lines.extend(["", t("template_validation.top_fixes", language)])
    if fixes:
        for index, item in enumerate(fixes[:3], start=1):
            lines.append("  {0}. {1}: {2}".format(index, item.name, item.fix))
    else:
        lines.append("  1. No smoke-test fixes needed. Configure .env when you are ready to use a live endpoint.")
    lines.extend([
        "",
        "Next steps:",
        "  1. Run inferdoctor template validate {0}".format(report.path),
        "  2. Edit .env or config.yaml for your local endpoint.",
        "  3. Run the generated app when your endpoint is ready.",
        "",
        "Smoke tests are read-only. They do not install dependencies, start services, call endpoints, or run model inference.",
    ])
    return "\n".join(lines)


def render_template_validation(report: TemplateValidationReport, language: str = "auto") -> str:
    lines = [
        t("template_validation.title", language),
        "=" * 57,
        "Path: {0}".format(report.path),
        "Detected template: {0}".format(report.template_type),
        "Overall status: {0}".format(report.status),
        "Project Readiness: {0} / 100 (heuristic)".format(report.readiness_score),
        "",
        t("template_validation.checks", language),
    ]
    for item in report.items:
        lines.append("  {0:<18} {1:<4} {2}".format(item.name, item.status, item.summary))
    fixes = [item for item in report.items if item.status in {"WARN", "FAIL"} and item.fix]
    lines.extend(["", t("template_validation.top_fixes", language)])
    if fixes:
        for index, item in enumerate(fixes[:3], start=1):
            lines.append("  {0}. {1}: {2}".format(index, item.name, item.fix))
    else:
        lines.append("  1. No fixes needed. Configure .env if you want to use a different local endpoint.")
    if report.template_type in {"customer-service", "restaurant-ordering"}:
        next_commands = [
            "python app.py --dry-run",
            "python app.py --check-config",
            "inferdoctor template smoke-test {0}".format(report.path),
        ]
    elif report.template_type == "local-doc-qa":
        next_commands = [
            "python ingest.py --help",
            "python query.py --dry-run",
            "python query.py --check-config",
            "inferdoctor template smoke-test {0}".format(report.path),
        ]
    else:
        next_commands = ["inferdoctor template list"]
    lines.extend([
        "",
        t("template_validation.next_command", language),
        "  cd {0}".format(report.path),
    ])
    lines.extend("  {0}".format(command) for command in next_commands)
    lines.extend([
        "",
        "Supported layouts: customer-service, restaurant-ordering, local-doc-qa.",
        "This validation is read-only. It does not install dependencies, call endpoints, or run inference.",
    ])
    return "\n".join(lines)
