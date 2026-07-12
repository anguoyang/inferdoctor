from __future__ import annotations

import json
import os
import re
import secrets
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from inferdoctor import __version__
from inferdoctor.core.endpoint_safety import classify_endpoint, redact_endpoint

DIFY_KIT_SCHEMA_VERSION = "inferdoctor.dify.kit.v1"
DIFY_SMOKE_SCHEMA_VERSION = "inferdoctor.dify.smoke.v1"
DIFY_PERF_SCHEMA_VERSION = "inferdoctor.dify.performance.v1"
DIFY_OPTIMIZE_SCHEMA_VERSION = "inferdoctor.dify.optimize.v1"
DIFY_KNOWLEDGE_SCHEMA_VERSION = "inferdoctor.dify.knowledge.v1"

DEFAULT_APP_BASE_URL = "http://127.0.0.1:5001/v1"
DEFAULT_APP_KEY_ENV = "DIFY_APP_API_KEY"
DEFAULT_KNOWLEDGE_KEY_ENV = "DIFY_KNOWLEDGE_API_KEY"
DEFAULT_DATASET_ENV = "DIFY_DATASET_ID"
CHAT_MODES = {"chat", "advanced-chat", "agent-chat", "agent"}
WORKFLOW_MODES = {"workflow"}
KNOWN_APP_MODES = CHAT_MODES | WORKFLOW_MODES | {"completion"}
SUPPORTED_LIVE_MODES = CHAT_MODES | WORKFLOW_MODES
KIT_ALIASES = {"local-rag": "local-private-rag", "private-rag": "local-private-rag"}


class DifyError(ValueError):
    pass


class DifyAPIError(DifyError):
    def __init__(self, message: str, *, status_code: Optional[int] = None, category: str = "api_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.category = category


@dataclass(frozen=True)
class DifyConfig:
    app_base_url: str
    app_key_env: str = DEFAULT_APP_KEY_ENV
    app_api_key: Optional[str] = None
    knowledge_base_url: Optional[str] = None
    knowledge_key_env: str = DEFAULT_KNOWLEDGE_KEY_ENV
    knowledge_api_key: Optional[str] = None
    dataset_id: Optional[str] = None
    timeout: float = 30.0
    allow_non_local: bool = False
    allow_public: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generated_user_id() -> str:
    return "inferdoctor-smoke-{0}".format(secrets.token_hex(4))


def canonical_kit_name(name: str) -> str:
    normalized = name.strip().lower()
    return KIT_ALIASES.get(normalized, normalized)


def available_dify_template_names() -> List[str]:
    return ["local-private-rag"]


def load_dify_config(
    *,
    base_url: Optional[str] = None,
    app_key_env: Optional[str] = None,
    knowledge_base_url: Optional[str] = None,
    knowledge_key_env: Optional[str] = None,
    dataset_id: Optional[str] = None,
    timeout: float = 30.0,
    allow_non_local: bool = False,
    allow_public: bool = False,
) -> DifyConfig:
    app_env = app_key_env or DEFAULT_APP_KEY_ENV
    knowledge_env = knowledge_key_env or DEFAULT_KNOWLEDGE_KEY_ENV
    resolved_base = base_url or os.environ.get("DIFY_API_BASE_URL") or DEFAULT_APP_BASE_URL
    resolved_knowledge_base = knowledge_base_url or os.environ.get("DIFY_KNOWLEDGE_API_BASE_URL") or resolved_base
    return DifyConfig(
        app_base_url=resolved_base,
        app_key_env=app_env,
        app_api_key=os.environ.get(app_env),
        knowledge_base_url=resolved_knowledge_base,
        knowledge_key_env=knowledge_env,
        knowledge_api_key=os.environ.get(knowledge_env),
        dataset_id=dataset_id or os.environ.get(DEFAULT_DATASET_ENV),
        timeout=timeout,
        allow_non_local=allow_non_local,
        allow_public=allow_public,
    )


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def redact_secret_text(text: str, secrets_to_hide: Sequence[Optional[str]] = ()) -> str:
    redacted = text
    for secret in secrets_to_hide:
        if secret:
            redacted = redacted.replace(secret, "REDACTED")
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer REDACTED", redacted, flags=re.IGNORECASE)
    redacted = re.sub(
        r"(api[_-]?key|token|secret|password)(\s*[=:]\s*)[^\s,'\"]+",
        r"\1\2REDACTED",
        redacted,
        flags=re.IGNORECASE,
    )
    return redacted


def sanitize_endpoint(endpoint: str) -> str:
    return redact_endpoint(endpoint)


def ensure_endpoint_allowed(base_url: str, *, allow_non_local: bool = False, allow_public: bool = False) -> Dict[str, Any]:
    safety = classify_endpoint(base_url)
    allowed = True
    reason = "allowed"
    if safety.category == "invalid":
        allowed = False
        reason = "invalid_url"
    elif safety.category == "private" and not allow_non_local:
        allowed = False
        reason = "private_endpoint_requires_allow_non_local"
    elif safety.category == "public" and not allow_public:
        allowed = False
        reason = "public_endpoint_requires_allow_public"
    return {
        "endpoint": safety.sanitized_endpoint,
        "category": safety.category,
        "host": safety.host,
        "warnings": safety.warnings,
        "allowed": allowed,
        "reason": reason,
    }


class _NoCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        old_host = urllib.parse.urlsplit(req.full_url).netloc.lower()
        new_host = urllib.parse.urlsplit(newurl).netloc.lower()
        if old_host != new_host and req.headers.get("Authorization"):
            raise DifyAPIError("Refusing to follow redirect with Authorization to a different host", category="unsafe_redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class DifyAPIClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoCrossHostRedirect)

    def _url(self, path: str) -> str:
        return self.base_url + "/" + path.lstrip("/")

    def _headers(self, api_key: Optional[str] = None) -> Dict[str, str]:
        key = api_key if api_key is not None else self.api_key
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "InferDoctor/{0}".format(__version__),
        }
        if key:
            headers["Authorization"] = "Bearer {0}".format(key)
        return headers

    def request_json(self, method: str, path: str, *, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self._url(path), data=payload, method=method.upper(), headers=self._headers())
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                content = response.read(1024 * 1024 + 1)
                if len(content) > 1024 * 1024:
                    raise DifyAPIError("Dify response exceeded the 1 MiB safety limit", category="response_too_large")
                if not content:
                    return {}
                parsed = json.loads(content.decode("utf-8"))
                return parsed if isinstance(parsed, dict) else {"value": parsed}
        except urllib.error.HTTPError as exc:
            raise DifyAPIError(_http_error_message(exc), status_code=exc.code, category=_status_category(exc.code)) from exc
        except urllib.error.URLError as exc:
            raise DifyAPIError("Could not reach Dify endpoint: {0}".format(redact_secret_text(str(exc.reason), [self.api_key])), category="network_error") from exc
        except (TimeoutError, json.JSONDecodeError) as exc:
            raise DifyAPIError("Dify returned no usable JSON response", category="invalid_json") from exc

    def get_info(self) -> Dict[str, Any]:
        return self.request_json("GET", "/info")

    def stream_request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(self._url(path), data=json.dumps(payload).encode("utf-8"), method="POST", headers=self._headers())
        started = time.perf_counter()
        lines: List[bytes] = []
        content_type = ""
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                total = 0
                while True:
                    line = response.readline(65536)
                    if not line:
                        break
                    total += len(line)
                    if total > 1024 * 1024:
                        raise DifyAPIError("Dify streaming response exceeded the 1 MiB safety limit", category="response_too_large")
                    lines.append(line)
                ended = time.perf_counter()
        except urllib.error.HTTPError as exc:
            raise DifyAPIError(_http_error_message(exc), status_code=exc.code, category=_status_category(exc.code)) from exc
        except urllib.error.URLError as exc:
            raise DifyAPIError("Could not reach Dify endpoint: {0}".format(redact_secret_text(str(exc.reason), [self.api_key])), category="network_error") from exc
        events = parse_sse_lines(lines)
        metrics = interpret_dify_events(events, started_at=started, ended_at=ended)
        metrics["content_type"] = content_type
        return metrics

    def run_chat_stream(self, query: str, *, user: str, show_answer: bool = False) -> Dict[str, Any]:
        result = self.stream_request("/chat-messages", {"query": query, "inputs": {}, "response_mode": "streaming", "user": user})
        if not show_answer:
            result["answer_preview"] = None
            result["answer_retained"] = False
        return result

    def run_workflow_stream(self, query: str, *, user: str, show_answer: bool = False) -> Dict[str, Any]:
        result = self.stream_request("/workflows/run", {"inputs": {"query": query}, "response_mode": "streaming", "user": user})
        if not show_answer:
            result["answer_preview"] = None
            result["answer_retained"] = False
        return result

    def retrieve_chunks(self, dataset_id: str, query: str) -> Dict[str, Any]:
        body = {"query": query, "retrieval_model": {"search_method": "semantic_search", "top_k": 3}}
        return self.request_json("POST", "/datasets/{0}/retrieve".format(urllib.parse.quote(dataset_id, safe="")), body=body)


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    body = exc.read(4096) if hasattr(exc, "read") else b""
    detail = ""
    if body:
        try:
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict):
                detail = str(parsed.get("message") or parsed.get("error") or parsed.get("code") or "")
        except Exception:
            detail = ""
    return "Dify returned HTTP {0}{1}".format(exc.code, ": " + redact_secret_text(detail) if detail else "")


def _status_category(status_code: int) -> str:
    if status_code == 401:
        return "unauthorized"
    if status_code == 403:
        return "forbidden"
    if status_code == 404:
        return "not_found"
    if status_code >= 500:
        return "server_error"
    return "http_error"


def parse_sse_lines(lines: Iterable[bytes | str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    event_type: Optional[str] = None
    data_lines: List[str] = []

    def flush() -> None:
        nonlocal event_type, data_lines
        if not data_lines and event_type is None:
            return
        data_text = "\n".join(data_lines)
        parsed: Optional[Any] = None
        error: Optional[str] = None
        if data_text:
            try:
                parsed = json.loads(data_text)
            except json.JSONDecodeError:
                error = "malformed_json"
        events.append({"event": event_type, "data": data_text, "json": parsed, "error": error})
        event_type = None
        data_lines = []

    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line == "":
            flush()
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_type = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].lstrip()
            if data == "[DONE]":
                events.append({"event": event_type or "done", "data": data, "json": None, "error": None})
                event_type = None
                data_lines = []
            else:
                data_lines.append(data)
    flush()
    return events


def _visible_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("answer", "text", "delta", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    data = payload.get("data")
    return _visible_text(data) if isinstance(data, dict) else ""


def interpret_dify_events(events: Sequence[Dict[str, Any]], *, started_at: float = 0.0, ended_at: Optional[float] = None) -> Dict[str, Any]:
    first_event: Optional[int] = None
    first_text: Optional[int] = None
    node_events = 0
    workflow_events = 0
    answer: List[str] = []
    errors: List[str] = []
    unknown: List[str] = []
    known = {None, "message", "agent_message", "message_end", "workflow_started", "node_started", "node_finished", "workflow_finished", "error", "ping", "done"}
    status = "incomplete"
    for index, event in enumerate(events):
        event_type = event.get("event")
        if first_event is None and event_type != "ping":
            first_event = index
        if event.get("error"):
            errors.append("malformed streaming event")
        if event_type not in known and str(event_type) not in unknown:
            unknown.append(str(event_type))
        if event_type in {"node_started", "node_finished"}:
            node_events += 1
        if event_type in {"workflow_started", "workflow_finished"}:
            workflow_events += 1
        if event_type == "error":
            errors.append("Dify stream returned an error event")
            status = "error"
        if event_type in {"message_end", "workflow_finished", "done"} and status != "error":
            status = "completed"
        text = _visible_text(event.get("json"))
        if text:
            answer.append(text)
            if first_text is None:
                first_text = index
    total_latency = None if ended_at is None or not started_at else round(max(0.0, ended_at - started_at), 6)
    return {
        "event_count": len(events),
        "first_event_index": first_event,
        "first_visible_text_index": first_text,
        "first_event_latency_seconds": None,
        "ttft_seconds": None,
        "total_latency_seconds": total_latency,
        "node_event_count": node_events,
        "workflow_event_count": workflow_events,
        "unknown_event_types": unknown,
        "errors": errors,
        "completion_status": status,
        "answer_preview": "".join(answer)[:160] if answer else None,
        "answer_retained": bool(answer),
        "event_quality": "ok" if not errors else "partial",
    }


def _kit_files() -> Dict[str, str]:
    return {
        "manifest.yaml": """schema_version: inferdoctor.dify.kit.v1
name: local-private-rag
title: Local / Private RAG Starter Kit for Dify
description: Starter kit for a Dify Chatflow-style RAG app connected to local, LAN, private, or self-hosted model endpoints.
intended_app_mode: advanced-chat
validation_level: static_compatibility_validated
required_environment_variables:
  - DIFY_API_BASE_URL
  - DIFY_APP_API_KEY
  - DIFY_KNOWLEDGE_API_BASE_URL
  - DIFY_KNOWLEDGE_API_KEY
  - DIFY_DATASET_ID
unresolved_placeholders:
  - MODEL_PROVIDER_PLACEHOLDER
  - MODEL_NAME_PLACEHOLDER
  - KNOWLEDGE_DATASET_ID_PLACEHOLDER
compatibility_notes:
  - Static DSL validation is provided by InferDoctor.
  - Manual Dify import must be verified in the target workspace.
""",
        "dify_app.yaml": """app:
  mode: advanced-chat
  name: Local Private RAG Starter
  description: A small RAG flow for local or private model endpoints.
workflow:
  graph:
    nodes:
      - id: start
        type: start
        title: User question
      - id: retrieve_knowledge
        type: knowledge-retrieval
        title: Retrieve from selected knowledge base
        dataset_id: KNOWLEDGE_DATASET_ID_PLACEHOLDER
        top_k: 4
      - id: llm_answer
        type: llm
        title: Generate grounded answer
        provider: MODEL_PROVIDER_PLACEHOLDER
        model: MODEL_NAME_PLACEHOLDER
        streaming: true
        context_budget_tokens: 2500
      - id: answer
        type: answer
        title: Stream final answer
    edges:
      - source: start
        target: retrieve_knowledge
      - source: retrieve_knowledge
        target: llm_answer
      - source: llm_answer
        target: answer
notes:
  import_status: manual_import_required
  live_import_verified: false
""",
        "README.md": """# Local / Private RAG Starter Kit for Dify

This kit helps you build a Dify RAG app connected to a local, LAN, private, or self-hosted model endpoint.

It does not install Dify, import the DSL automatically, create a knowledge base, upload documents, or store API keys.

## Files

- `manifest.yaml`: kit metadata and safety boundaries
- `dify_app.yaml`: starter Chatflow-style DSL draft for manual import
- `.env.example`: environment variable names only
- `preflight.yaml`: checks to run before live smoke tests
- `smoke_cases.yaml`: harmless smoke-test prompts
- `experience_profile.yaml`: performance UX priorities
- `performance_guidance.yaml`: baseline and comparison commands
- `optimization_notes.md`: Dify-specific optimization notes
- `sample_docs/return_policy.md`: fictional sample document

## Manual Dify steps

1. Review `dify_app.yaml` before import.
2. Import or recreate the flow manually in your Dify workspace.
3. Configure the model provider and model name.
4. Select the knowledge base you want to use.
5. Publish the app.
6. Create an app API key in Dify.
7. Configure environment variables in your shell, not in this repository.

## InferDoctor workflow

```bash
inferdoctor dify validate ./dify-local-private-rag
inferdoctor dify smoke --kit ./dify-local-private-rag --dry-run
export DIFY_API_BASE_URL=http://127.0.0.1:5001/v1
export DIFY_APP_API_KEY=your-app-key
inferdoctor dify check --base-url "$DIFY_API_BASE_URL"
inferdoctor dify smoke --base-url "$DIFY_API_BASE_URL"
inferdoctor dify perf --base-url "$DIFY_API_BASE_URL" --runs 2 --warmup 1 --output dify-perf.json --format json
inferdoctor perf baseline create --report dify-perf.json --name dify-before
inferdoctor dify optimize --report dify-perf.json --kit ./dify-local-private-rag
```

Use `--allow-non-local` only for private endpoints you control. Use `--allow-public` only when you intentionally test Dify Cloud or another public endpoint with a harmless query.

Performance checks are bounded smoke tests, not formal benchmarks.
""",
        ".env.example": """# Use environment variables. Do not commit real keys.
DIFY_API_BASE_URL=http://127.0.0.1:5001/v1
DIFY_APP_API_KEY=replace-with-your-app-api-key
DIFY_KNOWLEDGE_API_BASE_URL=http://127.0.0.1:5001/v1
DIFY_KNOWLEDGE_API_KEY=replace-with-your-knowledge-api-key
DIFY_DATASET_ID=replace-with-your-dataset-id
""",
        "preflight.yaml": """checks:
  - confirm_app_is_published
  - confirm_app_api_key_is_available_in_environment
  - confirm_model_provider_is_configured_in_dify
  - confirm_knowledge_base_is_selected
  - run_inferdoctor_dify_validate
  - run_inferdoctor_dify_smoke_dry_run
""",
        "smoke_cases.yaml": """cases:
  - id: fictional_return_policy
    query: What is the fictional return policy in the sample document?
    sensitive: false
""",
        "experience_profile.yaml": """profile: rag
priorities:
  ttft: high
  streaming: high
  retrieval_latency: high
  total_latency: medium
""",
        "performance_guidance.yaml": """commands:
  live_smoke: inferdoctor dify smoke --base-url $DIFY_API_BASE_URL
  performance_report: inferdoctor dify perf --base-url $DIFY_API_BASE_URL --runs 2 --warmup 1 --format json --output dify-perf.json
  baseline: inferdoctor perf baseline create --report dify-perf.json --name dify-before
  compare: inferdoctor perf compare before.json after.json
  optimize: inferdoctor dify optimize --report dify-perf.json --kit ./dify-local-private-rag
limitations:
  - bounded smoke tests only
  - no concurrency testing
  - no model quality evaluation
""",
        "optimization_notes.md": """# Optimization Notes

- Show retrieval progress before generation.
- Keep `top_k` small enough for the app goal.
- Avoid rerank on the critical path unless quality requires it.
- Keep context within a clear budget.
- Enable streaming for the final answer.

Static DSL analysis cannot prove live runtime latency. Use live smoke tests only with endpoints you control and harmless prompts.
""",
        "sample_docs/return_policy.md": """# Fictional Return Policy

Acme Example Shop accepts returns within 30 days for unopened items. Opened items may be exchanged when the original receipt is available. Custom engraved items are final sale.

This sample document is fictional and safe for local RAG tests.
""",
    }


def get_dify_template(name: str) -> Dict[str, Any]:
    canonical = canonical_kit_name(name)
    if canonical != "local-private-rag":
        raise KeyError("Unknown Dify template: {0}".format(name))
    return {
        "name": canonical,
        "aliases": sorted(alias for alias, target in KIT_ALIASES.items() if target == canonical),
        "title": "Local / Private RAG Starter Kit for Dify",
        "description": "A read-only starter kit for manually importing a Dify RAG app and measuring UX safely.",
        "app_mode": "advanced-chat",
        "validation_level": "static_compatibility_validated",
        "files": _kit_files(),
    }


def render_dify_template_list() -> str:
    lines = ["InferDoctor Dify Templates", "=" * 57]
    for name in available_dify_template_names():
        template = get_dify_template(name)
        lines.append("- {0}: {1}".format(template["name"], template["description"]))
    lines.extend(["", "Next: inferdoctor dify template show local-private-rag"])
    return "\n".join(lines)


def render_dify_template_show(name: str) -> str:
    template = get_dify_template(name)
    lines = [
        template["title"],
        "=" * 57,
        "Name: {0}".format(template["name"]),
        "Aliases: {0}".format(", ".join(template["aliases"]) or "none"),
        "Intended app mode: {0}".format(template["app_mode"]),
        "Validation level: {0}".format(template["validation_level"]),
        "",
        template["description"],
        "",
        "Files exported:",
    ]
    lines.extend("- {0}".format(path) for path in sorted(template["files"]))
    lines.extend(["", "Export: inferdoctor dify template export local-private-rag --output ./dify-local-private-rag"])
    return "\n".join(lines)


def export_dify_template(name: str, output: str | Path, *, overwrite: bool = False) -> List[Path]:
    template = get_dify_template(name)
    root = Path(output).expanduser()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise DifyError("Output directory is not empty; rerun with --overwrite to replace generated kit files")
    written: List[Path] = []
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in template["files"].items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise DifyError("Unsafe kit path: {0}".format(relative))
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def render_dify_template_export(name: str, output: str | Path, written: Sequence[Path]) -> str:
    return "\n".join([
        "Dify template exported",
        "=" * 57,
        "Template: {0}".format(canonical_kit_name(name)),
        "Output: {0}".format(output),
        "Files written: {0}".format(len(written)),
        "Validation level: package-structure and static DSL checks only",
        "",
        "Next steps:",
        "1. inferdoctor dify validate {0}".format(output),
        "2. inferdoctor dify smoke --kit {0} --dry-run".format(output),
        "3. Manually review and import dify_app.yaml into Dify.",
    ])


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _scan_for_secrets(root: Path) -> List[str]:
    findings: List[str] = []
    patterns = [
        re.compile(r"Authorization:\s*Bearer\s+[^\s]+", re.IGNORECASE),
        re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
        re.compile(r"(api[_-]?key|token|secret|password)\s*[:=]\s*(?!replace|your|REDACTED|\$|<)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
        re.compile(r"https?://[^\s/@]+:[^\s/@]+@"),
    ]
    paths = list(root.rglob("*")) if root.is_dir() else [root]
    for path in paths:
        if path.is_file():
            text = _read_text(path)
            if any(pattern.search(text) for pattern in patterns):
                findings.append("possible secret-like value in {0}".format(path.relative_to(root) if root.is_dir() else path.name))
    return findings


def validate_dify_kit(path: str | Path) -> Dict[str, Any]:
    target = Path(path).expanduser()
    checks: List[Dict[str, Any]] = []

    def add(status: str, item: str, detail: str) -> None:
        checks.append({"status": status, "item": item, "detail": detail})

    if not target.exists():
        add("FAIL", "path", "Path does not exist")
        return _validation_result(target, checks)
    root = target if target.is_dir() else target.parent
    dsl_path = target if target.is_file() else root / "dify_app.yaml"
    if target.is_dir():
        for name in ("manifest.yaml", "dify_app.yaml", "README.md", ".env.example", "smoke_cases.yaml"):
            add("PASS" if (root / name).exists() else "FAIL", "file:{0}".format(name), "required file")
    else:
        add("WARN", "kit-structure", "Validating a single DSL file; kit-level files were not checked")
    manifest = _read_text(root / "manifest.yaml") if (root / "manifest.yaml").exists() else ""
    if manifest:
        add("PASS" if DIFY_KIT_SCHEMA_VERSION in manifest else "FAIL", "manifest schema", "expected {0}".format(DIFY_KIT_SCHEMA_VERSION))
        add("PASS" if "local-private-rag" in manifest else "WARN", "manifest name", "canonical kit name should be local-private-rag")
    if not dsl_path.exists():
        add("FAIL", "DSL", "dify_app.yaml was not found")
        return _validation_result(root, checks)
    dsl = _read_text(dsl_path)
    add("PASS" if "mode:" in dsl else "FAIL", "DSL app mode", "app mode should be present")
    node_ids = re.findall(r"^\s*-\s*id:\s*([A-Za-z0-9_.-]+)", dsl, flags=re.MULTILINE)
    add("PASS" if node_ids and len(node_ids) == len(set(node_ids)) else "FAIL", "node ids", "node ids should be unique")
    for required_type in ("start", "knowledge-retrieval", "llm", "answer"):
        add("PASS" if re.search(r"type:\s*{0}\b".format(re.escape(required_type)), dsl) else "FAIL", "node:{0}".format(required_type), "required node type")
    unresolved_edges = [item for item in re.findall(r"(?:source|target):\s*([A-Za-z0-9_.-]+)", dsl) if item not in node_ids]
    add("PASS" if not unresolved_edges else "FAIL", "edge references", "all edges reference known nodes" if not unresolved_edges else "unknown nodes: {0}".format(", ".join(sorted(set(unresolved_edges)))))
    for placeholder in ("MODEL_PROVIDER_PLACEHOLDER", "MODEL_NAME_PLACEHOLDER", "KNOWLEDGE_DATASET_ID_PLACEHOLDER"):
        add("WARN" if placeholder in dsl else "PASS", "placeholder:{0}".format(placeholder), "replace before live import" if placeholder in dsl else "resolved")
    for finding in _scan_for_secrets(root):
        add("FAIL", "secret scan", finding)
    return _validation_result(root, checks)


def _validation_result(path: Path, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed = sum(1 for check in checks if check["status"] == "FAIL")
    warned = sum(1 for check in checks if check["status"] == "WARN")
    passed = sum(1 for check in checks if check["status"] == "PASS")
    total = max(1, passed + warned + failed)
    score = max(0, int(round((passed + warned * 0.5) / total * 100)))
    return {
        "schema_version": "inferdoctor.dify.validation.v1",
        "timestamp": utc_now(),
        "path": str(path),
        "status": "FAIL" if failed else ("WARN" if warned else "PASS"),
        "readiness_score": score,
        "validation_level": "static_compatibility_validated" if not failed else "package_structure_checked",
        "checks": checks,
        "top_fixes": _top_fixes(checks),
        "warnings": [check["detail"] for check in checks if check["status"] == "WARN"],
    }


def _top_fixes(checks: Sequence[Dict[str, Any]]) -> List[str]:
    fixes = ["Fix {0}: {1}".format(check["item"], check["detail"]) for check in checks if check["status"] == "FAIL"]
    if fixes:
        return fixes[:5]
    warnings = ["Review {0}: {1}".format(check["item"], check["detail"]) for check in checks if check["status"] == "WARN"]
    return warnings[:5] or ["No release-blocking kit issues found by offline validation."]


def render_dify_validation(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return _json(result)
    if output_format == "markdown":
        lines = ["# Dify Kit Validation", "", "- Status: `{0}`".format(result["status"]), "- Readiness: `{0}/100`".format(result["readiness_score"]), "", "## Checks"]
        lines.extend("- `{0}` {1}: {2}".format(check["status"], check["item"], check["detail"]) for check in result["checks"])
        return "\n".join(lines)
    lines = ["Dify Kit Validation", "=" * 57, "Status: {0}".format(result["status"]), "Readiness: {0} / 100".format(result["readiness_score"]), "Validation level: {0}".format(result["validation_level"]), "", "Checks:"]
    lines.extend("- {0:<5} {1}: {2}".format(check["status"], check["item"], check["detail"]) for check in result["checks"])
    lines.extend(["", "Top fixes:"])
    lines.extend("- {0}".format(item) for item in result["top_fixes"])
    lines.extend(["", "Note: Offline validation does not prove live import into Dify."])
    return "\n".join(lines)


def _safe_app_info(info: Dict[str, Any]) -> Dict[str, Any]:
    return {key: info.get(key) for key in ("name", "description", "tags", "mode", "author_name") if key in info}


def _supported_operations_for_mode(mode: str) -> List[str]:
    operations = ["check"]
    if mode in SUPPORTED_LIVE_MODES:
        operations.extend(["smoke", "perf", "optimize"])
    elif mode == "completion":
        operations.append("check-only")
    return operations


def run_dify_check(config: DifyConfig, *, client_factory: Callable[..., DifyAPIClient] = DifyAPIClient) -> Dict[str, Any]:
    safety = ensure_endpoint_allowed(config.app_base_url, allow_non_local=config.allow_non_local, allow_public=config.allow_public)
    result = {
        "schema_version": "inferdoctor.dify.check.v1",
        "timestamp": utc_now(),
        "endpoint": safety["endpoint"],
        "endpoint_category": safety["category"],
        "endpoint_allowed": safety["allowed"],
        "authenticated": False,
        "app_info": None,
        "app_mode": None,
        "supported_operations": [],
        "status": "WARN",
        "warnings": list(safety["warnings"]),
        "errors": [],
    }
    if not safety["allowed"]:
        result["status"] = "FAIL"
        result["errors"].append(safety["reason"])
        return result
    if not config.app_api_key:
        result["warnings"].append("Application API inspection skipped because {0} is not set.".format(config.app_key_env))
        return result
    try:
        info = client_factory(config.app_base_url, config.app_api_key, timeout=config.timeout).get_info()
    except DifyAPIError as exc:
        result["status"] = "FAIL"
        result["errors"].append(str(exc))
        return result
    mode = str(info.get("mode") or "unknown")
    result.update({"authenticated": True, "app_info": _safe_app_info(info), "app_mode": mode, "supported_operations": _supported_operations_for_mode(mode), "status": "PASS" if mode in KNOWN_APP_MODES else "WARN"})
    if mode not in KNOWN_APP_MODES:
        result["warnings"].append("Unknown Dify app mode returned by /info: {0}".format(mode))
    return result


def render_dify_check(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return _json(result)
    if output_format == "markdown":
        return "\n".join(["# Dify Check", "", "- Status: `{0}`".format(result["status"]), "- Endpoint: `{0}`".format(result["endpoint"]), "- App mode: `{0}`".format(result.get("app_mode") or "unknown")])
    lines = ["Dify Application Check", "=" * 57, "Status: {0}".format(result["status"]), "Endpoint: {0}".format(result["endpoint"]), "Endpoint class: {0}".format(result["endpoint_category"]), "Authenticated: {0}".format("yes" if result["authenticated"] else "no"), "App mode: {0}".format(result.get("app_mode") or "unknown"), "Supported operations: {0}".format(", ".join(result.get("supported_operations") or ["check only"]))]
    if result.get("app_info"):
        lines.append("App: {0}".format(result["app_info"].get("name") or "unnamed"))
    if result.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend("- {0}".format(item) for item in result["errors"])
    if result.get("warnings"):
        lines.extend(["", "Warnings:"])
        lines.extend("- {0}".format(item) for item in result["warnings"])
    lines.extend(["", "Next: inferdoctor dify template export local-private-rag --output ./dify-rag"])
    return "\n".join(lines)


def _live_error_report(schema_version: str, endpoint: str, error: str, *, app_mode: Optional[str] = None) -> Dict[str, Any]:
    return {"schema_version": schema_version, "timestamp": utc_now(), "status": "FAIL", "endpoint": sanitize_endpoint(endpoint), "app_mode": app_mode, "request_sent": False, "metrics": {}, "warnings": [], "errors": [redact_secret_text(error)]}


def run_dify_smoke(config: DifyConfig, *, kit_path: Optional[str] = None, dry_run: bool = False, query: Optional[str] = None, show_answer: bool = False, client_factory: Callable[..., DifyAPIClient] = DifyAPIClient) -> Dict[str, Any]:
    if dry_run:
        validation = validate_dify_kit(kit_path or ".") if kit_path else None
        failed = validation is not None and validation["status"] == "FAIL"
        return {"schema_version": DIFY_SMOKE_SCHEMA_VERSION, "timestamp": utc_now(), "mode": "offline-dry-run", "status": "FAIL" if failed else "PASS", "endpoint": None, "app_mode": None, "request_sent": False, "validation": validation, "metrics": {}, "warnings": ["No Dify endpoint was contacted."], "errors": validation["top_fixes"] if failed else []}
    safety = ensure_endpoint_allowed(config.app_base_url, allow_non_local=config.allow_non_local, allow_public=config.allow_public)
    if not safety["allowed"]:
        return _live_error_report(DIFY_SMOKE_SCHEMA_VERSION, config.app_base_url, safety["reason"])
    if not config.app_api_key:
        return _live_error_report(DIFY_SMOKE_SCHEMA_VERSION, config.app_base_url, "{0} is not set".format(config.app_key_env))
    client = client_factory(config.app_base_url, config.app_api_key, timeout=config.timeout)
    try:
        info = client.get_info()
        mode = str(info.get("mode") or "unknown")
        if mode in CHAT_MODES:
            metrics = client.run_chat_stream(query or "Reply with one short sentence that the Dify app is reachable.", user=generated_user_id(), show_answer=show_answer)
        elif mode in WORKFLOW_MODES:
            metrics = client.run_workflow_stream(query or "Reply with one short sentence that the Dify workflow is reachable.", user=generated_user_id(), show_answer=show_answer)
        else:
            return _live_error_report(DIFY_SMOKE_SCHEMA_VERSION, config.app_base_url, "Live smoke is not supported for app mode {0}".format(mode), app_mode=mode)
    except DifyAPIError as exc:
        return _live_error_report(DIFY_SMOKE_SCHEMA_VERSION, config.app_base_url, str(exc))
    return {"schema_version": DIFY_SMOKE_SCHEMA_VERSION, "timestamp": utc_now(), "mode": "live-app-api", "status": "PASS" if not metrics.get("errors") else "WARN", "endpoint": sanitize_endpoint(config.app_base_url), "app_mode": mode, "request_sent": True, "answer_retained": bool(show_answer and metrics.get("answer_preview")), "metrics": metrics, "warnings": ["Answer content is suppressed by default." if not show_answer else "Answer preview was shown because --show-answer was used."], "errors": list(metrics.get("errors") or [])}


def render_dify_smoke(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return _json(result)
    if output_format == "markdown":
        return "\n".join(["# Dify Smoke Test", "", "- Status: `{0}`".format(result["status"]), "- Mode: `{0}`".format(result.get("mode") or "unknown"), "- Request sent: `{0}`".format(result.get("request_sent"))])
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    lines = ["Dify Smoke Test", "=" * 57, "Status: {0}".format(result["status"]), "Mode: {0}".format(result.get("mode") or "unknown"), "Request sent: {0}".format("yes" if result.get("request_sent") else "no"), "App mode: {0}".format(result.get("app_mode") or "unknown"), "First visible text / TTFT: {0}".format(metrics.get("ttft_seconds")), "Total latency: {0}".format(metrics.get("total_latency_seconds")), "Events: {0}".format(metrics.get("event_count", 0)), "Completion: {0}".format(metrics.get("completion_status") or "unknown")]
    if result.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend("- {0}".format(item) for item in result["errors"])
    if result.get("warnings"):
        lines.extend(["", "Notes:"])
        lines.extend("- {0}".format(item) for item in result["warnings"])
    lines.extend(["", "Note: This is a bounded smoke test, not a benchmark."])
    return "\n".join(lines)


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.median(clean), 6) if clean else None


def run_dify_perf(config: DifyConfig, *, runs: int = 1, warmup: int = 0, profile: Optional[str] = None, query: Optional[str] = None, client_factory: Callable[..., DifyAPIClient] = DifyAPIClient) -> Dict[str, Any]:
    if runs < 1 or runs > 3:
        raise DifyError("--runs must be between 1 and 3")
    if warmup < 0 or warmup > 1:
        raise DifyError("--warmup must be between 0 and 1")
    samples: List[Dict[str, Any]] = []
    errors: List[str] = []
    app_mode: Optional[str] = None
    for index in range(warmup + runs):
        smoke = run_dify_smoke(config, query=query, client_factory=client_factory)
        if smoke.get("app_mode"):
            app_mode = str(smoke.get("app_mode"))
        if index < warmup:
            continue
        if smoke.get("status") == "FAIL":
            errors.extend(smoke.get("errors") or [])
        else:
            samples.append(smoke.get("metrics") or {})
    aggregate = {"ttft_median": _median([sample.get("ttft_seconds") for sample in samples]), "total_latency_median": _median([sample.get("total_latency_seconds") for sample in samples]), "generation_duration_median": None, "generation_tps_median": None}
    return {
        "schema_version": DIFY_PERF_SCHEMA_VERSION,
        "timestamp": utc_now(),
        "endpoint": sanitize_endpoint(config.app_base_url),
        "model": "dify-app",
        "test_type": "dify-streaming",
        "source_type": "dify",
        "app_mode": app_mode,
        "streaming_requested": True,
        "streaming_observed": "confirmed" if any(sample.get("first_visible_text_index") is not None for sample in samples) else "unknown",
        "successful_runs": len(samples),
        "failed_runs": max(0, runs - len(samples)),
        "metrics": {"ttft_seconds": aggregate["ttft_median"], "total_latency_seconds": aggregate["total_latency_median"], "generation_duration_seconds": None, "generation_tokens_per_second": None, "aggregate": aggregate},
        "metric_quality": {"ttft": "observed" if aggregate["ttft_median"] is not None else "unavailable", "tps": "unavailable"},
        "experience_read": _dify_experience_read(aggregate["ttft_median"], aggregate["total_latency_median"], len(samples), runs),
        "dify_metrics": {"samples": samples, "profile": profile, "warmup_runs": warmup},
        "warnings": ["Dify performance results are bounded smoke tests, not formal benchmarks."],
        "errors": errors,
    }


def _dify_experience_read(ttft: Optional[float], total: Optional[float], success: int, runs: int) -> Dict[str, Any]:
    if success == 0:
        return {"category": "Endpoint/configuration failure", "confidence": "high", "reason": "no measured run succeeded"}
    if ttft is None:
        return {"category": "Inconclusive", "confidence": "low", "reason": "first visible text was not observed"}
    if ttft <= 1.5 and (total is None or total <= 8.0):
        return {"category": "Responsive for interactive Dify app", "confidence": "medium", "reason": "first visible text arrived quickly"}
    if ttft <= 3.0:
        return {"category": "Usable with streaming", "confidence": "medium", "reason": "streaming can keep the user engaged"}
    return {"category": "Likely frustrating without progress feedback", "confidence": "medium", "reason": "TTFT is high for an interactive RAG app"}


def render_dify_perf(report: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return _json(report)
    if output_format == "markdown":
        return "\n".join(["# Dify Performance Smoke Test", "", "- Endpoint: `{0}`".format(report["endpoint"]), "- Successful runs: `{0}`".format(report["successful_runs"]), "- Readiness: **{0}**".format(report["experience_read"]["category"])])
    metrics = report.get("metrics") or {}
    lines = ["Dify Performance Smoke Test", "=" * 57, "Endpoint: {0}".format(report.get("endpoint")), "App mode: {0}".format(report.get("app_mode") or "unknown"), "Successful runs: {0}".format(report.get("successful_runs", 0)), "Failed runs: {0}".format(report.get("failed_runs", 0)), "Median first visible text / TTFT: {0}".format(metrics.get("ttft_seconds")), "Median total latency: {0}".format(metrics.get("total_latency_seconds")), "Readiness: {0}".format(report.get("experience_read", {}).get("category")), "Reason: {0}".format(report.get("experience_read", {}).get("reason")), "", "Next: inferdoctor perf baseline create --report dify-perf.json --name dify-before", "Note: This is a smoke test, not a benchmark."]
    if report.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend("- {0}".format(item) for item in report["errors"])
    return "\n".join(lines)


def _rec(priority: str, evidence: str, action: str, verify: str, limitation: str) -> Dict[str, str]:
    return {"priority": priority, "evidence": evidence, "action": action, "verify": verify, "limitation": limitation}


def optimize_dify(*, report_path: Optional[str] = None, kit_path: Optional[str] = None, retrieval_ms: Optional[float] = None, rerank_ms: Optional[float] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    observations: List[Dict[str, str]] = []
    recommendations: List[Dict[str, str]] = []
    if report_path:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
        ttft = metrics.get("ttft_seconds")
        total = metrics.get("total_latency_seconds")
        if isinstance(ttft, (int, float)):
            observations.append({"evidence": "Observed", "text": "Dify first visible text latency was {0:.2f}s.".format(ttft)})
            if ttft > 3.0:
                recommendations.append(_rec("Do now", "Observed", "Show retrieval progress before generation and verify streaming is enabled.", "inferdoctor dify perf --base-url $DIFY_API_BASE_URL --runs 2 --warmup 1", "High TTFT can make a RAG app feel stuck."))
        if isinstance(total, (int, float)) and total > 12.0:
            recommendations.append(_rec("Test next", "Observed", "Reduce top_k or context budget, then compare before and after.", "inferdoctor perf compare before.json after.json", "Lower context may improve latency but can reduce answer coverage."))
        if report.get("failed_runs", 0):
            recommendations.append(_rec("Do now", "Observed", "Fix Dify app/API stability before optimizing speed.", "inferdoctor dify check --base-url $DIFY_API_BASE_URL", "Failures make performance comparisons unreliable."))
    if retrieval_ms is not None:
        observations.append({"evidence": "User supplied", "text": "Retrieval latency was provided as {0:.0f} ms.".format(retrieval_ms)})
        if retrieval_ms > 700:
            recommendations.append(_rec("Test next", "Strongly indicated", "Reduce top_k, add retrieval progress messages, or cache repeated retrievals.", "inferdoctor dify optimize --retrieval-ms {0:.0f}".format(retrieval_ms), "User-provided retrieval time suggests retrieval may be in the critical path."))
    if rerank_ms is not None and rerank_ms > 1000:
        recommendations.append(_rec("Consider later", "Possible", "Test whether rerank is necessary for the demo path.", "inferdoctor dify optimize --rerank-ms {0:.0f}".format(rerank_ms), "Rerank can improve quality but often adds visible latency."))
    if kit_path:
        validation = validate_dify_kit(kit_path)
        observations.append({"evidence": "Static analysis", "text": "Kit validation status: {0}.".format(validation["status"])})
        if validation["status"] == "FAIL":
            recommendations.append(_rec("Do now", "Observed", "Resolve kit validation failures before live import.", "inferdoctor dify validate {0}".format(kit_path), "Static issues can block manual import or live smoke tests."))
    if not recommendations:
        recommendations.append(_rec("Not enough evidence", "Not enough evidence", "Create a Dify performance report before tuning.", "inferdoctor dify perf --base-url $DIFY_API_BASE_URL --format json --output dify-perf.json", "Optimization advice is stronger after a bounded smoke test."))
    return {"schema_version": DIFY_OPTIMIZE_SCHEMA_VERSION, "timestamp": utc_now(), "profile": profile, "observations": observations, "recommendations": recommendations[:6], "limitations": ["No exact performance gain is promised.", "Static DSL analysis does not measure runtime latency.", "Model answer quality is not measured."]}


def render_dify_optimize(plan: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return _json(plan)
    if output_format == "markdown":
        lines = ["# Dify Optimization Guidance", "", "## Recommendations"]
        lines.extend("- **{0}** ({1}): {2}".format(item["priority"], item["evidence"], item["action"]) for item in plan["recommendations"])
        return "\n".join(lines)
    lines = ["Dify Optimization Guidance", "=" * 57]
    if plan.get("observations"):
        lines.extend(["", "Observations:"])
        lines.extend("- {0}: {1}".format(item["evidence"], item["text"]) for item in plan["observations"])
    lines.extend(["", "Recommendations:"])
    for item in plan["recommendations"]:
        lines.append("- {0} [{1}]: {2}".format(item["priority"], item["evidence"], item["action"]))
        lines.append("  Verify: {0}".format(item["verify"]))
    lines.extend(["", "Limitations:"])
    lines.extend("- {0}".format(item) for item in plan["limitations"])
    return "\n".join(lines)


def run_dify_knowledge_check(config: DifyConfig, *, query: str = "fictional return policy", show_content: bool = False, client_factory: Callable[..., DifyAPIClient] = DifyAPIClient) -> Dict[str, Any]:
    base_url = config.knowledge_base_url or config.app_base_url
    safety = ensure_endpoint_allowed(base_url, allow_non_local=config.allow_non_local, allow_public=config.allow_public)
    if not safety["allowed"]:
        return _live_error_report(DIFY_KNOWLEDGE_SCHEMA_VERSION, base_url, safety["reason"])
    if not config.knowledge_api_key:
        return _live_error_report(DIFY_KNOWLEDGE_SCHEMA_VERSION, base_url, "{0} is not set".format(config.knowledge_key_env))
    if not config.dataset_id:
        return _live_error_report(DIFY_KNOWLEDGE_SCHEMA_VERSION, base_url, "{0} is not set".format(DEFAULT_DATASET_ENV))
    client = client_factory(base_url, config.knowledge_api_key, timeout=config.timeout)
    started = time.perf_counter()
    try:
        data = client.retrieve_chunks(config.dataset_id, query)
    except DifyAPIError as exc:
        return _live_error_report(DIFY_KNOWLEDGE_SCHEMA_VERSION, base_url, str(exc))
    latency = round(time.perf_counter() - started, 6)
    records = data.get("records") if isinstance(data.get("records"), list) else []
    scores = [float(record["score"]) for record in records if isinstance(record, dict) and isinstance(record.get("score"), (int, float))]
    return {"schema_version": DIFY_KNOWLEDGE_SCHEMA_VERSION, "timestamp": utc_now(), "status": "PASS", "endpoint": sanitize_endpoint(base_url), "dataset_id_present": True, "result_count": len(records), "score_min": min(scores) if scores else None, "score_max": max(scores) if scores else None, "retrieval_latency_seconds": latency, "content_retained": show_content, "content_preview": _knowledge_preview(records) if show_content else None, "warnings": ["Retrieved content is suppressed by default." if not show_content else "Retrieved content preview was shown because --show-content was used."], "errors": []}


def _knowledge_preview(records: Sequence[Any]) -> List[str]:
    previews: List[str] = []
    for record in records[:3]:
        segment = record.get("segment") if isinstance(record, dict) else None
        content = segment.get("content") if isinstance(segment, dict) else None
        if isinstance(content, str):
            previews.append(content[:120])
    return previews


def render_dify_knowledge(result: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return _json(result)
    if output_format == "markdown":
        return "# Dify Knowledge Retrieval Check\n\n- Status: `{0}`\n- Results: `{1}`".format(result.get("status"), result.get("result_count"))
    lines = ["Dify Knowledge Retrieval Check", "=" * 57, "Status: {0}".format(result.get("status")), "Results: {0}".format(result.get("result_count", 0)), "Retrieval latency: {0}".format(result.get("retrieval_latency_seconds")), "Content retained: {0}".format("yes" if result.get("content_retained") else "no")]
    if result.get("errors"):
        lines.extend(["", "Errors:"])
        lines.extend("- {0}".format(item) for item in result["errors"])
    if result.get("warnings"):
        lines.extend(["", "Notes:"])
        lines.extend("- {0}".format(item) for item in result["warnings"])
    return "\n".join(lines)
