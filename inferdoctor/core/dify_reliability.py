from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from inferdoctor import __version__
from inferdoctor.core.dify import redact_secret_text
from inferdoctor.core.endpoint_safety import classify_endpoint, redact_endpoint

DIFY_EVIDENCE_SCHEMA_VERSION = "inferdoctor.dify.evidence.v1"
DIFY_PREFLIGHT_SCHEMA_VERSION = "inferdoctor.dify.selfhost.preflight.v1"
DIFY_INSPECT_SCHEMA_VERSION = "inferdoctor.dify.selfhost.inspect.v1"
DIFY_CONNECTIVITY_SCHEMA_VERSION = "inferdoctor.dify.connectivity.v1"
DIFY_EVIDENCE_BUNDLE_SCHEMA_VERSION = "inferdoctor.dify.evidence.bundle.v1"
DIFY_EVIDENCE_EXPLAIN_SCHEMA_VERSION = "inferdoctor.dify.evidence.explain.v1"

COMPOSE_FILENAMES = ("compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml")
SAFE_SERVICE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,96}$")
SECRET_QUERY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)=([^&\s]+)")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
VECTOR_TOKENS = ("weaviate", "qdrant", "milvus", "pgvector", "opensearch", "elasticsearch", "chroma", "couchbase", "tidb", "oracle")
SIGNATURES = [
    ("postgres_unavailable", re.compile(r"connection refused|could not connect.*postgres|database.*unavailable", re.I)),
    ("redis_unavailable", re.compile(r"redis.*(connection refused|unavailable|timeout)", re.I)),
    ("vector_store_unavailable", re.compile(r"(weaviate|qdrant|milvus|pgvector|opensearch|elasticsearch|chroma).*(unavailable|connection refused|timeout|dimension|schema)", re.I)),
    ("plugin_daemon_failure", re.compile(r"plugin daemon|plugin_daemon|plugin not found|marketplace|pypi|plugin.*certificate|plugin.*proxy", re.I)),
    ("ssrf_rejection", re.compile(r"ssrf|forbidden private|blocked address|proxy rejected", re.I)),
    ("sandbox_failure", re.compile(r"sandbox.*(failed|unavailable|connection refused|timeout)|failed to execute code", re.I)),
    ("migration_error", re.compile(r"migration|alembic|database schema|version mismatch", re.I)),
    ("oom_or_resource", re.compile(r"out of memory|oom|no space left|too many open files|thread", re.I)),
    ("dns_failure", re.compile(r"name or service not known|temporary failure in name resolution|dns", re.I)),
    ("auth_failure", re.compile(r"unauthorized|forbidden|401|403|authentication", re.I)),
]


class DifyReliabilityError(ValueError):
    pass


@dataclass(frozen=True)
class CommandResult:
    args: Tuple[str, ...]
    returncode: Optional[int]
    stdout: str
    stderr: str
    category: str
    timed_out: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_text(value: Any, secrets_to_hide: Sequence[Optional[str]] = ()) -> str:
    text = CONTROL_RE.sub("?", str(value))
    text = SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}=REDACTED", text)
    return redact_secret_text(text, secrets_to_hide)[:4096]


def sanitize_url(url: str) -> str:
    return redact_endpoint(SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}=REDACTED", url))


def _stable_id(parts: Sequence[Any]) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def make_observation(
    *,
    component: str,
    component_role: str,
    layer: str,
    source_type: str,
    source_reference: str,
    check_name: str,
    status: str,
    summary: str,
    sanitized_detail: str = "",
    evidence_strength: str = "observed",
    confidence: str = "medium",
    error_signature: Optional[str] = None,
    correlation_ids: Optional[Sequence[str]] = None,
    upstream_candidates: Optional[Sequence[str]] = None,
    downstream_effects: Optional[Sequence[str]] = None,
    next_checks: Optional[Sequence[str]] = None,
    redaction_applied: bool = True,
) -> Dict[str, Any]:
    if status not in {"pass", "warn", "fail", "unknown", "skipped"}:
        raise DifyReliabilityError(f"invalid evidence status: {status}")
    if evidence_strength not in {"observed", "strongly_indicated", "possible", "unknown"}:
        raise DifyReliabilityError(f"invalid evidence strength: {evidence_strength}")
    return {
        "schema_version": DIFY_EVIDENCE_SCHEMA_VERSION,
        "observation_id": _stable_id([component, layer, check_name, status, summary, source_reference]),
        "timestamp": utc_now(),
        "component": component,
        "component_role": component_role,
        "layer": layer,
        "source_type": source_type,
        "source_reference": sanitize_text(source_reference),
        "check_name": check_name,
        "status": status,
        "summary": sanitize_text(summary),
        "sanitized_detail": sanitize_text(sanitized_detail),
        "evidence_strength": evidence_strength,
        "confidence": confidence,
        "error_signature": error_signature,
        "correlation_ids": list(correlation_ids or []),
        "upstream_candidates": list(upstream_candidates or []),
        "downstream_effects": list(downstream_effects or []),
        "next_checks": list(next_checks or []),
        "redaction_applied": redaction_applied,
    }


def validate_evidence(observation: Dict[str, Any]) -> None:
    required = {
        "schema_version", "observation_id", "timestamp", "component", "component_role",
        "layer", "source_type", "source_reference", "check_name", "status", "summary",
        "sanitized_detail", "evidence_strength", "confidence", "redaction_applied",
    }
    missing = required - set(observation)
    if missing:
        raise DifyReliabilityError("malformed evidence missing: " + ", ".join(sorted(missing)))
    if observation["schema_version"] != DIFY_EVIDENCE_SCHEMA_VERSION:
        raise DifyReliabilityError("unsupported evidence schema: " + str(observation.get("schema_version")))


class BoundedCommandRunner:
    def __init__(self, allowed: Optional[Iterable[str]] = None) -> None:
        self.allowed = set(allowed or {"docker", "uname", "df", "stat", "ss", "getent"})

    def run(self, args: Sequence[str], *, timeout: float = 5.0, max_bytes: int = 65536) -> CommandResult:
        if not args:
            return CommandResult(tuple(), None, "", "empty command", "invalid")
        executable = Path(args[0]).name
        if executable not in self.allowed:
            return CommandResult(tuple(args), None, "", "executable is not allowed", "not_allowed")
        try:
            completed = subprocess.run(list(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
        except FileNotFoundError:
            return CommandResult(tuple(args), None, "", f"{executable} was not found", "missing_executable")
        except PermissionError:
            return CommandResult(tuple(args), None, "", f"permission denied running {executable}", "permission_denied")
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"")[:max_bytes].decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"")[:max_bytes].decode("utf-8", errors="replace")
            return CommandResult(tuple(args), None, sanitize_text(stdout), sanitize_text(stderr), "timeout", timed_out=True)
        stdout = completed.stdout[:max_bytes].decode("utf-8", errors="replace")
        stderr = completed.stderr[:max_bytes].decode("utf-8", errors="replace")
        category = "ok" if completed.returncode == 0 else "non_zero_exit"
        low = stderr.lower()
        if "permission denied" in low:
            category = "permission_denied"
        if "cannot connect to the docker daemon" in low or "is the docker daemon running" in low:
            category = "daemon_unavailable"
        return CommandResult(tuple(args), completed.returncode, sanitize_text(stdout), sanitize_text(stderr), category)


def resolve_compose_file(compose_file: Optional[str] = None, project_directory: Optional[str] = None, *, cwd: Optional[Path] = None) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    findings: List[Dict[str, Any]] = []
    base = Path.cwd() if cwd is None else cwd
    if compose_file:
        path = Path(compose_file).expanduser().resolve()
        if path.exists() and path.is_file():
            return path, findings
        return None, [make_observation(component="compose", component_role="compose", layer="compose", source_type="file", source_reference=str(path), check_name="compose_file", status="fail", summary="Compose file was not found.", confidence="high")]
    search_dir = Path(project_directory).expanduser().resolve() if project_directory else base.resolve()
    if not search_dir.exists() or not search_dir.is_dir():
        return None, [make_observation(component="compose", component_role="compose", layer="compose", source_type="directory", source_reference=str(search_dir), check_name="project_directory", status="fail", summary="Project directory was not found.", confidence="high")]
    for name in COMPOSE_FILENAMES:
        candidate = search_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate, findings
    findings.append(make_observation(component="compose", component_role="compose", layer="compose", source_type="directory", source_reference=str(search_dir), check_name="compose_discovery", status="unknown", summary="No Compose file was found in the selected directory.", next_checks=["Pass --compose-file PATH for self-host diagnostics."]))
    return None, findings


def _read_text_limited(path: Path, limit: int = 262144) -> str:
    return path.read_bytes()[:limit].decode("utf-8", errors="replace")


def parse_compose_services(compose_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not compose_path.exists():
        return [], ["compose file does not exist"]
    text = _read_text_limited(compose_path)
    services: List[Dict[str, Any]] = []
    warnings: List[str] = []
    in_services = False
    current: Optional[Dict[str, Any]] = None
    current_section: Optional[str] = None
    for line in text.splitlines():
        if re.match(r"^services:\s*(#.*)?$", line):
            in_services = True
            current = None
            continue
        if not in_services:
            continue
        if re.match(r"^[A-Za-z0-9_.-]+:\s*(#.*)?$", line) and not line.startswith(" "):
            break
        service_match = re.match(r"^  ([A-Za-z0-9_.-]+):\s*(#.*)?$", line)
        if service_match:
            current = {"name": service_match.group(1), "image": "", "environment_keys": [], "ports": [], "env_files": [], "volumes": []}
            services.append(current)
            current_section = None
            continue
        if current is None:
            continue
        stripped = line.strip()
        section_match = re.match(r"^([A-Za-z0-9_-]+):\s*$", stripped)
        if section_match:
            current_section = section_match.group(1)
            continue
        if stripped.startswith("image:"):
            current["image"] = stripped.split(":", 1)[1].strip().strip("\"'")
        elif current_section == "environment":
            key = None
            if stripped.startswith("-"):
                key = stripped[1:].strip().split("=", 1)[0].split(":", 1)[0].strip()
            elif ":" in stripped:
                key = stripped.split(":", 1)[0].strip()
            if key and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
                current["environment_keys"].append(key)
        elif current_section == "ports" and stripped.startswith("-"):
            current["ports"].append(stripped[1:].strip().strip("\"'"))
        elif current_section == "env_file" and stripped.startswith("-"):
            current["env_files"].append(stripped[1:].strip().strip("\"'"))
        elif current_section == "volumes" and stripped.startswith("-"):
            current["volumes"].append(stripped[1:].strip().strip("\"'"))
    if not services:
        warnings.append("no services were detected by the lightweight parser")
    return services, warnings


def detect_service_role(service: Dict[str, Any]) -> Dict[str, Any]:
    name = service.get("name", "").lower().replace("-", "_")
    image = service.get("image", "").lower().replace("-", "_")
    env_keys = " ".join(str(key).lower() for key in service.get("environment_keys", []))
    haystack = f"{name} {image} {env_keys}"
    checks = [
        ("plugin_daemon", ("plugin_daemon", "plugin daemon")),
        ("worker_beat", ("worker_beat", "scheduler", "beat")),
        ("ssrf_proxy", ("ssrf_proxy", "ssrf")),
        ("sandbox", ("sandbox",)),
        ("worker", ("worker", "celery")),
        ("api", ("api", "server")),
        ("web", ("web", "frontend")),
        ("postgres", ("postgres", "postgresql")),
        ("redis", ("redis",)),
        ("nginx", ("nginx",)),
    ]
    role = "unknown"
    confidence = "low"
    evidence = ["no known Dify role token matched"]
    for candidate, tokens in checks:
        if any(token in haystack for token in tokens):
            role = candidate
            confidence = "high" if candidate in name else "medium"
            evidence = [f"matched {candidate} in service name/image/env keys"]
            break
    if role == "unknown":
        for token in VECTOR_TOKENS:
            if token in haystack:
                role = "vector_store"
                confidence = "medium"
                evidence = [f"matched vector store token {token}"]
                break
    return {
        "service": service.get("name"),
        "role": role,
        "confidence": confidence,
        "evidence": evidence,
        "image": service.get("image", ""),
        "environment_keys": sorted(set(service.get("environment_keys", []))),
        "ports": service.get("ports", []),
        "env_files": service.get("env_files", []),
        "volumes": service.get("volumes", []),
    }


def service_inventory(compose_path: Optional[Path]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not compose_path:
        return [], ["no compose file selected"]
    services, warnings = parse_compose_services(compose_path)
    return [detect_service_role(service) for service in services], warnings


def host_snapshot(project_directory: Optional[Path] = None) -> Dict[str, Any]:
    mem_total = None
    mem_available = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1]) * 1024
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1]) * 1024
    target = project_directory or Path.cwd()
    disk = shutil.disk_usage(target if target.exists() else Path.cwd())
    statvfs = os.statvfs(target if target.exists() else Path.cwd())
    return {
        "os": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": mem_total,
        "memory_available_bytes": mem_available,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "inode_free": statvfs.f_favail,
        "inode_total": statvfs.f_files,
    }


def parse_host_ports(inventory: Sequence[Dict[str, Any]]) -> List[int]:
    ports: List[int] = []
    for service in inventory:
        for mapping in service.get("ports", []):
            clean = str(mapping).strip().strip("\"'")
            parts = clean.split(":")
            candidate = parts[-2] if len(parts) >= 2 else parts[0]
            if "/" in candidate:
                candidate = candidate.split("/", 1)[0]
            try:
                value = int(candidate)
            except ValueError:
                continue
            if 0 < value < 65536 and value not in ports:
                ports.append(value)
    return ports


def check_port(port: int) -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        return "occupied_by_unknown" if sock.connect_ex(("127.0.0.1", port)) == 0 else "available"
    except OSError:
        return "unknown"
    finally:
        sock.close()


def _overall_status(observations: Sequence[Dict[str, Any]], ready_word: str) -> str:
    statuses = {item.get("status") for item in observations}
    if "fail" in statuses:
        return "BLOCKED"
    if "warn" in statuses:
        return "ATTENTION"
    if statuses and statuses <= {"pass"}:
        return ready_word
    return "UNKNOWN"


def run_dify_selfhost_preflight(*, compose_file: Optional[str] = None, project_directory: Optional[str] = None, project_name: Optional[str] = None, runner: Optional[BoundedCommandRunner] = None) -> Dict[str, Any]:
    runner = runner or BoundedCommandRunner()
    compose_path, observations = resolve_compose_file(compose_file, project_directory)
    project_dir = compose_path.parent if compose_path else (Path(project_directory).resolve() if project_directory else Path.cwd())
    snapshot = host_snapshot(project_dir)
    inventory, warnings = service_inventory(compose_path)
    observations = list(observations)
    observations.append(make_observation(component="host", component_role="host", layer="host", source_type="system", source_reference=platform.node(), check_name="host_snapshot", status="pass", summary="Host resource snapshot collected.", sanitized_detail=f"cpu={snapshot.get('cpu_count')} arch={snapshot.get('architecture')}", confidence="high"))
    if snapshot.get("memory_available_bytes") is not None and snapshot["memory_available_bytes"] < 4 * 1024 ** 3:
        observations.append(make_observation(component="host", component_role="host", layer="host", source_type="system", source_reference="/proc/meminfo", check_name="memory", status="warn", summary="Available memory is low for a comfortable Dify deployment.", evidence_strength="strongly_indicated"))
    if snapshot.get("disk_free_bytes", 0) < 10 * 1024 ** 3:
        observations.append(make_observation(component="host", component_role="host", layer="host", source_type="filesystem", source_reference=str(project_dir), check_name="disk", status="warn", summary="Available disk space is low for logs, database, vectors, and model caches.", evidence_strength="strongly_indicated"))
    docker_version = runner.run(["docker", "version", "--format", "{{json .}}"], timeout=5)
    observations.append(make_observation(component="docker", component_role="docker", layer="docker", source_type="command", source_reference="docker version", check_name="docker_daemon", status="pass" if docker_version.category == "ok" else "warn", summary="Docker daemon is reachable." if docker_version.category == "ok" else "Docker CLI or daemon is unavailable.", sanitized_detail=docker_version.stderr or docker_version.stdout[:240], confidence="high", next_checks=[] if docker_version.category == "ok" else ["Start Docker or fix current-user Docker access before inspecting a running deployment."]))
    compose_version = runner.run(["docker", "compose", "version", "--short"], timeout=5)
    observations.append(make_observation(component="docker", component_role="compose", layer="docker", source_type="command", source_reference="docker compose version", check_name="compose_v2", status="pass" if compose_version.category == "ok" else "warn", summary="Docker Compose v2 is available." if compose_version.category == "ok" else "Docker Compose v2 was not confirmed.", sanitized_detail=compose_version.stderr or compose_version.stdout[:120]))
    if compose_path:
        observations.append(make_observation(component="compose", component_role="compose", layer="compose", source_type="file", source_reference=str(compose_path), check_name="compose_file", status="pass", summary="Compose file selected for Dify diagnostics.", confidence="high"))
    for warning in warnings:
        observations.append(make_observation(component="compose", component_role="compose", layer="compose", source_type="file", source_reference=str(compose_path or ""), check_name="compose_parse", status="warn", summary=warning))
    for service in inventory:
        observations.append(make_observation(component=str(service.get("service")), component_role=str(service.get("role")), layer="compose", source_type="compose_service", source_reference=str(compose_path or ""), check_name="service_role", status="pass" if service.get("role") != "unknown" else "unknown", summary=f"Detected service role: {service.get('role')}.", sanitized_detail="; ".join(service.get("evidence", [])), confidence=str(service.get("confidence", "low"))))
    for port in parse_host_ports(inventory):
        observations.append(make_observation(component="port", component_role="host_port", layer="host", source_type="compose_ports", source_reference=str(port), check_name="declared_port", status="pass", summary=f"Declared host port {port} is {check_port(port)}."))
    role_set = {service.get("role") for service in inventory}
    for role in ("api", "worker", "plugin_daemon", "sandbox", "ssrf_proxy", "postgres", "redis"):
        if inventory and role not in role_set:
            observations.append(make_observation(component=role, component_role=role, layer="compose", source_type="role_inventory", source_reference=str(compose_path or ""), check_name="required_role", status="warn", summary=f"No {role} service was detected in the selected Compose file.", evidence_strength="possible"))
    return {
        "schema_version": DIFY_PREFLIGHT_SCHEMA_VERSION,
        "timestamp": utc_now(),
        "inferdoctor_version": __version__,
        "status": _overall_status(observations, "READY"),
        "compose_file": str(compose_path) if compose_path else None,
        "project_directory": str(project_dir),
        "project_name": project_name,
        "host": snapshot,
        "service_inventory": inventory,
        "findings": observations,
        "limitations": ["Preflight is read-only and heuristic.", "It does not start containers, pull images, or verify a live Dify app."],
    }


def _parse_docker_ps(text: str) -> List[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows


def extract_log_signatures(text: str) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    seen = set()
    for line in text.splitlines()[:1000]:
        clean = sanitize_text(line)[:300]
        for name, pattern in SIGNATURES:
            if pattern.search(clean):
                key = (name, clean[:80])
                if key in seen:
                    continue
                seen.add(key)
                role = "plugin_daemon" if "plugin" in name else "ssrf_proxy" if "ssrf" in name else "sandbox" if "sandbox" in name else "unknown"
                findings.append({"signature": name, "summary": f"Log signature detected: {name}.", "excerpt": clean, "component": role, "component_role": role})
                break
        if len(findings) >= 40:
            break
    return findings


def _select_services(inventory: Sequence[Dict[str, Any]], services: Optional[Sequence[str]]) -> List[str]:
    if services:
        wanted = {item.strip() for item in services if item.strip()}
        return [str(item.get("service")) for item in inventory if item.get("service") in wanted or item.get("role") in wanted]
    return [str(item.get("service")) for item in inventory if item.get("role") in {"api", "worker", "plugin_daemon", "sandbox", "ssrf_proxy", "postgres", "redis", "vector_store"}]


def run_dify_selfhost_inspect(*, compose_file: Optional[str] = None, project_directory: Optional[str] = None, project_name: Optional[str] = None, since: str = "10m", services: Optional[Sequence[str]] = None, details: bool = False, runner: Optional[BoundedCommandRunner] = None) -> Dict[str, Any]:
    runner = runner or BoundedCommandRunner()
    compose_path, observations = resolve_compose_file(compose_file, project_directory)
    inventory, warnings = service_inventory(compose_path)
    observations = list(observations)
    for warning in warnings:
        observations.append(make_observation(component="compose", component_role="compose", layer="compose", source_type="file", source_reference=str(compose_path or ""), check_name="compose_parse", status="warn", summary=warning))
    docker_ps: List[Dict[str, Any]] = []
    if compose_path:
        observations.append(make_observation(component="compose", component_role="compose", layer="compose", source_type="file", source_reference=str(compose_path), check_name="compose_file", status="pass", summary="Compose file selected for inspection."))
        ps = runner.run(["docker", "compose", "-f", str(compose_path), "ps", "--format", "json"], timeout=8)
        if ps.category == "ok":
            docker_ps = _parse_docker_ps(ps.stdout)
            observations.append(make_observation(component="docker", component_role="compose", layer="docker", source_type="command", source_reference="docker compose ps", check_name="compose_ps", status="pass", summary="Docker Compose project state was read."))
        else:
            observations.append(make_observation(component="docker", component_role="compose", layer="docker", source_type="command", source_reference="docker compose ps", check_name="compose_ps", status="warn", summary="Docker Compose project state could not be read.", sanitized_detail=ps.stderr or ps.stdout, next_checks=["Check Docker daemon and current-user Docker access."]))
    for item in inventory:
        observations.append(make_observation(component=str(item.get("service")), component_role=str(item.get("role")), layer="container", source_type="docker_compose", source_reference=str(compose_path or ""), check_name="container_state", status="pass" if docker_ps else "unknown", summary=f"Service {item.get('service')} role {item.get('role')} inspected.", sanitized_detail=json.dumps({"image": item.get("image")}, sort_keys=True), confidence=str(item.get("confidence", "medium"))))
    log_signatures: List[Dict[str, Any]] = []
    if details and compose_path:
        safe_services = [name for name in _select_services(inventory, services) if SAFE_SERVICE_RE.match(name)]
        logs = runner.run(["docker", "compose", "-f", str(compose_path), "logs", "--no-color", "--tail", "80", "--since", since] + safe_services[:8], timeout=10)
        if logs.category == "ok":
            log_signatures = extract_log_signatures(logs.stdout)
            for signature in log_signatures[:20]:
                observations.append(make_observation(component=signature.get("component", "logs"), component_role=signature.get("component_role", "unknown"), layer="logs", source_type="bounded_logs", source_reference="docker compose logs", check_name="log_signature", status="warn", summary=signature["summary"], sanitized_detail=signature.get("excerpt", ""), error_signature=signature.get("signature"), evidence_strength="observed"))
        else:
            observations.append(make_observation(component="logs", component_role="logs", layer="logs", source_type="command", source_reference="docker compose logs", check_name="bounded_logs", status="skipped", summary="Bounded logs could not be collected.", sanitized_detail=logs.stderr or logs.stdout))
    return {
        "schema_version": DIFY_INSPECT_SCHEMA_VERSION,
        "timestamp": utc_now(),
        "inferdoctor_version": __version__,
        "status": _overall_status(observations, "HEALTHY"),
        "compose_file": str(compose_path) if compose_path else None,
        "project_name": project_name,
        "service_inventory": inventory,
        "container_states": docker_ps,
        "log_signatures": log_signatures,
        "findings": observations,
        "limitations": ["Inspection is read-only and bounded.", "Environment values and full Docker inspect payloads are not collected."],
    }


def _host_endpoint_probe(endpoint: str, *, role: str, allow_non_local: bool, allow_public: bool, path: Optional[str]) -> List[Dict[str, Any]]:
    safety = classify_endpoint(endpoint)
    if safety.category == "invalid":
        return [make_observation(component="model_endpoint", component_role=role, layer="host", source_type="endpoint", source_reference=sanitize_url(endpoint), check_name="url_parse", status="fail", summary="Endpoint URL is malformed.", sanitized_detail="; ".join(safety.warnings))]
    if safety.category == "private" and not allow_non_local:
        return [make_observation(component="model_endpoint", component_role=role, layer="host", source_type="endpoint", source_reference=safety.sanitized_endpoint, check_name="endpoint_policy", status="fail", summary="LAN/private endpoint requires --allow-non-local before a live probe.")]
    if safety.category == "public" and not allow_public:
        return [make_observation(component="model_endpoint", component_role=role, layer="host", source_type="endpoint", source_reference=safety.sanitized_endpoint, check_name="endpoint_policy", status="fail", summary="Public endpoint requires --allow-public before a live probe.")]
    parsed = urllib.parse.urlsplit(endpoint)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    observations: List[Dict[str, Any]] = []
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="dns", source_reference=safety.sanitized_endpoint, check_name="dns", status="pass", summary="Host DNS resolution succeeded.", sanitized_detail=str(addresses[0][4][0]), confidence="high"))
    except OSError as exc:
        return [make_observation(component="model_endpoint", component_role=role, layer="host", source_type="dns", source_reference=safety.sanitized_endpoint, check_name="dns", status="fail", summary="Host DNS resolution failed.", sanitized_detail=str(exc), error_signature="dns_failure")]
    family = socket.AF_INET6 if ":" in addresses[0][4][0] else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        sock.connect(addresses[0][4])
        observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="tcp", source_reference=safety.sanitized_endpoint, check_name="tcp", status="pass", summary="Host TCP connection succeeded."))
    except OSError as exc:
        observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="tcp", source_reference=safety.sanitized_endpoint, check_name="tcp", status="fail", summary="Host TCP connection failed.", sanitized_detail=str(exc), error_signature="connection_failed"))
        return observations
    finally:
        sock.close()
    route = path or "/models"
    url = endpoint.rstrip("/") + "/" + route.lstrip("/")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET", headers={"User-Agent": f"InferDoctor/{__version__}"}), timeout=3) as response:
            observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="http", source_reference=sanitize_url(url), check_name="http_route", status="pass", summary=f"Host HTTP route returned {response.status}."))
    except urllib.error.HTTPError as exc:
        signature = "auth_failure" if exc.code in {401, 403} else "route_not_found" if exc.code == 404 else "http_error"
        observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="http", source_reference=sanitize_url(url), check_name="http_route", status="warn" if exc.code in {401, 403, 404} else "fail", summary=f"Host HTTP route returned {exc.code}.", error_signature=signature))
    except Exception as exc:
        observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="http", source_reference=sanitize_url(url), check_name="http_route", status="warn", summary="Host HTTP route was inconclusive after TCP succeeded.", sanitized_detail=str(exc)))
    return observations


def _container_endpoint_probe(compose_path: Path, service: str, endpoint: str, *, runner: BoundedCommandRunner) -> Dict[str, Any]:
    if not SAFE_SERVICE_RE.match(service):
        return make_observation(component=service, component_role="container_direct", layer="container", source_type="compose", source_reference=str(compose_path), check_name="container_probe", status="skipped", summary="Service name is not safe for docker compose exec probing.")
    code = "import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=3).read(1); print('ok')"
    result = runner.run(["docker", "compose", "-f", str(compose_path), "exec", "-T", service, "python", "-c", code, endpoint.rstrip("/") + "/models"], timeout=6, max_bytes=8192)
    if result.category == "ok":
        return make_observation(component=service, component_role="container_direct", layer="container", source_type="docker_exec", source_reference=service, check_name="container_model_probe", status="pass", summary="Container-direct model probe succeeded.")
    return make_observation(component=service, component_role="container_direct", layer="container", source_type="docker_exec", source_reference=service, check_name="container_model_probe", status="warn", summary="Container-direct model probe did not succeed or no safe client was available.", sanitized_detail=result.stderr or result.stdout, error_signature=result.category)


def _layer_summary(observations: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    order = {"fail": 4, "warn": 3, "unknown": 2, "skipped": 1, "pass": 0}
    summary: Dict[str, str] = {}
    for obs in observations:
        layer = str(obs.get("layer", "unknown"))
        status = str(obs.get("status", "unknown"))
        if layer not in summary or order.get(status, 2) > order.get(summary[layer], 2):
            summary[layer] = status
    return summary


def diagnose_root_causes(observations: Sequence[Dict[str, Any]], *, role: str = "chat") -> List[Dict[str, Any]]:
    layers = _layer_summary(observations)
    signatures = {obs.get("error_signature") for obs in observations if obs.get("error_signature")}
    texts = "\n".join(str(obs.get("summary", "")) + " " + str(obs.get("sanitized_detail", "")) for obs in observations).lower()
    candidates: List[Dict[str, Any]] = []

    def add(candidate: str, confidence: str, evidence: str, next_check: str, proven: bool = False) -> None:
        candidates.append({"candidate": candidate, "confidence": confidence, "evidence": evidence, "downstream_symptoms": [], "safe_next_check": next_check, "repair_risk": "low" if not proven else "medium", "verification_command": next_check, "proven": proven})

    if layers.get("host") == "pass" and layers.get("container") in {"fail", "warn"}:
        add("Docker-network or container addressing problem", "high", "Host path passed while container path failed or was inconclusive.", "inferdoctor dify connectivity check --compose-file ./docker-compose.yaml --endpoint <url> --details")
    if layers.get("host") == "fail":
        add("Endpoint unavailable before Dify-specific layers", "medium", "Host direct path failed; do not blame Docker or Dify first.", "Check the model endpoint from the host with the same base URL.")
    if "route_not_found" in signatures:
        add("Base URL or path problem", "high", "DNS/TCP worked but HTTP route returned 404.", "Try the correct /v1 base URL or provider-specific route.")
    if "auth_failure" in signatures:
        add("Authentication, Provider, or SSRF policy issue", "medium", "HTTP 401/403 was observed.", "Verify the key, Provider configuration, and SSRF policy evidence.")
    if "ssrf" in texts or "blocked address" in texts:
        add("Dify SSRF Proxy or security policy rejection", "high", "SSRF-related evidence was observed.", "inferdoctor dify selfhost inspect --compose-file ./docker-compose.yaml --services ssrf_proxy --details")
    if "plugin daemon" in texts or "plugin not found" in texts:
        add("Plugin Daemon or plugin availability problem", "high", "Plugin Daemon/plugin-not-found evidence was observed.", "inferdoctor dify selfhost inspect --compose-file ./docker-compose.yaml --services plugin_daemon --details")
    if "worker" in texts and ("indexing" in texts or "retrieval" in texts or "queue" in texts):
        add("Worker or indexing pipeline problem", "medium", "Worker/indexing evidence appeared before retrieval symptoms.", "inferdoctor dify selfhost inspect --compose-file ./docker-compose.yaml --services worker --details")
    if "migration" in texts or "version mismatch" in texts or "mixed version" in texts:
        add("Upgrade or version drift risk", "medium", "Migration or version-drift evidence was observed.", "Collect an evidence bundle before changing versions or rolling back.")
    if role == "embedding" and layers.get("host") == "pass":
        add("Embedding role may be unavailable even if chat works", "medium", "Role-specific check requested for embedding.", "Run a role-specific Provider test; do not treat chat success as embedding success.")
    if not candidates:
        add("Not enough evidence", "low", "No strong cross-layer pattern was detected.", "Collect a bounded evidence bundle before changing configuration.")
    return candidates[:5]


def run_dify_connectivity_check(*, endpoint: Optional[str], runtime: str = "auto", role: str = "chat", compose_file: Optional[str] = None, project_directory: Optional[str] = None, project_name: Optional[str] = None, services: Optional[Sequence[str]] = None, path: Optional[str] = None, through_dify: bool = False, app_api_base: Optional[str] = None, app_key_env: str = "DIFY_APP_API_KEY", allow_non_local: bool = False, allow_public: bool = False, details: bool = False, runner: Optional[BoundedCommandRunner] = None) -> Dict[str, Any]:
    runner = runner or BoundedCommandRunner()
    compose_path, observations = resolve_compose_file(compose_file, project_directory)
    observations = list(observations)
    inventory, _warnings = service_inventory(compose_path)
    target = endpoint or app_api_base
    if not target:
        observations.append(make_observation(component="model_endpoint", component_role=role, layer="host", source_type="cli", source_reference="--endpoint", check_name="endpoint_required", status="fail", summary="No model endpoint was supplied.", next_checks=["Pass --endpoint URL for the model provider path you want to diagnose."]))
    else:
        observations.extend(_host_endpoint_probe(target, role=role, allow_non_local=allow_non_local, allow_public=allow_public, path=path))
        if compose_path and details:
            for service in _select_services(inventory, services)[:3]:
                observations.append(_container_endpoint_probe(compose_path, service, target, runner=runner))
        elif compose_path:
            observations.append(make_observation(component="containers", component_role="container_direct", layer="container", source_type="compose", source_reference=str(compose_path), check_name="container_probe", status="skipped", summary="Container-direct model probe skipped; pass --details to attempt bounded read-only probes from detected services.", evidence_strength="unknown"))
    if through_dify:
        if not app_api_base:
            observations.append(make_observation(component="dify_app", component_role="app_api", layer="dify_provider", source_type="cli", source_reference="--app-api-base", check_name="dify_mediated", status="fail", summary="--through-dify requires --app-api-base."))
        elif not os.environ.get(app_key_env):
            observations.append(make_observation(component="dify_app", component_role="app_api", layer="dify_provider", source_type="environment", source_reference=app_key_env, check_name="dify_mediated", status="warn", summary="Dify-mediated path skipped because the app API key environment variable is missing."))
        else:
            observations.append(make_observation(component="dify_app", component_role="app_api", layer="dify_provider", source_type="app_api", source_reference=sanitize_url(app_api_base), check_name="dify_mediated", status="unknown", summary="Dify-mediated provider path should be verified with the existing Dify smoke command.", next_checks=["inferdoctor dify smoke --base-url <app-api-base> --allow-non-local"]))
    status = "FAIL" if any(obs.get("status") == "fail" for obs in observations) else "WARN" if any(obs.get("status") in {"warn", "unknown", "skipped"} for obs in observations) else "PASS"
    return {"schema_version": DIFY_CONNECTIVITY_SCHEMA_VERSION, "timestamp": utc_now(), "inferdoctor_version": __version__, "status": status, "endpoint": sanitize_url(target or ""), "runtime": runtime, "role": role, "compose_file": str(compose_path) if compose_path else None, "layers": _layer_summary(observations), "findings": observations, "root_cause_candidates": diagnose_root_causes(observations, role=role), "limitations": ["Direct-network success is not the same as Dify Provider success.", "Container and Dify-mediated probes are bounded and opt-in."]}


def run_dify_evidence_collect(*, compose_file: Optional[str] = None, project_directory: Optional[str] = None, project_name: Optional[str] = None, since: str = "10m", services: Optional[Sequence[str]] = None, details: bool = False, runner: Optional[BoundedCommandRunner] = None) -> Dict[str, Any]:
    runner = runner or BoundedCommandRunner()
    preflight = run_dify_selfhost_preflight(compose_file=compose_file, project_directory=project_directory, project_name=project_name, runner=runner)
    inspect = run_dify_selfhost_inspect(compose_file=compose_file, project_directory=project_directory, project_name=project_name, since=since, services=services, details=details, runner=runner)
    evidence = list(preflight.get("findings", [])) + list(inspect.get("findings", []))
    for item in evidence:
        validate_evidence(item)
    return {"schema_version": DIFY_EVIDENCE_BUNDLE_SCHEMA_VERSION, "timestamp": utc_now(), "inferdoctor_version": __version__, "compose_file": preflight.get("compose_file"), "project_identifier": project_name or Path(preflight.get("project_directory") or ".").name, "deployment_shape": {"service_count": len(preflight.get("service_inventory", [])), "roles": sorted({str(item.get("role")) for item in preflight.get("service_inventory", [])})}, "evidence": evidence, "redaction_applied": True, "limitations": ["Bundle is bounded and redacted.", "It does not include complete logs, environment values, database rows, or private documents."]}


def run_dify_evidence_explain(bundle_or_path: str | Dict[str, Any]) -> Dict[str, Any]:
    bundle = bundle_or_path if isinstance(bundle_or_path, dict) else json.loads(Path(bundle_or_path).read_text(encoding="utf-8"))
    evidence = bundle.get("evidence", []) if isinstance(bundle, dict) else []
    for item in evidence:
        validate_evidence(item)
    components: Dict[str, int] = {}
    signatures: Dict[str, int] = {}
    for item in evidence:
        key = str(item.get("component_role") or item.get("component"))
        components[key] = components.get(key, 0) + 1
        if item.get("error_signature"):
            sig = str(item["error_signature"])
            signatures[sig] = signatures.get(sig, 0) + 1
    return {"schema_version": DIFY_EVIDENCE_EXPLAIN_SCHEMA_VERSION, "timestamp": utc_now(), "inferdoctor_version": __version__, "source_schema_version": bundle.get("schema_version"), "evidence_count": len(evidence), "component_counts": components, "error_signature_counts": signatures, "root_cause_candidates": diagnose_root_causes(evidence), "first_failures": [item for item in evidence if item.get("status") == "fail"][:5], "limitations": ["This is correlation guidance, not proof of root cause unless marked proven."]}


def render_reliability_report(report: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return json.dumps(report, indent=2, sort_keys=True)
    if output_format == "markdown":
        return _render_markdown(report)
    return _render_console(report)


def _title_for_schema(schema: str) -> str:
    if "preflight" in schema:
        return "Dify Self-Host Preflight"
    if "inspect" in schema:
        return "Dify Self-Host Inspect"
    if "connectivity" in schema:
        return "Dify Model Connectivity Doctor"
    if "explain" in schema:
        return "Dify Evidence Explanation"
    if "bundle" in schema:
        return "Dify Evidence Bundle"
    return "Dify Reliability Report"


def _render_console(report: Dict[str, Any]) -> str:
    title = _title_for_schema(str(report.get("schema_version", "")))
    lines = [title, "=" * min(len(title), 72)]
    for key, label in (("status", "Status"), ("compose_file", "Compose"), ("endpoint", "Endpoint"), ("role", "Role")):
        if report.get(key):
            lines.append(f"{label}: {report.get(key)}")
    candidates = report.get("root_cause_candidates") or []
    if candidates:
        top = candidates[0]
        lines.extend(["", f"Likely blocker: {top.get('candidate')}", f"Confidence: {top.get('confidence')}", f"Next: {top.get('safe_next_check')}"])
    findings = report.get("findings") or report.get("evidence") or []
    if findings:
        lines.extend(["", "Findings:"])
        for item in findings[:12]:
            lines.append(f"- {str(item.get('status', 'unknown')).upper()} {item.get('component_role') or item.get('component')}: {item.get('summary')}")
        if len(findings) > 12:
            lines.append(f"- ... {len(findings) - 12} more findings omitted; use --format json --output report.json for details.")
    if report.get("limitations"):
        lines.extend(["", "Limits: " + " ".join(str(item) for item in report.get("limitations", [])[:2])])
    return "\n".join(lines)


def _render_markdown(report: Dict[str, Any]) -> str:
    title = _title_for_schema(str(report.get("schema_version", "")))
    lines = [f"# {title}", "", f"- Schema: `{report.get('schema_version')}`", f"- Generated: `{report.get('timestamp')}`", f"- InferDoctor: `{report.get('inferdoctor_version')}`"]
    if report.get("status"):
        lines.append(f"- Status: **{report.get('status')}**")
    if report.get("compose_file"):
        lines.append(f"- Compose: `{report.get('compose_file')}`")
    if report.get("endpoint"):
        lines.append(f"- Endpoint: `{report.get('endpoint')}`")
    candidates = report.get("root_cause_candidates") or []
    if candidates:
        lines.extend(["", "## Root-Cause Candidates", ""])
        for item in candidates:
            lines.append(f"- **{item.get('candidate')}** ({item.get('confidence')}): {item.get('evidence')} Next: `{item.get('safe_next_check')}`")
    findings = report.get("findings") or report.get("evidence") or []
    if findings:
        lines.extend(["", "## Findings", ""])
        for item in findings[:40]:
            lines.append(f"- `{item.get('status')}` `{item.get('component_role')}`: {item.get('summary')}")
    if report.get("limitations"):
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines)


def write_report_if_requested(rendered: str, output: Optional[str]) -> None:
    if not output:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent)) as handle:
        handle.write(rendered)
        tmp = Path(handle.name)
    tmp.replace(path)
