from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "local-model"


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def as_bool(value: str | None, default: bool) -> bool:
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def as_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def settings() -> dict[str, object]:
    env_file = read_env(ROOT / ".env")
    config = read_config(ROOT / "config.yaml")
    base_url = os.environ.get("LOCAL_AI_BASE_URL") or env_file.get("LOCAL_AI_BASE_URL") or config.get("endpoint") or DEFAULT_ENDPOINT
    model = os.environ.get("LOCAL_AI_MODEL") or env_file.get("LOCAL_AI_MODEL") or config.get("model") or DEFAULT_MODEL
    timeout = os.environ.get("LOCAL_AI_TIMEOUT") or env_file.get("LOCAL_AI_TIMEOUT") or config.get("timeout_seconds")
    streaming = os.environ.get("LOCAL_AI_STREAMING") or env_file.get("LOCAL_AI_STREAMING") or config.get("streaming")
    context = os.environ.get("LOCAL_AI_MAX_CONTEXT_CHARS") or env_file.get("LOCAL_AI_MAX_CONTEXT_CHARS") or config.get("max_context_chars")
    warmup = os.environ.get("LOCAL_AI_WARMUP_PROMPT") or env_file.get("LOCAL_AI_WARMUP_PROMPT") or config.get("warmup_prompt") or ""
    return {
        "base_url": str(base_url).rstrip("/"),
        "model": model,
        "timeout": as_int(timeout, 30),
        "streaming": as_bool(streaming, True),
        "max_context_chars": as_int(context, 4000),
        "warmup_prompt": warmup,
    }


def print_settings(config: dict[str, object]) -> None:
    print("Endpoint: {0}".format(config["base_url"]))
    print("Model: {0}".format(config["model"]))
    print("Streaming: {0}".format("enabled" if config["streaming"] else "disabled"))
    print("Timeout: {0}s".format(config["timeout"]))
    print("Context budget: {0} chars".format(config["max_context_chars"]))
    if config["warmup_prompt"]:
        print("Warmup prompt: {0}".format(config["warmup_prompt"]))


def faq_context(config: dict[str, object]) -> str:
    text = (ROOT / "data" / "faq.md").read_text(encoding="utf-8")
    return text[: int(config["max_context_chars"])]


def models_url(base_url: str) -> str:
    return base_url + "/models" if base_url.endswith("/v1") else base_url + "/v1/models"


def chat_url(base_url: str) -> str:
    return base_url + "/chat/completions" if base_url.endswith("/v1") else base_url + "/v1/chat/completions"


def check_endpoint(config: dict[str, object]) -> str:
    request = urllib.request.Request(models_url(str(config["base_url"])), headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=int(config["timeout"])) as response:
            return "Endpoint check returned HTTP {0}.".format(response.getcode())
    except urllib.error.HTTPError as exc:
        return "Endpoint returned HTTP {0}. Check base URL, auth, and runtime logs.".format(exc.code)
    except urllib.error.URLError as exc:
        return "Endpoint check failed: {0}".format(exc.reason)


def read_stream(response) -> str:
    parts: list[str] = []
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
            content = event["choices"][0].get("delta", {}).get("content") or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, AttributeError):
            content = ""
        if content:
            print(content, end="", flush=True)
            parts.append(str(content))
    if parts:
        print()
    return "".join(parts)


def ask(message: str, config: dict[str, object]) -> str:
    context = faq_context(config)
    system_prompt = (ROOT / "instructions.md").read_text(encoding="utf-8")
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "FAQ context:\n{0}\n\nQuestion: {1}".format(context, message)},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": bool(config["streaming"]),
    }
    request = urllib.request.Request(
        chat_url(str(config["base_url"])),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(config["timeout"])) as response:
            content_type = response.headers.get("Content-Type", "")
            if config["streaming"] and "text/event-stream" in content_type:
                return read_stream(response)
            body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"]).strip()
    except urllib.error.HTTPError as exc:
        return "Endpoint returned HTTP {0}. Check model name, auth, and base URL.".format(exc.code)
    except urllib.error.URLError as exc:
        return "Could not reach endpoint: {0}".format(exc.reason)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "Endpoint response did not look like OpenAI-compatible chat completions JSON."


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Customer-service reference app for a local or private OpenAI-compatible endpoint.")
    p.add_argument("--message", help="Ask one customer-service question and exit")
    p.add_argument("--dry-run", action="store_true", help="Show a safe sample interaction without calling an endpoint")
    p.add_argument("--check-config", action="store_true", help="Print resolved config without calling an endpoint")
    p.add_argument("--check-endpoint", action="store_true", help="Call /models once to verify the configured endpoint")
    return p


def main() -> None:
    args = parser().parse_args()
    config = settings()
    if args.check_config:
        print_settings(config)
        print("No endpoint call was made.")
        return
    if args.dry_run:
        print_settings(config)
        print("Dry run: would answer a FAQ question using local context and streaming={0}.".format(config["streaming"]))
        print("No endpoint call was made.")
        return
    if args.check_endpoint:
        print_settings(config)
        print(check_endpoint(config))
        return
    message = args.message or input("Question: ")
    print("Connecting to {0} ...".format(config["base_url"]))
    answer = ask(message, config)
    if answer:
        print(answer)


if __name__ == "__main__":
    main()
