from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.txt"
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "local-model"


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
    config = read_config(ROOT / "config.yaml")
    return {
        "base_url": os.environ.get("LOCAL_AI_BASE_URL") or config.get("endpoint") or DEFAULT_ENDPOINT,
        "model": os.environ.get("LOCAL_AI_MODEL") or config.get("model") or DEFAULT_MODEL,
        "timeout": as_int(os.environ.get("LOCAL_AI_TIMEOUT") or config.get("timeout_seconds"), 30),
        "streaming": as_bool(os.environ.get("LOCAL_AI_STREAMING") or config.get("streaming"), True),
        "top_k": as_int(os.environ.get("LOCAL_AI_TOP_K") or config.get("top_k"), 4),
        "context_budget": as_int(os.environ.get("LOCAL_AI_CONTEXT_BUDGET") or config.get("context_budget"), 4000),
        "show_progress": as_bool(os.environ.get("LOCAL_AI_SHOW_PROGRESS") or config.get("show_progress"), True),
    }


def print_settings(config: dict[str, object]) -> None:
    print("Endpoint: {0}".format(config["base_url"]))
    print("Model: {0}".format(config["model"]))
    print("Streaming: {0}".format("enabled" if config["streaming"] else "disabled"))
    print("top_k: {0}".format(config["top_k"]))
    print("Context budget: {0} chars".format(config["context_budget"]))
    print("Show progress: {0}".format("yes" if config["show_progress"] else "no"))


def tokens(text: str) -> set[str]:
    return {part.lower().strip(".,?!:;()[]{}") for part in text.split() if len(part) > 2}


def load_chunks() -> list[str]:
    if not INDEX.exists():
        docs = sorted((ROOT / "docs").glob("*.md"))
        return [path.read_text(encoding="utf-8") for path in docs]
    return [item for item in INDEX.read_text(encoding="utf-8").split("\\n\\n---\\n\\n") if item.strip()]


def retrieve(question: str, config: dict[str, object]) -> list[str]:
    query_terms = tokens(question)
    ranked = []
    for chunk in load_chunks():
        score = sum(1 for term in query_terms if term in chunk.lower())
        ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for score, chunk in ranked[: int(config["top_k"])] if score > 0]
    if not selected and ranked:
        selected = [ranked[0][1]]
    return selected


def chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/chat/completions" if base.endswith("/v1") else base + "/v1/chat/completions"


def generate(question: str, context: str, config: dict[str, object]) -> str:
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": "Answer from the provided local context. If context is insufficient, say so."},
            {"role": "user", "content": "Context:\\n{0}\\n\\nQuestion: {1}".format(context, question)},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": bool(config["streaming"]),
    }
    request = urllib.request.Request(chat_url(str(config["base_url"])), data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=int(config["timeout"])) as response:
            body = response.read().decode("utf-8")
            return body[:800]
    except urllib.error.URLError as exc:
        return "Endpoint call failed: {0}".format(exc.reason)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query local Markdown context with optional OpenAI-compatible generation.")
    parser.add_argument("question", nargs="?", default="What is InferDoctor?")
    parser.add_argument("--check-config", action="store_true", help="Print config without calling an endpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show retrieval progress without calling an endpoint")
    parser.add_argument("--generate", action="store_true", help="Call the configured endpoint after retrieval")
    args = parser.parse_args()
    config = settings()
    if args.check_config:
        print_settings(config)
        print("No endpoint call was made.")
        return
    if config["show_progress"]:
        print("Searching local documents...")
    selected = retrieve(args.question, config)
    context = "\\n\\n---\\n\\n".join(selected)[: int(config["context_budget"])]
    print("Retrieval complete: selected {0} local context chunk(s).".format(len(selected)))
    if args.dry_run:
        print("Dry run: no endpoint call was made.")
        print(context[:500])
        return
    if not args.generate:
        print("Use --generate to call the configured endpoint.")
        return
    print(generate(args.question, context, config))


if __name__ == "__main__":
    main()
