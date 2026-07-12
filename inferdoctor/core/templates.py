from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class TemplateInfo:
    name: str
    title: str
    description: str
    target_user: str
    required_stack: List[str]
    optional_stack: List[str]
    hardware_recommendation: str
    estimated_difficulty: str
    generated_files_planned: List[str]
    next_command: str


TEMPLATES: Dict[str, TemplateInfo] = {
    "customer-service": TemplateInfo(
        name="customer-service",
        title="Customer Service Chatbot",
        description="A local FAQ assistant for small support teams or product sites.",
        target_user="Teams that want a simple local support chatbot before adding RAG.",
        required_stack=["OpenAI-compatible local endpoint"],
        optional_stack=["Ollama", "vLLM", "SGLang", "Dify"],
        hardware_recommendation="CPU works for small tests; NVIDIA GPU improves latency.",
        estimated_difficulty="easy",
        generated_files_planned=["README.md", "app.py", "requirements.txt", ".env.example", "data/faq.md", "config.yaml", "prompts/system_prompt.md", "troubleshooting.md"],
        next_command="inferdoctor template create customer-service --output ./customer-service-demo",
    ),
    "restaurant-ordering": TemplateInfo(
        name="restaurant-ordering",
        title="Restaurant Ordering Assistant",
        description="A local ordering helper with menu data and clear ordering policies.",
        target_user="Restaurants or demo builders who need a concrete local chatbot scenario.",
        required_stack=["OpenAI-compatible local endpoint"],
        optional_stack=["Ollama", "LM Studio", "Open WebUI"],
        hardware_recommendation="Runs as a small demo on CPU; GPU helps interactive response time.",
        estimated_difficulty="easy",
        generated_files_planned=["README.md", "app.py", "requirements.txt", ".env.example", "data/menu.yaml", "data/policies.md", "config.yaml", "prompts/system_prompt.md", "examples/sample_orders.md", "troubleshooting.md"],
        next_command="inferdoctor template create restaurant-ordering --output ./restaurant-ordering-demo",
    ),
    "local-doc-qa": TemplateInfo(
        name="local-doc-qa",
        title="Local Document Q&A",
        description="A starter for asking questions over local Markdown documents.",
        target_user="Developers building a small document assistant before adding a vector DB.",
        required_stack=["OpenAI-compatible local endpoint"],
        optional_stack=["Dify", "Xinference", "embedding endpoint"],
        hardware_recommendation="CPU is fine for keyword fallback; GPU helps generation.",
        estimated_difficulty="medium",
        generated_files_planned=["README.md", "ingest.py", "query.py", "requirements.txt", ".env.example", "docs/sample.md", "config.yaml", "troubleshooting.md"],
        next_command="inferdoctor template create local-doc-qa --output ./local-doc-qa-demo",
    ),
    "personal-kb": TemplateInfo(
        name="personal-kb",
        title="Personal Knowledge Base",
        description="A personal notes assistant that can later grow into local RAG.",
        target_user="Individual developers organizing private notes locally.",
        required_stack=["OpenAI-compatible local endpoint"],
        optional_stack=["Dify", "Open WebUI", "embedding endpoint"],
        hardware_recommendation="CPU is enough for early notes experiments.",
        estimated_difficulty="medium",
        generated_files_planned=["README.md", "config.yaml", "notes/sample.md"],
        next_command="inferdoctor template show local-doc-qa",
    ),
    "meeting-notes": TemplateInfo(
        name="meeting-notes",
        title="Meeting Notes Assistant",
        description="A local summarization workflow for pasted meeting notes.",
        target_user="Teams that want local-first meeting note cleanup.",
        required_stack=["OpenAI-compatible local endpoint"],
        optional_stack=["Ollama", "LM Studio"],
        hardware_recommendation="Small quantized models are usually enough.",
        estimated_difficulty="easy",
        generated_files_planned=["README.md", "summarize.py", "examples/meeting.txt"],
        next_command="inferdoctor template show meeting-notes",
    ),
    "openai-compatible-api": TemplateInfo(
        name="openai-compatible-api",
        title="OpenAI-compatible API Demo",
        description="A minimal client for any local /v1/chat/completions endpoint.",
        target_user="Developers validating local API compatibility.",
        required_stack=["OpenAI-compatible local endpoint"],
        optional_stack=["vLLM", "SGLang", "LM Studio", "llama.cpp server"],
        hardware_recommendation="Depends on the serving runtime and model size.",
        estimated_difficulty="easy",
        generated_files_planned=["README.md", "client.py", ".env.example"],
        next_command="inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
    ),
    "ollama-chat": TemplateInfo(
        name="ollama-chat",
        title="Ollama Chat Starter",
        description="A beginner-friendly local chat path using Ollama-style setup.",
        target_user="Beginners who want the easiest local chat workflow.",
        required_stack=["Ollama API or OpenAI-compatible endpoint"],
        optional_stack=["Open WebUI"],
        hardware_recommendation="CPU works for small models; GPU improves speed.",
        estimated_difficulty="easy",
        generated_files_planned=["README.md", "chat.py", "config.yaml"],
        next_command="inferdoctor check ollama",
    ),
    "dify-rag": TemplateInfo(
        name="dify-rag",
        title="Dify Local RAG Starter",
        description="A readiness path for Dify connected to an existing local OpenAI-compatible model endpoint.",
        target_user="Developers who want Dify as the RAG/app layer without hiding endpoint diagnostics.",
        required_stack=["Dify endpoint", "local OpenAI-compatible model endpoint"],
        optional_stack=["Ollama", "vLLM", "SGLang", "Xinference", "Docker Compose guidance"],
        hardware_recommendation="Dify can run separately; model endpoint hardware decides generation speed and capacity.",
        estimated_difficulty="medium",
        generated_files_planned=["README.md", "inferdoctor.yaml", "sample_prompt.md", "optional docker-compose.yml guidance"],
        next_command="inferdoctor template compose dify-rag --output ./dify-rag-compose",
    ),
    "open-webui": TemplateInfo(
        name="open-webui",
        title="Open WebUI Starter",
        description="A browser-chat starter path for Open WebUI connected to Ollama or any local OpenAI-compatible endpoint.",
        target_user="Beginners who want a local chat UI while keeping model backend diagnosis explicit.",
        required_stack=["Open WebUI endpoint", "local model backend"],
        optional_stack=["Ollama", "vLLM", "SGLang", "LM Studio", "Docker Compose guidance"],
        hardware_recommendation="Open WebUI itself is light; the backend model decides CPU/GPU needs.",
        estimated_difficulty="easy",
        generated_files_planned=["README.md", "inferdoctor.yaml", "optional docker-compose.yml guidance"],
        next_command="inferdoctor template compose open-webui --output ./open-webui-compose",
    ),
    "vllm-api": TemplateInfo(
        name="vllm-api",
        title="vLLM API Starter",
        description="A high-throughput local API readiness path for vLLM.",
        target_user="Developers serving OpenAI-compatible APIs on NVIDIA GPUs.",
        required_stack=["NVIDIA GPU", "vLLM OpenAI-compatible endpoint"],
        optional_stack=["Docker", "CUDA toolkit for builds"],
        hardware_recommendation="Prefer 16 GiB+ VRAM; 24 GiB+ gives more headroom.",
        estimated_difficulty="advanced",
        generated_files_planned=["README.md", "client.py", "inferdoctor.yaml"],
        next_command="inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
    ),
    "sglang-api": TemplateInfo(
        name="sglang-api",
        title="SGLang API Starter",
        description="A local API readiness path for SGLang OpenAI-compatible serving.",
        target_user="Developers testing SGLang servers and API clients.",
        required_stack=["NVIDIA GPU", "SGLang OpenAI-compatible endpoint"],
        optional_stack=["Docker", "CUDA toolkit for builds"],
        hardware_recommendation="Prefer 16 GiB+ VRAM; tune model and context to fit.",
        estimated_difficulty="advanced",
        generated_files_planned=["README.md", "client.py", "inferdoctor.yaml"],
        next_command="inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1",
    ),
}


def list_templates() -> List[TemplateInfo]:
    return [TEMPLATES[name] for name in sorted(TEMPLATES)]


def get_template(name: str) -> TemplateInfo:
    try:
        return TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(TEMPLATES))
        raise KeyError("unknown template '{0}'. Available templates: {1}".format(name, available)) from exc


def template_names() -> List[str]:
    return sorted(TEMPLATES)


def _join(values: Iterable[str]) -> str:
    return ", ".join(values) if values else "none"


def render_template_list() -> str:
    lines = [
        "InferDoctor Local AI App Templates",
        "=" * 57,
        "Pick a small starter app after your machine and endpoint look healthy.",
        "This command only shows options. It does not create files.",
        "",
    ]
    for template in list_templates():
        lines.extend([
            "{0}  ({1})".format(template.name, template.estimated_difficulty),
            "  Builds: {0}".format(template.title),
            "  Needs: {0}".format(_join(template.required_stack)),
            "  Good for: {0}".format(template.target_user),
            "",
        ])
    lines.extend([
        "Next steps:",
        "  1. Show details: inferdoctor template show customer-service",
        "  2. Create a starter: inferdoctor template create customer-service --output ./customer-service-demo",
        "  3. Configure the generated .env or config.yaml for your local endpoint.",
    ])
    return "\n".join(lines)


def render_template_detail(name: str) -> str:
    template = get_template(name)
    lines = [
        "InferDoctor Template: {0}".format(template.name),
        "=" * 57,
        template.title,
        "",
        "What it builds:",
        "  {0}".format(template.description),
        "",
        "Who it is for:",
        "  {0}".format(template.target_user),
        "",
        "Stack fit:",
        "  Required: {0}".format(_join(template.required_stack)),
        "  Optional: {0}".format(_join(template.optional_stack)),
        "  Hardware: {0}".format(template.hardware_recommendation),
        "  Difficulty: {0}".format(template.estimated_difficulty),
        "",
        "Files planned:",
    ]
    lines.extend("  - {0}".format(item) for item in template.generated_files_planned)
    lines.extend([
        "",
        "Beginner path:",
        "  1. Run inferdoctor to check the machine.",
        "  2. Create this starter project.",
        "  3. Edit .env or config.yaml to point at your local endpoint.",
        "  4. Run the generated app from its README.",
        "",
        "Create it:",
        "  {0}".format(template.next_command),
    ])
    return "\n".join(lines)



DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1"

ENDPOINT_EXAMPLES = [
    ("Ollama OpenAI-compatible", "http://127.0.0.1:11434/v1"),
    ("LM Studio", "http://127.0.0.1:1234/v1"),
    ("vLLM", "http://127.0.0.1:8000/v1"),
    ("SGLang", "http://127.0.0.1:30000/v1"),
    ("Xinference OpenAI-compatible", "http://127.0.0.1:9997/v1"),
]

CHAT_ENV_EXAMPLE = """LOCAL_AI_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_AI_MODEL=local-model
LOCAL_AI_TIMEOUT=30
LOCAL_AI_STREAMING=true
LOCAL_AI_MAX_CONTEXT_CHARS=4000
LOCAL_AI_WARMUP_PROMPT=Say hello in one short sentence.
"""

CHAT_CONFIG = """endpoint: http://127.0.0.1:8000/v1
model: local-model
timeout_seconds: 30
streaming: true
max_context_chars: 4000
warmup_prompt: Say hello in one short sentence.
show_progress: true
endpoint_health_check: true
"""

DOC_QA_ENV_EXAMPLE = """LOCAL_AI_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_AI_MODEL=local-model
LOCAL_AI_TIMEOUT=30
LOCAL_AI_STREAMING=true
LOCAL_AI_TOP_K=4
LOCAL_AI_CONTEXT_BUDGET=4000
LOCAL_AI_SHOW_PROGRESS=true
LOCAL_AI_WARMUP_PROMPT=Summarize the retrieved context in one sentence.
"""

DOC_QA_CONFIG = """endpoint: http://127.0.0.1:8000/v1
model: local-model
timeout_seconds: 30
streaming: true
retrieval: keyword
top_k: 4
context_budget: 4000
show_progress: true
endpoint_health_check: true
warmup_prompt: Summarize the retrieved context in one sentence.
"""


APP_CLIENT = r"""from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "__DEFAULT_ENDPOINT__"
DEFAULT_MODEL = "local-model"
ROOT = Path(__file__).resolve().parent


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def read_simple_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def as_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def load_settings() -> dict[str, object]:
    env_file = read_env_file(ROOT / ".env")
    config = read_simple_config(ROOT / "config.yaml")
    base_url = os.environ.get("LOCAL_AI_BASE_URL") or env_file.get("LOCAL_AI_BASE_URL") or config.get("endpoint") or DEFAULT_BASE_URL
    model = os.environ.get("LOCAL_AI_MODEL") or env_file.get("LOCAL_AI_MODEL") or config.get("model") or DEFAULT_MODEL
    timeout_text = os.environ.get("LOCAL_AI_TIMEOUT") or env_file.get("LOCAL_AI_TIMEOUT") or config.get("timeout_seconds") or config.get("timeout")
    streaming_text = os.environ.get("LOCAL_AI_STREAMING") or env_file.get("LOCAL_AI_STREAMING") or config.get("streaming")
    max_context_text = os.environ.get("LOCAL_AI_MAX_CONTEXT_CHARS") or env_file.get("LOCAL_AI_MAX_CONTEXT_CHARS") or config.get("max_context_chars")
    warmup_prompt = os.environ.get("LOCAL_AI_WARMUP_PROMPT") or env_file.get("LOCAL_AI_WARMUP_PROMPT") or config.get("warmup_prompt") or ""
    timeout_text = os.environ.get("LOCAL_AI_TIMEOUT") or env_file.get("LOCAL_AI_TIMEOUT") or config.get("timeout_seconds") or config.get("timeout")
    show_progress_text = os.environ.get("LOCAL_AI_SHOW_PROGRESS") or env_file.get("LOCAL_AI_SHOW_PROGRESS") or config.get("show_progress")
    health_check_text = os.environ.get("LOCAL_AI_ENDPOINT_HEALTH_CHECK") or env_file.get("LOCAL_AI_ENDPOINT_HEALTH_CHECK") or config.get("endpoint_health_check")
    return {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "timeout": as_int(timeout_text, 30),
        "streaming": as_bool(streaming_text, True),
        "max_context_chars": as_int(max_context_text, 4000),
        "warmup_prompt": warmup_prompt,
        "show_progress": as_bool(show_progress_text, True),
        "endpoint_health_check": as_bool(health_check_text, True),
    }


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def load_context() -> str:
__DATA_LOADER__


def load_system_prompt() -> str:
    prompt_path = ROOT / "prompts" / "system_prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "__SYSTEM_PROMPT__"


def trim_context(context: str, max_chars: int) -> str:
    if len(context) <= max_chars:
        return context
    return context[:max_chars].rstrip() + "\n\n[Context trimmed by max_context_chars]"


def extract_non_streaming_content(body: dict[str, object]) -> str:
    try:
        choices = body["choices"]  # type: ignore[index]
        return choices[0]["message"]["content"].strip()  # type: ignore[index]
    except (KeyError, IndexError, TypeError, AttributeError):
        return "Endpoint response did not look like OpenAI chat completions JSON."


def read_streaming_response(response, echo: bool = False) -> str:
    chunks: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        try:
            delta = event["choices"][0].get("delta", {}).get("content") or event["choices"][0].get("text") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            delta = ""
        if delta:
            chunks.append(delta)
            if echo:
                print(delta, end="", flush=True)
    return "" if echo else "".join(chunks).strip()


def models_url(base_url: str) -> str:
    return base_url + "/models" if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/") + "/v1/models"


def check_endpoint(settings: dict[str, object]) -> str:
    request = urllib.request.Request(str(models_url(str(settings["base_url"]))), headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=int(settings["timeout"])) as response:
            if 200 <= response.getcode() < 300:
                return "Endpoint health check passed: /models responded."
            return "Endpoint health check returned HTTP {0}.".format(response.getcode())
    except urllib.error.HTTPError as exc:
        return "Endpoint health check returned HTTP {0}. Check base URL, auth, and runtime logs.".format(exc.code)
    except urllib.error.URLError as exc:
        return "Endpoint health check failed. Check LOCAL_AI_BASE_URL, port, container networking, and whether the runtime is running. Details: {0}".format(exc.reason)


def ask_local_model(message: str, echo_stream: bool = False) -> str:
    settings = load_settings()
    context = trim_context(load_context(), int(settings["max_context_chars"]))
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": load_system_prompt() + "\n\nContext:\n" + context},
            {"role": "user", "content": message},
        ],
        "temperature": 0.2,
        "stream": bool(settings["streaming"]),
    }
    request = urllib.request.Request(
        str(settings["base_url"]) + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(settings["timeout"])) as response:
            content_type = response.headers.get("Content-Type", "")
            if bool(settings["streaming"]) and "text/event-stream" in content_type:
                streamed = read_streaming_response(response, echo=echo_stream)
                return streamed or ""
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return "Endpoint returned HTTP {0}. Check base URL, model name, and whether auth is required.".format(exc.code)
    except urllib.error.URLError as exc:
        return "Could not reach local endpoint. Check LOCAL_AI_BASE_URL or config.yaml. Details: {0}".format(exc.reason)
    except json.JSONDecodeError:
        return "Endpoint responded, but the response was not valid JSON. Verify OpenAI-compatible mode."

    return extract_non_streaming_content(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a streaming-first local OpenAI-compatible starter chat loop. No cloud API key is required by default."
    )
    parser.add_argument("--check-config", action="store_true", help="Print resolved endpoint/model/performance settings and exit without calling the model")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt, local context, and endpoint settings without calling the model")
    parser.add_argument("--check-endpoint", action="store_true", help="Call /models once to verify the configured endpoint, then exit")
    parser.add_argument("--warmup", action="store_true", help="Send the configured warmup prompt once before a demo, then exit")
    return parser


def print_settings(settings: dict[str, object]) -> None:
    print("Endpoint: {0}".format(settings["base_url"]))
    print("Model: {0}".format(settings["model"]))
    print("Timeout: {0}s".format(settings["timeout"]))
    print("Streaming: {0}".format("enabled" if settings["streaming"] else "disabled"))
    print("Max context chars: {0}".format(settings["max_context_chars"]))
    print("Show progress: {0}".format("yes" if settings["show_progress"] else "no"))
    print("Endpoint health check hint: {0}".format("enabled" if settings["endpoint_health_check"] else "disabled"))
    if settings["warmup_prompt"]:
        print("Warmup prompt: {0}".format(settings["warmup_prompt"]))


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    if args.check_config:
        print_settings(settings)
        print("No endpoint call was made.")
        return
    if args.check_endpoint:
        print_settings(settings)
        print(check_endpoint(settings))
        return
    if args.warmup:
        print_settings(settings)
        prompt = str(settings["warmup_prompt"] or "Say hello in one short sentence.")
        print("Running one explicit warmup prompt. This will call the configured endpoint.")
        print("Warmup prompt: {0}".format(prompt))
        answer = ask_local_model(prompt, "", settings, echo_stream=False)
        print("Warmup response received: {0}".format("yes" if answer else "no"))
        return
    if args.dry_run:
        print("Dry run: no endpoint call was made.")
        print_settings(settings)
        print("System prompt preview:")
        print(load_system_prompt()[:600])
        print("Context preview:")
        print(trim_context(load_context(), int(settings["max_context_chars"]))[:1000])
        print("Next: run python app.py when your local endpoint is ready.")
        print("UX tip: streaming lowers perceived latency; use inferdoctor perf streaming to check TTFT.")
        return
    print("Local AI starter configured for {0} with model '{1}'.".format(settings["base_url"], settings["model"]))
    print("Streaming is {0}. A live endpoint call happens only after you send a message.".format("enabled" if settings["streaming"] else "disabled"))
    print("Type 'exit' to quit. No cloud API key is required by default.")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        if settings["show_progress"]:
            print("Connecting to local endpoint and waiting for first generated content...")
        print("assistant> ", end="", flush=True)
        answer = ask_local_model(message, echo_stream=bool(settings["streaming"]))
        if answer:
            print(answer)
        else:
            print()


if __name__ == "__main__":
    main()
"""


def _endpoint_examples_markdown() -> str:
    return "\n".join("- {0}: `{1}`".format(name, url) for name, url in ENDPOINT_EXAMPLES)


def _help_command(command: str) -> str:
    return "python query.py --help" if "query.py" in command else "python app.py --help"


def _dry_run_command(command: str) -> str:
    return "python query.py --dry-run" if "query.py" in command else "python app.py --dry-run"


def _check_config_command(command: str) -> str:
    return "python query.py --check-config" if "query.py" in command else "python app.py --check-config"


def _base_readme(title: str, purpose: str, command: str, extra: str = "") -> str:
    return """# {title}

{purpose}

This starter is generated by InferDoctor for a local OpenAI-compatible endpoint.
It does not require a cloud API key by default and does not download or run models for you.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
inferdoctor template validate .
inferdoctor template smoke-test .
{help_command}
{dry_run_command}
{check_config_command}
{command}
```

## Expected File Tree

```text
.
├── README.md
├── config.yaml
├── .env.example
├── requirements.txt
├── troubleshooting.md
├── app.py or query.py
├── prompts/
└── data/ or docs/
```

## Configure Your Local Endpoint

Edit `.env` or `config.yaml`:

```bash
LOCAL_AI_BASE_URL=http://127.0.0.1:8000/v1
LOCAL_AI_MODEL=local-model
LOCAL_AI_TIMEOUT=30
LOCAL_AI_STREAMING=true
LOCAL_AI_MAX_CONTEXT_CHARS=4000
```

Endpoint examples:

{endpoint_examples}

Use whichever runtime you already have running. The app sends requests to `/chat/completions` under that base URL.

## Performance UX Defaults

- `streaming: true` is enabled by default because users care about time to first token (TTFT), not just total latency.
- `timeout_seconds` prevents a broken endpoint from hanging the demo.
- `max_context_chars` or `context_budget` keeps prompts from becoming too large.
- `warmup_prompt` is a safe reminder for demos; run a tiny warmup question manually before customer-facing demos.
- Use `--dry-run` and `--check-config` before making any live endpoint call.
- Use `--check-endpoint` only when you want one live `/models` check.
- Use `--warmup` only when you explicitly want to send the configured warmup prompt.

Switch streaming off by setting `LOCAL_AI_STREAMING=false` or `streaming: false` in `config.yaml`.

## Diagnose Before Running

```bash
inferdoctor
inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1
inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1
inferdoctor explain openai-compatible-connection-refused
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
inferdoctor optimize plan --report before.json --goal customer-service
# After a change, save after.json and compare:
inferdoctor perf compare before.json after.json
inferdoctor optimize endpoint --runtime vllm --vram 24 --model-size 14b
```

For a LAN or private endpoint you control, add `--allow-non-local` to live `inferdoctor perf` smoke tests. Never send private documents or secrets in smoke-test prompts.
{extra}

## Validate This Template

```bash
inferdoctor template validate .
inferdoctor template smoke-test .
```

Validation is read-only. Smoke tests only run safe help, dry-run, and config-check commands. They do not install dependencies, call endpoints, or run inference.

## Troubleshooting

See `troubleshooting.md` for common local endpoint problems and fixes.

## Safety

The template sends prompts only to the endpoint you configure. Keep it pointed at a local service if you want a local-only demo.
""".format(
        title=title,
        purpose=purpose,
        command=command,
        help_command=_help_command(command),
        dry_run_command=_dry_run_command(command),
        check_config_command=_check_config_command(command),
        endpoint_examples=_endpoint_examples_markdown(),
        extra=extra,
    )


def _chat_client_py(system_prompt: str, data_loader: str) -> str:
    return (
        APP_CLIENT.replace("__DEFAULT_ENDPOINT__", DEFAULT_ENDPOINT)
        .replace("__SYSTEM_PROMPT__", system_prompt.replace('"', '\\"'))
        .replace("__DATA_LOADER__", data_loader.rstrip())
    )


def _common_troubleshooting() -> str:
    return """# Troubleshooting

## Connection refused

The local AI server is probably not running or the port is wrong.

Try:

```bash
inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1
inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1
inferdoctor check ollama --endpoint http://127.0.0.1:11434
```

## 404 or wrong base URL

Many OpenAI-compatible servers expect the base URL to include `/v1`.

Examples:

- `http://127.0.0.1:8000/v1` for vLLM
- `http://127.0.0.1:30000/v1` for SGLang
- `http://127.0.0.1:1234/v1` for LM Studio
- `http://127.0.0.1:11434/v1` for Ollama OpenAI-compatible mode

## 401 or 403

Your endpoint may require an API key. This starter does not require one by default.
If your local server requires auth, extend `app.py` to add an `Authorization` header.

## Invalid JSON or response shape

The server may not expose OpenAI-compatible chat completions. Run:

```bash
inferdoctor explain openai-compatible-invalid-json
inferdoctor explain openai-compatible-404
```

## Slow responses

Enable streaming, warm up the endpoint before demos, reduce context size, or use a smaller/quantized model.
Run:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
inferdoctor optimize plan --report before.json --goal customer-service
inferdoctor optimize endpoint --runtime vllm --vram 24 --model-size 14b
inferdoctor model fit --size 14b --quant q4 --vram 24
inferdoctor recommend --goal customer-service --vram 24
```
"""


def _customer_service_files() -> dict[str, str]:
    data_loader = '    return read_text("data/faq.md")'
    return {
        "README.md": _base_readme(
            "Customer Service Chatbot",
            "A local FAQ assistant for answering product, billing, shipping, warranty, and support questions.",
            "python app.py",
            extra="""
## Try These Prompts

- What is the return policy?
- How long does shipping take?
- Is there a warranty?
- Can support see my password?
""",
        ),
        "app.py": _chat_client_py(
            "You are a concise customer service assistant. Answer only from the FAQ context. If the answer is missing, say you do not know and suggest contacting support.",
            data_loader,
        ),
        "requirements.txt": "# Standard library only. Add packages here if you extend the demo.\n",
        ".env.example": CHAT_ENV_EXAMPLE,
        "config.yaml": CHAT_CONFIG,
        "prompts/system_prompt.md": """You are a concise customer service assistant for Acme Local Shop.

Rules:
- Answer only from the FAQ context.
- If the FAQ does not contain the answer, say you do not know.
- Do not invent policies, prices, or account information.
- Keep answers short and helpful.
- Suggest contacting human support for account-specific issues.
""",
        "data/faq.md": """# Example FAQ

## Shipping
Standard shipping usually takes 3-5 business days. Expedited shipping takes 1-2 business days when available.

## Billing
Invoices are emailed after checkout. Customers can update billing details before an order ships, but payment card numbers are never shown to support agents.

## Returns
Unused items can be returned within 30 days with the original receipt. Opened software, personalized items, and gift cards are not refundable.

## Warranty
Hardware accessories include a one-year limited warranty against manufacturing defects. The warranty does not cover accidental damage or normal wear.

## Account Help
Customers can reset their password from the sign-in page. Support cannot view, recover, or share passwords.

## Support Hours
Support is available Monday to Friday, 09:00-18:00 local time. Urgent outage reports are monitored after hours.
""",
        "troubleshooting.md": _common_troubleshooting(),
    }


def _restaurant_ordering_files() -> dict[str, str]:
    data_loader = '    return read_text("data/menu.yaml") + "\\n\\n" + read_text("data/policies.md")'
    return {
        "README.md": _base_readme(
            "Restaurant Ordering Assistant",
            "A local restaurant ordering assistant with menu data, ordering policy, examples, and local endpoint configuration.",
            "python app.py",
            extra="""
## Try These Prompts

- I want a spicy ramen and a drink.
- What vegetarian options do you have?
- Build an order for two people under 30 dollars.
- I have a soy allergy. What should I avoid?
""",
        ),
        "app.py": _chat_client_py(
            "You are a restaurant ordering assistant. Help users choose items, clarify options, ask about allergies, and follow the ordering policies. Do not invent unavailable menu items or take payment details.",
            data_loader,
        ),
        "requirements.txt": "# Standard library only. Add packages here if you extend the demo.\n",
        ".env.example": CHAT_ENV_EXAMPLE,
        "config.yaml": CHAT_CONFIG,
        "prompts/system_prompt.md": """You are a local restaurant ordering assistant.

Rules:
- Use only the menu and policies provided.
- Ask about pickup or dine-in before finalizing.
- Ask about allergies before confirming an order.
- Do not accept payment information in chat.
- End with a concise order summary.
""",
        "data/menu.yaml": """restaurant: Local Noodle House
currency: USD
items:
  - name: Classic Ramen
    price: 12
    options: [mild, spicy, extra spicy]
    allergens: [wheat, soy]
  - name: Miso Ramen
    price: 13
    options: [pork, chicken, tofu]
    allergens: [soy]
  - name: Vegetable Rice Bowl
    price: 10
    options: [regular, large]
    allergens: [sesame]
  - name: Chicken Karaage
    price: 8
    options: [regular, spicy mayo]
    allergens: [wheat, egg]
  - name: Gyoza
    price: 6
    options: [pork, vegetable]
    allergens: [wheat, soy]
  - name: Iced Tea
    price: 3
  - name: Sparkling Water
    price: 2
""",
        "data/policies.md": """# Ordering Policies

- Confirm pickup or dine-in before finalizing an order.
- Ask about spice level when ramen is selected.
- Ask about allergies before confirming the order.
- Mention that prices exclude tax.
- Do not accept payment information in chat.
- If a requested item is unavailable, suggest the closest listed menu item.
- End with a concise order summary.
""",
        "examples/sample_orders.md": """# Sample Orders

## Quick lunch
User: I want something vegetarian and not too expensive.
Assistant should suggest the Vegetable Rice Bowl and ask about size and allergies.

## Ramen order
User: I want spicy ramen and gyoza for pickup.
Assistant should confirm spice level, gyoza type, pickup, allergies, and summarize the order.
""",
        "troubleshooting.md": _common_troubleshooting(),
    }


def _local_doc_qa_files() -> dict[str, str]:
    return {
        "README.md": _base_readme(
            "Local Document Q&A",
            "A small document question-answering starter with simple Markdown ingestion and keyword retrieval fallback. It does not require a vector database.",
            "python ingest.py && python query.py --dry-run",
            extra="""
## How It Works

1. Put Markdown files in `docs/`.
2. Run `python ingest.py` to build a plain-text local index.
3. Run `python query.py` to find relevant local context.
4. Run `python query.py --dry-run` or `python query.py --check-config` before using a live endpoint.
5. Keep `streaming: true`, start with `top_k: 4`, and show retrieval progress before generation.
6. Use the printed context with your local endpoint, or extend `query.py` to call `/chat/completions`.

## RAG UX Notes

Retrieval and generation are separate phases. If retrieval takes time, show progress before the first generated token. Too many chunks increase prompt size and can make TTFT worse, so keep a context budget and tune `top_k` carefully.
""",
        ),
        "ingest.py": """from __future__ import annotations

import argparse
from pathlib import Path

DOCS = Path("docs")
INDEX = Path("index.txt")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Build a plain-text keyword index from Markdown files in docs/.")


def main() -> None:
    build_parser().parse_args()
    chunks = []
    for path in sorted(DOCS.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        chunks.append("# " + path.name + "\\n" + text.strip())
    INDEX.write_text("\\n\\n---\\n\\n".join(chunks), encoding="utf-8")
    print("Indexed {0} Markdown file(s) into {1}".format(len(chunks), INDEX))


if __name__ == "__main__":
    main()
""",
        "query.py": """from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

INDEX = Path("index.txt")
CONFIG = Path("config.yaml")
ENV_FILE = Path(".env")
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "local-model"


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def read_simple_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def as_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value or "")
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def load_settings() -> dict[str, object]:
    env_file = read_env_file(ENV_FILE)
    config = read_simple_config(CONFIG)
    base_url = os.environ.get("LOCAL_AI_BASE_URL") or env_file.get("LOCAL_AI_BASE_URL") or config.get("endpoint") or DEFAULT_BASE_URL
    model = os.environ.get("LOCAL_AI_MODEL") or env_file.get("LOCAL_AI_MODEL") or config.get("model") or DEFAULT_MODEL
    retrieval = config.get("retrieval") or "keyword"
    streaming_text = os.environ.get("LOCAL_AI_STREAMING") or env_file.get("LOCAL_AI_STREAMING") or config.get("streaming")
    top_k_text = os.environ.get("LOCAL_AI_TOP_K") or env_file.get("LOCAL_AI_TOP_K") or config.get("top_k")
    context_text = os.environ.get("LOCAL_AI_CONTEXT_BUDGET") or env_file.get("LOCAL_AI_CONTEXT_BUDGET") or config.get("context_budget")
    show_progress_text = os.environ.get("LOCAL_AI_SHOW_PROGRESS") or env_file.get("LOCAL_AI_SHOW_PROGRESS") or config.get("show_progress")
    warmup_prompt = os.environ.get("LOCAL_AI_WARMUP_PROMPT") or env_file.get("LOCAL_AI_WARMUP_PROMPT") or config.get("warmup_prompt") or ""
    timeout_text = os.environ.get("LOCAL_AI_TIMEOUT") or env_file.get("LOCAL_AI_TIMEOUT") or config.get("timeout_seconds") or config.get("timeout")
    return {
        "base_url": base_url.rstrip("/"),
        "model": model,
        "retrieval": retrieval,
        "streaming": as_bool(streaming_text, True),
        "top_k": as_int(top_k_text, 4),
        "context_budget": as_int(context_text, 4000),
        "show_progress": as_bool(show_progress_text, True),
        "warmup_prompt": warmup_prompt,
        "timeout": as_int(timeout_text, 30),
    }


def tokenize(text: str) -> set[str]:
    return {term.lower().strip(".,?!:;()[]{}") for term in text.split() if len(term) > 2}


def score(text: str, question: str) -> int:
    terms = tokenize(question)
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search the local keyword index and print the most relevant Markdown chunks. No model endpoint is called."
    )
    parser.add_argument("question", nargs="?", help="Question to search for; omit for interactive prompt")
    parser.add_argument("--check-config", action="store_true", help="Print endpoint/model/retrieval/performance settings and exit")
    parser.add_argument("--dry-run", action="store_true", help="Show a safe retrieval preview without calling a model endpoint")
    parser.add_argument("--check-endpoint", action="store_true", help="Call /models once to verify the configured endpoint, then exit")
    parser.add_argument("--warmup", action="store_true", help="Send the configured warmup prompt once, then exit")
    parser.add_argument("--generate", action="store_true", help="After retrieval, send selected context to the configured endpoint")
    return parser


def print_settings(settings: dict[str, object]) -> None:
    print("Endpoint: {0}".format(settings["base_url"]))
    print("Model: {0}".format(settings["model"]))
    print("Retrieval: {0}".format(settings["retrieval"]))
    print("Streaming: {0}".format("enabled" if settings["streaming"] else "disabled"))
    print("top_k: {0}".format(settings["top_k"]))
    print("Context budget: {0} chars".format(settings["context_budget"]))
    print("Timeout: {0}s".format(settings["timeout"]))
    print("Show progress: {0}".format("yes" if settings["show_progress"] else "no"))
    if settings["warmup_prompt"]:
        print("Warmup prompt: {0}".format(settings["warmup_prompt"]))


def load_chunks() -> list[str]:
    if not INDEX.exists():
        return []
    return [chunk for chunk in INDEX.read_text(encoding="utf-8").split("\\n\\n---\\n\\n") if chunk.strip()]


def models_url(base_url: str) -> str:
    return base_url + "/models" if base_url.rstrip("/").endswith("/v1") else base_url.rstrip("/") + "/v1/models"


def check_endpoint(settings: dict[str, object]) -> str:
    request = urllib.request.Request(str(models_url(str(settings["base_url"]))), headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=int(settings["timeout"])) as response:
            if 200 <= response.getcode() < 300:
                return "Endpoint health check passed: /models responded."
            return "Endpoint health check returned HTTP {0}.".format(response.getcode())
    except urllib.error.HTTPError as exc:
        return "Endpoint health check returned HTTP {0}. Check base URL, auth, and runtime logs.".format(exc.code)
    except urllib.error.URLError as exc:
        return "Endpoint health check failed. Check LOCAL_AI_BASE_URL, port, container networking, and whether the runtime is running. Details: {0}".format(exc.reason)


def selected_context(ranked: list[str], settings: dict[str, object]) -> str:
    joined = "\\n\\n---\\n\\n".join(ranked[: int(settings["top_k"])])
    return joined[: int(settings["context_budget"])]


def read_streaming_response(response) -> str:
    parts: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
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
            parts.append(str(content))
            print(str(content), end="", flush=True)
    return "".join(parts)


def ask_local_model(question: str, context: str, settings: dict[str, object], echo_stream: bool = True) -> str:
    payload = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": "Answer from the provided local context. If the context is insufficient, say so."},
            {"role": "user", "content": "Context:\\n{0}\\n\\nQuestion: {1}".format(context, question)},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": bool(settings["streaming"]),
    }
    request = urllib.request.Request(
        str(settings["base_url"]) + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(settings["timeout"])) as response:
            content_type = response.headers.get("Content-Type", "")
            if settings["streaming"] and "text/event-stream" in content_type:
                return read_streaming_response(response) if echo_stream else ""
            body = json.loads(response.read().decode("utf-8"))
            return str(body["choices"][0]["message"]["content"]).strip()
    except urllib.error.HTTPError as exc:
        return "Endpoint returned HTTP {0}. Check base URL, model name, and whether auth is required.".format(exc.code)
    except urllib.error.URLError as exc:
        return "Could not reach local endpoint. Details: {0}".format(exc.reason)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "Endpoint response did not look like OpenAI chat completions JSON."


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    if args.check_config:
        print_settings(settings)
        print("No endpoint call was made.")
        return
    if args.check_endpoint:
        print_settings(settings)
        print(check_endpoint(settings))
        return
    if args.warmup:
        print_settings(settings)
        prompt = str(settings["warmup_prompt"] or "Say hello in one short sentence.")
        print("Running one explicit warmup prompt. This will call the configured endpoint.")
        print("Warmup prompt: {0}".format(prompt))
        answer = ask_local_model(prompt, "", settings, echo_stream=False)
        print("Warmup response received: {0}".format("yes" if answer else "no"))
        return
    if args.dry_run:
        print("Dry run: no endpoint call was made.")
        print_settings(settings)
        if settings["show_progress"]:
            print("Retrieval progress: load index -> rank chunks -> prepare context -> generate with streaming endpoint.")
        chunks = load_chunks()
        if chunks:
            print("Indexed chunks: {0}".format(len(chunks)))
            print("Context preview:")
            print(chunks[0][: int(settings["context_budget"])][:1000])
        else:
            print("Index is missing. Run python ingest.py before real queries.")
        print("Next: run python ingest.py, then python query.py 'your question'.")
        print("UX tip: too many chunks can hurt TTFT; start with top_k=4 and show retrieval progress before generation.")
        return
    chunks = load_chunks()
    if not chunks:
        print("Run python ingest.py first.")
        return
    question = args.question or input("question> ").strip()
    ranked = sorted(chunks, key=lambda chunk: score(chunk, question), reverse=True)
    print("\\nTop local context matches:\\n")
    for index, chunk in enumerate(ranked[: int(settings["top_k"])], start=1):
        print("[{0}] score={1}".format(index, score(chunk, question)))
        print(chunk[:1200])
        print()
    if not ranked:
        print("No documents indexed.")
    if args.generate:
        context = selected_context(ranked, settings)
        if settings["show_progress"]:
            print("Retrieval complete: selected {0} local context match(es).".format(min(len(ranked), int(settings["top_k"]))))
            print("Connecting to local endpoint and waiting for first generated content...")
        print("assistant> ", end="", flush=True)
        answer = ask_local_model(question, context, settings, echo_stream=bool(settings["streaming"]))
        if answer:
            print(answer)
        else:
            print()
    else:
        print("Next: send this context to your configured local OpenAI-compatible endpoint, or run with --generate when ready.")
    print("Tip: enable streaming and show retrieval progress so users are not left waiting silently.")
    print("Tip: run inferdoctor optimize rag --top-k {0} --streaming if the app feels slow.".format(settings["top_k"]))


if __name__ == "__main__":
    main()
""",
        "requirements.txt": "# Standard library only. Add packages here if you extend the demo.\n",
        ".env.example": DOC_QA_ENV_EXAMPLE,
        "config.yaml": DOC_QA_CONFIG,
        "docs/sample.md": """# Sample Local Document

InferDoctor helps diagnose local AI stacks and guide setup steps.

Use `inferdoctor` for a health dashboard, `inferdoctor recommend` for stack guidance, and `inferdoctor template list` for starter ideas.

## Local AI Terms

An OpenAI-compatible endpoint usually exposes `/v1/models` and `/v1/chat/completions`.
Ollama, LM Studio, vLLM, SGLang, and Xinference can be used as local model backends depending on your setup.
""",
        "troubleshooting.md": _common_troubleshooting(),
    }


TEMPLATE_FILE_BUILDERS = {
    "customer-service": _customer_service_files,
    "restaurant-ordering": _restaurant_ordering_files,
    "local-doc-qa": _local_doc_qa_files,
}


def create_template_project(name: str, output_dir: str) -> list[str]:
    get_template(name)
    if name not in TEMPLATE_FILE_BUILDERS:
        available = ", ".join(sorted(TEMPLATE_FILE_BUILDERS))
        raise KeyError("template '{0}' is catalog-only for now. Creatable templates: {1}".format(name, available))
    root = Path(output_dir).expanduser()
    files = TEMPLATE_FILE_BUILDERS[name]()
    written: list[str] = []
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(str(destination))
    return written


def render_template_create_summary(name: str, output_dir: str, written: list[str]) -> str:
    run_command = "python app.py" if name in {"customer-service", "restaurant-ordering"} else "python ingest.py && python query.py \"What is an OpenAI-compatible endpoint?\""
    dry_run_command = "python app.py --dry-run" if name in {"customer-service", "restaurant-ordering"} else "python query.py --dry-run"
    check_config_command = "python app.py --check-config" if name in {"customer-service", "restaurant-ordering"} else "python query.py --check-config"
    lines = [
        "InferDoctor Starter Project Created",
        "=" * 57,
        "Template: {0}".format(name),
        "Output directory: {0}".format(output_dir),
        "",
        "Generated files:",
    ]
    lines.extend("  - {0}".format(path) for path in written)
    lines.extend([
        "",
        "What to do next:",
        "  1. cd {0}".format(output_dir),
        "  2. cp .env.example .env",
        "  3. Validate the generated project: inferdoctor template validate {0}".format(output_dir),
        "  4. Smoke-test the project without calling a model: inferdoctor template smoke-test {0}".format(output_dir),
        "  5. Edit .env or config.yaml for your local endpoint.",
        "  6. Try safe generated commands before using a live endpoint:",
        "     {0}".format(dry_run_command),
        "     {0}".format(check_config_command),
        "  7. {0}".format(run_command),
        "",
        "Endpoint examples:",
    ])
    lines.extend("  {0}: {1}".format(name, url) for name, url in ENDPOINT_EXAMPLES)
    lines.extend([
        "",
        "If the app cannot connect, diagnose the endpoint first:",
        "  inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
        "  inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1",
        "  inferdoctor explain openai-compatible-connection-refused",
    ])
    return "\n".join(lines)


COMPOSE_TEMPLATE_NAMES = ["customer-service", "restaurant-ordering", "local-doc-qa", "open-webui", "dify-rag"]


def compose_template_names() -> List[str]:
    return list(COMPOSE_TEMPLATE_NAMES)


def _compose_readme(name: str, services: str) -> str:
    return """# InferDoctor Docker Compose Starter: {name}

This directory contains optional Docker Compose guidance generated by InferDoctor.
It is a starter scaffold only.

## Safety

InferDoctor only generated these files. It did not pull images, start containers,
install runtimes, download models, call endpoints, or modify system services.

## Files

- docker-compose.yml: optional service layout to review and adapt.
- .env.example: safe placeholder settings. Copy it to .env before use.
- config.yaml: InferDoctor endpoint hints for diagnostics.
- README.md: this guide.

## Services Included

{services}

## Suggested Review Flow

    cp .env.example .env
    inferdoctor
    inferdoctor template validate .

Review docker-compose.yml manually before running Docker commands.

## Important

The generated Compose file may reference public container images as examples, but
InferDoctor does not pull or run them. Review image names, ports, volumes, and
runtime requirements for your machine before using Docker.
""".format(name=name, services=services)


def _compose_files(name: str) -> dict[str, str]:
    common_env = """LOCAL_AI_BASE_URL=http://host.docker.internal:8000/v1
LOCAL_AI_MODEL=local-model
"""
    common_config = """endpoints:
  vllm: http://127.0.0.1:8000/v1
  sglang: http://127.0.0.1:30000/v1
"""
    if name in {"customer-service", "restaurant-ordering", "local-doc-qa"}:
        service_name = name.replace("-", "_")
        command = "python app.py --check-config" if name != "local-doc-qa" else "python query.py --check-config"
        compose = """services:
  {service_name}:
    image: python:3.12-slim
    working_dir: /app
    volumes:
      - ./app:/app:ro
    env_file:
      - .env
    command: ["sh", "-lc", "{command}"]
    extra_hosts:
      - "host.docker.internal:host-gateway"
""".format(service_name=service_name, command=command)
        services = "- {0}: placeholder Python service for the generated starter app.".format(service_name)
        return {
            "README.md": _compose_readme(name, services),
            "docker-compose.yml": compose,
            ".env.example": common_env,
            "config.yaml": common_config + "template: {0}".format(name) + chr(10),
        }
    if name == "open-webui":
        compose = """services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
      - OPENAI_API_KEY=local-placeholder
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - open-webui-data:/app/backend/data

volumes:
  open-webui-data:
"""
        services = "- open-webui: optional browser UI pointed at a local OpenAI-compatible endpoint."
        return {
            "README.md": _compose_readme(name, services),
            "docker-compose.yml": compose,
            ".env.example": """OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
OPENAI_API_KEY=local-placeholder
""",
            "config.yaml": """endpoints:
  openwebui: http://127.0.0.1:3000
  vllm: http://127.0.0.1:8000/v1
""",
        }
    if name == "dify-rag":
        compose = """services:
  dify-api-placeholder:
    image: langgenius/dify-api:latest
    profiles: ["manual-review"]
    environment:
      - CONSOLE_API_URL=http://127.0.0.1:5001
      - SERVICE_API_URL=http://127.0.0.1:5001
      - OPENAI_API_BASE=http://host.docker.internal:8000/v1
    extra_hosts:
      - "host.docker.internal:host-gateway"

  dify-web-placeholder:
    image: langgenius/dify-web:latest
    profiles: ["manual-review"]
    ports:
      - "5001:3000"
"""
        services = "- dify-api-placeholder and dify-web-placeholder: review-only Dify service hints using a manual-review profile."
        return {
            "README.md": _compose_readme(name, services),
            "docker-compose.yml": compose,
            ".env.example": """DIFY_BASE_URL=http://127.0.0.1:5001
LOCAL_AI_BASE_URL=http://host.docker.internal:8000/v1
""",
            "config.yaml": """endpoints:
  dify: http://127.0.0.1:5001
  vllm: http://127.0.0.1:8000/v1
""",
        }
    raise KeyError("compose generation is not available for template '{0}'".format(name))

def create_compose_project(name: str, output_dir: str) -> list[str]:
    if name not in COMPOSE_TEMPLATE_NAMES:
        available = ", ".join(COMPOSE_TEMPLATE_NAMES)
        raise KeyError("compose generation is available for: {0}".format(available))
    root = Path(output_dir).expanduser()
    files = _compose_files(name)
    written: list[str] = []
    for relative, content in files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(str(destination))
    return written


def render_compose_create_summary(name: str, output_dir: str, written: list[str]) -> str:
    lines = [
        "InferDoctor Docker Compose Files Created",
        "=" * 57,
        "Template: {0}".format(name),
        "Output directory: {0}".format(output_dir),
        "",
        "Generated files:",
    ]
    lines.extend("  - {0}".format(path) for path in written)
    lines.extend([
        "",
        "Safety:",
        "  InferDoctor only generated files. It did not pull images, start containers, install runtimes, or call endpoints.",
        "",
        "Next steps:",
        "  1. cd {0}".format(output_dir),
        "  2. cp .env.example .env",
        "  3. Review docker-compose.yml manually.",
        "  4. Run inferdoctor to diagnose your host before starting services.",
        "",
        "Optional only after review:",
        "  docker compose config",
    ])
    return chr(10).join(lines)



def render_template_registry() -> str:
    lines = [
        "InferDoctor Template Registry",
        "=" * 57,
        "Current source: built-in templates shipped with InferDoctor.",
        "Remote/community template registries are not enabled yet.",
        "",
        "Built-in templates:",
    ]
    for template in list_templates():
        lines.append("  - {0}: {1}".format(template.name, template.title))
    lines.extend([
        "",
        "Future registry principles:",
        "  - No remote template execution by default.",
        "  - Templates should be generated locally and validated before use.",
        "  - Template metadata should declare files, runtime assumptions, and safety boundaries.",
        "  - InferDoctor should never install heavy runtimes, download models, or start services without explicit user action.",
        "",
        "Useful commands:",
        "  inferdoctor template list",
        "  inferdoctor template show customer-service",
        "  inferdoctor template validate ./customer-service-demo",
        "  inferdoctor template smoke-test ./customer-service-demo",
    ])
    return chr(10).join(lines)
