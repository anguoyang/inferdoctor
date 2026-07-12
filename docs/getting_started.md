# Getting Started with InferDoctor

InferDoctor helps you diagnose a local AI machine, understand a practical stack path, and create small local AI starter projects.

## Install

Install the published package from PyPI:

```bash
pip install inferdoctor
```

Alternative developer install from GitHub:

```bash
python -m pip install "git+https://github.com/anguoyang/inferdoctor.git@dev"
```

For local development:

```bash
git clone https://github.com/anguoyang/inferdoctor.git
cd inferdoctor
python -m pip install -e ".[dev]"
```

## First Command

```bash
inferdoctor
```

This shows the local AI stack health dashboard, overall health score, component status table, and Top Fixes.

## Language Option

The first i18n release localizes the health dashboard and `inferdoctor check` console summary. Other commands may still render English output.

```bash
inferdoctor --language zh
inferdoctor --language ja
inferdoctor check --language en
```

Use `language: zh`, `language: ja`, `language: en`, or `language: auto` in `inferdoctor.yaml` to set the dashboard language. Unsupported language values fail with a configuration error so mistakes are visible.

## Recommended Beginner Flow

Start with diagnosis:

```bash
inferdoctor
```

Pick a goal and get a practical recommendation:

```bash
inferdoctor quickstart customer-service --preference easiest
inferdoctor init --goal customer-service --preference easiest
inferdoctor recommend --goal customer-service --preference easiest
inferdoctor stack plan --goal customer-service
inferdoctor stack bootstrap --goal customer-service --dry-run
```

Create, validate, and smoke-test a starter project:

```bash
inferdoctor template create customer-service --output ./customer-service-demo
inferdoctor template validate ./customer-service-demo
inferdoctor template smoke-test ./customer-service-demo
```

Configure the generated project:

```bash
cd ./customer-service-demo
cp .env.example .env
# Edit .env or config.yaml to point at your local OpenAI-compatible endpoint.
python app.py --help
python app.py --dry-run
python app.py --check-config
```

Then run the app when your local endpoint is ready:

```bash
python app.py
```


## Measure, Compare, and Optimize

After the generated app can reach your local or private endpoint, create a small performance baseline before changing model, runtime, prompt, context size, or networking:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output before.json
inferdoctor perf baseline create --report before.json --name before
```

After a change, run another bounded smoke test and compare:

```bash
inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output after.json
inferdoctor perf compare before.json after.json
inferdoctor optimize plan --baseline before.json --candidate after.json --goal customer-service
```

For a LAN or private endpoint you control, add `--allow-non-local` to live `inferdoctor perf` smoke tests. Do not send private documents or customer data in smoke-test prompts.

These commands are smoke tests, not formal benchmarks. They do not download models, start runtimes, run long load tests, or evaluate answer quality.

## Endpoint Examples

Most generated examples use a local OpenAI-compatible endpoint. Common base URLs include:

| Runtime | Example base URL |
| --- | --- |
| Ollama OpenAI-compatible API | `http://127.0.0.1:11434/v1` |
| LM Studio | `http://127.0.0.1:1234/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |
| SGLang | `http://127.0.0.1:30000/v1` |
| Xinference OpenAI-compatible API | `http://127.0.0.1:9997/v1` |

## Safe Defaults

InferDoctor does not install runtimes, download models, run inference, or modify system settings by default. Template generation writes files only to the output directory you choose. Smoke tests only run safe help, dry-run, and config-check commands.
