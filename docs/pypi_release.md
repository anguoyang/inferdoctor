# PyPI Release Workflow

InferDoctor is designed to stay lightweight on PyPI. Runtime dependencies should
remain empty unless a future dependency is small, pure Python, and clearly worth
its cost.

## Pre-release Checks

Run these from a clean checkout on the release branch or tag candidate:

```bash
python -m pip install -e ".[dev]"
pytest
inferdoctor
inferdoctor report --format markdown
inferdoctor profile --format markdown
python -m build
twine check dist/*
```

Confirm:

- `pyproject.toml` version matches the intended release tag.
- `README.md` renders correctly in `twine check`.
- The wheel contains only the `inferdoctor` package and metadata.
- No heavy AI runtime packages are listed as dependencies.
- No private paths, tokens, or machine-specific files are included.

## Build Artifacts

`python -m build` creates:

- `dist/inferdoctor-<version>.tar.gz`
- `dist/inferdoctor-<version>-py3-none-any.whl`

The `dist/` directory is ignored by git and should not be committed.

## Publishing

Do not publish from automation unless credentials and release ownership are
explicitly configured. For a manual release:

```bash
twine upload dist/*
```

Use a scoped PyPI token. Do not store PyPI tokens in the repository, examples,
issue templates, or generated reports.

## Version Policy

- Patch releases: bug fixes and documentation corrections.
- Minor releases: new checkers, CLI commands, report formats, or diagnosis
  improvements.
- Major releases: breaking CLI or report schema changes.

InferDoctor diagnostics are read-only by default. A package release must not add
automatic model downloads, automatic model invocation, GPU framework imports, or
system modification steps. Explicit bounded smoke probes must remain opt-in and
document exactly what evidence they retain.
