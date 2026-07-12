from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
MARKDOWN_LINK = re.compile(r"(!?)\[([^\]]+)\]\(([^)]+)\)")


def _target_without_optional_title(raw: str) -> str:
    target = raw.strip()
    if target.startswith("<"):
        end = target.find(">")
        if end != -1:
            return target[1:end]
    return target.split()[0] if target else target


def test_pypi_readme_uses_absolute_links_only():
    text = README.read_text(encoding="utf-8")
    bad_links: list[str] = []
    for match in MARKDOWN_LINK.finditer(text):
        target = _target_without_optional_title(match.group(3))
        if target.startswith(("https://", "http://", "#", "mailto:")):
            continue
        bad_links.append(target)

    assert bad_links == []


def test_pypi_readme_japanese_link_points_to_github():
    text = README.read_text(encoding="utf-8")

    assert "[日本語クイックスタート](https://github.com/anguoyang/inferdoctor/blob/main/README.ja.md)" in text
    assert "[日本語クイックスタート](README.ja.md)" not in text
