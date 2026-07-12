from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
INDEX = ROOT / "index.txt"


def chunks() -> list[str]:
    items: list[str] = []
    for path in sorted(DOCS.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            items.append("# Source: {0}\\n{1}".format(path.name, text))
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny local keyword index from Markdown files. No model endpoint is called.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed without writing index.txt")
    args = parser.parse_args()
    items = chunks()
    print("Documents found: {0}".format(len(items)))
    if args.dry_run:
        print("Dry run: no files were written and no endpoint call was made.")
        return
    INDEX.write_text("\\n\\n---\\n\\n".join(items), encoding="utf-8")
    print("Wrote {0}".format(INDEX))


if __name__ == "__main__":
    main()
