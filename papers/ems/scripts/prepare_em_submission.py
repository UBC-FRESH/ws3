#!/usr/bin/env python3
"""
Flatten the LaTeX source tree for Editorial Manager (no subdirectories).

Creates an ``em-submission`` directory in the current working directory that
contains the modified sources and a ``em-submission.zip`` archive.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT
DEST = BASE / "em-submission"
ARCHIVE = BASE / "em-submission.zip"


def copy_flattened_sources() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True, exist_ok=True)

    # Files to copy directly.
    direct_files = [
        "references.bib",
        "highlights.txt",
        "cover-letter.pdf",
        "declarationStatement.docx",
        "paper.bbl",
        "ws3-manuscript-graphical-abstract.pdf",
        "prisma-flow-diagram.sty",
    ]
    for rel in direct_files:
        src = BASE / rel
        if src.exists():
            shutil.copy2(src, DEST / Path(rel).name)

    # Copy figures and tables into the flat directory.
    for folder in ("figs", "tables"):
        src_dir = BASE / folder
        if src_dir.exists():
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, DEST / item.name)

    # Flattened paper.tex (remove path prefixes).
    text = (BASE / "paper.tex").read_text(encoding="utf-8")
    for prefix in ("figs/", "tables/", "papers/ems/"):
        text = text.replace(prefix, "")
    (DEST / "paper.tex").write_text(text, encoding="utf-8")


def make_archive() -> None:
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(ARCHIVE.with_suffix(""), "zip", DEST)


def main() -> None:
    copy_flattened_sources()
    make_archive()
    print(f"Created flattened submission archive at {ARCHIVE}")


if __name__ == "__main__":
    main()
