#!/usr/bin/env python3
"""
Validate BibTeX entries against the Crossref API.

Example:
    python validate_references.py \
        --bib ../references.bib \
        --output ../reference-validation-report.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import bibtexparser  # type: ignore
import requests


def normalise_title(title: str) -> str:
    """Lowercase and strip punctuation/extra whitespace for comparisons."""
    cleaned = re.sub(r"[^0-9a-zA-Z\s]", " ", title)
    return " ".join(cleaned.lower().split())


def iter_entries(bib_path: Path) -> Iterable[bibtexparser.model.Entry]:
    """Yield bib entries from the provided .bib file."""
    library = bibtexparser.parse_string(bib_path.read_text(encoding="utf-8"))
    for entry in library.entries:
        yield entry


def validate_entry(entry: bibtexparser.model.Entry) -> list[str]:
    """Validate a single entry and return report lines."""
    fields = {field.key.lower(): field.value.strip() for field in entry.fields}
    lines = [f"Entry: {entry.key}"]

    doi = fields.get("doi")
    if not doi:
        lines.append("  DOI: (missing)")
        lines.append("  Crossref status: n/a")
        return lines

    doi_clean = doi.replace("https://doi.org/", "").strip()
    url = f"https://api.crossref.org/works/{doi_clean}"
    lines.append(f"  DOI: {doi}")

    try:
        resp = requests.get(url, timeout=20)
    except requests.RequestException as exc:  # network or HTTP error
        lines.append(f"  Crossref status: error ({exc})")
        return lines

    lines.append(f"  Crossref status: {resp.status_code}")

    if resp.status_code != 200:
        return lines

    try:
        message = resp.json()["message"]
    except (ValueError, KeyError):
        lines.append("  Title match: unavailable (invalid JSON)")
        return lines

    api_title = " ".join(message.get("title", []))
    bib_title = fields.get("title", "").strip("{}")
    if api_title and bib_title:
        match = normalise_title(api_title) == normalise_title(bib_title)
        lines.append(f"  Title match: {'yes' if match else 'no'}")
        if not match:
            lines.append(f"    Bib title: {bib_title}")
            lines.append(f"    API title: {api_title}")
    else:
        lines.append("  Title match: unavailable")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate BibTeX references against Crossref."
    )
    parser.add_argument(
        "--bib",
        default="../references.bib",
        type=Path,
        help="Path to the BibTeX file (default: ../references.bib relative to script).",
    )
    parser.add_argument(
        "--output",
        default="../reference-validation-report.txt",
        type=Path,
        help="Path to write the validation report (default: ../reference-validation-report.txt).",
    )
    args = parser.parse_args()

    bib_path = args.bib.resolve()
    if not bib_path.exists():
        raise FileNotFoundError(f"BibTeX file not found: {bib_path}")

    output_lines = ["Reference validation report (Crossref)", ""]
    for entry in iter_entries(bib_path):
        output_lines.extend(validate_entry(entry))
        output_lines.append("")  # blank line between entries

    output_path = args.output.resolve()
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
