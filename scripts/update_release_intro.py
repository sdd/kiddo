#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path


INTRO_START = "<!-- release-intro:start -->"
INTRO_END = "<!-- release-intro:end -->"
INTRO_BLOCK_RE = re.compile(
    rf"\n?{re.escape(INTRO_START)}\n.*?\n{re.escape(INTRO_END)}\n?",
    re.DOTALL,
)
FIRST_RELEASE_HEADING_RE = re.compile(r"(?m)^## \[[^\n]+\].*\n")
NEXT_H2_RE = re.compile(r"(?m)^## ")


def clean_intro_markers(section_body: str) -> str:
    section_body = INTRO_BLOCK_RE.sub("\n", section_body)
    return section_body.lstrip("\n")


def render_intro_block(intro: str) -> str:
    intro = intro.rstrip()
    if not intro:
        return ""
    return f"{INTRO_START}\n{intro}\n{INTRO_END}\n\n"


def update_changelog(changelog: str, intro: str) -> str:
    heading_match = FIRST_RELEASE_HEADING_RE.search(changelog)
    if heading_match is None:
        raise RuntimeError("Could not find a release heading in CHANGELOG.md")

    section_start = heading_match.end()
    next_heading = NEXT_H2_RE.search(changelog, section_start)
    section_end = next_heading.start() if next_heading else len(changelog)

    before = changelog[:section_start]
    section_body = changelog[section_start:section_end]
    after = changelog[section_end:]

    cleaned_body = clean_intro_markers(section_body)
    intro_block = render_intro_block(intro)

    if cleaned_body:
        new_section_body = "\n" + intro_block + cleaned_body
    else:
        new_section_body = "\n" + intro_block

    return before + new_section_body + after


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: update_release_intro.py <changelog-path> <intro-path>", file=sys.stderr)
        return 2

    changelog_path = Path(sys.argv[1])
    intro_path = Path(sys.argv[2])

    changelog = changelog_path.read_text(encoding="utf-8")
    intro = intro_path.read_text(encoding="utf-8").rstrip()

    updated = update_changelog(changelog, intro)
    changelog_path.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
