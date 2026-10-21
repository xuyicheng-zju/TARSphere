#!/usr/bin/env python3
"""Copy the active project-page videos from a local resource tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil


def media_paths(website: Path) -> list[PurePosixPath]:
    text = (website / "media-config.js").read_text(encoding="utf-8")
    config, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    paths: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value.split("?")[0].endswith(".mp4"):
            paths.add(value.split("?")[0])

    visit(config)
    html = (website / "index.html").read_text(encoding="utf-8")
    paths.update(re.findall(r'(?:src|href)="(assets/[^"?]*\.mp4)(?:\?[^" ]*)?"', html))
    result = []
    for relative in sorted(paths):
        path = PurePosixPath(relative)
        if (
            path.parts[:2] != ("assets", "videos")
            or ".." in path.parts
            or "\\" in relative
        ):
            raise ValueError("Video paths must stay inside assets/videos.")
        result.append(path)
    return result


def copy_media(source_root: Path, website: Path, *, overwrite: bool = False) -> int:
    paths = media_paths(website)
    sources = [(source_root / "website" / path).resolve() for path in paths]
    source_base = (source_root / "website").resolve()
    targets = [(website / path).resolve() for path in paths]
    if any(not path.is_relative_to(source_base) for path in sources):
        raise ValueError("Source video points outside the resource tree.")
    if any(not path.is_relative_to(website.resolve()) for path in targets):
        raise ValueError("Destination video points outside the website.")
    missing = [relative for relative, path in zip(paths, sources) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} videos; first: {missing[0]}")
    copied = 0
    for source, target in zip(sources, targets):
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--website", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        count = copy_media(args.source_root, args.website, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Copied {count} videos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
