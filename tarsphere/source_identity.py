"""Portable identity for the executable TARSphere-Bench runtime source tree."""

from __future__ import annotations

import hashlib
from pathlib import Path


def is_runtime_source_path(relative: Path) -> bool:
    """Select executable source while excluding caches and private config."""

    return (
        bool(relative.parts)
        and relative.parts[0] in {"tarsphere", "scripts"}
        and not any(
            part.startswith(".") or part == "__pycache__" for part in relative.parts
        )
        and relative.suffix not in {".pyc", ".pyo", ".log", ".pid"}
        and not relative.name.startswith(("local_secrets.", "host_env.sh"))
    )


def runtime_source_tree_sha256(repository_root: Path) -> str:
    """Hash relative paths and bytes for every executable runtime source file."""

    root = repository_root.resolve()
    paths: list[Path] = []
    for source_root in ("tarsphere", "scripts"):
        candidate = root / source_root
        if not candidate.is_dir():
            raise FileNotFoundError(f"Runtime source root is missing: {candidate}")
        paths.extend(
            path
            for path in candidate.rglob("*")
            if path.is_file() and is_runtime_source_path(path.relative_to(root))
        )

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        raw_path = (
            path.relative_to(root).as_posix().encode("utf-8", errors="surrogateescape")
        )
        data = path.read_bytes()
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


__all__ = ["is_runtime_source_path", "runtime_source_tree_sha256"]
