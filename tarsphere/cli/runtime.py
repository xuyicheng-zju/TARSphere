"""Check the bundled simulation source without starting Isaac Sim."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SOURCE = (
    "simulator/core/gym_env.py",
    "simulator/core/vec_env.py",
    "simulator/core/runner.py",
    "simulator/extensions/__init__.py",
    "simulator/extensions/robots/mobile_franka.py",
    "simulator/extensions/sensors/rep_camera.py",
    "embodiment/robot_profiles.py",
    "embodiment/robots/g1.py",
)


def runtime_issues(package_root: Path = PACKAGE_ROOT) -> list[str]:
    """Report missing runtime files and unresolved local module imports.

    This is a source integrity check, not a GPU or simulator readiness check.
    Both source checkouts and installed wheels use the same package layout.
    """
    issues = [
        f"Missing {name}"
        for name in REQUIRED_SOURCE
        if not (package_root / name).is_file()
    ]
    for directory in ("simulator", "embodiment"):
        for source in sorted((package_root / directory).rglob("*.py")):
            try:
                tree = ast.parse(source.read_text(encoding="utf-8"))
            except SyntaxError as error:
                issues.append(
                    f"Invalid Python: {source.relative_to(package_root)}:{error.lineno}"
                )
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif (
                    isinstance(node, ast.ImportFrom) and not node.level and node.module
                ):
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    if not module.startswith("tarsphere."):
                        continue
                    target = package_root.joinpath(*module.split(".")[1:])
                    if not target.with_suffix(".py").is_file() and not target.is_dir():
                        issues.append(
                            f"Unresolved {module} in {source.relative_to(package_root)}"
                        )
    return sorted(set(issues))


def main() -> int:
    issues = runtime_issues()
    if issues:
        print("\n".join(issues))
        return 1
    print(
        "TARSphere simulation source is complete (GPU readiness is checked separately)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
