"""Local installation helpers preserve inputs and reject conflicting edits."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tarsphere.cli.runtime import PACKAGE_ROOT, runtime_issues
from tarsphere.cli.media import copy_media, media_paths


def test_default_evaluation_plans_without_gpu_or_network_services(monkeypatch, capsys):
    from tarsphere.cli.evaluate import build_parser, main

    monkeypatch.setattr(
        "tarsphere.cli.evaluate.detected_gpu_count",
        lambda: pytest.fail("Offline planning must not probe GPU hardware"),
    )
    main(build_parser().parse_args([]))
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_count"] == 1
    assert plan["episode_run_count"] == 1
    assert len(plan["episode_bindings"]) == 1
    assert len(plan["slots"]) == 1


def test_bundled_simulator_has_no_missing_local_modules():
    assert runtime_issues() == []


def test_runtime_check_rejects_incomplete_installation(tmp_path):
    import shutil

    for directory in ("simulator", "embodiment"):
        shutil.copytree(PACKAGE_ROOT / directory, tmp_path / directory)
    (tmp_path / "simulator/core/gym_env.py").unlink()
    issues = runtime_issues(tmp_path)
    assert "Missing simulator/core/gym_env.py" in issues
    config = tmp_path / "simulator/macros.py"
    config.write_text(
        config.read_text() + "\nimport tarsphere.simulator.missing_backend\n"
    )
    assert any(
        "Unresolved tarsphere.simulator.missing_backend" in issue
        for issue in runtime_issues(tmp_path)
    )


def test_robot_configs_use_the_bundled_runtime_and_asset_root(tmp_path):
    # Isolate module-level configuration caches, as deployment does at startup.
    import os

    code = """
from tarsphere.assets import configure_simulator_assets
from tarsphere.simulator.extensions.configs.robots.g1 import G1RobotCfg
from tarsphere.simulator.extensions.configs.robots.mobile_franka import MobileFrankaRobotCfg
from tarsphere.embodiment.robot_profiles import load_simulator_robot_cfg
root = configure_simulator_assets()
assert G1RobotCfg().usd_path.startswith(str(root))
assert MobileFrankaRobotCfg().usd_path.startswith(str(root))
load_simulator_robot_cfg('H1Robot')
_, gr1_camera, _, _ = load_simulator_robot_cfg('GR1Robot')
assert gr1_camera.prim_path == 'head_yaw_link/CameraLeft'
load_simulator_robot_cfg('MobileFrankaRobot')
"""
    result = subprocess.run(
        [__import__("sys").executable, "-c", code],
        env={**os.environ, "TARSPHERE_ASSETS_PATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def website(tmp_path: Path) -> Path:
    path = tmp_path / "site"
    path.mkdir()
    config = {"cases": [{"src": "assets/videos/demos/task.mp4?v=1"}]}
    (path / "media-config.js").write_text(
        "window.TARSPHERE_MEDIA = " + json.dumps(config) + ";"
    )
    (path / "index.html").write_text(
        '<video src="assets/videos/hero/task.mp4"></video>'
    )
    return path


def test_media_copy_uses_active_paths_and_preserves_existing_files(tmp_path):
    site = website(tmp_path)
    source = tmp_path / "resources"
    for relative in media_paths(site):
        path = source / "website" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"recorded video")
    assert copy_media(source, site) == 2
    assert copy_media(source, site) == 0
    target = site / "assets/videos/demos/task.mp4"
    target.write_bytes(b"local video")
    assert copy_media(source, site) == 0
    assert target.read_bytes() == b"local video"


def test_missing_media_fails_before_copying(tmp_path):
    site = website(tmp_path)
    with pytest.raises(FileNotFoundError, match="Missing 2 videos"):
        copy_media(tmp_path / "missing", site)
    assert not (site / "assets/videos").exists()


def test_activate_selects_the_declared_conda_prefix(tmp_path):
    import os

    root = Path(__file__).parents[1]
    declared = tmp_path / "declared"
    other = tmp_path / "other"
    commands = tmp_path / "commands"
    isaac = tmp_path / "isaac"
    for directory in (declared / "bin", other / "bin", commands):
        directory.mkdir(parents=True)

    declared_python = declared / "bin/python"
    declared_python.write_text("#!/usr/bin/env bash\nexit 0\n")
    declared_python.chmod(0o755)
    # Both environments resolve to the same binary. The prefix must still switch.
    (other / "bin/python").symlink_to(declared_python)
    fake_conda = commands / "conda"
    fake_conda.write_text(
        """#!/usr/bin/env bash
if [[ "$1 $2" != "shell.bash hook" ]]; then exit 2; fi
cat <<'EOF'
conda() {
  [[ "$1" == "activate" ]] || return 2
  export CONDA_PREFIX="$2"
  export PATH="$2/bin:$PATH"
}
EOF
"""
    )
    fake_conda.chmod(0o755)

    (isaac / "exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib").mkdir(
        parents=True
    )
    nvjitlink = (
        isaac
        / "exts/omni.isaac.ml_archive/pip_prebundle/nvidia/nvjitlink/lib"
        / "libnvJitLink.so.12"
    )
    nvjitlink.touch()
    (isaac / "extscache/omni.usd.libs-test/bin").mkdir(parents=True)
    (isaac / "setup_conda_env.sh").write_text(":\n")

    env = os.environ.copy()
    env.update(
        {
            "CONDA_PREFIX": str(other),
            "PATH": f"{other / 'bin'}:{commands}:/usr/bin:/bin",
            "ISAAC_SIM_PATH": str(isaac),
            "TARSPHERE_PYTHON": str(declared_python),
            "TARSPHERE_ASSETS_PATH": str(root / "assets"),
        }
    )
    env.pop("TARSPHERE_LOCAL_SCRATCH", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{root}/scripts/activate.sh" >/dev/null && '
            'printf \'%s\\n%s\\n\' "$CONDA_PREFIX" "$(command -v python)"',
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(declared), str(declared_python)]


def test_grasp_wrapper_help_runs_outside_the_checkout(tmp_path):
    import sys

    wrapper = Path(__file__).parents[1] / "tarsphere/infra/services/graspgenx.py"
    result = subprocess.run(
        [sys.executable, str(wrapper), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--checkpoint-root" in result.stdout


def test_media_rejects_path_traversal(tmp_path):
    site = website(tmp_path)
    (site / "media-config.js").write_text(
        'window.TARSPHERE_MEDIA = {"src":"assets/videos/../../secret.mp4"};'
    )
    with pytest.raises(ValueError, match="paths"):
        media_paths(site)
