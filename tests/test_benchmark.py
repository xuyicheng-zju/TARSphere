from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pytest

from tarsphere.bench.case_loader import BenchmarkCaseLoader
from tarsphere.bench.production_layout import foot_placement_clear, load_contact_profile

ROOT = Path(__file__).resolve().parents[1]


def test_user_episode_directory_loads_independently_of_working_directory(
    tmp_path, monkeypatch
):
    user_cases = tmp_path / "user_cases"
    user_cases.mkdir()
    loader = BenchmarkCaseLoader()
    expected = []
    for name, family in [
        ("a.yaml", "fetch_and_deliver"),
        ("z.yaml", "navigate_and_report"),
    ]:
        source = ROOT / "episodes/case" / family / "base.yaml"
        shutil.copyfile(source, user_cases / name)
        expected.append(loader.load_file(source).episode_id)

    monkeypatch.chdir(tmp_path)
    external = BenchmarkCaseLoader(str(user_cases))
    assert [episode.episode_id for episode in external] == expected
    assert external.count == 2
    assert external.load_file(user_cases / "a.yaml").episode_id == expected[0]


def test_casebook_has_twelve_current_cells():
    loader = BenchmarkCaseLoader()
    episodes = [
        loader.load_file(p) for p in sorted((ROOT / "episodes/case").rglob("*.yaml"))
    ]
    cells = {
        (e.evaluation.task_profile.mainline, e.evaluation.task_profile.capability_slice)
        for e in episodes
    }
    assert len(episodes) == len(cells) == 12
    assert len({episode.episode_id for episode in episodes}) == 12
    for episode in episodes:
        assert sum(node.kind == "scored" for node in episode.evaluation.milestones) == 5


@pytest.mark.parametrize("invalid", ["episode_id: invalid", "episode_id: ["])
def test_directory_failure_preserves_existing_cases_without_partial_append(
    tmp_path, invalid
):
    loader = BenchmarkCaseLoader()
    existing = loader.load_file(ROOT / "episodes/case/navigate_and_report/base.yaml")
    loader.add_episode(existing)
    shutil.copyfile(
        ROOT / "episodes/case/fetch_and_deliver/base.yaml", tmp_path / "a.yaml"
    )
    (tmp_path / "b.yaml").write_text(invalid)
    with pytest.raises(ValueError, match="b.yaml"):
        loader.load_directory(str(tmp_path))
    assert loader.episodes == [existing]


def test_directory_appends_all_valid_cases():
    loader = BenchmarkCaseLoader()
    assert loader.load_directory(str(ROOT / "episodes/case/navigate_and_report")) == 6
    assert loader.count == 6


def test_user_contact_profile_checks_source_freshness_and_foot_clearance(tmp_path):
    source = tmp_path / "robot_geometry.json"
    source.write_text("geometry")
    profile_path = tmp_path / "contact_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "tarsphere.g1_lower_body_profile.v1",
                "bounds_min_xy_m": [-0.1, -0.1],
                "bounds_max_xy_m": [0.1, 0.1],
                "source_hashes": {
                    source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                },
            }
        )
    )
    profile = load_contact_profile(profile_path)
    mask = np.zeros((21, 21), dtype=bool)
    meta = {"bev_bounds_min_xy": [-1.0, -1.0], "bev_span_m": 2.0, "resolution": 21}
    assert foot_placement_clear(mask, [0, 0], [0.5, 0], meta, profile=profile)
    mask[10, 10] = True
    assert not foot_placement_clear(mask, [0, 0], [0.5, 0], meta, profile=profile)
    source.write_text("changed geometry")
    with pytest.raises(ValueError, match="Stale G1 lower-body profile"):
        load_contact_profile(profile_path)
