"""Staged assets keep working after their original trees are unavailable."""

import os
import shutil

from tarsphere.infra.simulation.local_stage import plan_tree_copy, stage_tree


def test_external_tree_aliases_share_one_local_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shared = tmp_path / "shared"
    (shared / "nested").mkdir(parents=True)
    (shared / "nested" / "asset.usd").write_text("asset payload")
    (source / "a_models").symlink_to(shared, target_is_directory=True)
    (source / "b_models").symlink_to(shared, target_is_directory=True)
    (source / "c_nested").symlink_to(shared / "nested", target_is_directory=True)
    (shared / "back_to_models").symlink_to(shared, target_is_directory=True)

    destination = tmp_path / "staged"
    plan = plan_tree_copy(source, destination)
    assert [src for kind, src, dst in plan if kind == "file"] == [
        shared / "nested" / "asset.usd"
    ]
    stats = stage_tree(source, destination, workers=2)
    assert stats.errors == []
    assert stats.files_copied == 1
    assert not (destination / "a_models").is_symlink()
    assert (destination / "b_models").is_symlink()
    assert not os.path.isabs(os.readlink(destination / "b_models"))
    assert (destination / "b_models").resolve() == destination / "a_models"
    assert (destination / "c_nested").resolve() == destination / "a_models/nested"
    assert (
        destination / "a_models/back_to_models"
    ).resolve() == destination / "a_models"
    resumed = stage_tree(source, destination, workers=2)
    assert resumed.errors == []
    assert resumed.files_copied == 0
    assert resumed.files_skipped == 1

    shutil.rmtree(source)
    shutil.rmtree(shared)
    moved = tmp_path / "moved"
    destination.rename(moved)
    assert (moved / "b_models/nested/asset.usd").read_text() == "asset payload"
    assert (moved / "c_nested/asset.usd").read_text() == "asset payload"


def test_internal_absolute_and_relative_links_point_inside_staged_tree(tmp_path):
    source = tmp_path / "source"
    (source / "models").mkdir(parents=True)
    (source / "models/asset.usd").write_text("asset payload")
    (source / "scene").mkdir()
    (source / "scene/models").symlink_to("../models", target_is_directory=True)
    (source / "absolute_models").symlink_to(source / "models", target_is_directory=True)
    (source / "absolute_file").symlink_to(source / "models/asset.usd")
    (source / "absolute_missing").symlink_to(source / "models/missing.usd")
    destination = tmp_path / "staged"
    stats = stage_tree(source, destination, workers=2)
    assert stats.errors == []
    for name in (
        "scene/models",
        "absolute_models",
        "absolute_file",
        "absolute_missing",
    ):
        assert (destination / name).is_symlink()
        assert not os.path.isabs(os.readlink(destination / name))
        assert (destination / name).resolve().is_relative_to(destination)

    shutil.rmtree(source)
    assert (destination / "scene/models/asset.usd").read_text() == "asset payload"
    assert (destination / "absolute_models/asset.usd").read_text() == "asset payload"
    assert (destination / "absolute_file").read_text() == "asset payload"
    assert not (destination / "absolute_missing").exists()
