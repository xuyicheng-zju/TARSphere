"""Saved trajectory scoring must use the task and model that produced it."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from tarsphere.cli.evaluate import (
    ModelEndpoint,
    build_parser,
    load_episodes,
    main,
    validate_plan_reuse,
)
from tarsphere.cli.score import validate_saved_plan
from tarsphere.configs.secrets import EvaluationSupportConfig
from tarsphere.harness.harness_cli import agent_harness_config_from_args


def test_saved_plan_rejects_changed_task_model_and_harness(capsys):
    args = build_parser().parse_args(
        ["--model", "test-model", "--api-base", "https://example.org/v1"]
    )
    main(args)
    plan = json.loads(capsys.readouterr().out)
    episodes = load_episodes(args.episodes)
    model = ModelEndpoint("agent", args.model, args.api_base, args.api_key_env)
    config = agent_harness_config_from_args(args)
    judge = EvaluationSupportConfig(
        api_key="unused",
        base_url=plan["evaluation_support_binding"]["base_url"],
        model=plan["evaluation_support_binding"]["model"],
        timeout_seconds=plan["evaluation_support_binding"]["timeout_seconds"],
    )
    validate_saved_plan(
        plan,
        episodes=episodes,
        model=model,
        harness_config=config,
        judge_config=judge,
    )
    with pytest.raises(ValueError, match="episode definitions"):
        validate_saved_plan(
            plan,
            episodes=[replace(episodes[0], sha256="changed")],
            model=model,
            harness_config=config,
            judge_config=judge,
        )
    with pytest.raises(ValueError, match="model endpoint"):
        validate_saved_plan(
            plan,
            episodes=episodes,
            model=replace(model, model="other"),
            harness_config=config,
            judge_config=judge,
        )
    with pytest.raises(ValueError, match="model endpoint"):
        validate_saved_plan(
            plan,
            episodes=episodes,
            model=replace(model, max_tokens=2048),
            harness_config=config,
            judge_config=judge,
        )
    altered = deepcopy(plan)
    altered["agent_harness_config"] = {}
    with pytest.raises(ValueError, match="harness configuration"):
        validate_saved_plan(
            altered,
            episodes=episodes,
            model=model,
            harness_config=config,
            judge_config=judge,
        )


def test_saved_plan_requires_the_recorded_evaluation_support_binding(capsys):
    args = build_parser().parse_args(
        ["--model", "test-model", "--api-base", "https://example.org/v1"]
    )
    main(args)
    plan = json.loads(capsys.readouterr().out)
    episodes = load_episodes(args.episodes)
    model = ModelEndpoint("agent", args.model, args.api_base, args.api_key_env)
    config = agent_harness_config_from_args(args)
    saved = plan["evaluation_support_binding"]
    saved["base_url"] = "https://judge.example/v1"
    judge = EvaluationSupportConfig(
        api_key="unused",
        base_url=f"{saved['base_url']}/",
        model=saved["model"],
        timeout_seconds=saved["timeout_seconds"],
    )
    validate_saved_plan(
        plan,
        episodes=episodes,
        model=model,
        harness_config=config,
        judge_config=judge,
    )

    missing = deepcopy(plan)
    missing.pop("evaluation_support_binding")
    with pytest.raises(ValueError, match="no evaluation support binding"):
        validate_saved_plan(
            missing,
            episodes=episodes,
            model=model,
            harness_config=config,
            judge_config=judge,
        )

    changed = replace(judge, model="different-judge")
    with pytest.raises(ValueError, match="different evaluation support service"):
        validate_saved_plan(
            plan,
            episodes=episodes,
            model=model,
            harness_config=config,
            judge_config=changed,
        )

    changed_plan = deepcopy(plan)
    changed_plan["evaluation_support_binding"]["model"] = "different-judge"
    with pytest.raises(ValueError, match="evaluation_support_binding"):
        validate_plan_reuse(plan, changed_plan)
