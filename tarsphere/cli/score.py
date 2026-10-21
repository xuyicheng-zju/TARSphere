#!/usr/bin/env python3
"""Score saved trajectories without launching Isaac Sim."""

from __future__ import annotations

from collections import Counter
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tarsphere.cli.evaluate import (
    ModelEndpoint,
    _check_chat_inference,
    build_jobs,
    build_parser,
    evaluation_support_binding,
    _validated_result_state,
    load_episodes,
    result_path,
    retry_pending_evaluations,
)


def validate_saved_plan(plan, *, episodes, model, harness_config, judge_config):
    """Reject scoring with inputs or a semantic evaluator not saved in the plan."""
    if plan.get("schema_version") != "tarsphere.evaluation_plan.v7":
        raise ValueError("Offline evaluation requires a current physical-run plan")
    if plan.get("agent_harness_config") != harness_config.as_dict():
        raise ValueError("Output plan belongs to a different harness configuration")
    binding = dict(
        name=model.name,
        model=model.model,
        api_base=model.api_base,
        api_key_env=model.api_key_env,
        api_key_fallback_env=model.api_key_fallback_env,
        api_key_fallback_2_env=model.api_key_fallback_2_env,
        max_tokens=model.max_tokens,
    )
    if plan.get("selected_model_bindings") != [binding]:
        raise ValueError("Output plan belongs to a different model endpoint")
    saved_judge = plan.get("evaluation_support_binding")
    if not isinstance(saved_judge, dict):
        raise ValueError("Output plan has no evaluation support binding")
    try:
        saved_binding = evaluation_support_binding(saved_judge)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Output plan has an invalid evaluation support binding"
        ) from exc
    if saved_binding != evaluation_support_binding(judge_config):
        raise ValueError(
            "Output plan belongs to a different evaluation support service"
        )
    expected = [
        dict(episode_id=row.episode_id, path=row.path, sha256=row.sha256)
        for row in episodes
    ]
    if plan.get("episode_bindings") != expected:
        raise ValueError("Output plan has different episode definitions or selection")


def result_inventory(output, jobs):
    """Count every expected row; missing physics must never look evaluated."""

    rows = []
    for job in jobs:
        for scene in job.scene_runs:
            for model in scene.models:
                for row in scene.episodes:
                    rows.append(
                        dict(
                            episode_id=row.episode_id,
                            model=model.name,
                            status=_validated_result_state(
                                result_path(output, scene, model, row),
                                row,
                                scene=scene,
                                model=model,
                            ),
                        )
                    )
    counts = Counter(r["status"] for r in rows)
    return dict(
        expected=len(rows),
        counts={
            k: counts[k]
            for k in ("physical_pending", "evaluation_pending", "evaluated")
        },
        incomplete=[r for r in rows if r["status"] != "evaluated"],
    )


def main() -> None:
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Score saved trajectories using the support API; default only inspects files.",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    from tarsphere.infra.process_lifecycle import control_directory, exclusive_lock
    from tarsphere.harness.harness_cli import agent_harness_config_from_args
    from tarsphere.evaluation.judge import JudgeConfig

    control = control_directory(output) / "host0"
    plan_path = control / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing physical-run plan: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    episodes = load_episodes(args.episodes, args.episode_id)
    model = ModelEndpoint(
        name="agent",
        model=args.model,
        api_base=args.api_base.rstrip("/"),
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
    )
    jobs = build_jobs([model], episodes)
    support = JudgeConfig.from_environment()
    validate_saved_plan(
        plan,
        episodes=episodes,
        model=model,
        harness_config=agent_harness_config_from_args(args),
        judge_config=support,
    )
    inventory = result_inventory(output, jobs)
    print(json.dumps(inventory, indent=2), flush=True)
    if not args.execute:
        return
    if not inventory["counts"]["evaluation_pending"]:
        if inventory["counts"]["physical_pending"]:
            print(
                "[OFFLINE EVALUATION] Physical results are still pending; "
                "no saved trajectories are ready to score yet.",
                file=sys.stderr,
            )
            raise SystemExit(75)
        print(
            "[OFFLINE EVALUATION] all expected rows already evaluated; no calls made.",
            flush=True,
        )
        return
    if not support.api_key.strip() or not support.base_url.strip():
        raise ValueError(
            "Offline evaluation requires TARSPHERE_EVALUATION_SUPPORT_API_KEY "
            "and TARSPHERE_EVALUATION_SUPPORT_API_BASE."
        )
    # One scorer owns result updates. A second invocation must not duplicate
    # paid calls or race the atomic result/attempt-history writes.
    with (
        exclusive_lock(control / "results.lock"),
        (control / ".offline_evaluation.lock").open("a") as lock,
    ):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another offline evaluator owns this output directory"
            ) from error
        before = result_inventory(output, jobs)
        from tarsphere.cli.evaluate import atomic_json

        atomic_json(control / "offline_status.json", before)
        _check_chat_inference(
            name="evaluation support service",
            model=support.model,
            api_base=support.base_url,
            api_key=support.api_key,
        )
        retry_pending_evaluations(output, jobs)
        after = result_inventory(output, jobs)
        atomic_json(control / "offline_status.json", after)
    print(
        f"[OFFLINE EVALUATION] evaluated={after['counts']['evaluated']} "
        f"expected={after['expected']} model={support.model}",
        flush=True,
    )
    if after["incomplete"]:
        raise SystemExit(
            f"{len(after['incomplete'])} expected rows remain incomplete; rerun this command "
            "after resolving the reported evaluation errors."
        )


if __name__ == "__main__":
    from tarsphere.infra.process_lifecycle import require_parent_lifetime

    try:
        require_parent_lifetime("TARSPHERE_SUPERVISOR_PID")
        main()
    except (ValueError, FileNotFoundError) as error:
        print(f"[OFFLINE EVALUATION] blocked: {error}", file=sys.stderr)
        raise SystemExit(78)
