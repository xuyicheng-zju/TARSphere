# Evaluation

The launcher runs supplied task specifications with the default agent. Navigate-and-report and fetch-and-deliver examples cover Base, perceptual robustness (PR), disruption adaptation (DA), task update (TU), human coordination (HC), and intent understanding (IU). Conditions change the task or environment while retaining the same agent mechanisms.

## Task outcomes

Each example scores five dependency-checked outcomes:

1. Understand the task and submit a grounded plan.
2. Observe the onsite target.
3. Complete the onsite task.
4. Observe the intended recipient.
5. Complete the report or accepted handover.

Events become effective after the required NPC actions complete and information is delivered. Task updates revise active requirements without adding a scored stage. Selected actions and undelivered messages do not count as completed physical evidence.

Success measures completion; progress aggregates stage quality. Execution quality reports path efficiency and safety. Inference records contain model decisions, calls, tokens, and time. Interrupted execution and unavailable scoring retain separate states.

## Select episode inputs

`--episodes PATH` accepts one YAML file, a directory of YAML files searched recursively, or a JSON index. The default is [examples/episodes/navigation.yaml](../examples/episodes/navigation.yaml).

```bash
scripts/evaluate.sh --plan
scripts/evaluate.sh --plan --episodes episodes/case
scripts/evaluate.sh --plan --episodes /path/to/tasks/index.json
```

JSON indexes contain an `episodes` list. Each entry requires `path`; relative paths resolve against the index directory. An optional `episode_id` must match the YAML:

```json
{
  "episodes": [
    {"path": "navigation.yaml"},
    {"path": "delivery.yaml", "episode_id": "my_delivery_task"}
  ]
}
```

Episode IDs and paths must be unique. Repeat `--episode-id ID` to select a subset from the supplied input. Planning works offline and prints the resolved schedule.

## Run on one server

Configure the physical runtime, assets, checkpoints, and model APIs as described in [installation](INSTALLATION.md), then run:

```bash
scripts/evaluate.sh --episodes examples/episodes/navigation.yaml
scripts/evaluate.sh --status
scripts/evaluate.sh --stop
```

One worker per GPU is the default; a single RTX 4090 runs episodes sequentially. Agent and support models use remote APIs, leaving the local GPU for simulation and tool services. One SAM3 service and one GraspGenX service are shared on the server. Each Isaac worker has its own stateful X-NavDP service.

The agent uses physical locomotion, measured ego grounding, contact-based attachment, online-map navigation, authored NPC behavior, memory, and the default skills. `--help` lists ordinary worker, memory, port, camera, timing, and model settings.

`--max-tokens` sets the agent's maximum output tokens per model call (default: `16384`). It is a positive integer and is recorded in the run plan. For example:

```bash
scripts/evaluate.sh --max-tokens 2048
```

A smaller output budget can truncate long responses; use a budget sufficient for the complete decision JSON.

Output defaults to `~/.local/share/tarsphere/runs`. Keep independent runs in separate directories; pass the same `--output-dir` when checking, stopping, or scoring a chosen run:

```bash
scripts/evaluate.sh --episodes /path/to/tasks --output-dir "$HOME/tarsphere-runs/task-set"
scripts/evaluate.sh --status --output-dir "$HOME/tarsphere-runs/task-set"
scripts/evaluate.sh --stop --output-dir "$HOME/tarsphere-runs/task-set"
```

## Score saved trajectories

```bash
scripts/evaluate.sh --score
scripts/evaluate.sh --score --status
```

Offline scoring reuses recorded physical evidence and can call the configured support model for semantic judgments. The run plan binds the support model ID, normalized API base URL, and timeout. Deferred scoring and run resume require the same binding; use a new output directory when changing the semantic evaluator. Keep the episode identity, policy model, scoring version, trajectory, and result status together when comparing runs.

For a custom run, pass its original episode input and selection alongside its output directory:

```bash
scripts/evaluate.sh --score --episodes /path/to/tasks \
  --output-dir "$HOME/tarsphere-runs/task-set"
```

Repeat any original `--episode-id` selections, `--model`, `--api-base`, and `--max-tokens` overrides, and harness parameters such as `--text-history-horizon`, `--visual-history-horizon`, `--action-horizon`, `--camera-views`, and `--online-mapping-hz`. Environment-provided model and endpoint settings must also match the run. Scoring verifies these bindings against the saved plan before updating results.

## Aggregate official reports

`tarsphere.bench.summarize_paper_metrics` aggregates `EpisodeMeasurement` records for one evaluated model. Mark invalid or excluded runs with `included_in_metrics=False`, assign the same nonempty `pair_id` to matched Base and condition runs, and set `policy_usage_complete=True` only when an episode has complete policy-token usage. The result contains per-cell metrics, Base-relative paired differences, and paper-table family rows when all six condition cells are present. Success and Progress in a family row are unweighted means of the six cells; execution metrics pool the valid episodes in that task family:

```python
from tarsphere.bench import summarize_paper_metrics

metrics = summarize_paper_metrics(measurements)
```

Efficiency is defined only over successful Base episodes with complete path evidence. Safety combines native scene and Human touch-pair onsets. Token cost averages each episode's input-plus-output tokens per policy decision. Conditional metrics expose their evidence counts and remain `None` when the required cohort is incomplete.
