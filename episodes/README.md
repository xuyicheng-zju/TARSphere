# Episode examples

`case/` contains twelve task specifications: navigate-and-report and fetch-and-deliver, each with Base, perceptual robustness, disruption adaptation, task update, human coordination, and intent understanding. They demonstrate the schema, causal task events, dialogue, and five-stage scoring.

A standalone navigation task is available at [examples/episodes/navigation.yaml](../examples/episodes/navigation.yaml), the evaluation launcher's default input.

```bash
python examples/inspect_episode.py examples/episodes/navigation.yaml
scripts/evaluate.sh --plan --episodes episodes/case
scripts/evaluate.sh --episodes /path/to/your/episode.yaml
```

Scenes, robot packages, animation clips, and model checkpoints are installed separately; see [assets](../assets/README.md). Schema inspection and planning require no simulator or asset downloads. Physical execution requires the resources referenced by the chosen YAML.

The launcher also accepts a directory of YAML files or a JSON path index through `--episodes`. See [evaluation](../docs/EVALUATION.md) and [task authoring](../docs/BENCHMARK_AUTHORING.md).
