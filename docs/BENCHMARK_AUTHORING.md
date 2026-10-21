# Task specifications

## Examples and schema

[episodes/case](../episodes/case) contains twelve examples covering two task families and six conditions. [examples/episodes/navigation.yaml](../examples/episodes/navigation.yaml) is the default task for the evaluation launcher.

A specification binds a scene, robot start pose, NPC identities, authorized dialogue, task profile, spatial anchors, and a causal milestone graph. The schema is `tarsphere.bench.schema.TARSphereEpisode`.

```bash
python examples/inspect_episode.py examples/episodes/navigation.yaml
scripts/evaluate.sh --plan --episodes /path/to/tasks
```

## Task design

Navigate-and-report requires onsite observation and an accurate report. Fetch-and-deliver requires physical acquisition, transport, and acceptance by the intended recipient.

Base and its five conditions share a task family and five-stage evaluation. PR changes sensing. DA changes execution availability while retaining the goal. TU revises an authorized requirement. HC requires another person's information or onsite work. IU requires clarification or contextual inference of intent.

## Authoring and validation

The `tarsphere.bench` modules provide schemas, task synthesis, geometry checks, condition construction, and certification. Build task semantics, event dependencies, and interaction positions together. Load a user-supplied specification with:

```python
from tarsphere.bench.case_loader import BenchmarkCaseLoader

episode = BenchmarkCaseLoader().load_file("/path/to/episode.yaml")
```

Review goal and recipient bindings, dialogue authorization, event gates, approach positions, and actor separation. NPC actions must complete before dependent task transitions; messages must be delivered before updating active requirements. Keep private evaluator answers and geometric evidence out of agent-visible task messages.

Authoring evidence is optional for loading a runtime episode. When using authoring promotion checks, supply the geometry, overlays, and certification files referenced by the specification. Fetch placement screens require an explicit robot contact profile rather than a bundled generated profile:

```python
from tarsphere.bench.production_layout import load_contact_profile

profile = load_contact_profile(
    "/path/to/contact_profile.json",
    source_root="/path/to/profile-sources",
)
```

Relative profile source paths resolve against `source_root`, or the profile's directory when omitted. Pass `contact_profile_path` to `validate_task_layout` or `validate_episode_layout` when checking fetch placements. These authoring checks preserve the declared source hashes and geometry constraints.

Physical validation requires the simulator, assets, and tool services. Schema validation alone does not establish collision-free execution or task success. Store generated authoring evidence outside the source checkout and version your task specifications with their required resources. The evaluation launcher accepts external YAML files, directories, and JSON path indexes; see [evaluation](EVALUATION.md).
