# Entry points

Run commands from the repository root. Physical execution uses a local `host_env.sh` copied from [host_env.example.sh](host_env.example.sh); offline planning requires only the Python development installation.

| Script | Purpose |
| --- | --- |
| `setup.sh` | Prepare the simulator and separate GraspGenX environments. |
| `activate.sh` | Source the configured runtime into the current shell. |
| `evaluate.sh` | Plan, launch, monitor, stop, or score supplied episodes. |
| `teleop.sh` | Interactively control the robot and animated NPCs. |
| `stage.sh` | Stage runtime and assets on local storage. |

```bash
cp scripts/host_env.example.sh host_env.sh
# Fill paths and model endpoints before physical setup.
scripts/setup.sh
scripts/evaluate.sh --plan
scripts/evaluate.sh --episodes examples/episodes/navigation.yaml
scripts/evaluate.sh --status
scripts/evaluate.sh --stop
```

`--episodes PATH` accepts YAML, a directory of YAML files, or a JSON path index. The default is `examples/episodes/navigation.yaml`. Run output defaults to `~/.local/share/tarsphere/runs`; use `--output-dir` for another location. Teleoperation sessions use `~/.local/share/tarsphere/demos/teleop`.

[runtime-constraints.txt](runtime-constraints.txt) pins the simulator dependency stack. [graspgenx-requirements.txt](graspgenx-requirements.txt) is installed into its own environment. Command implementations live in `tarsphere.cli`; tool services live in `tarsphere.infra.services`.

See [installation](../docs/INSTALLATION.md), [evaluation](../docs/EVALUATION.md), and [teleoperation](../docs/TELEOPERATION.md).
