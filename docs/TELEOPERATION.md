# Robot & NPC teleoperation

[中文说明](TELEOPERATION.zh-CN.md) · [Installation](INSTALLATION.md)

Control the robot and animated NPCs interactively in a TARSphere scene. The operator supplies keyboard input or terminal commands. Robot motion uses the physical controller; NPC motion uses the native animation controller and NavMesh. Walking, turning, posture control, and NPC actions do not call a language model. Video and still captures are optional.

## Start a session

Prepare the Linux NVIDIA runtime and external assets using the [installation guide](INSTALLATION.md). Configure `host_env.sh` with the Isaac Sim 5.1 SDK, project Python environment, and asset paths. NPC assets must include the rigged character and native animation resources described in [assets](../assets/README.md).

From the repository root:

```bash
scripts/teleop.sh --episode examples/episodes/navigation.yaml
```

The episode selects the scene, initial robot pose, and NPCs. Use `--episode /path/to/episode.yaml` for another scene configuration, and `--human-id Nora` to select the initial NPC. The first scene load may prepare materials and bake a NavMesh; wait for `[TELEOP] Ready` before operating.

For a server without a desktop window:

```bash
scripts/teleop.sh --headless --episode examples/episodes/navigation.yaml
```

Headless sessions accept terminal commands over SSH. Keyboard and GUI camera switching require a visible Isaac window. `--stream` enables Isaac Sim's native livestream transport. Ordinary robot and NPC control does not require agent API credentials or running perception/navigation model services.

## Choose who to control

The robot is active initially. Focus the Isaac window to send motion keys; return to the launching terminal to enter commands.

| Input | Effect |
| --- | --- |
| `list` | List the NPC IDs in this scene. |
| `human <id>` | Give keyboard control to that NPC. |
| `human` | Select the initial NPC chosen for the session. |
| `robot` | Give keyboard control to the robot. |
| Tab | Toggle between the robot and selected NPC. |
| `status` | Show active control, robot posture, and NPC positions/actions. |

For the supplied navigation example, an interactive sequence is:

```text
list
human Nora
look Nora 2
robot
move 0 0 15
status
```

While `human Nora` is active, use the window keys to walk Nora. Switch back with Tab or `robot`. Explicit robot commands such as `move` always address the robot; NPC commands always include their NPC ID.

## Window keyboard

| Keys | Robot | Animated NPC |
| --- | --- | --- |
| I / K | Forward / backward | Walk forward / backward relative to its control heading. |
| J / L | Move left / right | Walk left / right relative to its control heading. |
| U / O | Turn left / right | Adjust the direction of the next walk. |
| F8 | Save a still from the robot-eye camera. | Save a still from the robot-eye camera. |

Motion keys combine. Releasing one key preserves the others. Short U/O taps are retained for at least 0.15 seconds of simulation time. Switching control or issuing a terminal command clears held motion keys, so press them again to resume walking.

Each NPC has its own control heading. U/O changes that heading; a walking animation turns and travels toward the requested target. It does not rotate the character in place. NPC movement is admitted through native NavMesh and animation actions; a rejected destination leaves the NPC in its current physical state. The terminal reports rejection feedback.

## Robot commands

| Command | Effect |
| --- | --- |
| `move <forward_m> <left_m> <yaw_deg>` | One body-relative motion; positive yaw turns left. |
| `crouch [height_m]` | Lower the pelvis; the default height is 0.55 m. |
| `stand` | Return to the walking posture. |
| `tilt <degrees>` | Adjust robot-eye camera gaze within the supported limits. |
| `wait <seconds>` | Let simulation and animated actions advance. |
| `ego on` / `ego off` | Switch the GUI between robot-eye and third-person views. |

Stand before walking or turning. A crouched locomotion request is refused with feedback. Robot walking retains the existing carry posture when an object is held.

For physical grasping, configure the SAM3 and GraspGenX services and start them before launching:

```bash
source host_env.sh
source scripts/activate.sh
tarsphere/infra/services/start.sh
scripts/teleop.sh --episode episodes/case/fetch_and_deliver/base.yaml --with-grasp
```

Enter `grasp <visual description>` in the terminal. The current ego RGB-D is used to ground the object and execute the default physical grasp in one call. Feedback reports the physical outcome. A delivery episode must provide the matching physical task and object assets.

## Animated NPC commands

| Command | Effect |
| --- | --- |
| `goto <id> <x> <y> [z] [end_rot]` | Walk to a world-coordinate destination through NavMesh and native animation. If omitted, z uses the current NPC height; optional `end_rot` is native character yaw in degrees. |
| `look <id> [seconds]` | Run the native looking action. |
| `idle <id>` | Request the native idle action. |
| `sit <id> <seat_prim_path> [seconds]` | Approach and sit at a seat supported by the scene and native animation controller. |

Choose NPC IDs with `list`. For `goto`, select a reachable point on the scene's walkable surface; for `sit`, use the seat's actual USD prim path. `executed=True` means the controller accepted the request. Walking or sitting may still be in progress: allow simulation time to pass and inspect `status` for completion or failure.

NPC control always uses the animation backend. Keyboard commands become native walking targets; they do not teleport characters or replace the animation controller with kinematic motion.

## Optional captures

Enter `capture [label]` or press F8 to save a lossless 2560×1920 robot-eye PNG. Labels use letters, digits, `_`, or `-`. `--capture-width` and `--capture-height` set another resolution; `--capture-layers` adds transparent PNGs for visible actors using native instance segmentation.

Capture uses a temporary independent camera and verifies that physics time, robot/NPC poses, and the source cameras remain unchanged. It publishes the PNGs and metadata together, then releases its camera and render buffers.

Add `--record-video` for continuous robot-eye video (`ffmpeg` required), or `--record-ego` to retain the ego observations used by grasping. These options are not required to control the scene.

Session logs and optional media are saved outside the source tree under `~/.local/share/tarsphere/demos/teleop/<timestamp>/`. `--output /path/to/new/session.jsonl` selects another location; existing session logs are not overwritten.

## Exit and common issues

Enter `help` to print commands or `quit` to close the session. Closing headless stdin also exits. `--max-loops` can bound a session for development; its default is unlimited.

| Symptom | Action |
| --- | --- |
| Motion keys have no effect | Focus the Isaac window, check the active actor with `status`, and press the keys again after a terminal command. |
| NPC does not turn while standing | U/O selects its next walking direction; press a walking key to execute a native walk. |
| NPC destination is rejected | Read terminal feedback, choose a reachable NavMesh point, and check that another native action is not still running. |
| NPC or animation is missing | Check the episode's NPC IDs and rigged/native animation assets. |
| Robot refuses motion while crouched | Enter `stand`, then retry the motion. |
| Grasp is unavailable | Launch with `--with-grasp` after preparing the required services and physical task. |

Use `scripts/teleop.sh --help` for numerical motion, timing, camera, and output options.
