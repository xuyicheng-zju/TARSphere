# Simulation assets

The source tree contains character descriptions and small authored USD compositions. Full worlds, scene metadata, robot packages, native character clips, and model checkpoints are separate resources. Package installation does not download them.

## Sources and paths

| Resource | Source | Required destination/configuration |
| --- | --- | --- |
| GRScenes scene assets | [InternRobotics/GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes); follow its dataset access and attribution terms. | `TARSPHERE_ASSETS_PATH/scenes/GRScenes-100/{home_scenes,commercial_scenes}/`, with shared `Materials/`, `models/`, and per-scene `scenes/<scene_id>/`. |
| Robot USD asset packages | [Robot asset distribution](https://huggingface.co/datasets/InternRobotics/GRScenes/tree/main/robots); preserve the packages’ original notices. | Extract the selected robot package under `TARSPHERE_ASSETS_PATH/robots/`, including referenced meshes and policy files. |
| G1 Dex3 donor (required even for G1 navigation) | NVIDIA Isaac Sim 5.1 [Groot G1 USD](https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Samples/Groot/Robots/g1_29dof_with_hand_rev_1_0.usd). Preserve NVIDIA/Unitree asset terms. | Save as `TARSPHERE_ASSETS_PATH/robots/g1_with_hand/g1_29dof_with_hand_rev_1_0.usd`. This is the donor for articulated hands and body colliders, not a replacement for the GRScenes locomotion body. |
| GRScenes navigation metadata | Official [benchmark/meta.zip](https://huggingface.co/datasets/InternRobotics/GRScenes/tree/main/benchmark). | Extract to `TARSPHERE_ASSETS_PATH/benchmark/meta/<scene_id>/`; the examples require `occupancy.npy`. |
| SAM3.1 multiplex checkpoint | [facebook/sam3.1](https://huggingface.co/facebook/sam3.1); access approval/authentication may be required. | `TARSPHERE_SAM3_CHECKPOINT` points to `sam3.1_multiplex.pt`. |
| X-NavDP policy | [InternRobotics/X-NavDP](https://huggingface.co/InternRobotics/X-NavDP/tree/main). | `TARSPHERE_XNAVDP_CHECKPOINT` points to `x-navdp_posttrain.ckpt`. The inference source is bundled in `third_party/NavDP/baselines/x-navdp`. |
| FrontierNet RGB-D model | The checkpoint linked by [cvg/FrontierNet](https://github.com/cvg/FrontierNet#setup). | `TARSPHERE_FRONTIERNET_CHECKPOINT` points to `rgbd_11cls.pth`. |
| GraspGenX models | [adithyamurali/GraspGenXModel](https://huggingface.co/adithyamurali/GraspGenXModel). | `third_party/GraspGenX/ext/graspgenx_checkpoints/release/{gen,dis}/`, each with `config.yaml` and `epoch_*.pth`. |
| GraspGenX grippers | [adithyamurali/gripper_descriptions](https://huggingface.co/datasets/adithyamurali/gripper_descriptions). | `third_party/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets/`, including `x_grippers/unitree_g1/` for the default service. |
| Native characters and animations | [NVIDIA Isaac Sim asset packs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html#local-assets-packs), using the [Isaac People character layout](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/action_and_event_data_generation/ext_replicator-agent/customization.html). | Copy `Isaac/People/Characters/` to `characters/IsaacPeople/` and `Isaac/People/Animations/` to `characters/Animations/`. Preserve the complete rig, material and animation dependencies. |

Upstream archive folder names can differ from TARSphere's flat layout. Extract the complete scene packages, rename their top-level directories into the layout above, and preserve relative scene-to-material/model references. Copy this repository's `assets/` contents over the prepared tree to add character descriptions and authored compositions:

```bash
# TARSPHERE_ASSETS_PATH must already point to your downloaded/extracted asset tree.
rsync -a assets/ "$TARSPHERE_ASSETS_PATH/"
```

The bundled GraspGenX source excludes its Git-LFS-backed procedural-gripper meshes and sample inputs. TARSphere does not use those upstream demo resources: its default `unitree_g1` inference path reads the separately installed `gripper_descriptions` checkout listed above.

For episodes with humans, `characters/characters.json` lists the supported character directories and exact USD filenames. Keep `characters/IsaacPeople/Biped_Setup.usd`, its `biped_demo/` rig payload and the sibling `characters/Animations/` directory together. Copy each selected character’s official asset-browser thumbnail to the path declared by its `reference_images` field under `characters/reference_images/`. These profile images are used by the human interaction tools.

Default evaluation requires the selected scene USD and its declared occupancy map. BEV video recording additionally needs the episode's `rgb_bev` and `projection` files; these are task-specific authoring resources. Use the documented `--ego-only` recording mode when recording without a top-down scene image.

Install the robot package selected by your episode, including its referenced meshes and controller policies. The default G1 example uses `robots/g1/g1_29dof_color.usd`. Episodes with humans also require the corresponding character assets and native animation clips. A root folder with only metadata is insufficient. Run the environment preflight from [installation](../docs/INSTALLATION.md); it checks the selected episodes, their USD dependencies, model files, and service readiness before execution.

Robot meshes bundled with cuRobo retain `third_party/curobo/LICENSE_ASSETS` and do not replace the simulator robot package. Original source licensing does not override scene, robot, animation, or model-weight terms. See [acknowledgments](../README.md#acknowledgments).

## G1 camera and donor preparation

The public GRScenes `g1_29dof_color.usd` contains `d435_link` but does not
contain the ego Camera. TARSphere creates the missing `d435_link/Camera` in
its temporary robot composition before sensor initialization. No modified G1
download or manual USD edit is required. An existing Camera keeps its authored
rest pose; a conflicting non-Camera prim is rejected.

The rest transform, relative to `d435_link`, matches the asset used for
evaluation:

| Property | Value |
| --- | --- |
| Local translation (metres) | `(0.053944, 0.0, 0.059076)` |
| Local orientation, quaternion **WXYZ** | `(0.2557072, 0.6592525, -0.6592525, -0.2557072)` |
| Transform order | translate, orient |
| Evaluated resolution / horizontal FOV | `640 × 480` / `100°` |

The runtime applies the G1 profile's **additional −15° mount pitch** to this
rest orientation, then adds any commanded gaze tilt. Keep that calibration
enabled for evaluation; do not bake it into the rest quaternion a second time.
The runtime sets the evaluated FOV after creating the render product.

Both NVIDIA's link-nested donor joints and the older URDF import's `/joints/`
layout are supported. Preflight composes the robot and validates its Camera,
articulated hands and body colliders, including unresolved USD references.
For the two environment and asset checks, see [installation](../docs/INSTALLATION.md).
