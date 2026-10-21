"""Task abstraction for the end2end demo.

A `Task` owns the action sequence that runs AFTER cuRobo's grasp plan
(which gives us approach + linear-grasp + lift trajectories). Different
tasks layer different post-pick behaviour on top:

- :class:`PickAndLiftTask` — just lift. The current default behaviour.
- :class:`PickAndDropInBinTask` — after the lift, plan to a transport
  waypoint over the bin, lower to the drop pose, open the fingers, and
  retract back up.

Each task returns the **full joint trajectory** (arm columns +
master-gripper columns) to feed into kinematic playback / dynamic
simulation, along with the named segment lengths for phase logging.

Adding a new task = add a subclass + register it in :data:`TASKS`.
The CLI's ``--task`` flag selects by name.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import trimesh.transformations as tra

log = logging.getLogger("tasks")


@dataclass
class TaskResult:
    """Output of a task's planner.

    Attributes:
        joint_traj: ``(T, n_arm + n_gripper)`` numpy array. Same column
            layout as the existing pick-and-lift output so all
            downstream code (export_trajectory, dynamic_playback,
            viser_visualize, render_trajectory_mp4) works unchanged.
        segments: ordered list of ``(phase_name, n_frames)`` pairs
            describing what each chunk of the trajectory is. Used only
            for logging.
    """

    joint_traj: np.ndarray
    segments: List[Tuple[str, int]]


class Task:
    """Base class for action sequences. Implement :meth:`plan_actions`.

    The state machine each task constructs is a list of named segments
    appended to the output trajectory. The naming is exposed via
    :attr:`TaskResult.segments` so the JSON + logs are self-describing.
    """

    NAME: str = "base"

    def plan_actions(
        self,
        *,
        planner,
        bundle,
        profile,
        grasps_world: np.ndarray,
        conf: np.ndarray,
        target_idx: int,
        pregrasp_traj: np.ndarray,
        lift_traj: np.ndarray | None,
        env_cfg: Dict[str, Any],
        close_frames: int,
        hold_frames: int,
        playback_mode: str,
        # `result` is cuRobo's GraspPlanResult; tasks can pull
        # individual segments off `result._segments` (approach / grasp /
        # lift, each as a separate (T, n_arm+n_grip?) numpy array) for a
        # finer-grained state machine. Optional for back-compat — if
        # None, tasks fall back to ``pregrasp_traj`` (approach+grasp
        # concatenated).
        result=None,
    ) -> TaskResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _open_close_vals(profile) -> tuple[np.ndarray, np.ndarray]:
    names = list(profile.gripper_joint_names)
    open_vals = np.array([profile.open_value(n) for n in names], dtype=np.float32)
    close_vals = np.array([profile.close_value(n) for n in names], dtype=np.float32)
    return open_vals, close_vals


def _stack_arm_and_gripper(
    arm_traj: np.ndarray, grip_q: np.ndarray, n_grip: int
) -> np.ndarray:
    """``(T, n_arm)`` arm trajectory + per-joint ``grip_q`` ->
    ``(T, n_arm + n_grip)``.
    """
    T = arm_traj.shape[0]
    grip_block = np.broadcast_to(grip_q[None, :], (T, n_grip)).astype(np.float32)
    return np.concatenate([arm_traj.astype(np.float32), grip_block], axis=1)


def _hold(last_row: np.ndarray, n_frames: int) -> np.ndarray:
    if n_frames <= 0:
        return np.empty((0, last_row.shape[0]), dtype=np.float32)
    return np.tile(last_row.astype(np.float32), (n_frames, 1))


def _resample_traj(traj: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample a (T_src, n) trajectory to (target_frames, n) via linear
    interpolation along the time axis. Useful for slowing down or
    speeding up segments without losing intermediate poses.
    """
    if traj.shape[0] == target_frames or traj.shape[0] < 2:
        return traj.astype(np.float32)
    src_idx = np.arange(traj.shape[0], dtype=np.float32)
    dst_idx = np.linspace(0, traj.shape[0] - 1, target_frames, dtype=np.float32)
    out = np.empty((target_frames, traj.shape[1]), dtype=np.float32)
    for j in range(traj.shape[1]):
        out[:, j] = np.interp(dst_idx, src_idx, traj[:, j])
    return out


def _ramp(start_vals: np.ndarray, end_vals: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear interpolation from start_vals to end_vals over n_frames.
    Returns shape ``(n_frames, len(start_vals))``.
    """
    if n_frames <= 0:
        return np.empty((0, len(start_vals)), dtype=np.float32)
    alpha = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)[:, None]
    return (start_vals[None, :] + alpha * (end_vals - start_vals)[None, :]).astype(
        np.float32
    )


def _slice_arm(traj: np.ndarray, n_arm: int) -> np.ndarray:
    """cuRobo may return planned values for the gripper joints too;
    drop them so the task can append its own gripper schedule.
    """
    if traj.shape[1] > n_arm:
        return traj[:, :n_arm].astype(np.float32)
    return traj.astype(np.float32)


# ---------------------------------------------------------------------------
# pick_and_lift  (current default behaviour)
# ---------------------------------------------------------------------------


class PickAndLiftTask(Task):
    """State machine: go_to_pre_grasp_pose → go_from_pre_grasp_to_grasp_pose
    → close_fingers → lift_object.

    Each step is a separately named segment in the output trajectory so
    the JSON + logs document the action sequence explicitly. The
    individual cuRobo segments (``result._segments``) are used when
    available; otherwise the combined ``pregrasp_traj`` is split
    in half as a fallback (only matters if a caller passes
    ``result=None``).
    """

    NAME = "pick_and_lift"

    # Frames the lift segment is resampled to (default 240 = 4 sec at
    # 60 fps). cuRobo's lift_interpolated_trajectory is only ~41
    # waypoints, which at 60 fps plays in 0.7 sec — too fast for the
    # gripper to keep grip on the object under physics. Interpolating
    # over 4 sec gives a controlled, deliberate vertical lift.
    LIFT_FRAMES = 240
    # Same idea for pre-grasp → grasp: cuRobo's segment is ~41 waypoints
    # at 60 fps = 0.7s, which is fast enough that the approach can bump
    # / tip the object before the fingers close. 120 frames ≈ 2s.
    GRASP_FRAMES = 120

    def _build_pick_and_lift(
        self,
        profile,
        pregrasp_traj,
        lift_traj,
        close_frames,
        hold_frames,
        result,
        hold_after_close_frames=None,
    ) -> tuple[list[np.ndarray], list[tuple[str, int]]]:
        """Return the (chunks, segments) for the pick+lift state machine.

        Subclasses (e.g. PickAndDropInBinTask) call this and then append
        their own transport / release segments.
        """
        n_arm = profile.n_arm
        n_grip = profile.n_gripper
        open_vals, close_vals = _open_close_vals(profile)

        # Prefer the per-segment trajectories from cuRobo (so we can
        # name the approach (joint-space) and grasp (linear cartesian)
        # parts separately). Fall back to splitting the concatenated
        # pregrasp_traj down the middle if separated segments aren't
        # available.
        seg_dict = getattr(result, "_segments", None) if result is not None else None
        if (
            seg_dict is not None
            and seg_dict.get("approach") is not None
            and seg_dict.get("grasp") is not None
        ):
            approach_arm = _slice_arm(seg_dict["approach"], n_arm)
            grasp_arm = _slice_arm(seg_dict["grasp"], n_arm)
        else:
            full = _slice_arm(pregrasp_traj, n_arm)
            mid = full.shape[0] // 2
            approach_arm = full[:mid]
            grasp_arm = full[mid:]
        lift_arm = _slice_arm(lift_traj, n_arm) if lift_traj is not None else None

        chunks: list[np.ndarray] = []
        segments: list[tuple[str, int]] = []

        # 1. go_to_pre_grasp_pose — free-space joint plan from start
        #    config to the pre-grasp pose (offset above the grasp).
        if approach_arm.shape[0] > 0:
            seg = _stack_arm_and_gripper(approach_arm, open_vals, n_grip)
            chunks.append(seg)
            segments.append(("go_to_pre_grasp_pose", seg.shape[0]))
            if hold_frames > 0:
                chunks.append(_hold(seg[-1], hold_frames))
                segments.append(("hold_at_pre_grasp", hold_frames))

        # 2. go_from_pre_grasp_to_grasp_pose — linear cartesian motion
        #    along the tool's z-axis from pre-grasp to grasp. Resample
        #    to GRASP_FRAMES so the approach is slow enough not to bump
        #    the object before the fingers close.
        if grasp_arm.shape[0] > 0:
            grasp_slow = _resample_traj(grasp_arm, self.GRASP_FRAMES)
            seg = _stack_arm_and_gripper(grasp_slow, open_vals, n_grip)
            chunks.append(seg)
            segments.append(("go_from_pre_grasp_to_grasp_pose", seg.shape[0]))
            if hold_frames > 0:
                chunks.append(_hold(seg[-1], hold_frames))
                segments.append(("hold_at_grasp", hold_frames))

        # 3. close_fingers — hold arm at grasp pose; ramp gripper closed.
        n_close = max(1, close_frames)
        grasp_arm_end = grasp_arm[-1] if grasp_arm.shape[0] > 0 else approach_arm[-1]
        close_arm = np.tile(grasp_arm_end, (n_close, 1))
        if n_grip > 0:
            ramp = _ramp(open_vals, close_vals, n_close)
            close_full = np.concatenate(
                [close_arm.astype(np.float32), ramp],
                axis=1,
            )
        else:
            close_full = close_arm.astype(np.float32)
        chunks.append(close_full)
        segments.append(("close_fingers", n_close))
        # Hold AFTER close before lifting. Default = hold_frames, but a
        # gripper that closes slowly (e.g. velocity-mode multi-finger hands)
        # needs LONGER here so the fingers fully settle on the object before
        # the lift — otherwise the object slips out (premature lift).
        hac = (
            hold_after_close_frames
            if hold_after_close_frames is not None
            else hold_frames
        )
        if hac > 0:
            chunks.append(_hold(close_full[-1], hac))
            segments.append(("hold_after_close", hac))

        # 4. lift_object — linear cartesian motion lifting the gripper
        #    (gripper held closed around the object). Resampled to
        #    ``LIFT_FRAMES`` so the upward motion is slow enough that
        #    the held object stays in the gripper through the lift
        #    instead of slipping out under inertia.
        if lift_arm is not None and lift_arm.shape[0] > 0:
            lift_slow = _resample_traj(lift_arm, self.LIFT_FRAMES)
            seg = _stack_arm_and_gripper(lift_slow, close_vals, n_grip)
            chunks.append(seg)
            segments.append(("lift_object", seg.shape[0]))
            # Pause AFTER the lift so the gripper / object settle before
            # the next phase (e.g. move_to_above_bin in
            # pick_and_drop_in_bin). For pick_and_lift on its own this is
            # the last segment in the trajectory; the hold just gives the
            # MP4 a stable end frame.
            if hold_frames > 0:
                chunks.append(_hold(seg[-1], hold_frames))
                segments.append(("hold_after_lift", hold_frames))

        return chunks, segments

    def plan_actions(
        self,
        *,
        planner,
        bundle,
        profile,
        grasps_world,
        conf,
        target_idx,
        pregrasp_traj,
        lift_traj,
        env_cfg,
        close_frames,
        hold_frames,
        playback_mode,
        result=None,
        hold_after_close_frames=None,
    ) -> TaskResult:
        chunks, segments = self._build_pick_and_lift(
            profile,
            pregrasp_traj,
            lift_traj,
            close_frames,
            hold_frames,
            result,
            hold_after_close_frames=hold_after_close_frames,
        )
        joint_traj = np.concatenate(chunks, axis=0).astype(np.float32)
        return TaskResult(joint_traj=joint_traj, segments=segments)


# ---------------------------------------------------------------------------
# pick_and_drop_in_bin  (the new task)
# ---------------------------------------------------------------------------


class PickAndDropInBinTask(PickAndLiftTask):
    """State machine: pick_and_lift + move_to_above_bin + open_fingers_to_drop.

    Builds on :class:`PickAndLiftTask` by adding:

      * ``move_to_above_bin`` — cuRobo plan_pose from lift-end to
        ``[bin_x, bin_y, bin_z + 0.20]`` (20 cm above the bin),
        gripper still closed.
      * ``hold_above_bin`` — settle before opening.
      * ``open_fingers_to_drop`` — ramp the gripper from close → open
        while holding the arm at the above-bin pose. The object falls
        into the bin during this segment + the trailing hold.

    The bin pose is read from ``env_cfg.assets`` (the asset with
    ``id == "bin"``). The gripper orientation at the drop pose matches
    the lift-end orientation (gripper pointing down).

    The move_to_above_bin segment is intentionally s l o w — cuRobo
    plans a fast trajectory; we downsample to ``MOVE_TO_BIN_FRAMES``
    frames so the arm doesn't jerk the held object out of the gripper.
    """

    NAME = "pick_and_drop_in_bin"

    # Frames for the move_to_above_bin segment. At sim_fps=60 this is
    # 360/60 = 6 sec for the long swing across the workspace.
    MOVE_TO_BIN_FRAMES = 360

    # Drop height above the bin (m). Set high enough that the
    # gripper's fingertips clear the bin rim with margin — the panda_hand
    # frame sits ~10 cm above the fingertips along its z-axis, so a
    # 30 cm drop height puts the fingertips ~20 cm above the rim.
    DROP_HEIGHT_ABOVE_BIN = 0.30

    def _bin_world_pose(self, env_cfg: Dict[str, Any]) -> np.ndarray | None:
        """Return the bin's 4x4 world transform from env_cfg, or None."""
        for a in env_cfg.get("assets", []):
            if a.get("id") == "bin":
                pose = a.get("pose", {})
                T = np.eye(4)
                T[:3, 3] = pose.get("translation", [0.0, 0.0, 0.0])
                q_xyzw = pose.get("quaternion_xyzw", [0, 0, 0, 1])
                if not (
                    abs(q_xyzw[0]) < 1e-9
                    and abs(q_xyzw[1]) < 1e-9
                    and abs(q_xyzw[2]) < 1e-9
                    and abs(q_xyzw[3] - 1) < 1e-9
                ):
                    T[:3, :3] = tra.quaternion_matrix(
                        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
                    )[:3, :3]
                return T
        return None

    def _plan_to_world_pose(
        self,
        planner,
        profile,
        q_start_arr: np.ndarray,
        T_world: np.ndarray,
        robot_base_T: np.ndarray,
        target_frames: int = 120,
    ) -> np.ndarray | None:
        # planner.joint_names is the planning DOFs only (e.g. 7 panda
        # arm joints for franka). If we were handed back a (T, n_arm+n_grip)
        # trajectory row, drop the gripper cols.
        n_plan = len(planner.joint_names)
        if q_start_arr.shape[0] > n_plan:
            q_start_arr = q_start_arr[:n_plan]
        """Plan an arm trajectory from q_start_arr to a world-frame pose
        target on the tool frame. Returns ``(T, n_arm)`` numpy array
        or None if planning failed.
        """
        import torch
        from curobo.types import JointState, Pose

        target_link = profile.tool_frame
        T_world_robot_inv = tra.inverse_matrix(robot_base_T)
        # T_world is already a tool-frame pose (`panda_hand` convention),
        # NOT a GraspGenX-convention grasp pose, so we do NOT apply
        # `grasp_to_tool_transform` here. That offset is only for
        # converting GraspGenX's X-closing convention to the URDF's
        # Y-closing convention — it doesn't belong on free-space
        # waypoints whose rotation comes from URDF FK.
        T_target_robot = T_world_robot_inv @ T_world
        pos = T_target_robot[:3, 3]
        quat_wxyz = tra.quaternion_from_matrix(T_target_robot)  # wxyz
        pos_t = torch.tensor(
            [pos.tolist()], device="cuda", dtype=torch.float32
        ).unsqueeze(0)
        quat_t = torch.tensor(
            [quat_wxyz.tolist()], device="cuda", dtype=torch.float32
        ).unsqueeze(0)
        q_start = JointState.from_position(
            torch.tensor([q_start_arr.tolist()], device="cuda", dtype=torch.float32),
            joint_names=planner.joint_names,
        )
        try:
            from curobo_compat import grasp_goals

            result = planner.plan_pose(
                grasp_goals(target_link, pos_t, quat_t),
                q_start,
            )
        except Exception as e:
            log.warning("plan_pose raised: %s", e)
            return None
        if result is None or result.success is None or not bool(result.success.any()):
            log.warning(
                "plan_pose to transport waypoint failed: status=%s",
                getattr(result, "status", "<no status>"),
            )
            return None
        traj = result.interpolated_trajectory
        if traj is None:
            return None
        pos_np = traj.position.detach().cpu().numpy()
        while pos_np.ndim > 2:
            pos_np = pos_np[0]
        pos_np = pos_np.astype(np.float32)
        # Slice off any planner-extra dofs (e.g. cuRobo's franka.yml
        # has 9 cspace dofs but our profile only treats the 7 arm
        # joints as 'arm' — we appended the gripper schedule ourselves).
        n_arm = profile.n_arm
        if pos_np.shape[1] > n_arm:
            pos_np = pos_np[:, :n_arm]
        # cuRobo's plan_pose interpolated_trajectory pads the trajectory
        # to a fixed length (~5000 frames) by holding the goal config
        # after the real motion ends — typically only the first ~30
        # frames contain meaningful waypoints. Naively linspace-
        # subsampling the full padded array compresses the entire
        # motion into 1-2 of our output frames (the arm teleports).
        # Find the last frame where joints actually move, then
        # linspace ONLY across that prefix.
        deltas = np.linalg.norm(np.diff(pos_np, axis=0), axis=1)
        motion_threshold = 1e-5
        moving_idx = np.where(deltas > motion_threshold)[0]
        if len(moving_idx) > 0:
            last_motion = int(moving_idx[-1]) + 1  # +1 because diff shifts by one
            log.info(
                "plan_pose trajectory: %d total frames, motion ends at %d "
                "(rest is goal-padding)",
                pos_np.shape[0],
                last_motion,
            )
            pos_np = pos_np[: last_motion + 1]
        # Resample to a smooth ``target_frames``-frame trajectory.
        # Linear interpolation across the (small) motion span gives a
        # constant-velocity playback at the demo's frame rate.
        if pos_np.shape[0] != target_frames and pos_np.shape[0] > 1:
            src_idx = np.arange(pos_np.shape[0], dtype=np.float32)
            dst_idx = np.linspace(
                0, pos_np.shape[0] - 1, target_frames, dtype=np.float32
            )
            resampled = np.empty((target_frames, pos_np.shape[1]), dtype=np.float32)
            for j in range(pos_np.shape[1]):
                resampled[:, j] = np.interp(dst_idx, src_idx, pos_np[:, j])
            pos_np = resampled
        return pos_np

    def plan_actions(
        self,
        *,
        planner,
        bundle,
        profile,
        grasps_world,
        conf,
        target_idx,
        pregrasp_traj,
        lift_traj,
        env_cfg,
        close_frames,
        hold_frames,
        playback_mode,
        result=None,
        hold_after_close_frames=None,
    ) -> TaskResult:
        n_arm = profile.n_arm
        n_grip = profile.n_gripper
        open_vals, close_vals = _open_close_vals(profile)

        # Step 1: build the pick + lift state machine via the parent
        # helper. Returns the ordered (chunks, segments) for steps:
        #   go_to_pre_grasp_pose, hold, go_from_pre_grasp_to_grasp_pose,
        #   hold, close_fingers, hold, lift_object
        chunks, segments = self._build_pick_and_lift(
            profile,
            pregrasp_traj,
            lift_traj,
            close_frames,
            hold_frames,
            result,
            hold_after_close_frames=hold_after_close_frames,
        )

        # Locate the bin in world frame. If absent, just return pick+lift.
        bin_T_world = self._bin_world_pose(env_cfg)
        if bin_T_world is None:
            log.warning(
                "pick_and_drop_in_bin: no 'bin' asset in env; "
                "falling back to pick_and_lift."
            )
            return TaskResult(
                joint_traj=np.concatenate(chunks, axis=0).astype(np.float32),
                segments=segments,
            )

        # Need a lift segment to anchor the drop orientation.
        if lift_traj is None or lift_traj.shape[0] == 0:
            log.warning(
                "pick_and_drop_in_bin: no lift segment; "
                "falling back to pick_and_lift."
            )
            return TaskResult(
                joint_traj=np.concatenate(chunks, axis=0).astype(np.float32),
                segments=segments,
            )
        lift_arm_end = _slice_arm(lift_traj, n_arm)[-1].copy()

        # Step 2: compute the gripper orientation at the end of the
        # lift via URDF FK. We re-use that orientation for the drop
        # pose so the gripper stays gripper-down across the swing.
        try:
            import yourdfpy

            fk = yourdfpy.URDF.load(
                profile.urdf_path, build_collision_scene_graph=False, load_meshes=False
            )
            cfg = {n: float(v) for n, v in zip(profile.arm_joint_names, lift_arm_end)}
            for gn in profile.gripper_joint_names:
                cfg[gn] = float(profile.close_value(gn))
            fk.update_cfg(cfg)
            T_lift_link = fk.get_transform(frame_to=profile.tool_frame)
            T_lift_world = bundle.robot_base_T @ T_lift_link
            R_drop = T_lift_world[:3, :3]
        except Exception as e:
            log.warning("Could not FK lift-end pose (%s); using identity rot", e)
            R_drop = np.eye(3)

        # Step 3: build the drop target = 20 cm above the bin, same
        # orientation as lift-end. Plan cuRobo plan_pose lift→drop.
        drop_T = np.eye(4)
        drop_T[:3, :3] = R_drop
        drop_T[:3, 3] = bin_T_world[:3, 3] + np.array(
            [0, 0, self.DROP_HEIGHT_ABOVE_BIN]
        )
        log.info(
            "pick_and_drop_in_bin: drop pose at %s (%.2f m above bin)",
            drop_T[:3, 3].tolist(),
            self.DROP_HEIGHT_ABOVE_BIN,
        )

        move_arm = self._plan_to_world_pose(
            planner,
            profile,
            lift_arm_end,
            drop_T,
            bundle.robot_base_T,
            target_frames=self.MOVE_TO_BIN_FRAMES,
        )
        if move_arm is None or move_arm.shape[0] == 0:
            log.warning(
                "pick_and_drop_in_bin: move_to_above_bin plan failed; "
                "falling back to pick_and_lift."
            )
            return TaskResult(
                joint_traj=np.concatenate(chunks, axis=0).astype(np.float32),
                segments=segments,
            )

        # Step 4: move_to_above_bin (gripper still closed)
        move_full = _stack_arm_and_gripper(move_arm, close_vals, n_grip)
        chunks.append(move_full)
        segments.append(("move_to_above_bin", move_full.shape[0]))
        if hold_frames > 0:
            chunks.append(_hold(move_full[-1], hold_frames))
            segments.append(("hold_above_bin", hold_frames))

        # Step 5: open_fingers_to_drop — ramp gripper open while
        # holding the arm at the above-bin pose. Object falls into bin
        # during this segment + the trailing hold.
        if n_grip > 0:
            n_open = max(1, close_frames)
            arm_hold = np.tile(move_arm[-1], (n_open, 1))
            ramp_open = _ramp(close_vals, open_vals, n_open)
            release_full = np.concatenate(
                [arm_hold.astype(np.float32), ramp_open],
                axis=1,
            )
            chunks.append(release_full)
            segments.append(("open_fingers_to_drop", n_open))
            # Long hold so the object has time to fall into the bin
            # before the trajectory ends.
            if hold_frames > 0:
                chunks.append(_hold(release_full[-1], hold_frames))
                segments.append(("hold_after_drop", hold_frames))

        joint_traj = np.concatenate(chunks, axis=0).astype(np.float32)
        return TaskResult(joint_traj=joint_traj, segments=segments)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


TASKS: Dict[str, type] = {
    PickAndLiftTask.NAME: PickAndLiftTask,
    PickAndDropInBinTask.NAME: PickAndDropInBinTask,
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Known: {sorted(TASKS)}")
    return TASKS[name]()
