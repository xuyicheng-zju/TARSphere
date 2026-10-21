# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Third Party
import numpy as np
import torch
import trimesh.transformations as tra
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from torch import Tensor, nn
from torch.autograd import Function

import graspgenx.utils.so3 as so3
from graspgenx.robot import GripperInfo
from graspgenx.utils.transformations import matrix_to_quaternion


def compute_recall(
    pose_set_a: np.ndarray, pose_set_b: np.ndarray, radius: float = 0.02
) -> float:

    # Recompute for tighter tolerance
    tree = KDTree(pose_set_a[:, :3, 3])
    visited = set()

    for i, grasp in enumerate(pose_set_b):
        close_indexes = tree.query_ball_point(grasp[:3, 3], radius)
        visited.update([(close_index) for close_index in close_indexes])

    recall = len(visited) / len(pose_set_a)
    return recall


def compute_metrics_given_two_sets_of_poses(
    poses_A: torch.Tensor,
    poses_B: torch.Tensor,
    gripper_info: GripperInfo,
    consider_symmetry: bool = False,
) -> Dict[str, float]:
    """
    Note that poses_A and poses_B have the same batch size and shape. e.g. [N, 4, 4]
    """

    actual_noise_pts_quat = normalize_quaternion(
        matrix_to_quaternion(poses_A[:, :3, :3])
    )
    pred_noise_pts_quat = normalize_quaternion(matrix_to_quaternion(poses_B[:, :3, :3]))

    phi3 = angular_distance_phi3(actual_noise_pts_quat, pred_noise_pts_quat)
    criterion = GeodesicLoss(reduction="none")
    geodesic_dist = criterion(poses_A[:, :3, :3], poses_B[:, :3, :3])
    device = geodesic_dist.device

    if consider_symmetry and gripper_info.symmetric:

        poses_A_mirror = np.array(
            [
                g @ tra.euler_matrix(0, 0, np.pi)
                for g in poses_A.clone().detach().cpu().numpy()
            ]
        )
        poses_A_mirror = torch.from_numpy(poses_A_mirror).to(geodesic_dist.device)
        geodesic_dist_mirror = criterion(poses_A_mirror[:, :3, :3], poses_B[:, :3, :3])

        geodesic_dist = torch.vstack([geodesic_dist_mirror, geodesic_dist])
        geodesic_dist = geodesic_dist.min(axis=0)[0]

        actual_noise_pts_quat = normalize_quaternion(
            matrix_to_quaternion(poses_A_mirror[:, :3, :3])
        )
        phi3_mirror = angular_distance_phi3(actual_noise_pts_quat, pred_noise_pts_quat)
        phi3 = torch.vstack([phi3_mirror, phi3])
        phi3 = phi3.min(axis=0)[0]

    phi3 = phi3.mean()
    geodesic_dist = geodesic_dist.mean()

    depth = gripper_info.depth
    # TODO - read this depth value from cfg file
    poses_A_shifted = np.array(
        [
            g @ tra.translation_matrix([0, 0, depth])
            for g in poses_A.clone().detach().cpu().numpy()
        ]
    )
    poses_B_shifted = np.array(
        [
            g @ tra.translation_matrix([0, 0, depth])
            for g in poses_B.clone().detach().cpu().numpy()
        ]
    )

    actual_noise_pts_t = poses_A_shifted[:, :3, 3]
    pred_noise_pts_t = poses_B_shifted[:, :3, 3]
    error = torch.tensor(actual_noise_pts_t - pred_noise_pts_t)
    translation_error = torch.linalg.norm(error, dim=1).mean().to(geodesic_dist.device)
    stats = {
        "error_trans_l2": translation_error,
        "error_rot_geodesic": geodesic_dist,
        "error_rot_phi3": phi3,
    }
    return stats


def compute_metrics_given_two_sets_of_xgripper_poses(
    poses_A: torch.Tensor,
    poses_B: torch.Tensor,
    gripper_depth: torch.Tensor,
    gripper_symmetry: torch.Tensor,
    consider_symmetry: bool = False,
) -> Dict[str, float]:
    """
    Note that poses_A and poses_B have the same batch size and shape. e.g. [N, 4, 4]
    """

    if poses_A.shape[0] == 0 or poses_B.shape[0] == 0:
        stats = {
            "error_trans_l2": 0.0,
            "error_rot_geodesic": 0.0,
            "error_rot_phi3": 0.0,
        }
        return stats

    actual_noise_pts_quat = normalize_quaternion(
        matrix_to_quaternion(poses_A[:, :3, :3])
    )
    pred_noise_pts_quat = normalize_quaternion(matrix_to_quaternion(poses_B[:, :3, :3]))

    phi3 = angular_distance_phi3(actual_noise_pts_quat, pred_noise_pts_quat)
    criterion = GeodesicLoss(reduction="none")
    geodesic_dist = criterion(poses_A[:, :3, :3], poses_B[:, :3, :3])
    device = geodesic_dist.device

    if consider_symmetry:

        gripper_symmetry = gripper_symmetry.float().unsqueeze(-1)
        rot_mtx = (
            torch.from_numpy(tra.euler_matrix(0, 0, np.pi))
            .unsqueeze(0)
            .to(device)
            .to(torch.float32)
        )
        idt_mtx = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
        offset_mtx = gripper_symmetry * rot_mtx + (1.0 - gripper_symmetry) * idt_mtx

        poses_A_mirror = poses_A @ offset_mtx
        geodesic_dist_mirror = criterion(poses_A_mirror[:, :3, :3], poses_B[:, :3, :3])

        geodesic_dist = torch.vstack([geodesic_dist_mirror, geodesic_dist])
        geodesic_dist = geodesic_dist.min(axis=0)[0]

        actual_noise_pts_quat = normalize_quaternion(
            matrix_to_quaternion(poses_A_mirror[:, :3, :3])
        )
        phi3_mirror = angular_distance_phi3(actual_noise_pts_quat, pred_noise_pts_quat)
        phi3 = torch.vstack([phi3_mirror, phi3])
        phi3 = phi3.min(axis=0)[0]

    phi3 = phi3.mean()
    geodesic_dist = geodesic_dist.mean()

    offset_mtx = (
        torch.eye(4, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat_interleave(gripper_depth.shape[0], dim=0)
    )
    offset_mtx[:, 2:3, 3:4] = gripper_depth.unsqueeze(-1)

    poses_A_shifted = poses_A @ offset_mtx
    poses_B_shifted = poses_B @ offset_mtx

    actual_noise_pts_t = poses_A_shifted[:, :3, 3]
    pred_noise_pts_t = poses_B_shifted[:, :3, 3]
    error = torch.tensor(actual_noise_pts_t - pred_noise_pts_t)
    translation_error = torch.linalg.norm(error, dim=1).mean().to(geodesic_dist.device)
    stats = {
        "error_trans_l2": translation_error,
        "error_rot_geodesic": geodesic_dist,
        "error_rot_phi3": phi3,
    }
    return stats


def angular_distance_phi3(
    goal_quat: torch.Tensor, current_quat: torch.Tensor
) -> torch.Tensor:
    """This function computes the angular distance phi_3.

    See Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of Mathematical
    Imaging and Vision 35 (2009): 155-164.

    Args:
        goal_quat: _description_
        current_quat: _description_

    Returns:
        Angular distance in range [0,1]
    """
    dot_prod = (
        goal_quat[..., 0] * current_quat[..., 0]
        + goal_quat[..., 1] * current_quat[..., 1]
        + goal_quat[..., 2] * current_quat[..., 2]
        + goal_quat[..., 3] * current_quat[..., 3]
    )

    dot_prod = torch.abs(dot_prod)
    distance = dot_prod
    distance = torch.arccos(dot_prod) / (torch.pi * 0.5)
    return distance


def quat_multiply(
    q1: torch.Tensor, q2: torch.Tensor, q_res: torch.Tensor
) -> torch.Tensor:
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]

    q_res[..., 0] = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z

    q_res[..., 1] = a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z
    q_res[..., 2] = a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x
    q_res[..., 3] = a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y
    return q_res


class OrientationError(Function):
    @staticmethod
    def geodesic_distance(goal_quat, current_quat, quat_res):
        conjugate_quat = current_quat.clone()
        conjugate_quat[..., 1:] *= -1.0
        quat_res = quat_multiply(goal_quat, conjugate_quat, quat_res)

        quat_res = -1.0 * quat_res * torch.sign(quat_res[..., 0]).unsqueeze(-1)
        quat_res[..., 0] = 0.0
        # quat_res = conjugate_quat * 0.0
        return quat_res

    @staticmethod
    def forward(ctx, goal_quat, current_quat, quat_res):
        quat_res = OrientationError.geodesic_distance(goal_quat, current_quat, quat_res)
        rot_error = torch.norm(quat_res, dim=-1, keepdim=True)
        ctx.save_for_backward(quat_res, rot_error)
        return rot_error

    @staticmethod
    def backward(ctx, grad_out):
        grad_mul = None
        if ctx.needs_input_grad[1]:
            quat_error, r_err = ctx.saved_tensors
            scale = 1 / r_err
            scale = torch.nan_to_num(scale, 0, 0, 0)

            grad_mul = grad_out * scale * quat_error
            # print(grad_out.shape)
            # if grad_out.shape[0] == 6:
            #    #print(grad_out.view(-1))
            #    #print(grad_mul.view(-1)[-6:])
            #    #exit()
        return None, grad_mul, None


def normalize_quaternion(in_quaternion: torch.Tensor) -> torch.Tensor:
    k = torch.sign(in_quaternion[..., 0:1])
    # NOTE: torch sign returns 0 as sign value when value is 0.0
    k = torch.where(k == 0, 1.0, k)
    k2 = k / torch.linalg.norm(in_quaternion, dim=-1, keepdim=True)
    # normalize quaternion
    in_q = k2 * in_quaternion
    return in_q


class GeodesicLoss(nn.Module):
    r"""Creates a criterion that measures the distance between rotation matrices, which is
    useful for pose estimation problems.
    The distance ranges from 0 to :math:`pi`.
    See: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices and:
    "Metrics for 3D Rotations: Comparison and Analysis" (https://link.springer.com/article/10.1007/s10851-009-0161-2).

    Both `input` and `target` consist of rotation matrices, i.e., they have to be Tensors
    of size :math:`(minibatch, 3, 3)`.

    The loss can be described as:

    .. math::
        \text{loss}(R_{S}, R_{T}) = \arccos\left(\frac{\text{tr} (R_{S} R_{T}^{T}) - 1}{2}\right)

    Args:
        eps (float, optional): term to improve numerical stability (default: 1e-7). See:
            https://github.com/pytorch/pytorch/issues/8069.

        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will
            be applied, ``'mean'``: the weighted mean of the output is taken,
            ``'sum'``: the output will be summed. Default: ``'mean'``

    Shape:
        - Input: Shape :math:`(N, 3, 3)`.
        - Target: Shape :math:`(N, 3, 3)`.
        - Output: If :attr:`reduction` is ``'none'``, then :math:`(N)`. Otherwise, scalar.
    """

    def __init__(self, eps: float = 1e-7, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.double()
        target = target.double()
        R_diffs = input @ target.permute(0, 2, 1)
        # See: https://github.com/pytorch/pytorch/issues/7500#issuecomment-502122839.
        traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
        dists = torch.acos(torch.clamp((traces - 1) / 2, -1 + self.eps, 1 - self.eps))
        if self.reduction == "none":
            return dists
        elif self.reduction == "mean":
            return dists.mean()
        elif self.reduction == "sum":
            return dists.sum()


def compute_pose_distance_batch(
    poses1: torch.Tensor, poses2: torch.Tensor
) -> torch.Tensor:
    """
    Compute distances between two sets of poses in a batched manner.

    Args:
        poses1: First set of poses [N1, 4, 4]
        poses2: Second set of poses [N2, 4, 4]

    Returns:
        torch.Tensor: Distance matrix [N1, N2]
    """
    # Extract positions [N1, 3] and [N2, 3]
    pos1 = poses1[:, :3, 3]  # [N1, 3]
    pos2 = poses2[:, :3, 3]  # [N2, 3]

    # Compute pairwise position distances
    pos1_expanded = pos1.unsqueeze(1)  # [N1, 1, 3]
    pos2_expanded = pos2.unsqueeze(0)  # [1, N2, 3]
    pos_dist = torch.norm(pos1_expanded - pos2_expanded, dim=2)  # [N1, N2]

    # Extract rotation matrices [N1, 3, 3] and [N2, 3, 3]
    R1 = poses1[:, :3, :3]  # [N1, 3, 3]
    R2 = poses2[:, :3, :3]  # [N2, 3, 3]

    # Compute relative rotations
    R1_expanded = R1.unsqueeze(1)  # [N1, 1, 3, 3]
    R2_expanded = R2.unsqueeze(0)  # [1, N2, 3, 3]
    R_diff = torch.matmul(R1_expanded, R2_expanded.transpose(-2, -1))  # [N1, N2, 3, 3]

    # Reshape for so3_log_map
    N1, N2 = R_diff.shape[:2]
    R_diff_flat = R_diff.reshape(-1, 3, 3)

    # Apply so3_log_map and compute rotation distances
    log_maps = so3.so3_log_map(R_diff_flat)  # [-1, 3]
    rot_dist = torch.norm(log_maps, dim=1).reshape(N1, N2) / torch.pi

    # Combine distances with equal weights
    return pos_dist + rot_dist


def compute_pose_emd(poses1: torch.Tensor, poses2: torch.Tensor) -> float:
    """
    Compute EMD between two sets of poses with equal weighting of position and rotation.

    Args:
        poses1: First set of poses [N1, 4, 4]
        poses2: Second set of poses [N2, 4, 4]

    Returns:
        float: EMD distance considering both position and rotation equally
    """
    # Ensure input is torch tensor
    if isinstance(poses1, np.ndarray):
        poses1 = torch.from_numpy(poses1).float()
    if isinstance(poses2, np.ndarray):
        poses2 = torch.from_numpy(poses2).float()

    # Compute cost matrix using vectorized operations
    M = compute_pose_distance_batch(poses1, poses2)

    # Convert to numpy for linear_sum_assignment
    M_np = M.cpu().numpy()

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(M_np)

    # Compute the total cost and normalize
    emd = M_np[row_ind, col_ind].sum() / len(row_ind)

    return float(emd)
