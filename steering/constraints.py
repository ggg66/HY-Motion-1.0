"""
Differentiable constraint losses for FlowSteer-Motion.

All constraints receive `joints: (B, T, 22, 3)` world-space joint positions
(output of MotionDecoder) and return a scalar loss tensor.

Joint index reference (SMPL-H 22-joint body):
    0  Pelvis    7  L_Ankle   10 L_Foot   15 Head
    1  L_Hip     8  R_Ankle   11 R_Foot   16 L_Shoulder
    2  R_Hip     9  Spine3    12 Neck     17 R_Shoulder
    3  Spine1   ...           13 L_Collar 20 L_Wrist
    4  L_Knee                 14 R_Collar 21 R_Wrist
    5  R_Knee
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .decode import FOOT_JOINTS, ROOT_JOINT


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseConstraint:
    """Interface for all constraint losses."""

    weight: float = 1.0

    def loss(self, joints: Tensor) -> Tensor:
        """
        Args:
            joints: (B, T, 22, 3) world-space joint positions

        Returns:
            scalar loss (higher = further from constraint)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# CompositeConstraint
# ---------------------------------------------------------------------------

class CompositeConstraint(BaseConstraint):
    """
    Weighted sum of multiple constraints.

    Args:
        constraints:       list of (constraint, weight) pairs
        normalize_losses:  if True, divide each constraint's loss by its
                           detached value before combining, so every term
                           contributes ~1.0 regardless of absolute scale.
                           Fixes gradient-magnitude imbalance when mixing
                           constraints with very different loss magnitudes
                           (e.g. TerminalConstraint ~4 m² vs FootContact ~0.02).
                           A single backward pass is still used, preserving
                           FK-chain temporal coupling (unlike per-gradient
                           normalisation which causes jerk artefacts).

    Usage:
        composite = CompositeConstraint([
            (TerminalConstraint(...), 1.0),
            (FootContactConstraint(), 1.0),
        ], normalize_losses=True)
        loss = composite.loss(joints)
    """

    def __init__(
        self,
        constraints: List[Tuple[BaseConstraint, float]],
        normalize_losses: bool = False,
    ):
        self.constraints = constraints        # list of (constraint, weight)
        self.normalize_losses = normalize_losses

    def loss(self, joints: Tensor) -> Tensor:
        total = joints.new_zeros(())
        for constraint, w in self.constraints:
            l = constraint.loss(joints)
            if self.normalize_losses:
                l = l / (l.detach() + 1e-8)
            total = total + w * l
        return total

    def __call__(self, joints: Tensor) -> Tensor:
        return self.loss(joints)


# ---------------------------------------------------------------------------
# TerminalConstraint  (L_end)
# ---------------------------------------------------------------------------

class TerminalConstraint(BaseConstraint):
    """
    Penalize deviation from a target joint configuration at the end of motion.

    Args:
        target_joints:  (J, 3) or (K, 3) – target positions for specified joints
        joint_indices:  list of joint indices to constrain (length K)
        tail_frames:    number of final frames to average over (default 4 ≈ 0.13s)
        weight:         overall loss weight
    """

    def __init__(
        self,
        target_joints: Tensor,          # (K, 3) target positions
        joint_indices: List[int],        # which joints to constrain
        tail_frames: int = 4,
        weight: float = 1.0,
    ):
        self.register_target(target_joints, joint_indices)
        self.tail_frames = tail_frames
        self.weight = weight

    def register_target(self, target_joints: Tensor, joint_indices: List[int]):
        # Store as plain tensors (move to device on first call)
        self._target = target_joints.float()   # (K, 3)
        self._indices = joint_indices

    def loss(self, joints: Tensor) -> Tensor:
        """
        joints: (B, T, 22, 3)
        Returns MSE between tail frames and target for specified joints.
        """
        device = joints.device
        target = self._target.to(device)   # (K, 3)

        # Select tail frames and specified joints
        tail = joints[:, -self.tail_frames:, :, :]   # (B, tail, 22, 3)
        tail_k = tail[:, :, self._indices, :]         # (B, tail, K, 3)
        target_exp = target.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 3)

        return F.mse_loss(tail_k, target_exp.expand_as(tail_k))


# ---------------------------------------------------------------------------
# TrajectoryConstraint  (L_traj)
# ---------------------------------------------------------------------------

class TrajectoryConstraint(BaseConstraint):
    """
    Sparse keyframe constraints: at specified frame indices, joints must be
    near target positions.

    Args:
        waypoints: list of (frame_idx, joint_idx, target_xyz) tuples
                   frame_idx can be negative (Python-style, -1 = last frame)
        weight:    overall loss weight
    """

    def __init__(
        self,
        waypoints: List[Tuple[int, int, Tensor]],
        weight: float = 1.0,
    ):
        self.waypoints = waypoints   # [(frame_idx, joint_idx, (3,) tensor), ...]
        self.weight = weight

    def loss(self, joints: Tensor) -> Tensor:
        """
        joints: (B, T, 22, 3)
        """
        device = joints.device
        T = joints.shape[1]
        total = joints.new_zeros(())

        for (t_idx, j_idx, target_xyz) in self.waypoints:
            # Support negative indexing
            if t_idx < 0:
                t_idx = T + t_idx
            t_idx = max(0, min(t_idx, T - 1))

            pred = joints[:, t_idx, j_idx, :]   # (B, 3)
            tgt = target_xyz.to(device).float()  # (3,)
            total = total + F.mse_loss(pred, tgt.unsqueeze(0).expand_as(pred))

        return total / max(len(self.waypoints), 1)


# ---------------------------------------------------------------------------
# FootContactConstraint  (L_contact)
# ---------------------------------------------------------------------------

class FootContactConstraint(BaseConstraint):
    """
    Penalizes foot sliding and levitation during contact phases.

    Contact detection is differentiable: uses sigmoid on relative foot height
    and foot velocity to produce a soft contact mask.

    Two penalties:
        1. velocity loss:  at contact frames, foot velocity ≈ 0
        2. height loss:    at contact frames, foot height ≈ floor height

    Floor height is estimated as the minimum foot height across the sequence
    (relative detection), avoiding dependence on the absolute y-axis offset.

    Args:
        foot_joint_indices: which joints to treat as feet (default: standard 4)
        height_thresh:      relative height above floor below which contact is
                            assumed (metres; default 0.05)
        vel_thresh:         per-frame velocity above which contact is penalized
                            (default 0.02)
        sigmoid_sharpness:  steepness of the soft contact mask sigmoid
        weight:             overall loss weight
    """

    def __init__(
        self,
        foot_joint_indices: List[int] = FOOT_JOINTS,
        height_thresh: float = 0.05,
        vel_thresh: float = 0.02,
        sigmoid_sharpness: float = 20.0,
        weight: float = 1.0,
        detach_mask: bool = True,
    ):
        self.foot_joints = foot_joint_indices
        self.height_thresh = height_thresh
        self.vel_thresh = vel_thresh
        self.sharpness = sigmoid_sharpness
        self.weight = weight
        self.detach_mask = detach_mask

    def loss(self, joints: Tensor) -> Tensor:
        """
        joints: (B, T, 22, 3)
        """
        feet = joints[:, :, self.foot_joints, :]   # (B, T, 4, 3)

        if self.detach_mask:
            # --- Contact mask: computed on detached joints so the gradient cannot
            #     "cheat" by reclassifying frames as non-contact (e.g. lifting feet).
            #     The mask is a fixed binary-ish indicator; only velocity and height
            #     are differentiable. ---
            with torch.no_grad():
                foot_y_det = feet[..., 1].detach()                            # (B, T, 4)
                floor_y_det = foot_y_det.min(dim=1, keepdim=True).values      # (B, 1, 4)
                rel_h_det = foot_y_det - floor_y_det                          # (B, T, 4)
                fvel_det = (feet[:, 1:] - feet[:, :-1]).detach().norm(dim=-1) # (B, T-1, 4)
                h_mask = torch.sigmoid(-self.sharpness * (rel_h_det[:, :-1] - self.height_thresh))
                v_mask = torch.sigmoid(-self.sharpness * (fvel_det - self.vel_thresh))
                contact = h_mask * v_mask   # (B, T-1, 4)  [detached]
        else:
            # --- Ablation: differentiable contact mask (gradient can cheat) ---
            foot_y_nd = feet[..., 1]                                          # (B, T, 4)
            floor_y_nd = foot_y_nd.min(dim=1, keepdim=True).values           # (B, 1, 4)
            rel_h_nd = foot_y_nd - floor_y_nd                                # (B, T, 4)
            fvel_nd = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)             # (B, T-1, 4)
            h_mask = torch.sigmoid(-self.sharpness * (rel_h_nd[:, :-1] - self.height_thresh))
            v_mask = torch.sigmoid(-self.sharpness * (fvel_nd - self.vel_thresh))
            contact = h_mask * v_mask   # (B, T-1, 4)  [differentiable]

        # --- Differentiable velocity and height: gradient flows here only ---
        foot_vel = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)   # (B, T-1, 4)
        foot_y = feet[..., 1]                                    # (B, T, 4)
        # Detach floor estimate: prevent gradient from "lowering the floor" to
        # trivially satisfy rel_height ≈ 0.
        floor_y = foot_y.min(dim=1, keepdim=True).values.detach()  # (B, 1, 4)
        rel_height = foot_y - floor_y                               # (B, T, 4)

        # --- Velocity penalty at contact frames ---
        vel_loss = (contact * foot_vel).mean()

        # --- Height penalty: foot should be at floor level when in contact ---
        height_loss = (contact * rel_height[:, :-1]).mean()

        return vel_loss + height_loss


# ---------------------------------------------------------------------------
# RootTrajectoryConstraint  (helper for locomotion tasks)
# ---------------------------------------------------------------------------

class RootTrajectoryConstraint(BaseConstraint):
    """
    Constrain the root joint (Pelvis, idx=0) to follow a 2D XZ path.

    Useful for "walk from A to B" scenarios where exact timing is flexible.

    Args:
        waypoints_xz:  (N, 2) target XZ positions at uniformly spaced frames
        weight:        overall loss weight
    """

    def __init__(self, waypoints_xz: Tensor, weight: float = 1.0):
        self._waypoints_xz = waypoints_xz.float()   # (N, 2)
        self.weight = weight

    def loss(self, joints: Tensor) -> Tensor:
        device = joints.device
        B, T, _, _ = joints.shape
        root_xz = joints[:, :, ROOT_JOINT, [0, 2]]   # (B, T, 2)

        waypoints = self._waypoints_xz.to(device)    # (N, 2)
        N = waypoints.shape[0]

        # Sample root trajectory at N uniformly-spaced frames
        indices = torch.linspace(0, T - 1, N, device=device).long()
        sampled = root_xz[:, indices, :]             # (B, N, 2)

        return F.mse_loss(sampled, waypoints.unsqueeze(0).expand_as(sampled))
