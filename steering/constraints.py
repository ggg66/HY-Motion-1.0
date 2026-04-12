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

    def temporal_mask(self, T: int, device) -> Optional[Tensor]:
        """
        Return a (T,) soft window in [0, 1] indicating where this constraint
        should focus its gradient energy.  Frames with low weight are masked
        before normalization so they don't receive spurious steering.

        Default: None (no masking — uniform over all valid frames).
        Override in subclasses with time-localised constraints.
        """
        return None


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

    def temporal_mask(self, T: int, device) -> Optional[Tensor]:
        """Aggregate sub-constraint masks by element-wise maximum."""
        mask = None
        for c, _ in self.constraints:
            m = c.temporal_mask(T, device)
            if m is not None:
                mask = m if mask is None else torch.maximum(mask, m)
        return mask

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


# ---------------------------------------------------------------------------
# WaypointConstraint  (L_waypoint)
# ---------------------------------------------------------------------------

class WaypointConstraint(BaseConstraint):
    """
    Multi-waypoint root trajectory constraint with temporal soft windows.

    Constrains the root joint (pelvis) XZ position to pass through a sequence
    of target positions at specified normalised times.  Only horizontal (XZ)
    position is constrained; height (Y) remains free.

    Complements PoseConstraint (body configuration) and
    FootContactConstraint (physical stability): spatial path is this
    constraint's sole responsibility.

    Args:
        waypoints:   list of (t_norm, target) pairs where
                     t_norm  ∈ [0, 1]  is normalised sequence time and
                     target  is a (2,) [XZ] or (3,) [XYZ, Y ignored] tensor.
        weight:      overall loss weight.
        sigma_frac:  temporal Gaussian σ as a fraction of sequence length.
                     Smaller → tighter time pinning.  Default 0.05 ≈ 5 %.
    """

    def __init__(
        self,
        waypoints: List[Tuple[float, Tensor]],
        weight: float = 1.0,
        sigma_frac: float = 0.05,
    ):
        self.waypoints   = [(float(t), w.float()) for t, w in waypoints]
        self.weight      = weight
        self.sigma_frac  = sigma_frac

    def loss(self, joints: Tensor) -> Tensor:
        B, T, _, _ = joints.shape
        device = joints.device
        root_xz = joints[:, :, ROOT_JOINT, [0, 2]]      # (B, T, 2)

        sigma = max(1.0, self.sigma_frac * T)
        t_idx = torch.arange(T, device=device, dtype=torch.float32)
        total = joints.new_zeros(())

        for t_norm, target in self.waypoints:
            t_frame = float(t_norm) * (T - 1)
            # Temporal Gaussian window, normalised to sum = 1
            w = torch.exp(-0.5 * ((t_idx - t_frame) / sigma) ** 2)
            w = w / w.sum()                              # (T,)

            tgt = target.to(device)
            if tgt.shape[-1] == 3:
                tgt = tgt[[0, 2]]                        # extract XZ from XYZ

            # Weighted MSE over time
            diff     = (root_xz - tgt.view(1, 1, 2)) ** 2   # (B, T, 2)
            weighted = (w.view(1, T, 1) * diff).sum() / B
            total    = total + weighted

        return total

    def temporal_mask(self, T: int, device) -> Optional[Tensor]:
        """
        (T,) soft window in [0, 1]: element-wise max of Gaussians around each waypoint.

        Focuses gradient energy near the time of each waypoint, suppressing
        spurious steering updates at frames that are far from any constraint target.
        """
        sigma = max(1.0, self.sigma_frac * T)
        t_idx = torch.arange(T, device=device, dtype=torch.float32)
        mask  = torch.zeros(T, device=device)
        for t_norm, _ in self.waypoints:
            t_frame = float(t_norm) * (T - 1)
            w = torch.exp(-0.5 * ((t_idx - t_frame) / sigma) ** 2)
            mask = torch.maximum(mask, w)
        return mask


# ---------------------------------------------------------------------------
# PoseConstraint  (L_pose)
# ---------------------------------------------------------------------------

# Hip indices used for yaw-canonicalization (SMPL-H 22-joint layout)
_LEFT_HIP  = 1
_RIGHT_HIP = 2

# Convenience body-part masks (subset of 0-21 joint indices)
UPPER_BODY_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_BODY_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
ARM_JOINTS        = [16, 17, 18, 19, 20, 21]
LEG_JOINTS        = [1, 2, 4, 5, 7, 8, 10, 11]

# Hierarchical joint groupings for weighted pose loss
TORSO_JOINTS        = [0, 3, 6, 9, 12, 15]           # Pelvis, Spine1-3, Neck, Head
LIMB_JOINTS         = [1, 2, 4, 5, 7, 8, 10, 11]     # Hips, Knees, Ankles, Feet
END_EFFECTOR_JOINTS = [10, 11, 20, 21]               # Feet, Wrists

# Per-joint importance weights for hierarchical pose loss (SMPL-H 22-joint)
# Torso/spine: low (global shape rarely the target of pose constraints)
# Upper limb end-effectors (shoulders, elbows, wrists): high
# Lower end-effectors (ankles, feet): medium-high
import torch as _torch
_HIER_WEIGHTS_22: Tensor = _torch.ones(22)
_HIER_WEIGHTS_22[[0, 3, 6, 9, 12, 15]] = 0.5           # torso/spine
_HIER_WEIGHTS_22[[16, 17, 18, 19, 20, 21]] = 3.0        # shoulders, elbows, wrists
_HIER_WEIGHTS_22[[7, 8, 10, 11]] = 2.0                  # ankles, feet
del _torch


class PoseConstraint(BaseConstraint):
    """
    Sparse keyframe pose constraint in canonical (root-aligned) space.

    Constrains body *configuration* — joint positions relative to the pelvis
    with global yaw removed — at one or more keyframes, without coupling to
    absolute world position (handled by Terminal/WaypointConstraint) or foot
    physics (handled by FootContactConstraint).

    Canonical space definition
    --------------------------
    Given world-space joints at frame t:
      1. Root-centred:  subtract pelvis (joint 0) position.
      2. Yaw-aligned:   rotate around Y-axis so the character faces +Z.
         Facing direction is derived from the left→right hip vector.
      3. Height (Y) is preserved.

    The canonicalization transform is computed from **detached** joints, so
    the gradient cannot shift the reference frame during optimisation — the
    same principle as FootContactConstraint's detached contact mask.

    Args:
        keyframes:      list of (t_norm, target_joints) where
                        t_norm         ∈ [0, 1] normalised sequence time,
                        target_joints  (22, 3) canonical-space joint positions.
                        Targets should themselves be canonicalized (pelvis
                        at origin, facing +Z).
        joint_mask:     list of joint indices to constrain.  None = all 22.
                        Use UPPER_BODY_JOINTS, ARM_JOINTS, etc. for partial
                        body control.
        sigma_frac:     temporal Gaussian σ as fraction of sequence length.
                        Larger → softer time pinning.  Default 0.04.
        facing_weight:  if > 0, add a cosine yaw-alignment term so the
                        character also faces the target direction in world
                        space (rather than only matching body shape).
        weight:         overall loss weight.
        mode:           "canonical" (default) — root-centred + yaw-aligned.
                        "world" — raw world-space, for ablation only; target
                        must be in world coordinates.
    """

    def __init__(
        self,
        keyframes: List[Tuple[float, Tensor]],
        joint_mask: Optional[List[int]] = None,
        sigma_frac: float = 0.04,
        facing_weight: float = 0.0,
        weight: float = 1.0,
        mode: str = "canonical",
        use_hierarchical: bool = False,
    ):
        assert mode in ("canonical", "world"), f"Unknown PoseConstraint mode: {mode!r}"
        self.keyframes        = [(float(t), kf.float()) for t, kf in keyframes]
        self.joint_mask       = joint_mask
        self.sigma_frac       = sigma_frac
        self.facing_weight    = facing_weight
        self.weight           = weight
        self.mode             = mode
        self.use_hierarchical = use_hierarchical

    # ------------------------------------------------------------------
    # Canonicalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def canonicalize(joints: Tensor) -> Tensor:
        """
        Map a joint sequence to root-centred + yaw-aligned canonical space.

        The yaw transform is computed from DETACHED joints (gradient cannot
        escape through the reference frame).

        joints:  (B, T, 22, 3)  world-space joint positions
        returns: (B, T, 22, 3)  canonical-space joint positions
        """
        with torch.no_grad():
            # Pelvis position (detached reference frame origin)
            root_det = joints[:, :, ROOT_JOINT, :].detach()          # (B, T, 3)

            # Hip vector → facing direction in XZ plane (detached)
            hip_vec = (
                joints[:, :, _RIGHT_HIP, :] - joints[:, :, _LEFT_HIP, :]
            ).detach()                                                 # (B, T, 3)
            hx, hz  = hip_vec[..., 0], hip_vec[..., 2]               # (B, T)
            norm    = (hx ** 2 + hz ** 2).sqrt().clamp(min=1e-6)
            # Forward direction perpendicular to hip in XZ: (-hz, hx)/norm
            fx, fz  = -hz / norm, hx / norm
            # Yaw angle: rotate so forward → +Z
            yaw     = torch.atan2(fx, fz)                             # (B, T)
            cos_y   = yaw.cos().unsqueeze(-1)                         # (B, T, 1)
            sin_y   = yaw.sin().unsqueeze(-1)                         # (B, T, 1)

        # Centre around root (differentiable)
        centred = joints - root_det.unsqueeze(2)                      # (B, T, 22, 3)
        x = centred[..., 0]   # (B, T, 22)
        y = centred[..., 1]
        z = centred[..., 2]

        # Rotate around Y-axis by +yaw:  x′ = x·cos − z·sin
        #                                 z′ = x·sin + z·cos
        x_rot = x * cos_y - z * sin_y
        z_rot = x * sin_y + z * cos_y
        return torch.stack([x_rot, y, z_rot], dim=-1)                 # (B, T, 22, 3)

    @staticmethod
    def _facing_loss(joints: Tensor, target_kf: Tensor) -> Tensor:
        """
        Cosine yaw-alignment loss: 1 − cos(Δyaw) ∈ [0, 2].

        joints:    (B, T, 22, 3)  world-space
        target_kf: (22, 3)        canonical-space target (used only for hip direction)
        """
        # Target facing yaw from canonical hip vector
        t_hip   = target_kf[_RIGHT_HIP] - target_kf[_LEFT_HIP]
        t_hx, t_hz = t_hip[0], t_hip[2]
        t_norm  = (t_hx ** 2 + t_hz ** 2).sqrt().clamp(min=1e-6)
        # In canonical space the character faces +Z, so target yaw = 0
        # (forward = +Z means fx=0, fz=1 → atan2(0,1)=0).
        # The target_yaw is implicitly 0; no need to compute from target_kf.
        target_yaw = joints.new_zeros(())

        hip_vec = (
            joints[:, :, _RIGHT_HIP, :] - joints[:, :, _LEFT_HIP, :]
        )                                                              # (B, T, 3)
        hx, hz   = hip_vec[..., 0], hip_vec[..., 2]
        norm     = (hx ** 2 + hz ** 2).sqrt().clamp(min=1e-6)
        fx, fz   = -hz / norm, hx / norm
        gen_yaw  = torch.atan2(fx, fz)                                # (B, T)
        delta    = gen_yaw - target_yaw
        return (1.0 - delta.cos()).mean(dim=0)                        # (T,)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, joints: Tensor) -> Tensor:
        """
        joints: (B, T, 22, 3)
        """
        B, T, _, _ = joints.shape
        device     = joints.device
        sigma      = max(1.0, self.sigma_frac * T)
        t_idx      = torch.arange(T, device=device, dtype=torch.float32)

        # Canonicalize full sequence once (efficient: no per-frame loop)
        if self.mode == "canonical":
            pose_seq = self.canonicalize(joints)                      # (B, T, 22, 3)
        else:
            pose_seq = joints

        total = joints.new_zeros(())

        for t_norm, target_kf in self.keyframes:
            t_frame = float(t_norm) * (T - 1)
            # Temporal Gaussian weights (normalised)
            w = torch.exp(-0.5 * ((t_idx - t_frame) / sigma) ** 2)
            w = w / w.sum()                                            # (T,)

            target = target_kf.to(device)                             # (22, 3)

            # Apply joint mask
            if self.joint_mask is not None:
                pose_m   = pose_seq[:, :, self.joint_mask, :]         # (B, T, J, 3)
                target_m = target[self.joint_mask, :]                  # (J, 3)
            else:
                pose_m   = pose_seq                                    # (B, T, 22, 3)
                target_m = target                                      # (22, 3)

            # Per-joint squared error, averaged over xyz
            diff          = (pose_m - target_m[None, None]) ** 2     # (B, T, J, 3)
            per_joint_mse = diff.mean(dim=-1)                         # (B, T, J)

            # Hierarchical weighting: wrists/elbows/shoulders high, torso low
            if self.use_hierarchical:
                joint_indices = self.joint_mask if self.joint_mask is not None \
                    else list(range(22))
                jw = _HIER_WEIGHTS_22[joint_indices].to(device)       # (J,)
                frame_loss = (jw * per_joint_mse).sum(dim=-1) / jw.sum()  # (B, T)
            else:
                frame_loss = per_joint_mse.mean(dim=-1)               # (B, T)

            total = total + (w * frame_loss).sum(dim=1).mean()        # scalar

            # Optional yaw-alignment term
            if self.facing_weight > 0.0 and self.mode == "canonical":
                face_loss = self._facing_loss(joints, target_kf.to(device))  # (T,)
                total     = total + self.facing_weight * (w * face_loss).sum()

        return total

    def temporal_mask(self, T: int, device) -> Optional[Tensor]:
        """
        (T,) soft window in [0, 1]: element-wise max of Gaussians around each keyframe.

        Focuses gradient energy near keyframe positions; frames far from every
        keyframe are attenuated so they don't receive spurious pose steering.
        The Gaussian width matches sigma_frac used in the loss itself.
        """
        sigma = max(1.0, self.sigma_frac * T)
        t_idx = torch.arange(T, device=device, dtype=torch.float32)
        mask  = torch.zeros(T, device=device)
        for t_norm, _ in self.keyframes:
            t_frame = float(t_norm) * (T - 1)
            w = torch.exp(-0.5 * ((t_idx - t_frame) / sigma) ** 2)
            mask = torch.maximum(mask, w)
        return mask
