"""
Evaluation metrics for FlowSteer-Motion.

Two metric families:

1. ConstraintMetrics  — measure how well the generated motion satisfies
   the imposed constraints.  These are the PRIMARY contribution metrics.

   foot_sliding_score       – mean foot velocity at detected contact frames (↓ better)
   contact_violation_rate   – fraction of contact frames where foot is airborne (↓ better)
   floor_penetration_depth  – mean below-floor distance at contact frames (↓ better)
   terminal_position_error  – L2 at final-frame root/joint target (↓ better)
   waypoint_mae             – mean L2 across sparse keyframe targets (↓ better)

2. MotionQualityMetrics  — measure how much the steering degrades generation quality.
   No pre-trained evaluator required; these are model-agnostic proxies.

   mean_jerk          – mean 3rd-order joint acceleration (↓ = smoother)
   kinematic_variance – std of per-frame root speed (diversity proxy)
   foot_floor_dist    – mean minimum foot height (↑ = less floor penetration)

   NOTE: For FID / R-Precision you need a pre-trained motion evaluator
         (e.g. MoMask's Comp_v6_KLD005).  We provide a stub
         `compute_fid_rprec` that calls into that evaluator when available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Joint / skeleton constants
# ---------------------------------------------------------------------------

FOOT_JOINTS   = [7, 8, 10, 11]   # L_Ankle, R_Ankle, L_Foot, R_Foot
ANKLE_JOINTS  = [7, 8]
TOE_JOINTS    = [10, 11]
ROOT_JOINT    = 0                 # Pelvis

# Pose canonicalization constants (mirrors PoseConstraint)
_POSE_ROOT      = 0
_POSE_LEFT_HIP  = 1
_POSE_RIGHT_HIP = 2

# Contact detection defaults (shared with FootContactConstraint)
DEFAULT_HEIGHT_THRESH = 0.05     # metres above estimated floor
DEFAULT_VEL_THRESH    = 0.02     # metres per frame


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _canonicalize_frame_np(joints_frame: np.ndarray) -> np.ndarray:
    """
    Root-centred + yaw-aligned canonicalization in numpy.

    Mirrors PoseConstraint.canonicalize() so metric space matches constraint space.

    Args:
        joints_frame: (22, 3) world-space joint positions

    Returns:
        (22, 3) canonical-space joint positions
    """
    root    = joints_frame[_POSE_ROOT]
    centred = joints_frame - root
    hip_vec = joints_frame[_POSE_RIGHT_HIP] - joints_frame[_POSE_LEFT_HIP]
    hx, hz  = hip_vec[0], hip_vec[2]
    norm    = math.sqrt(hx**2 + hz**2) + 1e-6
    fx, fz  = -hz / norm, hx / norm
    yaw     = math.atan2(fx, fz)
    cos_y   = math.cos(yaw)
    sin_y   = math.sin(yaw)
    x, y, z = centred[:, 0], centred[:, 1], centred[:, 2]
    x_rot   = x * cos_y - z * sin_y
    z_rot   = x * sin_y + z * cos_y
    return np.stack([x_rot, y, z_rot], axis=-1)  # (22, 3)

def _soft_contact_mask(
    joints_np: np.ndarray,
    foot_idx: List[int] = FOOT_JOINTS,
    height_thresh: float = DEFAULT_HEIGHT_THRESH,
    vel_thresh: float = DEFAULT_VEL_THRESH,
    sharpness: float = 20.0,
) -> np.ndarray:
    """
    Compute soft contact mask for foot joints.

    Args:
        joints_np: (T, J, 3) world-space joint positions

    Returns:
        contact: (T-1, len(foot_idx)) float array in [0, 1]
    """
    feet = joints_np[:, foot_idx, :]               # (T, 4, 3)
    foot_y = feet[:, :, 1]                         # (T, 4) y-axis

    # Floor at minimum y across sequence
    floor_y = foot_y.min(axis=0, keepdims=True)    # (1, 4)
    rel_h = foot_y - floor_y                        # (T, 4)

    # Per-frame foot speed
    foot_vel = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)  # (T-1, 4)

    # Sigmoid approximation of hard threshold
    h_mask = 1.0 / (1.0 + np.exp( sharpness * (rel_h[:-1] - height_thresh)))
    v_mask = 1.0 / (1.0 + np.exp( sharpness * (foot_vel - vel_thresh)))
    return h_mask * v_mask   # (T-1, 4)


def _hard_contact_mask(
    joints_np: np.ndarray,
    foot_idx: List[int] = FOOT_JOINTS,
    height_thresh: float = DEFAULT_HEIGHT_THRESH,
    vel_thresh: float = DEFAULT_VEL_THRESH,
) -> np.ndarray:
    """
    Binary contact mask (bool).

    Returns:
        contact: (T-1, len(foot_idx)) bool array
    """
    feet = joints_np[:, foot_idx, :]
    foot_y = feet[:, :, 1]
    floor_y = foot_y.min(axis=0, keepdims=True)
    rel_h = foot_y - floor_y
    foot_vel = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    return (rel_h[:-1] < height_thresh) & (foot_vel < vel_thresh)


# ---------------------------------------------------------------------------
# ConstraintMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConstraintMetrics:
    """
    Constraint satisfaction scores for a batch of generated motions.
    Each field is a float (mean across batch and foot joints / waypoints).
    """
    # Foot contact
    foot_sliding_score: float = 0.0        # mean foot vel at contact frames (m/frame)
    contact_violation_rate: float = 0.0    # fraction of contact frames where foot is airborne
    floor_penetration_depth: float = 0.0  # mean negative height at contact frames (m)

    # Terminal / waypoint / pose (NaN when not applicable)
    terminal_position_error: float = float("nan")
    waypoint_mae: float = float("nan")
    pose_hit_error: float = float("nan")   # mean L2 in canonical pose space at keyframe(s)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __repr__(self) -> str:
        lines = ["ConstraintMetrics:"]
        for k, v in self.to_dict().items():
            lines.append(f"  {k:32s} = {v:.5f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# MotionQualityMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class MotionQualityMetrics:
    """
    Motion quality scores (no pre-trained evaluator required).
    """
    mean_jerk: float = 0.0              # mean 3rd-order acceleration (smoothness)
    kinematic_variance: float = 0.0    # std of per-frame root speed (diversity)
    foot_floor_dist: float = 0.0       # mean minimum foot height (floor alignment)

    # Placeholders: require pre-trained evaluator (MoMask Comp_v6 etc.)
    fid: float = float("nan")
    r_precision_top1: float = float("nan")
    r_precision_top3: float = float("nan")

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def __repr__(self) -> str:
        lines = ["MotionQualityMetrics:"]
        for k, v in self.to_dict().items():
            tag = "(requires evaluator)" if k in ("fid", "r_precision_top1", "r_precision_top3") else ""
            lines.append(f"  {k:32s} = {v:.5f}  {tag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_constraint_metrics(
    joints_batch: np.ndarray,
    terminal_targets: Optional[List[Tuple[List[int], np.ndarray]]] = None,
    waypoints: Optional[List[List[Tuple[int, int, np.ndarray]]]] = None,
    pose_targets: Optional[List[Tuple[float, np.ndarray, Optional[List[int]]]]] = None,
    height_thresh: float = DEFAULT_HEIGHT_THRESH,
    vel_thresh: float = DEFAULT_VEL_THRESH,
) -> ConstraintMetrics:
    """
    Compute constraint satisfaction metrics for a batch of motions.

    Args:
        joints_batch: (B, T, 22, 3)  world-space joint positions
        terminal_targets: per-sample list of (joint_indices, target_xyz (K,3))
                          or None if no terminal constraint was applied
        waypoints: per-sample list of (frame_idx, joint_idx, target_xyz (3,))
                   or None
        pose_targets: list of (t_norm, target_pose (22,3), joint_mask) tuples
                      shared across the batch; or None
        height_thresh, vel_thresh: contact detection thresholds (match training)

    Returns:
        ConstraintMetrics with mean values across the batch
    """
    B, T, J, _ = joints_batch.shape

    # ---- Foot contact metrics ----
    all_sliding, all_violation, all_penetration = [], [], []

    for b in range(B):
        j = joints_batch[b]   # (T, 22, 3)

        soft = _soft_contact_mask(j, FOOT_JOINTS, height_thresh, vel_thresh)  # (T-1, 4)
        hard = _hard_contact_mask(j, FOOT_JOINTS, height_thresh, vel_thresh)  # (T-1, 4)

        feet = j[:, FOOT_JOINTS, :]              # (T, 4, 3)
        foot_y = feet[:, :, 1]
        floor_y = foot_y.min(axis=0)             # (4,)
        rel_h = foot_y - floor_y                  # (T, 4)

        foot_vel = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)  # (T-1, 4)

        # Foot sliding: mean velocity weighted by soft contact
        w = soft.sum()
        sliding = (soft * foot_vel).sum() / (w + 1e-8)
        all_sliding.append(sliding)

        # Contact violation: fraction of height-contact frames where foot is sliding
        # A "contact" frame = foot near floor (height < thresh).
        # Violation = foot is near floor but still moving fast (sliding).
        in_contact_by_height = rel_h[:-1] < height_thresh   # (T-1, 4) bool
        if in_contact_by_height.any():
            sliding_violation = in_contact_by_height & (foot_vel > vel_thresh)
            violation_rate = sliding_violation.sum() / (in_contact_by_height.sum() + 1e-8)
        else:
            violation_rate = 0.0
        all_violation.append(violation_rate)

        # Floor penetration: how far below estimated floor when in contact
        below_floor = np.clip(-rel_h[:-1], 0, None)   # positive = below floor
        penetration = (soft * below_floor).sum() / (w + 1e-8)
        all_penetration.append(penetration)

    metrics = ConstraintMetrics(
        foot_sliding_score=float(np.mean(all_sliding)),
        contact_violation_rate=float(np.mean(all_violation)),
        floor_penetration_depth=float(np.mean(all_penetration)),
    )

    # ---- Terminal position error ----
    if terminal_targets is not None:
        errors = []
        for b, (j_indices, target_xyz) in enumerate(terminal_targets):
            # target_xyz: (K, 3)
            pred = joints_batch[b, -1, j_indices, :]   # (K, 3)  last frame
            err = np.linalg.norm(pred - target_xyz, axis=-1).mean()
            errors.append(err)
        metrics.terminal_position_error = float(np.mean(errors))

    # ---- Waypoint MAE ----
    if waypoints is not None:
        all_errs = []
        for b, wps in enumerate(waypoints):
            for (t_idx, j_idx, tgt) in wps:
                if t_idx < 0:
                    t_idx = T + t_idx
                t_idx = max(0, min(t_idx, T - 1))
                pred = joints_batch[b, t_idx, j_idx]   # (3,)
                all_errs.append(np.linalg.norm(pred - tgt))
        if all_errs:
            metrics.waypoint_mae = float(np.mean(all_errs))

    # ---- Pose hit error ----
    if pose_targets is not None:
        all_errs = []
        for b in range(B):
            for (t_norm, target_pose, joint_mask) in pose_targets:
                frame = int(round(t_norm * (T - 1)))
                frame = max(0, min(frame, T - 1))
                canonical = _canonicalize_frame_np(joints_batch[b, frame])  # (22, 3)
                if joint_mask is not None:
                    canonical    = canonical[joint_mask]
                    target_sel   = target_pose[joint_mask]
                else:
                    target_sel   = target_pose
                err = np.linalg.norm(canonical - target_sel, axis=-1).mean()
                all_errs.append(err)
        if all_errs:
            metrics.pose_hit_error = float(np.mean(all_errs))

    return metrics


def pipeline_output_to_world_joints(output_dict: dict) -> np.ndarray:
    """
    Convert pipeline.generate() output to world-space joint positions.

    pipeline.generate() returns:
        keypoints3d: (B, T, 52, 3)  LOCAL-space (FK without global transl)
        transl:      (B, T, 3)      global root translation

    World-space = local_joints + transl

    Returns:
        world_joints: (B, T, 22, 3)  first 22 body joints in world space
    """
    import torch
    k3d = output_dict["keypoints3d"]   # (B, T, J, 3)
    transl = output_dict["transl"]     # (B, T, 3)

    if isinstance(k3d, torch.Tensor):
        k3d = k3d.numpy()
    if isinstance(transl, torch.Tensor):
        transl = transl.numpy()

    # Add global translation to get world-space positions
    world = k3d[:, :, :22, :] + transl[:, :, np.newaxis, :]  # (B, T, 22, 3)
    return world


def compute_quality_metrics(
    joints_batch: np.ndarray,
) -> MotionQualityMetrics:
    """
    Compute model-agnostic motion quality metrics.

    Args:
        joints_batch: (B, T, 22, 3)  world-space joint positions
                      Use pipeline_output_to_world_joints() to convert
                      from pipeline.generate() output.

    Returns:
        MotionQualityMetrics
    """
    B, T, J, _ = joints_batch.shape

    jerks, variances, floor_dists = [], [], []

    for b in range(B):
        j = joints_batch[b]   # (T, 22, 3)

        # Jerk: 3rd finite difference of position (all joints)
        if T >= 4:
            vel  = j[1:]   - j[:-1]     # (T-1, J, 3)
            acc  = vel[1:] - vel[:-1]   # (T-2, J, 3)
            jerk = acc[1:] - acc[:-1]   # (T-3, J, 3)
            mean_jerk = np.linalg.norm(jerk, axis=-1).mean()
        else:
            mean_jerk = 0.0
        jerks.append(mean_jerk)

        # Root (Pelvis) speed variance in world space — diversity proxy.
        # In world space, root moves as the character walks/runs.
        root = j[:, ROOT_JOINT, :]                                      # (T, 3)
        root_vel = np.linalg.norm(root[1:] - root[:-1], axis=-1)       # (T-1,) m/frame
        variances.append(float(root_vel.std()))

        # Floor distance: mean relative foot height above estimated floor
        feet_y = j[:, FOOT_JOINTS, 1]    # (T, 4) world y
        floor_y = feet_y.min(axis=0)     # (4,)   estimated floor per foot
        rel_h = (feet_y - floor_y).mean()
        floor_dists.append(rel_h)

    return MotionQualityMetrics(
        mean_jerk=float(np.mean(jerks)),
        kinematic_variance=float(np.mean(variances)),
        foot_floor_dist=float(np.mean(floor_dists)),
    )


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    joints_batch: np.ndarray,
    terminal_targets: Optional[List] = None,
    waypoints: Optional[List] = None,
) -> Dict[str, float]:
    """
    Compute all available metrics and return as flat dict.

    Args:
        joints_batch: (B, T, 22, 3) numpy array

    Returns:
        flat dict of metric_name → float
    """
    c = compute_constraint_metrics(joints_batch, terminal_targets, waypoints)
    q = compute_quality_metrics(joints_batch)
    return {**c.to_dict(), **q.to_dict()}


# ---------------------------------------------------------------------------
# FID / R-Precision stub (requires external evaluator)
# ---------------------------------------------------------------------------

def compute_fid_rprec(
    joints_batch: np.ndarray,
    texts: List[str],
    evaluator_wrapper=None,
) -> Dict[str, float]:
    """
    Compute FID and R-Precision using a pre-trained motion-text evaluator.

    Requires:
        evaluator_wrapper: EvaluatorModelWrapper from MoMask
                           (models/t2m_eval_wrapper.py)

    If evaluator_wrapper is None, returns NaN for all metrics.

    The pipeline is:
        joints (B, T, 22, 3)
          → convert to HumanML3D 263D format  [external conversion needed]
          → evaluator.get_motion_embeddings()
          → evaluator.get_text_embeddings()
          → compute FID, R-Precision in evaluator space

    NOTE: HY-Motion uses o6dp 201D representation; HumanML3D uses 263D
    local-coordinate features.  Direct feature comparison is not valid.
    You should either:
      (a) Fit an evaluator trained on HY-Motion features (preferred), or
      (b) Convert HY-Motion output to HumanML3D features via FK + normalization
          and use the MoMask evaluator.
    """
    if evaluator_wrapper is None:
        return {
            "fid": float("nan"),
            "r_precision_top1": float("nan"),
            "r_precision_top3": float("nan"),
        }

    # --- Placeholder: implement conversion + evaluator call here ---
    raise NotImplementedError(
        "compute_fid_rprec: implement HY-Motion → HumanML3D feature conversion, "
        "then call evaluator_wrapper.get_motion_embeddings()."
    )


# ---------------------------------------------------------------------------
# Pretty-print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    baseline_metrics: Dict[str, float],
    steered_metrics: Dict[str, float],
    title: str = "Metric Comparison",
) -> None:
    """Print a side-by-side comparison table of baseline vs steered metrics."""
    all_keys = list(baseline_metrics.keys())
    col_w = 32

    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Metric':<{col_w}} {'Baseline':>12}  {'Steered':>12}  {'Δ':>10}")
    print(f"  {'-'*col_w} {'----------':>12}  {'----------':>12}  {'----------':>10}")

    for k in all_keys:
        b_val = baseline_metrics.get(k, float("nan"))
        s_val = steered_metrics.get(k, float("nan"))
        if np.isnan(b_val) or np.isnan(s_val):
            delta_str = "    N/A"
        else:
            delta = s_val - b_val
            delta_str = f"{delta:+10.5f}"
        print(f"  {k:<{col_w}} {b_val:>12.5f}  {s_val:>12.5f}  {delta_str}")

    print(f"{'='*70}\n")
