"""
FlowSteer-Motion demo.

The original pose-keyframe demo is kept, but the recommended path is now
segment-level attribute editing via --edit_mode.  These edits expose user-level
controls such as "raise the right hand by 0.35 m" instead of asking users to
interpret a raw steering alpha.

Examples:
    python demo_steer.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person walks forward" \
        --duration 4.0 \
        --edit_mode raise_hand \
        --edit_delta_y 0.35 \
        --edit_t_start 0.30 --edit_t_end 0.70 \
        --alpha_max 20 \
        --output_dir output/raise_hand_demo

    python demo_steer.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person performs a side kick." \
        --duration 3.0 \
        --edit_mode higher_kick \
        --edit_delta_y 0.25 \
        --alpha_max 24 \
        --output_dir output/higher_kick_demo
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from hymotion.utils.loaders import load_object
from steering import (
    ARM_JOINTS,
    CompositeConstraint,
    FlowSteerer,
    FootContactConstraint,
    JointReachConstraint,
    PoseConstraint,
    RootDisplacementScaleConstraint,
    StagedScheduler,
    TemporalJointOffsetConstraint,
    TerminalConstraint,
    UPPER_BODY_JOINTS,
)


_JOINT_MASK_MAP = {
    "all": None,
    "upper_body": UPPER_BODY_JOINTS,
    "arms": ARM_JOINTS,
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "legs": [1, 2, 4, 5, 7, 8, 10, 11],
}

_EDIT_JOINTS: Dict[str, List[int]] = {
    "left_wrist": [20],
    "right_wrist": [21],
    "both_wrists": [20, 21],
    "left_arm": [18, 20],
    "right_arm": [19, 21],
    "both_arms": [18, 19, 20, 21],
    "left_foot": [7, 10],
    "right_foot": [8, 11],
    "both_feet": [7, 8, 10, 11],
}


def load_pipeline(model_path: str, device: torch.device):
    """Load MotionFlowMatching from a HY-Motion checkpoint directory."""
    cfg_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")
    assert os.path.exists(cfg_path), f"config.yml not found at {cfg_path}"
    assert os.path.exists(ckpt_path), f"latest.ckpt not found at {ckpt_path}"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    pipeline = load_object(
        cfg["train_pipeline"],
        cfg["train_pipeline_args"],
        network_module=cfg["network_module"],
        network_module_args=cfg["network_module_args"],
    )
    pipeline.load_in_demo(ckpt_path, build_text_encoder=True)
    pipeline.to(device)
    pipeline.eval()
    return pipeline


def canonicalize_frame_np(joints_frame: np.ndarray) -> np.ndarray:
    """Root-center and yaw-align one 22-joint frame to match PoseConstraint."""
    root = joints_frame[0]
    centered = joints_frame - root
    hip_vec = joints_frame[2] - joints_frame[1]
    hx, hz = hip_vec[0], hip_vec[2]
    norm = np.sqrt(hx ** 2 + hz ** 2) + 1e-6
    fx, fz = -hz / norm, hx / norm
    yaw = np.arctan2(fx, fz)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x, y, z = centered[:, 0], centered[:, 1], centered[:, 2]
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y
    return np.stack([x_rot, y, z_rot], axis=-1)


def pipeline_output_to_world_joints(output: dict) -> np.ndarray:
    """Convert HY-Motion output dict to (B, T, 22, 3) world-space joints."""
    k3d = output["keypoints3d"].numpy()
    transl = output["transl"].numpy()
    return k3d[:, :, :22, :] + transl[:, :, np.newaxis, :]


def save_joints_npy(output_dir: str, tag: str, keypoints3d: np.ndarray):
    """Save (B, T, J, 3) keypoints as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    for b in range(keypoints3d.shape[0]):
        path = os.path.join(output_dir, f"{tag}_seed{b}.npy")
        np.save(path, keypoints3d[b])
        print(f"  Saved: {path}")


def pose_hit_error(joints_np: np.ndarray, t_norm: float, target: np.ndarray, joint_mask) -> float:
    """Mean L2 canonical pose error for one motion."""
    T = joints_np.shape[0]
    frame = int(round(t_norm * (T - 1)))
    frame = max(0, min(frame, T - 1))
    pred = canonicalize_frame_np(joints_np[frame])
    if joint_mask is not None:
        pred = pred[joint_mask]
        target = target[joint_mask]
    return float(np.linalg.norm(pred - target, axis=-1).mean())


def _window_indices(T: int, t_start: float, t_end: float) -> np.ndarray:
    lo = int(round(t_start * (T - 1)))
    hi = int(round(t_end * (T - 1))) + 1
    lo = max(0, min(lo, T - 1))
    hi = max(lo + 1, min(hi, T))
    return np.arange(lo, hi)


def _mean_joint_delta(
    baseline: np.ndarray,
    steered: np.ndarray,
    joint_indices: List[int],
    direction: np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """Per-sample average displacement along direction in the edit window."""
    idx = _window_indices(baseline.shape[1], t_start, t_end)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    delta = steered[:, idx][:, :, joint_indices, :] - baseline[:, idx][:, :, joint_indices, :]
    return np.tensordot(delta, direction, axes=([-1], [0])).mean(axis=(1, 2))


def _mean_jerk(joints: np.ndarray) -> float:
    """Mean third finite difference magnitude over batch/time/joints."""
    if joints.shape[1] < 4:
        return 0.0
    jerk = np.diff(joints, n=3, axis=1)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def _edit_achievement(args, baseline_joints: np.ndarray, steered_joints: np.ndarray) -> np.ndarray:
    """Return per-sample achievement percentage for edit modes with scalar targets."""
    if args.edit_mode in ("raise_hand", "higher_kick"):
        joint_indices = _EDIT_JOINTS[args.edit_joint]
        direction = np.array([args.edit_delta_x, args.edit_delta_y, args.edit_delta_z], dtype=np.float32)
        achieved = _mean_joint_delta(
            baseline_joints,
            steered_joints,
            joint_indices,
            direction,
            args.edit_t_start,
            args.edit_t_end,
        )
        target = np.linalg.norm(direction)
        return achieved / (target + 1e-8) * 100.0

    if args.edit_mode == "reach_hand":
        joint_indices = _EDIT_JOINTS[args.edit_joint]
        frame = int(round(args.edit_target_t * (baseline_joints.shape[1] - 1)))
        source = baseline_joints[:, frame, joint_indices, :].mean(axis=(0, 1))
        target = source + np.array([args.edit_delta_x, args.edit_delta_y, args.edit_delta_z], dtype=np.float32)
        b_dist = np.linalg.norm(baseline_joints[:, frame][:, joint_indices, :] - target, axis=-1).mean(axis=1)
        s_dist = np.linalg.norm(steered_joints[:, frame][:, joint_indices, :] - target, axis=-1).mean(axis=1)
        return (b_dist - s_dist) / (b_dist + 1e-8) * 100.0

    if args.edit_mode == "faster_root":
        b = baseline_joints[:, -1, 0, [0, 2]] - baseline_joints[:, 0, 0, [0, 2]]
        s = steered_joints[:, -1, 0, [0, 2]] - steered_joints[:, 0, 0, [0, 2]]
        b_len = np.linalg.norm(b, axis=-1)
        s_len = np.linalg.norm(s, axis=-1)
        requested = max(args.root_scale - 1.0, 1e-8)
        achieved = (s_len / (b_len + 1e-8)) - 1.0
        return achieved / requested * 100.0

    return np.zeros((baseline_joints.shape[0],), dtype=np.float32)


def build_constraints(
    args,
    baseline_joints: np.ndarray,
    pose_target: Optional[torch.Tensor] = None,
) -> CompositeConstraint:
    constraint_list = []

    if "foot_contact" in args.constraint:
        fc = FootContactConstraint(
            height_thresh=0.05,
            vel_thresh=0.02,
            sigmoid_sharpness=20.0,
        )
        constraint_list.append((fc, 1.0))

    if "terminal" in args.constraint:
        target = torch.tensor([[args.terminal_x, 0.9, args.terminal_z]])
        tc = TerminalConstraint(target_joints=target, joint_indices=[0], tail_frames=4)
        constraint_list.append((tc, 1.5))

    if "pose" in args.constraint:
        assert pose_target is not None, "pose constraint requires a target pose"
        pc = PoseConstraint(
            keyframes=[(args.pose_t, pose_target)],
            joint_mask=_JOINT_MASK_MAP[args.pose_joint_mask],
            sigma_frac=args.pose_sigma_frac,
            use_hierarchical=args.use_hierarchical,
        )
        constraint_list.append((pc, 1.0))

    if args.edit_mode in ("raise_hand", "higher_kick"):
        joint_indices = _EDIT_JOINTS[args.edit_joint]
        offset = torch.tensor(
            [args.edit_delta_x, args.edit_delta_y, args.edit_delta_z],
            dtype=torch.float32,
        )
        c = TemporalJointOffsetConstraint(
            reference_joints=torch.from_numpy(baseline_joints).float(),
            joint_indices=joint_indices,
            offset_xyz=offset,
            t_start=args.edit_t_start,
            t_end=args.edit_t_end,
            edge_frac=args.edit_edge_frac,
        )
        constraint_list.append((c, args.edit_weight))

    elif args.edit_mode == "reach_hand":
        joint_indices = _EDIT_JOINTS[args.edit_joint]
        frame = int(round(args.edit_target_t * (baseline_joints.shape[1] - 1)))
        source = baseline_joints[:, frame, joint_indices, :].mean(axis=(0, 1))
        target = source + np.array(
            [args.edit_delta_x, args.edit_delta_y, args.edit_delta_z],
            dtype=np.float32,
        )
        c = JointReachConstraint(
            target_xyz=torch.from_numpy(target).float(),
            joint_indices=joint_indices,
            t_start=args.edit_t_start,
            t_end=args.edit_t_end,
            edge_frac=args.edit_edge_frac,
        )
        constraint_list.append((c, args.edit_weight))

    elif args.edit_mode == "faster_root":
        c = RootDisplacementScaleConstraint(
            reference_joints=torch.from_numpy(baseline_joints).float(),
            scale=args.root_scale,
            t_start=args.edit_t_start,
            t_end=args.edit_t_end,
            edge_frac=args.edit_edge_frac,
        )
        constraint_list.append((c, args.edit_weight))

    assert constraint_list, f"No constraints built from constraint={args.constraint}, edit_mode={args.edit_mode}"
    return CompositeConstraint(constraint_list, normalize_losses=args.normalize_losses)


def print_edit_metrics(args, baseline_joints: np.ndarray, steered_joints: np.ndarray):
    if args.edit_mode in ("raise_hand", "higher_kick"):
        joint_indices = _EDIT_JOINTS[args.edit_joint]
        direction = np.array([args.edit_delta_x, args.edit_delta_y, args.edit_delta_z], dtype=np.float32)
        achieved = _mean_joint_delta(
            baseline_joints,
            steered_joints,
            joint_indices,
            direction,
            args.edit_t_start,
            args.edit_t_end,
        )
        target = np.linalg.norm(direction)
        print("\n--- Temporal attribute edit ---")
        for i, value in enumerate(achieved):
            pct = value / (target + 1e-8) * 100.0
            print(f"  sample {i:2d} | target delta={target:.3f} m | achieved={value:.3f} m ({pct:.1f}%)")

    elif args.edit_mode == "reach_hand":
        joint_indices = _EDIT_JOINTS[args.edit_joint]
        frame = int(round(args.edit_target_t * (baseline_joints.shape[1] - 1)))
        source = baseline_joints[:, frame, joint_indices, :].mean(axis=(0, 1))
        target = source + np.array([args.edit_delta_x, args.edit_delta_y, args.edit_delta_z], dtype=np.float32)
        b_dist = np.linalg.norm(baseline_joints[:, frame][:, joint_indices, :] - target, axis=-1).mean(axis=1)
        s_dist = np.linalg.norm(steered_joints[:, frame][:, joint_indices, :] - target, axis=-1).mean(axis=1)
        print("\n--- Reach edit ---")
        for i, (b, s) in enumerate(zip(b_dist, s_dist)):
            imp = (b - s) / (b + 1e-8) * 100.0
            print(f"  sample {i:2d} | reach error {b:.3f} -> {s:.3f} m ({imp:+.1f}%)")

    elif args.edit_mode == "faster_root":
        b = baseline_joints[:, -1, 0, [0, 2]] - baseline_joints[:, 0, 0, [0, 2]]
        s = steered_joints[:, -1, 0, [0, 2]] - steered_joints[:, 0, 0, [0, 2]]
        b_len = np.linalg.norm(b, axis=-1)
        s_len = np.linalg.norm(s, axis=-1)
        print("\n--- Root displacement edit ---")
        for i, (bv, sv) in enumerate(zip(b_len, s_len)):
            print(f"  sample {i:2d} | root distance {bv:.3f} -> {sv:.3f} m (target scale {args.root_scale:.2f})")


def build_steerer(args, pipeline, constraints, scheduler):
    pose_like_edit = args.edit_mode in ("raise_hand", "higher_kick", "reach_hand")
    return FlowSteerer.from_pipeline(
        pipeline=pipeline,
        stats_dir=os.path.join(os.path.dirname(__file__), "stats"),
        body_model_path=os.path.join(
            os.path.dirname(__file__),
            "scripts/gradio/static/assets/dump_wooden",
        ),
        constraints=constraints,
        scheduler=scheduler,
        steps=args.steps,
        smooth_kernel=args.smooth_kernel,
        soft_norm_tau=args.soft_norm_tau,
        max_steer_ratio=args.max_steer_ratio,
        ema_momentum=args.ema_momentum,
        apply_latent_mask=args.apply_latent_mask or ("pose" in args.constraint) or pose_like_edit,
        latent_mask_transl=args.latent_mask_transl,
        latent_mask_root_rot=args.latent_mask_root_rot,
        use_temporal_mask=not args.no_temporal_mask,
        verbose=args.verbose,
    )


def main():
    parser = argparse.ArgumentParser(description="FlowSteer-Motion demo")
    parser.add_argument("--model_path", required=True, help="Path to HY-Motion-1.0 ckpt dir")
    parser.add_argument("--prompt", default="a person walks forward and stops")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated steer seed list")
    parser.add_argument("--cfg_scale", type=float, default=5.0)

    parser.add_argument(
        "--edit_mode",
        default="none",
        choices=["none", "raise_hand", "higher_kick", "reach_hand", "faster_root"],
        help="User-level temporal edit. Prefer this over raw pose keyframes for visible demos.",
    )
    parser.add_argument("--edit_joint", default=None, help="Joint group for edit_mode")
    parser.add_argument("--edit_delta_x", type=float, default=0.0)
    parser.add_argument("--edit_delta_y", type=float, default=0.35)
    parser.add_argument("--edit_delta_z", type=float, default=0.0)
    parser.add_argument("--edit_t_start", type=float, default=0.30)
    parser.add_argument("--edit_t_end", type=float, default=0.70)
    parser.add_argument("--edit_target_t", type=float, default=0.50)
    parser.add_argument("--edit_edge_frac", type=float, default=0.05)
    parser.add_argument("--edit_weight", type=float, default=1.0)
    parser.add_argument("--root_scale", type=float, default=1.3)
    parser.add_argument("--normalize_losses", action="store_true")
    parser.add_argument(
        "--auto_tune",
        action="store_true",
        help="Try a small alpha/trust-region grid and select the best edit under a jerk budget.",
    )
    parser.add_argument("--target_achievement_pct", type=float, default=60.0)
    parser.add_argument("--jerk_budget", type=float, default=2.0)
    parser.add_argument("--auto_alpha_values", default="20,40,80")
    parser.add_argument("--auto_max_steer_ratios", default="0.3,0.6,1.0")

    parser.add_argument(
        "--constraint",
        nargs="+",
        default=[],
        choices=["foot_contact", "terminal", "pose"],
        help="Legacy low-level constraints to apply in addition to edit_mode.",
    )
    parser.add_argument("--terminal_x", type=float, default=2.0)
    parser.add_argument("--terminal_z", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=None, help="Internal steering strength")
    parser.add_argument("--target_seed", type=int, default=42, help="Reference seed for pose targets")
    parser.add_argument("--pose_t", type=float, default=0.5, help="Normalized pose keyframe time")
    parser.add_argument(
        "--pose_joint_mask",
        default="upper_body",
        choices=["all", "upper_body", "arms", "lower_body", "legs"],
    )
    parser.add_argument("--pose_sigma_frac", type=float, default=0.04)
    parser.add_argument("--use_hierarchical", action="store_true")
    parser.add_argument("--apply_latent_mask", action="store_true")
    parser.add_argument("--latent_mask_transl", type=float, default=0.1)
    parser.add_argument("--latent_mask_root_rot", type=float, default=0.3)
    parser.add_argument("--no_temporal_mask", action="store_true")
    parser.add_argument("--steps", type=int, default=50, help="Euler steps")
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--soft_norm_tau", type=float, default=0.1)
    parser.add_argument("--max_steer_ratio", type=float, default=0.3)
    parser.add_argument("--ema_momentum", type=float, default=0.7)
    parser.add_argument("--scheduler", default="cosine", choices=["constant", "cosine", "staged"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", default="output/steer_demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.edit_joint is None:
        if args.edit_mode == "higher_kick":
            args.edit_joint = "right_foot"
        elif args.edit_mode in ("raise_hand", "reach_hand"):
            args.edit_joint = "right_arm"
        else:
            args.edit_joint = "right_wrist"
    if args.edit_joint not in _EDIT_JOINTS:
        raise ValueError(f"Unknown --edit_joint {args.edit_joint!r}; choices: {sorted(_EDIT_JOINTS)}")

    if args.alpha_max is None and args.edit_mode != "none":
        args.alpha_max = 20.0
        print("[Note] Using alpha_max=20.0 for temporal attribute editing. Pass --alpha_max to override.")
    if args.alpha_max is None and "pose" in args.constraint:
        args.alpha_max = 6.0
        print("[Note] Using alpha_max=6.0 for pose steering. Pass --alpha_max to override.")
    if args.alpha_max is None:
        args.alpha_max = 80.0

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    if "pose" in args.constraint and args.target_seed in seeds and len(seeds) > 1:
        seeds = [s for s in seeds if s != args.target_seed]
        print(f"[Note] Removed target_seed={args.target_seed} from steer seeds: {seeds}")
    random.seed(seeds[0] if seeds else 0)

    print("Loading HY-Motion pipeline...")
    pipeline = load_pipeline(args.model_path, device)

    pose_target_t = None
    pose_target_np = None
    if "pose" in args.constraint:
        print(f"Extracting pose target from seed {args.target_seed} at t={args.pose_t:.2f}...")
        with torch.no_grad():
            target_out = pipeline.generate(
                text=args.prompt,
                seed_input=[args.target_seed],
                duration_slider=args.duration,
                cfg_scale=args.cfg_scale,
            )
        target_joints = pipeline_output_to_world_joints(target_out)
        target_frame = int(round(args.pose_t * (target_joints.shape[1] - 1)))
        pose_target_np = canonicalize_frame_np(target_joints[0, target_frame])
        pose_target_t = torch.from_numpy(pose_target_np).float()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating baseline (no steering)...")
    with torch.no_grad():
        baseline_out = pipeline.generate(
            text=args.prompt,
            seed_input=seeds,
            duration_slider=args.duration,
            cfg_scale=args.cfg_scale,
        )
    baseline_joints = pipeline_output_to_world_joints(baseline_out)
    save_joints_npy(args.output_dir, "baseline", baseline_joints)

    print("Building steerer...")
    constraints = build_constraints(args, baseline_joints=baseline_joints, pose_target=pose_target_t)

    if args.edit_mode != "none":
        scheduler = StagedScheduler(alpha_max=args.alpha_max, mode="cosine", t_start=0.35, t_end=0.92)
    elif set(args.constraint) == {"pose"}:
        scheduler = StagedScheduler(alpha_max=args.alpha_max, mode="cosine", t_start=0.5, t_end=0.88)
    elif args.scheduler == "cosine":
        scheduler = StagedScheduler.cosine(alpha_max=args.alpha_max)
    elif args.scheduler == "constant":
        scheduler = StagedScheduler.constant(alpha_max=args.alpha_max)
    else:
        scheduler = StagedScheduler.make_staged(
            alpha_terminal=args.alpha_max,
            alpha_waypoint=args.alpha_max * 0.8,
            alpha_contact=args.alpha_max * 0.6,
        )

    if args.auto_tune and args.edit_mode != "none":
        alpha_values = [float(x.strip()) for x in args.auto_alpha_values.split(",") if x.strip()]
        ratio_values = [float(x.strip()) for x in args.auto_max_steer_ratios.split(",") if x.strip()]
        base_jerk = _mean_jerk(baseline_joints)
        candidates = []
        print("\nAuto-tuning steering parameters...")
        for alpha in alpha_values:
            for ratio in ratio_values:
                args.alpha_max = alpha
                args.max_steer_ratio = ratio
                trial_scheduler = StagedScheduler(alpha_max=alpha, mode="cosine", t_start=0.35, t_end=0.92)
                trial_steerer = build_steerer(args, pipeline, constraints, trial_scheduler)
                print(f"  trial alpha={alpha:.1f}, max_steer_ratio={ratio:.2f}")
                trial_out = trial_steerer.generate(
                    text=args.prompt,
                    seed_input=seeds,
                    duration_slider=args.duration,
                    cfg_scale=args.cfg_scale,
                )
                trial_joints = pipeline_output_to_world_joints(trial_out)
                achievement = _edit_achievement(args, baseline_joints, trial_joints)
                jerk_ratio = _mean_jerk(trial_joints) / (base_jerk + 1e-9)
                mean_achievement = float(achievement.mean())
                meets = mean_achievement >= args.target_achievement_pct and jerk_ratio <= args.jerk_budget
                within_quality = jerk_ratio <= args.jerk_budget
                if meets:
                    # Best case: hit the requested edit while staying under the quality budget.
                    tier = 0
                    score = abs(mean_achievement - args.target_achievement_pct) + 5.0 * max(0.0, jerk_ratio - 1.0)
                elif within_quality:
                    # Preserve quality first, then get as close as possible to the target edit.
                    tier = 1
                    score = abs(args.target_achievement_pct - mean_achievement)
                else:
                    # Last resort: no quality-feasible candidate exists.
                    tier = 2
                    score = (
                        max(0.0, args.target_achievement_pct - mean_achievement)
                        + 10.0 * max(0.0, jerk_ratio - args.jerk_budget)
                    )
                print(f"    achievement={mean_achievement:.1f}% jerk={jerk_ratio:.3f} score={score:.2f}")
                candidates.append((tier, score, meets, within_quality, mean_achievement, jerk_ratio, alpha, ratio, trial_joints))

        candidates.sort(key=lambda x: x[0])
        _, _, meets, within_quality, mean_achievement, jerk_ratio, best_alpha, best_ratio, steered_joints = candidates[0]
        args.alpha_max = best_alpha
        args.max_steer_ratio = best_ratio
        if meets:
            status = "met target and quality budget"
        elif within_quality:
            status = "quality-feasible closest target"
        else:
            status = "closest available; quality budget unmet"
        print(
            f"\nAuto-tune selected alpha={best_alpha:.1f}, max_steer_ratio={best_ratio:.2f} "
            f"({status}; achievement={mean_achievement:.1f}%, jerk={jerk_ratio:.3f})"
        )
    else:
        steerer = build_steerer(args, pipeline, constraints, scheduler)
        print(f"\nGenerating with steering (alpha={args.alpha_max}, steps={args.steps})...")
        steered_out = steerer.generate(
            text=args.prompt,
            seed_input=seeds,
            duration_slider=args.duration,
            cfg_scale=args.cfg_scale,
        )
        steered_joints = pipeline_output_to_world_joints(steered_out)

    save_joints_npy(args.output_dir, "steered", steered_joints)

    print_edit_metrics(args, baseline_joints, steered_joints)

    if "pose" in args.constraint:
        print("\n--- Pose keyframe comparison ---")
        joint_mask = _JOINT_MASK_MAP[args.pose_joint_mask]
        for idx, seed in enumerate(seeds):
            b_err = pose_hit_error(baseline_joints[idx], args.pose_t, pose_target_np, joint_mask)
            s_err = pose_hit_error(steered_joints[idx], args.pose_t, pose_target_np, joint_mask)
            imp = (b_err - s_err) / (b_err + 1e-8) * 100.0
            print(f"  seed {seed:4d} | pose error {b_err:.4f} -> {s_err:.4f} m ({imp:+.1f}%)")

    print("\n--- Foot sliding comparison ---")
    for tag, joints_np in [("baseline", baseline_joints), ("steered", steered_joints)]:
        joints_t = torch.from_numpy(joints_np)
        foot_idx = [7, 8, 10, 11]
        feet = joints_t[:, :, foot_idx, :]
        foot_vel = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        foot_y = feet[..., 1]
        floor_y = foot_y.min(dim=1, keepdim=True).values
        rel_h = foot_y - floor_y
        contact = (
            torch.sigmoid(-20 * (rel_h[:, :-1] - 0.05))
            * torch.sigmoid(-20 * (foot_vel - 0.02))
        )
        contact_vel = (contact * foot_vel).sum() / (contact.sum() + 1e-8)
        print(f"  {tag:10s} | mean contact velocity = {contact_vel.item():.5f}")

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
