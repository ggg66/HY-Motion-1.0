"""
FlowSteer-Motion demo.

Runs HY-Motion 1.0 with and without constraint steering, then saves results.

Examples:
    python demo_steer.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person walks forward and stops" \
        --duration 3.0 \
        --constraint foot_contact \
        --alpha_max 80 \
        --output_dir output/steer_demo

    python demo_steer.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person walks forward and stops" \
        --duration 3.0 \
        --constraint pose \
        --target_seed 42 \
        --seeds 43,44 \
        --alpha_max 6 \
        --use_hierarchical \
        --output_dir output/pose_steer_demo
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import yaml

from hymotion.utils.loaders import load_object
from steering import (
    ARM_JOINTS,
    CompositeConstraint,
    FlowSteerer,
    FootContactConstraint,
    PoseConstraint,
    StagedScheduler,
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


def build_constraints(args, pose_target: Optional[torch.Tensor] = None) -> CompositeConstraint:
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

    assert constraint_list, f"No constraints built from: {args.constraint}"
    return CompositeConstraint(constraint_list)


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


def main():
    parser = argparse.ArgumentParser(description="FlowSteer-Motion demo")
    parser.add_argument("--model_path", required=True, help="Path to HY-Motion-1.0 ckpt dir")
    parser.add_argument("--prompt", default="a person walks forward and stops")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated steer seed list")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument(
        "--constraint",
        nargs="+",
        default=["foot_contact"],
        choices=["foot_contact", "terminal", "pose"],
        help="Which constraints to apply",
    )
    parser.add_argument("--terminal_x", type=float, default=2.0)
    parser.add_argument("--terminal_z", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=80.0, help="Steering strength")
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
    parser.add_argument("--scheduler", default="cosine", choices=["constant", "cosine", "staged"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", default="output/steer_demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if "pose" in args.constraint and args.alpha_max == 80.0:
        args.alpha_max = 6.0
        print("[Note] Using alpha_max=6.0 for pose steering. Pass --alpha_max to override.")

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

    print("Building steerer...")
    constraints = build_constraints(args, pose_target=pose_target_t)

    if set(args.constraint) == {"pose"}:
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

    steerer = FlowSteerer.from_pipeline(
        pipeline=pipeline,
        stats_dir=os.path.join(os.path.dirname(__file__), "stats"),
        body_model_path=os.path.join(
            os.path.dirname(__file__),
            "scripts/gradio/static/assets/dump_wooden",
        ),
        constraints=constraints,
        scheduler=scheduler,
        steps=args.steps,
        apply_latent_mask=args.apply_latent_mask or ("pose" in args.constraint),
        latent_mask_transl=args.latent_mask_transl,
        latent_mask_root_rot=args.latent_mask_root_rot,
        use_temporal_mask=not args.no_temporal_mask,
        verbose=args.verbose,
    )

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

    print(f"\nGenerating with steering (alpha={args.alpha_max}, steps={args.steps})...")
    steered_out = steerer.generate(
        text=args.prompt,
        seed_input=seeds,
        duration_slider=args.duration,
        cfg_scale=args.cfg_scale,
    )
    steered_joints = pipeline_output_to_world_joints(steered_out)
    save_joints_npy(args.output_dir, "steered", steered_joints)

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
