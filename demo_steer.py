"""
FlowSteer-Motion demo.

Runs HY-Motion 1.0 with and without constraint steering, then saves results.

Usage:
    python demo_steer.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person walks forward and stops" \
        --duration 3.0 \
        --constraint foot_contact \
        --alpha_max 80 \
        --steps 50 \
        --output_dir output/steer_demo

    # With terminal position constraint:
    python demo_steer.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person walks to a destination" \
        --duration 3.0 \
        --constraint terminal \
        --terminal_x 2.0 --terminal_z 1.5 \
        --alpha_max 120 \
        --output_dir output/steer_demo
"""

import argparse
import os
import random

import numpy as np
import torch
import yaml

from hymotion.utils.loaders import load_object
from hymotion.utils.t2m_runtime import T2MRuntime
from steering import (
    CompositeConstraint,
    FlowSteerer,
    FootContactConstraint,
    MotionDecoder,
    StagedScheduler,
    TerminalConstraint,
    TrajectoryConstraint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline(model_path: str, device: torch.device):
    """Load MotionFlowMatching from a HY-Motion checkpoint directory."""
    cfg_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")
    assert os.path.exists(cfg_path), f"config.yml not found at {cfg_path}"
    assert os.path.exists(ckpt_path), f"latest.ckpt not found at {ckpt_path}"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Config layout matches T2MRuntime.load():
    #   train_pipeline / train_pipeline_args / network_module / network_module_args
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


def build_constraints(args) -> CompositeConstraint:
    constraint_list = []

    if "foot_contact" in args.constraint:
        fc = FootContactConstraint(
            height_thresh=0.05,
            vel_thresh=0.02,
            sigmoid_sharpness=20.0,
        )
        constraint_list.append((fc, 1.0))

    if "terminal" in args.constraint:
        target = torch.tensor([
            [args.terminal_x, 0.9, args.terminal_z],   # root (pelvis height ≈ 0.9m)
        ])
        tc = TerminalConstraint(
            target_joints=target,
            joint_indices=[0],   # Pelvis
            tail_frames=4,
        )
        constraint_list.append((tc, 1.5))

    assert constraint_list, f"No constraints built from: {args.constraint}"
    return CompositeConstraint(constraint_list)


def save_joints_npy(output_dir: str, tag: str, keypoints3d: np.ndarray):
    """Save (B, T, J, 3) keypoints as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    B = keypoints3d.shape[0]
    for b in range(B):
        path = os.path.join(output_dir, f"{tag}_seed{b}.npy")
        np.save(path, keypoints3d[b])
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FlowSteer-Motion demo")
    parser.add_argument("--model_path", required=True, help="Path to HY-Motion-1.0 ckpt dir")
    parser.add_argument("--prompt", default="a person walks forward and stops")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration in seconds")
    parser.add_argument("--seeds", default="42,43", help="Comma-separated seed list")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument(
        "--constraint",
        nargs="+",
        default=["foot_contact"],
        choices=["foot_contact", "terminal"],
        help="Which constraints to apply",
    )
    parser.add_argument("--terminal_x", type=float, default=2.0)
    parser.add_argument("--terminal_z", type=float, default=0.0)
    parser.add_argument("--alpha_max", type=float, default=80.0, help="Steering strength")
    parser.add_argument("--steps", type=int, default=50, help="Euler steps")
    parser.add_argument("--scheduler", default="cosine", choices=["constant", "cosine", "staged"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", default="output/steer_demo")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]

    # --- Load pipeline ---
    print("Loading HY-Motion pipeline...")
    pipeline = load_pipeline(args.model_path, device)

    # --- Build steerer components ---
    print("Building steerer...")
    constraints = build_constraints(args)

    if args.scheduler == "cosine":
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
            os.path.dirname(__file__), "scripts/gradio/static/assets/dump_wooden"
        ),
        constraints=constraints,
        scheduler=scheduler,
        steps=args.steps,
        verbose=args.verbose,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Baseline: no steering ---
    print(f"\nGenerating baseline (no steering)...")
    with torch.no_grad():
        baseline_out = pipeline.generate(
            text=args.prompt,
            seed_input=seeds,
            duration_slider=args.duration,
            cfg_scale=args.cfg_scale,
        )
    # World-space = local_joints + transl (WoodenMesh keypoints3d is local FK only)
    k3d_b = baseline_out["keypoints3d"].numpy()        # (B, T, 52, 3) local
    transl_b = baseline_out["transl"].numpy()           # (B, T, 3)
    baseline_joints = k3d_b[:, :, :22, :] + transl_b[:, :, np.newaxis, :]
    save_joints_npy(args.output_dir, "baseline", baseline_joints)

    # --- Steered generation ---
    print(f"\nGenerating with steering (α={args.alpha_max}, steps={args.steps})...")
    steered_out = steerer.generate(
        text=args.prompt,
        seed_input=seeds,
        duration_slider=args.duration,
        cfg_scale=args.cfg_scale,
    )
    k3d_s = steered_out["keypoints3d"].numpy()
    transl_s = steered_out["transl"].numpy()
    steered_joints = k3d_s[:, :, :22, :] + transl_s[:, :, np.newaxis, :]
    save_joints_npy(args.output_dir, "steered", steered_joints)

    # --- Quick quantitative comparison ---
    print("\n--- Foot sliding comparison ---")
    for tag, joints_np in [("baseline", baseline_joints), ("steered", steered_joints)]:
        joints_t = torch.from_numpy(joints_np)   # (B, T, J, 3)
        foot_idx = [7, 8, 10, 11]
        feet = joints_t[:, :, foot_idx, :]   # (B, T, 4, 3)
        foot_vel = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)   # (B, T-1, 4)

        # Soft contact mask (same as constraint)
        foot_y = feet[..., 1]
        floor_y = foot_y.min(dim=1, keepdim=True).values
        rel_h = foot_y - floor_y
        contact = (torch.sigmoid(-20 * (rel_h[:, :-1] - 0.05)) *
                   torch.sigmoid(-20 * (foot_vel - 0.02)))
        contact_vel = (contact * foot_vel).sum() / (contact.sum() + 1e-8)
        print(f"  {tag:10s} | mean contact velocity = {contact_vel.item():.5f}")

    print(f"\nDone. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
