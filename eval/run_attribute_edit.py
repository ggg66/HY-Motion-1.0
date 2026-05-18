"""
Attribute-edit sweep for FlowSteer-Motion.

This script tests whether user-level temporal edits are controllable.  It runs
one baseline motion per seed, then applies a TemporalJointOffsetConstraint under
multiple steering settings and reports:

  - achieved_delta_m: mean displacement along the requested edit direction
  - achievement_pct: achieved_delta_m / requested_delta_m
  - jerk_ratio: motion smoothness cost relative to baseline
  - foot_sliding_ratio: contact-foot velocity cost relative to baseline

Example:
    python eval/run_attribute_edit.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt "a person walks forward." \
        --duration 4.0 \
        --seeds 43 \
        --edit_joint right_arm \
        --delta_y_values 0.20,0.35 \
        --alpha_values 20,40,80 \
        --max_steer_ratios 0.3,0.6,1.0 \
        --output_dir output/attribute_sweep_raise_hand
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from demo_steer import load_pipeline, pipeline_output_to_world_joints
from eval.metrics import compute_quality_metrics
from steering import CompositeConstraint, FlowSteerer, MotionDecoder, TemporalJointOffsetConstraint


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


def _parse_floats(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _window_indices(T: int, t_start: float, t_end: float) -> np.ndarray:
    lo = int(round(t_start * (T - 1)))
    hi = int(round(t_end * (T - 1))) + 1
    lo = max(0, min(lo, T - 1))
    hi = max(lo + 1, min(hi, T))
    return np.arange(lo, hi)


def _mean_delta_along(
    baseline: np.ndarray,
    steered: np.ndarray,
    joint_indices: List[int],
    direction: np.ndarray,
    t_start: float,
    t_end: float,
) -> float:
    idx = _window_indices(baseline.shape[0], t_start, t_end)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    delta = steered[idx][:, joint_indices, :] - baseline[idx][:, joint_indices, :]
    return float(np.tensordot(delta, direction, axes=([-1], [0])).mean())


def _foot_sliding_proxy(joints_np: np.ndarray) -> float:
    feet = joints_np[:, [7, 8, 10, 11], :]
    foot_y = feet[:, :, 1]
    floor_y = foot_y.min(axis=0, keepdims=True)
    rel_h = foot_y - floor_y
    foot_vel = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    contact = rel_h[:-1] < 0.05
    return float((contact * foot_vel).sum() / (contact.sum() + 1e-8))


def run_one(
    pipeline,
    decoder,
    baseline_joints: np.ndarray,
    prompt: str,
    duration: float,
    seed: int,
    args,
    delta_y: float,
    alpha: float,
    max_steer_ratio: float,
) -> Dict:
    joint_indices = _EDIT_JOINTS[args.edit_joint]
    offset = torch.tensor([args.delta_x, delta_y, args.delta_z], dtype=torch.float32)
    constraint = CompositeConstraint([
        (
            TemporalJointOffsetConstraint(
                reference_joints=torch.from_numpy(baseline_joints).float(),
                joint_indices=joint_indices,
                offset_xyz=offset,
                t_start=args.t_start,
                t_end=args.t_end,
                edge_frac=args.edge_frac,
            ),
            1.0,
        )
    ])

    steerer = FlowSteerer(
        pipeline=pipeline,
        decoder=decoder,
        constraints=constraint,
        scheduler=args.scheduler_factory(alpha),
        steps=args.steps,
        smooth_kernel=args.smooth_kernel,
        soft_norm_tau=args.soft_norm_tau,
        max_steer_ratio=max_steer_ratio,
        ema_momentum=args.ema_momentum,
        apply_latent_mask=not args.no_latent_mask,
        latent_mask_transl=args.latent_mask_transl,
        latent_mask_root_rot=args.latent_mask_root_rot,
        use_temporal_mask=not args.no_temporal_mask,
    )

    out = steerer.generate(
        text=prompt,
        seed_input=[seed],
        duration_slider=duration,
        cfg_scale=args.cfg_scale,
    )
    steered = pipeline_output_to_world_joints(out)[0]

    requested = float(np.linalg.norm([args.delta_x, delta_y, args.delta_z]))
    achieved = _mean_delta_along(
        baseline_joints,
        steered,
        joint_indices,
        np.array([args.delta_x, delta_y, args.delta_z], dtype=np.float32),
        args.t_start,
        args.t_end,
    )

    q_base = compute_quality_metrics(baseline_joints[None])
    q_steer = compute_quality_metrics(steered[None])
    foot_base = _foot_sliding_proxy(baseline_joints)
    foot_steer = _foot_sliding_proxy(steered)

    return {
        "prompt": prompt,
        "seed": seed,
        "edit_joint": args.edit_joint,
        "delta_y": delta_y,
        "requested_delta_m": requested,
        "alpha": alpha,
        "max_steer_ratio": max_steer_ratio,
        "achieved_delta_m": achieved,
        "achievement_pct": achieved / (requested + 1e-8) * 100.0,
        "jerk_ratio": q_steer.mean_jerk / (q_base.mean_jerk + 1e-9),
        "foot_sliding_base": foot_base,
        "foot_sliding_steered": foot_steer,
        "foot_sliding_ratio": foot_steer / (foot_base + 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal attribute edit sweep")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default="a person walks forward.")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seeds", default="43")
    parser.add_argument("--output_dir", default="output/attribute_sweep")
    parser.add_argument("--edit_joint", default="right_arm", choices=sorted(_EDIT_JOINTS))
    parser.add_argument("--delta_x", type=float, default=0.0)
    parser.add_argument("--delta_y_values", default="0.20,0.35")
    parser.add_argument("--delta_z", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.30)
    parser.add_argument("--t_end", type=float, default=0.70)
    parser.add_argument("--edge_frac", type=float, default=0.05)
    parser.add_argument("--alpha_values", default="20,40,80")
    parser.add_argument("--max_steer_ratios", default="0.3,0.6,1.0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--soft_norm_tau", type=float, default=0.1)
    parser.add_argument("--ema_momentum", type=float, default=0.7)
    parser.add_argument("--latent_mask_transl", type=float, default=0.1)
    parser.add_argument("--latent_mask_root_rot", type=float, default=0.3)
    parser.add_argument("--no_latent_mask", action="store_true")
    parser.add_argument("--no_temporal_mask", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    from steering import StagedScheduler

    args.scheduler_factory = lambda alpha: StagedScheduler(
        alpha_max=alpha,
        mode="cosine",
        t_start=0.35,
        t_end=0.92,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    delta_y_values = _parse_floats(args.delta_y_values)
    alpha_values = _parse_floats(args.alpha_values)
    max_steer_ratios = _parse_floats(args.max_steer_ratios)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Loading HY-Motion pipeline...")
    pipeline = load_pipeline(args.model_path, device)
    decoder = MotionDecoder.from_stats_dir(
        stats_dir=os.path.join(_repo_root, "stats"),
        body_model_path=os.path.join(_repo_root, "scripts/gradio/static/assets/dump_wooden"),
    )

    rows = []
    for seed in seeds:
        print(f"\nBaseline seed={seed}")
        with torch.no_grad():
            base_out = pipeline.generate(
                text=args.prompt,
                seed_input=[seed],
                duration_slider=args.duration,
                cfg_scale=args.cfg_scale,
            )
        baseline = pipeline_output_to_world_joints(base_out)[0]
        np.save(os.path.join(args.output_dir, f"baseline_seed{seed}.npy"), baseline)

        for delta_y in delta_y_values:
            for alpha in alpha_values:
                for ratio in max_steer_ratios:
                    t0 = time.time()
                    row = run_one(
                        pipeline=pipeline,
                        decoder=decoder,
                        baseline_joints=baseline,
                        prompt=args.prompt,
                        duration=args.duration,
                        seed=seed,
                        args=args,
                        delta_y=delta_y,
                        alpha=alpha,
                        max_steer_ratio=ratio,
                    )
                    rows.append(row)
                    print(
                        f"  dy={delta_y:.2f} alpha={alpha:.1f} ratio={ratio:.2f} | "
                        f"achieved={row['achieved_delta_m']:.3f}m "
                        f"({row['achievement_pct']:.1f}%) | "
                        f"jerk={row['jerk_ratio']:.3f} foot={row['foot_sliding_ratio']:.3f} "
                        f"[{time.time() - t0:.1f}s]"
                    )

    json_path = os.path.join(args.output_dir, "results.json")
    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
