"""Attribute-edit evaluation for FlowSteer-Motion.

This script evaluates segment-level user edits such as "raise the right arm by
0.35 m from 30%-70% of the motion".  It supports both:

  - steer: sampling-time FlowSteer updates during Euler integration
  - refine: post-sampling latent optimization on the generated baseline

The key metrics are edit achievement and quality cost, so results can be used
to build controllability curves and choose budget-feasible settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, Iterable, List

import numpy as np
import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from demo_steer import load_pipeline, pipeline_output_to_world_joints
from eval.metrics import compute_quality_metrics
from steering import (
    CompositeConstraint,
    FlowSteerer,
    LatentRefiner,
    MotionDecoder,
    StagedScheduler,
    TemporalJointOffsetConstraint,
)


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


def _parse_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_prompt_cases(args) -> List[Dict]:
    if args.prompts_file is None:
        return [{"prompt": args.prompt}]
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        cases = [item if isinstance(item, dict) else {"prompt": str(item)} for item in data]
    elif isinstance(data, dict) and "prompts" in data:
        cases = [item if isinstance(item, dict) else {"prompt": str(item)} for item in data["prompts"]]
    else:
        raise ValueError(f"Unsupported prompts file format: {args.prompts_file}")
    return [case for case in cases if str(case.get("prompt", "")).strip()]


def _window_indices(T: int, t_start: float, t_end: float) -> np.ndarray:
    lo = int(round(t_start * (T - 1)))
    hi = int(round(t_end * (T - 1))) + 1
    lo = max(0, min(lo, T - 1))
    hi = max(lo + 1, min(hi, T))
    return np.arange(lo, hi)


def _mean_delta_along(
    baseline: np.ndarray,
    edited: np.ndarray,
    joint_indices: List[int],
    direction: np.ndarray,
    t_start: float,
    t_end: float,
) -> float:
    idx = _window_indices(baseline.shape[0], t_start, t_end)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    delta = edited[idx][:, joint_indices, :] - baseline[idx][:, joint_indices, :]
    return float(np.tensordot(delta, direction, axes=([-1], [0])).mean())


def _mean_l2_delta(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=-1).mean())


def _preservation_metrics(
    baseline: np.ndarray,
    edited: np.ndarray,
    edited_joint_indices: List[int],
    t_start: float,
    t_end: float,
) -> Dict[str, float]:
    T, J, _ = baseline.shape
    in_idx = _window_indices(T, t_start, t_end)
    outside_mask = np.ones(T, dtype=bool)
    outside_mask[in_idx] = False
    nonedited = [j for j in range(J) if j not in set(edited_joint_indices)]
    root = [0]

    outside_drift = _mean_l2_delta(baseline[outside_mask], edited[outside_mask]) if outside_mask.any() else 0.0
    outside_edited_drift = (
        _mean_l2_delta(baseline[outside_mask][:, edited_joint_indices], edited[outside_mask][:, edited_joint_indices])
        if outside_mask.any()
        else 0.0
    )
    outside_nonedited_drift = (
        _mean_l2_delta(baseline[outside_mask][:, nonedited], edited[outside_mask][:, nonedited])
        if outside_mask.any() and nonedited
        else 0.0
    )
    in_nonedited_drift = _mean_l2_delta(baseline[in_idx][:, nonedited], edited[in_idx][:, nonedited]) if nonedited else 0.0
    root_drift = _mean_l2_delta(baseline[:, root], edited[:, root])

    return {
        "outside_window_drift_m": outside_drift,
        "outside_edited_joint_drift_m": outside_edited_drift,
        "outside_nonedited_joint_drift_m": outside_nonedited_drift,
        "inside_nonedited_joint_drift_m": in_nonedited_drift,
        "root_drift_m": root_drift,
    }


def _foot_sliding_proxy(joints_np: np.ndarray) -> float:
    feet = joints_np[:, [7, 8, 10, 11], :]
    foot_y = feet[:, :, 1]
    floor_y = foot_y.min(axis=0, keepdims=True)
    rel_h = foot_y - floor_y
    foot_vel = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
    contact = rel_h[:-1] < 0.05
    return float((contact * foot_vel).sum() / (contact.sum() + 1e-8))


def _denorm_latent_to_norm(pipeline, latent_denorm: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = pipeline.mean.to(device).view(1, 1, -1)
    std = pipeline.std.to(device).view(1, 1, -1)
    std_safe = std.clone()
    std_safe[std_safe < 1e-3] = 1.0
    return (latent_denorm.to(device) - mean) / std_safe


def _build_constraint(args, baseline_joints: np.ndarray, delta_y: float) -> CompositeConstraint:
    joint_indices = _EDIT_JOINTS[args.edit_joint]
    offset = torch.tensor([args.delta_x, delta_y, args.delta_z], dtype=torch.float32)
    return CompositeConstraint([
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


def _metrics_row(
    args,
    baseline_joints: np.ndarray,
    edited_joints: np.ndarray,
    seed: int,
    delta_y: float,
    method: str,
    params: Dict[str, float],
    elapsed_sec: float,
) -> Dict:
    joint_indices = _EDIT_JOINTS[args.edit_joint]
    requested = float(np.linalg.norm([args.delta_x, delta_y, args.delta_z]))
    achieved = _mean_delta_along(
        baseline_joints,
        edited_joints,
        joint_indices,
        np.array([args.delta_x, delta_y, args.delta_z], dtype=np.float32),
        args.t_start,
        args.t_end,
    )
    achievement_pct = achieved / (requested + 1e-8) * 100.0

    q_base = compute_quality_metrics(baseline_joints[None])
    q_edit = compute_quality_metrics(edited_joints[None])
    foot_base = _foot_sliding_proxy(baseline_joints)
    foot_edit = _foot_sliding_proxy(edited_joints)
    jerk_ratio = q_edit.mean_jerk / (q_base.mean_jerk + 1e-9)
    foot_ratio = foot_edit / (foot_base + 1e-9)
    preservation = _preservation_metrics(
        baseline_joints,
        edited_joints,
        joint_indices,
        args.t_start,
        args.t_end,
    )

    row = {
        "prompt": args.prompt,
        "seed": seed,
        "method": method,
        "edit_joint": args.edit_joint,
        "delta_y": delta_y,
        "requested_delta_m": requested,
        "achieved_delta_m": achieved,
        "achievement_pct": achievement_pct,
        "jerk_ratio": jerk_ratio,
        "foot_sliding_base": foot_base,
        "foot_sliding_edited": foot_edit,
        "foot_sliding_ratio": foot_ratio,
        **preservation,
        "meets_target": achievement_pct >= args.target_achievement_pct,
        "meets_jerk_budget": jerk_ratio <= args.jerk_budget,
        "meets_budget": achievement_pct >= args.target_achievement_pct and jerk_ratio <= args.jerk_budget,
        "elapsed_sec": elapsed_sec,
    }
    row.update(params)
    return row


def _run_steer_one(
    pipeline,
    decoder,
    baseline_joints: np.ndarray,
    seed: int,
    args,
    delta_y: float,
    alpha: float,
    max_steer_ratio: float,
) -> np.ndarray:
    constraint = _build_constraint(args, baseline_joints, delta_y)
    steerer = FlowSteerer(
        pipeline=pipeline,
        decoder=decoder,
        constraints=constraint,
        scheduler=StagedScheduler(alpha_max=alpha, mode="cosine", t_start=0.35, t_end=0.92),
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
        text=args.prompt,
        seed_input=[seed],
        duration_slider=args.duration,
        cfg_scale=args.cfg_scale,
    )
    return pipeline_output_to_world_joints(out)[0]


def _run_refine_one(
    pipeline,
    decoder,
    baseline_out: dict,
    baseline_joints: np.ndarray,
    args,
    delta_y: float,
    steps: int,
    lr: float,
    constraint_weight: float,
    joint_proximity: float,
    smoothness: float,
    delta_smoothness: float,
) -> np.ndarray:
    device = next(pipeline.parameters()).device
    latent0 = _denorm_latent_to_norm(pipeline, baseline_out["latent_denorm"], device)
    decoder.to(device)
    with torch.no_grad():
        reference_joints = decoder(latent0).detach().cpu().numpy()[0]
    constraint = _build_constraint(args, reference_joints, delta_y)
    refiner = LatentRefiner(
        decoder=decoder,
        constraints=constraint,
        steps=steps,
        lr=lr,
        constraint_weight=constraint_weight,
        latent_proximity_weight=args.refine_latent_proximity,
        delta_smoothness_weight=delta_smoothness,
        joint_proximity_weight=joint_proximity,
        smoothness_weight=smoothness,
        max_delta=args.refine_max_delta,
        apply_latent_mask=not args.no_latent_mask,
        latent_mask_transl=args.latent_mask_transl,
        latent_mask_root_rot=args.latent_mask_root_rot,
        use_temporal_mask=not args.no_temporal_mask,
        log_every=max(steps, 1),
        verbose=False,
    )
    result = refiner.refine(latent0)
    out = pipeline.decode_motion_from_latent(result.latent, should_apply_smooothing=True)
    return pipeline_output_to_world_joints(out)[0]


def _select_budget_rows(rows: Iterable[Dict], args) -> List[Dict]:
    selected = []
    grouped: Dict[tuple, List[Dict]] = {}
    for row in rows:
        key = (row["prompt"], row["seed"], row["delta_y"], row["method"])
        grouped.setdefault(key, []).append(row)

    for key, group in grouped.items():
        feasible = [r for r in group if r["meets_budget"]]
        if feasible:
            best = min(
                feasible,
                key=lambda r: (
                    abs(r["achievement_pct"] - args.target_achievement_pct),
                    r["jerk_ratio"],
                    r["elapsed_sec"],
                ),
            )
            status = "target_and_quality"
        else:
            quality_ok = [r for r in group if r["meets_jerk_budget"]]
            if quality_ok:
                best = min(quality_ok, key=lambda r: abs(r["achievement_pct"] - args.target_achievement_pct))
                status = "quality_feasible_closest"
            else:
                best = min(
                    group,
                    key=lambda r: (
                        max(0.0, args.target_achievement_pct - r["achievement_pct"])
                        + 10.0 * max(0.0, r["jerk_ratio"] - args.jerk_budget)
                    ),
                )
                status = "closest_available"
        out = dict(best)
        out["selection_status"] = status
        selected.append(out)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Temporal attribute edit evaluation")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", default="a person walks forward.")
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seeds", default="43")
    parser.add_argument("--output_dir", default="output/attribute_sweep")
    parser.add_argument("--method", default="both", choices=["steer", "refine", "target_only", "both"])
    parser.add_argument("--edit_joint", default="right_arm", choices=sorted(_EDIT_JOINTS))
    parser.add_argument("--delta_x", type=float, default=0.0)
    parser.add_argument("--delta_y_values", default="0.20,0.35")
    parser.add_argument("--delta_z", type=float, default=0.0)
    parser.add_argument("--t_start", type=float, default=0.30)
    parser.add_argument("--t_end", type=float, default=0.70)
    parser.add_argument("--edge_frac", type=float, default=0.05)
    parser.add_argument("--target_achievement_pct", type=float, default=75.0)
    parser.add_argument("--jerk_budget", type=float, default=2.0)

    parser.add_argument("--alpha_values", default="20,80")
    parser.add_argument("--max_steer_ratios", default="0.3,1.0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smooth_kernel", type=int, default=7)
    parser.add_argument("--soft_norm_tau", type=float, default=0.1)
    parser.add_argument("--ema_momentum", type=float, default=0.7)

    parser.add_argument("--refine_steps_values", default="40,100")
    parser.add_argument("--refine_lr_values", default="0.05")
    parser.add_argument("--refine_constraint_weights", default="10,20,40")
    parser.add_argument("--refine_delta_smoothness_values", default="0.0,0.01")
    parser.add_argument("--refine_joint_proximity_values", default="0.01,0.02")
    parser.add_argument("--refine_smoothness_values", default="0.002,0.01")
    parser.add_argument("--refine_latent_proximity", type=float, default=1e-3)
    parser.add_argument("--refine_max_delta", type=float, default=3.0)

    parser.add_argument("--latent_mask_transl", type=float, default=0.1)
    parser.add_argument("--latent_mask_root_rot", type=float, default=0.3)
    parser.add_argument("--no_latent_mask", action="store_true")
    parser.add_argument("--no_temporal_mask", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_best_npy", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    prompt_cases = _load_prompt_cases(args)
    default_edit_joint = args.edit_joint
    default_delta_x = args.delta_x
    default_delta_y_values = args.delta_y_values
    default_delta_z = args.delta_z
    default_t_start = args.t_start
    default_t_end = args.t_end
    default_edge_frac = args.edge_frac
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print("Loading HY-Motion pipeline...")
    pipeline = load_pipeline(args.model_path, device)
    decoder = MotionDecoder.from_stats_dir(
        stats_dir=os.path.join(_repo_root, "stats"),
        body_model_path=os.path.join(_repo_root, "scripts/gradio/static/assets/dump_wooden"),
    )

    run_steer = args.method in ("steer", "both")
    run_refine = args.method in ("refine", "target_only", "both")
    run_target_only = args.method == "target_only"
    rows = []

    for prompt_idx, case in enumerate(prompt_cases):
        args.prompt = str(case["prompt"])
        args.edit_joint = str(case.get("edit_joint", default_edit_joint))
        if args.edit_joint not in _EDIT_JOINTS:
            raise ValueError(f"Unknown edit_joint in case {prompt_idx}: {args.edit_joint}")
        args.delta_x = float(case.get("delta_x", default_delta_x))
        args.delta_z = float(case.get("delta_z", default_delta_z))
        args.t_start = float(case.get("t_start", default_t_start))
        args.t_end = float(case.get("t_end", default_t_end))
        args.edge_frac = float(case.get("edge_frac", default_edge_frac))
        delta_y_values = _parse_floats(str(case.get("delta_y_values", default_delta_y_values)))
        alpha_values = _parse_floats(str(case.get("alpha_values", args.alpha_values)))
        max_steer_ratios = _parse_floats(str(case.get("max_steer_ratios", args.max_steer_ratios)))
        refine_steps_values = _parse_ints(str(case.get("refine_steps_values", args.refine_steps_values)))
        refine_lr_values = _parse_floats(str(case.get("refine_lr_values", args.refine_lr_values)))
        refine_constraint_weights = _parse_floats(
            str(case.get("refine_constraint_weights", args.refine_constraint_weights))
        )
        refine_delta_smoothness_values = _parse_floats(
            str(case.get("refine_delta_smoothness_values", args.refine_delta_smoothness_values))
        )
        refine_joint_proximity_values = _parse_floats(
            str(case.get("refine_joint_proximity_values", args.refine_joint_proximity_values))
        )
        refine_smoothness_values = _parse_floats(
            str(case.get("refine_smoothness_values", args.refine_smoothness_values))
        )

        case_id = str(case.get("id", f"case_{prompt_idx:02d}"))
        print(f"\nPrompt {prompt_idx} [{case_id}]: {args.prompt}")
        print(
            f"  edit_joint={args.edit_joint}, dy={delta_y_values}, "
            f"window=({args.t_start:.2f}, {args.t_end:.2f})"
        )
        if run_refine:
            print(
                f"  refine grid: steps={refine_steps_values}, lr={refine_lr_values}, "
                f"cw={refine_constraint_weights}, ds={refine_delta_smoothness_values}, "
                f"jp={refine_joint_proximity_values}, sm={refine_smoothness_values}"
            )
        for seed in seeds:
            print(f"\nBaseline seed={seed}")
            with torch.no_grad():
                baseline_out = pipeline.generate(
                    text=args.prompt,
                    seed_input=[seed],
                    duration_slider=args.duration,
                    cfg_scale=args.cfg_scale,
                )
            baseline = pipeline_output_to_world_joints(baseline_out)[0]
            np.save(os.path.join(args.output_dir, f"baseline_p{prompt_idx:02d}_seed{seed}.npy"), baseline)

            for delta_y in delta_y_values:
                if run_steer:
                    for alpha in alpha_values:
                        for ratio in max_steer_ratios:
                            t0 = time.time()
                            edited = _run_steer_one(
                                pipeline, decoder, baseline, seed, args, delta_y, alpha, ratio
                            )
                            row = _metrics_row(
                                args,
                                baseline,
                                edited,
                                seed,
                                delta_y,
                                "steer",
                                {
                                    "prompt_idx": prompt_idx,
                                    "case_id": case_id,
                                    "alpha": alpha,
                                    "max_steer_ratio": ratio,
                                    "refine_steps": 0,
                                    "refine_lr": 0.0,
                                    "refine_constraint_weight": 0.0,
                                    "refine_delta_smoothness": 0.0,
                                    "refine_joint_proximity": 0.0,
                                    "refine_smoothness": 0.0,
                                },
                                time.time() - t0,
                            )
                            rows.append(row)
                            print(
                                f"  steer  dy={delta_y:.2f} alpha={alpha:.1f} ratio={ratio:.2f} | "
                                f"achieved={row['achieved_delta_m']:.3f}m "
                                f"({row['achievement_pct']:.1f}%) | jerk={row['jerk_ratio']:.3f}"
                            )

                if run_refine:
                    for r_steps in refine_steps_values:
                        for lr in refine_lr_values:
                            for c_weight in refine_constraint_weights:
                                for d_smooth in refine_delta_smoothness_values:
                                    for j_prox in refine_joint_proximity_values:
                                        for smooth in refine_smoothness_values:
                                            if run_target_only:
                                                args.no_latent_mask = True
                                                args.no_temporal_mask = True
                                                args.refine_latent_proximity = 0.0
                                                d_smooth = 0.0
                                                j_prox = 0.0
                                                smooth = 0.0
                                            t0 = time.time()
                                            edited = _run_refine_one(
                                                pipeline,
                                                decoder,
                                                baseline_out,
                                                baseline,
                                                args,
                                                delta_y,
                                                r_steps,
                                                lr,
                                                c_weight,
                                                j_prox,
                                                smooth,
                                                d_smooth,
                                            )
                                            row = _metrics_row(
                                                args,
                                                baseline,
                                                edited,
                                                seed,
                                                delta_y,
                                                "target_only" if run_target_only else "refine",
                                                {
                                                    "prompt_idx": prompt_idx,
                                                    "case_id": case_id,
                                                    "alpha": 0.0,
                                                    "max_steer_ratio": 0.0,
                                                    "refine_steps": r_steps,
                                                    "refine_lr": lr,
                                                    "refine_constraint_weight": c_weight,
                                                    "refine_delta_smoothness": d_smooth,
                                                    "refine_joint_proximity": j_prox,
                                                    "refine_smoothness": smooth,
                                                },
                                                time.time() - t0,
                                            )
                                            rows.append(row)
                                            print(
                                                f"  refine dy={delta_y:.2f} steps={r_steps} cw={c_weight:.1f} "
                                                f"ds={d_smooth:g} jp={j_prox:g} sm={smooth:g} | "
                                                f"achieved={row['achieved_delta_m']:.3f}m "
                                                f"({row['achievement_pct']:.1f}%) | jerk={row['jerk_ratio']:.3f}"
                                            )

    selected = _select_budget_rows(rows, args)
    if args.save_best_npy:
        # Keep the flag for future batch generation; best videos are better made
        # with eval/visualize_attribute_edit.py from explicitly chosen rows.
        print("[Note] --save_best_npy is reserved; rerun selected configs for paper videos.")

    json_path = os.path.join(args.output_dir, "results.json")
    csv_path = os.path.join(args.output_dir, "results.csv")
    selected_path = os.path.join(args.output_dir, "selected.csv")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(selected_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {selected_path}")


if __name__ == "__main__":
    main()
