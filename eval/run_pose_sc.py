"""
Pose self-consistency evaluation for PoseConstraint.

Protocol (per prompt × per seed):
    1. Run baseline(seed=s)  — no steering
    2. Canonicalize pose at t_norm  →  in-memory target  (no external .npy needed)
    3. Run steered(seed=s, PoseConstraint(target))
    4. Measure pose_hit_error, jerk / kinvar ratios

Each seed's target comes from that seed's own baseline, so the protocol
is pure self-recovery.  Results are reported per seed and then aggregated.
A low/high variance split (from the "variance" field in the prompt JSON)
is also reported separately.

Cross-seed alignment (42→43/44) is intentionally excluded from this script;
it is a harder, different task that should be reported under a separate table.

Usage
-----
    # Single alpha, 3 seeds:
    python eval/run_pose_sc.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt_file eval/prompts/pose_eval_raw.json \
        --seeds 42,43,44 \
        --alpha_pose 8.0 \
        --output_dir output/eval_pose_sc

    # Alpha sweep (seed=42 only, fastest):
    python eval/run_pose_sc.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt_file eval/prompts/pose_eval_raw.json \
        --seeds 42 \
        --alpha_sweep 1,2,4,6,8,10,12,15 \
        --output_dir output/eval_pose_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hymotion.utils.loaders import load_object
from steering import (
    ARM_JOINTS,
    CompositeConstraint,
    FlowSteerer,
    MotionDecoder,
    PoseConstraint,
    StagedScheduler,
    UPPER_BODY_JOINTS,
)
from eval.metrics import (
    canonicalize_frame_np,
    compute_quality_metrics,
    pipeline_output_to_world_joints,
)


# ---------------------------------------------------------------------------
# Joint mask map
# ---------------------------------------------------------------------------

_JOINT_MASK_MAP: Dict[str, Optional[List[int]]] = {
    "all":        None,
    "upper_body": UPPER_BODY_JOINTS,
    "arms":       ARM_JOINTS,
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "legs":       [1, 2, 4, 5, 7, 8, 10, 11],
}


# ---------------------------------------------------------------------------
# Pipeline loader (shared with run_eval.py)
# ---------------------------------------------------------------------------

def load_pipeline(model_path: str, device: torch.device):
    cfg_path  = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")
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


# ---------------------------------------------------------------------------
# Per-frame pose helpers
# ---------------------------------------------------------------------------

def _extract_target(joints_np: np.ndarray, t_norm: float) -> np.ndarray:
    """
    Canonicalize the pose at t_norm from a single-sample motion array.

    Args:
        joints_np: (T, 22, 3) world-space joints (one sample)
        t_norm: normalised time in [0, 1]

    Returns:
        (22, 3) canonical pose
    """
    T = joints_np.shape[0]
    frame = int(round(t_norm * (T - 1)))
    frame = max(0, min(frame, T - 1))
    return canonicalize_frame_np(joints_np[frame])   # (22, 3)


def _pose_hit_error(
    joints_np: np.ndarray,
    t_norm: float,
    target: np.ndarray,
    joint_mask: Optional[List[int]],
) -> float:
    """
    Compute mean L2 error in canonical pose space at t_norm.

    Args:
        joints_np: (T, 22, 3) world-space joints
        t_norm:    normalised target time
        target:    (22, 3) canonical target pose
        joint_mask: subset of joints to evaluate (None = all)

    Returns:
        scalar error (m, lower is better)
    """
    T = joints_np.shape[0]
    frame = int(round(t_norm * (T - 1)))
    frame = max(0, min(frame, T - 1))
    canonical = canonicalize_frame_np(joints_np[frame])    # (22, 3)
    if joint_mask is not None:
        canonical = canonical[joint_mask]
        target    = target[joint_mask]
    return float(np.linalg.norm(canonical - target, axis=-1).mean())


# ---------------------------------------------------------------------------
# Per-prompt × per-seed self-consistency run
# ---------------------------------------------------------------------------

def run_single(
    pipeline,
    decoder: MotionDecoder,
    prompt_cfg: Dict,
    seed: int,
    alpha_pose: float,
    args,
) -> Dict:
    """
    Run one self-consistency trial: baseline → extract target → steer → metrics.

    Returns a dict with keys:
        seed, prompt, duration, variance,
        pose_hit_baseline, pose_hit_steered, pose_hit_improvement_pct,
        jerk_baseline, jerk_steered, jerk_ratio,
        kinvar_baseline, kinvar_steered, kinvar_ratio,
        t_norm, joint_mask, alpha_pose
    """
    prompt   = prompt_cfg["prompt"]
    duration = prompt_cfg.get("duration", 3.0)
    variance = prompt_cfg.get("variance", "unknown")

    # ---- Extract keyframe specs ----
    kfs = prompt_cfg.get("pose_keyframes_raw", [])
    assert kfs, f"No pose_keyframes_raw in prompt: {prompt}"
    t_norm     = float(kfs[0]["t_norm"])
    mask_key   = kfs[0].get("joint_mask", "upper_body")
    joint_mask = _JOINT_MASK_MAP.get(mask_key, None)
    sigma_frac = float(kfs[0].get("sigma_frac", 0.04))

    # ---- 1. Baseline (no steering) ----
    with torch.no_grad():
        base_out = pipeline.generate(
            text=prompt,
            seed_input=[seed],
            duration_slider=duration,
            cfg_scale=args.cfg_scale,
        )
    base_joints = pipeline_output_to_world_joints(base_out)   # (1, T, 22, 3)

    # ---- 2. Extract canonical pose target from this seed's baseline ----
    target_np = _extract_target(base_joints[0], t_norm)        # (22, 3)
    target_t  = torch.from_numpy(target_np).float()

    # Compute baseline pose error (should be ~0 by construction)
    err_base = _pose_hit_error(base_joints[0], t_norm, target_np, joint_mask)

    # ---- 3. Build PoseConstraint and scheduler ----
    pc        = PoseConstraint(
        keyframes=[(t_norm, target_t)],
        joint_mask=joint_mask,
        sigma_frac=sigma_frac,
    )
    constraint = CompositeConstraint([(pc, 1.0)])

    # Mid-to-late cosine schedule: pose is a local structural constraint,
    # should not lock down global trajectory in early ODE steps.
    scheduler = StagedScheduler(
        alpha_max=alpha_pose,
        mode="cosine",
        t_start=0.2,
        t_end=0.95,
    )

    steerer = FlowSteerer(
        pipeline=pipeline,
        decoder=decoder,
        constraints=constraint,
        scheduler=scheduler,
        steps=args.steps,
        smooth_kernel=args.smooth_kernel,
    )

    # ---- 4. Steered generation (same seed) ----
    steer_out    = steerer.generate(
        text=prompt,
        seed_input=[seed],
        duration_slider=duration,
        cfg_scale=args.cfg_scale,
    )
    steer_joints = pipeline_output_to_world_joints(steer_out)  # (1, T, 22, 3)

    # ---- 5. Metrics ----
    err_steer   = _pose_hit_error(steer_joints[0], t_norm, target_np, joint_mask)
    improvement = (err_base - err_steer) / (err_base + 1e-8) * 100.0

    q_base  = compute_quality_metrics(base_joints)
    q_steer = compute_quality_metrics(steer_joints)

    jerk_ratio  = q_steer.mean_jerk        / (q_base.mean_jerk        + 1e-9)
    kinvar_ratio = q_steer.kinematic_variance / (q_base.kinematic_variance + 1e-9)

    return {
        "seed":                   seed,
        "prompt":                 prompt,
        "duration":               duration,
        "variance":               variance,
        "t_norm":                 t_norm,
        "joint_mask":             mask_key,
        "alpha_pose":             alpha_pose,
        "pose_hit_baseline":      err_base,
        "pose_hit_steered":       err_steer,
        "pose_hit_improvement_pct": improvement,
        "jerk_baseline":          q_base.mean_jerk,
        "jerk_steered":           q_steer.mean_jerk,
        "jerk_ratio":             jerk_ratio,
        "kinvar_baseline":        q_base.kinematic_variance,
        "kinvar_steered":         q_steer.kinematic_variance,
        "kinvar_ratio":           kinvar_ratio,
    }


# ---------------------------------------------------------------------------
# Aggregate + print helpers
# ---------------------------------------------------------------------------

def _agg(rows: List[Dict], key: str) -> Tuple[float, float]:
    vals = [r[key] for r in rows if not math.isnan(r.get(key, float("nan")))]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std())


def _print_summary(rows: List[Dict], title: str) -> None:
    if not rows:
        return
    phi_m, phi_s   = _agg(rows, "pose_hit_improvement_pct")
    jerr_m, jerr_s = _agg(rows, "jerk_ratio")
    kvr_m, kvr_s   = _agg(rows, "kinvar_ratio")
    n = len(rows)
    print(f"\n  {title}  (n={n})")
    print(f"    pose_hit improvement  : {phi_m:+.1f}%  ±{phi_s:.1f}%")
    print(f"    jerk ratio            : {jerr_m:.3f}  ±{jerr_s:.3f}  "
          f"({(jerr_m-1)*100:+.1f}%)")
    print(f"    kinvar ratio          : {kvr_m:.3f}  ±{kvr_s:.3f}  "
          f"({(kvr_m-1)*100:+.1f}%)")


def _print_per_prompt(rows: List[Dict]) -> None:
    print(f"\n  {'prompt':42} {'var':4} {'base_err':9} {'steer_err':9} {'Δerr%':7} {'jerk×':7} {'kv×':7}")
    print("  " + "-" * 90)
    for r in rows:
        p = r["prompt"][:40] + ".." if len(r["prompt"]) > 42 else r["prompt"]
        print(f"  {p:42} {r['variance']:4} "
              f"{r['pose_hit_baseline']:9.4f} {r['pose_hit_steered']:9.4f} "
              f"{r['pose_hit_improvement_pct']:+7.1f}% "
              f"{r['jerk_ratio']:7.2f} {r['kinvar_ratio']:7.2f}")


def _json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pose self-consistency evaluation (Protocol 1)"
    )
    parser.add_argument("--model_path",    required=True)
    parser.add_argument("--prompt_file",   required=True,
                        help="pose_eval_raw.json (contains pose_keyframes_raw + variance field)")
    parser.add_argument("--output_dir",    default="output/eval_pose_sc")
    parser.add_argument("--seeds",         default="42",
                        help="Comma-separated seeds.  Each seed is run independently "
                             "with its own target (self-consistency).")
    parser.add_argument("--alpha_pose",    type=float, default=8.0,
                        help="Steering strength for PoseConstraint.")
    parser.add_argument("--alpha_sweep",   default=None,
                        help="Comma-separated list of alpha values to sweep "
                             "(overrides --alpha_pose; output_dir gets alpha suffix).")
    parser.add_argument("--steps",         type=int,   default=50)
    parser.add_argument("--smooth_kernel", type=int,   default=5)
    parser.add_argument("--cfg_scale",     type=float, default=5.0)
    parser.add_argument("--gpu_id",        type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seeds  = [int(s) for s in args.seeds.split(",")]

    if args.alpha_sweep is not None:
        alphas = [float(a) for a in args.alpha_sweep.split(",")]
    else:
        alphas = [args.alpha_pose]

    with open(args.prompt_file) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts  seeds={seeds}  alphas={alphas}")

    # --- Load pipeline + decoder ---
    print("Loading HY-Motion pipeline...")
    pipeline = load_pipeline(args.model_path, device)
    decoder  = MotionDecoder.from_stats_dir(
        stats_dir=os.path.join(_repo_root, "stats"),
        body_model_path=os.path.join(_repo_root, "scripts/gradio/static/assets/dump_wooden"),
    )

    sweep_summaries = {}  # alpha → {all_rows, low_rows, high_rows}

    for alpha in alphas:
        out_dir = args.output_dir
        if len(alphas) > 1:
            out_dir = f"{args.output_dir}_a{alpha:.0f}"
        os.makedirs(out_dir, exist_ok=True)

        all_rows: List[Dict] = []
        t0_total = time.time()

        for p_idx, prompt_cfg in enumerate(prompts):
            for seed in seeds:
                label = f"[{p_idx+1}/{len(prompts)}] seed={seed}  α={alpha}"
                print(f"\n{label}  {prompt_cfg['prompt'][:50]!r}")

                t0 = time.time()
                row = run_single(pipeline, decoder, prompt_cfg, seed, alpha, args)
                elapsed = time.time() - t0

                print(f"  pose_hit: {row['pose_hit_baseline']:.4f} → {row['pose_hit_steered']:.4f} "
                      f"({row['pose_hit_improvement_pct']:+.1f}%)  "
                      f"jerk×{row['jerk_ratio']:.2f}  kv×{row['kinvar_ratio']:.2f}  "
                      f"[{elapsed:.1f}s]")
                all_rows.append(row)

        total_time = time.time() - t0_total
        print(f"\n{'='*70}")
        print(f"  SELF-CONSISTENCY RESULTS  α={alpha}  "
              f"({len(prompts)} prompts × {len(seeds)} seeds = {len(all_rows)} trials)")
        print(f"  Total time: {total_time/60:.1f} min")
        print(f"{'='*70}")

        low_rows  = [r for r in all_rows if r.get("variance") == "low"]
        high_rows = [r for r in all_rows if r.get("variance") == "high"]

        _print_summary(all_rows,  "ALL")
        _print_summary(low_rows,  "LOW-variance  (walk/run/march/skip)")
        _print_summary(high_rows, "HIGH-variance (dance/kick)")
        _print_per_prompt(all_rows)

        # Save
        result_path = os.path.join(out_dir, "results.json")
        with open(result_path, "w") as f:
            json.dump(all_rows, f, indent=2, default=_json_safe)
        print(f"\nSaved: {result_path}")

        sweep_summaries[alpha] = {
            "all":  _agg(all_rows,  "pose_hit_improvement_pct"),
            "low":  _agg(low_rows,  "pose_hit_improvement_pct"),
            "high": _agg(high_rows, "pose_hit_improvement_pct"),
            "jerk": _agg(all_rows,  "jerk_ratio"),
            "kv":   _agg(all_rows,  "kinvar_ratio"),
        }

    # ---- Sweep summary table ----
    if len(alphas) > 1:
        print(f"\n{'='*70}")
        print(f"  ALPHA SWEEP SUMMARY")
        print(f"{'='*70}")
        print(f"  {'alpha':>6}  {'pose_imp_all':>13}  {'pose_imp_low':>13}  "
              f"{'pose_imp_high':>13}  {'jerk×':>7}  {'kv×':>7}")
        print("  " + "-" * 68)
        for alpha in alphas:
            s = sweep_summaries[alpha]
            print(f"  {alpha:6.1f}  "
                  f"{s['all'][0]:+12.1f}%  "
                  f"{s['low'][0]:+12.1f}%  "
                  f"{s['high'][0]:+12.1f}%  "
                  f"{s['jerk'][0]:7.3f}  "
                  f"{s['kv'][0]:7.3f}")

        sweep_path = os.path.join(args.output_dir + "_sweep_summary.json")
        with open(sweep_path, "w") as f:
            json.dump({str(a): v for a, v in sweep_summaries.items()},
                      f, indent=2, default=_json_safe)
        print(f"\nSweep summary saved: {sweep_path}")


if __name__ == "__main__":
    main()
