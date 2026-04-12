"""
Multi-constraint self-consistency evaluation (Priority 2).

Tests timed-composite steering on three combinations:
    A: pose + foot_contact
    B: pose + waypoint
    C: pose + foot_contact + waypoint

Protocol (cross-seed, mirrors run_pose_sc.py):
    1. Generate baseline with target_seed=42.
       - Extract canonical pose at t_norm  → PoseConstraint target.
       - Extract root XZ at t=1.0          → WaypointConstraint terminal target.
    2. Generate baseline with steer_seed (43 or 44).
       - Measure err_base_pose    = pose distance to target.
       - Measure err_base_term    = terminal XZ distance to target.
       - Measure foot_slide_base  = FootContactConstraint proxy metric.
    3. Generate steered with steer_seed using timed-composite:
       - waypoint  [0.15, 0.65]
       - pose      [0.50, 0.88]
       - foot      [0.72, 1.00]
    4. Report per-constraint improvement + quality cost.

Usage:
    python eval/run_multiconstraint_sc.py \\
        --model_path ckpts/tencent/HY-Motion-1.0 \\
        --prompt_file eval/prompts/pose_eval_raw.json \\
        --combo pose+foot \\
        --alpha 6.0 \\
        --output_dir output/ablation_mc_pose_foot

    --combo choices: pose+foot  pose+waypoint  pose+foot+waypoint
    --prompts_subset low   (only low-variance prompts, faster)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

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
    FootContactConstraint,
    MotionDecoder,
    PoseConstraint,
    StagedScheduler,
    UPPER_BODY_JOINTS,
    WaypointConstraint,
)
from steering.decode import ROOT_JOINT
from eval.metrics import (
    canonicalize_frame_np,
    compute_quality_metrics,
    pipeline_output_to_world_joints,
)

_JOINT_MASK_MAP = {
    "all":        None,
    "upper_body": UPPER_BODY_JOINTS,
    "arms":       ARM_JOINTS,
}

# Default timed windows (matches run_eval.py _DEFAULT_TIMED_WINDOWS)
_T_WAYPOINT = (0.15, 0.65)
_T_POSE     = (0.50, 0.88)
_T_FOOT     = (0.72, 1.00)


def load_pipeline(model_path: str, device: torch.device):
    cfg_path  = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    pipeline = load_object(
        cfg["train_pipeline"], cfg["train_pipeline_args"],
        network_module=cfg["network_module"],
        network_module_args=cfg["network_module_args"],
    )
    pipeline.load_in_demo(ckpt_path, build_text_encoder=True)
    pipeline.to(device)
    pipeline.eval()
    return pipeline


def _foot_slide_proxy(joints_np: np.ndarray) -> float:
    """Mean foot velocity at estimated contact frames (proxy for sliding)."""
    feet = joints_np[:, [7, 8, 10, 11], :]   # (T, 4, 3)
    foot_y = feet[:, :, 1]                    # (T, 4)
    floor_y = foot_y.min(axis=0, keepdims=True)
    rel_h = foot_y - floor_y                  # (T, 4)
    fvel  = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)   # (T-1, 4)
    contact = (rel_h[:-1] < 0.05).astype(float)
    return float((contact * fvel).mean())


def run_single(pipeline, decoder, prompt_cfg, steer_seed, alpha, combo, args):
    prompt   = prompt_cfg["prompt"]
    duration = prompt_cfg.get("duration", 3.0)
    variance = prompt_cfg.get("variance", "unknown")

    kfs        = prompt_cfg.get("pose_keyframes_raw", [])
    assert kfs, f"No pose_keyframes_raw in: {prompt}"
    t_norm     = float(kfs[0]["t_norm"])
    mask_key   = kfs[0].get("joint_mask", "upper_body")
    joint_mask = _JOINT_MASK_MAP.get(mask_key, None)
    sigma_frac = float(kfs[0].get("sigma_frac", 0.04))

    # --- 1. Target-seed baseline → extract pose + terminal waypoint ---
    with torch.no_grad():
        tgt_out = pipeline.generate(
            text=prompt, seed_input=[args.target_seed],
            duration_slider=duration, cfg_scale=args.cfg_scale,
        )
    tgt_joints  = pipeline_output_to_world_joints(tgt_out)   # (1,T,22,3)
    T           = tgt_joints.shape[1]
    target_pose = canonicalize_frame_np(tgt_joints[0, int(round(t_norm*(T-1)))])  # (22,3)
    # Terminal XZ: root position at last frame
    term_xz_np  = tgt_joints[0, -1, ROOT_JOINT, [0, 2]]      # (2,) numpy

    # --- 2. Steer-seed baseline → err_base ---
    with torch.no_grad():
        base_out = pipeline.generate(
            text=prompt, seed_input=[steer_seed],
            duration_slider=duration, cfg_scale=args.cfg_scale,
        )
    base_joints = pipeline_output_to_world_joints(base_out)

    def pose_err(joints_np):
        frame = int(round(t_norm * (joints_np.shape[0] - 1)))
        canon = canonicalize_frame_np(joints_np[frame])
        jm = joint_mask
        return float(np.linalg.norm(
            (canon[jm] if jm else canon) - (target_pose[jm] if jm else target_pose),
            axis=-1,
        ).mean())

    def term_err(joints_np):
        root_xz = joints_np[-1, ROOT_JOINT, [0, 2]]
        return float(np.linalg.norm(root_xz - term_xz_np))

    err_base_pose = pose_err(base_joints[0])
    err_base_term = term_err(base_joints[0])
    foot_base     = _foot_slide_proxy(base_joints[0])

    # --- 3. Build timed-composite constraint list ---
    target_pose_t = torch.from_numpy(target_pose).float()
    term_xz_t     = torch.from_numpy(term_xz_np.astype(np.float32))

    timed = []
    if "pose" in combo:
        pc = PoseConstraint(
            keyframes=[(t_norm, target_pose_t)],
            joint_mask=joint_mask, sigma_frac=sigma_frac,
            use_hierarchical=args.use_hierarchical,
        )
        timed.append((pc, 1.0, _T_POSE[0], _T_POSE[1]))
    if "foot" in combo:
        fc = FootContactConstraint(height_thresh=0.05, vel_thresh=0.02, detach_mask=True)
        timed.append((fc, 1.0, _T_FOOT[0], _T_FOOT[1]))
    if "waypoint" in combo:
        wc = WaypointConstraint([(1.0, term_xz_t)], sigma_frac=0.05)
        timed.append((wc, 1.0, _T_WAYPOINT[0], _T_WAYPOINT[1]))

    scheduler = StagedScheduler.cosine(alpha_max=alpha)
    steerer   = FlowSteerer(
        pipeline=pipeline, decoder=decoder,
        timed_constraints=timed, scheduler=scheduler,
        steps=args.steps, smooth_kernel=args.smooth_kernel,
        soft_norm_tau=args.soft_norm_tau,
        max_steer_ratio=args.max_steer_ratio,
        ema_momentum=args.ema_momentum,
        apply_latent_mask=args.apply_latent_mask,
        latent_mask_transl=args.latent_mask_transl,
        latent_mask_root_rot=args.latent_mask_root_rot,
        use_temporal_mask=not args.no_temporal_mask,
    )

    # --- 4. Steered generation ---
    steer_out    = steerer.generate(
        text=prompt, seed_input=[steer_seed],
        duration_slider=duration, cfg_scale=args.cfg_scale,
    )
    steer_joints = pipeline_output_to_world_joints(steer_out)

    err_steer_pose = pose_err(steer_joints[0])
    err_steer_term = term_err(steer_joints[0])
    foot_steer     = _foot_slide_proxy(steer_joints[0])

    pose_imp = (err_base_pose - err_steer_pose) / (err_base_pose + 1e-8) * 100.0
    term_imp = (err_base_term - err_steer_term) / (err_base_term + 1e-8) * 100.0
    foot_imp = (foot_base - foot_steer) / (foot_base + 1e-8) * 100.0

    q_base  = compute_quality_metrics(base_joints)
    q_steer = compute_quality_metrics(steer_joints)
    jerk_ratio   = q_steer.mean_jerk         / (q_base.mean_jerk         + 1e-9)
    kinvar_ratio = q_steer.kinematic_variance / (q_base.kinematic_variance + 1e-9)

    return {
        "target_seed": args.target_seed, "steer_seed": steer_seed,
        "prompt": prompt, "duration": duration, "variance": variance,
        "combo": combo, "alpha": alpha, "t_norm": t_norm,
        "pose_hit_baseline": err_base_pose, "pose_hit_steered": err_steer_pose,
        "pose_hit_improvement_pct": pose_imp,
        "term_err_baseline": err_base_term, "term_err_steered": err_steer_term,
        "term_improvement_pct": term_imp,
        "foot_slide_baseline": foot_base, "foot_slide_steered": foot_steer,
        "foot_improvement_pct": foot_imp,
        "jerk_ratio": jerk_ratio, "kinvar_ratio": kinvar_ratio,
    }


def _agg(rows, key):
    v = [r[key] for r in rows if not math.isnan(r.get(key, float('nan')))]
    if not v: return float('nan'), float('nan')
    arr = np.array(v)
    return float(arr.mean()), float(arr.std())


def _json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj): return None
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.integer): return int(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description="Multi-constraint self-consistency eval")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--prompt_file",  required=True)
    parser.add_argument("--output_dir",   default="output/ablation_mc")
    parser.add_argument("--combo",        default="pose+foot",
                        choices=["pose+foot", "pose+waypoint", "pose+foot+waypoint"],
                        help="Which constraint combination to test.")
    parser.add_argument("--prompts_subset", default="all",
                        choices=["all", "low", "high"],
                        help="Run only low/high variance prompts (faster).")
    parser.add_argument("--target_seed",  type=int, default=42)
    parser.add_argument("--seeds",        default="43",
                        help="Steer seeds (comma-separated, must differ from target_seed).")
    parser.add_argument("--alpha",        type=float, default=6.0)
    parser.add_argument("--steps",        type=int,   default=50)
    parser.add_argument("--smooth_kernel",type=int,   default=7)
    parser.add_argument("--soft_norm_tau",type=float, default=0.1)
    parser.add_argument("--max_steer_ratio", type=float, default=0.3)
    parser.add_argument("--ema_momentum",    type=float, default=0.7)
    parser.add_argument("--use_hierarchical", action="store_true")
    parser.add_argument("--apply_latent_mask", action="store_true")
    parser.add_argument("--latent_mask_transl",   type=float, default=0.1)
    parser.add_argument("--latent_mask_root_rot", type=float, default=0.3)
    parser.add_argument("--no_temporal_mask", action="store_true")
    parser.add_argument("--cfg_scale",    type=float, default=5.0)
    parser.add_argument("--gpu_id",       type=int,   default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seeds  = [int(s) for s in args.seeds.split(",")]
    seeds  = [s for s in seeds if s != args.target_seed]
    assert seeds, "seeds must contain at least one seed != target_seed"

    with open(args.prompt_file) as f:
        prompts = json.load(f)
    if args.prompts_subset != "all":
        prompts = [p for p in prompts if p.get("variance") == args.prompts_subset]
    print(f"Prompts: {len(prompts)}  combo={args.combo}  α={args.alpha}  "
          f"tgt={args.target_seed}→steer={seeds}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading pipeline...")
    pipeline = load_pipeline(args.model_path, device)
    decoder  = MotionDecoder.from_stats_dir(
        stats_dir=os.path.join(_repo_root, "stats"),
        body_model_path=os.path.join(_repo_root, "scripts/gradio/static/assets/dump_wooden"),
    )

    all_rows: List[Dict] = []
    t0 = time.time()

    for p_idx, prompt_cfg in enumerate(prompts):
        for seed in seeds:
            print(f"\n[{p_idx+1}/{len(prompts)}] tgt={args.target_seed}→steer={seed}  "
                  f"{prompt_cfg['prompt'][:55]!r}")
            t1 = time.time()
            row = run_single(pipeline, decoder, prompt_cfg, seed, args.alpha, args.combo, args)
            print(f"  pose_imp={row['pose_hit_improvement_pct']:+.1f}%  "
                  f"term_imp={row['term_improvement_pct']:+.1f}%  "
                  f"foot_imp={row['foot_improvement_pct']:+.1f}%  "
                  f"jerk×{row['jerk_ratio']:.3f}  [{time.time()-t1:.1f}s]")
            all_rows.append(row)

    total = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  combo={args.combo}  α={args.alpha}  n={len(all_rows)}  "
          f"({total/60:.1f} min)")
    print(f"{'='*65}")

    low  = [r for r in all_rows if r.get("variance") == "low"]
    high = [r for r in all_rows if r.get("variance") == "high"]

    for label, subset in [("ALL", all_rows), ("low", low), ("high", high)]:
        if not subset: continue
        pi, ps = _agg(subset, "pose_hit_improvement_pct")
        ti, ts = _agg(subset, "term_improvement_pct")
        fi, fs = _agg(subset, "foot_improvement_pct")
        jr, _  = _agg(subset, "jerk_ratio")
        kv, _  = _agg(subset, "kinvar_ratio")
        print(f"  {label:>5}  n={len(subset):2d}  "
              f"pose_imp={pi:+.1f}%±{ps:.1f}  "
              f"term_imp={ti:+.1f}%±{ts:.1f}  "
              f"foot_imp={fi:+.1f}%±{fs:.1f}  "
              f"jerk×{jr:.3f}  kv×{kv:.3f}")

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=_json_safe)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
