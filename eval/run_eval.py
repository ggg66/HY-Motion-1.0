"""
Batch evaluation script for FlowSteer-Motion.

Runs HY-Motion 1.0 with and without steering on a set of prompts,
computes all constraint + quality metrics, prints a comparison table,
and saves side-by-side videos and analysis plots.

Usage:
    # Quick 5-prompt sanity check (default):
    python eval/run_eval.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --output_dir output/eval

    # Full Phase-1 evaluation (100 prompts, 3 seeds):
    python eval/run_eval.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt_file eval/prompts/full_eval.json \
        --seeds 42,43,44 \
        --output_dir output/eval_full \
        --no_video

    # Resume an interrupted run:
    python eval/run_eval.py \
        --model_path ckpts/tencent/HY-Motion-1.0 \
        --prompt_file eval/prompts/full_eval.json \
        --seeds 42,43,44 \
        --output_dir output/eval_full \
        --resume

Prompt JSON format:
    [
        {
            "prompt": "a person walks forward and stops",
            "duration": 3.0,
            "constraint": ["foot_contact"],
            "terminal_xz": [2.0, 0.0]     # optional
        },
        ...
    ]

Output per prompt:
    output/eval/<idx>_baseline.npy
    output/eval/<idx>_steered.npy
    output/eval/<idx>_comparison.mp4   (skipped with --no_video)
    output/eval/<idx>_contact_analysis.png

Final output:
    output/eval/results_summary.json
    output/eval/results_table.txt
    output/eval/aggregate_stats.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# numpy is used for loading pose target files (.npy) — ensure it is imported
# before any constraint-building code.

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hymotion.utils.loaders import load_object
from steering import (
    ARM_JOINTS,
    CompositeConstraint,
    FlowSteerer,
    FootContactConstraint,
    LEG_JOINTS,
    LOWER_BODY_JOINTS,
    MotionDecoder,
    PoseConstraint,
    StagedScheduler,
    TerminalConstraint,
    UPPER_BODY_JOINTS,
    WaypointConstraint,
)
from eval.metrics import (
    compute_constraint_metrics,
    compute_quality_metrics,
    pipeline_output_to_world_joints,
    print_comparison_table,
)
from eval.visualize import save_comparison_video, plot_foot_contact_analysis


# ---------------------------------------------------------------------------
# Default evaluation prompts (5-prompt sanity check)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    {
        "prompt": "a person walks forward and stops",
        "duration": 3.0,
        "constraint": ["foot_contact"],
    },
    {
        "prompt": "a person runs and then slows down",
        "duration": 4.0,
        "constraint": ["foot_contact"],
    },
    {
        "prompt": "a person walks to a destination",
        "duration": 4.0,
        "constraint": ["foot_contact", "terminal"],
        "terminal_xz": [2.5, 0.0],
    },
    {
        "prompt": "a person jogs in a circle",
        "duration": 5.0,
        "constraint": ["foot_contact"],
    },
    {
        "prompt": "a person dances",
        "duration": 4.0,
        "constraint": ["foot_contact"],
    },
]


# ---------------------------------------------------------------------------
# Pipeline loader
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
# Per-prompt helpers
# ---------------------------------------------------------------------------

_JOINT_MASK_MAP: Dict[str, Optional[List[int]]] = {
    "all":        None,
    "upper_body": UPPER_BODY_JOINTS,
    "lower_body": LOWER_BODY_JOINTS,
    "arms":       ARM_JOINTS,
    "legs":       LEG_JOINTS,
}

# Default ODE-time windows for each constraint type in timed-composite mode.
# (t_start, t_end, weight)
# These encode the heterogeneous staged steering schedule:
#   waypoint/terminal: early-mid  [0.15, 0.65] — establish path/destination
#   pose:              mid-late   [0.50, 0.88] — refine body configuration
#   foot_contact:      late       [0.72, 1.0]  — enforce physical stability
_DEFAULT_TIMED_WINDOWS: Dict[str, tuple] = {
    "foot_contact": (0.72, 1.0,  1.0),
    "terminal":     (0.15, 0.65, 1.5),
    "waypoint":     (0.15, 0.65, 1.0),
    "pose":         (0.50, 0.88, 1.0),
}


def build_timed_constraints_for_prompt(prompt_cfg: Dict, args) -> List[tuple]:
    """
    Build a timed-composite constraint list for the heterogeneous staged steering path.

    Returns:
        list of (BaseConstraint, weight, t_start, t_end) tuples.

    Time windows are read from the prompt's "timed" field (if present), with fallback
    to _DEFAULT_TIMED_WINDOWS.  The "timed" field maps constraint type → dict with
    optional keys: t_start, t_end, weight.

    Example prompt JSON:
        {
            "prompt": "...",
            "constraint": ["foot_contact", "pose", "waypoint"],
            "timed": {
                "pose":         {"t_start": 0.45, "t_end": 0.90, "weight": 1.2},
                "foot_contact": {"t_start": 0.70, "t_end": 1.0,  "weight": 1.0}
            }
        }
    """
    constraint_types = prompt_cfg.get("constraint", args.constraint)
    timed_overrides  = prompt_cfg.get("timed", {})
    result = []

    for ctype in constraint_types:
        defaults = _DEFAULT_TIMED_WINDOWS.get(ctype, (0.0, 1.0, 1.0))
        ov       = timed_overrides.get(ctype, {})
        t_start  = float(ov.get("t_start", defaults[0]))
        t_end    = float(ov.get("t_end",   defaults[1]))
        weight   = float(ov.get("weight",  defaults[2]))

        if ctype == "foot_contact":
            c = FootContactConstraint(
                height_thresh=0.05, vel_thresh=0.02,
                detach_mask=not args.no_detach_mask,
            )
        elif ctype == "terminal":
            xz     = prompt_cfg.get("terminal_xz", [args.terminal_x, args.terminal_z])
            target = torch.tensor([[xz[0], 0.9, xz[1]]], dtype=torch.float32)
            c      = TerminalConstraint(target_joints=target, joint_indices=[0], tail_frames=4)
        elif ctype == "waypoint":
            raw_wps = prompt_cfg.get("waypoints", [])
            wps = [(float(w["t_norm"]),
                    torch.tensor(w["xz"], dtype=torch.float32))
                   for w in raw_wps]
            c   = WaypointConstraint(wps, sigma_frac=0.05)
        elif ctype == "pose":
            raw_kfs   = prompt_cfg.get("pose_keyframes", [])
            keyframes = []
            for kf in raw_kfs:
                target_np = np.load(kf["target_file"])
                target    = torch.from_numpy(target_np).float()
                keyframes.append((float(kf["t_norm"]), target))
            mask_key      = raw_kfs[0].get("joint_mask", "upper_body") if raw_kfs else "upper_body"
            joint_mask    = _JOINT_MASK_MAP.get(mask_key, None)
            sigma_frac    = raw_kfs[0].get("sigma_frac", 0.04) if raw_kfs else 0.04
            c = PoseConstraint(
                keyframes, joint_mask=joint_mask, sigma_frac=sigma_frac,
                use_hierarchical=args.use_hierarchical,
            )
        else:
            continue

        result.append((c, weight, t_start, t_end))

    assert result, f"No timed constraints for prompt: {prompt_cfg['prompt']}"
    return result


def build_constraints_for_prompt(prompt_cfg: Dict, args) -> CompositeConstraint:
    """
    Build a static CompositeConstraint for the given prompt.

    Supported constraint types (set via prompt JSON "constraint" field):
        foot_contact  weight 1.0  — physical foot stability
        terminal      weight 1.5  — root XZ terminal position
        waypoint      weight 1.0  — multi-point root XZ path
        pose          weight 1.0  — canonical keyframe body configuration
    """
    constraint_types = prompt_cfg.get("constraint", args.constraint)
    constraint_list  = []

    if "foot_contact" in constraint_types:
        fc = FootContactConstraint(
            height_thresh=0.05, vel_thresh=0.02,
            detach_mask=not args.no_detach_mask,
        )
        constraint_list.append((fc, 1.0))

    if "terminal" in constraint_types:
        xz     = prompt_cfg.get("terminal_xz", [args.terminal_x, args.terminal_z])
        target = torch.tensor([[xz[0], 0.9, xz[1]]], dtype=torch.float32)
        tc     = TerminalConstraint(target_joints=target, joint_indices=[0], tail_frames=4)
        constraint_list.append((tc, 1.5))

    if "waypoint" in constraint_types:
        raw_wps = prompt_cfg.get("waypoints", [])
        wps = [(float(w["t_norm"]),
                torch.tensor(w["xz"], dtype=torch.float32))
               for w in raw_wps]
        wc = WaypointConstraint(wps, sigma_frac=0.05)
        constraint_list.append((wc, 1.0))

    if "pose" in constraint_types:
        raw_kfs = prompt_cfg.get("pose_keyframes", [])
        keyframes = []
        for kf in raw_kfs:
            target_np = np.load(kf["target_file"])             # (22, 3)
            target    = torch.from_numpy(target_np).float()
            keyframes.append((float(kf["t_norm"]), target))
        mask_key   = raw_kfs[0].get("joint_mask", "upper_body") if raw_kfs else "upper_body"
        joint_mask = _JOINT_MASK_MAP.get(mask_key, None)
        sigma_frac = raw_kfs[0].get("sigma_frac", 0.04) if raw_kfs else 0.04
        pc = PoseConstraint(
            keyframes, joint_mask=joint_mask, sigma_frac=sigma_frac,
            use_hierarchical=args.use_hierarchical,
        )
        constraint_list.append((pc, 1.0))

    assert constraint_list, f"No constraints for prompt: {prompt_cfg['prompt']}"
    return CompositeConstraint(constraint_list, normalize_losses=args.normalize_losses)


def extract_terminal_targets(prompt_cfg: Dict, args) -> Optional[List]:
    if "terminal" not in prompt_cfg.get("constraint", args.constraint):
        return None
    xz = prompt_cfg.get("terminal_xz", [args.terminal_x, args.terminal_z])
    target_np = np.array([[xz[0], 0.9, xz[1]]], dtype=np.float32)
    return [([0], target_np)]


def extract_pose_targets_for_metrics(prompt_cfg: Dict, args) -> Optional[List]:
    """
    Extract pose targets from prompt config for use in compute_constraint_metrics.

    Returns:
        List of (t_norm, target_pose (22,3) np.ndarray, joint_mask list or None)
        or None if no pose constraint.
    """
    constraint_types = prompt_cfg.get("constraint", args.constraint)
    if "pose" not in constraint_types:
        return None
    raw_kfs = prompt_cfg.get("pose_keyframes", [])
    if not raw_kfs:
        return None
    result = []
    for kf in raw_kfs:
        target_np  = np.load(kf["target_file"])                        # (22, 3)
        t_norm     = float(kf["t_norm"])
        mask_key   = kf.get("joint_mask", "upper_body")
        joint_mask = _JOINT_MASK_MAP.get(mask_key, None)
        result.append((t_norm, target_np, joint_mask))
    return result


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, "_progress.json")


def _load_progress(output_dir: str) -> Dict:
    path = _checkpoint_path(output_dir)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_idx": [], "results": []}


def _save_progress(output_dir: str, progress: Dict) -> None:
    with open(_checkpoint_path(output_dir), "w") as f:
        json.dump(progress, f, default=_json_safe)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    seeds  = [int(s) for s in args.seeds.split(",")]

    # --- Load prompts ---
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using {len(prompts)} default prompts (5-prompt sanity check)")

    # --- Resume support ---
    os.makedirs(args.output_dir, exist_ok=True)
    progress = _load_progress(args.output_dir) if args.resume else {"completed_idx": [], "results": []}
    completed = set(progress["completed_idx"])
    all_results = progress["results"]

    if completed:
        print(f"Resuming: {len(completed)}/{len(prompts)} already done")

    # --- Load pipeline ---
    print("Loading HY-Motion pipeline...")
    pipeline = load_pipeline(args.model_path, device)

    # --- Build common decoder ---
    decoder = MotionDecoder.from_stats_dir(
        stats_dir=os.path.join(_repo_root, "stats"),
        body_model_path=os.path.join(_repo_root, "scripts/gradio/static/assets/dump_wooden"),
    )

    t_total_start = time.time()

    for idx, prompt_cfg in enumerate(prompts):
        global_idx = idx + 1
        if global_idx in completed:
            print(f"[{global_idx}/{len(prompts)}] SKIP (already done)")
            continue

        prompt   = prompt_cfg["prompt"]
        duration = prompt_cfg.get("duration", 3.0)
        print(f"\n[{global_idx}/{len(prompts)}] {prompt!r}  ({duration}s)")

        # --- Build constraints ---
        constraint_types = prompt_cfg.get("constraint", args.constraint)
        terminal_targets = extract_terminal_targets(prompt_cfg, args)

        if args.no_timed:
            # Legacy static path (ablation / backward compat)
            constraints = build_constraints_for_prompt(prompt_cfg, args)
            if set(constraint_types) == {"pose"}:
                scheduler = StagedScheduler(
                    alpha_max=args.alpha_pose, mode="cosine", t_start=0.5, t_end=0.88,
                )
            elif args.scheduler == "staged":
                scheduler = StagedScheduler.make_staged(
                    alpha_terminal=args.alpha_terminal,
                    alpha_waypoint=args.alpha_terminal,
                    alpha_contact=args.alpha_contact,
                )
            elif args.scheduler == "constant":
                scheduler = StagedScheduler.constant(alpha_max=args.alpha_terminal)
            elif args.scheduler == "cosine":
                scheduler = StagedScheduler.cosine(alpha_max=args.alpha_terminal)
            else:
                raise ValueError(f"Unknown scheduler: {args.scheduler}")
            steerer = FlowSteerer(
                pipeline=pipeline, decoder=decoder,
                constraints=constraints, scheduler=scheduler,
                steps=args.steps, smooth_kernel=args.smooth_kernel,
                soft_norm_tau=args.soft_norm_tau, use_unit_grad=args.use_unit_grad,
                max_steer_ratio=args.max_steer_ratio, ema_momentum=args.ema_momentum,
                apply_latent_mask=args.apply_latent_mask,
                latent_mask_transl=args.latent_mask_transl,
                latent_mask_root_rot=args.latent_mask_root_rot,
                verbose=args.verbose,
            )
        else:
            # Timed-composite path (default): heterogeneous staged steering.
            # waypoint/terminal [0.15,0.65] → pose [0.50,0.88] → foot [0.72,1.0]
            # A single cosine StagedScheduler controls global alpha magnitude;
            # individual constraint time windows gate which constraints are active.
            timed_constraints = build_timed_constraints_for_prompt(prompt_cfg, args)

            # Pick alpha scale based on dominant constraint type
            if set(constraint_types) == {"pose"}:
                alpha_max = args.alpha_pose
            elif "foot_contact" in constraint_types and not any(
                c in constraint_types for c in ("terminal", "waypoint", "pose")
            ):
                alpha_max = args.alpha_contact
            else:
                alpha_max = args.alpha_terminal

            scheduler = StagedScheduler.cosine(alpha_max=alpha_max)

            steerer = FlowSteerer(
                pipeline=pipeline, decoder=decoder,
                timed_constraints=timed_constraints, scheduler=scheduler,
                steps=args.steps, smooth_kernel=args.smooth_kernel,
                soft_norm_tau=args.soft_norm_tau, use_unit_grad=args.use_unit_grad,
                max_steer_ratio=args.max_steer_ratio, ema_momentum=args.ema_momentum,
                apply_latent_mask=args.apply_latent_mask,
                latent_mask_transl=args.latent_mask_transl,
                latent_mask_root_rot=args.latent_mask_root_rot,
                verbose=args.verbose,
            )

        # ---- Baseline (no steering) ----
        t0 = time.time()
        with torch.no_grad():
            baseline_out = pipeline.generate(
                text=prompt,
                seed_input=seeds,
                duration_slider=duration,
                cfg_scale=args.cfg_scale,
            )
        baseline_joints = pipeline_output_to_world_joints(baseline_out)  # (B, T, 22, 3)
        t_base = time.time() - t0
        print(f"  Baseline:  {t_base:.1f}s  (B={baseline_joints.shape[0]}, T={baseline_joints.shape[1]})")

        # ---- Steered ----
        t0 = time.time()
        steered_out    = steerer.generate(
            text=prompt,
            seed_input=seeds,
            duration_slider=duration,
            cfg_scale=args.cfg_scale,
        )
        steered_joints = pipeline_output_to_world_joints(steered_out)    # (B, T, 22, 3)
        t_steer = time.time() - t0
        print(f"  Steered:   {t_steer:.1f}s")

        # ---- Metrics ----
        pose_targets = extract_pose_targets_for_metrics(prompt_cfg, args)
        b_c = compute_constraint_metrics(baseline_joints, terminal_targets=terminal_targets, pose_targets=pose_targets)
        s_c = compute_constraint_metrics(steered_joints,  terminal_targets=terminal_targets, pose_targets=pose_targets)
        b_q = compute_quality_metrics(baseline_joints)
        s_q = compute_quality_metrics(steered_joints)

        b_all = {**b_c.to_dict(), **b_q.to_dict()}
        s_all = {**s_c.to_dict(), **s_q.to_dict()}

        constraint_label = "+".join(prompt_cfg.get("constraint", args.constraint))
        print_comparison_table(b_all, s_all,
                               title=f'[{global_idx}] "{prompt}"  [{constraint_label}]')

        # ---- Save .npy ----
        tag = f"{global_idx:04d}"
        np.save(os.path.join(args.output_dir, f"{tag}_baseline.npy"), baseline_joints)
        np.save(os.path.join(args.output_dir, f"{tag}_steered.npy"),  steered_joints)

        # ---- Video + plot (optional) ----
        if not args.no_video:
            try:
                save_comparison_video(
                    baseline_joints=baseline_joints[0],
                    steered_joints=steered_joints[0],
                    output_path=os.path.join(args.output_dir, f"{tag}_comparison.mp4"),
                    fps=30,
                    prompt=prompt,
                    constraint_label=constraint_label,
                )
                plot_foot_contact_analysis(
                    baseline_joints=baseline_joints[0],
                    steered_joints=steered_joints[0],
                    output_path=os.path.join(args.output_dir, f"{tag}_contact_analysis.png"),
                    fps=30,
                )
            except Exception as e:
                print(f"  [Warning] Visualization failed: {e}")

        # ---- Record ----
        result = {
            "idx": global_idx,
            "prompt": prompt,
            "duration": duration,
            "constraint": prompt_cfg.get("constraint", args.constraint),
            "seeds": seeds,
            "time_baseline": round(t_base, 2),
            "time_steered":  round(t_steer, 2),
            "baseline": b_all,
            "steered":  s_all,
        }
        if "terminal_xz" in prompt_cfg:
            result["terminal_xz"] = prompt_cfg["terminal_xz"]

        all_results.append(result)
        completed.add(global_idx)

        # ---- Checkpoint ----
        progress["completed_idx"] = list(completed)
        progress["results"]       = all_results
        _save_progress(args.output_dir, progress)

    # ---- Aggregate summary ----
    t_total = time.time() - t_total_start
    print(f"\n{'='*70}")
    print(f"  AGGREGATE SUMMARY  ({len(all_results)} prompts, seeds={seeds})")
    print(f"  Total wall time: {t_total/60:.1f} min")
    print(f"{'='*70}")

    agg = _aggregate_stats(all_results)
    for k, v in agg.items():
        b_mean = v["baseline_mean"]
        s_mean = v["steered_mean"]
        delta  = s_mean - b_mean
        pct    = delta / (abs(b_mean) + 1e-9) * 100
        print(f"  {k:<32s}  B={b_mean:.5f}  S={s_mean:.5f}  "
              f"Δ={delta:+.5f} ({pct:+.1f}%)  n={v['n']}")

    # ---- Save JSON ----
    results_path = os.path.join(args.output_dir, "results_summary.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    print(f"\nResults saved: {results_path}")

    agg_path = os.path.join(args.output_dir, "aggregate_stats.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=_json_safe)
    print(f"Aggregate stats: {agg_path}")

    # ---- Save plain-text table ----
    _save_text_table(all_results, os.path.join(args.output_dir, "results_table.txt"))

    # ---- Clean up checkpoint ----
    prog_path = _checkpoint_path(args.output_dir)
    if os.path.exists(prog_path):
        os.remove(prog_path)
        print("Checkpoint removed (run complete).")


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def _aggregate_stats(results: List[Dict]) -> Dict:
    """Compute per-metric mean ± std across all results, separated by constraint type."""
    if not results:
        return {}

    all_keys = [k for k in results[0]["baseline"]
                if not _is_nan(results[0]["baseline"][k])]

    agg = {}
    for k in all_keys:
        b_vals = [r["baseline"][k] for r in results if not _is_nan(r["baseline"].get(k, float("nan")))]
        s_vals = [r["steered"][k]  for r in results if not _is_nan(r["steered"].get(k, float("nan")))]
        if not b_vals:
            continue
        b_arr, s_arr = np.array(b_vals), np.array(s_vals)
        agg[k] = {
            "baseline_mean": float(b_arr.mean()),
            "baseline_std":  float(b_arr.std()),
            "steered_mean":  float(s_arr.mean()),
            "steered_std":   float(s_arr.std()),
            "delta_mean":    float((s_arr - b_arr).mean()),
            "delta_pct":     float((s_arr - b_arr).mean() / (abs(b_arr.mean()) + 1e-9) * 100),
            "n":             len(b_vals),
        }
    return agg


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_nan(v):
    return isinstance(v, float) and math.isnan(v)


def _json_safe(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def _save_text_table(results: List[Dict], path: str) -> None:
    lines = []
    header = (f"{'#':>4}  {'Prompt':40}  {'Constraint':22}  "
              f"{'foot_sliding':>14}  {'violation_rate':>15}  {'term_err':>12}  "
              f"{'mean_jerk':>11}  {'kin_var':>9}")
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        prompt_short = r["prompt"][:38] + ".." if len(r["prompt"]) > 40 else r["prompt"]
        cname = "+".join(r["constraint"])[:20]
        b, s  = r["baseline"], r["steered"]

        def fmt(k, decimals=4):
            bv = b.get(k, float("nan"))
            sv = s.get(k, float("nan"))
            if _is_nan(bv):
                return "    N/A"
            return f"{bv:.{decimals}f}→{sv:.{decimals}f}"

        row = (
            f"{r['idx']:>4}  {prompt_short:40}  {cname:22}  "
            f"{fmt('foot_sliding_score'):>14}  "
            f"{fmt('contact_violation_rate'):>15}  "
            f"{fmt('terminal_position_error'):>12}  "
            f"{fmt('mean_jerk', 5):>11}  "
            f"{fmt('kinematic_variance', 4):>9}"
        )
        lines.append(row)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Table saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FlowSteer-Motion batch evaluation")
    parser.add_argument("--model_path",   required=True)
    parser.add_argument("--prompt_file",  default=None,
                        help="JSON prompt file (from scripts/sample_prompts.py). "
                             "Omit to use built-in 5-prompt sanity check.")
    parser.add_argument("--output_dir",   default="output/eval")
    parser.add_argument("--constraint",   nargs="+", default=["foot_contact"],
                        choices=["foot_contact", "terminal", "waypoint", "pose"])
    parser.add_argument("--terminal_x",  type=float, default=2.0)
    parser.add_argument("--terminal_z",  type=float, default=0.0)
    parser.add_argument("--scheduler",      default="staged",
                        choices=["staged", "constant", "cosine"],
                        help="Alpha schedule: staged (default), constant, or cosine.")
    parser.add_argument("--no_detach_mask", action="store_true",
                        help="[Ablation] Disable contact mask detach — lets gradient "
                             "reclassify frames as non-contact (original buggy behavior).")
    parser.add_argument("--normalize_losses", action="store_true",
                        help="Normalize each constraint loss by its detached value before "
                             "combining, equalising gradient magnitudes across constraints. "
                             "Fixes foot+terminal imbalance without breaking FK-chain coupling.")
    parser.add_argument("--alpha_terminal", type=float, default=80.0,
                        help="Peak steering strength for terminal/waypoint constraints "
                             "(StagedScheduler stages [0,0.7] and [0.2,0.9]).")
    parser.add_argument("--alpha_contact",  type=float, default=15.0,
                        help="Steering strength for foot-contact constraint "
                             "(StagedScheduler stage [0.5,1.0]). "
                             "Keep lower than alpha_terminal to avoid jerk.")
    parser.add_argument("--alpha_pose",    type=float, default=1.0,
                        help="Steering strength for pose constraint (timed path: peak α for "
                             "the global cosine schedule when pose-only; per-frame normalization "
                             "is 14× stronger per-element than flat, so ~1.0 is correct operating "
                             "point vs the legacy 8.0 with flat normalization).")
    parser.add_argument("--steps",        type=int,   default=50)
    parser.add_argument("--smooth_kernel", type=int,  default=7,
                        help="Temporal smoothing window for steering vector (odd int). "
                             "Suppresses high-frequency jerk. Set to 1 to disable.")
    parser.add_argument("--soft_norm_tau", type=float, default=0.1,
                        help="τ (relative multiplier) for per-frame adaptive soft-norm. "
                             "τ_abs = τ × mean(‖g_frame‖).  Frames near keyframe get "
                             "scale≈1; off-keyframe frames get scale≈0.  Default 0.1.")
    parser.add_argument("--no_timed",      action="store_true",
                        help="[Ablation] Use legacy static-composite path instead of "
                             "timed-composite heterogeneous staged steering (default: timed).")
    parser.add_argument("--use_hierarchical", action="store_true",
                        help="Enable hierarchical joint weighting in PoseConstraint: "
                             "wrists/elbows/shoulders high (3×), torso low (0.5×), others 1×.")
    parser.add_argument("--apply_latent_mask", action="store_true",
                        help="Apply latent trust mask: attenuate steering on transl/root_rot "
                             "dims to avoid position drift and yaw flips from pose steering.")
    parser.add_argument("--latent_mask_transl",   type=float, default=0.1,
                        help="Scale for translation dims [0:3] under latent trust mask. Default 0.1.")
    parser.add_argument("--latent_mask_root_rot", type=float, default=0.3,
                        help="Scale for root-rotation dims [3:9] under latent trust mask. Default 0.3.")
    parser.add_argument("--use_unit_grad", action="store_true",
                        help="[Ablation] Use per-frame unit-norm instead of adaptive "
                             "soft-norm. Restores pre-fix behaviour.")
    parser.add_argument("--max_steer_ratio", type=float, default=0.3,
                        help="Trust region: per-frame ‖α·s‖ clamped to at most "
                             "max_steer_ratio × ‖v‖.  Prevents steering from overriding "
                             "model dynamics.  0.0 = disabled.  Default 0.3.")
    parser.add_argument("--ema_momentum", type=float, default=0.7,
                        help="EMA coefficient for steering direction across ODE steps. "
                             "s_ema = μ·s_prev + (1-μ)·s_new.  0.0 = disabled.  Default 0.7.")
    parser.add_argument("--seeds",       default="42",
                        help="Comma-separated seed list (one motion per seed). "
                             "For Phase-1 eval use: 42,43,44")
    parser.add_argument("--cfg_scale",   type=float, default=5.0)
    parser.add_argument("--gpu_id",      type=int,   default=0)
    parser.add_argument("--no_video",    action="store_true",
                        help="Skip video/plot generation (strongly recommended for large evals)")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume a previously interrupted run from checkpoint")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()

    run_eval(args)


if __name__ == "__main__":
    main()
