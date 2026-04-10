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

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hymotion.utils.loaders import load_object
from steering import (
    CompositeConstraint,
    FlowSteerer,
    FootContactConstraint,
    MotionDecoder,
    PerConstraintScheduler,
    StagedScheduler,
    TerminalConstraint,
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

def build_constraint_schedulers(prompt_cfg: Dict, args) -> list:
    """
    Build a list of (BaseConstraint, StagedScheduler) pairs for per-constraint
    steering.  Each constraint gets an independent gradient and alpha schedule,
    preventing a high-magnitude constraint from suppressing a low-magnitude one.
    """
    constraint_types = prompt_cfg.get("constraint", args.constraint)
    pairs = []

    if "foot_contact" in constraint_types:
        fc = FootContactConstraint(height_thresh=0.05, vel_thresh=0.02)
        # Contact steers fine-grained physical detail.  Activate from t=0.2 so
        # the x̂_1 estimate is stable enough, ramp up over 15% of the window to
        # avoid a sudden velocity shock, then hold at full alpha.
        sched = StagedScheduler(
            alpha_max=args.alpha_contact,
            mode="linear_ramp",
            t_start=0.2,
            t_end=1.0,
            warmup_frac=0.15,
        )
        pairs.append((fc, sched))

    if "terminal" in constraint_types:
        xz = prompt_cfg.get("terminal_xz", [args.terminal_x, args.terminal_z])
        target = torch.tensor([[xz[0], 0.9, xz[1]]], dtype=torch.float32)
        tc = TerminalConstraint(target_joints=target, joint_indices=[0], tail_frames=4)
        # Terminal steers global structure — must be strong from the very first
        # ODE step when the trajectory skeleton is being set.  Use constant mode
        # (no warmup) so the full alpha is active immediately.
        sched = StagedScheduler(
            alpha_max=args.alpha_terminal,
            mode="constant",
            t_start=0.0,
            t_end=0.75,
        )
        pairs.append((tc, sched))

    assert pairs, f"No constraints for prompt: {prompt_cfg['prompt']}"
    return pairs


def extract_terminal_targets(prompt_cfg: Dict, args) -> Optional[List]:
    if "terminal" not in prompt_cfg.get("constraint", args.constraint):
        return None
    xz = prompt_cfg.get("terminal_xz", [args.terminal_x, args.terminal_z])
    target_np = np.array([[xz[0], 0.9, xz[1]]], dtype=np.float32)
    return [([0], target_np)]


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

        # --- Build per-constraint (constraint, scheduler) pairs ---
        constraint_schedulers = build_constraint_schedulers(prompt_cfg, args)
        terminal_targets      = extract_terminal_targets(prompt_cfg, args)

        steerer = FlowSteerer(
            pipeline=pipeline,
            decoder=decoder,
            constraint_schedulers=constraint_schedulers,
            steps=args.steps,
            smooth_kernel=args.smooth_kernel,
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
        b_c = compute_constraint_metrics(baseline_joints, terminal_targets=terminal_targets)
        s_c = compute_constraint_metrics(steered_joints,  terminal_targets=terminal_targets)
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
                        choices=["foot_contact", "terminal"])
    parser.add_argument("--terminal_x",  type=float, default=2.0)
    parser.add_argument("--terminal_z",  type=float, default=0.0)
    parser.add_argument("--alpha_max",      type=float, default=80.0,
                        help="Global alpha for cosine/constant schedulers")
    parser.add_argument("--alpha_terminal", type=float, default=80.0,
                        help="Alpha for terminal/waypoint constraints (staged scheduler)")
    parser.add_argument("--alpha_contact",  type=float, default=80.0,
                        help="Alpha for foot contact constraint.")
    parser.add_argument("--steps",        type=int,   default=50)
    parser.add_argument("--smooth_kernel", type=int,  default=5,
                        help="Temporal smoothing window for steering vector (odd int). "
                             "Suppresses high-frequency jerk. Set to 1 to disable.")
    parser.add_argument("--scheduler",   default="staged",
                        choices=["constant", "cosine", "staged"],
                        help="Legacy arg retained for cosine/constant single-alpha runs. "
                             "Per-constraint steering ignores this.")
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
