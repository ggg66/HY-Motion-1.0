"""
Generate paper-quality figures for FlowSteer-Motion.

Produces three PNG figures:
    fig1_pose_comparison.png   Main comparison: Target | Baseline | Ours (×3 prompts)
    fig2_ablation.png          Ablation: no-mask vs full method (×2 prompts)
    fig3_multiconstraint.png   Multi-constraint: pose-only vs pose+foot (×2 prompts)

For each prompt the motions are generated once and cached as .npy arrays.
Re-running the script reads from cache (skips generation) unless --force is set.

Usage
-----
    python eval/make_paper_figures.py --model_path ckpts/tencent/HY-Motion-1.0

Optional flags
--------------
    --output_dir    Where to write figures (default: output/paper_figures)
    --cache_dir     Where to cache .npy arrays (default: output/paper_figures/cache)
    --force         Discard cache and regenerate all motions
    --draft         Low-DPI (100) for quick preview
    --fig1_only / --fig2_only / --fig3_only  Generate only one figure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
    PoseConstraint,
    StagedScheduler,
    UPPER_BODY_JOINTS,
    WaypointConstraint,
)
from eval.metrics import canonicalize_frame_np, pipeline_output_to_world_joints
from eval.visualize_paper import (
    RowSpec,
    save_comparison_figure,
    save_multi_prompt_figure,
)

# ── Default timed windows (mirrors run_eval.py) ───────────────────────────────
_T_WAYPOINT = (0.15, 0.65)
_T_POSE     = (0.50, 0.88)
_T_FOOT     = (0.72, 1.00)


# ── Representative prompts ────────────────────────────────────────────────────
# (prompt_text, duration_s, t_norm, joint_mask_key)
_PROMPTS_MAIN = [
    ("a person walks forward and stops",                3.0, 0.5,  "upper_body"),
    ("a person does a hip-hop dance.",                  4.0, 0.4,  "upper_body"),
    ("a person performs a taekwondo kick, extending their leg.", 3.0, 0.5, "all"),
]

_PROMPTS_ABLATION = [
    ("a person walks forward and stops",    3.0, 0.5, "upper_body"),
    ("a person dances jazz, jumping rhythmically.", 5.0, 0.5, "upper_body"),
]

_PROMPTS_MC = [
    ("a person walks forward and stops",    3.0, 0.5, "upper_body"),
    ("a person marches in place, swinging their arms.", 4.0, 0.4, "arms"),
]

_JOINT_MASK_MAP = {
    "all":        None,
    "upper_body": UPPER_BODY_JOINTS,
    "arms":       [16, 17, 18, 19, 20, 21],   # shoulders, elbows, wrists
}


# ── Pipeline helpers ──────────────────────────────────────────────────────────

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


def _generate(pipeline, text, seed, duration, cfg_scale=5.0) -> np.ndarray:
    """Generate one motion and return (T, 22, 3) world-space joints."""
    with torch.no_grad():
        out = pipeline.generate(
            text=text, seed_input=[seed],
            duration_slider=duration, cfg_scale=cfg_scale,
        )
    return pipeline_output_to_world_joints(out)[0]   # (T, 22, 3)


def _build_pose_steerer(
    pipeline, decoder, prompt, duration, t_norm, joint_mask,
    target_joints_T22_3: np.ndarray, alpha: float = 6.0,
    apply_latent_mask: bool = True, use_hier: bool = True,
    cfg_scale: float = 5.0, steps: int = 50,
) -> FlowSteerer:
    """Build a FlowSteerer for pose-only constraint."""
    T = target_joints_T22_3.shape[0]
    kf_idx = int(round(t_norm * (T - 1)))
    target_pose = canonicalize_frame_np(target_joints_T22_3[kf_idx])   # (22, 3)
    target_pose_t = torch.from_numpy(target_pose).float()

    pc = PoseConstraint(
        keyframes=[(t_norm, target_pose_t)],
        joint_mask=joint_mask,
        sigma_frac=0.04,
        use_hierarchical=use_hier,
    )
    timed = [(pc, 1.0, _T_POSE[0], _T_POSE[1])]
    scheduler = StagedScheduler.cosine(alpha_max=alpha)
    return FlowSteerer(
        pipeline=pipeline, decoder=decoder,
        timed_constraints=timed, scheduler=scheduler,
        steps=steps, smooth_kernel=7,
        soft_norm_tau=0.1, max_steer_ratio=0.3, ema_momentum=0.7,
        apply_latent_mask=apply_latent_mask,
        latent_mask_transl=0.1, latent_mask_root_rot=0.3,
        use_temporal_mask=True,
    )


def _build_pose_foot_steerer(
    pipeline, decoder, t_norm, joint_mask,
    target_pose_t: torch.Tensor, alpha: float = 6.0,
    steps: int = 50,
) -> FlowSteerer:
    """Build a FlowSteerer for pose + foot contact."""
    pc = PoseConstraint(
        keyframes=[(t_norm, target_pose_t)],
        joint_mask=joint_mask,
        sigma_frac=0.04,
        use_hierarchical=True,
    )
    fc = FootContactConstraint(height_thresh=0.05, vel_thresh=0.02, detach_mask=True)
    timed = [
        (pc, 1.0, _T_POSE[0],  _T_POSE[1]),
        (fc, 1.0, _T_FOOT[0],  _T_FOOT[1]),
    ]
    scheduler = StagedScheduler.cosine(alpha_max=alpha)
    return FlowSteerer(
        pipeline=pipeline, decoder=decoder,
        timed_constraints=timed, scheduler=scheduler,
        steps=steps, smooth_kernel=7,
        soft_norm_tau=0.1, max_steer_ratio=0.3, ema_momentum=0.7,
        apply_latent_mask=True,
        latent_mask_transl=0.1, latent_mask_root_rot=0.3,
        use_temporal_mask=True,
    )


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, key: str) -> str:
    safe_key = key.replace(" ", "_").replace(".", "").replace(",", "")[:60]
    return os.path.join(cache_dir, f"{safe_key}.npy")


def _load_or_generate(
    cache_dir: str,
    key: str,
    generate_fn,
    force: bool = False,
) -> np.ndarray:
    path = _cache_path(cache_dir, key)
    if not force and os.path.exists(path):
        arr = np.load(path)
        print(f"  [cache] {key[:60]}  shape={arr.shape}")
        return arr
    t0 = time.time()
    arr = generate_fn()
    os.makedirs(cache_dir, exist_ok=True)
    np.save(path, arr)
    print(f"  [gen]   {key[:60]}  shape={arr.shape}  ({time.time()-t0:.1f}s)")
    return arr


# ── Figure builders ───────────────────────────────────────────────────────────

def make_fig1_pose_comparison(
    pipeline, decoder, cache_dir, output_dir,
    dpi: int = 200, force: bool = False,
) -> None:
    """
    Figure 1: Pose constraint comparison.
    Rows: Target (seed 42) | Baseline (seed 43) | Ours (seed 43, full method).
    3 prompt blocks stacked vertically.
    """
    prompts_data = []
    for prompt, dur, t_norm, mask_key in _PROMPTS_MAIN:
        jm = _JOINT_MASK_MAP[mask_key]
        pfx = f"p_{prompt[:30]}"

        # Target (seed 42)
        tgt = _load_or_generate(
            cache_dir, f"{pfx}_tgt42",
            lambda: _generate(pipeline, prompt, 42, dur),
            force=force,
        )

        # Baseline (seed 43)
        base = _load_or_generate(
            cache_dir, f"{pfx}_base43",
            lambda: _generate(pipeline, prompt, 43, dur),
            force=force,
        )

        # Build steerer using target (seed 42) pose
        T = tgt.shape[0]
        kf = int(round(t_norm * (T - 1)))
        target_pose_t = torch.from_numpy(
            canonicalize_frame_np(tgt[kf]).astype(np.float32)
        )

        def _steer(p=prompt, d=dur, tn=t_norm, jm_=jm, tpt=target_pose_t):
            steerer = _build_pose_steerer(
                pipeline, decoder, p, d, tn, jm_, tgt,
                alpha=6.0, apply_latent_mask=True, use_hier=True,
            )
            out = steerer.generate(
                text=p, seed_input=[43],
                duration_slider=d, cfg_scale=5.0,
            )
            return pipeline_output_to_world_joints(out)[0]

        steered = _load_or_generate(
            cache_dir, f"{pfx}_steer43_full",
            _steer,
            force=force,
        )

        row_specs: List[RowSpec] = [
            ("Target\n(ref)", tgt,     "target", t_norm),
            ("Baseline",      base,    "muted",  t_norm),
            ("Ours",          steered, "normal", t_norm),
        ]
        prompts_data.append((prompt, row_specs))

    out_path = os.path.join(output_dir, "fig1_pose_comparison.png")
    save_multi_prompt_figure(
        prompts_data, out_path,
        n_frames=7, dpi=dpi,
    )


def make_fig2_ablation(
    pipeline, decoder, cache_dir, output_dir,
    dpi: int = 200, force: bool = False,
) -> None:
    """
    Figure 2: Ablation.
    Rows: w/o latent mask | Full method.
    2 prompt blocks.
    """
    prompts_data = []
    for prompt, dur, t_norm, mask_key in _PROMPTS_ABLATION:
        jm = _JOINT_MASK_MAP[mask_key]
        pfx = f"p_{prompt[:30]}"

        tgt = _load_or_generate(
            cache_dir, f"{pfx}_tgt42",
            lambda: _generate(pipeline, prompt, 42, dur),
            force=force,
        )

        def _steer_no_mask(p=prompt, d=dur, tn=t_norm, jm_=jm, tgt_=tgt):
            steerer = _build_pose_steerer(
                pipeline, decoder, p, d, tn, jm_, tgt_,
                alpha=6.0, apply_latent_mask=False, use_hier=True,
            )
            out = steerer.generate(
                text=p, seed_input=[43],
                duration_slider=d, cfg_scale=5.0,
            )
            return pipeline_output_to_world_joints(out)[0]

        def _steer_full(p=prompt, d=dur, tn=t_norm, jm_=jm, tgt_=tgt):
            steerer = _build_pose_steerer(
                pipeline, decoder, p, d, tn, jm_, tgt_,
                alpha=6.0, apply_latent_mask=True, use_hier=True,
            )
            out = steerer.generate(
                text=p, seed_input=[43],
                duration_slider=d, cfg_scale=5.0,
            )
            return pipeline_output_to_world_joints(out)[0]

        no_mask = _load_or_generate(
            cache_dir, f"{pfx}_steer43_no_mask",
            _steer_no_mask, force=force,
        )
        full = _load_or_generate(
            cache_dir, f"{pfx}_steer43_full",
            _steer_full, force=force,
        )

        row_specs: List[RowSpec] = [
            ("w/o latent\nmask", no_mask, "muted",  t_norm),
            ("Full\nmethod",    full,    "normal", t_norm),
        ]
        prompts_data.append((prompt, row_specs))

    out_path = os.path.join(output_dir, "fig2_ablation.png")
    save_multi_prompt_figure(
        prompts_data, out_path,
        n_frames=7, dpi=dpi,
    )


def make_fig3_multiconstraint(
    pipeline, decoder, cache_dir, output_dir,
    dpi: int = 200, force: bool = False,
) -> None:
    """
    Figure 3: Multi-constraint (pose only vs pose + foot contact).
    2 prompt blocks.
    """
    prompts_data = []
    for prompt, dur, t_norm, mask_key in _PROMPTS_MC:
        jm = _JOINT_MASK_MAP[mask_key]
        pfx = f"p_{prompt[:30]}"

        tgt = _load_or_generate(
            cache_dir, f"{pfx}_tgt42",
            lambda: _generate(pipeline, prompt, 42, dur),
            force=force,
        )

        T = tgt.shape[0]
        kf = int(round(t_norm * (T - 1)))
        target_pose_t = torch.from_numpy(
            canonicalize_frame_np(tgt[kf]).astype(np.float32)
        )

        def _steer_pose_only(p=prompt, d=dur, tn=t_norm, jm_=jm, tgt_=tgt):
            steerer = _build_pose_steerer(
                pipeline, decoder, p, d, tn, jm_, tgt_,
                alpha=6.0, apply_latent_mask=True, use_hier=True,
            )
            out = steerer.generate(
                text=p, seed_input=[43],
                duration_slider=d, cfg_scale=5.0,
            )
            return pipeline_output_to_world_joints(out)[0]

        def _steer_pose_foot(p=prompt, d=dur, tn=t_norm, jm_=jm, tpt=target_pose_t):
            steerer = _build_pose_foot_steerer(
                pipeline, decoder, tn, jm_, tpt, alpha=6.0,
            )
            out = steerer.generate(
                text=p, seed_input=[43],
                duration_slider=d, cfg_scale=5.0,
            )
            return pipeline_output_to_world_joints(out)[0]

        pose_only = _load_or_generate(
            cache_dir, f"{pfx}_steer43_full",
            _steer_pose_only, force=force,
        )
        pose_foot = _load_or_generate(
            cache_dir, f"{pfx}_steer43_pose_foot",
            _steer_pose_foot, force=force,
        )

        row_specs: List[RowSpec] = [
            ("Pose only",     pose_only, "muted",  t_norm),
            ("Pose + Foot",   pose_foot, "normal", t_norm),
        ]
        prompts_data.append((prompt, row_specs))

    out_path = os.path.join(output_dir, "fig3_multiconstraint.png")
    save_multi_prompt_figure(
        prompts_data, out_path,
        n_frames=7, dpi=dpi,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--model_path",  required=True)
    parser.add_argument("--output_dir",  default="output/paper_figures")
    parser.add_argument("--cache_dir",   default="output/paper_figures/cache")
    parser.add_argument("--force",       action="store_true", help="Discard cache and regenerate all motions")
    parser.add_argument("--draft",       action="store_true", help="Low DPI (100) for quick preview")
    parser.add_argument("--fig1_only",   action="store_true")
    parser.add_argument("--fig2_only",   action="store_true")
    parser.add_argument("--fig3_only",   action="store_true")
    parser.add_argument("--gpu_id",      type=int, default=0)
    args = parser.parse_args()

    dpi = 100 if args.draft else 200
    any_specific = args.fig1_only or args.fig2_only or args.fig3_only
    do1 = args.fig1_only or not any_specific
    do2 = args.fig2_only or not any_specific
    do3 = args.fig3_only or not any_specific

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Loading pipeline from {args.model_path} ...")
    pipeline = load_pipeline(args.model_path, device)
    decoder  = MotionDecoder.from_stats_dir(
        stats_dir=os.path.join(_repo_root, "stats"),
        body_model_path=os.path.join(_repo_root, "scripts/gradio/static/assets/dump_wooden"),
    )

    t0 = time.time()

    if do1:
        print("\n── Figure 1: Pose comparison ────────────────────────────────")
        make_fig1_pose_comparison(
            pipeline, decoder, args.cache_dir, args.output_dir,
            dpi=dpi, force=args.force,
        )

    if do2:
        print("\n── Figure 2: Ablation ───────────────────────────────────────")
        make_fig2_ablation(
            pipeline, decoder, args.cache_dir, args.output_dir,
            dpi=dpi, force=args.force,
        )

    if do3:
        print("\n── Figure 3: Multi-constraint ───────────────────────────────")
        make_fig3_multiconstraint(
            pipeline, decoder, args.cache_dir, args.output_dir,
            dpi=dpi, force=args.force,
        )

    print(f"\nDone.  Total: {(time.time()-t0)/60:.1f} min")
    print(f"Figures written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
