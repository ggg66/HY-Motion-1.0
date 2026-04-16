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


# ── Expanded prompt pool (~60 prompts) ───────────────────────────────────────
# Larger pool → better chance of finding visually compelling top-3 for paper.
# (prompt_text, duration_s, t_norm, joint_mask_key)
_ALL_PROMPTS = [
    # ── Locomotion ────────────────────────────────────────────────────────────
    ("a person walks forward and stops",                                    3.0, 0.5,  "upper_body"),
    ("a person walks forward.",                                             4.0, 0.5,  "upper_body"),
    ("a person runs forward.",                                              4.0, 0.4,  "upper_body"),
    ("a person sprints forward.",                                           4.0, 0.5,  "arms"),
    ("a person skips forward.",                                             4.0, 0.5,  "all"),
    ("a person marches in place, swinging their arms.",                     4.0, 0.4,  "arms"),
    ("a person walks forward, moving arms and legs while looking left and right.", 6.0, 0.5, "arms"),
    ("a person jogs in place.",                                             4.0, 0.4,  "arms"),
    ("a person walks backward slowly.",                                     4.0, 0.5,  "upper_body"),
    ("a person shuffles sideways to the left.",                             3.0, 0.5,  "arms"),
    ("a person crawls forward on all fours.",                               4.0, 0.4,  "all"),
    ("a person tiptoes forward quietly.",                                   4.0, 0.5,  "upper_body"),

    # ── Jumps & acrobatics ───────────────────────────────────────────────────
    ("a person jumps upward with both legs together.",                      3.0, 0.5,  "all"),
    ("a person jumps up.",                                                  3.0, 0.45, "all"),
    ("a person does a star jump, spreading arms and legs wide.",            3.0, 0.45, "all"),
    ("a person leaps forward.",                                             3.0, 0.45, "all"),
    ("a person does a standing long jump.",                                 3.0, 0.4,  "all"),
    ("a person hops on one foot.",                                          3.0, 0.5,  "upper_body"),

    # ── Martial arts / combat ────────────────────────────────────────────────
    ("a person performs a taekwondo kick, extending their leg.",            3.0, 0.5,  "all"),
    ("a person performs a side kick.",                                      3.0, 0.5,  "all"),
    ("a person does a front kick followed by a punch.",                     4.0, 0.35, "all"),
    ("a person throws a punch with their right arm.",                       3.0, 0.4,  "arms"),
    ("a person does a karate chop with their right hand.",                  3.0, 0.4,  "arms"),
    ("a person performs a spinning back kick.",                             3.0, 0.45, "all"),
    ("a person does a boxing combination, throwing jabs and a cross.",      4.0, 0.4,  "arms"),
    ("a person assumes a fighting stance, fists raised.",                   3.0, 0.5,  "arms"),
    ("a person performs a roundhouse kick.",                                3.0, 0.45, "all"),
    ("a person does a judo throw, grabbing and tossing an opponent.",       4.0, 0.35, "upper_body"),

    # ── Sports ───────────────────────────────────────────────────────────────
    ("a person shoots a basketball with both hands.",                       3.0, 0.4,  "arms"),
    ("a person swings a tennis racket overhead.",                           3.0, 0.4,  "arms"),
    ("a person throws a ball overhand.",                                    3.0, 0.35, "arms"),
    ("a person swings a baseball bat.",                                     3.0, 0.4,  "arms"),
    ("a person does a volleyball spike, jumping and hitting overhead.",     3.0, 0.4,  "all"),
    ("a person performs a swimming freestyle stroke.",                      4.0, 0.4,  "arms"),
    ("a person does a soccer kick.",                                        3.0, 0.45, "all"),
    ("a person performs a golf swing.",                                     3.0, 0.45, "arms"),

    # ── Dance ────────────────────────────────────────────────────────────────
    ("a person dances.",                                                    4.0, 0.3,  "upper_body"),
    ("a person dances jazz, jumping rhythmically.",                         5.0, 0.5,  "upper_body"),
    ("a person does a hip-hop dance.",                                      4.0, 0.4,  "upper_body"),
    ("a person does a salsa step.",                                         4.0, 0.5,  "upper_body"),
    ("a person performs a breakdance toprock.",                             4.0, 0.5,  "arms"),
    ("a person dances flamenco, stamping their feet.",                      5.0, 0.4,  "upper_body"),
    ("a person performs a ballet arabesque.",                               3.0, 0.5,  "all"),
    ("a person does the robot dance.",                                      4.0, 0.4,  "arms"),
    ("a person waves their arms above their head while dancing.",           4.0, 0.4,  "arms"),
    ("a person does a contemporary dance, sweeping arms wide.",             5.0, 0.4,  "arms"),
    ("a person performs a waltz, stepping and turning.",                    5.0, 0.4,  "upper_body"),

    # ── Slow / controlled movements ──────────────────────────────────────────
    ("a person practices tai chi, performing slow circular movements.",     5.0, 0.3,  "upper_body"),
    ("a person does a yoga warrior pose, arms extended.",                   4.0, 0.5,  "arms"),
    ("a person stretches both arms overhead.",                              3.0, 0.5,  "arms"),
    ("a person does a T-pose, arms held horizontally.",                     3.0, 0.5,  "arms"),
    ("a person reaches up with one arm to grab something.",                 3.0, 0.5,  "arms"),
    ("a person does a side stretch, leaning and raising one arm.",          4.0, 0.5,  "arms"),
    ("a person crosses their arms over their chest.",                       3.0, 0.5,  "arms"),

    # ── Object interaction / expressive ──────────────────────────────────────
    ("a person lifts a long gun, then walks forward.",                      5.0, 0.3,  "upper_body"),
    ("a person waves goodbye with their right hand.",                       3.0, 0.5,  "arms"),
    ("a person claps their hands.",                                         3.0, 0.4,  "arms"),
    ("a person raises both hands in celebration.",                          3.0, 0.45, "arms"),
    ("a person shrugs their shoulders.",                                    3.0, 0.5,  "upper_body"),
    ("a person bows deeply.",                                               3.0, 0.5,  "upper_body"),
    ("a person points forward with their right arm.",                       3.0, 0.5,  "arms"),
    ("a person mimics playing a drum kit, striking with both arms.",        4.0, 0.4,  "arms"),
    ("a person performs a cartwheel.",                                      3.0, 0.45, "all"),
]

# ── Default subset for each figure ───────────────────────────────────────────
# Override with --fig1_prompts / --fig2_prompts on the command line.
_PROMPTS_MAIN = [
    ("a person walks forward and stops",                            3.0, 0.5,  "upper_body"),
    ("a person does a hip-hop dance.",                              4.0, 0.4,  "upper_body"),
    ("a person performs a taekwondo kick, extending their leg.",    3.0, 0.5,  "all"),
]

_PROMPTS_ABLATION = [
    ("a person walks forward and stops",                            3.0, 0.5,  "upper_body"),
    ("a person dances jazz, jumping rhythmically.",                 5.0, 0.5,  "upper_body"),
]

_PROMPTS_MC = [
    ("a person walks forward and stops",                            3.0, 0.5,  "upper_body"),
    ("a person marches in place, swinging their arms.",             4.0, 0.4,  "arms"),
]

def _prompt_lookup(name: str):
    """Return (prompt, dur, t_norm, mask) by matching prompt text prefix."""
    name_l = name.lower()
    for row in _ALL_PROMPTS:
        if row[0].lower().startswith(name_l) or name_l in row[0].lower():
            return row
    raise ValueError(f"Prompt not found: {name!r}")

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

        # canonical target pose (22,3) for ghost overlay at keyframe column
        target_pose_np = canonicalize_frame_np(tgt[kf])   # (22,3)

        row_specs: List[RowSpec] = [
            ("Target\n(ref)", tgt,     "target", t_norm, None),
            ("Baseline",      base,    "muted",  t_norm, target_pose_np),
            ("Ours",          steered, "normal", t_norm, target_pose_np),
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
            ("w/o latent\nmask", no_mask, "muted",  t_norm, None),
            ("Full\nmethod",     full,    "normal", t_norm, None),
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
            ("Pose only",   pose_only, "muted",  t_norm, None),
            ("Pose + Foot", pose_foot, "normal", t_norm, None),
        ]
        prompts_data.append((prompt, row_specs))

    out_path = os.path.join(output_dir, "fig3_multiconstraint.png")
    save_multi_prompt_figure(
        prompts_data, out_path,
        n_frames=7, dpi=dpi,
    )


# ── Cache-all helper ─────────────────────────────────────────────────────────

def generate_all_cache(
    pipeline, decoder, cache_dir: str, force: bool = False,
) -> None:
    """
    Pre-generate and cache target + baseline + steered for every prompt in
    _ALL_PROMPTS.  No figures are produced; use pick_best_cases.py afterwards
    to rank the prompts, then rerun with --fig1_prompts to make figures.
    """
    print(f"\nGenerating cache for all {len(_ALL_PROMPTS)} prompts …")
    results = []
    for i, (prompt, dur, t_norm, mask_key) in enumerate(_ALL_PROMPTS):
        jm   = _JOINT_MASK_MAP[mask_key]
        pfx  = f"p_{prompt[:30]}"
        print(f"\n[{i+1}/{len(_ALL_PROMPTS)}] {prompt[:60]}")

        tgt  = _load_or_generate(
            cache_dir, f"{pfx}_tgt42",
            lambda p=prompt, d=dur: _generate(pipeline, p, 42, d),
            force=force,
        )
        base = _load_or_generate(
            cache_dir, f"{pfx}_base43",
            lambda p=prompt, d=dur: _generate(pipeline, p, 43, d),
            force=force,
        )

        T  = tgt.shape[0]
        kf = int(round(t_norm * (T - 1)))

        def _steer_fn(p=prompt, d=dur, tn=t_norm, jm_=jm, tgt_=tgt):
            steerer = _build_pose_steerer(
                pipeline, decoder, p, d, tn, jm_, tgt_,
                alpha=6.0, apply_latent_mask=True, use_hier=True,
            )
            out = steerer.generate(
                text=p, seed_input=[43],
                duration_slider=d, cfg_scale=5.0,
            )
            return pipeline_output_to_world_joints(out)[0]

        steered = _load_or_generate(
            cache_dir, f"{pfx}_steer43_full",
            _steer_fn, force=force,
        )

        # Quick inline PKE for ranking printout
        from eval.metrics import canonicalize_frame_np as _canon
        UPPER = [9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        Q  = _canon(tgt[kf])
        eb = float(np.linalg.norm((_canon(base[kf])    - Q)[UPPER], axis=-1).mean())
        es = float(np.linalg.norm((_canon(steered[kf]) - Q)[UPPER], axis=-1).mean())
        imp = (eb - es) / (eb + 1e-8) * 100
        abs_impr = eb - es
        score = abs_impr * max(imp / 100.0, 0.0)
        results.append(dict(prompt=prompt, dur=dur, t_norm=t_norm, mask=mask_key,
                            eb=eb, es=es, imp=imp, abs_impr=abs_impr, score=score))
        print(f"  PKE  base={eb:.4f}m  steer={es:.4f}m  impr={imp:+.1f}%  score={score:.4f}")

    # Sort and print ranking
    results.sort(key=lambda r: r["score"], reverse=True)
    print("\n" + "="*90)
    print("  RANKING — all prompts by visual score (abs_impr × rel_impr)")
    print("="*90)
    print(f"  {'#':<3} {'prompt':<52} {'e_base':>7} {'e_steer':>7} {'abs':>8} {'rel':>8}  score")
    print(f"  {'-'*3} {'-'*52} {'-'*7} {'-'*7} {'-'*8} {'-'*8}  -----")
    for i, r in enumerate(results):
        star = " ★" if i < 3 else ""
        print(f"  {i+1:<3} {r['prompt'][:52]:<52} {r['eb']:>7.4f} {r['es']:>7.4f} "
              f"{r['abs_impr']:>+7.4f}m {r['imp']:>+7.1f}%  {r['score']:.4f}{star}")
    print()
    print("  Suggested --fig1_prompts value:")
    top3 = [r["prompt"][:30] for r in results[:3]]
    print(f"    --fig1_prompts \"{','.join(top3)}\"")
    print()

    # Save ranking JSON next to cache
    rank_path = os.path.join(os.path.dirname(cache_dir), "prompt_ranking.json")
    with open(rank_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Ranking saved to: {rank_path}")


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
    parser.add_argument("--generate_cache_only", action="store_true",
                        help="Cache all 20 prompts then print ranking; no figures produced")
    parser.add_argument("--fig1_prompts", type=str, default="",
                        help="Comma-separated prompt name prefixes for fig1 (overrides _PROMPTS_MAIN)")
    parser.add_argument("--gpu_id",      type=int, default=0)
    args = parser.parse_args()

    dpi = 100 if args.draft else 200

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

    # ── Cache-only mode ────────────────────────────────────────────────────────
    if args.generate_cache_only:
        generate_all_cache(pipeline, decoder, args.cache_dir, force=args.force)
        print(f"\nDone.  Total: {(time.time()-t0)/60:.1f} min")
        return

    # ── Override fig1 prompts if requested ────────────────────────────────────
    global _PROMPTS_MAIN
    if args.fig1_prompts:
        names = [n.strip() for n in args.fig1_prompts.split(",") if n.strip()]
        _PROMPTS_MAIN = [_prompt_lookup(n) for n in names]
        print(f"fig1 prompts overridden ({len(_PROMPTS_MAIN)} prompts):")
        for row in _PROMPTS_MAIN:
            print(f"  • {row[0]}")

    any_specific = args.fig1_only or args.fig2_only or args.fig3_only
    do1 = args.fig1_only or not any_specific
    do2 = args.fig2_only or not any_specific
    do3 = args.fig3_only or not any_specific

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
