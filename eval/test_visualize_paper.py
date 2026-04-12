"""
Smoke-test for visualize_paper.py using synthetic random joints.
Runs without the model — just checks that the layout renders correctly.

Usage:
    python eval/test_visualize_paper.py
"""
from __future__ import annotations
import os, sys
import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_eval_dir  = os.path.dirname(os.path.abspath(__file__))
# Import visualize_paper directly to avoid triggering eval/__init__.py (which needs torch)
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

import visualize_paper as vp
save_comparison_figure  = vp.save_comparison_figure
save_multi_prompt_figure = vp.save_multi_prompt_figure

RNG = np.random.default_rng(0)


def _random_walk(T=90, noise=0.01) -> np.ndarray:
    """Synthesize a plausible-ish skeleton walking sequence."""
    # SMPL-H T-pose (rough relative offsets)
    _TPOSE_22 = np.array([
        [ 0.00, 0.90, 0.00],   # 0  Pelvis
        [-0.09, 0.85, 0.00],   # 1  L_Hip
        [ 0.09, 0.85, 0.00],   # 2  R_Hip
        [ 0.00, 1.00, 0.00],   # 3  Spine1
        [-0.09, 0.50, 0.00],   # 4  L_Knee
        [ 0.09, 0.50, 0.00],   # 5  R_Knee
        [ 0.00, 1.10, 0.00],   # 6  Spine2
        [-0.09, 0.10, 0.00],   # 7  L_Ankle
        [ 0.09, 0.10, 0.00],   # 8  R_Ankle
        [ 0.00, 1.25, 0.00],   # 9  Spine3
        [-0.09, 0.02, 0.07],   # 10 L_Foot
        [ 0.09, 0.02, 0.07],   # 11 R_Foot
        [ 0.00, 1.40, 0.00],   # 12 Neck
        [-0.05, 1.35, 0.00],   # 13 L_Collar
        [ 0.05, 1.35, 0.00],   # 14 R_Collar
        [ 0.00, 1.55, 0.00],   # 15 Head
        [-0.20, 1.35, 0.00],   # 16 L_Shoulder
        [ 0.20, 1.35, 0.00],   # 17 R_Shoulder
        [-0.40, 1.15, 0.00],   # 18 L_Elbow
        [ 0.40, 1.15, 0.00],   # 19 R_Elbow
        [-0.55, 0.95, 0.00],   # 20 L_Wrist
        [ 0.55, 0.95, 0.00],   # 21 R_Wrist
    ], dtype=np.float32)

    frames = []
    for t in range(T):
        jitter = RNG.normal(0, noise, (22, 3)).astype(np.float32)
        frame = _TPOSE_22.copy() + jitter
        # Move forward (Z direction)
        frame[:, 2] += t * 0.03
        # Slight arm swing
        swing = np.sin(t * 0.3) * 0.08
        frame[20, 2] += swing   # L_Wrist
        frame[21, 2] -= swing   # R_Wrist
        frames.append(frame)
    return np.stack(frames, axis=0)   # (T, 22, 3)


def main():
    out_dir = "output/paper_figures_test"
    os.makedirs(out_dir, exist_ok=True)

    T = 90
    seq_tgt  = _random_walk(T, noise=0.005)
    seq_base = _random_walk(T, noise=0.015)
    seq_ours = _random_walk(T, noise=0.007)
    seq_nomask = _random_walk(T, noise=0.025)

    # ── Test 1: single-prompt comparison figure ───────────────────────────────
    print("Test 1: save_comparison_figure ...")
    row_specs = [
        ("Target\n(ref)", seq_tgt,    "target", 0.5, None),
        ("Baseline",      seq_base,   "muted",  0.5, None),
        ("Ours",          seq_ours,   "normal", 0.5, None),
    ]
    save_comparison_figure(
        row_specs,
        os.path.join(out_dir, "test_comparison.png"),
        n_frames=7, dpi=120,
        suptitle="a person walks forward and stops",
    )

    # ── Test 2: multi-prompt figure ───────────────────────────────────────────
    print("Test 2: save_multi_prompt_figure ...")
    prompts_data = [
        ("a person walks forward and stops", [
            ("Target\n(ref)", seq_tgt,  "target", 0.5),
            ("Baseline",      seq_base, "muted",  0.5),
            ("Ours",          seq_ours, "normal", 0.5),
        ]),
        ("a person does a hip-hop dance.", [
            ("Target\n(ref)", seq_tgt,    "target", 0.4, None),
            ("Baseline",      seq_base,   "muted",  0.4, None),
            ("Ours",          seq_nomask, "normal", 0.4, None),
        ]),
    ]
    save_multi_prompt_figure(
        prompts_data,
        os.path.join(out_dir, "test_multi_prompt.png"),
        n_frames=7, dpi=120,
    )

    # ── Test 3: ablation figure ───────────────────────────────────────────────
    print("Test 3: ablation layout ...")
    ablation_data = [
        ("a person walks forward and stops", [
            ("w/o latent\nmask", seq_nomask, "muted",  0.5, None),
            ("Full\nmethod",     seq_ours,   "normal", 0.5, None),
        ]),
    ]
    save_multi_prompt_figure(
        ablation_data,
        os.path.join(out_dir, "test_ablation.png"),
        n_frames=7, dpi=120,
    )

    print(f"\nAll tests passed.  Figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
