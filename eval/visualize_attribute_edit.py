"""Generate comparison videos for attribute-edit experiments."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from eval.visualize import save_comparison_video, save_skeleton_video


def main():
    parser = argparse.ArgumentParser(description="Visualize baseline vs edited joints")
    parser.add_argument("--baseline", required=True, help="Path to baseline .npy")
    parser.add_argument("--edited", required=True, help="Path to edited .npy")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--label", default="attribute edit")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    baseline = np.load(args.baseline)
    edited = np.load(args.edited)

    save_comparison_video(
        baseline,
        edited,
        os.path.join(args.output_dir, "comparison_full.mp4"),
        fps=args.fps,
        prompt=args.prompt,
        constraint_label=args.label,
    )
    save_skeleton_video(
        edited,
        os.path.join(args.output_dir, "edited_only.mp4"),
        fps=args.fps,
        title=args.label,
    )


if __name__ == "__main__":
    main()
