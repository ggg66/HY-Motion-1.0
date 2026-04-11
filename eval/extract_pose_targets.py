"""
Extract canonical keyframe poses from existing baseline motion files.

For each prompt in a pose-eval JSON, loads the corresponding baseline
joints from an eval directory, canonicalizes the pose at a specified
normalised time, and saves the result as a .npy file.

These .npy files are then referenced by pose_eval.json as PoseConstraint
targets, enabling a clean self-consistency evaluation:

    baseline → extract canonical pose @ t_norm
             → re-run with PoseConstraint targeting that pose
             → measure: does steering recover the pose without quality loss?

Usage:
    python eval/extract_pose_targets.py \
        --eval_dir output/eval_full \
        --prompt_file eval/prompts/pose_eval_raw.json \
        --output_dir eval/pose_targets \
        --out_prompt_file eval/prompts/pose_eval.json

Output
------
  eval/pose_targets/{idx:04d}_t{t_pct:02d}.npy   shape (22, 3) canonical joints
  eval/prompts/pose_eval.json                     prompts with target_file fields filled in
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Numpy canonicalization (mirrors PoseConstraint.canonicalize)
# ---------------------------------------------------------------------------

_LEFT_HIP  = 1
_RIGHT_HIP = 2
_ROOT      = 0


def _canonicalize_frame_np(joints_frame: np.ndarray) -> np.ndarray:
    """
    Root-centred + yaw-aligned canonicalization in numpy.

    joints_frame: (22, 3)  world-space joint positions
    returns:      (22, 3)  canonical-space joint positions

    Mirrors PoseConstraint.canonicalize() so targets and predictions are
    in the same space.
    """
    # 1. Root-centre
    root    = joints_frame[_ROOT]                  # (3,)
    centred = joints_frame - root                  # (22, 3)

    # 2. Compute facing direction from hip vector
    hip_vec = joints_frame[_RIGHT_HIP] - joints_frame[_LEFT_HIP]  # (3,)
    hx, hz  = hip_vec[0], hip_vec[2]
    norm    = math.sqrt(hx**2 + hz**2) + 1e-6
    fx, fz  = -hz / norm, hx / norm               # forward dir in XZ

    # 3. Yaw: rotate so forward → +Z
    yaw     = math.atan2(fx, fz)
    cos_y   = math.cos(yaw)
    sin_y   = math.sin(yaw)

    # 4. Apply Y-axis rotation
    x = centred[:, 0]
    y = centred[:, 1]
    z = centred[:, 2]
    x_rot = x * cos_y - z * sin_y
    z_rot = x * sin_y + z * cos_y

    return np.stack([x_rot, y, z_rot], axis=-1)   # (22, 3)


def extract_pose(
    joints: np.ndarray,          # (B, T, 22, 3)  world-space joints
    t_norm: float,               # target normalised time in [0, 1]
    seed_idx: int = 0,           # which seed (batch index) to use
) -> np.ndarray:
    """Extract and canonicalize a single frame's pose."""
    T     = joints.shape[1]
    frame = int(round(t_norm * (T - 1)))
    frame = max(0, min(frame, T - 1))
    return _canonicalize_frame_np(joints[seed_idx, frame])   # (22, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

JOINT_MASK_NAMES = {
    "all":        None,
    "upper_body": [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
    "lower_body": [1, 2, 4, 5, 7, 8, 10, 11],
    "arms":       [16, 17, 18, 19, 20, 21],
    "legs":       [1, 2, 4, 5, 7, 8, 10, 11],
}


def main():
    parser = argparse.ArgumentParser(
        description="Extract canonical pose targets from baseline motion files"
    )
    parser.add_argument("--eval_dir",       required=True,
                        help="Directory containing XXXX_baseline.npy files")
    parser.add_argument("--prompt_file",    required=True,
                        help="Input pose prompt JSON (pose_eval_raw.json)")
    parser.add_argument("--output_dir",     default="eval/pose_targets",
                        help="Where to save extracted .npy pose files")
    parser.add_argument("--out_prompt_file", default=None,
                        help="Output prompt JSON with target_file fields filled. "
                             "Default: prompt_file with '_raw' removed.")
    parser.add_argument("--seed_idx",       type=int, default=0,
                        help="Which seed (batch index) to use when extracting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.prompt_file) as f:
        prompts = json.load(f)

    if args.out_prompt_file is None:
        base = os.path.basename(args.prompt_file).replace("_raw", "")
        args.out_prompt_file = os.path.join(os.path.dirname(args.prompt_file), base)

    updated = []
    for entry in prompts:
        idx      = entry["idx"]
        npy_path = os.path.join(args.eval_dir, f"{idx:04d}_baseline.npy")

        if not os.path.exists(npy_path):
            print(f"[SKIP] #{idx}: {npy_path} not found")
            updated.append(entry)
            continue

        joints = np.load(npy_path)           # (B, T, 22, 3)
        T      = joints.shape[1]

        pose_keyframes = []
        for kf in entry.get("pose_keyframes_raw", []):
            t_norm    = float(kf["t_norm"])
            jmask_key = kf.get("joint_mask", "upper_body")

            # Extract and save canonical pose
            frame     = int(round(t_norm * (T - 1)))
            tag       = f"{idx:04d}_t{int(t_norm*100):02d}"
            out_path  = os.path.join(args.output_dir, f"{tag}.npy")
            pose      = extract_pose(joints, t_norm, seed_idx=args.seed_idx)
            np.save(out_path, pose)

            print(f"  #{idx} t={t_norm:.2f} (frame {frame}/{T-1}) "
                  f"mask={jmask_key} → {out_path}")

            pose_keyframes.append({
                "t_norm":      t_norm,
                "target_file": out_path,
                "joint_mask":  jmask_key,
                "sigma_frac":  kf.get("sigma_frac", 0.04),
            })

        new_entry = {k: v for k, v in entry.items()
                     if k != "pose_keyframes_raw"}
        new_entry["pose_keyframes"] = pose_keyframes
        new_entry["constraint"]     = ["pose"]
        updated.append(new_entry)
        print(f"[OK] #{idx}: {entry['prompt'][:50]}")

    with open(args.out_prompt_file, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"\nWrote {len(updated)} prompts → {args.out_prompt_file}")


if __name__ == "__main__":
    main()
