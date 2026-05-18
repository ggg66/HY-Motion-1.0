"""Create static paper figures for temporal attribute-edit experiments."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from eval.visualize import BONES, BONE_COLOURS


_CASE_LABELS = {
    "walk_arm_quality": "Walk arm",
    "march_arms_quality": "March arms",
    "dance_arm_visible": "Dance arm",
    "kick_foot_quality": "Kick foot",
    "exercise_arms_visible": "Exercise arms",
    "walk_smooth_sweep": "Walk arm",
    "kick_smooth_sweep": "Kick foot",
}

_QUAL_CASES = [
    ("Dance arm +25 cm", "output/paper_dance_refine", [0.30, 0.50, 0.70], [19, 21]),
    ("Exercise arms +25 cm", "output/paper_exercise_refine", [0.25, 0.50, 0.70], [18, 19, 20, 21]),
    ("Walk arm +10 cm", "output/paper_walk_refine", [0.30, 0.50, 0.70], [19, 21]),
    ("Kick foot +10 cm", "output/paper_kick_refine", [0.35, 0.52, 0.70], [8, 11]),
]


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(row: Dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def _case_label(case_id: str) -> str:
    return _CASE_LABELS.get(case_id, case_id.replace("_", " "))


def _save_method_comparison(selected_csv: str, output_dir: str) -> None:
    rows = _read_csv(selected_csv)
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["case_id"], row["method"])].append(row)

    case_ids = []
    for row in rows:
        if row["case_id"] not in case_ids:
            case_ids.append(row["case_id"])

    x = np.arange(len(case_ids))
    width = 0.36
    steer = [mean(_float(r, "achievement_pct") for r in grouped[(cid, "steer")]) for cid in case_ids]
    refine = [mean(_float(r, "achievement_pct") for r in grouped[(cid, "refine")]) for cid in case_ids]
    steer_pass = [
        100.0 * sum(str(r.get("meets_budget", "")).lower() == "true" for r in grouped[(cid, "steer")])
        / max(len(grouped[(cid, "steer")]), 1)
        for cid in case_ids
    ]
    refine_pass = [
        100.0 * sum(str(r.get("meets_budget", "")).lower() == "true" for r in grouped[(cid, "refine")])
        / max(len(grouped[(cid, "refine")]), 1)
        for cid in case_ids
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), dpi=220)
    ax = axes[0]
    ax.bar(x - width / 2, steer, width, label="Sampling steer", color="#9CA3AF")
    ax.bar(x + width / 2, refine, width, label="Latent refine", color="#2563EB")
    ax.axhline(75, color="#111827", lw=1.0, ls="--", label="Target threshold")
    ax.set_ylabel("Target achievement (%)")
    ax.set_ylim(-10, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([_case_label(c) for c in case_ids], rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.bar(x - width / 2, steer_pass, width, color="#9CA3AF")
    ax.bar(x + width / 2, refine_pass, width, color="#2563EB")
    ax.set_ylabel("Budget pass rate (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([_case_label(c) for c in case_ids], rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Sampling-time steering vs. budget-aware latent refinement", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "fig_attribute_method_comparison.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _save_tradeoff(walk_csv: str, kick_csv: str, output_dir: str) -> None:
    rows = []
    for source, path in [("Walk arm", walk_csv), ("Kick foot", kick_csv)]:
        for row in _read_csv(path):
            row = dict(row)
            row["source"] = source
            rows.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), dpi=220)
    colors = {"Walk arm": "#2563EB", "Kick foot": "#F97316"}
    for source in ["Walk arm", "Kick foot"]:
        group = sorted([r for r in rows if r["source"] == source], key=lambda r: _float(r, "delta_y"))
        xs = [_float(r, "delta_y") for r in group]
        ach = [_float(r, "achievement_pct") for r in group]
        jerk = [_float(r, "jerk_ratio") for r in group]
        axes[0].plot(xs, ach, marker="o", lw=2.2, label=source, color=colors[source])
        axes[1].plot(xs, jerk, marker="o", lw=2.2, label=source, color=colors[source])

    axes[0].axhline(75, color="#111827", lw=1.0, ls="--")
    axes[0].set_ylabel("Target achievement (%)")
    axes[0].set_xlabel("Requested vertical offset (m)")
    axes[0].set_ylim(70, 105)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(2.0, color="#111827", lw=1.0, ls="--", label="Jerk budget")
    axes[1].set_ylabel("Jerk ratio")
    axes[1].set_xlabel("Requested vertical offset (m)")
    axes[1].set_ylim(1.0, 3.8)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle("Edit-strength vs. motion-quality tradeoff", fontsize=12)
    fig.tight_layout()
    out = os.path.join(output_dir, "fig_attribute_tradeoff.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def _axis_limits(motions: Iterable[np.ndarray]) -> Tuple[np.ndarray, float]:
    pts = np.concatenate([m.reshape(-1, 3) for m in motions], axis=0)
    center = pts.mean(axis=0)
    ranges = pts.max(axis=0) - pts.min(axis=0)
    radius = float(max(ranges.max() * 0.55, 0.8))
    return center, radius


def _draw_overlay(ax, baseline: np.ndarray, edited: np.ndarray, target_joints: List[int]) -> None:
    ax.set_facecolor("white")
    for child, parent in BONES:
        b0, b1 = baseline[parent], baseline[child]
        e0, e1 = edited[parent], edited[child]
        ax.plot([b0[0], b1[0]], [b0[2], b1[2]], [b0[1], b1[1]], color="#9CA3AF", lw=1.4, alpha=0.55)
        color = "#2563EB" if child in target_joints or parent in target_joints else "#111827"
        ax.plot([e0[0], e1[0]], [e0[2], e1[2]], [e0[1], e1[1]], color=color, lw=2.1, alpha=0.95)
    tj = edited[target_joints]
    ax.scatter(tj[:, 0], tj[:, 2], tj[:, 1], s=16, c="#F97316", depthshade=False)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=14, azim=-62)


def _save_qualitative(output_dir: str) -> None:
    fig = plt.figure(figsize=(9.5, 8.2), dpi=220)
    row_centers = []
    for row_idx, (title, rel_dir, frame_fracs, target_joints) in enumerate(_QUAL_CASES):
        baseline = np.load(os.path.join(_repo_root, rel_dir, "baseline_seed0.npy"))
        edited = np.load(os.path.join(_repo_root, rel_dir, "steered_seed0.npy"))
        center, radius = _axis_limits([baseline, edited])
        row_centers.append(0.815 - row_idx * 0.205)
        for col_idx, frac in enumerate(frame_fracs):
            ax = fig.add_subplot(len(_QUAL_CASES), len(frame_fracs), row_idx * len(frame_fracs) + col_idx + 1, projection="3d")
            t = int(round(frac * (min(len(baseline), len(edited)) - 1)))
            _draw_overlay(ax, baseline[t], edited[t], target_joints)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[2] - radius, center[2] + radius)
            ax.set_zlim(max(0.0, center[1] - radius), center[1] + radius)
            if row_idx == 0:
                ax.set_title(f"{int(frac * 100)}% time", fontsize=9)

    for y, (title, _, _, _) in zip(row_centers, _QUAL_CASES):
        fig.text(0.02, y, title, va="center", ha="left", fontsize=9, rotation=90)

    handles = [
        plt.Line2D([0], [0], color="#9CA3AF", lw=2, label="Baseline"),
        plt.Line2D([0], [0], color="#111827", lw=2, label="Edited"),
        plt.Line2D([0], [0], color="#2563EB", lw=2, label="Edited target limb"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#F97316", markersize=6, label="Target joint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Temporal attribute edits: baseline (gray) vs. refined edit", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.08, wspace=0.02, hspace=0.02)
    out = os.path.join(output_dir, "fig_attribute_qualitative_snapshots.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Make static attribute-edit paper figures")
    parser.add_argument("--output_dir", default="output/paper_static_figures")
    parser.add_argument("--compare_csv", default="output/attribute_paper_protocol_compare_3seed/selected.csv")
    parser.add_argument("--walk_tradeoff_csv", default="output/walk_smooth_sweep/selected.csv")
    parser.add_argument("--kick_tradeoff_csv", default="output/kick_smooth_sweep/selected.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _save_method_comparison(args.compare_csv, args.output_dir)
    _save_tradeoff(args.walk_tradeoff_csv, args.kick_tradeoff_csv, args.output_dir)
    _save_qualitative(args.output_dir)


if __name__ == "__main__":
    main()
