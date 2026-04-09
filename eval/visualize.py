"""
Skeleton visualization for FlowSteer-Motion.

Generates side-by-side MP4 videos comparing baseline vs steered motions.
Uses matplotlib FuncAnimation (requires ffmpeg for MP4 output).

Joint indices follow the WoodenMesh / SMPL-H 22-joint layout:
    0  Pelvis    7  L_Ankle   14 R_Collar  21 R_Wrist
    1  L_Hip     8  R_Ankle   15 Head
    2  R_Hip     9  Spine3    16 L_Shoulder
    3  Spine1   10  L_Foot    17 R_Shoulder
    4  L_Knee   11  R_Foot    18 L_Elbow
    5  R_Knee   12  Neck      19 R_Elbow
    6  Spine2   13  L_Collar  20 L_Wrist
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (registers 3d projection)
import numpy as np


# ---------------------------------------------------------------------------
# Skeleton definition
# ---------------------------------------------------------------------------

# Parent index for each joint (-1 = root)
SMPL22_PARENTS = [
    -1,   # 0  Pelvis
     0,   # 1  L_Hip
     0,   # 2  R_Hip
     0,   # 3  Spine1
     1,   # 4  L_Knee
     2,   # 5  R_Knee
     3,   # 6  Spine2
     4,   # 7  L_Ankle
     5,   # 8  R_Ankle
     6,   # 9  Spine3
     7,   # 10 L_Foot
     8,   # 11 R_Foot
     9,   # 12 Neck
     9,   # 13 L_Collar
     9,   # 14 R_Collar
    12,   # 15 Head
    13,   # 16 L_Shoulder
    14,   # 17 R_Shoulder
    16,   # 18 L_Elbow
    17,   # 19 R_Elbow
    18,   # 20 L_Wrist
    19,   # 21 R_Wrist
]

# Bone list derived from parent array (child_idx, parent_idx)
BONES = [(i, p) for i, p in enumerate(SMPL22_PARENTS) if p >= 0]

# Colour scheme: left = blue, right = red, spine = green
_LEFT   = [1, 4, 7, 10, 13, 16, 18, 20]   # L_* joints
_RIGHT  = [2, 5, 8, 11, 14, 17, 19, 21]   # R_* joints

def _bone_colour(child: int) -> str:
    if child in _LEFT:
        return "#4477CC"
    if child in _RIGHT:
        return "#CC4444"
    return "#44AA44"

BONE_COLOURS = [_bone_colour(c) for c, _ in BONES]

# Foot joints for contact overlay
FOOT_JOINTS = [7, 8, 10, 11]


# ---------------------------------------------------------------------------
# Single-panel skeleton draw
# ---------------------------------------------------------------------------

def _draw_skeleton(
    ax: "Axes3D",
    joints: np.ndarray,          # (22, 3)
    contact_mask: Optional[np.ndarray] = None,   # (4,) bool, foot contact
    alpha: float = 1.0,
    lw: float = 2.0,
) -> None:
    """Draw one skeleton frame into a 3D axes."""
    ax.cla()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 2.2)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.grid(False)
    ax.set_facecolor("#F8F8F8")

    # Draw bones
    for (child, parent), col in zip(BONES, BONE_COLOURS):
        p0 = joints[parent]
        p1 = joints[child]
        # Swap Y/Z so Y is vertical in the plot
        ax.plot(
            [p0[0], p1[0]],   # X
            [p0[2], p1[2]],   # Z (depth) → Y-axis of plot
            [p0[1], p1[1]],   # Y (height) → Z-axis of plot
            color=col, linewidth=lw, alpha=alpha,
        )

    # Draw joints
    ax.scatter(
        joints[:, 0], joints[:, 2], joints[:, 1],
        c="white", edgecolors="black", s=20, zorder=5, alpha=alpha,
    )

    # Highlight foot joints by contact status
    if contact_mask is not None:
        for fi, ji in enumerate(FOOT_JOINTS):
            colour = "#FF6600" if not contact_mask[fi] else "#00CC44"
            j = joints[ji]
            ax.scatter([j[0]], [j[2]], [j[1]],
                       c=colour, s=60, zorder=6, alpha=alpha)

    # Draw ground plane grid
    xs = np.linspace(-1, 1, 5)
    zs = np.linspace(-1, 1, 5)
    for x in xs:
        ax.plot([x, x], [zs[0], zs[-1]], [0, 0], color="#CCCCCC", lw=0.5)
    for z in zs:
        ax.plot([xs[0], xs[-1]], [z, z], [0, 0], color="#CCCCCC", lw=0.5)

    ax.view_init(elev=15, azim=-60)


# ---------------------------------------------------------------------------
# Contact mask helper
# ---------------------------------------------------------------------------

def _get_contact_mask(
    joints: np.ndarray,   # (T, 22, 3)
    t: int,
    height_thresh: float = 0.05,
    vel_thresh: float = 0.02,
) -> np.ndarray:
    """Return (4,) bool contact mask for frame t."""
    feet = joints[:, FOOT_JOINTS, :]   # (T, 4, 3)
    foot_y = feet[:, :, 1]
    floor_y = foot_y.min(axis=0)
    rel_h = foot_y - floor_y           # (T, 4)

    if t == 0:
        return (rel_h[0] < height_thresh)

    foot_vel = np.linalg.norm(feet[t] - feet[t - 1], axis=-1)  # (4,)
    height_ok = rel_h[t] < height_thresh
    vel_ok = foot_vel < vel_thresh
    return height_ok & vel_ok


# ---------------------------------------------------------------------------
# Public API: single-motion video
# ---------------------------------------------------------------------------

def save_skeleton_video(
    joints: np.ndarray,        # (T, 22, 3) or (B, T, 22, 3) — first sample used
    output_path: str,
    fps: int = 30,
    title: str = "",
    show_contact: bool = True,
    dpi: int = 100,
) -> None:
    """
    Save a skeleton animation as MP4.

    Args:
        joints:      (T, 22, 3) or (B, T, 22, 3) joint positions
        output_path: where to save the .mp4 file
        fps:         playback frame rate
        title:       text label shown in the figure
        show_contact: overlay foot contact detection
        dpi:         render resolution
    """
    if joints.ndim == 4:
        joints = joints[0]   # use first sample in batch
    T = joints.shape[0]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    if title:
        fig.suptitle(title, fontsize=10)

    def update(t: int):
        contact = _get_contact_mask(joints, t) if show_contact else None
        _draw_skeleton(ax, joints[t], contact)
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Public API: side-by-side comparison video
# ---------------------------------------------------------------------------

def save_comparison_video(
    baseline_joints: np.ndarray,   # (T, 22, 3) or (B, T, 22, 3)
    steered_joints:  np.ndarray,   # (T, 22, 3) or (B, T, 22, 3)
    output_path: str,
    fps: int = 30,
    prompt: str = "",
    constraint_label: str = "",
    show_contact: bool = True,
    dpi: int = 100,
) -> None:
    """
    Save a side-by-side baseline vs steered comparison as MP4.

    Args:
        baseline_joints: (T, 22, 3)  generated without steering
        steered_joints:  (T, 22, 3)  generated with steering
        output_path:     where to save the .mp4
        fps:             playback frame rate
        prompt:          text prompt shown as subtitle
        constraint_label: e.g. "FootContact + Terminal"
        show_contact:    overlay foot contact colours
        dpi:             render resolution
    """
    if baseline_joints.ndim == 4:
        baseline_joints = baseline_joints[0]
    if steered_joints.ndim == 4:
        steered_joints = steered_joints[0]

    T = min(baseline_joints.shape[0], steered_joints.shape[0])

    fig = plt.figure(figsize=(10, 5))
    ax_b = fig.add_subplot(121, projection="3d")
    ax_s = fig.add_subplot(122, projection="3d")

    if prompt:
        fig.suptitle(f'"{prompt}"', fontsize=9, y=0.98)
    ax_b.set_title("Baseline (no steering)", fontsize=9)
    ax_s.set_title(f"FlowSteer  [{constraint_label}]", fontsize=9)

    def update(t: int):
        c_b = _get_contact_mask(baseline_joints, t) if show_contact else None
        c_s = _get_contact_mask(steered_joints,  t) if show_contact else None
        _draw_skeleton(ax_b, baseline_joints[t], c_b)
        _draw_skeleton(ax_s, steered_joints[t],  c_s)
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2400)
    ani.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Public API: metric overlay plot
# ---------------------------------------------------------------------------

def plot_foot_contact_analysis(
    baseline_joints: np.ndarray,   # (T, 22, 3)
    steered_joints:  np.ndarray,   # (T, 22, 3)
    output_path: str,
    fps: int = 30,
) -> None:
    """
    Plot foot velocity and contact mask over time for both motions.
    Saves a static PNG.

    Useful for paper figures: shows that steering reduces sliding.
    """
    if baseline_joints.ndim == 4:
        baseline_joints = baseline_joints[0]
    if steered_joints.ndim == 4:
        steered_joints = steered_joints[0]

    T = min(baseline_joints.shape[0], steered_joints.shape[0])
    time_axis = np.arange(T - 1) / fps

    def _foot_stats(joints: np.ndarray):
        feet = joints[:, FOOT_JOINTS, :]
        foot_vel = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)   # (T-1, 4)
        foot_y = feet[:, :, 1]
        floor_y = foot_y.min(axis=0)
        rel_h = foot_y - floor_y
        contact = (rel_h[:-1] < 0.05) & (foot_vel < 0.02)          # (T-1, 4)
        return foot_vel.mean(axis=1), contact.any(axis=1)

    b_vel, b_contact = _foot_stats(baseline_joints[:T])
    s_vel, s_contact = _foot_stats(steered_joints[:T])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Foot Contact Analysis: Baseline vs FlowSteered", fontsize=12)

    # Top: foot velocity
    ax = axes[0]
    ax.plot(time_axis, b_vel, label="Baseline",       color="#CC4444", linewidth=1.5)
    ax.plot(time_axis, s_vel, label="FlowSteered",    color="#4477CC", linewidth=1.5)
    ax.axhline(0.02, color="gray", linestyle="--", linewidth=1, label="vel threshold")
    ax.fill_between(time_axis, 0, b_vel, where=b_contact, alpha=0.15, color="#CC4444", label="contact (B)")
    ax.fill_between(time_axis, 0, s_vel, where=s_contact, alpha=0.15, color="#4477CC", label="contact (S)")
    ax.set_ylabel("Mean foot velocity (m/frame)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: contact indicator
    ax = axes[1]
    ax.fill_between(time_axis, 0, b_contact.astype(float),
                    step="mid", alpha=0.6, color="#CC4444", label="Contact (Baseline)")
    ax.fill_between(time_axis, 0, s_contact.astype(float) * 0.8,
                    step="mid", alpha=0.6, color="#4477CC", label="Contact (FlowSteered)")
    ax.set_ylabel("Contact detected")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")
