"""
Paper-quality figure generator for FlowSteer-Motion.

Generates static PNG figures showing temporal frame sequences
suitable for CVPR submission. Mirrors the style of HY-Motion Fig. 6:
multiple evenly-spaced frames per row, rows = configurations, clean white
background, color-coded bones, ground-plane grid.

Main API
--------
save_comparison_figure(row_specs, output_path, ...)
    Renders n_frames evenly-spaced skeleton frames per row.
    Rows are labeled on the left; keyframe column is starred (★).

save_ablation_figure(row_specs, output_path, ...)
    Identical layout but uses a muted color scheme for disabled components.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: registers 3D projection


# ── Skeleton definition (SMPL-H 22-joint) ────────────────────────────────────

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
BONES = [(i, p) for i, p in enumerate(SMPL22_PARENTS) if p >= 0]

_LEFT  = {1, 4, 7, 10, 13, 16, 18, 20}
_RIGHT = {2, 5, 8, 11, 14, 17, 19, 21}


def _bone_colour(child: int, style: str = "normal") -> str:
    """
    style:
        "normal"  – vivid blue/red/green
        "muted"   – desaturated for ablation (disabled) rows
        "target"  – lighter tint for target/reference rows
    """
    if style == "muted":
        if child in _LEFT:  return "#99AACC"
        if child in _RIGHT: return "#CCAAAA"
        return "#AACCAA"
    if style == "target":
        if child in _LEFT:  return "#6699DD"
        if child in _RIGHT: return "#DD6666"
        return "#66BB66"
    # normal
    if child in _LEFT:  return "#2255BB"
    if child in _RIGHT: return "#BB2222"
    return "#22AA22"


_COLOURS = {
    style: [_bone_colour(c, style) for c, _ in BONES]
    for style in ("normal", "muted", "target")
}


# ── Sequence helpers ──────────────────────────────────────────────────────────

def normalize_sequence(joints: np.ndarray) -> np.ndarray:
    """
    Normalize a motion sequence for display.
    - XZ: centre on pelvis at frame 0 so trajectory starts at origin.
    - Y:  shift so minimum foot height = 0.

    Args:
        joints: (T, 22, 3) world-space joint positions

    Returns:
        (T, 22, 3) normalized joint positions
    """
    j = joints.copy()
    j[:, :, 0] -= j[0, 0, 0]   # pelvis X at frame 0 → 0
    j[:, :, 2] -= j[0, 0, 2]   # pelvis Z at frame 0 → 0
    floor = j[:, [7, 8, 10, 11], 1].min()
    j[:, :, 1] -= floor
    return j


def _seq_bounds(
    joints_list: List[np.ndarray],
    margin: float = 0.25,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute a shared (xlim, ylim, zlim) bounding box covering all sequences.
    ylim is the depth axis (Z in world space).
    """
    all_j = np.concatenate(joints_list, axis=0)   # (total_T, 22, 3)
    xs, ys, zs = all_j[:, :, 0], all_j[:, :, 1], all_j[:, :, 2]
    xlim = (xs.min() - margin, xs.max() + margin)
    ylim = (zs.min() - margin, zs.max() + margin)   # depth axis
    zlim = (max(0.0, ys.min() - 0.05), ys.max() + 0.3)   # height axis
    return xlim, ylim, zlim


# ── Low-level draw helpers ────────────────────────────────────────────────────

def _draw_skeleton(
    ax,
    joints22: np.ndarray,       # (22, 3)
    colours: List[str],
    lw: float = 1.5,
    alpha: float = 1.0,
    ms: float = 8.0,
) -> None:
    """Draw one skeleton frame into a 3D axes (Y = height, Z = depth)."""
    for (child, par), col in zip(BONES, colours):
        p0, p1 = joints22[par], joints22[child]
        ax.plot(
            [p0[0], p1[0]],   # X
            [p0[2], p1[2]],   # Z → depth axis in the plot
            [p0[1], p1[1]],   # Y → height axis in the plot
            color=col, lw=lw, alpha=alpha, solid_capstyle="round",
        )
    ax.scatter(
        joints22[:, 0], joints22[:, 2], joints22[:, 1],
        c="white", edgecolors="#444444", s=ms,
        zorder=5, alpha=alpha, linewidths=0.4,
    )


def _setup_axis(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],   # depth axis
    zlim: Tuple[float, float],   # height axis
    elev: float = 18,
    azim: float = -70,
) -> None:
    """Configure 3D axis for paper-quality rendering."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_axis_off()
    ax.set_facecolor("white")
    # Remove pane backgrounds
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    # Ground-plane grid
    for gx in np.linspace(xlim[0], xlim[1], 5):
        ax.plot([gx, gx], [ylim[0], ylim[1]], [0, 0], color="#CCCCCC", lw=0.4)
    for gz in np.linspace(ylim[0], ylim[1], 5):
        ax.plot([xlim[0], xlim[1]], [gz, gz], [0, 0], color="#CCCCCC", lw=0.4)


# ── RowSpec type ──────────────────────────────────────────────────────────────
# label:      row label shown on the left
# joints:     (T, 22, 3) motion sequence
# style:      bone colour style ("normal" | "muted" | "target")
# t_norm:     keyframe position in [0,1]; used to mark the nearest column (None = no mark)
RowSpec = Tuple[str, np.ndarray, str, Optional[float]]


# ── Public API ────────────────────────────────────────────────────────────────

def save_comparison_figure(
    row_specs: List[RowSpec],
    output_path: str,
    n_frames: int = 7,
    cell_w: float = 1.7,     # inches per skeleton cell
    cell_h: float = 2.3,     # inches per row
    label_w: float = 1.05,   # inches for row label column
    dpi: int = 200,
    elev: float = 18,
    azim: float = -70,
    suptitle: str = "",
    fps: int = 30,
) -> None:
    """
    Generate a paper-style temporal frame-sequence comparison figure.

    Each row shows n_frames evenly-spaced skeleton frames.  Rows share
    a common bounding box and viewing angle for consistent comparison.
    The column nearest the keyframe is marked with ★.

    Args:
        row_specs:    list of (label, joints(T,22,3), style, t_norm|None)
            style choices: "normal" (vivid), "muted" (greyed-out), "target" (lighter)
        output_path:  path to save PNG
        n_frames:     number of frames to show per row (default 7)
        cell_w/h:     subplot cell dimensions in inches
        label_w:      row label column width in inches
        dpi:          output resolution (200 for paper, 100 for draft)
        elev, azim:   3D viewing angle
        suptitle:     figure title (prompt text)
        fps:          frames per second for time labels
    """
    n_rows = len(row_specs)
    fig_w  = label_w + n_frames * cell_w
    fig_h  = cell_h * n_rows + (0.5 if suptitle else 0.1)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    if suptitle:
        fig.text(
            0.5, 1.0 - 0.1 / fig_h, f'"{suptitle}"',
            ha="center", va="top", fontsize=9, fontstyle="italic",
            color="#333333",
        )

    # Normalise all sequences and build shared bounding box
    normed_seqs = [normalize_sequence(joints) for _, joints, _, _ in row_specs]
    xlim, ylim, zlim = _seq_bounds(normed_seqs)

    for row_idx, ((label, joints, style, t_norm), joints_n) in enumerate(
        zip(row_specs, normed_seqs)
    ):
        T = joints_n.shape[0]
        cols = [_bone_colour(c, style) for c, _ in BONES]

        # Evenly-spaced frame indices
        frame_idxs = [int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)]

        # Column closest to the keyframe
        kf_frame = int(round(t_norm * (T - 1))) if t_norm is not None else None
        kf_col   = (
            int(np.argmin([abs(fi - kf_frame) for fi in frame_idxs]))
            if kf_frame is not None else None
        )

        for col_idx, fi in enumerate(frame_idxs):
            # Normalised axes bounds [0, 1] in figure space
            l = (label_w + col_idx * cell_w) / fig_w
            b = 1.0 - (row_idx + 1) * cell_h / fig_h
            w = cell_w  / fig_w
            h = cell_h  / fig_h

            ax = fig.add_axes([l, b, w, h], projection="3d")
            _setup_axis(ax, xlim, ylim, zlim, elev=elev, azim=azim)
            _draw_skeleton(ax, joints_n[fi], cols, lw=1.8, alpha=1.0, ms=9)

            # Time label
            t_sec = fi / fps
            ax.text2D(
                0.5, 0.0, f"t={t_sec:.1f}s", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=6.5, color="#777777",
            )

            # Keyframe marker
            if kf_col is not None and col_idx == kf_col:
                ax.text2D(
                    0.5, 1.01, "★", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=10, color="#CC8800",
                )

        # Row label
        lax = fig.add_axes(
            [0.0, 1.0 - (row_idx + 0.5) * cell_h / fig_h, label_w / fig_w, 0.0]
        )
        lax.set_axis_off()
        lax.text(
            0.95, 0.5, label, transform=lax.transAxes,
            ha="right", va="center", fontsize=9, fontweight="bold",
            color="#222222",
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_multi_prompt_figure(
    prompts_data: List[Tuple[str, List[RowSpec]]],
    output_path: str,
    n_frames: int = 7,
    cell_w: float = 1.55,
    cell_h: float = 2.0,
    label_w: float = 1.05,
    prompt_h: float = 0.28,   # inches for prompt title row
    dpi: int = 200,
    elev: float = 18,
    azim: float = -70,
    fps: int = 30,
) -> None:
    """
    Generate a combined multi-prompt comparison figure.

    Layout:
        ┌─ Prompt A ─────────────────────────────────────┐
        │  [Label] [f0] [f1] [f2] [f3★] [f4] [f5] [f6]  │  row A1
        │  [Label] [f0] [f1] [f2] [f3★] [f4] [f5] [f6]  │  row A2
        ├─ Prompt B ─────────────────────────────────────┤
        │  ...                                            │
        └────────────────────────────────────────────────┘

    Args:
        prompts_data: list of (prompt_text, row_specs) — one entry per prompt
        output_path:  output PNG path
        n_frames:     frames per row
        cell_w/h:     cell size in inches
        label_w:      row label column width
        prompt_h:     height of prompt title separator row
        dpi:          output resolution
        elev, azim:   3D viewing angle
        fps:          frame rate for time labels
    """
    # Total rows accounting for prompt separators
    # Each prompt group contributes: 1 separator + len(rows) cell rows
    total_h = sum(
        prompt_h + len(rows) * cell_h
        for _, rows in prompts_data
    ) + 0.1  # bottom margin

    fig_w = label_w + n_frames * cell_w
    fig = plt.figure(figsize=(fig_w, total_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Normalise all sequences and compute shared bounding box
    all_normed: List[np.ndarray] = []
    for _, row_specs in prompts_data:
        for _, joints, _, _ in row_specs:
            all_normed.append(normalize_sequence(joints))

    xlim, ylim, zlim = _seq_bounds(all_normed)

    # Track current top offset as a fraction of figure height
    normed_iter = iter(all_normed)
    y_offset = 0.0   # cumulative from top (in inches)

    for p_idx, (prompt_text, row_specs) in enumerate(prompts_data):
        # ── Prompt separator ──────────────────────────────────────────────────
        sep_b = 1.0 - (y_offset + prompt_h) / total_h
        sep_h = prompt_h / total_h
        sep_ax = fig.add_axes([0.0, sep_b, 1.0, sep_h])
        sep_ax.set_axis_off()
        sep_ax.set_facecolor("#F0F0F0")
        sep_ax.axhline(0.5, color="#CCCCCC", lw=0.5)
        prompt_label = f'"{prompt_text}"' if len(prompt_text) < 70 else f'"{prompt_text[:67]}..."'
        sep_ax.text(
            0.02, 0.5, prompt_label,
            transform=sep_ax.transAxes, ha="left", va="center",
            fontsize=8.5, fontstyle="italic", color="#444444",
        )
        y_offset += prompt_h

        for row_idx, (label, joints, style, t_norm) in enumerate(row_specs):
            joints_n = next(normed_iter)
            T = joints_n.shape[0]
            cols = [_bone_colour(c, style) for c, _ in BONES]

            frame_idxs = [int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)]
            kf_frame = int(round(t_norm * (T - 1))) if t_norm is not None else None
            kf_col   = (
                int(np.argmin([abs(fi - kf_frame) for fi in frame_idxs]))
                if kf_frame is not None else None
            )

            for col_idx, fi in enumerate(frame_idxs):
                l = (label_w + col_idx * cell_w) / fig_w
                b = 1.0 - (y_offset + cell_h) / total_h
                w = cell_w   / fig_w
                h = cell_h   / total_h

                ax = fig.add_axes([l, b, w, h], projection="3d")
                _setup_axis(ax, xlim, ylim, zlim, elev=elev, azim=azim)
                _draw_skeleton(ax, joints_n[fi], cols, lw=1.6, alpha=1.0, ms=7)

                t_sec = fi / fps
                ax.text2D(
                    0.5, 0.0, f"t={t_sec:.1f}s", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=6, color="#777777",
                )

                if kf_col is not None and col_idx == kf_col:
                    ax.text2D(
                        0.5, 1.01, "★", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=9, color="#CC8800",
                    )

            # Row label
            lax = fig.add_axes(
                [0.0, 1.0 - (y_offset + 0.5 * cell_h) / total_h,
                 label_w / fig_w, 0.0]
            )
            lax.set_axis_off()
            lax.text(
                0.95, 0.5, label, transform=lax.transAxes,
                ha="right", va="center", fontsize=8.5, fontweight="bold",
                color="#222222",
            )

            y_offset += cell_h

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(
        output_path, dpi=dpi, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.close(fig)
    print(f"  Saved: {output_path}")
