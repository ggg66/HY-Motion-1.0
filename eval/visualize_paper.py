"""
Paper-quality figure generator for FlowSteer-Motion.

Generates static PNG figures suitable for CVPR submission.

Layout philosophy
-----------------
- Each skeleton cell is independently centred on the current frame's pelvis
  so the character fills the cell even in long travelling motions.
- The keyframe column (★) uses a canonical pose view (root-centred,
  yaw-aligned) so the pose comparison matches the evaluation metric exactly.
  An optional ghost skeleton shows the constraint target.
- Three bone-colour styles: "normal" (vivid), "muted" (greyed ablation row),
  "target" (lighter reference row).

Main API
--------
save_comparison_figure(row_specs, output_path, ...)
save_multi_prompt_figure(prompts_data, output_path, ...)
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D   # noqa: registers 3D projection


# ── Skeleton ──────────────────────────────────────────────────────────────────

SMPL22_PARENTS = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
    12, 13, 14, 16, 17, 18, 19,
]
BONES = [(i, p) for i, p in enumerate(SMPL22_PARENTS) if p >= 0]

_LEFT  = {1, 4, 7, 10, 13, 16, 18, 20}
_RIGHT = {2, 5, 8, 11, 14, 17, 19, 21}


def _bone_colour(child: int, style: str = "normal") -> str:
    if style == "muted":
        if child in _LEFT:  return "#99AACC"
        if child in _RIGHT: return "#CCAAAA"
        return "#AACCAA"
    if style == "target":
        if child in _LEFT:  return "#6699DD"
        if child in _RIGHT: return "#DD6666"
        return "#66BB66"
    if child in _LEFT:  return "#2255BB"
    if child in _RIGHT: return "#BB2222"
    return "#22AA22"


def _colours(style: str) -> List[str]:
    return [_bone_colour(c, style) for c, _ in BONES]

_GHOST_COLOURS = ["#BBBBBB"] * len(BONES)


# ── Pose helpers ──────────────────────────────────────────────────────────────

def _canonicalize(joints22: np.ndarray) -> np.ndarray:
    """
    Root-centred + yaw-aligned canonical pose.
    Mirrors eval/metrics.py::canonicalize_frame_np.
    joints22: (22, 3)  →  (22, 3)
    """
    root = joints22[0]
    centred = joints22 - root
    hip_vec = joints22[2] - joints22[1]   # R_Hip - L_Hip
    hx, hz  = hip_vec[0], hip_vec[2]
    norm    = math.sqrt(hx**2 + hz**2) + 1e-6
    fx, fz  = -hz / norm, hx / norm
    yaw     = math.atan2(fx, fz)
    cy, sy  = math.cos(yaw), math.sin(yaw)
    x, y, z = centred[:, 0], centred[:, 1], centred[:, 2]
    return np.stack([x * cy - z * sy, y, x * sy + z * cy], axis=-1)


def normalize_sequence(joints: np.ndarray) -> np.ndarray:
    """
    Translate so frame-0 pelvis is at XZ=(0,0); shift Y so floor=0.
    joints: (T, 22, 3)
    """
    j = joints.copy()
    j[:, :, 0] -= j[0, 0, 0]
    j[:, :, 2] -= j[0, 0, 2]
    floor = j[:, [7, 8, 10, 11], 1].min()
    j[:, :, 1] -= floor
    return j


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_skeleton(
    ax,
    joints22: np.ndarray,      # (22, 3)  — world or canonical, already centred
    colours: List[str],
    lw: float = 1.6,
    alpha: float = 1.0,
    ms: float = 8.0,
) -> None:
    """Draw skeleton into a 3D axes (Y=height, Z=depth)."""
    for (child, par), col in zip(BONES, colours):
        p0, p1 = joints22[par], joints22[child]
        ax.plot(
            [p0[0], p1[0]], [p0[2], p1[2]], [p0[1], p1[1]],
            color=col, lw=lw, alpha=alpha, solid_capstyle="round",
        )
    ax.scatter(
        joints22[:, 0], joints22[:, 2], joints22[:, 1],
        c="white", edgecolors="#444444", s=ms,
        zorder=5, alpha=alpha, linewidths=0.4,
    )


def _setup_ax(ax, cx: float, cz: float, cy_floor: float,
              half: float = 0.75, h_top: float = 2.1,
              elev: float = 18, azim: float = -70,
              canonical: bool = False) -> None:
    """
    Configure 3D axes centred at (cx, cz) in the XZ plane.
    canonical=True uses a tighter window suitable for canonical pose comparison.
    """
    if canonical:
        half = 0.65
        h_top = 2.0
    xlim = (cx - half, cx + half)
    ylim = (cz - half, cz + half)
    zlim = (cy_floor, cy_floor + h_top)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_axis_off()
    ax.set_facecolor("white")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("none")
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)
    # Ground grid
    for gx in np.linspace(xlim[0], xlim[1], 5):
        ax.plot([gx, gx], [ylim[0], ylim[1]], [zlim[0], zlim[0]], color="#CCCCCC", lw=0.4)
    for gz in np.linspace(ylim[0], ylim[1], 5):
        ax.plot([xlim[0], xlim[1]], [gz, gz], [zlim[0], zlim[0]], color="#CCCCCC", lw=0.4)


# ── RowSpec ───────────────────────────────────────────────────────────────────
# (row_label, joints(T,22,3), style, t_norm|None, target_pose(22,3)|None)
# target_pose: canonical target joint positions, shown as ghost at keyframe col
RowSpec = Tuple[str, np.ndarray, str, Optional[float], Optional[np.ndarray]]


# ── Core figure builder ───────────────────────────────────────────────────────

def _render_rows(
    fig,
    row_specs: List[RowSpec],
    n_frames: int,
    cell_w: float,
    cell_h: float,
    label_w: float,
    fig_w: float,
    fig_h: float,
    y_top_inch: float,    # distance from figure top to first row top (inches)
    elev: float,
    azim: float,
    fps: int,
) -> None:
    """Render a block of rows into fig, starting at y_top_inch from the top."""
    for row_idx, (label, joints, style, t_norm, target_pose) in enumerate(row_specs):
        joints_n = normalize_sequence(joints)
        T = joints_n.shape[0]
        cols = _colours(style)

        frame_idxs = [int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)]
        kf_frame = int(round(t_norm * (T - 1))) if t_norm is not None else None
        kf_col   = (
            int(np.argmin([abs(fi - kf_frame) for fi in frame_idxs]))
            if kf_frame is not None else None
        )

        for col_idx, fi in enumerate(frame_idxs):
            is_kf = (kf_col is not None and col_idx == kf_col)

            # ── Axes position ──────────────────────────────────────────────
            row_top = y_top_inch + row_idx * cell_h
            l = (label_w + col_idx * cell_w) / fig_w
            b = 1.0 - (row_top + cell_h) / fig_h
            w = cell_w / fig_w
            h = cell_h / fig_h

            ax = fig.add_axes([l, b, w, h], projection="3d")

            if is_kf:
                # ── Canonical view: front-facing, removes root transl+rot ─────
                canon_frame = _canonicalize(joints_n[fi])
                pelvis_y = joints_n[fi, 0, 1]
                canon_frame[:, 1] += pelvis_y
                floor_y = max(0.0, pelvis_y - 1.0)

                # Front view: azim=-90 looks directly at the character's face,
                # showing lateral (X) and vertical (Y) arm positions clearly.
                _setup_ax(ax, cx=0.0, cz=0.0, cy_floor=floor_y,
                          half=0.65, h_top=2.1,
                          elev=10, azim=-90, canonical=True)

                # Ghost: canonical target pose (prominent orange)
                if target_pose is not None:
                    ghost = target_pose.copy()
                    ghost[:, 1] += pelvis_y
                    _draw_skeleton(ax, ghost, ["#FF8C00"] * len(BONES),
                                   lw=1.4, alpha=0.55, ms=6)

                # Actual skeleton on top
                _draw_skeleton(ax, canon_frame, cols, lw=2.0, alpha=1.0, ms=10)

                # Highlight constrained joints (shoulders/elbows/wrists) with
                # larger dots so the viewer knows exactly where to look
                _CONSTRAINED = [16, 17, 18, 19, 20, 21]  # shoulders,elbows,wrists
                cj = canon_frame[_CONSTRAINED]
                ax.scatter(cj[:, 0], cj[:, 2], cj[:, 1],
                           c="#FFDD00", edgecolors="#333333", s=55,
                           zorder=10, linewidths=0.8)

                ax.patch.set_facecolor("#FFFBE6")
                ax.patch.set_alpha(0.6)

            else:
                # ── World view: per-frame centred ──────────────────────────
                px = joints_n[fi, 0, 0]   # pelvis X
                pz = joints_n[fi, 0, 2]   # pelvis Z
                floor_y = 0.0

                _setup_ax(ax, cx=px, cz=pz, cy_floor=floor_y,
                          half=0.75, h_top=2.1,
                          elev=elev, azim=azim)
                _draw_skeleton(ax, joints_n[fi], cols, lw=1.6, alpha=1.0, ms=8)

            # ── Time label ─────────────────────────────────────────────────
            t_sec = fi / fps
            ax.text2D(0.5, 0.0, f"t={t_sec:.1f}s", transform=ax.transAxes,
                      ha="center", va="bottom", fontsize=6.5, color="#777777")

            # ── Keyframe star ──────────────────────────────────────────────
            if is_kf:
                ax.text2D(0.5, 1.01, "★", transform=ax.transAxes,
                          ha="center", va="bottom", fontsize=10, color="#CC8800")

        # ── Row label ───────────────────────────────────────────────────────
        row_mid = y_top_inch + (row_idx + 0.5) * cell_h
        lax = fig.add_axes([0.0, 1.0 - row_mid / fig_h, label_w / fig_w, 0.0])
        lax.set_axis_off()
        lax.text(0.95, 0.5, label, transform=lax.transAxes,
                 ha="right", va="center", fontsize=9, fontweight="bold",
                 color="#222222")


# ── Public API ────────────────────────────────────────────────────────────────

def save_comparison_figure(
    row_specs: List[RowSpec],
    output_path: str,
    n_frames: int = 7,
    cell_w: float = 1.7,
    cell_h: float = 2.3,
    label_w: float = 1.05,
    dpi: int = 200,
    elev: float = 18,
    azim: float = -70,
    suptitle: str = "",
    fps: int = 30,
) -> None:
    """
    Single-prompt comparison figure.

    Args:
        row_specs: list of (label, joints(T,22,3), style, t_norm|None, target_pose(22,3)|None)
            style: "normal" | "muted" | "target"
            target_pose: optional canonical ghost skeleton shown at keyframe column
        output_path: PNG output path
        n_frames: frames per row (default 7)
        dpi: 200 for paper, 100 for draft
        suptitle: prompt text shown as figure title
    """
    n_rows = len(row_specs)
    fig_w  = label_w + n_frames * cell_w
    fig_h  = cell_h * n_rows + (0.45 if suptitle else 0.1)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    if suptitle:
        fig.text(0.5, 1.0 - 0.08 / fig_h, f'"{suptitle}"',
                 ha="center", va="top", fontsize=9, fontstyle="italic",
                 color="#333333")

    y_top = 0.45 if suptitle else 0.1
    _render_rows(fig, row_specs, n_frames, cell_w, cell_h, label_w,
                 fig_w, fig_h, y_top, elev, azim, fps)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_multi_prompt_figure(
    prompts_data: List[Tuple[str, List[RowSpec]]],
    output_path: str,
    n_frames: int = 7,
    cell_w: float = 1.55,
    cell_h: float = 2.0,
    label_w: float = 1.05,
    prompt_h: float = 0.28,
    dpi: int = 200,
    elev: float = 18,
    azim: float = -70,
    fps: int = 30,
) -> None:
    """
    Multi-prompt stacked comparison figure.

    Each prompt block has a grey separator bar with the prompt text,
    followed by its rows of skeleton sequences.

    Args:
        prompts_data: list of (prompt_text, row_specs)
        prompt_h: height of the separator bar in inches
    """
    total_h = sum(
        prompt_h + len(rows) * cell_h
        for _, rows in prompts_data
    ) + 0.1

    fig_w = label_w + n_frames * cell_w
    fig = plt.figure(figsize=(fig_w, total_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    y_offset = 0.0

    for prompt_text, row_specs in prompts_data:
        # ── Separator bar ──────────────────────────────────────────────────
        sep_b = 1.0 - (y_offset + prompt_h) / total_h
        sep_ax = fig.add_axes([0.0, sep_b, 1.0, prompt_h / total_h])
        sep_ax.set_facecolor("#F2F2F2")
        sep_ax.set_axis_off()
        label_txt = f'"{prompt_text}"' if len(prompt_text) < 68 else f'"{prompt_text[:65]}…"'
        sep_ax.text(0.015, 0.5, label_txt,
                    transform=sep_ax.transAxes, ha="left", va="center",
                    fontsize=8.5, fontstyle="italic", color="#444444")
        y_offset += prompt_h

        # ── Skeleton rows ──────────────────────────────────────────────────
        _render_rows(fig, row_specs, n_frames, cell_w, cell_h, label_w,
                     fig_w, total_h, y_offset, elev, azim, fps)
        y_offset += len(row_specs) * cell_h

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved: {output_path}")
