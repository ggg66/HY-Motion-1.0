"""
Differentiable decode: 201D normalized latent → 3D world-space joint positions.

The standard pipeline in MotionFlowMatching.decode_motion_from_latent() applies
Savitzky-Golay smoothing and slerp (both numpy-based, non-differentiable).
This module skips those steps to keep gradients flowing for constraint steering.

Latent layout (o6dp, 201D after denormalization):
    [0:3]   transl      – absolute root position (x, y, z)
    [3:9]   root_rot6d  – root joint 6D rotation
    [9:135] body_rot6d  – body joints 1-21, each 6D  (21 × 6 = 126D)
    [135:201] unused    – extra features (velocity etc.), not needed for FK

Joint indices (WoodenMesh / SMPL-H 22 body joints):
    0  Pelvis   1  L_Hip    2  R_Hip    3  Spine1
    4  L_Knee   5  R_Knee   6  Spine2   7  L_Ankle
    8  R_Ankle  9  Spine3   10 L_Foot   11 R_Foot
    12 Neck     13 L_Collar 14 R_Collar 15 Head
    16 L_Shoulder 17 R_Shoulder 18 L_Elbow 19 R_Elbow
    20 L_Wrist  21 R_Wrist
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# Make sure hymotion is importable when this module is run standalone
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hymotion.pipeline.body_model import WoodenMesh
from hymotion.utils.geometry import rot6d_to_rotation_matrix


# ---------------------------------------------------------------------------
# Joint index constants
# ---------------------------------------------------------------------------

FOOT_JOINTS = [7, 8, 10, 11]   # L_Ankle, R_Ankle, L_Foot, R_Foot
ANKLE_JOINTS = [7, 8]           # L_Ankle, R_Ankle
TOE_JOINTS = [10, 11]           # L_Foot, R_Foot
ROOT_JOINT = 0                  # Pelvis
HAND_JOINTS = [20, 21]          # L_Wrist, R_Wrist
HEAD_JOINT = 15                 # Head


# ---------------------------------------------------------------------------
# MotionDecoder
# ---------------------------------------------------------------------------

class MotionDecoder(nn.Module):
    """
    Wraps WoodenMesh to provide a differentiable 201D→keypoints3d path.

    Usage:
        decoder = MotionDecoder(mean, std, body_model)
        joints = decoder(latent)   # (B, T, 22, 3)
    """

    # Latent feature slices
    TRANSL_SLICE = slice(0, 3)
    ROOT_R6D_SLICE = slice(3, 9)
    BODY_R6D_SLICE = slice(9, 135)   # 21 joints × 6D = 126D
    NUM_BODY_JOINTS = 21
    NUM_JOINTS = 22   # body joints (no fingers)

    def __init__(
        self,
        mean: Tensor,              # (201,) or (1, 201)
        std: Tensor,               # (201,) or (1, 201)
        body_model: Optional[WoodenMesh] = None,
        body_model_path: str = "scripts/gradio/static/assets/dump_wooden",
    ):
        super().__init__()

        mean = mean.view(1, 1, -1).float()   # (1, 1, 201)
        std = std.view(1, 1, -1).float()
        # Zero-variance dims: replace std=0 with 1 to avoid /0
        std_safe = std.clone()
        std_safe[std_safe < 1e-3] = 1.0

        self.register_buffer("mean", mean)
        self.register_buffer("std_safe", std_safe)

        if body_model is None:
            body_model = WoodenMesh(model_path=body_model_path)
        self.body_model = body_model
        # Freeze body model (no trainable params anyway, but be explicit)
        for p in self.body_model.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Core differentiable path
    # ------------------------------------------------------------------

    def forward(self, latent: Tensor) -> Tensor:
        """
        Args:
            latent: (B, T, 201) normalized motion latent

        Returns:
            joints: (B, T, 22, 3) world-space 3D joint positions
                    NO smoothing applied, fully differentiable.
        """
        B, T, _ = latent.shape

        # 1. De-normalize
        x = latent * self.std_safe + self.mean   # (B, T, 201)

        # 2. Extract transl + rot6d
        transl = x[..., self.TRANSL_SLICE]                              # (B, T, 3)
        root_r6d = x[..., self.ROOT_R6D_SLICE].unsqueeze(-2)           # (B, T, 1, 6)
        body_r6d = x[..., self.BODY_R6D_SLICE].view(B, T, self.NUM_BODY_JOINTS, 6)  # (B, T, 21, 6)
        rot6d = torch.cat([root_r6d, body_r6d], dim=-2)                # (B, T, 22, 6)

        # 3. Forward kinematics via WoodenMesh (differentiable LBS)
        result = self.body_model.forward_batch({"rot6d": rot6d, "trans": transl})
        # keypoints3d from WoodenMesh is LOCAL space (FK without global transl).
        # Add global translation to get WORLD-space joint positions.
        local_joints = result["keypoints3d"][:, :, :self.NUM_JOINTS, :]  # (B, T, 22, 3)
        world_joints = local_joints + transl.unsqueeze(2)                 # (B, T, 22, 3)
        return world_joints

    # ------------------------------------------------------------------
    # Convenience: load from HY-Motion stats directory
    # ------------------------------------------------------------------

    @classmethod
    def from_stats_dir(
        cls,
        stats_dir: str = "stats",
        body_model_path: str = "scripts/gradio/static/assets/dump_wooden",
    ) -> "MotionDecoder":
        mean = torch.from_numpy(np.load(os.path.join(stats_dir, "Mean.npy"))).float()
        std = torch.from_numpy(np.load(os.path.join(stats_dir, "Std.npy"))).float()
        return cls(mean=mean, std=std, body_model_path=body_model_path)
