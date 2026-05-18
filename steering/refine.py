"""Post-sampling latent refinement for visible, controllable motion edits.

Flow steering changes the sampling trajectory through small per-step gradient
updates.  That path is useful as a training-free prior-preserving baseline, but
it can saturate when the final sample already sits in a strong model mode.  This
module optimizes the sampled motion latent directly after generation, using the
same differentiable decoder and constraint API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from .constraints import BaseConstraint, CompositeConstraint
from .decode import MotionDecoder


@dataclass
class RefinementResult:
    latent: Tensor
    joints: Tensor
    history: List[Dict[str, float]]


class LatentRefiner:
    """Optimize a generated normalized motion latent against constraints."""

    def __init__(
        self,
        decoder: MotionDecoder,
        constraints: Union[CompositeConstraint, BaseConstraint],
        steps: int = 200,
        lr: float = 0.05,
        constraint_weight: float = 10.0,
        latent_proximity_weight: float = 1e-3,
        joint_proximity_weight: float = 0.02,
        smoothness_weight: float = 0.01,
        max_delta: float = 3.0,
        apply_latent_mask: bool = True,
        latent_mask_transl: float = 0.1,
        latent_mask_root_rot: float = 0.3,
        use_temporal_mask: bool = True,
        log_every: int = 25,
        verbose: bool = False,
    ):
        self.decoder = decoder
        self.constraints = constraints
        self.steps = steps
        self.lr = lr
        self.constraint_weight = constraint_weight
        self.latent_proximity_weight = latent_proximity_weight
        self.joint_proximity_weight = joint_proximity_weight
        self.smoothness_weight = smoothness_weight
        self.max_delta = max_delta
        self.apply_latent_mask = apply_latent_mask
        self.latent_mask_transl = latent_mask_transl
        self.latent_mask_root_rot = latent_mask_root_rot
        self.use_temporal_mask = use_temporal_mask
        self.log_every = max(1, log_every)
        self.verbose = verbose

    def _latent_dim_mask(self, D: int, device) -> Tensor:
        mask = torch.ones(D, device=device)
        if self.apply_latent_mask:
            mask[0:3] = self.latent_mask_transl
            mask[3:9] = self.latent_mask_root_rot
        return mask.view(1, 1, D)

    def _temporal_mask(self, T: int, device) -> Tensor:
        mask = torch.ones(T, device=device)
        if self.use_temporal_mask and hasattr(self.constraints, "temporal_mask"):
            c_mask = self.constraints.temporal_mask(T, device)
            if c_mask is not None:
                mask = c_mask
        return mask.view(1, T, 1)

    @staticmethod
    def _jerk_loss(joints: Tensor) -> Tensor:
        if joints.shape[1] < 4:
            return joints.new_zeros(())
        jerk = joints[:, 3:] - 3 * joints[:, 2:-1] + 3 * joints[:, 1:-2] - joints[:, :-3]
        return (jerk ** 2).mean()

    def refine(self, initial_latent: Tensor) -> RefinementResult:
        """Refine normalized latent of shape (B, T, 201)."""
        device = initial_latent.device
        self.decoder.to(device)
        x0 = initial_latent.detach()
        B, T, D = x0.shape

        with torch.no_grad():
            baseline_joints = self.decoder(x0).detach()

        delta_param = torch.nn.Parameter(torch.zeros_like(x0))
        optimizer = torch.optim.Adam([delta_param], lr=self.lr)

        dim_mask = self._latent_dim_mask(D, device)
        t_mask = self._temporal_mask(T, device)
        update_mask = t_mask * dim_mask
        history: List[Dict[str, float]] = []

        best: Optional[Tuple[float, Tensor, Tensor]] = None
        for step in range(self.steps + 1):
            optimizer.zero_grad(set_to_none=True)

            delta = delta_param * update_mask
            if self.max_delta > 0:
                delta = delta.clamp(-self.max_delta, self.max_delta)
            latent = x0 + delta
            joints = self.decoder(latent)

            constraint_loss = self.constraints(joints)
            latent_prox = (delta ** 2).mean()
            joint_prox = F.mse_loss(joints, baseline_joints)
            smooth_loss = self._jerk_loss(joints)

            loss = (
                self.constraint_weight * constraint_loss
                + self.latent_proximity_weight * latent_prox
                + self.joint_proximity_weight * joint_prox
                + self.smoothness_weight * smooth_loss
            )

            if step < self.steps:
                loss.backward()
                optimizer.step()

            loss_value = float(loss.detach().cpu())
            if best is None or loss_value < best[0]:
                best = (loss_value, latent.detach().clone(), joints.detach().clone())

            if step % self.log_every == 0 or step == self.steps:
                row = {
                    "step": float(step),
                    "loss": loss_value,
                    "constraint_loss": float(constraint_loss.detach().cpu()),
                    "latent_proximity": float(latent_prox.detach().cpu()),
                    "joint_proximity": float(joint_prox.detach().cpu()),
                    "smoothness_loss": float(smooth_loss.detach().cpu()),
                    "delta_rms": float(delta.detach().pow(2).mean().sqrt().cpu()),
                }
                history.append(row)
                if self.verbose:
                    print(
                        f"  refine {step:4d}/{self.steps} | "
                        f"L={row['loss']:.5f} C={row['constraint_loss']:.5f} "
                        f"dRMS={row['delta_rms']:.4f}"
                    )

        assert best is not None
        return RefinementResult(latent=best[1], joints=best[2], history=history)
