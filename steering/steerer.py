"""
FlowSteerer: training-free inference-time steering for HY-Motion 1.0.

Replaces the odeint-based sampling loop in MotionFlowMatching.generate() with
a manual Euler loop, injecting constraint gradients at each timestep.

Steering step (at each t → t+dt):
    v = v_θ(x_t, text, t)              frozen model velocity (with CFG)
    x̂_1 = x_t + (1 - t) * v           one-step clean estimate (t=1 is data)
    joints = decoder(x̂_1)              differentiable 201D → 3D joints
    L = constraints(joints)             scalar constraint loss
    s_t = -SoftNorm(∇_{x_t} L)         steering direction (soft-normalized)
    x_{t+dt} = x_t + dt * (v + α(t) * s_t)   modified Euler step

Soft normalization (default):
    scale = ‖g‖ / (‖g‖ + τ)
    s_t   = -scale · g / ‖g‖

When the gradient is large (constraint far from satisfied), scale ≈ 1 and
the steering is near unit-strength.  When the gradient is small (constraint
nearly satisfied), scale ≈ 0 and the steering self-attenuates — preventing
the "keep pushing at full strength when already close" failure mode that
causes jerk and contact violations under the hard unit-norm scheme.

Set use_unit_grad=True to restore the original unit-norm behaviour (ablation).

Because v is computed with torch.no_grad(), the gradient ∇_{x_t} L flows only
through the analytical path  x_t → x̂_1 (identity + constant shift),
NOT through the DiT. This is O(1) memory overhead vs standard sampling.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from hymotion.pipeline.motion_diffusion import MotionFlowMatching, length_to_mask
from hymotion.utils.type_converter import get_module_device

import torch.nn.functional as F

from .constraints import BaseConstraint, CompositeConstraint
from .decode import MotionDecoder
from .scheduler import PerConstraintScheduler, StagedScheduler


class FlowSteerer:
    """
    Wraps a frozen MotionFlowMatching pipeline and adds constraint steering.

    Args:
        pipeline:        loaded MotionFlowMatching (weights already loaded)
        decoder:         MotionDecoder for differentiable latent → joints
        constraints:     CompositeConstraint (or any callable joints→scalar)
        scheduler:       StagedScheduler controlling α(t)
        steps:           number of Euler steps (default matches pipeline.validation_steps)
        grad_clip:       elementwise clip for raw gradient (default 1.0)
        smooth_kernel:   temporal smoothing window (odd int, default 7)
        soft_norm_tau:   τ in soft-norm  scale = ‖g‖/(‖g‖+τ).
                         Larger τ → steering auto-attenuates sooner when loss is small.
                         Default 0.001 (calibrated: empirical ‖g‖≈0.003 over T×D dims).
                         Set use_unit_grad=True to bypass.
        use_unit_grad:   if True, use original unit-norm steering (ablation baseline).
                         Default False (soft-norm is recommended).
        verbose:         print per-step loss + gradient diagnostics
    """

    def __init__(
        self,
        pipeline: MotionFlowMatching,
        decoder: MotionDecoder,
        constraints: Optional[Union[CompositeConstraint, BaseConstraint]] = None,
        scheduler: Optional[StagedScheduler] = None,
        constraint_schedulers: Optional[List[tuple]] = None,
        timed_constraints: Optional[List[tuple]] = None,
        steps: Optional[int] = None,
        grad_clip: float = 1.0,
        smooth_kernel: int = 7,
        soft_norm_tau: float = 0.001,
        use_unit_grad: bool = False,
        verbose: bool = False,
    ):
        self.pipeline = pipeline
        self.decoder = decoder
        # -- Path 1 (recommended): time-staged composite.
        # timed_constraints: list of (BaseConstraint, weight, t_start, t_end)
        # At each ODE step t, only constraints with t_start <= t <= t_end enter the
        # composite.  A single StagedScheduler controls the overall alpha magnitude.
        # This preserves the FK-chain temporal smoothness of Stage 1 while allowing
        # different constraints to act in different ODE phases.
        self.timed_constraints = timed_constraints
        # -- Path 2 (legacy, kept for ablation): per-constraint independent gradients.
        self.constraint_schedulers = constraint_schedulers
        # -- Path 3 (legacy): static composite + single scheduler (Stage 1 style).
        self.constraints = constraints
        self.scheduler = scheduler or StagedScheduler.cosine(alpha_max=0.0)
        self.steps = steps or pipeline.validation_steps
        self.grad_clip = grad_clip
        self.smooth_kernel = smooth_kernel
        self.soft_norm_tau = soft_norm_tau
        self.use_unit_grad = use_unit_grad
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API: mirrors MotionFlowMatching.generate()
    # ------------------------------------------------------------------

    def generate(
        self,
        text: Union[str, List[str]],
        seed_input: List[int],
        duration_slider: float,
        cfg_scale: float = 5.0,
        length: Optional[int] = None,
        hidden_state_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a motion with constraint steering.

        Args:
            text:            prompt (str or list of str, length must equal len(seed_input))
            seed_input:      list of int seeds; one motion per seed
            duration_slider: duration in seconds
            cfg_scale:       classifier-free guidance scale
            length:          explicit frame count (overrides duration_slider)
            hidden_state_dict: pre-computed text embeddings (skip re-encoding)

        Returns:
            dict with keys: latent_denorm, keypoints3d, rot6d, transl,
                            root_rotations_mat, text
        """
        pl = self.pipeline
        device = get_module_device(pl)

        # --- Length ---
        if length is None:
            length = int(round(duration_slider * pl.output_mesh_fps))
        length = min(max(length, min(pl.train_frames, 20)), pl.train_frames)

        repeat = len(seed_input)
        if isinstance(text, str):
            text_list = [text] * repeat
        else:
            assert len(text) == repeat
            text_list = text

        # --- Text encoding (frozen) ---
        if hidden_state_dict is None:
            hidden_state_dict = pl.encode_text({"text": text_list})
        vtxt_input = hidden_state_dict["text_vec_raw"]
        ctxt_input = hidden_state_dict["text_ctxt_raw"]
        ctxt_length = hidden_state_dict["text_ctxt_raw_length"]

        if vtxt_input.ndim == 2:
            vtxt_input = vtxt_input[None].repeat(repeat, 1, 1)
            ctxt_input = ctxt_input[None].repeat(repeat, 1, 1)
            ctxt_length = ctxt_length.repeat(repeat)

        ctxt_mask = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_length = torch.LongTensor([length] * repeat).to(device)
        x_mask = length_to_mask(x_length, pl.train_frames)

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            null_vtxt = pl.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input_cfg = torch.cat([null_vtxt, vtxt_input], dim=0)
            ctxt_input_cfg = torch.cat([ctxt_input, ctxt_input], dim=0) \
                if not pl.enable_ctxt_null_feat else \
                torch.cat([pl.null_ctxt_input.expand(*ctxt_input.shape), ctxt_input], dim=0)
            ctxt_mask_cfg = torch.cat([ctxt_mask] * 2, dim=0)
            x_mask_cfg = torch.cat([x_mask] * 2, dim=0)
        else:
            vtxt_input_cfg = vtxt_input
            ctxt_input_cfg = ctxt_input
            ctxt_mask_cfg = ctxt_mask
            x_mask_cfg = x_mask

        # --- Velocity function (frozen model, CFG) ---
        def velocity_fn(t_scalar: float, x: Tensor) -> Tensor:
            with torch.no_grad():
                n = x.shape[0] * (2 if do_cfg else 1)
                t_tensor = torch.full((n,), t_scalar, device=device, dtype=x.dtype)
                x_in = torch.cat([x, x], dim=0) if do_cfg else x
                v_pred = pl.motion_transformer(
                    x=x_in,
                    ctxt_input=ctxt_input_cfg,
                    vtxt_input=vtxt_input_cfg,
                    timesteps=t_tensor,
                    x_mask_temporal=x_mask_cfg,
                    ctxt_mask_temporal=ctxt_mask_cfg,
                )
                if do_cfg:
                    v_uncond, v_cond = v_pred.chunk(2, dim=0)
                    v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            return v_pred

        # --- Initial noise ---
        x = pl.noise_from_seeds(
            torch.zeros(1, pl.train_frames, pl._network_module_args["input_dim"], device=device),
            seed_input,
            random_generator_on_gpu=pl.random_generator_on_gpu,
        )

        # --- Move decoder to same device ---
        self.decoder.to(device)

        # --- Manual Euler loop with steering ---
        t_vals = torch.linspace(0.0, 1.0, self.steps + 1, device=device)

        for i in range(len(t_vals) - 1):
            t_cur = t_vals[i].item()
            t_next = t_vals[i + 1].item()
            dt = t_next - t_cur

            # Frozen model velocity
            v = velocity_fn(t_cur, x)   # (B, T_max, 201)

            # Constraint steering
            if self.timed_constraints is not None:
                # Path 1: time-staged composite (recommended).
                # Build a dynamic composite from constraints active at t_cur.
                active_pairs = [
                    (c, w) for c, w, t0, t1 in self.timed_constraints
                    if t0 <= t_cur <= t1
                ]
                alpha = self.scheduler(t_cur)
                if active_pairs and alpha > 0.0:
                    composite = CompositeConstraint(active_pairs)
                    steering, loss_val = self._compute_steering(
                        x, v, t_cur, length, constraints_override=composite
                    )
                    v = v + alpha * steering
                    if self.verbose:
                        names = [type(c).__name__ for c, _ in active_pairs]
                        print(f"  step {i:3d} | t={t_cur:.3f} | α={alpha:.1f} "
                              f"| loss={loss_val:.5f} | {names}")
            elif self.constraint_schedulers is not None:
                # Path 2: per-constraint independent gradients (ablation only).
                active = [(c, s) for c, s in self.constraint_schedulers if s(t_cur) > 0.0]
                if active:
                    steering, loss_val = self._compute_steering_per_constraint(
                        x, v, t_cur, length, active
                    )
                    v = v + steering
                    if self.verbose:
                        print(f"  step {i:3d} | t={t_cur:.3f} | loss={loss_val:.5f}")
            elif self.constraints is not None:
                # Path 3: static composite (Stage 1 style).
                alpha = self.scheduler(t_cur)
                if alpha > 0.0:
                    steering, loss_val = self._compute_steering(x, v, t_cur, length)
                    v = v + alpha * steering
                    if self.verbose:
                        print(f"  step {i:3d} | t={t_cur:.3f} | α={alpha:.1f} | loss={loss_val:.5f}")

            x = x + dt * v

        # --- Decode to output format (with smoothing, non-differentiable) ---
        sampled = x[:, :length, :]
        output = pl.decode_motion_from_latent(sampled, should_apply_smooothing=True)
        return {**output, "text": text}

    # ------------------------------------------------------------------
    # Internal: constraint gradient computation
    # ------------------------------------------------------------------

    def _apply_soft_norm(self, grad_flat: Tensor) -> Tensor:
        """
        Apply soft normalization to a (B, D) gradient tensor.

        use_unit_grad=True  → hard unit-norm (original behaviour, ablation)
        use_unit_grad=False → soft-norm: scale = ‖g‖/(‖g‖+τ)

        Soft-norm self-attenuates when the gradient (and hence the loss) is
        small, avoiding the "keep pushing at full strength when already close"
        failure mode that causes jerk under hard unit-norm.
        """
        grad_norm = grad_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        unit = grad_flat / grad_norm

        if self.use_unit_grad:
            return unit                                   # hard unit-norm (ablation)

        tau   = self.soft_norm_tau
        scale = grad_norm / (grad_norm + tau)             # ∈ (0, 1)
        return scale * unit

    def _compute_steering(
        self,
        x: Tensor,           # (B, T_max, 201) current state
        v: Tensor,           # (B, T_max, 201) frozen velocity (detached)
        t: float,
        length: int,
        constraints_override: Optional[CompositeConstraint] = None,
    ) -> tuple[Tensor, float]:
        """
        Returns (steering, loss_value).
        steering: (B, T_max, 201) – same shape as v, soft-normalised per sample.

        constraints_override: if provided, use this composite instead of
            self.constraints (used by the timed-composite path).
        """
        constraints = constraints_override if constraints_override is not None \
            else self.constraints

        x_req = x.detach().requires_grad_(True)

        # One-step estimate of clean motion (t=1 is data; t=0 is noise)
        # x̂_1 = x_t + (1 - t) * v    [v is already detached, no grad through model]
        x1_hat = x_req + (1.0 - t) * v.detach()  # (B, T_max, 201)

        # Differentiable decode: only use the valid length prefix
        latent_crop = x1_hat[:, :length, :]         # (B, L, 201)
        joints = self.decoder(latent_crop)           # (B, L, 22, 3)

        # Constraint loss
        loss = constraints(joints)                   # scalar
        loss_val = loss.item()

        # Gradient w.r.t. x_req (flows through x_req → x1_hat → joints → loss)
        grad = torch.autograd.grad(loss, x_req)[0]  # (B, T_max, 201)

        B, T_max, D = grad.shape

        # Zero out the padding frames beyond valid length so they do not
        # contribute to the norm estimate and receive no steering.
        if length < T_max:
            grad = grad.clone()
            grad[:, length:, :] = 0.0

        grad_flat = grad.view(B, -1)

        # Clip element-wise BEFORE normalizing so outlier latent dims don't
        # collapse the steering direction into a single dimension.
        if self.grad_clip < float("inf"):
            grad_flat = grad_flat.clamp(-self.grad_clip, self.grad_clip)

        # Soft (or hard-unit) normalization — see _apply_soft_norm docstring.
        grad_scaled = self._apply_soft_norm(grad_flat)

        if self.verbose:
            raw_norm = grad_flat.norm(dim=-1)
            print(f"    grad_norm mean={raw_norm.mean().item():.3f}  "
                  f"max={raw_norm.max().item():.3f}  "
                  f"loss={loss_val:.5f}  "
                  f"soft_norm={'off' if self.use_unit_grad else f'τ={self.soft_norm_tau}'}")

        steering = -grad_scaled.view(B, T_max, D)  # negative gradient → constraint satisfied

        # Temporal smoothing: suppress high-frequency jerk introduced by steering.
        if self.smooth_kernel > 1:
            pad = self.smooth_kernel // 2
            s_t = steering.permute(0, 2, 1)                                  # (B, D, T_max)
            s_t = F.avg_pool1d(s_t, kernel_size=self.smooth_kernel,
                               stride=1, padding=pad)                         # (B, D, T_max)
            steering = s_t.permute(0, 2, 1)[:, :T_max, :]                    # (B, T_max, D)

        return steering.detach(), loss_val

    def _compute_steering_per_constraint(
        self,
        x: Tensor,
        v: Tensor,
        t: float,
        length: int,
        active_pairs: list,   # [(BaseConstraint, StagedScheduler), ...] pre-filtered α > 0
    ) -> tuple[Tensor, float]:
        """
        Per-constraint steering.

        Each constraint is independently backpropagated through the shared
        x_t → x̂_1 → joints graph, normalized separately, then scaled by its
        own α(t) and accumulated in latent space:

            steering = Σ_i  α_i(t) · -SoftNorm(∇_{x_t} L_i)

        Returns:
            (total_steering, sum_of_loss_values)
            total_steering: (B, T_max, 201) ready to be added to velocity
        """
        x_req = x.detach().requires_grad_(True)
        x1_hat = x_req + (1.0 - t) * v.detach()       # (B, T_max, 201)
        latent_crop = x1_hat[:, :length, :]            # (B, L, 201)
        joints = self.decoder(latent_crop)              # (B, L, 22, 3)  shared graph

        B, T_max, D = x.shape
        total_steering = x.new_zeros(B, T_max, D)
        total_loss = 0.0

        for idx, (constraint, sched) in enumerate(active_pairs):
            alpha = sched(t)
            retain = (idx < len(active_pairs) - 1)   # free graph only on last call

            loss = constraint.loss(joints)
            total_loss += loss.item()

            grad = torch.autograd.grad(loss, x_req, retain_graph=retain)[0]  # (B, T_max, D)

            # Zero out padding frames
            if length < T_max:
                grad = grad.clone()
                grad[:, length:, :] = 0.0

            grad_flat = grad.view(B, -1)
            if self.grad_clip < float("inf"):
                grad_flat = grad_flat.clamp(-self.grad_clip, self.grad_clip)

            grad_scaled = self._apply_soft_norm(grad_flat)
            steering_i = -grad_scaled.view(B, T_max, D)

            if self.smooth_kernel > 1:
                pad = self.smooth_kernel // 2
                s_t = steering_i.permute(0, 2, 1)
                s_t = F.avg_pool1d(s_t, kernel_size=self.smooth_kernel,
                                   stride=1, padding=pad)
                steering_i = s_t.permute(0, 2, 1)[:, :T_max, :]

            total_steering = total_steering + alpha * steering_i.detach()

        return total_steering, total_loss

    # ------------------------------------------------------------------
    # Convenience: build from pipeline path + constraint config
    # ------------------------------------------------------------------

    @classmethod
    def from_pipeline(
        cls,
        pipeline: MotionFlowMatching,
        stats_dir: str = "stats",
        body_model_path: str = "scripts/gradio/static/assets/dump_wooden",
        constraints: Optional[CompositeConstraint] = None,
        scheduler: Optional[StagedScheduler] = None,
        timed_constraints: Optional[List[tuple]] = None,
        constraint_schedulers: Optional[List[tuple]] = None,
        steps: int = 50,
        grad_clip: float = 1.0,
        smooth_kernel: int = 7,
        soft_norm_tau: float = 0.001,
        use_unit_grad: bool = False,
        verbose: bool = False,
    ) -> "FlowSteerer":
        decoder = MotionDecoder.from_stats_dir(
            stats_dir=stats_dir,
            body_model_path=body_model_path,
        )
        return cls(
            pipeline=pipeline,
            decoder=decoder,
            constraints=constraints,
            scheduler=scheduler,
            timed_constraints=timed_constraints,
            constraint_schedulers=constraint_schedulers,
            steps=steps,
            grad_clip=grad_clip,
            smooth_kernel=smooth_kernel,
            soft_norm_tau=soft_norm_tau,
            use_unit_grad=use_unit_grad,
            verbose=verbose,
        )
