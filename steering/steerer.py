"""
FlowSteerer: training-free inference-time steering for HY-Motion 1.0.

Replaces the odeint-based sampling loop in MotionFlowMatching.generate() with
a manual Euler loop, injecting constraint gradients at each timestep.

Steering step (at each t → t+dt):
    v = v_θ(x_t, text, t)              frozen model velocity (with CFG)
    x̂_1 = x_t + (1 - t) * v           one-step clean estimate (t=1 is data)
    joints = decoder(x̂_1)              differentiable 201D → 3D joints
    L = constraints(joints)             scalar constraint loss
    s_t = -Norm(∇_{x_t} L)            steering direction
    x_{t+dt} = x_t + dt * (v + α(t) * s_t)   modified Euler step

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

from .constraints import BaseConstraint, CompositeConstraint
from .decode import MotionDecoder
from .scheduler import StagedScheduler


class FlowSteerer:
    """
    Wraps a frozen MotionFlowMatching pipeline and adds constraint steering.

    Args:
        pipeline:      loaded MotionFlowMatching (weights already loaded)
        decoder:       MotionDecoder for differentiable latent → joints
        constraints:   CompositeConstraint (or any callable joints→scalar)
        scheduler:     StagedScheduler controlling α(t)
        steps:         number of Euler steps (default matches pipeline.validation_steps)
        grad_clip:     max norm for the steering gradient (default 1.0)
        verbose:       print per-step loss
    """

    def __init__(
        self,
        pipeline: MotionFlowMatching,
        decoder: MotionDecoder,
        constraints: Optional[Union[CompositeConstraint, BaseConstraint]] = None,
        scheduler: Optional[StagedScheduler] = None,
        steps: Optional[int] = None,
        grad_clip: float = 1.0,
        verbose: bool = False,
    ):
        self.pipeline = pipeline
        self.decoder = decoder
        self.constraints = constraints
        self.scheduler = scheduler or StagedScheduler.cosine(alpha_max=0.0)
        self.steps = steps or pipeline.validation_steps
        self.grad_clip = grad_clip
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
            alpha = self.scheduler(t_cur) if self.constraints is not None else 0.0
            if alpha > 0.0 and self.constraints is not None:
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

    def _compute_steering(
        self,
        x: Tensor,           # (B, T_max, 201) current state
        v: Tensor,           # (B, T_max, 201) frozen velocity (detached)
        t: float,
        length: int,
    ) -> tuple[Tensor, float]:
        """
        Returns (steering, loss_value).
        steering: (B, T_max, 201) – same shape as v, L2-normalised per sample
        """
        x_req = x.detach().requires_grad_(True)

        # One-step estimate of clean motion (t=1 is data; t=0 is noise)
        # x̂_1 = x_t + (1 - t) * v    [v is already detached, no grad through model]
        x1_hat = x_req + (1.0 - t) * v.detach()  # (B, T_max, 201)

        # Differentiable decode: only use the valid length prefix
        latent_crop = x1_hat[:, :length, :]         # (B, L, 201)
        joints = self.decoder(latent_crop)           # (B, L, 22, 3)

        # Constraint loss
        loss = self.constraints(joints)              # scalar
        loss_val = loss.item()

        # Gradient w.r.t. x_req (flows through x_req → x1_hat → joints → loss)
        grad = torch.autograd.grad(loss, x_req)[0]  # (B, T_max, 201)

        # Normalize per-sample (avoid scale sensitivity to α)
        B = grad.shape[0]
        grad_flat = grad.view(B, -1)
        grad_norm = grad_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad_flat = grad_flat / grad_norm

        if self.grad_clip < float("inf"):
            grad_flat = torch.clamp(grad_flat, -self.grad_clip, self.grad_clip)

        steering = -grad_flat.view_as(grad)   # negative gradient = ascent → constraint satisfied

        return steering.detach(), loss_val

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
        steps: int = 50,
        grad_clip: float = 1.0,
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
            steps=steps,
            grad_clip=grad_clip,
            verbose=verbose,
        )
