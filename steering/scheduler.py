"""
α(t) time-staged steering strength scheduler.

Convention: t ∈ [0, 1], where t=0 is pure noise and t=1 is clean data.
The scheduler controls how strongly constraints are applied at each timestep.

Intuition:
    Early steps (t ≈ 0): global structure forms   → apply semantic / terminal constraints
    Mid steps   (t ≈ 0.5): local detail forms      → apply geometric / waypoint constraints
    Late steps  (t ≈ 1):  fine-grained refinement  → apply physical / contact constraints
"""

from __future__ import annotations

import math
from typing import Callable, Optional


class StagedScheduler:
    """
    Flexible α(t) scheduler with several built-in modes.

    Args:
        alpha_max:     peak steering strength (tune per task, typical 50–200)
        mode:          one of 'constant', 'cosine', 'linear_ramp', 'staged'
        t_start:       start applying steering at this t (default 0.0 = always)
        t_end:         stop applying steering at this t (default 1.0 = always)
        warmup_frac:   fraction of [t_start, t_end] used for linear warm-up
                       (only for 'linear_ramp' and 'staged')

    Mode descriptions:
        constant      α(t) = alpha_max  (uniform throughout)
        cosine        α(t) = alpha_max · sin(π·t)  (peaks at t=0.5)
        linear_ramp   α(t) ramps linearly from 0 to alpha_max then holds
        staged        α(t) follows a trapezoidal profile per constraint type
                      (use make_staged() factory for this)
    """

    def __init__(
        self,
        alpha_max: float = 100.0,
        mode: str = "cosine",
        t_start: float = 0.0,
        t_end: float = 1.0,
        warmup_frac: float = 0.1,
    ):
        assert mode in ("constant", "cosine", "linear_ramp", "staged")
        self.alpha_max = alpha_max
        self.mode = mode
        self.t_start = t_start
        self.t_end = t_end
        self.warmup_frac = warmup_frac

        # For 'staged' mode: filled in by make_staged()
        self._stages: list[tuple[float, float, float]] = []

    def __call__(self, t: float) -> float:
        """Return α at timestep t ∈ [0, 1]."""
        if t < self.t_start or t > self.t_end:
            return 0.0

        span = self.t_end - self.t_start
        if span < 1e-8:
            return self.alpha_max

        # Normalised position within active window
        u = (t - self.t_start) / span   # ∈ [0, 1]

        if self.mode == "constant":
            return self.alpha_max

        elif self.mode == "cosine":
            return self.alpha_max * math.sin(math.pi * u)

        elif self.mode == "linear_ramp":
            if u < self.warmup_frac:
                return self.alpha_max * (u / self.warmup_frac)
            return self.alpha_max

        elif self.mode == "staged":
            return self._eval_staged(t)

        return 0.0

    def _eval_staged(self, t: float) -> float:
        """Piecewise linear profile defined by self._stages."""
        if not self._stages:
            return self.alpha_max
        # stages = list of (t_lo, t_hi, alpha) trapezoids
        total = 0.0
        for (t_lo, t_hi, alpha) in self._stages:
            if t_lo <= t <= t_hi:
                total += alpha
        return min(total, self.alpha_max)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def constant(cls, alpha_max: float = 100.0) -> "StagedScheduler":
        return cls(alpha_max=alpha_max, mode="constant")

    @classmethod
    def cosine(cls, alpha_max: float = 100.0) -> "StagedScheduler":
        return cls(alpha_max=alpha_max, mode="cosine")

    @classmethod
    def make_staged(
        cls,
        alpha_terminal: float = 80.0,
        alpha_waypoint: float = 80.0,
        alpha_contact: float = 60.0,
    ) -> "StagedScheduler":
        """
        Pre-configured stage profile:
            Terminal constraint:  t ∈ [0.0, 0.7]   (global structure)
            Waypoint constraint:  t ∈ [0.2, 0.9]   (mid-level geometry)
            Contact constraint:   t ∈ [0.5, 1.0]   (fine physical detail)
        """
        sched = cls(alpha_max=max(alpha_terminal, alpha_waypoint, alpha_contact),
                    mode="staged")
        sched._stages = [
            (0.0, 0.7, alpha_terminal),
            (0.2, 0.9, alpha_waypoint),
            (0.5, 1.0, alpha_contact),
        ]
        return sched


# ---------------------------------------------------------------------------
# Per-constraint alpha wrapper
# ---------------------------------------------------------------------------

class PerConstraintScheduler:
    """
    Assigns a separate StagedScheduler to each constraint in a CompositeConstraint.
    Returns a dict of {constraint_idx: alpha} at time t.

    Usage:
        sched = PerConstraintScheduler([
            StagedScheduler(alpha_max=80, t_start=0.0, t_end=0.7),  # terminal
            StagedScheduler(alpha_max=60, t_start=0.5, t_end=1.0),  # contact
        ])
        alphas = sched(t)  # {0: ..., 1: ...}
    """

    def __init__(self, schedulers: list[StagedScheduler]):
        self.schedulers = schedulers

    def __call__(self, t: float) -> dict[int, float]:
        return {i: s(t) for i, s in enumerate(self.schedulers)}
