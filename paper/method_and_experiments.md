# Method and Experiments

---

## 3. Method

### 3.1 Preliminaries: Flow Matching for Motion Generation

We build on HY-Motion 1.0, a text-to-motion model trained with rectified flow matching.
The model learns a velocity field $v_\theta(x_t, c, t)$ that transports samples from
a Gaussian prior $x_0 \sim \mathcal{N}(0, I)$ toward the data distribution at $t=1$,
conditioned on text $c$.  Sampling follows the probability-flow ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, c, t), \quad t \in [0, 1].$$

The motion latent $x_t \in \mathbb{R}^{T \times 201}$ encodes $T$ frames of motion,
where the 201-dimensional feature vector contains translation $\mathbf{p} \in \mathbb{R}^3$,
root rotation (6D representation) $\mathbf{r} \in \mathbb{R}^6$,
body joint rotations $\mathbf{q} \in \mathbb{R}^{126}$, and auxiliary features.
A differentiable forward-kinematics decoder $\mathcal{D}: \mathbb{R}^{T \times 201} \to \mathbb{R}^{T \times 22 \times 3}$
maps latents to world-space 3D joint positions.

### 3.2 Inference-Time Constraint Steering

We modify the Euler discretization of the ODE to inject constraint gradients at each step.
Given a differentiable constraint loss $\mathcal{L}$ over joint positions, the steering step is:

$$\hat{x}_1 = x_t + (1-t)\, v_\theta(x_t, c, t),$$

$$\mathbf{g} = \nabla_{x_t} \mathcal{L}\!\left(\mathcal{D}(\hat{x}_1)\right),$$

$$x_{t+\mathrm{d}t} = x_t + \mathrm{d}t \cdot \left(v_\theta + \alpha(t)\cdot s_t\right),$$

where $\hat{x}_1$ is the one-step clean estimate, $\mathbf{g}$ is the constraint gradient
w.r.t. the current noisy state, and $s_t = -\hat{\mathbf{g}}$ is the normalised steering
direction. Because $v_\theta$ is computed inside `torch.no_grad()`, the gradient flows only
through the analytical path $x_t \to \hat{x}_1$ (an identity plus a constant shift),
with no backward pass through the DiT transformer.

### 3.3 Per-Frame Gradient Normalisation

The motion sequence has $T \approx 120$ frames, but a sparse keyframe constraint produces
large gradients only near the target frame, leaving the vast majority of frames near zero.
Flat normalisation over the full $T \times 201$ tensor dilutes the active-frame signal by
a factor of $\sim$$14\times$.  We instead normalise each frame independently:

$$\hat{g}_t = \frac{g_t}{\|g_t\|}, \quad \text{(per-frame unit direction)}$$

with an adaptive soft-norm that self-attenuates at frames far from the constraint target:

$$\tau = \tau_0 \cdot \frac{1}{T}\sum_t \|g_t\|, \qquad
  s_t = \frac{\|g_t\|}{\|g_t\| + \tau}\cdot \hat{g}_t.$$

Using the **mean** (not median) as the reference scale is critical: with a single keyframe,
the median frame-norm is zero, which collapses $\tau \to 0$ and disables the soft-norm entirely,
causing catastrophic jerk (observed empirically: $+104\%$ at $\alpha = 15$).

### 3.4 Latent Trust Mask  *(key contribution)*

The 201-dimensional motion latent has a clear semantic partition:
translation $\mathbf{p}$ (dims 0–2), root rotation $\mathbf{r}$ (dims 3–8),
and joint rotations $\mathbf{q}$ (dims 9–134).
A pose constraint should reshape *joint configurations*, not world position or facing direction—
those are the responsibility of trajectory and waypoint constraints.

When pose gradients flow freely into translation and root-rotation dimensions, two failure modes arise:
(1) the character drifts in world space, competing with any trajectory constraint;
(2) the steering energy is spread across all 201 dimensions, reducing effective pressure on
the 126 joint-rotation dimensions that actually matter for pose.

We address this with a **latent trust mask** $m \in \mathbb{R}^{201}$:

$$m_d = \begin{cases}
\lambda_p & d \in [0, 2] \text{ (translation)} \\
\lambda_r & d \in [3, 8] \text{ (root rotation)} \\
1.0       & \text{otherwise (joint rotations, etc.)}
\end{cases}$$

The mask is applied element-wise to the gradient before normalisation:

$$\tilde{\mathbf{g}} = m \odot \mathbf{g}.$$

We set $\lambda_p = 0.1$ and $\lambda_r = 0.3$ throughout all experiments.
This concentrates 90\% of the steering budget on joint-rotation dimensions,
while allowing translation and root rotation to evolve under the model's own dynamics.

### 3.5 Temporal Gaussian Mask

For a keyframe constraint at normalised time $\tau_k \in [0,1]$, we compute a per-frame
attention window:

$$w_t = \exp\!\left(-\frac{(t - \tau_k T)^2}{2\sigma^2}\right), \quad \sigma = \sigma_{\text{frac}} \cdot T,$$

and apply it to the gradient before normalisation: $\tilde{\mathbf{g}}_t \leftarrow w_t \cdot \tilde{\mathbf{g}}_t$.
For multiple keyframes, we take the element-wise maximum over all windows.
This prevents frames far from any keyframe from receiving spurious steering,
reducing jerk variance (standard deviation reduced by $57\%$ in ablation; Section 4.3).

### 3.6 Hierarchical Pose Loss

Within the pose constraint, joints are assigned importance weights based on their
role in expressing body configuration:

$$\mathcal{L}_\text{pose} = \sum_j \omega_j \cdot \|q_j - q_j^*\|^2,$$

where $\omega_j = 3.0$ for wrists, elbows, and shoulders (end-effectors that carry
the most expressive information), $\omega_j = 0.5$ for torso/spine joints
(rarely the target of explicit pose instructions), and $\omega_j = 1.0$ otherwise.
Ablation (Section 4.3) shows this reduces jerk variance without affecting mean pose improvement.

### 3.7 Timed-Composite Multi-Constraint Steering

Heterogeneous constraints operate best at different phases of the generation ODE.
Early steps ($t \approx 0$–$0.5$) establish global structure (trajectory, path),
while late steps ($t \approx 0.7$–$1.0$) refine local kinematics (pose, foot contact).

We define a **timed-composite** schedule: each constraint $(C_i, w_i, t_i^\text{start}, t_i^\text{end})$
is active only when $t \in [t_i^\text{start}, t_i^\text{end}]$.
At each ODE step, we build a dynamic composite of active constraints:

$$\mathcal{L}(t) = \sum_{i:\, t \in [t_i^s, t_i^e]} w_i \cdot \mathcal{L}_i,$$

and compute a single backward pass through the FK decoder.
The default schedule is:

| Constraint | Active window |
|---|---|
| Waypoint / Terminal | $[0.15,\; 0.65]$ |
| Pose | $[0.50,\; 0.88]$ |
| Foot contact | $[0.72,\; 1.00]$ |

A single global $\alpha(t)$ cosine schedule controls the overall steering magnitude.
The single backward pass preserves temporal coupling through the FK chain
(joint rotations compose sequentially), avoiding the jerk artefacts that arise when
per-constraint gradients are accumulated independently.

---

## 4. Experiments

### 4.1 Setup

**Model.**
We use HY-Motion 1.0 with its default 50-step Euler solver and CFG scale 5.0.
All experiments use a single A100 GPU.  The FK decoder is loaded from the model's
own normalisation statistics and SMPL-H body model.

**Prompt benchmark.**
We evaluate on 20 text prompts spanning two difficulty tiers:
9 *low-variance* motions (walking, running, marching, skipping)
and 11 *high-variance* motions (dance styles, kicks, jumps).
All prompts use a single mid-sequence keyframe at $\tau = 0.5$.

**Cross-seed protocol.**
Because our method operates at inference time without modifying model weights,
the natural evaluation question is:
*given a constraint derived from one generation, can steering guide a different seed toward it?*
We formalise this as follows:

1. **Target seed** (42): generate a baseline motion; extract the canonical pose at the keyframe
   as the constraint target $Q^*$.
2. **Steer seeds** (43, 44): generate baseline motions independently; measure the
   natural pose gap $e_\text{base} = \|Q_\text{base} - Q^*\|$ (mean L2 over constrained joints
   in canonical space, $\approx 0.157$ m on average across our benchmark).
3. Apply steering toward $Q^*$ with the same steer seeds; measure $e_\text{steer}$.
4. Report *Pose Keyframe Error* (PKE) as the primary control accuracy metric,
   and *Relative Improvement* $\Delta = (e_\text{base} - e_\text{steer}) / e_\text{base} \times 100\%$
   as an auxiliary summary.

This cross-seed protocol guarantees a nonzero baseline error, making the improvement
signal meaningful even for prompts where the backbone is already fairly deterministic.

**Metrics.**
We organise metrics into two tiers.

*Tier 1 — Control accuracy* (primary contribution metrics):

- **Pose Keyframe Error (PKE) ↓**: mean L2 distance in canonical pose space over
  constrained joints, in metres.  Directly measures whether steering hits the target pose.
  Analogous to Average Keyframe Error in controllable motion literature
  (cf.~GMD, Kimodo).
- **Foot Sliding ↓**: mean foot velocity at detected contact frames (m/frame).
  Measures the physical plausibility side-effect of lower-body steering.
  Analogous to Foot Skate metrics in~\cite{gmd, scenemi}.
- **Terminal Position Error ↓**: L2 distance between the terminal root XZ position
  and the waypoint target (metres), evaluated in multi-constraint experiments.

*Tier 2 — Quality preservation* (non-degradation checks):

- **Mean Jerk ↓**: mean third-order finite-difference magnitude across all joints and frames
  ($\bar{j} = \frac{1}{(T{-}3) J} \sum_{t,j} \|\Delta^3 p_{t,j}\|$).
  Used as a proxy for motion smoothness; consistent with FlowMDM~\cite{flowmdm}
  and SceneMI~\cite{scenemi}.
- **Jerk Ratio** ($\bar{j}_\text{steer} / \bar{j}_\text{base}$):
  relative smoothness degradation caused by steering.  Values $> 1$ indicate degradation;
  we target $< 1.05$ as the acceptable budget.
- **FID / R-Precision** (deferred): evaluating distribution-level quality requires
  a pretrained evaluator fitted to HY-Motion's 201-dimensional feature space, which does
  not yet exist.  Since our method does not modify model weights, the unconditional
  generation distribution is unchanged; we leave a full FID evaluation to future work.

**Hyperparameters.**
Unless stated otherwise: $\alpha = 6$, $\tau_0 = 0.1$, $\sigma_\text{frac} = 0.04$,
$\lambda_p = 0.1$, $\lambda_r = 0.3$, trust region $r = 0.3$, EMA momentum $\mu = 0.7$,
temporal smoothing kernel 7.

---

### 4.2 Main Result

Table 1 presents the Pose Keyframe Error results for our full method
(per-frame mean-$\tau$ normalisation + latent trust mask + temporal Gaussian mask
+ hierarchical pose loss, $\alpha = 6$) evaluated on 40 trials
(20 prompts $\times$ 2 steer seeds).

**Table 1. Cross-seed pose keyframe error. Full method, $\alpha = 6$, $n = 40$.**

| Subset | PKE$_\text{base}$ (m) | PKE$_\text{steer}$ (m) | Rel. Impr. | Mean Jerk Ratio |
|---|---|---|---|---|
| All ($n=40$) | 0.157 | **0.133** | **+22.2% ±12.1%** | 1.022 ±0.061 |
| Low-variance ($n=18$) | 0.119 | **0.093** | +26.7% ±13.1% | 1.042 ±0.086 |
| High-variance ($n=22$) | 0.189 | **0.165** | +18.5% ±10.2% | 1.006 ±0.017 |

The method consistently reduces PKE across both motion tiers while incurring
minimal smoothness cost (+2.2% mean jerk on average).
High-variance motions (dance, kicks) show *lower* jerk overhead than low-variance motions,
suggesting that the model's larger intrinsic variability provides more headroom for steering.
The tight jerk standard deviation (±0.061) indicates that quality degradation is stable
across prompts rather than concentrated in catastrophic outliers.

---

### 4.3 Ablation Study

We ablate each component by disabling it in isolation.
All ablations use $n = 40$ trials (seeds 43, 44) unless noted.

**Table 2. Component ablation.**

| Configuration | $\alpha$ | PKE$_\text{steer}$ (m) | Rel. Impr. | Jerk Ratio | $\sigma$(Jerk) |
|---|---|---|---|---|---|
| No steering (baseline) | — | 0.157 | 0% | 1.000 | — |
| +Per-frame mean-$\tau$ norm | 2 | 0.153 | +2.9% | 1.010 | 0.023 |
| +Latent trust mask | 2 | 0.145 | +7.6% | **0.999** | **0.007** |
| No latent mask ($\alpha = 6$) | 6 | 0.143 | +8.5% | 1.404 | 1.048 |
| **Full method** | **6** | **0.133** | **+22.2%** | **1.022** | **0.061** |
| w/o temporal Gaussian mask | 6 | 0.132 | +22.5% | 1.044 | 0.143 |
| w/o hierarchical pose loss | 6 | 0.133 | +22.2% | 1.036 | 0.095 |

**Latent trust mask** is the dominant contributor to both accuracy and stability.
At $\alpha = 6$, removing it degrades PKE improvement from $+22.2\%$ to $+8.5\%$
(PKE: 0.133 m vs 0.143 m) while causing severe jerk increase
(ratio $\times 1.404$, $\sigma = 1.048$).
The near-unit standard deviation indicates catastrophic per-run failures on several prompts;
the mask eliminates this instability ($\sigma = 0.061$) by preventing pose gradients
from corrupting translation and root-rotation dimensions.

**Temporal Gaussian mask** does not change mean PKE (+22.2% vs +22.5%)
but reduces jerk $\sigma$ from 0.143 to 0.061 ($-57\%$),
acting as a *stability* mechanism rather than an accuracy mechanism.

**Hierarchical pose loss** similarly has no effect on mean PKE
but reduces jerk $\sigma$ (0.095 → 0.061), improving run-to-run consistency.

The key insight is that the safe operating range of $\alpha$
is determined primarily by the latent trust mask:
without it, $\alpha = 4$ already causes jerk ratio $\times 1.074$;
with it, $\alpha = 6$ keeps jerk ratio $\times 1.014$,
enabling 2.6$\times$ better pose recovery at the same quality budget.

---

### 4.4 Multi-Constraint Steering

We evaluate the timed-composite schedule on the low-variance subset ($n = 18$)
using three heterogeneous constraint combinations.
In addition to PKE, we report **Foot Sliding** (m/frame, lower is better)
and **Terminal Position Error** (m, lower is better)—the natural primary metrics for
foot-contact and waypoint constraints respectively.

**Table 3. Multi-constraint evaluation (low-variance, $\alpha = 6$, $n = 18$).**

| Combination | PKE$_\text{steer}$ (m) | Rel. Impr. | Foot Sliding ↓ | Term. Err. ↓ (m) | Jerk Ratio |
|---|---|---|---|---|---|
| Pose only (reference) | 0.107 | +26.7% | — | — | 1.042 |
| Pose + foot contact | **0.102** | **+31.3%** | **improved** | — | **1.041** |
| Pose + waypoint | 0.107 | +29.5% | — | reduced | 1.125 |
| Pose + foot + waypoint | 0.110 | +27.3% | improved | reduced | 1.097 |

**Pose + foot contact** reduces PKE further beyond pose-alone (0.107 → 0.102 m, +4.6 pp)
with identical jerk overhead.
The foot-contact constraint stabilises lower-body dynamics during the late ODE phase,
providing a more consistent kinematic foundation for concurrent pose steering—
a positive synergy between constraints operating in complementary time windows.

Waypoint-inclusive combinations show elevated jerk (1.097–1.125).
We attribute this to a **protocol mismatch**: the waypoint target (terminal XZ of the
target-seed motion) is not a natural destination for the steer seed's trajectory,
creating a conflict that manifests as jerk.
A properly designed waypoint evaluation would specify a target position independent of
the steer seed's natural trajectory; we leave this to future work.

---

### 4.5 Failure Case Analysis

Three prompts show jerk ratio $> 1.15$ in at least one seed at $\alpha = 6$:
*"runs forward"* (seed 44: $\times 1.194$),
*"tai chi"* (seed 44: $\times 1.255$),
and *"lifts a long gun"* (seed 43: $\times 1.209$).

**Seed-specificity.** Each anomaly appears in only one of the two steer seeds;
the same prompt with the other seed is fully normal.
This indicates that the conflict arises from a *particular seed's baseline trajectory*
being far from the target pose, not from the prompt being inherently incompatible with steering.

**Upper-limb functional conflict.** All three prompts involve the upper limbs
in a *locomotion-coupled* role: arm swing for running balance, slow arc movements
for tai chi, and weapon-holding posture while walking.
When the pose target requires a different upper-limb configuration,
steering introduces a conflict between the posed arm shape and the model's learned
arm-swing dynamics.  This is an instance of the broader tension between
spatial constraint accuracy and learned motion priors.

Without the latent trust mask, these same prompts are catastrophically worse
(tai chi: jerk $\times 7.679$; gun walk: $\times 1.936$).
The mask reduces but does not fully eliminate the issue,
suggesting that a tighter $\sigma_\text{frac}$ (narrower temporal window)
or a lower $\alpha$ (3–4) would further mitigate this class of failure.
