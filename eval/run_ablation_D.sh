#!/usr/bin/env bash
# Phase-D ablation: temporal mask + hierarchical loss + latent trust mask
#
# Run order:
#   D1 alpha sweep  →  inspect output, set BEST_ALPHA below  →  D2 / D3 / D4
#
# Usage:
#   bash eval/run_ablation_D.sh
#
# After D1 finishes, look at the "ALPHA SWEEP SUMMARY" table printed to stdout
# (or output/ablation_pose_D1_sweep_summary.json) and set BEST_ALPHA to the
# alpha with the best pose_imp_all before continuing D2-D4.
# ----------------------------------------------------------------------------

MODEL="ckpts/tencent/HY-Motion-1.0"
PROMPTS="eval/prompts/pose_eval_raw.json"
TAU=0.1

# Set this to the best alpha from D1 before running D2/D3/D4.
# Default 2.0 is a reasonable starting point; adjust after D1.
BEST_ALPHA=2.0

set -e

# --- D1: alpha sweep — find operating point under new infrastructure ---
# Tests: median→mean bug fix + temporal_mask + per-frame norm + trust region + EMA
# Expected: alpha 1-4 range is safe (temporal_mask suppresses C1-style jerk spikes)
echo "============================================================"
echo " D1  alpha sweep  seeds=42  alphas=0.5,1,2,4,8"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --seeds 42 --alpha_sweep 0.5,1.0,2.0,4.0,8.0 --soft_norm_tau $TAU --output_dir output/ablation_pose_D1

echo ""
echo ">>> D1 done.  Check output/ablation_pose_D1_sweep_summary.json"
echo ">>> then edit BEST_ALPHA in this script and re-run from D2 if needed."
echo ""

# --- D2: hierarchical joint weighting ---
# Tests: wrists/elbows/shoulders 3x, torso 0.5x  vs uniform weights (D1)
# Expected gain: high-variance subset (dance/kick) pose improvement increases
echo "============================================================"
echo " D2  hierarchical  alpha=${BEST_ALPHA}  seeds=42"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --seeds 42 --alpha_pose $BEST_ALPHA --soft_norm_tau $TAU --use_hierarchical --output_dir output/ablation_pose_D2

# --- D3: latent trust mask ---
# Tests: attenuation of transl[0:3]/root_rot[3:9] dims (0.1x / 0.3x)
# Expected gain: lower foot_sliding + jerk_ratio closer to 1.0
echo "============================================================"
echo " D3  latent_mask  alpha=${BEST_ALPHA}  seeds=42"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --seeds 42 --alpha_pose $BEST_ALPHA --soft_norm_tau $TAU --apply_latent_mask --output_dir output/ablation_pose_D3

# --- D4: full combo — paper main result row ---
# D1 best alpha + hierarchical + latent_mask, 3 seeds for statistical stability
echo "============================================================"
echo " D4  full combo  alpha=${BEST_ALPHA}  seeds=42,43,44"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --seeds 42,43,44 --alpha_pose $BEST_ALPHA --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --output_dir output/ablation_pose_D4

echo ""
echo "All Phase-D experiments complete."
echo "Results:"
echo "  D1 sweep : output/ablation_pose_D1_sweep_summary.json"
echo "  D2       : output/ablation_pose_D2/results.json"
echo "  D3       : output/ablation_pose_D3/results.json"
echo "  D4       : output/ablation_pose_D4/results.json"
