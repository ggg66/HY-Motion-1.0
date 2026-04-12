#!/usr/bin/env bash
# Phase-D ablation: temporal mask + hierarchical loss + latent trust mask
#
# Protocol: cross-seed pose recovery
#   target_seed=42 generates the reference pose target.
#   steer_seeds={43,44} are the seeds being steered toward that target.
#   err_base = natural distance between steer_seed pose and target_seed pose.
#   improvement = (err_base - err_steer) / err_base * 100%  (well-defined, no ÷0).
#
# Run order:
#   D1 alpha sweep  →  inspect output, set BEST_ALPHA below  →  D2 / D3 / D4
#
# Usage:
#   bash eval/run_ablation_D.sh
# ----------------------------------------------------------------------------

MODEL="ckpts/tencent/HY-Motion-1.0"
PROMPTS="eval/prompts/pose_eval_raw.json"
TAU=0.1
TARGET_SEED=42
STEER_SEEDS_SINGLE=43
STEER_SEEDS_MULTI=43,44

# Set this to the best alpha from D1 before running D2/D3/D4.
BEST_ALPHA=2.0

set -e

# --- D1: alpha sweep — find operating point under new infrastructure ---
# Tests: median→mean fix + temporal_mask + per-frame norm + trust region + EMA
# Cross-seed: tgt=42 → steer=43
echo "============================================================"
echo " D1  alpha sweep  tgt=42 steer=43  alphas=0.5,1,2,4,8"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds $STEER_SEEDS_SINGLE --alpha_sweep 0.5,1.0,2.0,4.0,8.0 --soft_norm_tau $TAU --output_dir output/ablation_pose_D1

echo ""
echo ">>> D1 done. Check output/ablation_pose_D1_sweep_summary.json"
echo ">>> Set BEST_ALPHA in this script, then continue D2-D4."
echo ""

# --- D2: hierarchical joint weighting ---
# wrists/elbows/shoulders 3x, torso 0.5x  vs D1 uniform weights
echo "============================================================"
echo " D2  hierarchical  alpha=${BEST_ALPHA}  tgt=42 steer=43"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds $STEER_SEEDS_SINGLE --alpha_pose $BEST_ALPHA --soft_norm_tau $TAU --use_hierarchical --output_dir output/ablation_pose_D2

# --- D3: latent trust mask ---
# attenuation of transl[0:3]/root_rot[3:9] dims (0.1x / 0.3x)
echo "============================================================"
echo " D3  latent_mask  alpha=${BEST_ALPHA}  tgt=42 steer=43"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds $STEER_SEEDS_SINGLE --alpha_pose $BEST_ALPHA --soft_norm_tau $TAU --apply_latent_mask --output_dir output/ablation_pose_D3

# --- D4: full combo — paper main result row ---
# D1 best alpha + hierarchical + latent_mask, 2 steer seeds for statistical stability
echo "============================================================"
echo " D4  full combo  alpha=${BEST_ALPHA}  tgt=42 steer=43,44"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds $STEER_SEEDS_MULTI --alpha_pose $BEST_ALPHA --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --output_dir output/ablation_pose_D4

echo ""
echo "All Phase-D experiments complete."
echo "Results:"
echo "  D1 sweep : output/ablation_pose_D1_sweep_summary.json"
echo "  D2       : output/ablation_pose_D2/results.json"
echo "  D3       : output/ablation_pose_D3/results.json"
echo "  D4       : output/ablation_pose_D4/results.json"
