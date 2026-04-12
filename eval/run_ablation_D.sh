#!/usr/bin/env bash
# Phase-D ablation + multi-constraint experiments
#
# Cross-seed protocol: target_seed=42 → steer_seeds={43,44}
#
# Usage:
#   bash eval/run_ablation_D.sh
# ----------------------------------------------------------------------------

MODEL="ckpts/tencent/HY-Motion-1.0"
PROMPTS="eval/prompts/pose_eval_raw.json"
TAU=0.1
TARGET_SEED=42

set -e

# ============================================================
# D1–D5: already done in previous runs.
# Kept here for reference / re-run.
# ============================================================

# --- D1: alpha sweep, no latent_mask ---
# python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43 --alpha_sweep 0.5,1.0,2.0,4.0,6.0,8.0 --soft_norm_tau $TAU --output_dir output/ablation_pose_D1

# --- D5: full combo alpha sweep ---
# python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43 --alpha_sweep 2.0,4.0,6.0,8.0,12.0 --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --output_dir output/ablation_pose_D5

# ============================================================
# Priority 1: temporal_mask ablation at alpha=6
# ============================================================
echo "============================================================"
echo " P1a  no_temporal_mask  α=6  full combo  seeds=43,44"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43,44 --alpha_pose 6.0 --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --no_temporal_mask --output_dir output/ablation_pose_P1_no_tmask

# ============================================================
# Priority 4: hierarchical ablation at alpha=6
# ============================================================
echo "============================================================"
echo " P4  latent_mask only (no hierarchical)  α=6  seeds=43,44"
echo "============================================================"
python eval/run_pose_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43,44 --alpha_pose 6.0 --soft_norm_tau $TAU --apply_latent_mask --output_dir output/ablation_pose_P4_no_hier

# ============================================================
# Priority 2: multi-constraint combos  (low-variance subset, faster)
# Main config: α=6, latent_mask+hierarchical
# ============================================================
echo "============================================================"
echo " P2a  pose+foot  α=6  seeds=43,44  low-variance"
echo "============================================================"
python eval/run_multiconstraint_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43,44 --combo pose+foot --prompts_subset low --alpha 6.0 --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --output_dir output/ablation_mc_pose_foot

echo "============================================================"
echo " P2b  pose+waypoint  α=6  seeds=43,44  low-variance"
echo "============================================================"
python eval/run_multiconstraint_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43,44 --combo pose+waypoint --prompts_subset low --alpha 6.0 --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --output_dir output/ablation_mc_pose_waypoint

echo "============================================================"
echo " P2c  pose+foot+waypoint  α=6  seeds=43,44  low-variance"
echo "============================================================"
python eval/run_multiconstraint_sc.py --model_path "$MODEL" --prompt_file "$PROMPTS" --target_seed $TARGET_SEED --seeds 43,44 --combo pose+foot+waypoint --prompts_subset low --alpha 6.0 --soft_norm_tau $TAU --use_hierarchical --apply_latent_mask --output_dir output/ablation_mc_all

echo ""
echo "All experiments complete."
echo "Results:"
echo "  P1 no_tmask : output/ablation_pose_P1_no_tmask/results.json"
echo "  P4 no_hier  : output/ablation_pose_P4_no_hier/results.json"
echo "  P2a pose+ft : output/ablation_mc_pose_foot/results.json"
echo "  P2b pose+wp : output/ablation_mc_pose_waypoint/results.json"
echo "  P2c all     : output/ablation_mc_all/results.json"
