"""
Phase 2: FID and R-Precision evaluation for FlowSteer-Motion.

Converts saved .npy files (HY-Motion 30fps world-space joints) to
HumanML3D 263D features, then computes:
  - FID between baseline and steered distributions
  - R-Precision (Top-1, Top-3) for text-motion alignment

Requires:
  - MoMask evaluator checkpoint:
      <momask_root>/checkpoints/t2m/text_mot_match/model/finest.tar
    Download with:
      cd <momask_root> && bash prepare/download_evaluator.sh

  - (Optional) GloVe word vectors for R-Precision text embeddings:
      <momask_root>/dataset/HumanML3D/  (contains glove_data.npy etc.)

Usage:
    python eval/compute_fid.py \\
        --npy_dir output/eval_phase1 \\
        --results_json output/eval_phase1/results_summary.json \\
        --momask_root /path/to/momask-codes \\
        --output_json output/eval_phase1/fid_results.json

    # With GPU:
    python eval/compute_fid.py \\
        --npy_dir output/eval_phase1 \\
        --results_json output/eval_phase1/results_summary.json \\
        --momask_root /path/to/momask-codes \\
        --gpu_id 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup: add momask_root to sys.path so we can import its utils
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _add_momask_to_path(momask_root: str):
    if momask_root not in sys.path:
        sys.path.insert(0, momask_root)


# ---------------------------------------------------------------------------
# HumanML3D skeleton constants (same as paramUtil.py)
# ---------------------------------------------------------------------------

_T2M_RAW_OFFSETS = np.array([
    [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0],
    [0, -1, 0], [0, -1, 0], [0, 1, 0], [0, -1, 0],
    [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1],
    [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1],
    [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
    [0, -1, 0], [0, -1, 0],
])

_T2M_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20],
]

# HumanML3D face/foot joint indices
_FACE_JOINT_IDX = [2, 1, 17, 16]   # r_hip, l_hip, sdr_r, sdr_l
_FID_R = [8, 11]                    # right foot joints
_FID_L = [7, 10]                    # left foot joints
_L_IDX1, _L_IDX2 = 5, 8            # lower leg indices for scale


# ---------------------------------------------------------------------------
# 30fps → 20fps downsampling
# ---------------------------------------------------------------------------

def downsample_30_to_20(joints: np.ndarray) -> np.ndarray:
    """
    Subsample (T, J, 3) from 30fps to 20fps.
    Takes 2 out of every 3 frames: [0,1, 3,4, 6,7, ...]
    """
    T = joints.shape[0]
    keep = [i for i in range(T) if i % 3 != 2]
    return joints[keep]


# ---------------------------------------------------------------------------
# Reference skeleton offsets
# ---------------------------------------------------------------------------

def _get_reference_offsets(momask_root: str, reference_joints: np.ndarray):
    """
    Compute target skeleton offsets.
    Prefers loading from HumanML3D reference pose (000021.npy);
    falls back to using the first frame of our generated data.
    """
    _add_momask_to_path(momask_root)
    from common.skeleton import Skeleton
    import torch

    n_raw_offsets = torch.from_numpy(_T2M_RAW_OFFSETS)
    skel = Skeleton(n_raw_offsets, _T2M_KINEMATIC_CHAIN, 'cpu')

    # Try HumanML3D reference pose
    candidate_paths = [
        os.path.join(momask_root, "dataset", "HumanML3D", "new_joints", "000021.npy"),
        os.path.join(momask_root, "dataset", "pose_data_raw", "joints", "000021.npy"),
    ]
    ref_pose = None
    for p in candidate_paths:
        if os.path.exists(p):
            data = np.load(p)
            ref_pose = torch.from_numpy(data[:1, :22, :].reshape(1, 22, 3))
            print(f"  Reference skeleton: {p}")
            break

    if ref_pose is None:
        # Fall back to first frame of our data (same SMPL skeleton → offsets ≈ identical)
        ref_pose = torch.from_numpy(reference_joints[:1].reshape(1, 22, 3))
        print("  Reference skeleton: using first generated frame (no HumanML3D dataset found)")

    tgt_offsets = skel.get_offsets_joints(ref_pose[0])
    return tgt_offsets


# ---------------------------------------------------------------------------
# World-space joints → 263D HumanML3D features
# ---------------------------------------------------------------------------

def joints_to_263d(
    joints_seq: np.ndarray,       # (T, 22, 3) world-space, 20fps
    momask_root: str,
    tgt_offsets,
    feet_thre: float = 0.002,
) -> Optional[np.ndarray]:        # (T-1, 263) or None on failure
    """
    Convert world-space 22-joint positions to HumanML3D 263D features.
    Input must already be at 20fps.
    """
    _add_momask_to_path(momask_root)
    from utils.motion_process import process_file

    try:
        # process_file expects (T, 22, 3) and uses module-level globals
        # We patch them in temporarily
        import utils.motion_process as mp
        import utils.paramUtil as pu
        import torch

        # Patch globals used by process_file
        mp.n_raw_offsets      = torch.from_numpy(_T2M_RAW_OFFSETS)
        mp.kinematic_chain    = _T2M_KINEMATIC_CHAIN
        mp.face_joint_indx    = _FACE_JOINT_IDX
        mp.fid_r              = _FID_R
        mp.fid_l              = _FID_L
        mp.l_idx1             = _L_IDX1
        mp.l_idx2             = _L_IDX2
        mp.tgt_offsets        = tgt_offsets

        data, _, _, _ = process_file(joints_seq.copy(), feet_thre)
        return data   # (T-1, 263)
    except Exception as e:
        print(f"    [Warning] 263D conversion failed: {e}")
        return None


def batch_joints_to_263d(
    joints_batch: np.ndarray,    # (B, T, 22, 3) world-space, 30fps
    momask_root: str,
    tgt_offsets,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert a batch of 30fps joint sequences to 263D features at 20fps.

    Returns:
        features: list of (T_i, 263) arrays (variable length)
        lengths:  list of T_i values
    """
    features, lengths = [], []
    B = joints_batch.shape[0]

    for b in range(B):
        joints_20fps = downsample_30_to_20(joints_batch[b])  # (T', 22, 3)
        feat = joints_to_263d(joints_20fps, momask_root, tgt_offsets)
        if feat is not None and len(feat) >= 8:
            features.append(feat)
            lengths.append(len(feat))
        else:
            print(f"    [Skip] sample {b}: conversion returned None or too short")

    return features, lengths


# ---------------------------------------------------------------------------
# Normalize features with HumanML3D mean/std
# ---------------------------------------------------------------------------

def _load_mean_std(momask_root: str):
    """Load HumanML3D normalization stats."""
    candidates = [
        (os.path.join(momask_root, "dataset", "HumanML3D", "Mean.npy"),
         os.path.join(momask_root, "dataset", "HumanML3D", "Std.npy")),
        (os.path.join(momask_root, "dataset", "t2m", "Mean.npy"),
         os.path.join(momask_root, "dataset", "t2m", "Std.npy")),
    ]
    for mean_path, std_path in candidates:
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean = np.load(mean_path)
            std  = np.load(std_path)
            print(f"  Normalization stats: {mean_path}")
            return mean, std
    print("  [Warning] HumanML3D mean/std not found; features will be unnormalized")
    return None, None


def normalize_features(features: List[np.ndarray], mean, std) -> List[np.ndarray]:
    if mean is None:
        return features
    return [(f - mean) / (std + 1e-8) for f in features]


# ---------------------------------------------------------------------------
# Pad/truncate to fixed length for batch processing
# ---------------------------------------------------------------------------

def pad_to_batch(
    features: List[np.ndarray],
    max_len: int = 196,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length feature sequences to a fixed max_len.

    Returns:
        padded: (N, max_len, 263) float tensor
        lengths: (N,) int tensor
    """
    N = len(features)
    D = features[0].shape[-1]
    padded = np.zeros((N, max_len, D), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int32)

    for i, f in enumerate(features):
        T = min(len(f), max_len)
        padded[i, :T] = f[:T]
        lengths[i] = T

    return torch.from_numpy(padded), torch.from_numpy(lengths)


# ---------------------------------------------------------------------------
# Load evaluator
# ---------------------------------------------------------------------------

def load_evaluator(momask_root: str, device: torch.device):
    """Load MoMask EvaluatorWrapper from checkpoints/t2m/text_mot_match/."""
    _add_momask_to_path(momask_root)
    from models.t2m_eval_wrapper import EvaluatorWrapper

    ckpt_path = os.path.join(
        momask_root, "checkpoints", "t2m", "text_mot_match", "model", "finest.tar"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Evaluator checkpoint not found: {ckpt_path}\n"
            f"Download with:\n"
            f"  cd {momask_root} && bash prepare/download_evaluator.sh"
        )

    wrapper = EvaluatorWrapper("humanml", device)
    return wrapper


# ---------------------------------------------------------------------------
# Text encoding
# ---------------------------------------------------------------------------

def load_text_encoder_and_vectorizer(momask_root: str, device: torch.device):
    """
    Load the text encoder from MoMask evaluator + GloVe word vectorizer.
    Returns (text_encoder, word_vectorizer) or (None, None) if unavailable.
    """
    _add_momask_to_path(momask_root)

    glove_candidates = [
        os.path.join(momask_root, "dataset", "HumanML3D"),
        os.path.join(momask_root, "dataset", "t2m"),
    ]
    glove_root = None
    for p in glove_candidates:
        if os.path.exists(os.path.join(p, "glove_data.npy")):
            glove_root = p
            break

    if glove_root is None:
        print("  [Warning] GloVe vectors not found; R-Precision will be skipped")
        return None, None

    try:
        from utils.word_vectorizer import WordVectorizer
        wv = WordVectorizer(glove_root, "glove")
        print(f"  GloVe loaded from: {glove_root}")
        return wv
    except Exception as e:
        print(f"  [Warning] WordVectorizer load failed: {e}")
        return None


def _encode_text(text: str, word_vectorizer, max_text_len: int = 20):
    """
    Tokenize and encode a single text string.
    Returns (word_embs, pos_ohot, cap_len) or None.
    """
    if word_vectorizer is None:
        return None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text.lower())
        tokens = [token.text for token in doc]
        word_embs, pos_ohot = word_vectorizer.get_vectors_from_tokens(tokens, max_text_len)
        cap_len = min(len(tokens), max_text_len)
        return word_embs, pos_ohot, cap_len
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# FID computation
# ---------------------------------------------------------------------------

def compute_fid_from_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute FID between two sets of embeddings (N1, D) and (N2, D)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "momask-codes"))
    from scipy import linalg

    mu1, sigma1 = emb1.mean(0), np.cov(emb1, rowvar=False)
    mu2, sigma2 = emb2.mean(0), np.cov(emb2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fid


# ---------------------------------------------------------------------------
# R-Precision
# ---------------------------------------------------------------------------

def compute_r_precision(
    text_embs: np.ndarray,    # (N, D)
    motion_embs: np.ndarray,  # (N, D)
    top_k: int = 3,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute R-Precision@k.

    For each motion, checks if its paired text is in the top-k
    most similar texts (out of a random batch of batch_size).
    Returns array of top-1, top-2, ..., top-k precision values.
    """
    N = len(motion_embs)
    rng = np.random.default_rng(42)
    hits = np.zeros(top_k)
    count = 0

    for i in range(N):
        # Build a batch: 1 positive + (batch_size-1) negatives
        neg_idx = rng.choice([j for j in range(N) if j != i],
                             size=min(batch_size - 1, N - 1), replace=False)
        all_idx = [i] + list(neg_idx)
        t_batch = text_embs[all_idx]    # (batch_size, D)
        m_q     = motion_embs[i:i+1]   # (1, D)

        # Cosine similarity
        t_norm = t_batch / (np.linalg.norm(t_batch, axis=-1, keepdims=True) + 1e-8)
        m_norm = m_q / (np.linalg.norm(m_q, axis=-1, keepdims=True) + 1e-8)
        sims = (t_norm @ m_norm.T).squeeze(-1)  # (batch_size,)

        # Rank of positive (index 0)
        rank = int((sims > sims[0]).sum())  # number of items ranked higher than positive
        for k in range(top_k):
            if rank <= k:
                hits[k] += 1
        count += 1

    return hits / count   # (top_k,)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_fid_eval(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    momask_root = os.path.abspath(args.momask_root)
    _add_momask_to_path(momask_root)

    # --- Load results metadata ---
    with open(args.results_json) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} prompt results from {args.results_json}")

    # --- Load evaluator ---
    print("\nLoading MoMask evaluator...")
    try:
        evaluator = load_evaluator(momask_root, device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # --- Load normalization stats ---
    mean, std = _load_mean_std(momask_root)

    # --- Get reference skeleton offsets ---
    # Load first available .npy to get a reference frame
    first_npy = os.path.join(args.npy_dir, "0001_baseline.npy")
    ref_joints = np.load(first_npy)  # (B, T, 22, 3)
    print("\nComputing reference skeleton offsets...")
    tgt_offsets = _get_reference_offsets(momask_root, ref_joints[0, 0])

    # --- Load word vectorizer (optional) ---
    print("\nLoading text encoder (optional for R-Precision)...")
    word_vectorizer = load_text_encoder_and_vectorizer(momask_root, device)

    # --- Process all samples ---
    print(f"\nConverting {len(results)} prompt × (baseline + steered) to 263D features...")
    baseline_feats, steered_feats = [], []
    texts = []
    skipped = 0

    for r in results:
        idx = r["idx"]
        tag = f"{idx:04d}"
        b_path = os.path.join(args.npy_dir, f"{tag}_baseline.npy")
        s_path = os.path.join(args.npy_dir, f"{tag}_steered.npy")

        if not (os.path.exists(b_path) and os.path.exists(s_path)):
            print(f"  [{idx}] Missing .npy, skipping")
            skipped += 1
            continue

        b_joints = np.load(b_path)  # (B, T, 22, 3)
        s_joints = np.load(s_path)

        b_feats, b_lens = batch_joints_to_263d(b_joints, momask_root, tgt_offsets)
        s_feats, s_lens = batch_joints_to_263d(s_joints, momask_root, tgt_offsets)

        if not b_feats or not s_feats:
            skipped += 1
            continue

        # Normalize
        b_feats = normalize_features(b_feats, mean, std)
        s_feats = normalize_features(s_feats, mean, std)

        baseline_feats.extend(b_feats)
        steered_feats.extend(s_feats)
        # One text per prompt (repeated for each seed)
        texts.extend([r["prompt"]] * len(b_feats))

        if idx % 10 == 0:
            print(f"  [{idx}/{len(results)}] OK — {len(b_feats)} samples each")

    print(f"\nConverted: {len(baseline_feats)} baseline, {len(steered_feats)} steered")
    print(f"Skipped: {skipped} prompts")

    if len(baseline_feats) < 10:
        print("ERROR: Too few samples for meaningful FID. Exiting.")
        sys.exit(1)

    # --- Pad to max_len and get embeddings ---
    max_len = 196
    b_padded, b_lens_t = pad_to_batch(baseline_feats, max_len)
    s_padded, s_lens_t = pad_to_batch(steered_feats,  max_len)

    print("\nComputing motion embeddings...")
    CHUNK = 32

    def get_embeddings(padded_t, lens_t):
        all_embs = []
        N = padded_t.shape[0]
        for i in range(0, N, CHUNK):
            m = padded_t[i:i+CHUNK].to(device)
            l = lens_t[i:i+CHUNK]
            emb = evaluator.get_motion_embeddings(m, l)
            all_embs.append(emb.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    baseline_embs = get_embeddings(b_padded, b_lens_t)
    steered_embs  = get_embeddings(s_padded, s_lens_t)
    print(f"  Embedding shapes: {baseline_embs.shape}, {steered_embs.shape}")

    # --- FID ---
    print("\nComputing FID...")
    fid_score = compute_fid_from_embeddings(baseline_embs, steered_embs)
    print(f"  FID(baseline || steered) = {fid_score:.4f}")
    print("  (Lower = steering preserves distribution better)")

    # --- R-Precision (if text encoder available) ---
    rprec_baseline = rprec_steered = None
    if word_vectorizer is not None:
        print("\nComputing R-Precision...")
        try:
            # Encode all texts
            text_embs_list = []
            for text in texts:
                enc = _encode_text(text, word_vectorizer)
                if enc is not None:
                    w, p, l = enc
                    t_emb = evaluator.text_encoder(
                        torch.from_numpy(w).unsqueeze(0).to(device),
                        torch.from_numpy(p).unsqueeze(0).to(device),
                        torch.tensor([l]),
                    )
                    text_embs_list.append(t_emb.cpu().detach().numpy()[0])
                else:
                    text_embs_list.append(np.zeros(512))

            text_embs = np.stack(text_embs_list)

            rprec_baseline = compute_r_precision(text_embs, baseline_embs, top_k=3)
            rprec_steered  = compute_r_precision(text_embs, steered_embs,  top_k=3)

            print(f"  Baseline  R@1={rprec_baseline[0]:.4f}  R@3={rprec_baseline[2]:.4f}")
            print(f"  Steered   R@1={rprec_steered[0]:.4f}  R@3={rprec_steered[2]:.4f}")
        except Exception as e:
            print(f"  R-Precision failed: {e}")

    # --- Save results ---
    out = {
        "n_baseline":   len(baseline_embs),
        "n_steered":    len(steered_embs),
        "fid_baseline_vs_steered": round(fid_score, 6),
        "r_precision_baseline": (rprec_baseline.tolist() if rprec_baseline is not None else None),
        "r_precision_steered":  (rprec_steered.tolist()  if rprec_steered  is not None else None),
    }

    out_path = args.output_json or os.path.join(args.npy_dir, "fid_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFID results saved: {out_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Phase 2 Summary")
    print(f"{'='*60}")
    print(f"  Samples:         {len(baseline_embs)} baseline / {len(steered_embs)} steered")
    print(f"  FID(B‖S):       {fid_score:.4f}  (↓ = steering preserves distribution)")
    if rprec_baseline is not None:
        print(f"  R@1  B={rprec_baseline[0]:.4f}  S={rprec_steered[0]:.4f}  "
              f"Δ={(rprec_steered[0]-rprec_baseline[0]):+.4f}")
        print(f"  R@3  B={rprec_baseline[2]:.4f}  S={rprec_steered[2]:.4f}  "
              f"Δ={(rprec_steered[2]-rprec_baseline[2]):+.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FID / R-Precision for FlowSteer-Motion")
    parser.add_argument("--npy_dir",        required=True,
                        help="Directory with *_baseline.npy / *_steered.npy files")
    parser.add_argument("--results_json",   default=None,
                        help="results_summary.json from run_eval.py (for prompt texts)")
    parser.add_argument("--momask_root",    required=True,
                        help="Root directory of momask-codes (contains utils/, models/, checkpoints/)")
    parser.add_argument("--output_json",    default=None)
    parser.add_argument("--gpu_id",         type=int, default=0)
    args = parser.parse_args()

    # Auto-find results_json if not specified
    if args.results_json is None:
        args.results_json = os.path.join(args.npy_dir, "results_summary.json")

    run_fid_eval(args)


if __name__ == "__main__":
    main()
