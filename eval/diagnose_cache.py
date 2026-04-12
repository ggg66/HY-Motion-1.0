"""
Diagnose whether steered motions in cache differ from baselines.

Usage:
    python eval/diagnose_cache.py
    python eval/diagnose_cache.py --cache_dir output/paper_figures/cache
"""
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="output/paper_figures/cache")
    args = parser.parse_args()

    cache = args.cache_dir
    if not os.path.isdir(cache):
        print(f"Cache dir not found: {cache}")
        return

    files = sorted(os.listdir(cache))
    print(f"\n{'='*70}")
    print(f"  Cache: {cache}  ({len(files)} files)")
    print(f"{'='*70}")
    print(f"  {'filename':<55}  shape       mean     std")
    print(f"  {'-'*55}  ----------  -------  -------")
    for f in files:
        if not f.endswith(".npy"):
            continue
        arr = np.load(os.path.join(cache, f))
        print(f"  {f[:55]:<55}  {str(arr.shape):<10}  {arr.mean():+.4f}  {arr.std():.4f}")

    # --- pairwise base vs steer comparison ---
    print(f"\n{'='*70}")
    print("  Base vs Steer diff (max / mean absolute difference)")
    print(f"{'='*70}")
    print(f"  {'prompt prefix':<40}  {'max_diff':>9}  {'mean_diff':>10}  status")
    print(f"  {'-'*40}  {'-'*9}  {'-'*10}  ------")

    # find all base43 files and look for matching steer43_full
    any_pair = False
    for f in files:
        if not f.endswith("_base43.npy"):
            continue
        prefix = f[:-len("_base43.npy")]
        steer_f = prefix + "_steer43_full.npy"
        bpath = os.path.join(cache, f)
        spath = os.path.join(cache, steer_f)
        if not os.path.exists(spath):
            print(f"  {prefix[-40:]:<40}  {'N/A':>9}  {'N/A':>10}  steer file missing")
            continue
        b = np.load(bpath)
        s = np.load(spath)
        if b.shape != s.shape:
            print(f"  {prefix[-40:]:<40}  {'N/A':>9}  {'N/A':>10}  shape mismatch {b.shape} vs {s.shape}")
            continue
        diff = np.abs(b - s)
        mx   = diff.max()
        mn   = diff.mean()
        status = "OK (differ)" if mx > 1e-4 else "*** IDENTICAL — steering may not have worked ***"
        print(f"  {prefix[-40:]:<40}  {mx:>9.4f}  {mn:>10.4f}  {status}")
        any_pair = True

    if not any_pair:
        print("  No base43 / steer43_full pairs found in cache.")

    print()

if __name__ == "__main__":
    main()
