"""
Read D5 results and rank prompts by visual distinctiveness for paper figures.

Visual quality score = e_base * improvement_pct
  - high e_base  → baseline is clearly "wrong"
  - high improvement → ours is clearly "better"
  - product maximises the absolute gap visible in the figure

Usage:
    python eval/pick_best_cases.py
    python eval/pick_best_cases.py --results_dir output
"""
import argparse, json, math, os

def load(path):
    with open(path) as f:
        return json.load(f)

def stats(rows, key):
    v = [r[key] for r in rows if not math.isnan(r.get(key, float('nan')))]
    if not v: return float('nan'), float('nan')
    n = len(v); m = sum(v)/n
    s = (sum((x-m)**2 for x in v)/max(n-1,1))**0.5
    return m, s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="output")
    args = parser.parse_args()

    # Try to load D5 combined results (seeds 43+44)
    candidates = [
        os.path.join(args.results_dir, "ablation_pose_D5_a6_s4344", "results.json"),
        os.path.join(args.results_dir, "ablation_pose_D5_a6",       "results.json"),
        os.path.join(args.results_dir, "ablation_pose_D5",          "results.json"),
    ]
    # Also try to combine D5_s44 with D5_a6
    d5_s43_path = os.path.join(args.results_dir, "ablation_pose_D5_a6",    "results.json")
    d5_s44_path = os.path.join(args.results_dir, "ablation_pose_D5_s44",   "results.json")

    rows = []
    for path in candidates:
        if os.path.exists(path):
            rows = load(path)
            print(f"Loaded: {path}  ({len(rows)} rows)")
            break

    if not rows and os.path.exists(d5_s43_path) and os.path.exists(d5_s44_path):
        rows = load(d5_s43_path) + load(d5_s44_path)
        print(f"Combined s43+s44: {len(rows)} rows")

    if not rows:
        print("No D5 results found. Tried:")
        for p in candidates: print(f"  {p}")
        return

    # Group by prompt
    by_prompt = {}
    for r in rows:
        p = r["prompt"]
        by_prompt.setdefault(p, []).append(r)

    # Compute per-prompt stats
    records = []
    for prompt, rlist in by_prompt.items():
        imp_m, imp_s = stats(rlist, "pose_hit_improvement_pct")
        eb_m,  _     = stats(rlist, "pose_hit_baseline")
        es_m,  _     = stats(rlist, "pose_hit_steered")
        jr_m,  jr_s  = stats(rlist, "jerk_ratio")
        variance     = rlist[0].get("variance", "?")
        n            = len(rlist)
        # visual score: how large is the ABSOLUTE improvement in metres?
        abs_impr_m   = eb_m - es_m          # metres closer to target
        visual_score = abs_impr_m * (imp_m / 100.0)   # weighted by relative improvement
        records.append(dict(
            prompt=prompt, variance=variance, n=n,
            eb=eb_m, es=es_m,
            imp=imp_m, imp_s=imp_s,
            abs_impr=abs_impr_m,
            jr=jr_m, jr_s=jr_s,
            score=visual_score,
        ))

    records.sort(key=lambda r: r["score"], reverse=True)

    print()
    print("="*100)
    print("  Prompts ranked by visual distinctiveness  (score = abs_improvement * rel_improvement)")
    print("="*100)
    print(f"  {'prompt':<52} {'var':>4}  {'e_base':>7} {'e_steer':>7} {'abs_impr':>9} {'rel_impr':>9} {'jerk':>6}  score")
    print(f"  {'-'*52} {'-'*4}  {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*6}  -----")
    for i, r in enumerate(records):
        marker = " <-- TOP PICK" if i < 3 else ""
        print(f"  {r['prompt'][:52]:<52} {r['variance']:>4}  "
              f"{r['eb']:>7.4f} {r['es']:>7.4f} {r['abs_impr']:>+8.4f}m "
              f"{r['imp']:>+8.1f}%  {r['jr']:>6.3f}  "
              f"{r['score']:.4f}{marker}")

    print()
    print("  TOP 5 recommended for paper figures:")
    print(f"  {'#':<3} {'prompt':<52} {'var':>4}  {'abs_impr':>9} {'rel_impr':>9} {'jerk':>6}")
    print(f"  {'-'*3} {'-'*52} {'-'*4}  {'-'*9} {'-'*9} {'-'*6}")
    for i, r in enumerate(records[:5]):
        print(f"  {i+1:<3} {r['prompt'][:52]:<52} {r['variance']:>4}  "
              f"{r['abs_impr']:>+8.4f}m {r['imp']:>+8.1f}%  {r['jr']:>6.3f}")
    print()

if __name__ == "__main__":
    main()
