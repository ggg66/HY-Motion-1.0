"""Print complete experimental data summary."""
import json, math, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load(path):
    with open(path) as f:
        return json.load(f)

def stats(rows, key):
    v = [r[key] for r in rows if not math.isnan(r.get(key, float('nan')))]
    if not v:
        return float('nan'), float('nan'), 0
    n = len(v); m = sum(v) / n
    s = (sum((x - m) ** 2 for x in v) / max(n - 1, 1)) ** 0.5
    return m, s, n

BASE = 'output'
d1 = {
    '0.5': load(f'{BASE}/ablation_pose_D1_a0/results.json'),
    '1.0': load(f'{BASE}/ablation_pose_D1_a1/results.json'),
    '2.0': load(f'{BASE}/ablation_pose_D1_a2/results.json'),
    '4.0': load(f'{BASE}/ablation_pose_D1_a4/results.json'),
    '8.0': load(f'{BASE}/ablation_pose_D1_a8/results.json'),
}
d1_a6   = load(f'{BASE}/ablation_pose_D1_a6_s4344/results.json')
d2      = load(f'{BASE}/ablation_pose_D2/results.json')
d3      = load(f'{BASE}/ablation_pose_D3/results.json')
d5 = {
    '2.0':  load(f'{BASE}/ablation_pose_D5_a2/results.json'),
    '4.0':  load(f'{BASE}/ablation_pose_D5_a4/results.json'),
    '6.0':  load(f'{BASE}/ablation_pose_D5_a6/results.json'),
    '8.0':  load(f'{BASE}/ablation_pose_D5_a8/results.json'),
    '12.0': load(f'{BASE}/ablation_pose_D5_a12/results.json'),
}
d5_s44  = load(f'{BASE}/ablation_pose_D5_s44/results.json')
d5_comb = d5['6.0'] + d5_s44


def pr(label, rows, note='', main=False):
    pi, ps, n = stats(rows, 'pose_hit_improvement_pct')
    jr, js, _ = stats(rows, 'jerk_ratio')
    kv, _,  _ = stats(rows, 'kinvar_ratio')
    eb, _,  _ = stats(rows, 'pose_hit_baseline')
    es, _,  _ = stats(rows, 'pose_hit_steered')
    tag = '  <-- MAIN RESULT' if main else ''
    print(f'  {label:<44} {note:<14} {n:>3}  '
          f'{eb:.4f}->{es:.4f}  '
          f'{pi:+7.1f}%+/-{ps:.1f}  '
          f'{jr:.3f}+/-{js:.3f}  '
          f'{kv:.3f}{tag}')


H   = (f'  {"config":<44} {"seeds":<14} {"n":>3}  '
       f'{"err(base->steer)":>16}  {"pose_imp+/-sd":>14}  '
       f'{"jerk+/-sd":>12}  kv')
SEP = '  ' + '-' * 112

print()
print('=' * 114)
print('  COMPLETE EXPERIMENTAL DATA -- FlowSteer-Motion PoseConstraint')
print('  Cross-seed protocol: target_seed=42')
print('  pose_imp = (err_base - err_steer) / err_base * 100%')
print('  err: mean L2 (m) in canonical pose space over constrained joints')
print('=' * 114)

# ── Section A ────────────────────────────────────────────────────────────────
print()
print('  SECTION A: Historical context (same-seed protocol -- pose_imp invalid)')
print(SEP)
for cfg, note in [
    ('B3: soft-norm tau=0.1 relative, flat norm',
     'violation_rate unchanged -- tau >> grad_norm'),
    ('B4: soft-norm tau=0.001 absolute, flat norm',
     'violation_rate -4.7%, jerk neutral'),
    ('A3: unit-norm alpha=6, flat norm',
     'violation_rate -2.7%, jerk +0.9%'),
    ('C1: per-frame + median-tau + alpha=15  [BUG]',
     'CRASH: jerk +104%  (median=0 for sparse grad)'),
]:
    print(f'  {cfg:<50}  {note}')

# ── Section B ────────────────────────────────────────────────────────────────
print()
print('  SECTION B: D1 -- per-frame mean-tau, NO latent_mask, alpha sweep '
      '(seed=43, n=20/alpha)')
print(H)
print(SEP)
for a in ['0.5', '1.0', '2.0', '4.0', '8.0']:
    pr(f'D1  no_mask  alpha={a}', d1[a], 'tgt=42 s=43')
print()
pr('D1  no_mask  alpha=6.0  (2 seeds)', d1_a6, 'tgt=42 s=43+44')

# ── Section C ────────────────────────────────────────────────────────────────
print()
print('  SECTION C: Component ablation at alpha=2 (seed=43, n=20)')
print(H)
print(SEP)
pr('D1  mean-tau only              alpha=2', d1['2.0'], 'tgt=42 s=43')
pr('D2  + hierarchical weighting   alpha=2', d2,        'tgt=42 s=43')
pr('D3  + latent_mask              alpha=2', d3,        'tgt=42 s=43')

# ── Section D ────────────────────────────────────────────────────────────────
print()
print('  SECTION D: D5 -- full combo (latent_mask + hier), alpha sweep '
      '(seed=43, n=20/alpha)')
print(H)
print(SEP)
for a in ['2.0', '4.0', '6.0', '8.0', '12.0']:
    pr(f'D5  full  alpha={a}', d5[a], 'tgt=42 s=43')

# ── Section E ────────────────────────────────────────────────────────────────
print()
print('  SECTION E: Main result -- D5 alpha=6, 2 steer seeds (n=40)')
print(H)
print(SEP)
pr('D5  full  alpha=6  seed=43',     d5['6.0'], 'tgt=42 s=43')
pr('D5  full  alpha=6  seed=44',     d5_s44,    'tgt=42 s=44')
pr('D5  full  alpha=6  COMBINED',    d5_comb,   'tgt=42 s=43+44', main=True)
low  = [r for r in d5_comb if r.get('variance') == 'low']
high = [r for r in d5_comb if r.get('variance') == 'high']
pr('    low-variance  (walk/run/march)',  low,  f'n={len(low)}')
pr('    high-variance (dance/kick)',      high, f'n={len(high)}')

# ── Section F ────────────────────────────────────────────────────────────────
print()
print('  SECTION F: Latent_mask isolation -- alpha=6, with vs without (n=40)')
print(H)
print(SEP)
pr('D1  NO  latent_mask  alpha=6', d1_a6,   'tgt=42 s=43+44')
pr('D5  YES latent_mask  alpha=6', d5_comb, 'tgt=42 s=43+44')
pi1, _, _ = stats(d1_a6,   'pose_hit_improvement_pct')
pi5, _, _ = stats(d5_comb, 'pose_hit_improvement_pct')
jr1, _, _ = stats(d1_a6,   'jerk_ratio')
jr5, _, _ = stats(d5_comb, 'jerk_ratio')
kv1, _, _ = stats(d1_a6,   'kinvar_ratio')
kv5, _, _ = stats(d5_comb, 'kinvar_ratio')
print(f'  pose_imp gain : {pi1:+.1f}% --> {pi5:+.1f}%  ({pi5/max(pi1, 0.01):.1f}x)')
print(f'  jerk reduction: x{jr1:.3f} --> x{jr5:.3f}  ({(jr1-jr5)*100:.1f}pp)')
print(f'  kv   reduction: x{kv1:.3f} --> x{kv5:.3f}  ({(kv1-kv5)*100:.1f}pp)')

# ── Section G ────────────────────────────────────────────────────────────────
print()
print('  SECTION G: Anomalous prompts at alpha=6 (jerk > 1.15 in any seed)')
hdr = (f'  {"prompt":<53} {"var":>5}  '
       f'{"imp43":>7}  {"imp44":>7}  '
       f'{"jk43":>6}  {"jk44":>6}  '
       f'{"D1-jk43":>9}  {"D1-jk44":>9}  diagnosis')
print(hdr)
print('  ' + '-' * 130)
anom_kws = ['a person runs forward.', 'tai chi', 'long gun']
for r43, r44 in zip(d5['6.0'], d5_s44):
    if not any(k in r43['prompt'] for k in anom_kws):
        continue
    d1r43 = next((r for r in d1_a6
                  if r['prompt'] == r43['prompt']
                  and r.get('steer_seed', r.get('seed')) == 43), None)
    d1r44 = next((r for r in d1_a6
                  if r['prompt'] == r44['prompt']
                  and r.get('steer_seed', r.get('seed')) == 44), None)
    j1_43 = d1r43['jerk_ratio'] if d1r43 else float('nan')
    j1_44 = d1r44['jerk_ratio'] if d1r44 else float('nan')
    if 'runs forward' in r43['prompt']:
        diag = 'seed-specific; s43 fully normal across all alphas'
    elif 'tai chi' in r43['prompt']:
        diag = 'slow large-arc arms conflict with pose target'
    else:
        diag = 'functional upper-limb pose + locomotion conflict'
    print(f'  {r43["prompt"][:53]:<53} {r43["variance"]:>5}  '
          f'{r43["pose_hit_improvement_pct"]:+7.1f}%  '
          f'{r44["pose_hit_improvement_pct"]:+7.1f}%  '
          f'{r43["jerk_ratio"]:6.3f}  {r44["jerk_ratio"]:6.3f}  '
          f'{j1_43:9.3f}  {j1_44:9.3f}  {diag}')

# ── Section H ────────────────────────────────────────────────────────────────
print()
print('  SECTION H: Full per-prompt breakdown, D5 alpha=6 (n=40)')
hdr2 = (f'  {"prompt":<53} {"var":>5}  '
        f'{"imp43":>7}  {"imp44":>7}  '
        f'{"jk43":>6}  {"jk44":>6}  '
        f'{"kv43":>6}  {"kv44":>6}')
print(hdr2)
print('  ' + '-' * 110)
for r43, r44 in zip(d5['6.0'], d5_s44):
    flag = '  *' if any(k in r43['prompt'] for k in anom_kws) else ''
    print(f'  {r43["prompt"][:53]:<53} {r43["variance"]:>5}  '
          f'{r43["pose_hit_improvement_pct"]:+7.1f}%  '
          f'{r44["pose_hit_improvement_pct"]:+7.1f}%  '
          f'{r43["jerk_ratio"]:6.3f}  {r44["jerk_ratio"]:6.3f}  '
          f'{r43["kinvar_ratio"]:6.3f}  {r44["kinvar_ratio"]:6.3f}{flag}')

# ── Section I: temporal_mask ablation (loaded when P1 results exist) ─────────
print()
print('  SECTION I: P1 -- temporal_mask ablation at alpha=6 (n=40 each)')
print('  Compare: full combo WITH temporal_mask vs WITHOUT temporal_mask')
print('  (FootContactConstraint now has late-phase ramp; TerminalConstraint has tail Gaussian)')
print(H)
print(SEP)

_p1_with_path = f'{BASE}/ablation_pose_P1_with_tmask/results.json'
_p1_no_path   = f'{BASE}/ablation_pose_P1_no_tmask/results.json'
_p1_loaded = True

try:
    p1_with = load(_p1_with_path)
    p1_no   = load(_p1_no_path)
except FileNotFoundError as e:
    print(f'  [NOT YET RUN]  {e}')
    print(f'  Run: bash eval/run_ablation_D.sh  (P1_with + P1_no groups)')
    _p1_loaded = False

if _p1_loaded:
    pr('P1  WITH temporal_mask  alpha=6', p1_with, 'tgt=42 s=43+44')
    pr('P1  NO  temporal_mask  alpha=6', p1_no,   'tgt=42 s=43+44')
    pi_w, _, _ = stats(p1_with, 'pose_hit_improvement_pct')
    pi_n, _, _ = stats(p1_no,   'pose_hit_improvement_pct')
    jr_w, _, _ = stats(p1_with, 'jerk_ratio')
    jr_n, _, _ = stats(p1_no,   'jerk_ratio')
    kv_w, _, _ = stats(p1_with, 'kinvar_ratio')
    kv_n, _, _ = stats(p1_no,   'kinvar_ratio')
    print(f'  pose_imp  WITH: {pi_w:+.1f}%  NO: {pi_n:+.1f}%  '
          f'  Δpose_imp = {pi_w - pi_n:+.1f}pp')
    print(f'  jerk      WITH: x{jr_w:.3f}  NO: x{jr_n:.3f}  '
          f'  Δjerk = {(jr_w - jr_n) * 100:+.1f}pp')
    print(f'  kv        WITH: x{kv_w:.3f}  NO: x{kv_n:.3f}  '
          f'  Δkv   = {(kv_w - kv_n) * 100:+.1f}pp')
    print()
    # Low/high split
    p1w_low  = [r for r in p1_with if r.get('variance') == 'low']
    p1w_high = [r for r in p1_with if r.get('variance') == 'high']
    p1n_low  = [r for r in p1_no   if r.get('variance') == 'low']
    p1n_high = [r for r in p1_no   if r.get('variance') == 'high']
    pr('  P1 WITH  low-variance',  p1w_low,  f'n={len(p1w_low)}')
    pr('  P1 NO    low-variance',  p1n_low,  f'n={len(p1n_low)}')
    pr('  P1 WITH  high-variance', p1w_high, f'n={len(p1w_high)}')
    pr('  P1 NO    high-variance', p1n_high, f'n={len(p1n_high)}')

# ── Section J: Pending ────────────────────────────────────────────────────────
print()
print('  SECTION J: Pending experiments')
print(SEP)
for exp in [
    'P4  no_hierarchical ablation        alpha=6  seeds=43+44  (drop --use_hierarchical)',
    'P2a pose + foot_contact combo       alpha=6  seeds=43+44',
    'P2b pose + waypoint combo           alpha=6  seeds=43+44',
    'P2c pose + foot + waypoint combo    alpha=6  seeds=43+44',
]:
    print(f'  {exp}')
print()
