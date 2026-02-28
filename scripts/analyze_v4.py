"""Analyze V4 research results in detail."""
import pandas as pd
import numpy as np

sweep = pd.read_csv("logs/master_sweep_v4.csv")
wf = pd.read_csv("logs/walk_forward_v4.csv")

print("=" * 90)
print("  V4 RESEARCH — DETAILED ANALYSIS")
print("=" * 90)

print(f"\nTotal combos: {len(sweep)}")
profitable = sweep[sweep["Return %"] > 0]
print(f"Profitable: {len(profitable)} ({len(profitable)/len(sweep)*100:.1f}%)")

# By phase
for phase in sweep["Phase"].unique():
    sub = sweep[sweep["Phase"] == phase]
    prof = sub[sub["Return %"] > 0]
    print(f"\n  {phase}:")
    print(f"    Total: {len(sub)}, Profitable: {len(prof)} ({len(prof)/len(sub)*100:.1f}%)")
    print(f"    Avg Return: {sub['Return %'].mean():.4f}%, Best: {sub['Return %'].max():.4f}%")
    if len(prof) > 0:
        best = prof.sort_values("Sharpe", ascending=False).iloc[0]
        print(f"    Best: {best['Strategy']} {best['Symbol']} {best['Interval']} "
              f"Return:{best['Return %']:+.2f}% Sharpe:{best['Sharpe']:.2f}")

# Walk-forward analysis
print(f"\n\nWALK-FORWARD RESULTS ({len(wf)} candidates)")
print(f"PASSED: {len(wf[wf['status']=='PASS'])} / {len(wf)}")

# Unique validated combos
wf_passed = wf[wf["status"] == "PASS"].sort_values("oos_return", ascending=False)
print(f"\nTOP UNIQUE VALIDATED COMBOS:")
seen = set()
count = 0
for _, r in wf_passed.iterrows():
    key = f"{r['Strategy']}_{r['Symbol']}_{r['Interval']}"
    if key in seen:
        continue
    seen.add(key)
    count += 1
    if count > 15:
        break
    ann = r["oos_return"] * (365 / 91)
    dollar = ann / 100 * 10000
    print(f"  {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
          f"| OOS {r['oos_return']:+.4f}% (~${dollar:+.0f}/yr) "
          f"| Sweep {r['Sweep_Return']:+.2f}% Sharpe {r['Sweep_Sharpe']:.2f}")

# V4 strategy performance
print(f"\n\nV4 STRATEGY PERFORMANCE:")
v4_sweep = sweep[sweep["Phase"] == "V4Sweep"]
for strat in sorted(v4_sweep["Strategy"].unique()):
    sub = v4_sweep[v4_sweep["Strategy"] == strat]
    prof = sub[sub["Return %"] > 0]
    avg_sharpe = sub["Sharpe"].mean()
    avg_ret = sub["Return %"].mean()
    print(f"  {strat:25s} | {len(prof):2d}/{len(sub):2d} profitable | "
          f"Avg Return: {avg_ret:+.4f}% | Avg Sharpe: {avg_sharpe:+.2f}")

# New pairs
new_pairs = sweep[sweep["Phase"] == "V3_NewPairs"]
if len(new_pairs) > 0:
    print(f"\nV3 WINNERS ON NEW PAIRS:")
    for _, r in new_pairs.sort_values("Sharpe", ascending=False).head(10).iterrows():
        print(f"  {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
              f"| {r['Return %']:+.4f}% | Sharpe {r['Sharpe']:.2f} | {int(r['Trades'])} trades")

# Grid search best validated params
print(f"\nBEST GRID-OPTIMIZED PARAMS (walk-forward validated):")
gs_validated = wf_passed[wf_passed["Params"] != "default"].head(5)
for _, r in gs_validated.iterrows():
    ann = r["oos_return"] * (365 / 91)
    dollar = ann / 100 * 10000
    print(f"  {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
          f"OOS:{r['oos_return']:+.4f}% (~${dollar:+.0f}/yr)")
    print(f"    Params: {r['Params'][:120]}")

# Comparison with V3
print("\n" + "=" * 90)
print("  COMPARISON: V3 vs V4")
print("=" * 90)
print("""
  V1/V2:  Best WF validated = $0.90/year on $10k
  V3:     Best WF validated = $303/year on $10k  (LSTM LTCUSDC 4h)
  V4:     Best WF validated = see above
""")

if len(wf_passed) > 0:
    best = wf_passed.iloc[0]
    ann = best["oos_return"] * (365 / 91)
    dollar = ann / 100 * 10000
    print(f"  V4 best: {best['Strategy']} {best['Symbol']} {best['Interval']}")
    print(f"    OOS return: {best['oos_return']:+.4f}% per fold")
    print(f"    Annualized: ~{ann:.2f}% = ~${dollar:.0f}/year on $10k")
    print(f"    Sweep in-sample: {best['Sweep_Return']:+.2f}% (Sharpe {best['Sweep_Sharpe']:.2f})")

# Count how many unique combos beat V3 best ($303/yr)
v3_best_oos = 0.7564  # LSTM LTCUSDC from V3
beat_v3 = wf_passed[wf_passed["oos_return"] > v3_best_oos]
# unique
seen2 = set()
unique_beat = 0
for _, r in beat_v3.iterrows():
    k = f"{r['Strategy']}_{r['Symbol']}_{r['Interval']}"
    if k not in seen2:
        seen2.add(k)
        unique_beat += 1
print(f"\n  Unique combos beating V3 best (OOS > {v3_best_oos}%): {unique_beat}")
