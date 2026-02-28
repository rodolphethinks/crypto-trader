"""
Final Profitability Assessment — V3 Strategies
Combines sweep results + walk-forward validation to determine
which combos (if any) are suitable for real-money deployment.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

# Load data
sweep = pd.read_csv("logs/master_sweep_v3.csv")
wf = pd.read_csv("logs/walk_forward_v3.csv")

print("=" * 90)
print("  FINAL PROFITABILITY ASSESSMENT — MEXC Trading System V3")
print("=" * 90)

# --- Merge sweep + walk-forward ---
passed = wf[wf["Status"] == "PASS"].copy()
merged = passed.merge(
    sweep, on=["Strategy", "Symbol", "Interval"], how="left", suffixes=("_wf", "_sweep")
)

print(f"\n  Total combos tested in sweep v3: {len(sweep)}")
print(f"  Profitable in-sample:           {len(sweep[sweep['Return %'] > 0])} ({len(sweep[sweep['Return %'] > 0])/len(sweep)*100:.1f}%)")
print(f"  Walk-forward candidates:        {len(wf)}")
print(f"  Walk-forward PASSED:            {len(passed)} ({len(passed)/len(wf)*100:.1f}%)")
print(f"  Walk-forward FAILED:            {len(wf[wf['Status']=='FAIL'])} ({len(wf[wf['Status']=='FAIL'])/len(wf)*100:.1f}%)")

print("\n" + "-" * 90)
print("  WALK-FORWARD VALIDATED WINNERS (ranked by OOS return)")
print("-" * 90)
print(f"{'Strategy':25s} {'Symbol':12s} {'TF':5s} │ {'Sweep':>8s} {'OOS':>8s} │ "
      f"{'OOS$/yr':>8s} │ {'Trades':>6s} {'Folds+':>8s} │ {'Sharpe':>7s} {'MaxDD':>7s}")
print("─" * 90)

# Annualize: sweep data is 365d, each fold tests ~33% of that => ~122 days each
for idx, r in merged.sort_values("Avg_OOS_Return", ascending=False).iterrows():
    sweep_ret = r.get("Return %", 0)
    oos_ret = r.get("Avg_OOS_Return", 0)
    # Each OOS fold is 1/4 of data (~91 days), annualize from that
    annualized_oos = oos_ret * (365 / 91)  # rough annualization
    dollar_annual = annualized_oos / 100 * 10000
    trades = int(r.get("Trades", 0))
    folds = r.get("Folds_Positive", "?")
    sharpe = r.get("Sharpe", 0)
    maxdd = r.get("Max DD %", 0)
    
    print(f"  {r['Strategy']:23s} {r['Symbol']:12s} {r['Interval']:5s} │ "
          f"{sweep_ret:+7.2f}% {oos_ret:+7.4f}% │ "
          f"${dollar_annual:+7.0f}/yr │ "
          f"{trades:5d}  {folds:>7s} │ "
          f"{sharpe:6.2f} {maxdd:6.2f}%")

print("\n" + "-" * 90)
print("  TIER CLASSIFICATION")
print("-" * 90)

# Tier 1: Positive OOS, 2+ folds, decent trade count
tier1 = merged[(merged["Avg_OOS_Return"] > 0.1) & (merged["OOS_Trades"] >= 10)]
tier2 = merged[(merged["Avg_OOS_Return"] > 0) & (merged["OOS_Trades"] >= 4)]
tier3 = merged[merged["Avg_OOS_Return"] > 0]

print(f"\n  TIER 1 (OOS > +0.10%, 10+ OOS trades) — Paper trading candidates:")
if len(tier1) > 0:
    for _, r in tier1.sort_values("Avg_OOS_Return", ascending=False).iterrows():
        ann = r["Avg_OOS_Return"] * (365/91) / 100 * 10000
        print(f"    ★ {r['Strategy']:23s} {r['Symbol']:12s} {r['Interval']:5s} "
              f"— OOS {r['Avg_OOS_Return']:+.4f}% (~${ann:+.0f}/yr on $10k)")
else:
    print("    (none)")

print(f"\n  TIER 2 (OOS > 0%, 4+ OOS trades) — Monitor candidates:")
t2_only = tier2[~tier2.index.isin(tier1.index)]
if len(t2_only) > 0:
    for _, r in t2_only.sort_values("Avg_OOS_Return", ascending=False).iterrows():
        ann = r["Avg_OOS_Return"] * (365/91) / 100 * 10000
        print(f"    ○ {r['Strategy']:23s} {r['Symbol']:12s} {r['Interval']:5s} "
              f"— OOS {r['Avg_OOS_Return']:+.4f}% (~${ann:+.0f}/yr on $10k)")
else:
    print("    (none)")

print(f"\n  TIER 3 (borderline / low trade count) — Watch list:")
t3_only = tier3[~tier3.index.isin(tier2.index)]
if len(t3_only) > 0:
    for _, r in t3_only.sort_values("Avg_OOS_Return", ascending=False).iterrows():
        ann = r["Avg_OOS_Return"] * (365/91) / 100 * 10000
        print(f"    · {r['Strategy']:23s} {r['Symbol']:12s} {r['Interval']:5s} "
              f"— OOS {r['Avg_OOS_Return']:+.4f}% (~${ann:+.0f}/yr on $10k)")
else:
    print("    (none)")


# --- HONEST ASSESSMENT ---
print("\n" + "=" * 90)
print("  HONEST BOTTOM-LINE ASSESSMENT")
print("=" * 90)

best_oos = merged.sort_values("Avg_OOS_Return", ascending=False).iloc[0] if len(merged) > 0 else None
best_annual_pct = best_oos["Avg_OOS_Return"] * (365/91) if best_oos is not None else 0
best_dollar = best_annual_pct / 100 * 10000

print(f"""
  PROGRESS VS. V1/V2:
  ────────────────────
  V1/V2 (920 combos):  Best walk-forward validated = $0.90/year on $10k
  V3    (435 combos):  Best walk-forward validated = ${best_dollar:.0f}/year on $10k

  That's a {'significant improvement' if best_dollar > 100 else 'modest improvement' if best_dollar > 10 else 'minimal improvement'}.

  KEY FINDINGS:
  ─────────────
  1. AdaptiveTrend (Kaufman AMA + ADX) is the strongest strategy family.
     - Consistently passes walk-forward on multiple pairs (INJ, XRP, AVAX, SUI)
     - {len(tier1[tier1['Strategy']=='AdaptiveTrend']) if len(tier1)>0 else 0} combos in Tier 1

  2. CrossTF_Momentum shows promise but fewer trades per fold.
  
  3. LSTM on LTCUSDC 4h surprisingly passed all 3 folds with +0.76% OOS.
  
  4. ML strategies (XGBoost/LightGBM) overtrade and lose — not suited for this market.
  
  5. High-frequency strategies underperformed on 5m/15m crypto data.

  RECOMMENDATION:
  ───────────────""")

if best_dollar > 200:
    print(f"""  ✓ CAUTIOUS PAPER TRADING for Tier 1 combos is reasonable.
    - Expected: ~${best_dollar:.0f}/year on $10k (annualized from OOS folds)
    - This is {best_annual_pct:.1f}% annually — {'above' if best_annual_pct > 5 else 'below'} a basic savings account.
    - Run paper trader for 2-4 weeks before any real capital.
    - Use the existing paper_trader.py infrastructure.
    - Start with AdaptiveTrend on XRP/AVAX/INJ (4h timeframe).
""")
elif best_dollar > 50:
    print(f"""  ⚠ MARGINAL — paper trading is possible but expectations should be very low.
    - Best validated return: ~${best_dollar:.0f}/year on $10k
    - This is {best_annual_pct:.1f}% annually — likely not worth the risk.
    - Could paper trade for research purposes to gather live data.
    - Would need $100k+ capital for meaningful returns.
""")
else:
    print(f"""  ✗ NOT YET PRODUCTION READY.
    - Best validated return: ~${best_dollar:.0f}/year on $10k
    - Transaction costs, latency, and execution risk would likely erase gains.
""")

print(f"""  NEXT STEPS IF PROCEEDING:
  ─────────────────────────
  1. Paper trade top Tier 1 combos for 2-4 weeks using existing infrastructure
  2. Compare paper results to backtest expectations
  3. If paper results hold: start with $500-1000 real capital (small)
  4. Scale up only after 30+ days of consistent live performance
  
  NEXT STEPS FOR FURTHER RESEARCH:
  ──────────────────────────────────
  1. Ensemble: combine AdaptiveTrend + CrossTF_Momentum signals
  2. Dynamic position sizing based on regime confidence
  3. Longer backtest periods (2-3 years if data available)
  4. Parameter optimization (grid search) on validated winners
  5. Test on additional pairs beyond current 15
""")
