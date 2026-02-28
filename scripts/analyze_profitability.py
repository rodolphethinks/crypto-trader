"""Quick analysis of all results for profitability assessment."""
import pandas as pd
import os

# ── 1. Master Sweep ──────────────────────────────────────────────────────
ms = pd.read_csv("logs/master_sweep_v2.csv")
profitable = ms[ms["Return %"] > 0].sort_values("Sharpe", ascending=False)

print("=" * 100)
print("  HONEST PROFITABILITY ASSESSMENT")
print("=" * 100)

print(f"\n--- MASTER SWEEP (920 backtests, 90 days, $10k capital, 1%/trade risk) ---")
print(f"  Total combos: {len(ms)}")
print(f"  Profitable:   {len(profitable)} ({len(profitable)/len(ms)*100:.1f}%)")
print(f"  Avg return:   {ms['Return %'].mean():+.4f}%")
print(f"  Median return:{ms['Return %'].median():+.4f}%")

print(f"\nTop 15 profitable by Sharpe:")
cols = ["Strategy", "Symbol", "Interval", "Trades", "Win Rate %",
        "Return %", "Max DD %", "Sharpe", "Profit Factor"]
print(profitable[cols].head(15).to_string(index=False))

# Dollar returns on $10k over 90 days
print(f"\n--- DOLLAR RETURNS (on $10,000 over 90 days) ---")
for _, r in profitable.head(10).iterrows():
    dollar = 10000 * r["Return %"] / 100
    annual = r["Return %"] * (365 / 90)
    print(f"  {r['Strategy']:25s} {r['Symbol']:10s} {r['Interval']:4s} "
          f"-> ${dollar:+.2f} ({r['Return %']:+.4f}%) = {annual:+.2f}% annualized "
          f"| {int(r['Trades'])} trades | DD {r['Max DD %']:.2f}%")

# ── 2. Grid Search Best ─────────────────────────────────────────────────
print(f"\n--- GRID SEARCH OPTIMIZED (best params found) ---")
gs = pd.read_csv("logs/grid_search_results.csv")
gs_top = gs.nlargest(10, "sharpe")
for _, r in gs_top.iterrows():
    dollar = 10000 * r["return_pct"] / 100
    annual = r["return_pct"] * (365 / 90)
    print(f"  {r['strategy']:12s} {r['symbol']:10s} {r['interval']:4s} "
          f"-> ${dollar:+.2f} ({r['return_pct']:+.4f}%) = {annual:+.2f}% annualized "
          f"| {int(r['trades'])} trades | Sharpe {r['sharpe']:.2f}")

# ── 3. Walk-Forward (the real test) ─────────────────────────────────────
print(f"\n--- WALK-FORWARD VALIDATION (out-of-sample truth) ---")
wf = pd.read_csv("logs/walk_forward_results.csv")
wf_agg = wf.groupby(["strategy", "symbol", "interval"]).agg({
    "train_return": "mean",
    "test_return": "mean",
    "train_sharpe": "mean",
    "test_sharpe": "mean",
    "train_trades": "sum",
    "test_trades": "sum",
}).round(4)

for idx, r in wf_agg.iterrows():
    strat, sym, intv = idx
    oos_dollar = 10000 * r["test_return"] / 100
    status = "PASS" if r["test_return"] > 0 else "FAIL"
    overfit = "OVERFIT" if r["train_sharpe"] > 0 and r["test_sharpe"] < -2 else "OK"
    print(f"  {strat:18s} {sym:10s} {intv:4s} "
          f"| Train: {r['train_return']:+.4f}% (Sharpe {r['train_sharpe']:+.2f}) "
          f"| Test: {r['test_return']:+.4f}% (Sharpe {r['test_sharpe']:+.2f}) "
          f"| OOS ${oos_dollar:+.2f} | {status} {overfit}")

# ── 4. Verdict ──────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("  VERDICT")
print("=" * 100)
best_return = profitable.iloc[0]["Return %"] if len(profitable) > 0 else 0
best_dollar = 10000 * best_return / 100
print(f"""
  The BEST combo in the entire sweep (BB_MACD on DOGEUSDC 4h) returned
  +{best_return:.4f}% over 90 days = ${best_dollar:.2f} on $10,000.

  That's about ${best_dollar*365/90:.2f}/year on $10k, or {best_return*365/90:.2f}% annualized.

  BUT the walk-forward test shows BB_MACD DOGEUSDC is OVERFIT:
    - In-sample:  +0.138% (Sharpe 6.94)
    - Out-sample: -0.060% (Sharpe -8.88)  <-- LOSES money on unseen data

  The ONLY combo that PASSED walk-forward with NO overfit:
    BB_MACD AVAXUSDC 4h:
    - In-sample:  +0.101% 
    - Out-sample: +0.009% (tiny but positive)
    - That's about $0.90 on $10k over the test period.

  77% of all strategy/pair/interval combos are UNPROFITABLE.
  Average return across everything: {ms['Return %'].mean():+.4f}%

  BOTTOM LINE: No strategy here is ready for real money.
  The returns are too small (~$1-$19 per $10k over 90 days), the edge
  is razor-thin, most don't survive out-of-sample testing, and
  transaction costs / slippage in real trading would likely erase
  what little edge exists.
""")
