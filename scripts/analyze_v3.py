"""Quick analysis of sweep v3 results."""
import pandas as pd

df = pd.read_csv("logs/master_sweep_v3.csv")
ok = df[df["Status"] == "OK"]

profitable = ok[ok["Return %"] > 0]
print(f"*** FULL RESULTS: {len(ok)} combos ***")
print(f"Profitable: {len(profitable)} ({len(profitable)/len(ok)*100:.1f}%)")
print(f"Avg Return: {ok['Return %'].mean():.4f}%  Median: {ok['Return %'].median():.4f}%")
print()

# TOP 30 by Sharpe (min 5 trades)
with_trades = ok[ok["Trades"] >= 5]
top30 = with_trades.nlargest(30, "Sharpe")
print("TOP 30 BY SHARPE (min 5 trades):")
header = f"{'Strategy':25s} {'Symbol':12s} {'TF':5s} {'Trades':>6s} {'Win%':>6s} {'Return%':>10s} {'DD%':>8s} {'Sharpe':>10s} {'PF':>8s}"
print(header)
print("-" * len(header))
for _, r in top30.iterrows():
    print(f"{r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
          f"{int(r['Trades']):6d} {r['Win Rate %']:6.1f} {r['Return %']:10.4f} "
          f"{r['Max DD %']:8.4f} {r['Sharpe']:10.4f} {r['Profit Factor']:8.4f}")

print()

# By strategy category
ML = {"XGBoost", "LightGBM", "LSTM"}
HF = {"MicroMomentum", "MeanReversion_HF", "OrderFlow_Imbalance", "Breakout_Micro"}
ALT = {"RegimeAdaptive", "CrossTF_Momentum", "Vol_Breakout", "StatArb", "AdaptiveTrend"}

print("BY CATEGORY:")
for cat_name, cat_set in [("ML", ML), ("HF", HF), ("Alt-Alpha", ALT)]:
    cat = ok[ok["Strategy"].isin(cat_set)]
    cat_prof = cat[cat["Return %"] > 0]
    if len(cat) > 0:
        print(f"  {cat_name:12s}: {len(cat_prof):3d}/{len(cat):3d} profitable "
              f"| Avg: {cat['Return %'].mean():+.4f}% "
              f"| Best Sharpe: {cat['Sharpe'].max():.2f} "
              f"| Avg trades: {cat['Trades'].mean():.0f}")

print()
print("PER STRATEGY AVG (sorted by Sharpe):")
header2 = f"{'Strategy':25s} {'Return%':>10s} {'Sharpe':>10s} {'Trades':>8s} {'Win%':>6s} {'Prof':>8s}"
print(header2)
print("-" * len(header2))
grp = ok.groupby("Strategy").agg({"Return %": "mean", "Sharpe": "mean", "Trades": "mean", "Win Rate %": "mean"})
grp = grp.sort_values("Sharpe", ascending=False)
for name, row in grp.iterrows():
    n_prof = len(ok[(ok["Strategy"] == name) & (ok["Return %"] > 0)])
    n_total = len(ok[ok["Strategy"] == name])
    print(f"{name:25s} {row['Return %']:+10.4f} {row['Sharpe']:+10.4f} "
          f"{row['Trades']:8.0f} {row['Win Rate %']:6.1f} {n_prof:3d}/{n_total:3d}")

print()

# Dollar returns for top 10 combos
print("DOLLAR RETURNS (on $10,000):")
top10 = with_trades.nlargest(10, "Sharpe")
for _, r in top10.iterrows():
    dollars = r["Return %"] / 100 * 10000
    annual = r["Return %"] * 4  # approximate annualization from 90-day window
    print(f"  {r['Strategy']:25s} {r['Symbol']:12s} {r['Interval']:5s} "
          f"-> ${dollars:+.0f} ({r['Return %']:+.4f}%) ~{annual:+.1f}% annual "
          f"| {int(r['Trades'])} trades | Sharpe {r['Sharpe']:.2f}")
