"""Quick 5-fold WF for best V7 ML models."""
import sys, time
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from data.fetcher import DataFetcher
from strategies.v7_ml import TransformerPredStrategy, CNNMomentumStrategy, LSTMTrendStrategy
from strategies.base import Signal

fetcher = DataFetcher()

CANDIDATES = [
    ("TfmrPred", "BNBUSDC", TransformerPredStrategy),
    ("CNNMom", "BTCUSDC", CNNMomentumStrategy),
    ("LSTMTrend", "BNBUSDC", LSTMTrendStrategy),
]

INITIAL = 10000
N_FOLDS = 5

def bt(strat_cls, df_seg):
    strat = strat_cls()
    sig_df = strat.generate_signals(df_seg.copy())
    cap = INITIAL; peak = cap; pos = None
    trades = 0; wins = 0; max_dd = 0
    for i in range(len(sig_df)):
        c = df_seg["close"].iloc[i]; lo = df_seg["low"].iloc[i]; hi = df_seg["high"].iloc[i]
        sig = sig_df["signal"].iloc[i]
        sl = sig_df["stop_loss"].iloc[i] if not pd.isna(sig_df["stop_loss"].iloc[i]) else 0
        tp = sig_df["take_profit"].iloc[i] if not pd.isna(sig_df["take_profit"].iloc[i]) else 0
        if pos:
            ex = None
            if pos["s"] == "BUY":
                if pos.get("sl", 0) > 0 and lo <= pos["sl"]: ex = pos["sl"]
                elif pos.get("tp", 0) > 0 and hi >= pos["tp"]: ex = pos["tp"]
                elif sig == Signal.SELL: ex = c
            else:
                if pos.get("sl", 0) > 0 and hi >= pos["sl"]: ex = pos["sl"]
                elif pos.get("tp", 0) > 0 and lo <= pos["tp"]: ex = pos["tp"]
                elif sig == Signal.BUY: ex = c
            if ex:
                pnl = (ex - pos["e"]) * pos["q"] if pos["s"] == "BUY" else (pos["e"] - ex) * pos["q"]
                cap += pnl; trades += 1
                if pnl > 0: wins += 1
                if cap > peak: peak = cap
                dd = (peak - cap) / peak * 100
                if dd > max_dd: max_dd = dd
                pos = None
        if pos is None and sig in (Signal.BUY, Signal.SELL):
            s = "BUY" if sig == Signal.BUY else "SELL"
            if sl > 0:
                risk = cap * 0.05; pr = abs(c - sl)
                qty = min(risk / pr if pr > 0 else 0, cap / c)
            else:
                qty = cap / c
            dd = (peak - cap) / peak * 100 if peak > 0 else 0
            if qty > 0 and dd < 35:
                pos = {"s": s, "e": c, "q": qty, "sl": sl, "tp": tp}
    if pos:
        c = df_seg["close"].iloc[-1]
        pnl = (c - pos["e"]) * pos["q"] if pos["s"] == "BUY" else (pos["e"] - c) * pos["q"]
        cap += pnl
    ret = (cap - INITIAL) / INITIAL * 100
    n_wk = len(df_seg) / 42
    wkly = ret / max(n_wk, 1)
    return ret, wkly, trades, max_dd

print("V7 ML 5-Fold Walk-Forward Validation")
print("=" * 90)

for name, pair, cls in CANDIDATES:
    df = fetcher.fetch_klines(pair, "4h", "2024-01-01", "2026-03-01")
    if df.empty:
        continue
    n = len(df)
    fs = n // N_FOLDS
    folds = []
    for f in range(N_FOLDS):
        seg = df.iloc[f * fs:min((f + 1) * fs, n)].copy().reset_index(drop=True)
        t0 = time.time()
        ret, wkly, trades, dd = bt(cls, seg)
        el = time.time() - t0
        folds.append((ret, wkly, trades, dd, el))

    pos_folds = sum(1 for r, _, _, _, _ in folds if r > 0)
    avg_wkly = np.mean([w for _, w, _, _, _ in folds])
    fold_strs = "  ".join([f"F{i+1}:{r:+.1f}%" for i, (r, _, _, _, _) in enumerate(folds)])
    status = "ALL+" if pos_folds == N_FOLDS else f"{pos_folds}/{N_FOLDS}"
    print(f"  {name:<12} {pair:<12} {status:<5} avg:{avg_wkly:+.2f}%/wk | {fold_strs}")
    time.sleep(0.3)

print("=" * 90)
