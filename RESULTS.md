# MEXC Trading System — Complete Results Summary

## System Overview
- **Exchange**: MEXC Spot V3 REST API (zero-fee USDC pairs)
- **Capital**: $10,000 initial
- **Risk Profile**: 5% risk per trade, 100% max position, 35% max drawdown
- **Timeframe**: 4h (validated as optimal across all strategy types)
- **Validation**: 5-fold walk-forward time-series cross-validation, all folds must be positive

---

## Walk-Forward Validated Strategies (10 Configs, ALL 5/5 Folds Positive)

### V6 Strategies (6 configs)

| # | Strategy | Pair | OOS Avg/wk | Full 2yr Return | Max DD | Sharpe |
|---|----------|------|-----------|-----------------|--------|--------|
| 1 | MomentumAccelerator | AVAXUSDC | **+1.81%/wk** | +295% | 22.2% | 1.53 |
| 2 | MultiEdgeComposite | XRPUSDC | +1.33%/wk | +317% | 23.7% | 1.57 |
| 3 | RegimeMomentumV2 | XRPUSDC | +1.32%/wk | +282% | **18.9%** | **1.58** |
| 4 | MultiEdgeComposite | DOGEUSDC | +1.13%/wk | +278% | 25.3% | 1.43 |
| 5 | CrossPairLeader | AVAXUSDC | +1.04%/wk | +126% | 22.1% | 1.10 |
| 6 | CrossPairLeader | BTCUSDC | +1.01%/wk | +143% | 22.8% | 1.15 |

### V7 Strategies (4 configs)

| # | Strategy | Pair | OOS Avg/wk | Full 2yr Return | Max DD | Sharpe |
|---|----------|------|-----------|-----------------|--------|--------|
| 7 | AdaptiveChannel | NEARUSDC | **+1.74%/wk** | +396% | 26.4% | 1.63 |
| 8 | AdaptiveChannel | SOLUSDC | +1.54%/wk | +236% | 28.5% | 1.61 |
| 9 | VolatilityCapture | AVAXUSDC | +1.53%/wk | +281% | **6.9%** | **1.99** |
| 10 | VolatilityCapture | NEARUSDC | +1.02%/wk | +128% | 29.9% | 1.15 |

### V8 Research Strategies (partial validation — long-only constraint)

Extracted 5 strategies from 4 LLM research documents (CHATGPT.md, CLAUDE.md, PERPLEXITY.md, GEMINI.md).
Backtested long-only on MEXC spot (no shorting). 220 sweep combos, 5-fold walk-forward on top 5.

| # | Strategy | Pair | OOS Avg/wk | Pos Folds | In-Sample Return | Max DD | Sharpe |
|---|----------|------|-----------|-----------|-----------------|--------|--------|
| 1 | TSMOM | SOLUSDC | **+0.80%/wk** | 4/5 | +113.6% | 21.0% | 0.96 |
| 2 | TSMOM | ADAUSDC | +0.65%/wk | 4/5 | +71.7% | 19.8% | 0.72 |
| 3 | VolBreakoutMom | DOGEUSDC | +0.52%/wk | 3/5 | +81.9% | 18.6% | 0.89 |
| 4 | VolBreakoutMom | XRPUSDC | +0.48%/wk | 3/5 | +90.6% | 16.0% | 0.98 |
| 5 | TSMOM | NEARUSDC | +0.19%/wk | 3/5 | +80.2% | 30.4% | 0.81 |

**None achieved 5/5 folds positive or 1%/wk.** Best: TSMOM SOL at +0.80%/wk (4/5 folds).

#### V8 Strategies That Failed
- **MeanRevLowVol** — Uniformly negative. Long-only can't short overbought conditions.
- **WeekendGapFade** — Too few trades (2-23/symbol), thin calendar edge on 4h.
- **BTCResidualMR** — Without short side, only captures half the residual reversion.

#### V8 Key Insight
Research docs heavily recommended funding-rate arbitrage (cited in ALL 4 docs) and cross-exchange strategies — both require futures/multi-exchange access unavailable on MEXC spot. The OHLCV-only strategies from research underperform V6/V7's hand-crafted indicators, especially under the long-only constraint.

### ML/DL Models (FAILED walk-forward)
- LSTM, CNN, Transformer models tested on 9 pairs — none achieved 5/5 folds positive
- Best: LSTMTrend BNBUSDC at 3/5 folds (+0.27%/wk avg)
- **Conclusion**: Conventional strategies with well-designed indicators outperform DL on 4h crypto

---

## Key Findings

### What Works
- **4h timeframe** is optimal — enough data points for significance, not too noisy
- **5% risk per trade** with **100% position sizing** — critical breakthrough (was capped at 2%)
- **Walk-forward validation** with 5 folds eliminates overfitting
- **Zero-fee USDC pairs** on MEXC remove commission drag entirely

### Strategy Types That Work
1. **MomentumAccelerator** — Acceleration of price momentum + volume confirmation
2. **MultiEdgeComposite** — 5-component ensemble (trend+momentum+reversion+breakout+volume)
3. **RegimeMomentumV2** — Fast regime detection with adaptive mode switching
4. **AdaptiveChannel** — Self-adjusting Keltner channels based on volatility percentile
5. **VolatilityCapture** — Bollinger/Keltner squeeze breakout (best risk-adjusted: 6.9% DD)
6. **CrossPairLeader** — BTC leads altcoin lag detection

### Best Risk-Adjusted Configs
1. **VolCapture AVAX**: 1.53%/wk, **6.9% max DD**, Sharpe 1.99
2. **RegimeMomV2 XRP**: 1.32%/wk, 18.9% max DD, Sharpe 1.58
3. **MomAccel AVAX**: 1.81%/wk (highest return), 22.2% max DD, Sharpe 1.53

---

## Production System

### Files
- `trading/live_trader.py` — Production paper/live trader with all 10 configs
- `scripts/run_production.py` — Launcher (`--check-signals`, `--mode paper/live`)
- `strategies/v6_aggressive.py` — 7 V6 strategies
- `strategies/v7_diverse.py` — 6 V7 strategies (OrderFlowMom, TrendPulse, VolCapture, MeanRevRSI, AdaptChan, MomSwitch)
- `strategies/v7_ml.py` — 3 DL strategies (LSTM, CNN, Transformer)
- `strategies/v8_research.py` — 5 research-extracted strategies (VolBreakoutMom, MeanRevLowVol, WeekendGapFade, BTCResidualMR, TSMOM)
- `scripts/run_v8_pipeline.py` — Automated research-to-backtest pipeline
- `results/` — V8 per-strategy JSONs, trade CSVs, sweep & walk-forward CSVs

### How to Run
```bash
# Check current signals (one-shot)
python scripts/run_production.py --check-signals

# Start paper trading
python scripts/run_production.py --mode paper --capital 10000

# Start live trading (use with caution)
python scripts/run_production.py --mode live --capital 10000
```

---

## Pair Exposure
| Pair | # Configs | Strategies |
|------|-----------|------------|
| AVAXUSDC | 3 | MomAccel, CrossPair, VolCapture |
| XRPUSDC | 2 | MultiEdge, RegimeMomV2 |
| NEARUSDC | 2 | AdaptChan, VolCapture |
| DOGEUSDC | 1 | MultiEdge |
| BTCUSDC | 1 | CrossPair |
| SOLUSDC | 1 | AdaptChan |

---

## Strategy Research Summary (V1 → V7)

| Version | Strategies | Testing | Result |
|---------|-----------|---------|--------|
| V1-V2 | Classic TA (SMA cross, RSI, Bollinger) | 920 combos | None profitable |
| V3 | ML + HF + Alt-alpha (12 strategies) | Walk-forward | Best: ~$303/yr |
| V4 | Ensembles + grid-optimized | 2789 combos | Best: ~$260/yr |
| V5 | DL + Sentiment (files lost) | Walk-forward | Moderate |
| **V6** | **Aggressive (7 new strategies)** | **Wide 4h sweep + 5-fold WF** | **6/7 pass, +1.0-1.8%/wk** |
| **V7** | **Diverse (6) + ML (3)** | **Wide 4h + 5-fold WF** | **4/8 pass, +1.0-1.7%/wk** |
| **V8** | **Research-extracted (5 from LLM docs)** | **220 sweep + 5-fold WF (long-only)** | **0/5 pass 5/5, best 4/5 at +0.80%/wk** |

**Total validated configs exceeding 1%/week: 10** (all with ALL 5/5 folds positive)

**V8 note**: Long-only constraint significantly limits research strategies. TSMOM SOL (+0.80%/wk, 4/5 folds) is closest candidate — needs further tuning or short-selling access to reach 1%/wk threshold.
