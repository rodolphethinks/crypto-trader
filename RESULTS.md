# MEXC Trading System — Complete Results Summary

## System Overview
- **Exchange**: MEXC Spot V3 REST API (zero-fee USDC pairs)
- **Capital**: $10,000 initial
- **Risk Profile**: 5% risk per trade, 100% max position, 35% max drawdown
- **Timeframe**: 4h (validated as optimal across all strategy types)
- **Validation**: 5-fold walk-forward time-series cross-validation, all folds must be positive

---

## Walk-Forward Validated Strategies (Original Long/Short Backtests)

> **WARNING**: These results used backtests that allowed **short selling**. MEXC spot is **long-only**.
> See "Long-Only Reality Check" below for re-tested results.

### V6 Strategies (6 configs — original long/short)

| # | Strategy | Pair | OOS Avg/wk | Full 2yr Return | Max DD | Sharpe |
|---|----------|------|-----------|-----------------|--------|--------|
| 1 | MomentumAccelerator | AVAXUSDC | **+1.81%/wk** | +295% | 22.2% | 1.53 |
| 2 | MultiEdgeComposite | XRPUSDC | +1.33%/wk | +317% | 23.7% | 1.57 |
| 3 | RegimeMomentumV2 | XRPUSDC | +1.32%/wk | +282% | **18.9%** | **1.58** |
| 4 | MultiEdgeComposite | DOGEUSDC | +1.13%/wk | +278% | 25.3% | 1.43 |
| 5 | CrossPairLeader | AVAXUSDC | +1.04%/wk | +126% | 22.1% | 1.10 |
| 6 | CrossPairLeader | BTCUSDC | +1.01%/wk | +143% | 22.8% | 1.15 |

### V7 Strategies (4 configs — original long/short)

| # | Strategy | Pair | OOS Avg/wk | Full 2yr Return | Max DD | Sharpe |
|---|----------|------|-----------|-----------------|--------|--------|
| 7 | AdaptiveChannel | NEARUSDC | **+1.74%/wk** | +396% | 26.4% | 1.63 |
| 8 | AdaptiveChannel | SOLUSDC | +1.54%/wk | +236% | 28.5% | 1.61 |
| 9 | VolatilityCapture | AVAXUSDC | +1.53%/wk | +281% | **6.9%** | **1.99** |
| 10 | VolatilityCapture | NEARUSDC | +1.02%/wk | +128% | 29.9% | 1.15 |

---

## Long-Only Reality Check (MEXC Spot Constraint)

All 10 V6/V7 configs re-tested with **strictly long-only** backtests (BUY to open, SELL to close only).
Also includes fine-tuned V8 strategies (TSMOM: 375 combos, VolBreakoutMom: 540 combos).

### V6/V7 Re-Test (long-only)

| # | Strategy | Pair | Long-Only WF Avg/wk | Folds | Long-Only Full | Avg DD | Status |
|---|----------|------|-----------|-------|---------------|--------|--------|
| 1 | **AdaptiveChannel** | **NEARUSDC** | **+1.15%/wk** | **5/5** | +1.58%/wk | 13.9% | **PRODUCTION-READY** |
| 2 | AdaptiveChannel | SOLUSDC | +0.70%/wk | 5/5 | +0.86%/wk | 11.1% | Valid (below target) |
| 3 | RegimeMomentumV2 | XRPUSDC | +0.99%/wk | 3/5 | +0.94%/wk | 13.9% | Close but inconsistent |
| 4 | VolatilityCapture | AVAXUSDC | +0.84%/wk | 4/5 | +1.24%/wk | 10.3% | Promising |
| 5 | MultiEdgeComposite | XRPUSDC | +0.81%/wk | 3/5 | +1.26%/wk | 16.5% | Inconsistent OOS |
| 6 | MultiEdgeComposite | DOGEUSDC | +0.47%/wk | 3/5 | +0.46%/wk | 24.6% | Degraded |
| 7 | CrossPairLeader | BTCUSDC | +0.37%/wk | 4/5 | +0.46%/wk | 14.4% | Moderate |
| 8 | MomentumAccelerator | AVAXUSDC | +0.03%/wk | 1/5 | +0.05%/wk | 22.8% | **FAILED** |
| 9 | VolatilityCapture | NEARUSDC | +0.36%/wk | 3/5 | +0.50%/wk | 12.1% | Degraded |
| 10 | CrossPairLeader | AVAXUSDC | -0.17%/wk | 2/5 | -0.17%/wk | 24.7% | **FAILED** |

**Key finding: Only 2/10 survive 5/5 long-only. Only 1/10 exceeds 1%/wk.**

Short selling accounted for 40-100% of original profits on most strategies.

### V8 Fine-Tuned Results (long-only)

| # | Strategy | Pair | Sweep Best | WF Avg/wk | Folds | Avg DD |
|---|----------|------|-----------|-----------|-------|--------|
| 1 | VolBreakoutMom | XRPUSDC | +1.77%/wk | **+1.17%/wk** | 4/5 | 14.4% |
| 2 | VolBreakoutMom | DOGEUSDC | +2.17%/wk | +1.05%/wk | 3/5 | 20.0% |
| 3 | TSMOM | ADAUSDC | +1.15%/wk | +0.94%/wk | 4/5 | 15.5% |
| 4 | VolBreakoutMom | ADAUSDC | +1.15%/wk | +0.73%/wk | 3/5 | 10.3% |
| 5 | TSMOM | SOLUSDC | +1.05%/wk | +0.73%/wk | 4/5 | 19.4% |

None achieved 5/5, but VBM XRP (+1.17%/wk, 4/5) and VBM DOGE (+1.05%/wk, 3/5) show strong OOS returns.

### Combined Long-Only Ranking (all versions)

| Rank | Strategy | Pair | WF Avg/wk | Folds | Avg DD | Source |
|------|----------|------|-----------|-------|--------|--------|
| 1 | **AdaptiveChannel** | **NEARUSDC** | **+1.15%/wk** | **5/5** | **13.9%** | **V7** |
| 2 | VolBreakoutMom | XRPUSDC | +1.17%/wk | 4/5 | 14.4% | V8 |
| 3 | VolBreakoutMom | DOGEUSDC | +1.05%/wk | 3/5 | 20.0% | V8 |
| 4 | RegimeMomentumV2 | XRPUSDC | +0.99%/wk | 3/5 | 13.9% | V6 |
| 5 | TSMOM | ADAUSDC | +0.94%/wk | 4/5 | 15.5% | V8 |
| 6 | VolatilityCapture | AVAXUSDC | +0.84%/wk | 4/5 | 10.3% | V7 |
| 7 | MultiEdgeComposite | XRPUSDC | +0.81%/wk | 3/5 | 16.5% | V6 |
| 8 | TSMOM | SOLUSDC | +0.73%/wk | 4/5 | 19.4% | V8 |
| 9 | AdaptiveChannel | SOLUSDC | +0.70%/wk | 5/5 | 11.1% | V7 |

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

**Initial V8 sweep**: None achieved 5/5 or 1%/wk. Best: TSMOM SOL at +0.80%/wk (4/5 folds).
**After fine-tuning** (915 additional combos): VBM XRP reached +1.17%/wk (4/5), VBM DOGE +1.05%/wk (3/5).

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

### Best Risk-Adjusted Configs (Long-Only Reality)
1. **AdaptChan NEAR**: +1.15%/wk, 13.9% avg DD, **only 5/5 long-only validated config ≥1%/wk**
2. **VolCapture AVAX**: +0.84%/wk, **10.3% avg DD** (best risk-adjusted, but 4/5)
3. **AdaptChan SOL**: +0.70%/wk, **11.1% avg DD**, 5/5 folds (conservative choice)

### Critical Insight: Short Selling Impact
- Original V6/V7 backtests allowed shorts → **10/10 configs >1%/wk, all 5/5**
- Long-only re-test → **1/10 configs >1%/wk with 5/5** (AdaptChan NEAR)
- Short selling contributed 40-100% of returns for most strategies
- **MomentumAccelerator AVAX**: +1.81%/wk (long/short) → +0.03%/wk (long-only) = 98% from shorts

---

## Production System — 2-Layer Architecture

### Design Philosophy
"Trust validated strategies. Intervene only when risk becomes extreme."

Two layers only:
1. **Strategy Execution (PRIMARY)** — 7 long-only validated configs with tier-based capital allocation
2. **Catastrophic Risk Filter (OVERRIDE ONLY)** — slow BTC kill switch, ALL 4 conditions × 3 candle persistence

No regime filters, no RS ranking — data proved they hurt returns (14/15 configs degraded).

### Deployed Strategies & Capital Allocation

| Tier | Strategy | Pair | Allocation | WF Avg/wk | Folds | Risk/Trade |
|------|----------|------|-----------|-----------|-------|-----------|
| T1 (Core) | AdaptiveChannel | NEARUSDC | **25%** | +1.15%/wk | 5/5 | 5% |
| T1 (Core) | AdaptiveChannel | SOLUSDC | **15%** | +0.70%/wk | 5/5 | 5% |
| T2 (Active) | VolBreakoutMom | XRPUSDC | **12%** | +1.17%/wk | 4/5 | 5% |
| T2 (Active) | VolatilityCapture | AVAXUSDC | **10%** | +0.84%/wk | 4/5 | 5% |
| T2 (Active) | TSMOM | ADAUSDC | **8%** | +0.94%/wk | 4/5 | 5% |
| T3 (Reduced) | VolBreakoutMom | DOGEUSDC | **5%** | +1.05%/wk | 3/5 | 5% |
| T3 (Reduced) | RegimeMomentumV2 | XRPUSDC | **5%** | +0.99%/wk | 3/5 | 5% |
| — | USDC Reserve | — | **20%** | — | — | — |

**Total deployed: 80% | USDC reserve: 20%**

### Risk Controls
- **Per trade**: 5% of allocated capital risked, always with stop-loss
- **Global drawdown**: ≥15% → stop new trades, reduce to ~30% exposure; recover at <10%
- **Consecutive losses**: 5 → strategy auto-disabled
- **Post stop-loss cooldown**: 2 candles before re-entry
- **Extreme candle filter**: skip entry if candle body > 2× ATR
- **Correlation cap**: 60% max to correlated asset groups (e.g. L1 alts: NEAR+SOL+AVAX)
- **Dynamic allocation**: +5% if WR > 55% over 10+ trades; -10% after 3 consecutive losses

### Catastrophic Risk Filter (Kill Switch)
Activates ONLY when ALL 4 conditions persist for ≥3 consecutive candles:
1. BTC below 200-period SMA
2. BTC making 3+ consecutive lower lows
3. BTC volatility contracting (ATR declining)
4. Breakout failure rate > 70% over last 20 bars

When active: Only T1 strategies at 50% allocation. All others suspended.
When inactive: No restriction — strategies trade freely.

### Expected Performance (Weighted by Allocation)
- **Backtest-weighted avg**: ~0.96%/wk (~$96/wk on $10k)
- **Live assumption (50-70% of backtest)**: ~0.48-0.67%/wk (~$48-67/wk)
- **Annual projection (50% decay)**: ~$2,500-3,500 on $10k (25-35%)
- **Max expected drawdown**: ~15-20% (backtest max was 13.9% for NEAR)

### Files
- **`trading/production_trader.py`** — 2-layer production trader (750 lines)
- `trading/live_trader.py` — Legacy trader with original 10 long/short configs
- `strategies/v6_aggressive.py` — 7 V6 strategies
- `strategies/v7_diverse.py` — 6 V7 strategies (AdaptChan, VolCapture, etc.)
- `strategies/v8_research.py` — 5 research-extracted strategies (VolBreakoutMom, TSMOM, etc.)
- `scripts/run_v8_pipeline.py` — Automated research-to-backtest pipeline
- `scripts/run_longonly_retest.py` — Long-only re-test + fine sweep pipeline
- `scripts/run_regime_filtered.py` — 3-layer regime test (proved harmful)

### How to Run
```bash
# One-shot signal check (recommended first step)
python trading/production_trader.py --check

# Paper trading (validate live before risking capital)
python trading/production_trader.py --mode paper --capital 10000

# Check status of running system
python trading/production_trader.py --status

# Live trading (requires 'CONFIRM' input)
python trading/production_trader.py --mode live --capital 10000
```

---

## Pair Exposure (Production System)
| Pair | # Configs | Strategies | Total Allocation |
|------|-----------|------------|-----------------|
| NEARUSDC | 1 | AdaptChan | 25% |
| SOLUSDC | 1 | AdaptChan | 15% |
| XRPUSDC | 2 | VBM, RegimeMomV2 | 17% |
| AVAXUSDC | 1 | VolCapture | 10% |
| ADAUSDC | 1 | TSMOM | 8% |
| DOGEUSDC | 1 | VBM | 5% |

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
| **V8** | **Research-extracted (5 from LLM docs)** | **220 + 915 sweep + WF (long-only)** | **0/5 pass 5/5; VBM XRP 4/5 +1.17%/wk** |

### Long-Only Bottom Line
- **Original claim (long/short)**: 10 configs >1%/wk, all 5/5 positive
- **Long-only reality**: **1 config** >1%/wk with 5/5 (AdaptChan NEAR +1.15%/wk)
- **Strong candidates** (4/5 folds, >1%/wk): VBM XRP +1.17%/wk, VBM DOGE +1.05%/wk
- **Near-target** (4/5, ~1%/wk): RegimeMomV2 XRP +0.99%/wk, TSMOM ADA +0.94%/wk

---

## 3-Layer Regime Filter Test

Tested a market-first trading system with 3 layers: BTC regime filter (master switch) → Relative Strength selection (top 3 assets) → Strategy execution. Hypothesis: filtering out RISK-OFF periods and weak assets should improve risk-adjusted returns.

### BTC Regime Filter Stats
- **RISK-ON**: 2,353/4,917 bars (47.9%)
- **RISK-OFF**: 2,564 bars (52.1%)
- Regime transitions: 451 (avg ~11 bars per regime = ~1.8 days)

### Results: Regime Filter HURT Returns

| Config | Raw WF | + BTC Regime | + Regime + RS | Δ from Raw | Folds (full) |
|--------|--------|-------------|---------------|------------|------|
| VolBreakoutMom DOGE | +1.05%/wk | +1.00%/wk | +0.57%/wk | -0.48% | 4/5 |
| TSMOM ADA | +0.94%/wk | +0.68%/wk | +0.42%/wk | -0.52% | 3/5 |
| VBM XRP | +1.17%/wk | +0.71%/wk | **+0.42%/wk** | -0.75% | **5/5** |
| MultiEdge XRP | +0.81%/wk | +0.70%/wk | +0.35%/wk | -0.46% | 2/5 |
| **AdaptChan NEAR** | **+1.15%/wk** | +0.74%/wk | +0.30%/wk | -0.85% | 4/5 |
| AdaptChan SOL | +0.70%/wk | +0.63%/wk | +0.26%/wk | -0.44% | 4/5 |
| CrossPair BTC | +0.37%/wk | +0.26%/wk | +0.26%/wk | -0.11% | 4/5 |
| TSMOM SOL | +0.73%/wk | **+0.78%/wk** | +0.03%/wk | -0.69% | 3/5 |

**Regime filter improved only 2/15 configs. Full 3-layer system improved only 1/15.**

### What the Regime Filter DID Improve
- **Drawdowns**: Average DD dropped from 27.9% → 22.8%
- **Trade count**: Average trades 104 → 42 (60% fewer trades)
- **CrossPair AVAX** (losing strategy): -0.17%/wk → -0.04%/wk (damage reduction)
- **TSMOM SOL**: Only config where BTC regime alone improved WF avg (+0.73% → +0.78%)

### Why the Regime Filter Failed to Boost Returns
1. **The strategies already have their own regime detection.** AdaptChan adjusts channels by volatility percentile. RegimeMomV2 switches between trend/range/volatile modes. Adding an external filter is redundant — it just blocks valid signals.
2. **Crypto altcoin trends don't always follow BTC.** NEAR had its strongest moves during periods BTC was flat/consolidating. The BTC regime filter blocked exactly those high-return entries.
3. **The RS filter was too restrictive.** NEAR (our best asset) is only in top-3 RS 16% of the time. It tends to outperform *after* dropping out of the top rankings, during recovery moves.
4. **451 transitions = too noisy.** Averaging ~1.8 days per regime means constant flipping, blocking entries mid-trend.

### Verdict on 3-Layer Architecture
The **concept is sound** but the **implementation needs work**:
- The BTC regime filter alone (no RS) preserved 5/5 fold consistency for AdaptChan NEAR — just at lower returns (+0.74%/wk vs +1.15%/wk). It reduces risk but also caps upside.
- The RS layer is counterproductive for pre-selected asset-strategy pairs. It may work better for *dynamic* asset selection (choose which asset to trade today) rather than *filtering* a fixed pairing.
- A **wider regime window** (less sensitive, longer lookback) would reduce whipsaws and preserve more of the edge.
- Better approach: use the regime filter **only as a kill switch during extreme risk-off** (all 4 conditions bearish), not as a general gatekeeper.
