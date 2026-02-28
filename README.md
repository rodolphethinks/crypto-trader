# MEXC Crypto Trading System

An algorithmic cryptocurrency trading system for the MEXC exchange with **10 walk-forward validated strategies** exceeding **1% weekly returns**. Features 25+ strategies across 7 research generations, a bar-by-bar backtesting engine, live/paper trading execution, and GPU-accelerated ML models — all optimized for **zero-fee USDC trading pairs**.

---

## Results

**10 strategy configs pass 5-fold walk-forward validation** (all 5/5 folds positive, all exceeding 1%/week OOS):

| # | Strategy | Pair | OOS Avg/wk | 2yr Return | Max DD | Sharpe |
|---|----------|------|-----------|------------|--------|--------|
| 1 | MomentumAccelerator | AVAXUSDC | **+1.81%** | +295% | 22.2% | 1.53 |
| 2 | AdaptiveChannel | NEARUSDC | **+1.74%** | +396% | 26.4% | 1.63 |
| 3 | AdaptiveChannel | SOLUSDC | +1.54% | +236% | 28.5% | 1.61 |
| 4 | VolatilityCapture | AVAXUSDC | +1.53% | +281% | **6.9%** | **1.99** |
| 5 | MultiEdgeComposite | XRPUSDC | +1.33% | +317% | 23.7% | 1.57 |
| 6 | RegimeMomentumV2 | XRPUSDC | +1.32% | +282% | 18.9% | 1.58 |
| 7 | MultiEdgeComposite | DOGEUSDC | +1.13% | +278% | 25.3% | 1.43 |
| 8 | VolatilityCapture | NEARUSDC | +1.02% | +128% | 29.9% | 1.15 |
| 9 | CrossPairLeader | AVAXUSDC | +1.04% | +126% | 22.1% | 1.10 |
| 10 | CrossPairLeader | BTCUSDC | +1.01% | +143% | 22.8% | 1.15 |

> Validated on 2-year data (2024–2026), 4h timeframe, $10k capital, 5% risk/trade, 0% commission (MEXC zero-fee USDC pairs), 0.05% slippage.

See [RESULTS.md](RESULTS.md) for the full research report.

---

## Project Structure

```
MEXC/
├── config/                  # Global settings, trading pair lists
│   ├── settings.py          # API config, risk defaults, paths
│   └── pairs.py             # Zero-fee pair definitions
│
├── api/                     # MEXC API integration
│   ├── client.py            # REST API client (HMAC-SHA256 auth)
│   └── websocket_client.py  # WebSocket for live streams
│
├── data/                    # Data acquisition & caching
│   └── fetcher.py           # Kline fetcher with auto-pagination
│
├── indicators/              # 40+ technical indicators
│   ├── trend.py             # SMA, EMA, MACD, ADX, Supertrend, Ichimoku
│   ├── momentum.py          # RSI, Stochastic, CCI, Williams %R, ROC, MFI
│   ├── volatility.py        # ATR, Bollinger, Keltner, Donchian, Squeeze
│   ├── volume.py            # OBV, VWAP, CMF, Force Index, Elder Ray
│   └── custom.py            # Pivots, Fibonacci, Market Regime
│
├── patterns/                # Pattern recognition
│   ├── chart_patterns.py    # Double Tops/Bottoms, H&S, Triangles, Flags
│   ├── structure.py         # S/R, BOS, ChoCH, Order Blocks, Liquidity
│   └── candlestick.py       # Doji, Hammer, Engulfing, Stars
│
├── strategies/              # 25+ strategies across 7 generations
│   ├── base.py              # BaseStrategy ABC, Signal enum
│   ├── v6_aggressive.py     # V6: MomAccel, MultiEdge, RegimeMomV2, CrossPair, ...
│   ├── v7_diverse.py        # V7: AdaptChan, VolCapture, TrendPulse, MomSwitch, ...
│   ├── v7_ml.py             # V7 ML: LSTM, CNN, Transformer (GPU)
│   ├── v4_research.py       # V4: Ensembles, tuned variants
│   ├── alt_alpha.py         # V3: AdaptiveTrend, CrossTFMomentum
│   └── (14 more)            # V1-V2: Classic TA strategies
│
├── risk/                    # Risk management
│   └── manager.py           # Position sizing, drawdown limits, SL/TP
│
├── backtesting/             # Backtesting engine
│   ├── engine.py            # Bar-by-bar simulation with SL/TP
│   ├── portfolio.py         # Portfolio-level backtester
│   └── runner.py            # Multi-strategy sweep orchestrator
│
├── trading/                 # Live/paper execution
│   ├── live_trader.py       # Production trader (10 validated configs)
│   ├── executor.py          # Paper + live order placement
│   ├── portfolio.py         # Portfolio tracking
│   └── order_manager.py     # Order lifecycle + SL/TP monitoring
│
├── scripts/                 # CLI entry points & research
│   ├── run_production.py    # Launch production paper/live trading
│   ├── run_v7_sweep.py      # V7 strategy sweep
│   ├── run_v7_wf.py         # V7 walk-forward validation
│   ├── run_v7_ml_sweep.py   # V7 ML/DL GPU sweep
│   ├── run_wide_4h_sweep.py # V6 wide strategy sweep
│   ├── run_wf_wide_4h.py    # V6 walk-forward validation
│   ├── verify_v6.py         # Independent verification backtests
│   └── (more)               # Grid search, optimization, analysis
│
├── dashboard/               # Web UI (Streamlit)
│   └── app.py
│
├── tests/                   # Test suite
│   └── test_system.py
│
├── RESULTS.md               # Full research results
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Install Dependencies

```bash
# Recommended: use uv for fast installs
pip install uv
uv pip install -r requirements.txt

# Or standard pip
pip install -r requirements.txt

# For GPU-accelerated ML strategies (optional)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env with your MEXC API key and secret
```

### 3. Verify Installation

```bash
python -m pytest tests/ -v
```

---

## Usage

### Production Trading (Validated Strategies)

```bash
# Check current signals across all 10 validated configs
python scripts/run_production.py --check-signals

# Start paper trading with all validated strategies
python scripts/run_production.py --mode paper --capital 10000

# Live trading (CAUTION — real funds)
python scripts/run_production.py --mode live --capital 10000

# Run specific configs only
python scripts/run_production.py --mode paper --configs MomAccel_AVAX_4h,VolCapture_AVAX_4h
```

### Research & Backtesting

```bash
# Run V7 strategy sweep (6 strategies × 9 pairs, 4h)
python scripts/run_v7_sweep.py

# Walk-forward validate winners
python scripts/run_v7_wf.py

# ML/DL sweep with GPU (LSTM, CNN, Transformer)
python scripts/run_v7_ml_sweep.py

# V6 wide sweep (7 strategies × 14 pairs)
python scripts/run_wide_4h_sweep.py

# Independent verification on full 2-year data
python scripts/verify_v6.py
```

### Data & Dashboard

```bash
# Fetch historical data
python scripts/fetch_data.py --symbols BTCUSDC ETHUSDC --interval 4h --days 730

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Strategies

### Production (Walk-Forward Validated)

| Strategy | Type | Description |
|----------|------|-------------|
| **MomentumAccelerator** | Momentum | Rate of change of momentum + volume surge detection |
| **MultiEdgeComposite** | Ensemble | 5-component system: trend + momentum + reversion + breakout + volume |
| **RegimeMomentumV2** | Adaptive | Fast regime detection with mode switching (trend/range) |
| **AdaptiveChannel** | Breakout | Self-adjusting Keltner channels based on volatility percentile |
| **VolatilityCapture** | Squeeze | Bollinger/Keltner squeeze breakout with momentum confirmation |
| **CrossPairLeader** | Cross-asset | BTC price leads altcoin lag detection |

### Research (V7 ML/DL — GPU)

| Strategy | Type | Description |
|----------|------|-------------|
| **LSTMTrend** | Deep Learning | Bidirectional LSTM with rolling retrain |
| **CNNMomentum** | Deep Learning | 1D-CNN on multi-feature windows |
| **TransformerPred** | Deep Learning | Lightweight transformer with positional encoding |

### Additional V7 (Non-ML)

| Strategy | Type | Description |
|----------|------|-------------|
| **OrderFlowMomentum** | Volume | OBV divergence + volume surge alignment |
| **TrendPulse** | Trend | Multi-period EMA alignment with acceleration |
| **MeanReversionRSI** | Reversion | RSI extremes with divergence & volume |
| **MomentumSwitch** | Adaptive | Regime-adaptive fast/slow momentum switching |

### Classic Strategies (V1–V4)

14 additional strategies including SMC Liquidity, Trend Following, Breakout, Scalping, Grid Trading, Bollinger Bands, MACD Divergence, VWAP, Ichimoku, Multi-Timeframe, Smart DCA, Pairs Trading, and ensemble variants.

---

## Research Journey (V1 → V7)

| Version | Strategies | Combos Tested | Result |
|---------|-----------|---------------|--------|
| V1–V2 | 14 classic TA | 920 | None profitable after costs |
| V3 | 12 ML + HF + alt-alpha | ~200 | Best: ~$303/yr on $10k |
| V4 | Ensembles + grid-optimized | 2,789 | Best: ~$260/yr |
| V5 | DL + Sentiment | ~100 | Moderate returns |
| **V6** | 7 aggressive | 98 | **6 pass WF at 1%+/wk** |
| **V7** | 6 diverse + 3 ML | 81 | **4 pass WF at 1%+/wk** |

**Key breakthrough**: Position sizing (5% risk, 100% max position) on 4h zero-fee pairs.

---

## Configuration

Key settings in `config/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `DEFAULT_INITIAL_CAPITAL` | $10,000 | Starting capital |
| `DEFAULT_COMMISSION` | 0.0% | Zero for MEXC USDC pairs |
| `DEFAULT_SLIPPAGE` | 0.05% | Simulated slippage |
| `RISK_PER_TRADE` | 1.0% | % equity risked (overridden to 5% in production) |
| `MAX_POSITION_SIZE_PCT` | 2.0% | Max position (overridden to 100% in production) |
| `MAX_DRAWDOWN_PCT` | 10% | Drawdown limit (overridden to 35% in production) |

Environment variables (`.env`):

| Variable | Description |
|----------|-------------|
| `MEXC_API_KEY` | Your MEXC API key |
| `MEXC_SECRET_KEY` | Your MEXC secret key |
| `TRADING_MODE` | `paper` or `live` |

---

## Tech Stack

- **Python 3.11** with `uv` package manager
- **PyTorch 2.6** (CUDA) for ML strategies
- **pandas / numpy** for data processing
- **MEXC Spot V3 REST API** + WebSocket
- **Streamlit** for dashboard

---

## Disclaimer

For personal/educational use only. Cryptocurrency trading involves substantial risk of loss. Past backtest performance does not guarantee future results. Always paper trade before risking real capital.
