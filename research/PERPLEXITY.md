Here are 5 fully rule-based, structurally motivated crypto strategies that can be implemented and backtested programmatically, with explicit rules and pseudo-code for each.

***

## 1. Delta‑Neutral Funding Rate Harvest

### Conceptual edge

Perpetual futures prices are kept close to spot via periodic funding payments between longs and shorts, but when the market becomes one‑sided, funding rates can remain extreme for multiple intervals, creating a predictable transfer from one side of the market to the other. [aeaweb](https://www.aeaweb.org/conference/2026/program/paper/ByyFEfr4)
Empirical work on delta‑neutral funding‑rate strategies in BTC shows that harvesting funding with a hedged spot–perp portfolio can be profitable even after transaction costs, especially when rebalancing is not too frequent. [digikogu.taltech](https://digikogu.taltech.ee/et/Download/dddbf9de-5dba-49dd-a2c8-60e87337731c)

### Market type & timeframe

- Market: BTC and ETH perpetual futures vs spot (same exchange where possible)
- Timeframe:
  - Signal: funding interval (e.g., 1h or 8h depending on venue)
  - Position horizon: 1–5 funding intervals (intra‑day to multi‑day)

### Execution style

- Core leg (perp and spot): primarily liquidity taker (market orders) at signal time, but prefer passive limit orders when spread and depth allow.
- Rebalancing: limit orders where possible; cap market‑order participation to a fixed fraction of interval volume.

### Indicators & features

All measured at time \(t\) using only information up to \(t-1\):

- Funding rate \(F_{t-1}\) (annualized or per interval)
- Perp–spot basis \(B_{t-1} = (P^{\text{perp}}_{t-1} - P^{\text{spot}}_{t-1}) / P^{\text{spot}}_{t-1}\)
- Open interest \(OI_{t-1}\) and its z‑score vs trailing 30‑day mean
- 24h realized volatility of spot returns \(RV_{t-1}\)
- 24h spot dollar volume \(V_{t-1}\)

### Exact trading rules

Assume we define:

- Thresholds:  
  - Funding extreme: \(F_{\text{high}} = +0.05\%\) per 8h (≈ 55% APR), \(F_{\text{low}} = -0.05\%\) per 8h  
  - Basis extreme: \(B_{\text{high}} = +0.50\%\), \(B_{\text{low}} = -0.50\%\)  
  - OI z‑score: \(z_{\text{OI,high}} = +1.5\), \(z_{\text{OI,low}} = -1.0\)  
- Volatility regime bounds: \(RV_{\min}, RV_{\max}\)

**Entry – short‑carry (bet against over‑leveraged longs):**

At the close of interval \(t-1\), if all hold:

1. \(F_{t-1} \ge F_{\text{high}}\)
2. \(B_{t-1} \ge B_{\text{high}}\)
3. \(z_{\text{OI},t-1} \ge z_{\text{OI,high}}\)
4. \(RV_{\min} \le RV_{t-1} \le RV_{\max}\)

Then at time \(t\):

- Short 1 unit notional of perp.
- Long 1 unit notional of spot (delta‑neutral).

**Entry – long‑carry (bet against over‑leveraged shorts):**

Same as above but with:

- \(F_{t-1} \le F_{\text{low}}, B_{t-1} \le B_{\text{low}}, z_{\text{OI},t-1} \le z_{\text{OI,low}}\)

Then:

- Long perp, short spot.

**Exit conditions (both sides):**

Exit entire position when any of:

1. Funding normalization: \(|F_{t-1}| < 0.02\%\) per 8h.
2. Basis mean‑reversion: \(|B_{t-1}| < 0.20\%\).
3. Time stop: position age ≥ \(H_{\max}\) funding intervals (e.g., 6).
4. Stop loss: cumulative P&L of the pair (including unrealized and funding) ≤ \(-S\%\) of allocated capital (e.g., \(-2\%\)).
5. Regime kill switch triggered (see below).

### Regime detection / kill switch

Do not enter new trades and close existing ones if any of:

- Extreme volatility: \(RV_{t-1} > RV_{\max}\) (e.g., realized 24h volatility > 150% annualized).
- Exchange‑specific stress: large intraday gaps \(> G\%\) between bars, or exchange halt flags.
- Funding structurally near zero over 30‑day window (indicates low directional leverage; edge is gone).

### Risk management & position sizing

- Volatility‑targeted sizing: For each asset, compute 30‑day realized volatility and size so that the expected 1‑day 1‑sigma move corresponds to a fixed fraction of equity (e.g., 0.5–1%).  
- Cap gross exposure: total pair notional across assets ≤ 2–3× equity; per‑asset cap (e.g., 50% of equity notional).
- Max drawdown circuit breaker: stop trading and reduce sizes by 50% if strategy equity drawdown exceeds 15% from peak; re‑enable after recovery above previous highwater.

### No‑trade zone (liquidity / spreads)

- Skip intervals where 24h spot volume \(V_{t-1}\) is below a fixed USD minimum (e.g., 20× your maximum trade size).
- Skip if high‑low range / close in last hour > 1.5% (proxy for extreme microstructure noise).
- If orderbook data is available, additionally skip when best‑bid–ask spread > 3× its 30‑day median or > 5 bps, whichever is larger.

### Backtesting specification (pseudo‑code)

```pseudo
for each asset in {BTC, ETH}:
  for each funding interval t from lookback_end+1 to T:
    # compute features from data up to t-1
    F = funding_rate[t-1]
    B = (perp_price[t-1] - spot_price[t-1]) / spot_price[t-1]
    OI_z = zscore(open_interest[1..t-1], window=30d)
    RV = realized_vol(spot_returns[1..t-1], window=24h)
    V = spot_dollar_volume[1..t-1 over 24h]

    # liquidity / regime filters
    if V < V_min or RV > RV_max: continue

    # manage existing positions
    update_PnL_including_funding_and_fees()
    if position_open:
      if abs(F) < F_norm or abs(B) < B_norm \
         or age >= H_max or pnl <= -S% or RV > RV_max:
        close_pair_at_mid_or_worse_with_slippage()
        continue

    # entry logic (only if flat)
    if not position_open:
      if F >= F_high and B >= B_high and OI_z >= OI_high and RV >= RV_min:
        size = vol_target_size(RV_30d)
        enter_short_perp_long_spot(size)
      elif F <= F_low and B <= B_low and OI_z <= OI_low and RV >= RV_min:
        size = vol_target_size(RV_30d)
        enter_long_perp_short_spot(size)
```

Include transaction cost model: assume 0.05–0.10% per side for taker, plus slippage proportional to participation rate.

### Parameters to optimize & anti‑overfitting

- Thresholds: \(F_{\text{high}}, F_{\text{low}}, B_{\text{high}}, B_{\text{low}}, z_{\text{OI}}\) bounds.
- Volatility band: \(RV_{\min}, RV_{\max}\).
- Holding cap \(H_{\max}\) and stop \(S\%\).

Use:

- Walk‑forward optimization: roll 1–2 year calibration windows with 3–6 month out‑of‑sample tests.
- Stability constraints: accept only parameter sets where sign of edge is consistent across assets (BTC, ETH) and sub‑periods (bull/bear).

### Expected behavior

- Best: prolonged trending markets with heavily one‑sided leverage (e.g., extreme positive funding during bull euphorias or strongly negative funding in panics). [yellow](https://yellow.com/learn/how-to-read-funding-rates-crypto-reversals)
- Weak / fails: flat markets with near‑zero funding, or violent regime shifts where funding collapses before you can enter or exit; also in extreme crash days when basis dislocates more than funding revenue and transaction costs.

### Supporting evidence

- Perpetuals’ funding mechanism and its role in aligning prices with spot are well documented. [semanticscholar](https://www.semanticscholar.org/paper/Fundamentals-of-Perpetual-Futures-He-Manela/82c0c4eb57515e60e0bfd431b44ced51b68507df)
- A dedicated BTC delta‑neutral funding‑rate thesis shows that realized returns depend strongly on rebalancing frequency and transaction costs but remain positive in many configurations. [digikogu.taltech](https://digikogu.taltech.ee/et/Download/dddbf9de-5dba-49dd-a2c8-60e87337731c)
- Exchange research and practitioner articles highlight how extreme funding episodes often precede large reversals and liquidation cascades. [binance](https://www.binance.com/en/square/post/35996851164730)

***

## 2. Intraday Time‑Series Momentum with Volatility Filter

### Conceptual edge

Time‑series momentum (TSMOM) – assets continuing in their own direction – is strongly documented in cryptocurrencies over horizons from days to weeks. [acfr.aut.ac](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
Intraday studies on BTC and other coins find both short‑horizon momentum and reversal patterns, and show that timing strategies based on intraday predictors outperform buy‑and‑hold. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)

### Market type & timeframe

- Market: BTC and ETH perpetual futures (or spot on major CEXs)
- Timeframe:
  - Bars: 15m or 1h OHLCV
  - Lookback: 4–24 hours
  - Holding: 2–8 bars

### Execution style

- Primarily liquidity taker; signals fire rarely (few times per day per asset).
- Use limit‑at‑touch orders when spreads are tight; fallback to market orders with capped participation.

### Indicators & features

Computed at close of bar \(t-1\):

- Return lookback: \(R_{t-1}(L) = P_{t-1} / P_{t-1-L} - 1\)
- 24h realized volatility of returns \(RV_{t-1}\)
- Volume filter: 24h dollar volume \(V_{t-1}\)
- Optional: funding rate sign to avoid fighting extreme leverage (\(F_{t-1}\))

### Exact trading rules

Example for 1h bars on BTC perp:

Parameters:

- Lookback \(L = 8\) hours.
- Return threshold: \(\theta_{\text{mom}} = 0.75 \times\) rolling 60‑day std of \(R(L)\) (z‑scored).
- Vol band: \(RV_{\min}, RV_{\max}\).
- Holding: \(H = 4\) bars.

**Entry – long:**

At bar close \(t-1\), if:

1. \(R_{t-1}(L) > \theta_{\text{mom}}\) (strong recent uptrend).
2. \(RV_{\min} \le RV_{t-1} \le RV_{\max}\).
3. 24h volume \(V_{t-1} \ge V_{\min}\).

Then at bar open \(t\): enter long position sized via volatility targeting.

**Entry – short:**

Symmetric: \(R_{t-1}(L) < -\theta_{\text{mom}}\) and same filters.

**Exit:**

At each bar open \(t\):

- Take profit: if trade P&L ≥ \(TP\%\) (e.g., 1.5× expected 1‑bar volatility).
- Stop loss: P&L ≤ \(-SL\%\) (e.g., −1× expected 1‑bar volatility).
- Time stop: bar age ≥ \(H\).
- Signal flip: opposite entry condition holds (e.g., long but now strong negative momentum) → flip direction (close and immediately open opposite).

### Regime detection / kill switch

- Kill switch when 24h realized volatility \(RV_{t-1} > RV_{\max}\) (hyper‑volatile regime where intraday patterns break down). [ideas.repec](https://ideas.repec.org/a/eee/ecofin/v62y2022ics1062940822000833.html)
- Optional macro filter: ignore signals during pre‑specified event windows (e.g., major FOMC) based on intraday predictability sensitivity. [papers.ssrn](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080253)

### Risk management & position sizing

- Volatility‑scaled notional per trade: \(size \propto \text{target\_risk} / RV_{30d}\).
- Cap open positions: at most one per asset and direction; optionally allow positions across BTC and ETH but cap total leverage.
- Equity risk: limit per‑trade VaR to ≤ 1–2% of equity; limit daily realized loss to 5%; if exceeded, stop trading for that day.

### No‑trade zone

- If bar high‑low / close > 2% (for 1h bars) or 4% (for 15m) → skip (likely illiquid jump or news).
- Skip if 24h dollar volume \(V_{t-1}\) below threshold (e.g., 10× your planned trade notional).
- If orderbook is available: skip when spread > 2 bps or when top‑of‑book depth < 5× trade size.

### Backtesting specification (pseudo‑code)

```pseudo
for each asset in {BTC, ETH}:
  for each bar t from lookback_end+1 to T:
    # features from bars up to t-1
    R_L = price[t-1] / price[t-1-L] - 1
    RV = realized_vol(returns[1..t-1], window=24h)
    V = dollar_volume[1..t-1 over 24h]

    if V < V_min or RV < RV_min or RV > RV_max:
      manage_existing_position_only()
      continue

    update_open_trade_PnL_including_costs()

    if position_open:
      if pnl >= TP or pnl <= -SL or age >= H or signal_flipped:
        close_at_open_with_costs()
        continue

    z_RL = (R_L - mean_RL_60d) / std_RL_60d
    if not position_open:
      if z_RL > z_thresh:
        size = vol_target_size(RV_30d)
        open_long(size)
      elif z_RL < -z_thresh:
        size = vol_target_size(RV_30d)
        open_short(size)
```

Include a per‑trade estimated round‑trip cost of at least 0.10–0.15% and reject configurations where average signal edge per trade is not at least 2× that.

### Parameters & anti‑overfitting

- Optimize: bar size (15m vs 1h), lookback \(L\), z‑threshold, holding \(H\), volatility band.
- Anti‑overfit:
  - Use coarse grids, not dense searches.
  - Require robustness across BTC/ETH and across bull/bear sub‑periods documented in TSMOM studies. [thesis.eur](https://thesis.eur.nl/pub/44390/Wisselink-NJ-483391-BA-thesis.pdf)
  - Use walk‑forward analysis and white‑out “event days” to ensure edges are not coming from a few outliers.

### Expected behavior

- Performs best in sustained intraday trends with moderate volatility and good liquidity. [scribd](https://www.scribd.com/document/934766694/Intraday-return-predictability-in-the-cryptocurrency-markets-momentum-reversal-or-both)
- Fails in choppy, mean‑reverting intraday regimes or during large jump events where stop‑outs dominate; also sensitive to fee and slippage on low‑timeframe bars.

### Supporting evidence

- Multiple studies document strong time‑series momentum in cryptocurrencies, including BTC‑centric and multi‑asset analyses. [semanticscholar](https://www.semanticscholar.org/paper/Time-series-Momentum-in-the-Cryptocurrency-Market-Wisselink/d09b4119b161b7050882229a65abaaada8ce5f65)
- Intraday momentum and reversal patterns and profitable timing strategies are specifically shown in “Intraday Return Predictability in the Cryptocurrency Markets.” [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940822000833)

***

## 3. Leverage Blow‑Off & Liquidation Reversal

### Conceptual edge

Crypto crashes are often driven by mechanical liquidation cascades when leverage becomes extremely one‑sided; liquidations create forced flows and temporary price dislocations rather than purely information‑driven repricing. [blog.amberdata](https://blog.amberdata.io/leverage-liquidations-the-31b-deleveraging)
Open interest peaks combined with extreme funding rates have preceded some of the largest deleveraging events, indicating that OI and funding can be used as fragility indicators. [blog.amberdata](https://blog.amberdata.io/leverage-liquidations-the-31b-deleveraging)

### Market type & timeframe

- Market: BTC and ETH perpetual futures (optionally others with deep derivatives markets)
- Timeframe:
  - Funding & OI: 1h snapshots
  - Price bars: 5m–15m
  - Holding: hours to 1–2 days

### Execution style

- Liquidity taker for entry/exit (speed is important around cascades).
- Reduce participation: cap each order to small percentage of current interval volume to limit slippage.

### Indicators & features

At time \(t-1\):

- Funding rate \(F_{t-1}\) (annualized).
- Open interest \(OI_{t-1}\) and its z‑score vs trailing 60‑day.
- Recent price return \(R_{t-1}(L)\) over 24–72 hours.
- Change in OI: \(\Delta OI_{t-1} = OI_{t-1} - OI_{t-1-24h}\).
- 24h realized volatility \(RV_{t-1}\).

### Exact trading rules

We define:

- Extreme long‑crowding (vulnerable to long squeeze):
  - \(F_{t-1} \ge F_{\text{crowd}}^{+}\) (e.g., > 20–30% APR).
  - \(z_{\text{OI},t-1} \ge +2\).
  - \(R_{t-1}(48h) > +15\%\).
- Extreme short‑crowding (vulnerable to short squeeze): symmetric with negative funding, OI, and price moves.

**Entry – short after long squeeze onset:**

At bar close \(t-1\) (e.g., 5m bar), if:

1. Long‑crowding conditions held over the prior 24–48h.
2. A breakdown trigger occurs:
   - Price has fallen by at least \(-X\%\) from the recent 24h high (e.g., \(X = 5\%\)), and
   - \(\Delta OI_{t-1} \le -Y\%\) of peak OI (e.g., > 10% OI reduction in 24h).

Then at time \(t\): enter short perp with notional sized via volatility targeting.

**Entry – long after short squeeze / capitulation:**

Symmetric:

- Extreme negative funding (e.g., \(\le -20\%\) APR), high negative OI z‑score then sharp upside move with OI reduction and evidence of short liquidations.
- Alternatively, after long squeeze crash, if funding flips negative and OI collapses > 30%, a mean‑reversion long can be considered.

**Exit:**

- Take profit when price rebounds (for longs) or extends down (for shorts) by \(TP\%\) relative to entry (e.g., 3–6%).
- Stop loss: fixed fractional move against you (e.g., 2–3%).
- Time stop: exit after 24h if neither TP nor SL hit.
- Optional: exit when funding returns to neutral range (e.g., between −5% and +5% APR) and OI returns within 1σ of 60‑day mean.

### Regime detection / kill switch

- Kill switch if OI is near historical lows (no leverage; nothing to squeeze).
- Kill switch if you detect exchange outages or derivatives halts.
- Kill switch if daily realized volatility > very high threshold (e.g., 250% annualized), indicating regime where entry/exit slippage is extreme.

### Risk management & position sizing

- Use small per‑trade risk (e.g., ≤ 0.5–1% of equity at SL).
- Limit concurrent positions: at most one direction per asset; total leverage capped.
- Enforce cool‑down: after a losing trade tied to a given burst, do not re‑enter for \(N\) hours to avoid chop.

### No‑trade zone

- Skip if bid‑ask spread in perp is > 5 bps or > 3× 30‑day median.
- Skip if top‑of‑book depth < 10× your intended trade size.
- Skip small cap alts: restrict to BTC, ETH, and maybe a small basket of most liquid perpetuals.

### Backtesting specification (pseudo‑code)

```pseudo
for each asset in {BTC, ETH}:
  precompute OI_z[t], funding[t], RV[t], OI_change[t], R_48h[t]

  for each bar t (5m) from start+lookback to T:
    update_PnL_and_apply_costs()

    if position_open:
      if pnl >= TP or pnl <= -SL or age >= 24h or (funding_in_neutral_band and OI_z < 1):
        close_position()
      continue

    # check crowding regime based on data up to t-1
    if funding[t-1] >= F_crowd_pos and OI_z[t-1] >= 2 and R_48h[t-1] > 0.15:
      if price[t-1] <= (1 - X%) * rolling_24h_high and OI_change_24h[t-1] <= -Y%:
        if spread_ok and depth_ok:
          size = vol_target_size(RV_30d)
          open_short(size)
    elif funding[t-1] <= F_crowd_neg and OI_z[t-1] >= 2 and R_48h[t-1] < -0.15:
      if price[t-1] >= (1 + X%) * rolling_24h_low and OI_change_24h[t-1] <= -Y%:
        if spread_ok and depth_ok:
          size = vol_target_size(RV_30d)
          open_long(size)
```

### Parameters & anti‑overfitting

- Optimize: thresholds for funding extremes, OI z‑score, price break %, OI drop %, TP/SL.
- Anti‑overfit:
  - Use long sample including multiple cascades and quiet periods.
  - Test robustness across BTC and ETH, and across exchanges.
  - Use simple, coarse thresholds and avoid fitting to individual events documented in narrative case studies. [cryptorank](https://cryptorank.io/news/feed/0972c-crypto-futures-liquidations-market-panic)

### Expected behavior

- Best: at the onset and early phase of liquidation cascades when pricing is still inefficient but liquidity has not fully evaporated. [yellow](https://yellow.com/learn/how-to-read-funding-rates-crypto-reversals)
- Fails: during slow grind trends without sharp leverage imbalances, or if exchanges change margin rules and dampen cascade dynamics.

### Supporting evidence

- Analyses of large deleveraging events show record OI accumulation, extreme positive funding, and then massive long liquidations coinciding with sharp crashes. [cryptorank](https://cryptorank.io/news/feed/0972c-crypto-futures-liquidations-market-panic)
- Exchange and research content emphasize negative funding plus collapsing OI as signals of bull capitulation and leverage washout, consistent with mean‑reversion opportunities. [binance](https://www.binance.com/en/square/post/35996851164730)

***

## 4. Cross‑Exchange Perpetual Spread Arbitrage

### Conceptual edge

Crypto markets are fragmented across exchanges with capital controls, settlement latency, and differential liquidity, leading to persistent cross‑exchange price discrepancies that are not fully arbitraged away. [osuva.uwasa](https://osuva.uwasa.fi/bitstream/handle/10024/19882/Uwasa_2025_Ruhanen_Samuel.pdf?sequence=2)
Studies document significant and persistent cross‑exchange inefficiencies and arbitrage spreads, limited by settlement latency and default risk, but still exploitable with appropriate risk controls. [arxiv](https://arxiv.org/abs/2501.17335)

### Market type & timeframe

- Market: BTC and ETH perpetual futures across 2–3 major CEXs (e.g., Exchange A vs B).
- Timeframe:
  - Tick / 1s–1m midprice snapshots for spread measurement.
  - Trades held minutes to hours until spreads mean‑revert.

### Execution style

- Prefer passive maker orders (post‑only) on both venues, to reduce fee drag.
- In emergencies (e.g., exchange outage risk), allow taker exit.

### Indicators & features

For each timestamp \(t\), per asset:

- Midprice on exchange A: \(m_A(t)\), on B: \(m_B(t)\).
- Spread: \(S(t) = (m_A(t) - m_B(t)) / m_B(t)\).
- Rolling mean \(\mu_S\) and std \(\sigma_S\) of spread over last 3–7 days.
- Orderbook depth at best 1–3 levels on each venue.
- Fee schedules and maker rebates per venue.

### Exact trading rules

Assume you treat one venue as “cheap” and one as “expensive” at time \(t\).

**Entry – basic convergence trade:**

At time \(t\), using only data up to \(t\):

1. Compute standardized spread \(z_S(t) = (S(t) - \mu_S) / \sigma_S\).
2. If \(z_S(t) > z_{\text{high}}\) (e.g., +2):

   - Short perp on expensive venue (A).
   - Long perp on cheap venue (B).
   - Size: proportional to \(z_S(t)\) up to cap, but adjusted for depth.

3. If \(z_S(t) < -z_{\text{high}}\): opposite (long on A, short on B).

Ensure:

- Depth at best 3 levels on both venues ≥ \(k \times\) desired trade size.
- Spreads ≤ max_spread threshold.

**Exit:**

At each time step:

- Close both legs when \(|z_S(t)| < z_{\text{close}}\) (e.g., 0).
- Hard time stop: after \(T_{\max}\) (e.g., 6 hours) if no convergence.
- Stop loss: if mark‑to‑market loss of the pair (using midprices) exceeds \(-L\%\) of allocated capital.

### Regime detection / kill switch

- Disable when cross‑exchange spreads structurally compress (e.g., median \(|S|\) below 2–3× total transaction cost).
- Disable if estimated settlement latency or withdrawal delays spike (based on chain congestion or status pages). [osuva.uwasa](https://osuva.uwasa.fi/bitstream/handle/10024/19882/Uwasa_2025_Ruhanen_Samuel.pdf?sequence=2)
- Disable if counterparty risk (default or regulatory) spikes for any venue.

### Risk management & position sizing

- Always delta‑neutral: equal notional on both legs.
- Cap exposure per pair to a fraction of available margin on each exchange.
- Maintain diversified collateral and pre‑deployed inventory to avoid transfer delays.
- Apply portfolio‑level drawdown limits as for other strategies.

### No‑trade zone

- If spread \(|S|\) < 1.5× all‑in round‑trip cost (fees + slippage) → no trade.
- If orderbook depth insufficient or spreads wide on either leg.
- Avoid low‑liquidity hours where depth collapses (e.g., weekends, certain time zones).

### Backtesting specification (pseudo‑code)

```pseudo
for each asset:
  compute spread_series S[t] from mid_A[t], mid_B[t]
  compute rolling mu_S[t], sigma_S[t]

  for each timestamp t:
    zS = (S[t] - mu_S[t]) / sigma_S[t]

    update_pair_PnL_with_costs()

    if pair_open:
      if abs(zS) < z_close or age > T_max or pnl <= -L%:
        close_both_legs()
      continue

    cost_floor = 1.5 * est_roundtrip_cost()
    if abs(S[t]) < cost_floor: continue
    if depth_A[t] < k*size_unit or depth_B[t] < k*size_unit: continue

    if zS > z_high:
      size = compute_size(zS, depth_A, depth_B)
      short_perp_on_A(size, maker_preferred)
      long_perp_on_B(size, maker_preferred)
    elif zS < -z_high:
      size = compute_size(...)
      long_perp_on_A(size)
      short_perp_on_B(size)
```

### Parameters & anti‑overfitting

- Optimize: window length for \(\mu_S, \sigma_S\), \(z_{\text{high}}\), \(z_{\text{close}}\), \(T_{\max}\).
- Anti‑overfit:
  - Use overlapping multi‑year data with regime changes in fiat rails and competition.
  - Ensure P&L is not dominated by a few outlier days.
  - Validate on multiple exchange pairs, not a single A/B pair. [en.wiwi.uni-paderborn](https://en.wiwi.uni-paderborn.de/fileadmin-wiwi/cetar/TAF_Working_Paper_Series/TAF_WP_067_CrepellierePelsterZeisberger2022_rev.pdf)

### Expected behavior

- Best: persistent but mean‑reverting spreads driven by frictions (fiat ramps, capital controls, exchange‑specific liquidity). [en.wiwi.uni-paderborn](https://en.wiwi.uni-paderborn.de/fileadmin-wiwi/cetar/TAF_Working_Paper_Series/TAF_WP_067_CrepellierePelsterZeisberger2022_rev.pdf)
- Fails: when spread reflects genuine credit/solvency risk of one venue or when arbitrage capital saturates and pays away all profits in fees.

### Supporting evidence

- Empirical studies confirm persistent cross‑exchange price inefficiencies in crypto and discuss settlement latency and default risk as major frictions limiting arbitrage. [osuva.uwasa](https://osuva.uwasa.fi/bitstream/handle/10024/19882/Uwasa_2025_Ruhanen_Samuel.pdf?sequence=2)
- Broader arbitrage research finds similar patterns across CEXs and cross‑chain DEXs, with documented profitability of non‑atomic arbitrage sequences. [arxiv](https://arxiv.org/abs/2501.17335)

***

## 5. BTC Volatility Risk Premium Harvest (Options)

### Conceptual edge

Bitcoin options exhibit a sizable and persistent variance risk premium: implied variance tends to exceed realized variance by a margin larger than in equity index markets, implying that systematically selling volatility can earn a premium. [acfr.aut.ac](https://acfr.aut.ac.nz/__data/assets/pdf_file/0006/969378/950002_Atanasova_Illiquidity-Premium-and-Crypto-Option-Returns.pdf)
Empirical analyses show a consistently negative volatility risk premium (implied minus realized) and high implied vol levels with pronounced skew in BTC options. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8418903/)

### Market type & timeframe

- Market: BTC (and maybe ETH) options on liquid venues (e.g., Deribit) plus underlying spot/perp for hedging.
- Timeframe:
  - Signal: 1× per day.
  - Instruments: 7–30 day ATM or slightly OTM options.
  - Rebalancing: once or a few times per day for delta hedging.

### Execution style

- Primarily liquidity taker for deltas; try to be maker for option orders (resting limit orders near mid).
- Conservative sizing because of tail risk.

### Indicators & features

At decision time \(t-1\):

- Implied volatility \(IV_{t-1}\) for selected tenor (e.g., 7‑day ATM).
- Realized volatility \(RV_{t-1}\) over past \(W\) days (e.g., 7 or 14).
- Volatility risk premium estimate: \(VRP_{t-1} = IV_{t-1} - RV_{t-1}\).
- Regime classification from studies: volatility regimes with different VRP behavior. [arxiv](https://arxiv.org/html/2410.15195v2)

### Exact trading rules

Example: short 7‑day delta‑hedged straddle on BTC:

**Entry:**

At daily close \(t-1\):

1. Compute \(IV_{t-1}\) (ATM 7‑day) and \(RV_{t-1}\) (realized over past 7 days).
2. If \(VRP_{t-1} = IV_{t-1} - RV_{t-1} \ge \theta_{VRP}\) (e.g., ≥ 10 vol points) and  
   \(IV_{t-1}\) not at historical extremes (e.g., within 5–95th percentile of last 2 years to avoid extreme crash regimes):

   - Sell 1× ATM call + 1× ATM put (same expiry).
   - Delta‑hedge immediately: adjust underlying so combined position is delta‑neutral.

**Daily management:**

Each day until expiry or forced exit:

- Recompute portfolio delta and re‑hedge to near zero using spot or perp.
- Monitor P&L; if loss exceeds \(-L\%\) of capital allocated to this trade (e.g., −5%), close entire position.
- Optionally partially take profit when realized theta + vega gains exceed \(+TP\%\) (e.g., +2–3%) and VRP has narrowed.

**Exit:**

- Natural expiry: let options expire; unwind delta hedge.
- Early exit on stop loss or if VRP flips negative (i.e., realized vol exceeds IV by margin).

### Regime detection / kill switch

- Kill switch during “high‑vol regime” where VRP is lower or unstable: e.g., when regime classification (based on clustering of implied distributions) indicates high‑vol cluster, in which VRP may be smaller. [linkedin](https://www.linkedin.com/posts/namnguyento_bitcoin-volatility-riskmanagement-activity-7385316324916461568-s2eL)
- Kill around known binary events (e.g., major ETF decisions, protocol forks).
- Kill when options orderbook illiquidity is high (wide spreads, low volume).

### Risk management & position sizing

- Allocate only a small fraction of portfolio (e.g., 5–10%) to short‑vol strategies due to tail risk.
- Use margin and notional limits per trade.
- Consider buying cheap OTM wings (turning straddle into an iron fly) to cap tail risk, at the cost of some carry.

### No‑trade zone

- Do not trade when implied vol is at or near all‑time highs or lows: edges may be regime‑dependent and asymmetrical. [arxiv](https://arxiv.org/html/2410.15195v2)
- Avoid tenors or strikes with wide bid‑ask spreads (e.g., > 5–10 vol points).
- Avoid days with extremely low underlying liquidity (as measured by underlying volume).

### Backtesting specification (pseudo‑code)

```pseudo
for each day t:
  compute IV = atm_iv_7d[t-1]
  RV = realized_vol(returns[1..t-1], window=7d)
  VRP = IV - RV

  classify_vol_regime()  # based on historical clustering

  update_existing_trades_delta_hedge_and_PnL()

  if new_trade_allowed and not in_high_vol_regime and VRP >= VRP_thresh \
     and IV in [IV_p5, IV_p95]:
    size = capital * risk_fraction / option_vega
    sell_atm_call_put(size)  # prefer maker
    delta_hedge(size)

  for each open_trade:
    if loss < -L% or near_expiry or VRP < 0:
      close_options_and_delta()
```

Include transaction costs for options (fees + bid‑ask) and underlying hedges; enforce at least 0.1–0.2% expected premium per day of vega‑weighted exposure.

### Parameters & anti‑overfitting

- Optimize: VRP threshold, lookback window for RV, regime rules, TP/SL levels.
- Anti‑overfit:
  - Use multi‑year Deribit data covering different regimes. [linkedin](https://www.linkedin.com/posts/namnguyento_bitcoin-volatility-riskmanagement-activity-7385316324916461568-s2eL)
  - Evaluate separately in low‑ and high‑volatility clusters.
  - Focus on robust rules (e.g., trade only when VRP in top decile of history) instead of precise numeric fits.

### Expected behavior

- Best: in low‑volatility regimes where VRP is high and implied vol tends to overstate realized vol. [arxiv](https://arxiv.org/html/2410.15195v2)
- Fails: during volatility spikes, structural breaks, or crash periods when realized volatility massively exceeds implied, leading to large losses.

### Supporting evidence

- Studies document that BTC options have a significantly higher volatility and variance risk premium than S&P 500, with implied variance persistently above realized. [acfr.aut.ac](https://acfr.aut.ac.nz/__data/assets/pdf_file/0006/969378/950002_Atanasova_Illiquidity-Premium-and-Crypto-Option-Returns.pdf)
- Work on BTC options stylized facts shows high and skewed implied vol levels, consistent with expensive protection. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8418903/)

***

## Strategy Comparison & Recommendations

### Comparative ranking

**Ease of implementation (1 = easiest):**

1. Intraday Time‑Series Momentum (Strategy 2) – needs only OHLCV, simplest execution logic.  
2. Delta‑Neutral Funding Harvest (Strategy 1) – uses funding/OI data but otherwise straightforward.  
3. Leverage Blow‑Off Reversal (Strategy 3) – requires integrating funding, OI, and intraday prices.  
4. Cross‑Exchange Perp Arbitrage (Strategy 4) – multi‑venue, inventory and operational complexity. [en.wiwi.uni-paderborn](https://en.wiwi.uni-paderborn.de/fileadmin-wiwi/cetar/TAF_Working_Paper_Series/TAF_WP_067_CrepellierePelsterZeisberger2022_rev.pdf)
5. Volatility Risk Premium Harvest (Strategy 5) – options, greeks, and hedging complexity. [acfr.aut.ac](https://acfr.aut.ac.nz/__data/assets/pdf_file/0006/969378/950002_Atanasova_Illiquidity-Premium-and-Crypto-Option-Returns.pdf)

### Data availability

- OHLCV and funding/OI (Strategies 1–3) are widely available from major derivatives venues. [aeaweb](https://www.aeaweb.org/conference/2026/program/paper/ByyFEfr4)
- Cross‑exchange midprice and depth (Strategy 4) require multi‑venue APIs and careful synchronization but are still public. [osuva.uwasa](https://osuva.uwasa.fi/bitstream/handle/10024/19882/Uwasa_2025_Ruhanen_Samuel.pdf?sequence=2)
- High‑quality options chains and implied vol (Strategy 5) are available but more specialized and sometimes paywalled. [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC8418903/)

### Likelihood of robustness in crypto

- **High robustness:**  
  - Strategy 1 (Funding Harvest): based on structural funding mechanism and leverage imbalances.  
  - Strategy 4 (Cross‑Exchange Arb): exploits fragmentation and frictions that are slow to disappear. [arxiv](https://arxiv.org/abs/2501.17335)

- **Moderate robustness:**  
  - Strategy 2 (TSMOM): time‑series momentum is well‑documented but can decay as more capital trades it. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1062940821000590)
  - Strategy 3 (Leverage Blow‑Off): depends on risk‑management practices and margin rules, which can change.

- **More fragile:**  
  - Strategy 5 (Short Vol Premium): statistically robust VRP, but payoff is negatively skewed and very sensitive to tail events and microstructure of options markets. [linkedin](https://www.linkedin.com/posts/namnguyento_bitcoin-volatility-riskmanagement-activity-7385316324916461568-s2eL)

### Which strategy to test first?

- **Recommended first test:**  
  - **Delta‑Neutral Funding Rate Harvest (Strategy 1).**  
    - Strong structural justification, extensive empirical and theoretical backing on perps and funding. [semanticscholar](https://www.semanticscholar.org/paper/Fundamentals-of-Perpetual-Futures-He-Manela/82c0c4eb57515e60e0bfd431b44ced51b68507df)
    - Uses relatively clean data and avoids predicting direction; easier to diagnose and risk‑manage.

- **Second candidate:**  
  - **Intraday Time‑Series Momentum (Strategy 2)** to build your pipeline for intraday OHLCV and vol‑scaled execution, leveraging well‑documented momentum anomalies. [acfr.aut.ac](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)

### Most prone to overfitting

- **Leverage Blow‑Off Reversal (Strategy 3)** is most prone to overfitting because it often relies on a small number of dramatic cascade episodes, making thresholds easy to tune to specific historical crashes. [blog.amberdata](https://blog.amberdata.io/leverage-liquidations-the-31b-deleveraging)
- **Volatility Risk Premium Harvest (Strategy 5)** can also be overfit via tenor, strike, and regime filters; it should be designed with very coarse, simple rules and tested across many regimes. [arxiv](https://arxiv.org/html/2410.15195v2)

These five strategies give you a diversified set of structurally motivated edges – carry, trend, mean‑reversion around liquidations, fragmentation arbitrage, and volatility premium – all expressible as explicit, backtestable rules with realistic friction and regime filters.