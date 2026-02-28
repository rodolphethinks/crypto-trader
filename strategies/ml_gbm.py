"""
XGBoost/LightGBM classification strategy.

Uses gradient-boosted trees to predict next-bar direction from 80+ features.
Trains on a walk-forward basis: at each point in time, only uses past data
for training (no look-ahead bias).

Two modes:
1. 'expanding' — train on all data up to bar N, predict bar N+1
2. 'rolling' — train on last `train_window` bars, predict next bar

The strategy produces BUY/SELL signals when predicted probability exceeds
a confidence threshold, with built-in stop-loss and take-profit from ATR.
"""
import logging
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from strategies.base import BaseStrategy, Signal
from indicators.features import compute_all_features, get_feature_columns
from indicators.regime import detect_regime_features

logger = logging.getLogger(__name__)


class GBMStrategy(BaseStrategy):
    """Gradient-boosted tree strategy (XGBoost or LightGBM)."""

    name = "GBM_Classifier"
    description = "ML gradient boosting for next-bar direction prediction"
    version = "1.0"

    default_params = {
        "model_type": "xgboost",        # 'xgboost' or 'lightgbm'
        "train_window": 500,             # bars for rolling training window
        "min_train_bars": 200,           # minimum bars before first prediction
        "retrain_every": 50,             # retrain every N bars
        "predict_horizon": 1,            # predict N bars ahead
        "buy_threshold": 0.55,           # predicted P(up) > this => BUY
        "sell_threshold": 0.45,          # predicted P(up) < this => SELL
        "sl_atr_mult": 2.0,             # stop loss = N * ATR
        "tp_atr_mult": 3.0,             # take profit = N * ATR
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "use_regime": True,              # add regime features
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train incrementally and generate signals with no look-ahead."""
        params = self.params

        # Compute features
        feat_df = compute_all_features(df, include_time=True)
        if params.get("use_regime", True):
            regime_df = detect_regime_features(df)
            for col in ["regime_trend", "regime_vol", "regime_trend_duration",
                        "regime_vol_duration", "regime_composite"]:
                if col in regime_df.columns:
                    feat_df[col] = regime_df[col]

        feature_cols = get_feature_columns(feat_df)
        
        # Keep ATR for SL/TP
        atr_14 = pd.Series(index=df.index, dtype=float)
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift(1)).abs(),
            "lc": (df["low"] - df["close"].shift(1)).abs(),
        }).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        # Initialize output
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5
        signals["predicted_prob"] = 0.5

        # Prepare feature matrix
        X_all = feat_df[feature_cols].copy()
        y_all = feat_df["target_class_1"].copy()

        # Drop rows where target is NaN (last row)
        valid_mask = y_all.notna() & X_all.notna().all(axis=1)

        min_train = params["min_train_bars"]
        retrain_every = params["retrain_every"]
        train_window = params["train_window"]
        buy_thresh = params["buy_threshold"]
        sell_thresh = params["sell_threshold"]
        sl_mult = params["sl_atr_mult"]
        tp_mult = params["tp_atr_mult"]

        model = None
        scaler = StandardScaler()
        last_train_idx = 0
        n_bars = len(df)

        for i in range(min_train, n_bars):
            # Retrain periodically
            if model is None or (i - last_train_idx) >= retrain_every:
                # Training data: only PAST bars
                train_start = max(0, i - train_window)
                train_end = i  # exclusive — no look-ahead

                train_mask = valid_mask.iloc[train_start:train_end]
                X_train = X_all.iloc[train_start:train_end][train_mask].values
                y_train = y_all.iloc[train_start:train_end][train_mask].values

                if len(X_train) < 50:
                    continue

                # Replace inf/nan
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

                scaler.fit(X_train)
                X_train_scaled = scaler.transform(X_train)

                model = self._build_model(params)
                try:
                    model.fit(X_train_scaled, y_train)
                    last_train_idx = i
                except Exception as e:
                    logger.debug(f"Training failed at bar {i}: {e}")
                    model = None
                    continue

            # Predict current bar
            if model is not None:
                X_now = X_all.iloc[i:i+1].values
                X_now = np.nan_to_num(X_now, nan=0.0, posinf=0.0, neginf=0.0)
                X_now_scaled = scaler.transform(X_now)

                try:
                    prob = model.predict_proba(X_now_scaled)[0]
                    p_up = prob[1] if len(prob) > 1 else prob[0]
                except Exception:
                    p_up = 0.5

                signals.iloc[i, signals.columns.get_loc("predicted_prob")] = p_up
                signals.iloc[i, signals.columns.get_loc("confidence")] = abs(p_up - 0.5) * 2

                current_atr = atr_14.iloc[i] if not np.isnan(atr_14.iloc[i]) else 0
                current_close = df["close"].iloc[i]

                if p_up > buy_thresh and current_atr > 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = current_close - sl_mult * current_atr
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = current_close + tp_mult * current_atr
                elif p_up < sell_thresh and current_atr > 0:
                    signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                    signals.iloc[i, signals.columns.get_loc("stop_loss")] = current_close + sl_mult * current_atr
                    signals.iloc[i, signals.columns.get_loc("take_profit")] = current_close - tp_mult * current_atr

        return signals

    @staticmethod
    def _build_model(params: dict):
        """Build the gradient boosting model."""
        model_type = params.get("model_type", "xgboost")
        n_est = params.get("n_estimators", 200)
        depth = params.get("max_depth", 4)
        lr = params.get("learning_rate", 0.05)

        if model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                random_state=42,
                verbose=-1,
                n_jobs=1,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
            )
        else:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=n_est,
                max_depth=depth,
                learning_rate=lr,
                random_state=42,
                verbosity=0,
                n_jobs=1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                use_label_encoder=False,
            )


class XGBoostStrategy(GBMStrategy):
    """XGBoost classifier strategy."""
    name = "XGBoost"
    default_params = {**GBMStrategy.default_params, "model_type": "xgboost"}


class LightGBMStrategy(GBMStrategy):
    """LightGBM classifier strategy."""
    name = "LightGBM"
    default_params = {**GBMStrategy.default_params, "model_type": "lightgbm"}
