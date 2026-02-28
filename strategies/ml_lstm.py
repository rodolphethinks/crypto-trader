"""
LSTM (Long Short-Term Memory) strategy.

Uses a PyTorch LSTM network to predict next-bar direction from
sequential feature data. Trains walk-forward with no look-ahead.

Architecture:
  Input (seq_len × n_features) → LSTM → Dropout → FC → Sigmoid → P(up)

GPU-accelerated on CUDA when available.
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal
from indicators.features import compute_all_features, get_feature_columns
from indicators.regime import detect_regime_features

logger = logging.getLogger(__name__)

# Lazy torch import — only when needed
_torch = None
_nn = None
_device = None


def _init_torch():
    global _torch, _nn, _device
    if _torch is None:
        import torch
        import torch.nn as nn
        _torch = torch
        _nn = nn
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"LSTM using device: {_device}")


class _LSTMModel:
    """Wrapper around a PyTorch LSTM classifier."""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 seq_len: int = 20, lr: float = 0.001, epochs: int = 30):
        _init_torch()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.net = None
        self.scaler_mean = None
        self.scaler_std = None

    def _build_net(self):
        """Build the LSTM network."""

        class LSTMNet(_nn.Module):
            def __init__(self2, n_feat, hidden, n_layers, drop):
                super().__init__()
                self2.lstm = _nn.LSTM(
                    input_size=n_feat, hidden_size=hidden,
                    num_layers=n_layers, batch_first=True,
                    dropout=drop if n_layers > 1 else 0.0
                )
                self2.dropout = _nn.Dropout(drop)
                self2.fc = _nn.Linear(hidden, 1)
                self2.sigmoid = _nn.Sigmoid()

            def forward(self2, x):
                lstm_out, _ = self2.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                out = self2.dropout(last_hidden)
                out = self2.fc(out)
                return self2.sigmoid(out).squeeze(-1)

        self.net = LSTMNet(self.n_features, self.hidden_size,
                           self.num_layers, self.dropout).to(_device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the LSTM on sequence data.
        
        X: (N, n_features) — will be converted to sequences
        y: (N,) — binary labels
        """
        # Normalize
        self.scaler_mean = np.nanmean(X, axis=0)
        self.scaler_std = np.nanstd(X, axis=0)
        self.scaler_std[self.scaler_std == 0] = 1.0
        X_norm = (X - self.scaler_mean) / self.scaler_std
        X_norm = np.nan_to_num(X_norm, 0.0)

        # Create sequences
        X_seq, y_seq = self._make_sequences(X_norm, y)
        if len(X_seq) < 10:
            return

        self._build_net()
        
        X_t = _torch.FloatTensor(X_seq).to(_device)
        y_t = _torch.FloatTensor(y_seq).to(_device)

        optimizer = _torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = _nn.BCELoss()

        self.net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.net(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability for the last sequence window.
        
        X: (seq_len, n_features) or (N, n_features) where N >= seq_len
        """
        if self.net is None or self.scaler_mean is None:
            return np.array([[0.5, 0.5]])

        X_norm = (X - self.scaler_mean) / self.scaler_std
        X_norm = np.nan_to_num(X_norm, 0.0)

        # Take last seq_len rows
        if len(X_norm) >= self.seq_len:
            X_seq = X_norm[-self.seq_len:].reshape(1, self.seq_len, -1)
        else:
            # Pad with zeros
            pad = np.zeros((self.seq_len - len(X_norm), X_norm.shape[1]))
            X_seq = np.vstack([pad, X_norm]).reshape(1, self.seq_len, -1)

        X_t = _torch.FloatTensor(X_seq).to(_device)

        self.net.eval()
        with _torch.no_grad():
            p_up = self.net(X_t).cpu().numpy()[0]

        return np.array([[1 - p_up, p_up]])

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        """Convert flat arrays to (N_seq, seq_len, features) sequences."""
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X)):
            X_seq.append(X[i - self.seq_len:i])
            y_seq.append(y[i])
        return np.array(X_seq) if X_seq else np.array([]), np.array(y_seq) if y_seq else np.array([])


class LSTMStrategy(BaseStrategy):
    """LSTM neural network strategy for next-bar prediction."""

    name = "LSTM"
    description = "LSTM sequence model for direction prediction (GPU-accelerated)"
    version = "1.0"

    default_params = {
        "train_window": 500,
        "min_train_bars": 250,
        "retrain_every": 100,
        "seq_len": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "epochs": 30,
        "lr": 0.001,
        "buy_threshold": 0.55,
        "sell_threshold": 0.45,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.0,
        "use_regime": True,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train LSTM walk-forward and generate signals."""
        _init_torch()  # ensure torch loaded
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

        # ATR for SL/TP
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift(1)).abs(),
            "lc": (df["low"] - df["close"].shift(1)).abs(),
        }).max(axis=1)
        atr_14 = tr.rolling(14).mean()

        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5
        signals["predicted_prob"] = 0.5

        X_all = feat_df[feature_cols].values
        y_all = feat_df["target_class_1"].values

        min_train = params["min_train_bars"]
        retrain_every = params["retrain_every"]
        train_window = params["train_window"]
        seq_len = params["seq_len"]
        buy_thresh = params["buy_threshold"]
        sell_thresh = params["sell_threshold"]
        sl_mult = params["sl_atr_mult"]
        tp_mult = params["tp_atr_mult"]

        model = None
        last_train_idx = 0
        n_bars = len(df)

        for i in range(min_train, n_bars):
            if model is None or (i - last_train_idx) >= retrain_every:
                train_start = max(0, i - train_window)
                train_end = i

                X_tr = X_all[train_start:train_end]
                y_tr = y_all[train_start:train_end]

                # Filter NaN targets
                valid = ~np.isnan(y_tr)
                X_tr = np.nan_to_num(X_tr[valid], 0.0)
                y_tr = y_tr[valid]

                if len(X_tr) < seq_len + 20:
                    continue

                model = _LSTMModel(
                    n_features=len(feature_cols),
                    hidden_size=params["hidden_size"],
                    num_layers=params["num_layers"],
                    dropout=params["dropout"],
                    seq_len=seq_len,
                    lr=params["lr"],
                    epochs=params["epochs"],
                )
                try:
                    model.fit(X_tr, y_tr)
                    last_train_idx = i
                except Exception as e:
                    logger.debug(f"LSTM training failed at bar {i}: {e}")
                    model = None
                    continue

            if model is not None and i >= seq_len:
                # Predict using last seq_len bars of features
                X_window = X_all[max(0, i - seq_len + 1):i + 1]
                X_window = np.nan_to_num(X_window, 0.0)

                try:
                    prob = model.predict_proba(X_window)
                    p_up = prob[0][1]
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
