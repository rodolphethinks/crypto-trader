"""
V7 ML/DL Strategies — GPU-accelerated deep learning strategies.

1. LSTMTrend       — Bidirectional LSTM trend predictor
2. CNNMomentum     — 1D-CNN on multi-feature candle windows
3. TransformerPred — Lightweight transformer for price direction

All models train on a rolling window of historical data and generate
signals based on predicted direction + confidence threshold.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from strategies.base import BaseStrategy, Signal

# Lazy torch imports (only loaded when actually used)
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
    return _torch, _nn, _device


# ── Feature Engineering ──────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """Build normalized feature matrix from OHLCV data."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    
    # Returns at multiple scales
    ret1 = np.zeros_like(close)
    ret1[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)
    
    ret5 = np.zeros_like(close)
    ret5[5:] = (close[5:] - close[:-5]) / (close[:-5] + 1e-10)
    
    ret20 = np.zeros_like(close)
    ret20[20:] = (close[20:] - close[:-20]) / (close[:-20] + 1e-10)
    
    # Volatility (rolling std of returns)
    vol10 = pd.Series(ret1).rolling(10).std().fillna(0).values
    vol30 = pd.Series(ret1).rolling(30).std().fillna(0).values
    
    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean().fillna(0).values
    avg_loss = pd.Series(loss).rolling(14).mean().fillna(1e-10).values
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    rsi_norm = (rsi - 50) / 50  # Center around 0
    
    # Volume ratio
    vol_avg = pd.Series(volume).rolling(20).mean().fillna(1).values
    vol_ratio = volume / (vol_avg + 1e-10)
    vol_ratio_norm = np.clip(vol_ratio - 1, -3, 3) / 3
    
    # High-low range normalized
    hl_range = (high - low) / (close + 1e-10)
    hl_avg = pd.Series(hl_range).rolling(20).mean().fillna(0.01).values
    hl_ratio = hl_range / (hl_avg + 1e-10) - 1
    
    # EMA crossovers 
    ema8 = pd.Series(close).ewm(span=8).mean().values
    ema21 = pd.Series(close).ewm(span=21).mean().values
    ema_cross = (ema8 - ema21) / (close + 1e-10)
    
    # OBV momentum
    obv_sign = np.sign(np.diff(close, prepend=close[0]))
    obv = np.cumsum(obv_sign * volume)
    obv_ema = pd.Series(obv).ewm(span=20).mean().values
    obv_diff = (obv - obv_ema) / (np.abs(obv_ema) + 1e-10)
    obv_diff = np.clip(obv_diff, -3, 3) / 3
    
    # Stack features: [ret1, ret5, ret20, vol10, vol30, rsi, vol_ratio, hl_ratio, ema_cross, obv]
    features = np.column_stack([
        ret1, ret5, ret20,
        vol10, vol30,
        rsi_norm,
        vol_ratio_norm,
        hl_ratio,
        ema_cross,
        obv_diff,
    ])
    
    return features


def _create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    """Create sequences for time series models."""
    X, y = [], []
    for i in range(seq_len, len(features)):
        X.append(features[i-seq_len:i])
        y.append(targets[i])
    return np.array(X), np.array(y)


def _make_targets(close: np.ndarray, horizon: int = 6) -> np.ndarray:
    """Binary target: 1 if price goes up over horizon bars, 0 otherwise."""
    targets = np.zeros(len(close))
    for i in range(len(close) - horizon):
        targets[i] = 1.0 if close[i + horizon] > close[i] else 0.0
    return targets


# ── LSTM Model ───────────────────────────────────────────────────────────────

def _build_lstm(input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
    torch, nn, device = _init_torch()
    
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout,
                batch_first=True, bidirectional=True,
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)
    
    return LSTMModel().to(device)


# ── CNN Model ────────────────────────────────────────────────────────────────

def _build_cnn(input_dim, seq_len, n_filters=64, dropout=0.3):
    torch, nn, device = _init_torch()
    
    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(n_filters, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            # x: (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.permute(0, 2, 1)
            x = self.conv(x).squeeze(-1)
            return self.fc(x).squeeze(-1)
    
    return CNNModel().to(device)


# ── Transformer Model ────────────────────────────────────────────────────────

def _build_transformer(input_dim, seq_len, d_model=64, nhead=4, num_layers=2, dropout=0.2):
    torch, nn, device = _init_torch()
    
    class TransformerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        
        def forward(self, x):
            x = self.input_proj(x) + self.pos_enc[:, :x.size(1), :]
            x = self.transformer(x)
            x = x[:, -1, :]  # Last timestep
            return self.fc(x).squeeze(-1)
    
    return TransformerModel().to(device)


# ── Training Utility ─────────────────────────────────────────────────────────

def _train_model(model, X_train, y_train, epochs=50, batch_size=64, lr=0.001):
    torch, nn, device = _init_torch()
    
    X_t = torch.FloatTensor(X_train).to(device)
    y_t = torch.FloatTensor(y_train).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss / len(loader))
    
    return model


def _predict(model, X):
    torch, nn, device = _init_torch()
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(device)
        pred = model(X_t).cpu().numpy()
    return pred


# ── ATR Helper ───────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Strategy Implementations ─────────────────────────────────────────────────

class LSTMTrendStrategy(BaseStrategy):
    """
    Bidirectional LSTM for trend direction prediction.
    
    Trains on rolling window, predicts probability of price going up
    over next 6 bars. Enters on high-confidence predictions (>0.65 / <0.35).
    Retrains every train_interval bars.
    """
    name = "LSTMTrend"
    description = "Bidirectional LSTM trend predictor"
    version = "7.0"
    
    default_params = {
        "seq_len": 30,
        "train_size": 500,
        "train_interval": 100,
        "horizon": 6,
        "buy_threshold": 0.65,
        "sell_threshold": 0.35,
        "epochs": 30,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        features = _build_features(df)
        targets = _make_targets(df["close"].values, p["horizon"])
        atr = _atr(df, p["atr_period"])
        
        input_dim = features.shape[1]
        model = None
        last_train = 0
        last_bar = -p["cooldown"] - 1
        
        start = p["train_size"] + p["seq_len"] + 50

        for i in range(start, len(df) - p["horizon"]):
            # Retrain periodically
            if model is None or (i - last_train) >= p["train_interval"]:
                train_start = max(0, i - p["train_size"] - p["seq_len"])
                feat_slice = features[train_start:i]
                tgt_slice = targets[train_start:i]
                
                X_train, y_train = _create_sequences(feat_slice, tgt_slice, p["seq_len"])
                if len(X_train) < 50:
                    continue
                
                model = _build_lstm(input_dim)
                model = _train_model(model, X_train, y_train, epochs=p["epochs"])
                last_train = i
            
            if i - last_bar < p["cooldown"]:
                continue
            
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            # Predict
            seq = features[i - p["seq_len"]:i]
            pred = _predict(model, seq[np.newaxis, :])[0]
            c = df["close"].iloc[i]
            
            if pred > p["buy_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = float(pred)
                last_bar = i
            
            elif pred < p["sell_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = float(1 - pred)
                last_bar = i

        return signals


class CNNMomentumStrategy(BaseStrategy):
    """
    1D-CNN on multi-feature candle windows.
    
    Uses convolutional layers to detect patterns in stacked OHLCV features.
    Advantage: captures local temporal patterns without sequential training.
    """
    name = "CNNMom"
    description = "1D-CNN multi-feature pattern detector"
    version = "7.0"
    
    default_params = {
        "seq_len": 20,
        "train_size": 500,
        "train_interval": 100,
        "horizon": 6,
        "buy_threshold": 0.62,
        "sell_threshold": 0.38,
        "epochs": 40,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 3.5,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        features = _build_features(df)
        targets = _make_targets(df["close"].values, p["horizon"])
        atr = _atr(df, p["atr_period"])
        
        input_dim = features.shape[1]
        model = None
        last_train = 0
        last_bar = -p["cooldown"] - 1
        
        start = p["train_size"] + p["seq_len"] + 50

        for i in range(start, len(df) - p["horizon"]):
            if model is None or (i - last_train) >= p["train_interval"]:
                train_start = max(0, i - p["train_size"] - p["seq_len"])
                feat_slice = features[train_start:i]
                tgt_slice = targets[train_start:i]
                
                X_train, y_train = _create_sequences(feat_slice, tgt_slice, p["seq_len"])
                if len(X_train) < 50:
                    continue
                
                model = _build_cnn(input_dim, p["seq_len"])
                model = _train_model(model, X_train, y_train, epochs=p["epochs"])
                last_train = i
            
            if i - last_bar < p["cooldown"]:
                continue
            
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            seq = features[i - p["seq_len"]:i]
            pred = _predict(model, seq[np.newaxis, :])[0]
            c = df["close"].iloc[i]
            
            if pred > p["buy_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = float(pred)
                last_bar = i
            
            elif pred < p["sell_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = float(1 - pred)
                last_bar = i

        return signals


class TransformerPredStrategy(BaseStrategy):
    """
    Lightweight transformer for price direction prediction.
    
    Self-attention captures long-range dependencies in feature sequences.
    Positional encoding helps model learn temporal patterns.
    """
    name = "TfmrPred"
    description = "Lightweight transformer price predictor"
    version = "7.0"
    
    default_params = {
        "seq_len": 30,
        "train_size": 500,
        "train_interval": 100,
        "horizon": 6,
        "buy_threshold": 0.63,
        "sell_threshold": 0.37,
        "epochs": 35,
        "atr_period": 14,
        "sl_atr_mult": 2.0,
        "tp_atr_mult": 4.0,
        "cooldown": 3,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        signals = df.copy()
        signals["signal"] = Signal.HOLD
        signals["stop_loss"] = np.nan
        signals["take_profit"] = np.nan
        signals["confidence"] = 0.5

        features = _build_features(df)
        targets = _make_targets(df["close"].values, p["horizon"])
        atr = _atr(df, p["atr_period"])
        
        input_dim = features.shape[1]
        model = None
        last_train = 0
        last_bar = -p["cooldown"] - 1
        
        start = p["train_size"] + p["seq_len"] + 50

        for i in range(start, len(df) - p["horizon"]):
            if model is None or (i - last_train) >= p["train_interval"]:
                train_start = max(0, i - p["train_size"] - p["seq_len"])
                feat_slice = features[train_start:i]
                tgt_slice = targets[train_start:i]
                
                X_train, y_train = _create_sequences(feat_slice, tgt_slice, p["seq_len"])
                if len(X_train) < 50:
                    continue
                
                model = _build_transformer(input_dim, p["seq_len"])
                model = _train_model(model, X_train, y_train, epochs=p["epochs"])
                last_train = i
            
            if i - last_bar < p["cooldown"]:
                continue
            
            a = atr.iloc[i]
            if np.isnan(a) or a <= 0:
                continue
            
            seq = features[i - p["seq_len"]:i]
            pred = _predict(model, seq[np.newaxis, :])[0]
            c = df["close"].iloc[i]
            
            if pred > p["buy_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.BUY
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c - p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c + p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = float(pred)
                last_bar = i
            
            elif pred < p["sell_threshold"]:
                signals.iloc[i, signals.columns.get_loc("signal")] = Signal.SELL
                signals.iloc[i, signals.columns.get_loc("stop_loss")] = c + p["sl_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("take_profit")] = c - p["tp_atr_mult"] * a
                signals.iloc[i, signals.columns.get_loc("confidence")] = float(1 - pred)
                last_bar = i

        return signals


def get_v7_ml_strategies():
    return {
        "LSTMTrend": LSTMTrendStrategy,
        "CNNMom": CNNMomentumStrategy,
        "TfmrPred": TransformerPredStrategy,
    }
