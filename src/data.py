# =============================================================================
# DO NOT MODIFY — provided infrastructure
# =============================================================================
"""
Data loading and feature engineering.

All features are lookahead-safe: feature at row t uses only data at indices <= t.
Fit the scaler on training data only and pass it to validation/test.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RISKY_ASSETS = ["asset_0", "asset_1", "asset_2"]
ALL_ASSETS   = RISKY_ASSETS + ["cash"]
RAW          = Path(__file__).parents[1] / "data" / "raw"


def load_prices(interval: str = "1h") -> pd.DataFrame:
    """Load raw OHLCV data for the given candle interval (15m, 30m, 1h)."""
    path = RAW / f"prices_{interval}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Data not found at {path}. Run: uv run python scripts/download_data.py --interval {interval}"
        )
    return pd.read_parquet(path)


def close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Extract just the close prices as a (T, 4) DataFrame."""
    cols = [f"{a}_close" for a in RISKY_ASSETS] + ["cash"]
    return data[cols].rename(columns={f"{a}_close": a for a in RISKY_ASSETS})


def split(data: pd.DataFrame, train_end: str, eval_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split with no overlap.

    LOOKAHEAD RULE: never fit scalers or compute statistics across both splits.
    The train split ends strictly at train_end; eval starts the next row.
    """
    train = data.loc[:train_end]
    eval_ = data.loc[train_end:eval_end].iloc[1:]
    return train, eval_


def build_features(
    data: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = False,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Compute feature matrix from raw OHLCV data.

    Features per asset:
      - log_return       : log(close_t / close_{t-1})
      - volatility_21    : rolling std of log returns over 21 periods
      - momentum_20      : log(close_t / close_{t-20})
      - atr_14           : Average True Range normalized by close (14 periods)
      - volume_ratio     : volume / rolling mean volume (21 periods)
      - taker_buy_ratio  : fraction of volume that was taker buys

    Correct usage:
        train_X, scaler = build_features(train_data, fit=True)
        val_X,   _      = build_features(val_data, scaler=scaler)
    """
    frames = []
    for asset in RISKY_ASSETS:
        close  = data[f"{asset}_close"]
        high   = data[f"{asset}_high"]
        low    = data[f"{asset}_low"]
        volume = data[f"{asset}_volume"]
        tbr    = data[f"{asset}_taker_buy_ratio"]

        log_ret = np.log(close / close.shift(1))

        # ATR: average of (high-low, |high-prev_close|, |low-prev_close|), normalized
        prev_close = close.shift(1)
        tr  = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean() / (close + 1e-8)

        vol_ratio = volume / (volume.rolling(21).mean() + 1e-8)

        f = pd.DataFrame({
            f"{asset}_log_ret":    log_ret,
            f"{asset}_vol_21":     log_ret.rolling(21).std(),
            f"{asset}_mom_20":     np.log(close / close.shift(20)),
            f"{asset}_atr_14":     atr,
            f"{asset}_vol_ratio":  vol_ratio,
            f"{asset}_tbr":        tbr,
        })
        frames.append(f)

    X = pd.concat(frames, axis=1).dropna()

    if scaler is None:
        scaler = StandardScaler()
    if fit:
        scaler.fit(X.values)

    return pd.DataFrame(scaler.transform(X.values), index=X.index, columns=X.columns), scaler
