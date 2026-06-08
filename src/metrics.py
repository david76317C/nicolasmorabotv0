# =============================================================================
# DO NOT MODIFY — provided infrastructure
# =============================================================================
import numpy as np


def compute_metrics(portfolio_values: np.ndarray, freq: int = 252) -> dict:
    v = np.asarray(portfolio_values, dtype=float)
    r = np.diff(v) / v[:-1]

    cum_ret  = v[-1] / v[0] - 1
    ann_ret  = (1 + cum_ret) ** (freq / max(len(r), 1)) - 1
    ann_vol  = r.std() * np.sqrt(freq)
    sharpe   = ann_ret / (ann_vol + 1e-8)
    down     = r[r < 0]
    sortino  = ann_ret / (down.std() * np.sqrt(freq) + 1e-8)
    peak     = np.maximum.accumulate(v)
    max_dd   = float(((v - peak) / peak).min())

    return dict(cum_ret=cum_ret, ann_ret=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, sortino=sortino, max_dd=max_dd)
