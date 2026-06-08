# =============================================================================
# DO NOT MODIFY — provided infrastructure
# =============================================================================
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from abc import abstractmethod

ASSETS       = ["asset_0", "asset_1", "asset_2", "cash"]
CLOSE_COLS   = ["asset_0_close", "asset_1_close", "asset_2_close", "cash"]


MAX_POSITION = 1.0   # maximum absolute weight per risky asset — no leverage


class BaseTradingEnv(gym.Env):
    """
    Portfolio allocation environment — base class.

    The framework handles: price stepping, portfolio tracking, transaction costs.
    You subclass this in agent.py and implement the three abstract methods.

    Shorting is supported: weights can be negative (short positions).
    Constraints enforced on every step:
      - weights must sum to 1
      - each risky asset weight must be in [-MAX_POSITION, MAX_POSITION]
    Cash cannot be shorted (weight must be >= 0).

    Note: this model does not charge a funding rate on short positions,
    which is a simplification. Real short positions in futures markets
    carry an ongoing borrowing cost.
    """

    def __init__(self, prices: pd.DataFrame, transaction_cost_bps: float = 10.0, initial_cash: float = 10_000.0):
        super().__init__()
        self.prices = prices[CLOSE_COLS].values.astype(np.float32)
        self.data   = prices
        self.tc = transaction_cost_bps / 10_000
        self.initial_cash = initial_cash

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self._lookback
        self._value = float(self.initial_cash)
        self._weights = np.array([0., 0., 0., 1.], dtype=np.float32)
        return self._obs(), {}

    def step(self, action: int):
        w = self._weights_from_action(action).astype(np.float32)

        assert np.isclose(w.sum(), 1.0, atol=1e-4),           "weights must sum to 1"
        assert (w[:3] >= -MAX_POSITION).all(),               f"risky asset weights must be >= {-MAX_POSITION}"
        assert (w[:3] <= MAX_POSITION).all(),                f"risky asset weights must be <= {MAX_POSITION}"
        assert w[3] >= 0,                                    "cash weight must be >= 0"

        turnover = float(np.abs(w - self._weights).sum())
        ret      = self.prices[self._t] / self.prices[self._t - 1]
        prev     = self._value
        self._value   = self._value * float(np.dot(w, ret)) - self._value * turnover * self.tc
        self._weights = w
        self._t += 1

        terminated = self._t >= len(self.prices)
        return self._obs(), self._reward(prev, self._value), terminated, False, {
            "portfolio_value": self._value,
            "turnover": turnover,
        }

    @abstractmethod
    def _obs(self) -> np.ndarray:
        """Return the observation vector. Must match self.observation_space.shape."""

    @abstractmethod
    def _weights_from_action(self, action: int) -> np.ndarray:
        """
        Map action index to portfolio weights, shape (4,): [asset_0, asset_1, asset_2, cash].
        Weights must sum to 1. Risky asset weights in [-MAX_POSITION, MAX_POSITION]. Cash >= 0.
        """

    @abstractmethod
    def _reward(self, prev_value: float, curr_value: float) -> float:
        """Scalar reward signal."""
