# =============================================================================
# DO NOT MODIFY — provided infrastructure
# =============================================================================
import numpy as np


class RandomPolicy:
    def __init__(self, n_actions): self.n = n_actions
    def act(self, obs): return np.random.randint(self.n)

class HoldCash:
    def act(self, obs): return 0   # assumes action 0 = all cash

class HoldAsset0:
    def act(self, obs): return 1   # assumes action 1 = 100% asset_0

class EqualWeight:
    def act(self, obs): return 4   # assumes action 4 = equal weight

class SMA:
    """Go risky when short-window momentum > long-window; else hold cash."""
    def __init__(self, short=5, long=20, risky_action=4, safe_action=0):
        self.short, self.long = short, long
        self.risky, self.safe = risky_action, safe_action

    def act(self, obs):
        n, lookback = 3, len(obs) // 3
        if lookback < self.long:
            return self.safe
        rets = obs[:lookback * n].reshape(lookback, n)
        if np.any(rets[-self.short:].sum(0) > rets[-self.long:].sum(0)):
            return self.risky
        return self.safe
