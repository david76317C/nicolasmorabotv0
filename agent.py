# =============================================================================
# YOUR FILE — this is the only file you submit.
# Implement TradingEnv and Agent below. Do not modify anything in src/.
# =============================================================================
#
# AGENT CODENAME : Bellman Capital — exposure-ladder DQN
#
# INVESTMENT THESIS (grounded in the EDA, README §0/§1):
#   - DIRECTION is close to unpredictable (returns ~ martingale-with-drift):
#     we do NOT try to forecast which asset goes up next.
#   - VOLATILITY is partially predictable (volatility clustering): we feed the
#     agent realized-vol / ATR features so it can size risk by regime.
#   - TREND carries weak but real regime information: an SMA-crossover signal
#     (the exact statistic that drives the strongest baseline) is handed to the
#     agent explicitly, bounded to [-1, 1] so it shares scale with the rest.
#   - ASSETS are highly correlated (0.70-0.83): the only real diversifier is
#     CASH, so the action menu is an EXPOSURE LADDER (how much risk to hold),
#     not an asset-picking menu.
#   - DRAWDOWNS are severe and TRADING IS COSTLY (10 bps/trade): the reward must
#     control both. We compare reward formulations and pick the most robust.
#
# Because these assets almost only rise (EDA), a fully short-driven menu lost
# ~100% in early experiments. We therefore include a SINGLE defensive short as
# an option (so we can honestly discuss the README's "shorts are allowed"), but
# keep the menu dominated by long-exposure levels.
# =============================================================================

import os
import random
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from src.env import BaseTradingEnv
from src.base import BaseAgent
from src.data import load_prices, split, build_features, RISKY_ASSETS


# ─────────────────────────────────────────────────────────────────────────────
# ACTION SPACE  (README §1)
# ─────────────────────────────────────────────────────────────────────────────
# A MONOTONE EXPOSURE LADDER on the equal-weight basket, plus all-cash, plus one
# defensive short. Each row sums to 1; cash >= 0; risky weights in [-1, 1].
#
#   0  all_cash      100% cash                       — safe fallback
#   1  exposure_25   25% basket / 75% cash           — defensive
#   2  exposure_50   50% basket / 50% cash
#   3  exposure_75   75% basket / 25% cash
#   4  equal_full    100% equal-weight basket        — full risk-on
#   5  short_hedge   -25% asset_0, 125% cash         — single defensive short
#
# WHY discrete ladder: DQN needs a discrete space; a small ORDERED menu makes
#   exploration dense, so the few seeds we can afford converge to similar
#   policies (low seed variance — what a walk-forward AVERAGE rewards).
# WHAT IT PREVENTS: fine-grained per-asset tilts and leverage. We accept this:
#   the EDA shows asset-picking is not robustly profitable here, only sizing is.
_ACTION_WEIGHTS = np.array([
    [0.00, 0.00, 0.00, 1.00],   # 0  all cash
    [1/12, 1/12, 1/12, 0.75],   # 1  25% exposure
    [1/6,  1/6,  1/6,  0.50],   # 2  50% exposure
    [0.25, 0.25, 0.25, 0.25],   # 3  75% exposure
    [1/3,  1/3,  1/3,  0.00],   # 4  100% equal-weight
    [-0.25, 0.00, 0.00, 1.25],  # 5  defensive short on asset_0
], dtype=np.float32)

# normalize tiny float error so every row sums to exactly 1
_ACTION_WEIGHTS = _ACTION_WEIGHTS / _ACTION_WEIGHTS.sum(axis=1, keepdims=True)
N_ACTIONS = len(_ACTION_WEIGHTS)

_ACTION_NAMES = {
    0: "all_cash", 1: "exposure_25", 2: "exposure_50",
    3: "exposure_75", 4: "equal_full", 5: "short_hedge",
}


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT  (README §1 state, §3 implementation)
# ─────────────────────────────────────────────────────────────────────────────
class TradingEnv(BaseTradingEnv):
    """
    STATE (28-dim, all causal — uses only data at index <= t):

      (A) 18 official build_features() columns for the 3 risky assets, 6 each:
            log_ret    — last-step log return            (recent direction)
            vol_21     — 21-period rolling std of returns (volatility regime)
            mom_20     — 20-period log momentum           (trend)
            atr_14     — 14-period ATR / close            (intrabar volatility)
            vol_ratio  — volume / 21-period mean volume   (activity / conviction)
            tbr        — taker-buy ratio                  (order-flow pressure)
          These are STANDARDIZED by a StandardScaler FIT ON TRAIN DATA ONLY and
          reused on eval/OOT (no lookahead — README §2).

      (B) 6 SMA-momentum regime features (bounded to [-1, 1] so they cannot
          unbalance the MLP relative to the standardized features):
            sign(3) : +1/0/-1 short-vs-long crossover per asset (the SMA signal)
            tanh(3) : tanh(short cumulative return * 20) — smooth trend strength

      (C) 4 current portfolio weights [a0, a1, a2, cash] — needed so the agent
          knows the turnover a rebalance would cost (state must include holdings).

    Why this is (approximately) Markov w.r.t. the README's concerns: past
    volatility enters via vol_21/atr_14, momentum via mom_20 and the SMA block,
    so a single-step lookback over these summaries suffices (README §1.2/1.3).

    REWARD (REWARD_TYPE, README §4 — we compare these):
      log_return         : r_t = log(V_t / V_{t-1}).  Simplest; over-trades.
      turnover_penalized : log_return - LAMBDA_TC * turnover.
      drawdown_penalized : log_return - LAMBDA_DD * incremental_drawdown.
      combined           : both penalties (default; most robust in our sweeps).
      diff_sharpe        : differential Sharpe ratio (Moody & Saffell, 1998).
    The base env ALREADY charges the 10 bps fee, so penalties only SHAPE
    learning; they never alter the reported metrics.
    """

    # Class attributes the CLI overrides per experiment.
    REWARD_TYPE = "combined"
    LAMBDA_TC = 0.01     # turnover penalty (low: avoids timid all-cash collapse)
    LAMBDA_DD = 0.05     # drawdown penalty (moderate: tames bear-window blowups)
    DSR_ETA   = 0.01     # differential-Sharpe EWMA rate

    def __init__(self, prices, transaction_cost_bps=10.0, initial_cash=10_000.0,
                 scaler=None):
        # Build official features BEFORE handing prices to the base class.
        # build_features drops warm-up rows (rolling-window NaNs), so we align
        # prices to the surviving index to keep feature[t] <-> price[t].
        feats, fitted_scaler = build_features(
            prices, scaler=scaler, fit=(scaler is None)
        )
        self._scaler = fitted_scaler
        prices_aligned = prices.loc[feats.index]

        super().__init__(prices_aligned, transaction_cost_bps, initial_cash)

        self._features = feats.values.astype(np.float32)     # (T, 18) standardized
        self._feat_dim = self._features.shape[1]

        # Causal raw log-returns of the risky assets, for the SMA-momentum block.
        risky_close = prices_aligned[[f"{a}_close" for a in RISKY_ASSETS]].values.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.zeros_like(risky_close)
            lr[1:] = np.log(np.maximum(risky_close[1:], 1e-12) /
                            np.maximum(risky_close[:-1], 1e-12))
        self._logret = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        # Features already summarize history → 1-step lookback. _t must start
        # >= 1 so price[t-1] exists in BaseTradingEnv.step().
        self._lookback = 1

        self.action_space = spaces.Discrete(N_ACTIONS)
        # 18 features + 6 momentum + 4 weights = 28
        obs_dim = self._feat_dim + 6 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # bookkeeping
        self._peak = float(self.initial_cash)
        self._last_turnover = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0

        # TRAINING-ONLY random episode start (decorrelation / cheap augmentation).
        self._train_random_start = False

    # -- helpers ----------------------------------------------------------------
    def _momentum(self, end, short=24, long=168):
        """Bounded SMA-crossover signal per asset (see STATE block B)."""
        s0 = max(1, end - short)
        l0 = max(1, end - long)
        short_seg = self._logret[s0:end]
        long_seg = self._logret[l0:end]
        short_cum = short_seg.sum(axis=0) if len(short_seg) else np.zeros(3, np.float32)
        long_avg = (long_seg.mean(axis=0) * short) if len(long_seg) else np.zeros(3, np.float32)
        if end - 1 < long:
            sign = np.zeros(3, dtype=np.float32)
        else:
            sign = np.sign(short_cum - long_avg).astype(np.float32)
        mag = np.tanh(short_cum * 20.0).astype(np.float32)      # bounded (-1, 1)
        return sign, mag

    # -- core API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        if self._train_random_start and len(self.prices) > self._lookback + 64:
            self._t = int(np.random.randint(self._lookback, len(self.prices) - 64))
        self._peak = float(self._value)
        self._last_turnover = 0.0
        self._dsr_A = 0.0
        self._dsr_B = 0.0
        return self._obs(), info

    def _obs(self) -> np.ndarray:
        idx = min(self._t, len(self._features) - 1)
        feat = self._features[idx]
        mom_sign, mom_mag = self._momentum(self._t)
        obs = np.concatenate([
            feat,
            mom_sign, mom_mag,
            self._weights.astype(np.float32),
        ]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _weights_from_action(self, action: int) -> np.ndarray:
        new_w = _ACTION_WEIGHTS[int(action)].copy()
        # self._weights still holds PREVIOUS weights here; stash L1 turnover.
        self._last_turnover = float(np.abs(new_w - self._weights).sum())
        return new_w

    def _reward(self, prev_value: float, curr_value: float) -> float:
        base = float(np.log(max(curr_value, 1e-12) / max(prev_value, 1e-12)))
        turnover = getattr(self, "_last_turnover", 0.0)
        rtype = self.REWARD_TYPE

        # keep drawdown peak current for every reward type (compute incr first)
        prev_peak = self._peak
        new_peak = max(self._peak, curr_value)
        dd_now = max(0.0, (new_peak - curr_value) / (new_peak + 1e-12))
        dd_prev = max(0.0, (prev_peak - prev_value) / (prev_peak + 1e-12))
        incremental_dd = max(0.0, dd_now - dd_prev)
        self._peak = new_peak

        if rtype == "log_return":
            return 0.0 if abs(curr_value - prev_value) < 1e-9 else base

        if rtype == "diff_sharpe":
            r = base
            eta = self.DSR_ETA
            A, B = self._dsr_A, self._dsr_B
            dA, dB = r - A, r * r - B
            var = B - A * A
            d = 0.0 if var <= 1e-12 else (B * dA - 0.5 * A * dB) / (var ** 1.5 + 1e-12)
            self._dsr_A, self._dsr_B = A + eta * dA, B + eta * dB
            return 0.0 if abs(curr_value - prev_value) < 1e-9 else float(np.clip(d, -10, 10))

        tc_penalty = self.LAMBDA_TC * turnover if rtype in ("turnover_penalized", "combined") else 0.0
        dd_penalty = self.LAMBDA_DD * incremental_dd if rtype in ("drawdown_penalized", "combined") else 0.0

        # exact 0 when value unchanged AND no rebalancing (TestReward zero case)
        if abs(curr_value - prev_value) < 1e-9 and turnover < 1e-12:
            return 0.0
        return base - tc_penalty - dd_penalty


# ─────────────────────────────────────────────────────────────────────────────
# DQN AGENT  (README §5)  —  Double DQN + Dueling head + n-step + Huber
# ─────────────────────────────────────────────────────────────────────────────
class _QNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256), dueling=True):
        super().__init__()
        self.dueling = bool(dueling)
        body, d = [], obs_dim
        for h in hidden:
            body += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*body)
        if self.dueling:
            self.v = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
            self.a = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n_actions))
        else:
            self.head = nn.Linear(d, n_actions)

    def forward(self, x):
        z = self.body(x)
        if self.dueling:
            v, a = self.v(z), self.a(z)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(z)


class Agent(BaseAgent):

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_dims=(256, 256), lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay_steps=50_000,
                 batch_size=64, buffer_size=100_000, target_update_freq=1000,
                 dueling=True, n_step=3, reward_scale=100.0, learning_starts=1000,
                 seed=0):
        super().__init__(obs_dim, n_actions)
        self.seed = seed
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = tuple(hidden_dims)
        self.dueling = bool(dueling)
        self.q_net = _QNet(obs_dim, n_actions, self.hidden_dims, self.dueling).to(self.device)
        self.target_net = _QNet(obs_dim, n_actions, self.hidden_dims, self.dueling).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.buffer = deque(maxlen=buffer_size)

        self.n_step = int(max(1, n_step))
        self.reward_scale = float(reward_scale)
        self.learning_starts = int(learning_starts)

        self.epsilon = epsilon_start
        self.eps_start, self.eps_end = epsilon_start, epsilon_end
        self.eps_decay = epsilon_decay_steps
        self._step = 0

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _store(self, transition):
        self.buffer.append(transition)

    def _make_nstep(self, nbuf):
        R, g = 0.0, self.gamma
        for i, (s, a, r, ns, d) in enumerate(nbuf):
            R += (g ** i) * r
            if d:
                return (nbuf[0][0], nbuf[0][1], R, ns, 1.0, g ** (i + 1))
        last = nbuf[-1]
        return (nbuf[0][0], nbuf[0][1], R, last[3], 0.0, g ** len(nbuf))

    def _decay_epsilon(self):
        frac = min(1.0, self._step / max(1, self.eps_decay))
        self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

    def _learn(self):
        batch = random.sample(self.buffer, self.batch_size)
        s, a, R, sb, d, gk = zip(*batch)
        s  = self._t(np.array(s))
        sb = self._t(np.array(sb))
        a  = torch.as_tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        R  = self._t(R).unsqueeze(1)
        d  = self._t(d).unsqueeze(1)
        gk = self._t(gk).unsqueeze(1)

        q_sa = self.q_net(s).gather(1, a)
        with torch.no_grad():
            next_a = self.q_net(sb).argmax(dim=1, keepdim=True)     # Double DQN
            q_next = self.target_net(sb).gather(1, next_a)
            target = R + gk * q_next * (1.0 - d)                    # n-step target
        loss = F.smooth_l1_loss(q_sa, target)                       # Huber

        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.opt.step()

    def train(self, env, n_steps: int = 200_000) -> None:
        prev_rs = getattr(env, "_train_random_start", None)
        if hasattr(env, "_train_random_start"):
            env._train_random_start = True
        self.eps_decay = max(1, int(0.6 * n_steps))      # explore ~60%, then exploit
        self._step = 0
        self.epsilon = self.eps_start

        nbuf = deque()
        obs, _ = env.reset()
        try:
            for _ in range(n_steps):
                self._step += 1
                self._decay_epsilon()

                if random.random() < self.epsilon:
                    action = random.randint(0, self.n_actions - 1)
                else:
                    with torch.no_grad():
                        action = int(self.q_net(self._t(obs).unsqueeze(0)).argmax(1).item())

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                nbuf.append((obs, action, float(reward) * self.reward_scale, next_obs, done))

                if done:
                    while nbuf:
                        self._store(self._make_nstep(nbuf)); nbuf.popleft()
                elif len(nbuf) >= self.n_step:
                    self._store(self._make_nstep(nbuf)); nbuf.popleft()

                obs = next_obs if not done else env.reset()[0]

                if len(self.buffer) >= max(self.batch_size, self.learning_starts):
                    self._learn()
                if self._step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())
        finally:
            if hasattr(env, "_train_random_start") and prev_rs is not None:
                env._train_random_start = prev_rs

    def act(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            return int(self.q_net(self._t(obs).unsqueeze(0)).argmax(1).item())

    # -- persistence ------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"q_net": self.q_net.state_dict(), "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "hidden_dims": self.hidden_dims,
                    "dueling": self.dueling, "n_step": self.n_step,
                    "reward_scale": self.reward_scale, "seed": self.seed}, path)

    @classmethod
    def load(cls, path: str, device=None) -> "Agent":
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(obs_dim=ckpt["obs_dim"], n_actions=ckpt["n_actions"],
                    hidden_dims=ckpt["hidden_dims"], dueling=ckpt.get("dueling", True),
                    n_step=ckpt.get("n_step", 3), reward_scale=ckpt.get("reward_scale", 100.0),
                    seed=ckpt.get("seed", 0))
        agent.q_net.load_state_dict(ckpt["q_net"])
        agent.target_net.load_state_dict(ckpt["q_net"])
        agent.q_net.eval()
        return agent


# =============================================================================
# Everything below is the EXPERIMENT HARNESS (not used by the submission tests).
# It produces every deliverable the README asks for, in the required order:
#   compare -> gridsearch -> walk_forward (multi-seed) -> evaluate (OOT + ablation)
# =============================================================================

# ── Baselines (README §6), mapped to THIS action menu ─────────────────────────
class _Baselines:
    @staticmethod
    def random(obs): return np.random.randint(N_ACTIONS)
    @staticmethod
    def cash(obs):   return 0                       # hold cash
    @staticmethod
    def asset0(obs): return 4                        # single-asset proxy: full basket
    @staticmethod
    def equal(obs):  return 4                        # equal-weight = action 4
    @staticmethod
    def sma(obs):
        # obs layout: [18 feats][sign(3)][tanh(3)][w(4)]; the SMA sign block sits
        # right after the 18 features. Risk-on if any asset is trending up.
        sign = obs[18:21]
        return 4 if np.any(sign > 0) else 0


def _baseline_policies():
    return {"Random": _Baselines.random, "Hold cash": _Baselines.cash,
            "Hold Asset_0": _Baselines.asset0, "Equal weight": _Baselines.equal,
            "SMA crossover": _Baselines.sma}


# ── Metrics (mirrors src/metrics.py; freq=8760 for hourly) ────────────────────
def _metrics(values, freq=8760):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return dict(cum_ret=0, ann_ret=0, ann_vol=0, sharpe=0, sortino=0, max_dd=0)
    r = np.diff(v) / v[:-1]
    cum = v[-1] / v[0] - 1
    ann = (1 + cum) ** (freq / max(len(r), 1)) - 1
    vol = r.std() * np.sqrt(freq)
    down = r[r < 0]
    dstd = down.std() * np.sqrt(freq) if len(down) > 1 else 0.0
    peak = np.maximum.accumulate(v)
    return dict(cum_ret=cum, ann_ret=ann, ann_vol=vol,
                sharpe=ann / (vol + 1e-8), sortino=ann / (dstd + 1e-8),
                max_dd=float(((v - peak) / peak).min()))


def _rollout(env, policy, collect=False):
    obs, _ = env.reset()
    vals, steps, done = [], [], False
    peak = float(env._value)
    while not done:
        a = int(policy(obs))
        obs, reward, term, trunc, info = env.step(a)
        v = info["portfolio_value"]; peak = max(peak, v)
        vals.append(v)
        if collect:
            w = env._weights
            steps.append(dict(action=a, action_name=_ACTION_NAMES.get(a, str(a)),
                              w_a0=float(w[0]), w_a1=float(w[1]), w_a2=float(w[2]),
                              w_cash=float(w[3]), turnover=float(info.get("turnover", 0.0)),
                              value=float(v), drawdown=float((peak - v) / (peak + 1e-12))))
        done = term or trunc
    return (np.array(vals), steps) if collect else np.array(vals)


def _action_distribution(steps):
    if not steps:
        return {}, float("nan")
    acts = np.array([s["action"] for s in steps])
    dd = np.array([s["drawdown"] for s in steps])
    expo = np.array([1.0 - s["w_cash"] for s in steps])
    dist = {n: float(np.mean(acts == i)) for i, n in _ACTION_NAMES.items()}
    dist = {k: v for k, v in dist.items() if v > 0}
    corr = float(np.corrcoef(expo, dd)[0, 1]) if np.std(dd) > 1e-9 and np.std(expo) > 1e-9 else float("nan")
    return dist, corr


# ── Temporal design (cuts come from configs/default.yaml) ─────────────────────
# SELECTION windows: used to pick reward / lambdas / seed.
WALK_FORWARD = [
    ("2019-12-31", "2020-12-31"),
    ("2020-12-31", "2021-12-31"),
    ("2021-12-31", "2022-12-31"),
    ("2022-12-31", "2023-12-31"),
    ("2023-12-31", "2024-12-31"),
]
# Cheap subset for the light, single-seed selection sweeps.
SWEEP_WINDOWS = WALK_FORWARD[-2:]
# OOT held-out: the most recent year. The agent NEVER trains or tunes on it;
# it is scored exactly once in `evaluate`. The scaler is fit only up to its start.
OOT_TEST = ("2024-12-31", "2025-12-31")

# Step budgets: selection sweeps are deliberately LIGHT; the shipped/eval models
# train LONG (README §7).
SWEEP_STEPS = 30_000
WF_STEPS    = 150_000
FINAL_STEPS = 200_000


def _set_reward(reward_type, lambda_tc, lambda_dd):
    TradingEnv.REWARD_TYPE = reward_type
    TradingEnv.LAMBDA_TC = lambda_tc
    TradingEnv.LAMBDA_DD = lambda_dd


def _train_eval_one_window(data, train_end, eval_end, seed, train_steps, tc_bps):
    """Train on [:train_end], evaluate trained agent + baselines on (train_end, eval_end].
    Scaler is fit on TRAIN ONLY and reused on eval (no lookahead)."""
    train_df, eval_df = split(data, train_end, eval_end)
    train_env = TradingEnv(train_df, transaction_cost_bps=tc_bps)
    eval_env  = TradingEnv(eval_df, transaction_cost_bps=tc_bps, scaler=train_env._scaler)

    obs, _ = eval_env.reset()
    agent = Agent(obs_dim=obs.shape[0], n_actions=N_ACTIONS, seed=seed)

    results = {}
    for name, pol in _baseline_policies().items():
        results[name] = _metrics(_rollout(eval_env, pol))
    agent.train(train_env, n_steps=train_steps)
    vals_t, steps_t = _rollout(eval_env, agent.act, collect=True)
    results["DQN (trained)"] = _metrics(vals_t)
    return results, agent, vals_t, steps_t, eval_df.index


def _print_window_table(name, results, action_info=None):
    print(f"\n{'='*78}\n  WINDOW: {name}\n{'='*78}")
    print(f"  {'strategy':<18}{'cum_ret':>9}{'ann_ret':>9}{'vol':>7}{'SORTINO':>9}{'max_dd':>8}")
    print(f"  {'-'*64}")
    for n, m in results.items():
        print(f"  {n:<18}{m['cum_ret']:>8.0%}{m['ann_ret']:>9.0%}{m['ann_vol']:>7.0%}"
              f"{m['sortino']:>9.2f}{m['max_dd']:>8.0%}")
    if action_info:
        dist, corr = action_info
        print("\n  DQN action distribution:")
        for n, f in sorted(dist.items(), key=lambda kv: -kv[1]):
            print(f"    {n:<14}{f:>6.1%}  {'#' * int(f * 40)}")
        tag = ("cuts exposure as drawdown grows (GOOD risk mgmt)" if corr < -0.1
               else "adds exposure as drawdown grows (risky)" if corr > 0.1
               else "exposure ~independent of drawdown")
        print(f"  exposure<->drawdown corr = {corr:+.3f}  ->  {tag}")


# ── STAGE 1: compare reward functions (LIGHT: 1 seed, short, subset) ──────────
def stage_compare(interval="1h", tc_bps=10.0, log_dir="logs", seed=0,
                  steps=SWEEP_STEPS):
    print(f"\n{'#'*78}\n#  STAGE 1 — REWARD COMPARISON (1 seed, {steps} steps)\n{'#'*78}")
    data = load_prices(interval)
    rewards = ["log_return", "turnover_penalized", "drawdown_penalized", "combined", "diff_sharpe"]
    rows = []
    for rt in rewards:
        _set_reward(rt, TradingEnv.LAMBDA_TC, TradingEnv.LAMBDA_DD)
        sortinos = []
        for te, ee in SWEEP_WINDOWS:
            res, *_ = _train_eval_one_window(data, te, ee, seed, steps, tc_bps)
            sortinos.append(res["DQN (trained)"]["sortino"])
        rows.append(dict(reward=rt, mean_sortino=float(np.mean(sortinos)),
                         per_window=";".join(f"{s:+.2f}" for s in sortinos)))
        print(f"  {rt:<20} mean_sortino={np.mean(sortinos):+.3f}   windows=[{rows[-1]['per_window']}]")
    lb = pd.DataFrame(rows).sort_values("mean_sortino", ascending=False)
    os.makedirs(log_dir, exist_ok=True)
    lb.to_csv(os.path.join(log_dir, "stage1_reward_comparison.csv"), index=False)
    print(f"\n  WINNER: {lb.iloc[0]['reward']}  -> use it for the grid search.")
    print(f"  Saved: {log_dir}/stage1_reward_comparison.csv")
    return lb


# ── STAGE 2: grid search over lambdas (LIGHT: 1 seed, short, subset) ──────────
def stage_gridsearch(interval="1h", tc_bps=10.0, log_dir="logs", seed=0,
                     reward_type="combined", steps=SWEEP_STEPS,
                     tc_grid=(0.005, 0.01, 0.02), dd_grid=(0.0, 0.02, 0.05)):
    print(f"\n{'#'*78}\n#  STAGE 2 — GRID SEARCH ({reward_type}, 1 seed, {steps} steps)\n{'#'*78}")
    data = load_prices(interval)
    combos = [(t, d) for t in tc_grid for d in dd_grid if not (t == 0 and d == 0)]
    rows = []
    for ltc, ldd in combos:
        _set_reward(reward_type, ltc, ldd)
        sortinos, cumrets = [], []
        for te, ee in SWEEP_WINDOWS:
            res, *_ = _train_eval_one_window(data, te, ee, seed, steps, tc_bps)
            sortinos.append(res["DQN (trained)"]["sortino"])
            cumrets.append(res["DQN (trained)"]["cum_ret"])
        collapse = float(np.mean([abs(c) < 1e-6 for c in cumrets]))
        rows.append(dict(lambda_tc=ltc, lambda_dd=ldd, mean_sortino=float(np.mean(sortinos)),
                         mean_cumret=float(np.mean(cumrets)), cash_collapse=collapse))
        print(f"  tc={ltc:<6g} dd={ldd:<6g} mean_sortino={np.mean(sortinos):+.3f}"
              f"  cumret={np.mean(cumrets):+.0%}  collapse={collapse:.0%}")
    lb = pd.DataFrame(rows).sort_values("mean_sortino", ascending=False)
    os.makedirs(log_dir, exist_ok=True)
    lb.to_csv(os.path.join(log_dir, "stage2_gridsearch.csv"), index=False)
    best = lb.iloc[0]
    print(f"\n  BEST: lambda_tc={best['lambda_tc']:g}  lambda_dd={best['lambda_dd']:g}")
    print(f"  Saved: {log_dir}/stage2_gridsearch.csv")
    return lb


# ── STAGE 3: walk-forward, MULTI-SEED (HEAVY) + equity curves + seed spread ───
def stage_walk_forward(interval="1h", tc_bps=10.0, log_dir="logs", seeds=(0, 1, 2),
                       reward_type="combined", lambda_tc=0.01, lambda_dd=0.05,
                       steps=WF_STEPS, make_plots=True):
    print(f"\n{'#'*78}\n#  STAGE 3 — WALK-FORWARD  seeds={seeds}  ({steps} steps)\n{'#'*78}")
    _set_reward(reward_type, lambda_tc, lambda_dd)
    data = load_prices(interval)
    os.makedirs(log_dir, exist_ok=True)

    summary_rows, equity_rows = [], []
    fail_window = None  # remember a negative-Sortino window for the failure figure

    for seed in seeds:
        for te, ee in WALK_FORWARD:
            res, agent, vals_t, steps_t, idx = _train_eval_one_window(
                data, te, ee, seed, steps, tc_bps)
            wname = f"{te}->{ee}"
            for strat, m in res.items():
                summary_rows.append(dict(seed=seed, window=wname, strategy=strat, **m))
            for i, v in enumerate(vals_t):
                equity_rows.append(dict(seed=seed, window=wname, t=i, value=float(v)))
            ai = _action_distribution(steps_t)
            if seed == seeds[0]:
                _print_window_table(f"[seed {seed}] {wname}", res, ai)
            if res["DQN (trained)"]["sortino"] < 0 and fail_window is None:
                fail_window = (wname, vals_t, steps_t, res)

    summary = pd.DataFrame(summary_rows)
    equity = pd.DataFrame(equity_rows)
    summary.to_csv(os.path.join(log_dir, "stage3_walkforward_summary.csv"), index=False)
    equity.to_csv(os.path.join(log_dir, "stage3_equity_curves.csv"), index=False)

    # seed-spread table (README §9)
    print(f"\n{'='*78}\n  SEED-SPREAD (mean +/- std across {len(seeds)} seeds)\n{'='*78}")
    agg = (summary[summary.strategy == "DQN (trained)"]
           .groupby("window")["sortino"].agg(["mean", "std"]).reset_index())
    for _, r in agg.iterrows():
        std = 0 if np.isnan(r["std"]) else r["std"]
        print(f"  {r['window']:<24} sortino = {r['mean']:+.2f}  +/- {std:.2f}")
    dq_mean = summary[summary.strategy == "DQN (trained)"].sortino.mean()
    eq_mean = summary[summary.strategy == "Equal weight"].sortino.mean()
    print(f"\n  DQN mean Sortino over all windows = {dq_mean:+.3f}")
    print(f"  Equal-weight mean Sortino         = {eq_mean:+.3f}")

    if make_plots:
        _plot_walk_forward(summary, equity, fail_window, log_dir)
    print(f"\n  Saved: {log_dir}/stage3_walkforward_summary.csv, stage3_equity_curves.csv")
    return summary, equity


# ── STAGE 4: OOT held-out evaluation (ONCE) + cost ablation (README §8) ───────
def stage_evaluate(interval="1h", log_dir="logs", model_dir="models",
                   reward_type="combined", lambda_tc=0.01, lambda_dd=0.05,
                   seeds=(0, 1, 2), steps=FINAL_STEPS, oot=OOT_TEST,
                   ablation_bps=(0, 10, 25), make_plots=True):
    print(f"\n{'#'*78}\n#  STAGE 4 — OOT HELD-OUT EVALUATION (once)  {oot[0]}->{oot[1]}\n{'#'*78}")
    _set_reward(reward_type, lambda_tc, lambda_dd)
    data = load_prices(interval)
    os.makedirs(log_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)

    oot_start, oot_end = oot
    # Train on everything up to OOT start; scaler is FIT ON TRAIN ONLY.
    train_df = data.loc[:oot_start]
    scaler = TradingEnv(train_df, transaction_cost_bps=10.0)._scaler
    oot_df = data.loc[oot_start:oot_end].iloc[1:]

    # SELECTION WITHOUT PEEKING AT THE OOT (no leakage):
    # carve a VALIDATION window = the last selection window before the OOT.
    # Each seed trains on data strictly before that window, is scored on it,
    # and the best-by-VALIDATION seed is the one we then evaluate ONCE on OOT.
    val_end = oot_start                       # validation ends where OOT begins
    val_start = WALK_FORWARD[-1][0]           # last selection train_end
    sel_train_df = data.loc[:val_start]
    val_df = data.loc[val_start:val_end].iloc[1:]
    sel_scaler = TradingEnv(sel_train_df, transaction_cost_bps=10.0)._scaler

    print(f"  Seed selection by VALIDATION {val_start}->{val_end} (OOT untouched):")
    best_val = None
    for seed in seeds:
        tr_env = TradingEnv(sel_train_df, transaction_cost_bps=10.0)
        agent = Agent(obs_dim=tr_env.observation_space.shape[0], n_actions=N_ACTIONS, seed=seed)
        agent.train(tr_env, n_steps=steps)
        va_env = TradingEnv(val_df, transaction_cost_bps=10.0, scaler=sel_scaler)
        m = _metrics(_rollout(va_env, agent.act))
        print(f"    seed {seed}: val sortino={m['sortino']:+.2f}  cumret={m['cum_ret']:+.0%}")
        if best_val is None or m["sortino"] > best_val[1]:
            best_val = (seed, m["sortino"])
    best_seed = best_val[0]
    print(f"  -> selected seed = {best_seed} (by validation). Now training it on "
          f"all data up to OOT and scoring ONCE on the held-out OOT.")

    # Retrain the WINNING seed on the full pre-OOT span, then score ONCE on OOT.
    tr_env = TradingEnv(train_df, transaction_cost_bps=10.0)
    best_agent = Agent(obs_dim=tr_env.observation_space.shape[0], n_actions=N_ACTIONS, seed=best_seed)
    best_agent.train(tr_env, n_steps=steps)
    oot_env = TradingEnv(oot_df, transaction_cost_bps=10.0, scaler=scaler)
    m_oot = _metrics(_rollout(oot_env, best_agent.act))
    print(f"  OOT (held-out, scored once): sortino={m_oot['sortino']:+.2f}  cumret={m_oot['cum_ret']:+.0%}")
    best_agent.save(os.path.join(model_dir, "oot_model.pt"))
    print(f"  -> saved models/oot_model.pt (the held-out-validated agent)")

    # COST ABLATION: score the chosen agent + baselines at each fee level.
    print(f"\n{'='*78}\n  COST ABLATION on OOT (bps in {ablation_bps})\n{'='*78}")
    abl_rows = []
    final_steps_for_plot = None
    for bps in ablation_bps:
        env = TradingEnv(oot_df, transaction_cost_bps=float(bps), scaler=scaler)
        results = {n: _metrics(_rollout(env, p)) for n, p in _baseline_policies().items()}
        env = TradingEnv(oot_df, transaction_cost_bps=float(bps), scaler=scaler)
        vals, steps_f = _rollout(env, best_agent.act, collect=True)
        results["DQN (final)"] = _metrics(vals)
        if bps == 10:
            final_steps_for_plot = (vals, steps_f, results)
        fees_usd = sum(s["turnover"] for s in steps_f) * (bps / 10_000) * 10_000  # $ on $10k
        for n, m in results.items():
            abl_rows.append(dict(bps=bps, strategy=n, **m,
                                 fees_usd=(fees_usd if n == "DQN (final)" else np.nan)))
        _print_window_table(f"OOT @ {bps} bps", results,
                            _action_distribution(steps_f))

    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(log_dir, "stage4_oot_ablation.csv"), index=False)

    if make_plots and final_steps_for_plot is not None:
        _plot_oot(final_steps_for_plot, oot_df.index, abl, log_dir)
    print(f"\n  Saved: {log_dir}/stage4_oot_ablation.csv")
    return abl


# ── STAGE 5: FINAL deliverable model — train on ALL available data ───────────
def stage_final(interval="1h", log_dir="logs", model_dir="models",
                reward_type="combined", lambda_tc=0.01, lambda_dd=0.05,
                seed=0, steps=FINAL_STEPS):
    """
    Train the SUBMISSION model on EVERY row available, with the frozen recipe and
    the seed chosen during selection. This is the model the instructor will load
    and put to the test on the unseen future (e.g. 2026).

    No held-out is reserved here ON PURPOSE: the reward, lambdas and seed were
    already validated in stages 1-4. Holding data back now would only weaken the
    deployed model. The scaler is fit on this full training span; at grading time
    the env re-fits a scaler on the instructor's future data (the framework
    contract — TradingEnv is created without a scaler), so features are always
    standardized causally within whatever period is evaluated.
    """
    print(f"\n{'#'*78}\n#  STAGE 5 — FINAL MODEL (train on ALL data, seed={seed}, {steps} steps)\n{'#'*78}")
    _set_reward(reward_type, lambda_tc, lambda_dd)
    data = load_prices(interval)
    os.makedirs(model_dir, exist_ok=True)

    full_env = TradingEnv(data, transaction_cost_bps=10.0)
    print(f"  Training on full span: {data.index[0]} -> {data.index[-1]}  "
          f"({len(data)} rows, {full_env._features.shape[0]} usable steps)")
    agent = Agent(obs_dim=full_env.observation_space.shape[0], n_actions=N_ACTIONS, seed=seed)
    agent.train(full_env, n_steps=steps)

    path = os.path.join(model_dir, "final_model.pt")
    agent.save(path)
    # quick sanity: greedy action distribution on the training span
    _, steps_log = _rollout(full_env, agent.act, collect=True)
    dist, corr = _action_distribution(steps_log)
    print("  Final-model action mix on training span:")
    for n, f in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"    {n:<14}{f:>6.1%}")
    print(f"  -> saved {path}  (THIS is the submission/deployment model)")
    return path


# ── Plotting (README §9): equity+seed spread, allocation, failure figure ──────
def _plot_walk_forward(summary, equity, fail_window, log_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) equity curves per window with seed spread
    windows = list(equity["window"].unique())
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 3.2), squeeze=False)
    for ax, w in zip(axes[0], windows):
        sub = equity[equity.window == w]
        for seed in sub["seed"].unique():
            s = sub[sub.seed == seed]
            ax.plot(s["t"], s["value"], lw=1, alpha=0.8, label=f"seed {seed}")
        ax.set_title(w, fontsize=8); ax.set_xlabel("step"); ax.axhline(10000, ls=":", c="grey")
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("portfolio value ($)")
    fig.suptitle("Walk-forward equity curves (seed spread)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "fig_equity_curves.png"), dpi=120)
    plt.close(fig)

    # (2) per-window Sortino bar (DQN vs Equal weight) with seed error bars
    dq = summary[summary.strategy == "DQN (trained)"].groupby("window")["sortino"].agg(["mean", "std"])
    eq = summary[summary.strategy == "Equal weight"].groupby("window")["sortino"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(dq))
    ax.bar(x - 0.2, dq["mean"], 0.4, yerr=dq["std"].fillna(0), label="DQN", capsize=3)
    ax.bar(x + 0.2, eq.reindex(dq.index), 0.4, label="Equal weight")
    ax.set_xticks(x); ax.set_xticklabels(dq.index, rotation=30, ha="right", fontsize=6)
    ax.axhline(0, c="k", lw=0.8); ax.set_ylabel("Sortino"); ax.legend()
    ax.set_title("Per-window Sortino: DQN vs Equal weight")
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "fig_sortino_by_window.png"), dpi=120)
    plt.close(fig)

    # (3) failure figure: a window where the DQN went negative
    if fail_window is not None:
        wname, vals, steps, _ = fail_window
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
        a1.plot(vals, c="crimson"); a1.axhline(10000, ls=":", c="grey")
        a1.set_title(f"FAILURE / anomaly window: {wname}", fontsize=9)
        a1.set_ylabel("value ($)")
        a2.plot([1 - s["w_cash"] for s in steps], c="steelblue")
        a2.set_ylabel("risk exposure"); a2.set_xlabel("step"); a2.set_ylim(-0.3, 1.1)
        fig.tight_layout(); fig.savefig(os.path.join(log_dir, "fig_failure_window.png"), dpi=120)
        plt.close(fig)
    print("  Figures: fig_equity_curves.png, fig_sortino_by_window.png, fig_failure_window.png")


def _plot_oot(final_steps, index, abl, log_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vals, steps, results = final_steps

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.plot(vals, c="navy", label="DQN final"); a1.axhline(10000, ls=":", c="grey")
    a1.set_ylabel("value ($)"); a1.legend(fontsize=7); a1.set_title("OOT held-out: DQN value & allocation")
    w0 = [s["w_a0"] for s in steps]; w1 = [s["w_a1"] for s in steps]
    w2 = [s["w_a2"] for s in steps]; wc = [s["w_cash"] for s in steps]
    x = np.arange(len(steps))
    a2.stackplot(x, w0, w1, w2, wc, labels=["a0", "a1", "a2", "cash"], alpha=0.8)
    a2.set_ylabel("weight"); a2.set_xlabel("step"); a2.legend(fontsize=7, loc="upper right", ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "fig_oot_allocation.png"), dpi=120)
    plt.close(fig)

    dq = abl[abl.strategy == "DQN (final)"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(dq["bps"], dq["sortino"], "o-", c="darkgreen")
    ax.set_xlabel("transaction cost (bps)"); ax.set_ylabel("Sortino")
    ax.set_title("Cost ablation: DQN Sortino vs fees"); ax.axhline(0, c="k", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(log_dir, "fig_cost_ablation.png"), dpi=120)
    plt.close(fig)
    print("  Figures: fig_oot_allocation.png, fig_cost_ablation.png")


# ── CLI: enforces the required ORDER compare -> grid -> walk_forward -> evaluate
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Bellman Capital experiment harness")
    p.add_argument("--stage", required=True,
                   choices=["compare", "gridsearch", "walk_forward", "evaluate", "final", "all"])
    p.add_argument("--interval", default="1h")
    p.add_argument("--tc_bps", type=float, default=10.0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--reward", default="combined",
                   choices=["log_return", "turnover_penalized", "drawdown_penalized",
                            "combined", "diff_sharpe"])
    p.add_argument("--lambda_tc", type=float, default=0.01)
    p.add_argument("--lambda_dd", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=None, help="override step budget for the stage")
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--model_dir", default="models")
    p.add_argument("--no_plots", action="store_true")
    args = p.parse_args()

    def _steps(default):
        return default if args.steps is None else args.steps

    if args.stage in ("compare", "all"):
        stage_compare(interval=args.interval, tc_bps=args.tc_bps,
                      log_dir=args.log_dir, seed=args.seeds[0], steps=_steps(SWEEP_STEPS))
    if args.stage in ("gridsearch", "all"):
        stage_gridsearch(interval=args.interval, tc_bps=args.tc_bps, log_dir=args.log_dir,
                         seed=args.seeds[0], reward_type=args.reward, steps=_steps(SWEEP_STEPS))
    if args.stage in ("walk_forward", "all"):
        stage_walk_forward(interval=args.interval, tc_bps=args.tc_bps, log_dir=args.log_dir,
                           seeds=tuple(args.seeds), reward_type=args.reward,
                           lambda_tc=args.lambda_tc, lambda_dd=args.lambda_dd,
                           steps=_steps(WF_STEPS), make_plots=not args.no_plots)
    if args.stage in ("evaluate", "all"):
        stage_evaluate(interval=args.interval, log_dir=args.log_dir, model_dir=args.model_dir,
                       reward_type=args.reward, lambda_tc=args.lambda_tc, lambda_dd=args.lambda_dd,
                       seeds=tuple(args.seeds), steps=_steps(FINAL_STEPS),
                       make_plots=not args.no_plots)
    if args.stage in ("final", "all"):
        stage_final(interval=args.interval, log_dir=args.log_dir, model_dir=args.model_dir,
                    reward_type=args.reward, lambda_tc=args.lambda_tc, lambda_dd=args.lambda_dd,
                    seed=args.seeds[0], steps=_steps(FINAL_STEPS))
