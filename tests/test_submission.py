"""
Bellman Capital — submission test suite.

Run with:
    uv run pytest tests/test_submission.py -v

A report is written to submission_report.txt after every run.
It shows what each test checked, whether it passed, and — on failure —
a plain-English hint explaining what to fix.

These tests use synthetic price data, so no real parquet files are needed.
All tests must pass before your submission will be graded.
"""
import numpy as np
import pandas as pd
import pytest

from agent import TradingEnv, Agent, N_ACTIONS, _ACTION_WEIGHTS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_prices(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with the columns BaseTradingEnv expects."""
    rng = np.random.default_rng(seed)
    assets = ["asset_0", "asset_1", "asset_2"]
    rows = {}
    for a in assets:
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n_rows))
        high  = close * (1 + rng.uniform(0, 0.005, n_rows))
        low   = close * (1 - rng.uniform(0, 0.005, n_rows))
        rows[f"{a}_close"]           = close
        rows[f"{a}_high"]            = high
        rows[f"{a}_low"]             = low
        rows[f"{a}_volume"]          = rng.uniform(1e4, 1e6, n_rows)
        rows[f"{a}_taker_buy_ratio"] = rng.uniform(0.3, 0.7, n_rows)
    rows["cash"] = np.ones(n_rows)  # cash price always 1
    idx = pd.date_range("2023-01-01", periods=n_rows, freq="1h")
    return pd.DataFrame(rows, index=idx)


@pytest.fixture
def prices():
    return _make_prices(n_rows=60)


@pytest.fixture
def env(prices):
    return TradingEnv(prices, transaction_cost_bps=10.0, initial_cash=10_000.0)


@pytest.fixture
def obs_dim(env):
    obs, _ = env.reset()
    return obs.shape[0]


@pytest.fixture
def agent(obs_dim):
    return Agent(obs_dim=obs_dim, n_actions=N_ACTIONS)


# ── Environment tests ─────────────────────────────────────────────────────────

class TestTradingEnv:

    def test_instantiation(self, prices):
        env = TradingEnv(prices)
        assert env is not None

    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32

    def test_reset_returns_dict_info(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_observation_is_finite(self, env):
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs)), "observation must not contain NaN or Inf"

    def test_step_returns_correct_types(self, env):
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_shape_consistent(self, env):
        obs0, _ = env.reset()
        obs1, *_ = env.step(0)
        assert obs1.shape == obs0.shape

    def test_all_actions_produce_valid_step(self, env):
        for action in range(N_ACTIONS):
            env.reset()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)

    def test_portfolio_value_in_info(self, env):
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "portfolio_value" in info
        assert info["portfolio_value"] > 0

    def test_episode_terminates(self, env):
        env.reset()
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
        assert steps > 0

    def test_action_space_consistent_with_n_actions(self, env):
        assert env.action_space.n == N_ACTIONS

    def test_observation_matches_observation_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs), \
            "obs must lie within declared observation_space"


class TestWeights:

    @pytest.mark.parametrize("action", range(N_ACTIONS))
    def test_weights_sum_to_one(self, action):
        w = _ACTION_WEIGHTS[action]
        assert np.isclose(w.sum(), 1.0, atol=1e-4), f"action {action}: weights sum to {w.sum()}"

    @pytest.mark.parametrize("action", range(N_ACTIONS))
    def test_cash_weight_non_negative(self, action):
        w = _ACTION_WEIGHTS[action]
        assert w[3] >= 0, f"action {action}: cash weight {w[3]} is negative"

    @pytest.mark.parametrize("action", range(N_ACTIONS))
    def test_risky_weights_in_bounds(self, action):
        w = _ACTION_WEIGHTS[action]
        assert np.all(w[:3] >= -1.0) and np.all(w[:3] <= 1.0), \
            f"action {action}: risky weights out of [-1, 1]"

    def test_all_cash_action_has_zero_risky(self):
        w = _ACTION_WEIGHTS[0]
        assert np.allclose(w[:3], 0.0)
        assert np.isclose(w[3], 1.0)


class TestReward:

    def test_reward_is_finite(self, env):
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert np.isfinite(reward)

    def test_reward_positive_when_value_grows(self, env):
        r = env._reward(10_000.0, 10_100.0)
        assert r > 0

    def test_reward_negative_when_value_shrinks(self, env):
        r = env._reward(10_000.0, 9_900.0)
        assert r < 0

    def test_reward_zero_when_value_unchanged(self, env):
        r = env._reward(10_000.0, 10_000.0)
        assert np.isclose(r, 0.0, atol=1e-5)


# ── Agent tests ───────────────────────────────────────────────────────────────

class TestAgent:

    def test_instantiation(self, agent):
        assert agent is not None

    def test_act_returns_valid_action(self, env, agent):
        obs, _ = env.reset()
        action = agent.act(obs)
        assert isinstance(action, int)
        assert 0 <= action < N_ACTIONS

    def test_act_is_deterministic(self, env, agent):
        obs, _ = env.reset()
        a1 = agent.act(obs)
        a2 = agent.act(obs)
        assert a1 == a2

    def test_train_runs_without_error(self, obs_dim):
        """Short training run (200 steps) to verify the loop doesn't crash."""
        env = TradingEnv(_make_prices(n_rows=60))
        ag  = Agent(obs_dim=obs_dim, n_actions=N_ACTIONS)
        ag.train(env, n_steps=200)

    def test_act_after_training_returns_valid_action(self, obs_dim):
        env = TradingEnv(_make_prices(n_rows=60))
        ag  = Agent(obs_dim=obs_dim, n_actions=N_ACTIONS)
        ag.train(env, n_steps=200)
        obs, _ = env.reset()
        action = ag.act(obs)
        assert 0 <= action < N_ACTIONS

    def test_epsilon_decays_during_training(self, prices, obs_dim):
        env = TradingEnv(_make_prices(n_rows=60))
        ag  = Agent(obs_dim=obs_dim, n_actions=N_ACTIONS)
        epsilon_before = ag.epsilon
        ag.train(env, n_steps=200)
        assert ag.epsilon < epsilon_before


# ── Integration test ──────────────────────────────────────────────────────────

class TestFullRollout:

    def test_untrained_agent_completes_full_episode(self, env, agent):
        obs, _ = env.reset()
        terminated = truncated = False
        portfolio_values = []
        while not (terminated or truncated):
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])

        assert len(portfolio_values) > 0
        assert all(np.isfinite(v) for v in portfolio_values)

    def test_portfolio_value_stays_positive(self, env, agent):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            assert info["portfolio_value"] > 0
