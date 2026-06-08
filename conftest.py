"""
Submission report generator.

Automatically writes submission_report.txt after every pytest run.
Open that file to see what each test checked and — on failure — exactly
what to fix before submitting.
"""
import pytest

# Human-readable description and guidance for each test node ID
_DESCRIPTIONS = {
    # ── Environment ──────────────────────────────────────────────────────────
    "tests/test_submission.py::TestTradingEnv::test_instantiation": (
        "TradingEnv can be created with a prices DataFrame",
        "Make sure TradingEnv.__init__ calls super().__init__ and sets self._lookback, "
        "self.action_space, and self.observation_space without raising."
    ),
    "tests/test_submission.py::TestTradingEnv::test_reset_returns_correct_shape": (
        "env.reset() returns an observation with the declared shape",
        "The array returned by _obs() must match observation_space.shape exactly. "
        "Double-check that (lookback-1)*4 + 4 equals the shape you declared."
    ),
    "tests/test_submission.py::TestTradingEnv::test_reset_returns_dict_info": (
        "env.reset() returns a dict as the second element",
        "BaseTradingEnv.reset() already does this — make sure you call super().reset()."
    ),
    "tests/test_submission.py::TestTradingEnv::test_observation_is_finite": (
        "Observation contains no NaN or Inf values",
        "NaNs often come from dividing by zero or taking log(0). Add a small epsilon "
        "(1e-8) to denominators in _obs()."
    ),
    "tests/test_submission.py::TestTradingEnv::test_step_returns_correct_types": (
        "env.step() returns (obs, float, bool, bool, dict)",
        "_reward() must return a Python float. _obs() must return np.ndarray. "
        "terminated/truncated are booleans from BaseTradingEnv."
    ),
    "tests/test_submission.py::TestTradingEnv::test_step_obs_shape_consistent": (
        "Observation shape is the same before and after a step",
        "_obs() must always return the same shape regardless of the current timestep."
    ),
    "tests/test_submission.py::TestTradingEnv::test_all_actions_produce_valid_step": (
        "Every action produces a finite obs and reward",
        "Test each of your N_ACTIONS actions. If one produces NaN, check "
        "_weights_from_action() for that index."
    ),
    "tests/test_submission.py::TestTradingEnv::test_portfolio_value_in_info": (
        "info dict from step() contains 'portfolio_value' key",
        "BaseTradingEnv.step() already adds this — don't override step()."
    ),
    "tests/test_submission.py::TestTradingEnv::test_episode_terminates": (
        "A full episode eventually ends (terminated=True)",
        "The episode ends when self._t >= len(self.prices). Make sure _lookback is "
        "smaller than the number of price rows."
    ),
    "tests/test_submission.py::TestTradingEnv::test_action_space_consistent_with_n_actions": (
        "action_space.n matches the number of actions defined in _ACTION_WEIGHTS",
        "N_ACTIONS and the Discrete space must agree."
    ),
    "tests/test_submission.py::TestTradingEnv::test_observation_matches_observation_space": (
        "Observation lies inside the declared observation_space",
        "The shape, dtype, and bounds of _obs() must all match observation_space."
    ),
    # ── Weights ───────────────────────────────────────────────────────────────
    "tests/test_submission.py::TestWeights::test_weights_sum_to_one[0]": (
        "Action 0 weights sum to 1",
        "All weight vectors must sum exactly to 1.0 (tolerance 1e-4)."
    ),
    "tests/test_submission.py::TestWeights::test_weights_sum_to_one[1]": (
        "Action 1 weights sum to 1",
        "All weight vectors must sum exactly to 1.0 (tolerance 1e-4)."
    ),
    "tests/test_submission.py::TestWeights::test_weights_sum_to_one[2]": (
        "Action 2 weights sum to 1",
        "All weight vectors must sum exactly to 1.0 (tolerance 1e-4)."
    ),
    "tests/test_submission.py::TestWeights::test_weights_sum_to_one[3]": (
        "Action 3 weights sum to 1",
        "All weight vectors must sum exactly to 1.0 (tolerance 1e-4)."
    ),
    "tests/test_submission.py::TestWeights::test_weights_sum_to_one[4]": (
        "Action 4 weights sum to 1",
        "All weight vectors must sum exactly to 1.0 (tolerance 1e-4)."
    ),
    "tests/test_submission.py::TestWeights::test_cash_weight_non_negative[0]": (
        "Action 0 cash weight >= 0",
        "w[3] is the cash allocation. Cash cannot be shorted."
    ),
    "tests/test_submission.py::TestWeights::test_cash_weight_non_negative[1]": (
        "Action 1 cash weight >= 0",
        "w[3] is the cash allocation. Cash cannot be shorted."
    ),
    "tests/test_submission.py::TestWeights::test_cash_weight_non_negative[2]": (
        "Action 2 cash weight >= 0",
        "w[3] is the cash allocation. Cash cannot be shorted."
    ),
    "tests/test_submission.py::TestWeights::test_cash_weight_non_negative[3]": (
        "Action 3 cash weight >= 0",
        "w[3] is the cash allocation. Cash cannot be shorted."
    ),
    "tests/test_submission.py::TestWeights::test_cash_weight_non_negative[4]": (
        "Action 4 cash weight >= 0",
        "w[3] is the cash allocation. Cash cannot be shorted."
    ),
    "tests/test_submission.py::TestWeights::test_risky_weights_in_bounds[0]": (
        "Action 0 risky weights in [-1, 1]",
        "w[0], w[1], w[2] must each be in [-1.0, 1.0]."
    ),
    "tests/test_submission.py::TestWeights::test_risky_weights_in_bounds[1]": (
        "Action 1 risky weights in [-1, 1]",
        "w[0], w[1], w[2] must each be in [-1.0, 1.0]."
    ),
    "tests/test_submission.py::TestWeights::test_risky_weights_in_bounds[2]": (
        "Action 2 risky weights in [-1, 1]",
        "w[0], w[1], w[2] must each be in [-1.0, 1.0]."
    ),
    "tests/test_submission.py::TestWeights::test_risky_weights_in_bounds[3]": (
        "Action 3 risky weights in [-1, 1]",
        "w[0], w[1], w[2] must each be in [-1.0, 1.0]."
    ),
    "tests/test_submission.py::TestWeights::test_risky_weights_in_bounds[4]": (
        "Action 4 risky weights in [-1, 1]",
        "w[0], w[1], w[2] must each be in [-1.0, 1.0]."
    ),
    "tests/test_submission.py::TestWeights::test_all_cash_action_has_zero_risky": (
        "Action 0 is a pure cash allocation (risky weights = 0)",
        "At least one action should be all-cash: [0, 0, 0, 1]. "
        "This is the safe baseline the agent can always fall back to."
    ),
    # ── Reward ────────────────────────────────────────────────────────────────
    "tests/test_submission.py::TestReward::test_reward_is_finite": (
        "_reward() returns a finite scalar",
        "Avoid log(0) or division by zero. Guard with: log(curr / (prev + 1e-8))."
    ),
    "tests/test_submission.py::TestReward::test_reward_positive_when_value_grows": (
        "_reward() is positive when portfolio value increases",
        "log(curr / prev) > 0 when curr > prev. Make sure you have the fraction the right way up."
    ),
    "tests/test_submission.py::TestReward::test_reward_negative_when_value_shrinks": (
        "_reward() is negative when portfolio value decreases",
        "log(curr / prev) < 0 when curr < prev."
    ),
    "tests/test_submission.py::TestReward::test_reward_zero_when_value_unchanged": (
        "_reward() is ~0 when portfolio value is unchanged",
        "log(1) = 0. Any constant multiplier will break this."
    ),
    # ── Agent ─────────────────────────────────────────────────────────────────
    "tests/test_submission.py::TestAgent::test_instantiation": (
        "Agent can be created with obs_dim and n_actions",
        "Make sure __init__ does not call raise NotImplementedError and initialises "
        "q_net, optimizer, and replay buffer."
    ),
    "tests/test_submission.py::TestAgent::test_act_returns_valid_action": (
        "agent.act(obs) returns an integer in [0, N_ACTIONS)",
        "act() should return int(q_net(obs).argmax()). Never return a tensor."
    ),
    "tests/test_submission.py::TestAgent::test_act_is_deterministic": (
        "agent.act(obs) returns the same action when called twice",
        "Use torch.no_grad() and don't apply epsilon-greedy inside act(). "
        "Exploration is only for training."
    ),
    "tests/test_submission.py::TestAgent::test_train_runs_without_error": (
        "agent.train(env, n_steps=200) completes without exception",
        "Common errors: buffer not large enough before sampling, wrong tensor shapes, "
        "forgetting env.reset() at episode end."
    ),
    "tests/test_submission.py::TestAgent::test_act_after_training_returns_valid_action": (
        "act() still returns a valid action after training",
        "Training must not corrupt the network or change obs_dim / n_actions."
    ),
    "tests/test_submission.py::TestAgent::test_epsilon_decays_during_training": (
        "Epsilon decreases during training (exploration decays)",
        "Epsilon must start at 1.0 and shrink each step. "
        "Without decay the agent never exploits what it has learned."
    ),
    # ── Integration ───────────────────────────────────────────────────────────
    "tests/test_submission.py::TestFullRollout::test_untrained_agent_completes_full_episode": (
        "Untrained agent can run a complete episode without errors",
        "This is an end-to-end smoke test. If it fails, the env or agent has a "
        "structural bug (wrong shape, illegal action, NaN propagation)."
    ),
    "tests/test_submission.py::TestFullRollout::test_portfolio_value_stays_positive": (
        "Portfolio value is always > 0 throughout an episode",
        "If value goes to 0 or negative, transaction costs are too high or your "
        "reward signal is leading the agent into ruin. Consider starting with long-only actions."
    ),
}

_WARNINGS = [
    "IMPORTANT: passing all tests does NOT guarantee a good Sortino ratio.",
    "These tests only verify that your code is structurally correct.",
    "Make sure you also run a walk-forward evaluation and compare against the baselines.",
    "Never train or evaluate on data after the train_end date — that is look-ahead.",
    "Fit your StandardScaler ONLY on training data (fit=True), then reuse it on eval data.",
    "Short positions require extra care: a wrong-way short loses money when the asset rises.",
]

_REPORT_PATH = "submission_report.txt"


class _ReportPlugin:
    def __init__(self):
        self._results: list[tuple[str, str, str | None]] = []  # (nodeid, outcome, longrepr)

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        nodeid  = report.nodeid
        outcome = report.outcome.upper()   # PASSED / FAILED / ERROR
        longrepr = str(report.longrepr) if report.failed else None
        self._results.append((nodeid, outcome, longrepr))

    def pytest_sessionfinish(self, session, exitstatus):
        passed = sum(1 for _, o, _ in self._results if o == "PASSED")
        failed = sum(1 for _, o, _ in self._results if o != "PASSED")
        total  = len(self._results)

        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("  BELLMAN CAPITAL — SUBMISSION REPORT")
        lines.append("=" * 72)
        lines.append(f"  Result : {'ALL TESTS PASSED' if failed == 0 else f'{failed} FAILED / {passed} PASSED'}")
        lines.append(f"  Total  : {total} tests")
        lines.append("=" * 72)
        lines.append("")

        sections = [
            ("ENVIRONMENT", "TestTradingEnv"),
            ("PORTFOLIO WEIGHTS", "TestWeights"),
            ("REWARD FUNCTION", "TestReward"),
            ("AGENT", "TestAgent"),
            ("FULL ROLLOUT (INTEGRATION)", "TestFullRollout"),
        ]

        for section_title, class_key in sections:
            section_results = [(n, o, l) for n, o, l in self._results if class_key in n]
            if not section_results:
                continue
            lines.append(f"── {section_title} {'─' * (66 - len(section_title))}")
            for nodeid, outcome, longrepr in section_results:
                icon   = "✓" if outcome == "PASSED" else "✗"
                desc, guidance = _DESCRIPTIONS.get(nodeid, (nodeid.split("::")[-1], ""))
                lines.append(f"  {icon} [{outcome:<6}]  {desc}")
                if outcome != "PASSED":
                    lines.append(f"             GUIDANCE : {guidance}")
                    if longrepr:
                        # Trim long tracebacks to the last 10 lines
                        tb_lines = longrepr.strip().splitlines()
                        trimmed  = tb_lines[-10:] if len(tb_lines) > 10 else tb_lines
                        lines.append("             TRACEBACK (last 10 lines):")
                        for tl in trimmed:
                            lines.append(f"               {tl}")
                    lines.append("")
            lines.append("")

        lines.append("── WARNINGS " + "─" * 60)
        for w in _WARNINGS:
            lines.append(f"  ⚠  {w}")
        lines.append("")
        lines.append("=" * 72)
        if failed == 0:
            lines.append("  All checks passed. Good luck on the evaluation!")
        else:
            lines.append(f"  Fix the {failed} failing test(s) above before submitting.")
        lines.append("=" * 72)

        report_text = "\n".join(lines)
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"\n  Report written to: {_REPORT_PATH}")


def pytest_configure(config):
    config.pluginmanager.register(_ReportPlugin(), "submission_report")
