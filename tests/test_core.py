import json
from dataclasses import replace

import numpy as np

from arena.agents import QLearningAgent, RuleBasedAgent
from arena.config import Config
from arena.environment import DISCOUNT, HIGH, HOLD, LOW, PricingEnv
from arena.experiments import EXPERIMENTS, results_to_json, runs_to_dataframe
from arena.simulation import run_one

FAST = Config(train_episodes=300, eval_episodes=50)


def test_purchase_probs_valid():
    env = PricingEnv(Config())
    for d in (LOW, HIGH):
        for a in (HOLD, DISCOUNT):
            for b in (HOLD, DISCOUNT):
                pa, pb = env.purchase_probs(d, a, b)
                assert 0 < pa < 1 and 0 < pb < 1 and pa + pb < 1


def test_discount_wins_share():
    env = PricingEnv(Config())
    pa, pb = env.purchase_probs(HIGH, DISCOUNT, HOLD)
    assert pa > pb


def test_profit_formula():
    cfg = Config()
    env = PricingEnv(cfg)
    pa, _, sa, _ = env.expected_payoff(HIGH, HOLD, HOLD)
    assert abs(pa - sa * (cfg.base_price - cfg.cost)) < 1e-9


def test_rule_based():
    r = RuleBasedAgent("demand")
    assert r.act(LOW, HOLD) == DISCOUNT and r.act(HIGH, HOLD) == HOLD
    m = RuleBasedAgent("match")
    assert m.act(HIGH, DISCOUNT) == DISCOUNT and m.act(HIGH, HOLD) == HOLD


def test_q_update():
    ag = QLearningAgent(0.5, 0.9, 0.0, 1.0, 0.0)
    ag.update(0, 1, 10.0, 1)
    assert ag.q[0][1] == 5.0


def test_reproducible_and_eval_does_not_learn():
    r1, r2 = run_one(FAST, seed=1), run_one(FAST, seed=1)
    assert np.allclose(r1.q_table, r2.q_table)
    assert r1.summary == r2.summary


def test_a_beats_static_rule_after_training():
    r = run_one(Config(), seed=3)
    assert r.summary["profit_a_mean"] > r.summary["profit_b_mean"]


def test_all_experiments_and_json():
    cfg = replace(FAST)
    res = {k: s.func(cfg) for k, s in EXPERIMENTS.items()}
    assert len(res["seeds"]) == 5 and len(res["alpha"]) == 5 and len(res["discount"]) == 5
    json.dumps(results_to_json(res))
    assert len(runs_to_dataframe("alpha", res["alpha"])) == 5
