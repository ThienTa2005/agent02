"""Vòng lặp mô phỏng, huấn luyện và đánh giá."""
import time
from dataclasses import dataclass, field, replace

import numpy as np

from .agents import QLearningAgent, RuleBasedAgent
from .config import Config
from .environment import DISCOUNT, HIGH, HOLD, LOW, PricingEnv


def play_episodes(env, agent, rule, n_episodes, rng, learn, explore):
    """Chạy n episode. Trả về các mảng thống kê theo từng episode.

    learn   : có cập nhật Q-Table hay không
    explore : có dùng epsilon-greedy hay không (False = chọn tham lam hoàn toàn)
    """
    cfg = env.cfg
    T = cfg.steps_per_episode
    pers, p_high = cfg.persistence, cfg.p_high
    margin = env.margin

    u_d = rng.random((n_episodes, T + 1)).tolist()
    z = rng.standard_normal((n_episodes, T, 2)).tolist()
    u_e = rng.random((n_episodes, T)).tolist()
    r_act = rng.integers(0, 2, (n_episodes, T)).tolist()

    keys = ["profit_a", "profit_b", "sales_a", "sales_b",
            "disc_a", "disc_b", "n_high", "disc_a_high", "disc_b_high"]
    out = {k: np.zeros(n_episodes) for k in keys}

    for e in range(n_episodes):
        ud, ze, ue, ra = u_d[e], z[e], u_e[e], r_act[e]
        d = HIGH if ud[0] < p_high else LOW
        a_last = b_last = HOLD
        pa_t = pb_t = sa_t = sb_t = da = db = nh = dah = dbh = 0
        for t in range(T):
            s = d * 4 + a_last * 2 + b_last
            if explore and ue[t] < agent.epsilon:
                a = ra[t]
            else:
                a = agent.greedy(s)
            b = rule.act(d, a_last)
            sa, sb = env.sample_sales(d, a, b, ze[t][0], ze[t][1])
            pa = sa * margin[a]
            pb = sb * margin[b]

            u = ud[t + 1]
            if u < pers or pers >= 1.0:
                d2 = d
            else:
                d2 = HIGH if (u - pers) / (1.0 - pers) < p_high else LOW

            if learn:
                agent.update(s, a, pa, d2 * 4 + a * 2 + b)

            pa_t += pa; pb_t += pb; sa_t += sa; sb_t += sb
            da += a; db += b
            if d == HIGH:
                nh += 1; dah += a; dbh += b
            d, a_last, b_last = d2, a, b
        if learn and explore:
            agent.decay()
        out["profit_a"][e], out["profit_b"][e] = pa_t, pb_t
        out["sales_a"][e], out["sales_b"][e] = sa_t, sb_t
        out["disc_a"][e], out["disc_b"][e] = da, db
        out["n_high"][e], out["disc_a_high"][e], out["disc_b_high"][e] = nh, dah, dbh
    return out


def summarize(res: dict, steps: int) -> dict:
    """Các chỉ số tổng hợp từ kết quả đánh giá."""
    n = len(res["profit_a"])
    total = n * steps
    nh = res["n_high"].sum()
    nl = total - nh

    def safe(x, y):
        return float(x / y) if y else float("nan")

    pa, pb = res["profit_a"], res["profit_b"]
    gap = pa - pb
    return {
        "profit_a_mean": float(pa.mean()), "profit_b_mean": float(pb.mean()),
        "profit_a_std": float(pa.std()), "profit_b_std": float(pb.std()),
        "profit_gap_mean": float(gap.mean()), "profit_gap_std": float(gap.std()),
        "win_rate_a": float((pa > pb).mean()),
        "cv_a": safe(pa.std(), pa.mean()), "cv_b": safe(pb.std(), pb.mean()),
        "sales_a_mean": float(res["sales_a"].mean()),
        "sales_b_mean": float(res["sales_b"].mean()),
        "disc_rate_a": safe(res["disc_a"].sum(), total),
        "disc_rate_b": safe(res["disc_b"].sum(), total),
        "disc_rate_a_high": safe(res["disc_a_high"].sum(), nh),
        "disc_rate_a_low": safe(res["disc_a"].sum() - res["disc_a_high"].sum(), nl),
        "disc_rate_b_high": safe(res["disc_b_high"].sum(), nh),
        "disc_rate_b_low": safe(res["disc_b"].sum() - res["disc_b_high"].sum(), nl),
    }


@dataclass
class RunResult:
    label: str
    params: dict
    config: dict
    seed: int
    train: dict            # đường cong huấn luyện theo episode (np.ndarray)
    eval: dict             # kết quả từng episode ở giai đoạn đánh giá
    summary: dict
    q_table: np.ndarray
    policy: list
    checkpoints: list = field(default_factory=list)
    final_epsilon: float = 0.0
    elapsed: float = 0.0


def run_one(cfg: Config, label="Baseline", params=None, seed=None,
            checkpoints=None, checkpoint_eval_episodes=200) -> RunResult:
    """Huấn luyện Agent A (train_episodes) rồi đánh giá riêng (eval_episodes)."""
    t0 = time.time()
    seed = cfg.seed if seed is None else seed
    env = PricingEnv(cfg)
    agent = QLearningAgent(cfg.alpha, cfg.gamma, cfg.epsilon,
                           cfg.epsilon_decay, cfg.epsilon_min)
    rule = RuleBasedAgent(cfg.rule)
    train_rng = np.random.default_rng(seed)

    E = cfg.train_episodes
    marks = sorted({int(c) for c in (checkpoints or []) if 0 < int(c) < E})
    bounds = marks + [E]
    chunks, ckpts, start = [], [], 0
    for i, end in enumerate(bounds):
        chunks.append(play_episodes(env, agent, rule, end - start, train_rng,
                                    learn=True, explore=True))
        start = end
        if end in marks:
            ck_rng = np.random.default_rng(seed + 1000 + i)
            r = play_episodes(env, agent, rule, checkpoint_eval_episodes, ck_rng,
                              learn=False, explore=False)
            ckpts.append({"episode": end,
                          "profit_a": float(r["profit_a"].mean()),
                          "profit_b": float(r["profit_b"].mean()),
                          "epsilon": agent.epsilon})
    tr = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    T = cfg.steps_per_episode
    train = {"profit_a": tr["profit_a"], "profit_b": tr["profit_b"],
             "disc_rate_a": tr["disc_a"] / T, "disc_rate_b": tr["disc_b"] / T}

    # Đánh giá riêng: epsilon = 0, không học, seed môi trường khác
    ev_rng = np.random.default_rng(seed + 99991)
    ev = play_episodes(env, agent, rule, cfg.eval_episodes, ev_rng,
                       learn=False, explore=False)
    if checkpoints:
        ckpts.append({"episode": E,
                      "profit_a": float(ev["profit_a"].mean()),
                      "profit_b": float(ev["profit_b"].mean()),
                      "epsilon": agent.epsilon})
    return RunResult(
        label=label, params=params or {}, config=cfg.to_dict(), seed=seed,
        train=train, eval={"profit_a": ev["profit_a"], "profit_b": ev["profit_b"]},
        summary=summarize(ev, T), q_table=agent.q_table, policy=agent.policy,
        checkpoints=ckpts, final_epsilon=agent.epsilon,
        elapsed=time.time() - t0)
