"""Các thí nghiệm: Baseline, diễn biến học, Alpha, Epsilon, 5 seed, mức giảm giá."""
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd

from .config import Config
from .simulation import RunResult, run_one

ALPHA_VALUES = (0.01, 0.05, 0.1, 0.3, 0.5)
EPSILON_VALUES = (0.05, 0.1, 0.3, 0.5, 0.9)
DISCOUNT_VALUES = (0.10, 0.15, 0.20, 0.25, 0.30)
N_SEEDS = 5


def _noop(*_):
    pass


def exp_baseline(cfg: Config, progress=_noop):
    progress(0, 1, "Baseline")
    runs = [run_one(cfg, label="Baseline")]
    progress(1, 1, "Baseline")
    return runs


def exp_learning(cfg: Config, progress=_noop):
    E = cfg.train_episodes
    marks = sorted({m for m in (50, 100, 250, 500, 1000, 2000, 3000, 4000) if m < E})
    progress(0, 1, "Diễn biến học")
    runs = [run_one(cfg, label="Diễn biến học", checkpoints=marks)]
    progress(1, 1, "Diễn biến học")
    return runs


def _sweep(cfg, field, values, fmt, progress):
    runs = []
    for i, v in enumerate(values):
        label = fmt(v)
        progress(i, len(values), label)
        runs.append(run_one(replace(cfg, **{field: v}), label=label, params={field: v}))
    progress(len(values), len(values), "xong")
    return runs


def exp_alpha(cfg: Config, progress=_noop):
    return _sweep(cfg, "alpha", ALPHA_VALUES, lambda v: f"α = {v:g}", progress)


def exp_epsilon(cfg: Config, progress=_noop):
    return _sweep(cfg, "epsilon", EPSILON_VALUES, lambda v: f"ε₀ = {v:g}", progress)


def exp_discount(cfg: Config, progress=_noop):
    return _sweep(cfg, "discount", DISCOUNT_VALUES, lambda v: f"Giảm {v:.0%}", progress)


def exp_seeds(cfg: Config, progress=_noop):
    runs = []
    for i in range(N_SEEDS):
        seed = cfg.seed + i
        progress(i, N_SEEDS, f"seed {seed}")
        runs.append(run_one(cfg, label=f"Seed {seed}", params={"seed": seed}, seed=seed))
    progress(N_SEEDS, N_SEEDS, "xong")
    return runs


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    title: str
    description: str
    func: Callable


EXPERIMENTS = {
    s.key: s for s in [
        ExperimentSpec("baseline", "1. Baseline",
                       "Cấu hình hiện tại: huấn luyện Agent A rồi đánh giá riêng.", exp_baseline),
        ExperimentSpec("learning", "2. Diễn biến học",
                       "Đánh giá Agent A tại nhiều mốc episode để xem quá trình học.", exp_learning),
        ExperimentSpec("alpha", "3. Thay đổi Alpha",
                       f"Alpha ∈ {list(ALPHA_VALUES)}.", exp_alpha),
        ExperimentSpec("epsilon", "4. Thay đổi Epsilon",
                       f"Epsilon ban đầu ∈ {list(EPSILON_VALUES)} (vẫn suy giảm theo decay).", exp_epsilon),
        ExperimentSpec("seeds", "5. Năm random seed",
                       "Lặp lại baseline với 5 seed liên tiếp để đo độ ổn định.", exp_seeds),
        ExperimentSpec("discount", "6. Mức giảm giá 10%–30%",
                       f"Mức giảm giá ∈ {[f'{v:.0%}' for v in DISCOUNT_VALUES]}.", exp_discount),
    ]
}


# --------------------------------------------------------------------------
# Bảng kết quả & xuất dữ liệu
# --------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "profit_a_mean", "profit_b_mean", "profit_gap_mean", "profit_a_std", "profit_b_std",
    "cv_a", "cv_b", "win_rate_a", "sales_a_mean", "sales_b_mean",
    "disc_rate_a", "disc_rate_b", "disc_rate_a_high", "disc_rate_a_low",
]


def runs_to_dataframe(exp_key: str, runs: list) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {"experiment": exp_key, "label": r.label, "seed": r.seed}
        row.update({k: r.summary[k] for k in SUMMARY_COLUMNS})
        row["policy"] = "".join(str(a) for a in r.policy)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_stability(runs: list) -> pd.DataFrame:
    """Trung bình ± độ lệch chuẩn giữa các seed (độ ổn định của thuật toán)."""
    df = runs_to_dataframe("seeds", runs)
    cols = ["profit_a_mean", "profit_b_mean", "profit_gap_mean", "win_rate_a", "disc_rate_a"]
    out = pd.DataFrame({"mean": df[cols].mean(), "std": df[cols].std(ddof=1),
                        "min": df[cols].min(), "max": df[cols].max()})
    out["cv"] = out["std"] / out["mean"].abs()
    return out


def _block_mean(x: np.ndarray, block: int = 50) -> list:
    n = (len(x) // block) * block
    if n == 0:
        return [float(np.mean(x))]
    return np.asarray(x[:n]).reshape(-1, block).mean(axis=1).round(3).tolist()


def run_to_dict(r: RunResult) -> dict:
    return {
        "label": r.label, "params": r.params, "seed": r.seed, "config": r.config,
        "summary": r.summary, "q_table": np.round(r.q_table, 3).tolist(),
        "policy": r.policy, "final_epsilon": r.final_epsilon,
        "checkpoints": r.checkpoints,
        "train_curve_block50": {k: _block_mean(v) for k, v in r.train.items()},
    }


def results_to_json(results: dict) -> dict:
    """results: {exp_key: list[RunResult]} -> dict có thể json.dumps."""
    return {k: [run_to_dict(r) for r in runs] for k, runs in results.items()}


def curves_to_dataframe(results: dict) -> pd.DataFrame:
    rows = []
    for k, runs in results.items():
        for r in runs:
            n = len(r.train["profit_a"])
            for i in range(0, n - 49, 50):
                rows.append({
                    "experiment": k, "label": r.label, "episode_end": i + 50,
                    "profit_a": r.train["profit_a"][i:i + 50].mean(),
                    "profit_b": r.train["profit_b"][i:i + 50].mean(),
                    "disc_rate_a": r.train["disc_rate_a"][i:i + 50].mean(),
                    "disc_rate_b": r.train["disc_rate_b"][i:i + 50].mean(),
                })
    return pd.DataFrame(rows)
