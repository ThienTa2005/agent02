"""Chạy toàn bộ thí nghiệm bằng dòng lệnh và xuất CSV/JSON vào thư mục `outputs/`.

    python run_experiments.py              # tham số mặc định
    python run_experiments.py --alpha 0.05 --discount 0.15 --train 5000
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from arena.config import Config
from arena.experiments import (EXPERIMENTS, curves_to_dataframe, results_to_json,
                               runs_to_dataframe)


def main():
    p = argparse.ArgumentParser()
    d = Config()
    p.add_argument("--base-price", type=float, default=d.base_price)
    p.add_argument("--cost", type=float, default=d.cost)
    p.add_argument("--discount", type=float, default=d.discount)
    p.add_argument("--alpha", type=float, default=d.alpha)
    p.add_argument("--gamma", type=float, default=d.gamma)
    p.add_argument("--epsilon", type=float, default=d.epsilon)
    p.add_argument("--train", type=int, default=d.train_episodes)
    p.add_argument("--eval", type=int, default=d.eval_episodes)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    cfg = replace(d, base_price=a.base_price, cost=a.cost, discount=a.discount, alpha=a.alpha,
                  gamma=a.gamma, epsilon=a.epsilon, train_episodes=a.train,
                  eval_episodes=a.eval, seed=a.seed)
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    results = {}
    for k, spec in EXPERIMENTS.items():
        print(f"Đang chạy {spec.title} ...")
        results[k] = spec.func(cfg)
    summary = pd.concat([runs_to_dataframe(k, r) for k, r in results.items()], ignore_index=True)
    summary.to_csv(out / "ket_qua_thi_nghiem.csv", index=False, encoding="utf-8-sig")
    curves_to_dataframe(results).to_csv(out / "duong_hoc.csv", index=False, encoding="utf-8-sig")
    (out / "ket_qua_day_du.json").write_text(
        json.dumps({"config": cfg.to_dict(), "experiments": results_to_json(results)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[["experiment", "label", "profit_a_mean", "profit_b_mean", "win_rate_a"]].round(2).to_string())
    print(f"Đã lưu kết quả vào {out.resolve()}")


if __name__ == "__main__":
    main()
