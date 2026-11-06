"""Cấu hình toàn bộ mô phỏng (thị trường + Q-Learning + huấn luyện)."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # ---- Thị trường -------------------------------------------------------
    base_price: float = 100.0      # giá bán niêm yết (nghìn đồng)
    cost: float = 60.0             # giá vốn
    discount: float = 0.20         # mức giảm giá khi chọn hành động "Giảm giá"
    quality: float = 8.0           # độ hấp dẫn cơ bản của sản phẩm (a)
    beta: float = 8.0              # độ nhạy cảm với giá (b)
    outside_high: float = 0.0      # tiện ích "không mua" khi nhu cầu cao
    outside_low: float = 1.0       # tiện ích "không mua" khi nhu cầu thấp
    n_high: int = 100              # số khách ghé thăm mỗi vòng khi nhu cầu cao
    n_low: int = 40                # số khách ghé thăm mỗi vòng khi nhu cầu thấp
    p_high: float = 0.5            # xác suất thị trường ở mức nhu cầu cao
    persistence: float = 0.7       # xác suất giữ nguyên mức nhu cầu ở vòng kế tiếp
    steps_per_episode: int = 30    # số vòng (ngày) trong 1 episode

    rule: str = "demand"          # luật của Agent B (xem agents.RULES)

    # ---- Q-Learning ---------------------------------------------------------
    alpha: float = 0.1             # tốc độ học
    gamma: float = 0.9             # hệ số chiết khấu
    epsilon: float = 0.5           # epsilon ban đầu (khám phá)
    epsilon_decay: float = 0.999   # hệ số suy giảm epsilon sau mỗi episode
    epsilon_min: float = 0.01      # epsilon tối thiểu

    # ---- Huấn luyện / đánh giá ---------------------------------------------
    train_episodes: int = 5000
    eval_episodes: int = 500
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)
