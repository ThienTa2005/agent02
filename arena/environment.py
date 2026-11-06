"""Môi trường thương mại điện tử: hai người bán cạnh tranh giá.

Mô hình mua hàng (multinomial logit):
    U_i   = quality - beta * (giá_i / giá_niêm_yết)         (i = A, B)
    U_0   = tiện ích "không mua" (phụ thuộc mức nhu cầu)
    P_i   = exp(U_i) / (exp(U_A) + exp(U_B) + exp(U_0))
    Doanh số_i  ~ N * P_i  (cộng nhiễu Gauss, xấp xỉ nhị thức)
    Lợi nhuận_i = Doanh số_i * (giá_i - giá_vốn)
"""
import math

from .config import Config

HOLD, DISCOUNT = 0, 1
LOW, HIGH = 0, 1
ACTION_NAMES = ["Giữ giá", "Giảm giá"]
DEMAND_NAMES = ["Nhu cầu thấp", "Nhu cầu cao"]


N_STATES = 8


def state_index(demand: int, own_last: int, opp_last: int) -> int:
    """Trạng thái của Agent A = (mức nhu cầu, hành động vòng trước của A, của B)."""
    return demand * 4 + own_last * 2 + opp_last


def state_label(s: int) -> str:
    demand, rest = divmod(s, 4)
    own, opp = divmod(rest, 2)
    return (f"{DEMAND_NAMES[demand]} | A trước: {ACTION_NAMES[own]}"
            f" | B trước: {ACTION_NAMES[opp]}")


class PricingEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_customers = {LOW: cfg.n_low, HIGH: cfg.n_high}
        self.prices = [cfg.base_price, cfg.base_price * (1.0 - cfg.discount)]
        self.margin = [p - cfg.cost for p in self.prices]
        self.outside = {LOW: cfg.outside_low, HIGH: cfg.outside_high}
        # Bảng xác suất / kỳ vọng / độ lệch chuẩn: [demand][a][b]
        self.prob_a = self._table()
        self.prob_b = self._table()
        self.mu_a = self._table()
        self.mu_b = self._table()
        self.sd_a = self._table()
        self.sd_b = self._table()
        for d in (LOW, HIGH):
            n = self.n_customers[d]
            for a in (HOLD, DISCOUNT):
                for b in (HOLD, DISCOUNT):
                    pa, pb = self.purchase_probs(d, a, b)
                    self.prob_a[d][a][b], self.prob_b[d][a][b] = pa, pb
                    self.mu_a[d][a][b], self.mu_b[d][a][b] = n * pa, n * pb
                    self.sd_a[d][a][b] = math.sqrt(n * pa * (1 - pa))
                    self.sd_b[d][a][b] = math.sqrt(n * pb * (1 - pb))

    @staticmethod
    def _table():
        return [[[0.0, 0.0], [0.0, 0.0]] for _ in range(2)]

    def purchase_probs(self, demand: int, act_a: int, act_b: int):
        c = self.cfg
        ua = c.quality - c.beta * self.prices[act_a] / c.base_price
        ub = c.quality - c.beta * self.prices[act_b] / c.base_price
        u0 = self.outside[demand]
        m = max(ua, ub, u0)
        ea, eb, e0 = math.exp(ua - m), math.exp(ub - m), math.exp(u0 - m)
        den = ea + eb + e0
        return ea / den, eb / den

    def sample_sales(self, d, a, b, z1, z2):
        n = self.n_customers[d]
        sa = int(max(0.0, self.mu_a[d][a][b] + self.sd_a[d][a][b] * z1) + 0.5)
        if sa > n:
            sa = n
        sb = int(max(0.0, self.mu_b[d][a][b] + self.sd_b[d][a][b] * z2) + 0.5)
        if sb > n - sa:
            sb = n - sa
        return sa, sb

    def expected_payoff(self, d, a, b):
        """Trả về (lợi nhuận A, lợi nhuận B, doanh số A, doanh số B) kỳ vọng."""
        sa, sb = self.mu_a[d][a][b], self.mu_b[d][a][b]
        return sa * self.margin[a], sb * self.margin[b], sa, sb
