"""Hai tác tử: Q-Learning (Agent A) và Rule-Based (Agent B)."""
import numpy as np

from .environment import DISCOUNT, HOLD, LOW


class QLearningAgent:
    N_STATES = 8
    N_ACTIONS = 2

    def __init__(self, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        self.alpha, self.gamma = alpha, gamma
        self.epsilon = epsilon
        self.epsilon_decay, self.epsilon_min = epsilon_decay, epsilon_min
        self.q = [[0.0] * self.N_ACTIONS for _ in range(self.N_STATES)]

    def greedy(self, s: int) -> int:
        q = self.q[s]
        return DISCOUNT if q[DISCOUNT] > q[HOLD] else HOLD  # hòa -> giữ giá

    def update(self, s, a, r, s2):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]"""
        q2 = self.q[s2]
        best = q2[0] if q2[0] > q2[1] else q2[1]
        self.q[s][a] += self.alpha * (r + self.gamma * best - self.q[s][a])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def q_table(self) -> np.ndarray:
        return np.array(self.q)

    @property
    def policy(self) -> list:
        return [self.greedy(s) for s in range(self.N_STATES)]


RULES = {
    "demand": "Theo nhu cầu: thấp → giảm giá, cao → giữ giá",
    "match": "Đối ứng: nhu cầu thấp → giảm giá; nhu cầu cao → giữ giá, "
             "trừ khi A vừa giảm giá ở vòng trước (khi đó giảm giá theo)",
    "always_hold": "Luôn giữ giá",
    "always_discount": "Luôn giảm giá",
}


class RuleBasedAgent:
    """Tác tử dựa trên luật cố định (không học). Xem `RULES` để biết các luật."""

    def __init__(self, rule: str = "demand"):
        if rule not in RULES:
            raise ValueError(f"Luật không hợp lệ: {rule}")
        self.rule = rule

    def act(self, demand: int, opp_last_action: int) -> int:
        r = self.rule
        if r == "demand":
            return DISCOUNT if demand == LOW else HOLD
        if r == "match":
            if demand == LOW:
                return DISCOUNT
            return DISCOUNT if opp_last_action == DISCOUNT else HOLD
        if r == "always_discount":
            return DISCOUNT
        return HOLD
