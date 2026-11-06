"""Biểu đồ Plotly cho dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .environment import ACTION_NAMES, state_label

COLOR_A, COLOR_B = "#2E86DE", "#EE5253"
LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
              legend=dict(orientation="h", y=-0.2))


def _roll(x, w):
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def fig_profit(runs):
    labels = [r.label for r in runs]
    fig = go.Figure()
    fig.add_bar(name="Agent A (Q-Learning)", x=labels, marker_color=COLOR_A,
                y=[r.summary["profit_a_mean"] for r in runs],
                error_y=dict(type="data", array=[r.summary["profit_a_std"] for r in runs]))
    fig.add_bar(name="Agent B (Rule-Based)", x=labels, marker_color=COLOR_B,
                y=[r.summary["profit_b_mean"] for r in runs],
                error_y=dict(type="data", array=[r.summary["profit_b_std"] for r in runs]))
    fig.update_layout(barmode="group", title="Lợi nhuận trung bình / episode (đánh giá, ± độ lệch chuẩn)",
                      yaxis_title="Lợi nhuận", **LAYOUT)
    return fig


def fig_discount_rate(runs):
    labels = [r.label for r in runs]
    fig = go.Figure()
    fig.add_bar(name="Agent A", x=labels, y=[r.summary["disc_rate_a"] * 100 for r in runs],
                marker_color=COLOR_A)
    fig.add_bar(name="Agent B", x=labels, y=[r.summary["disc_rate_b"] * 100 for r in runs],
                marker_color=COLOR_B)
    fig.update_layout(barmode="group", title="Tỷ lệ giảm giá (% số vòng, giai đoạn đánh giá)",
                      yaxis_title="%", **LAYOUT)
    return fig


def fig_discount_by_demand(runs):
    labels, vals = [], {"A · cao": [], "A · thấp": [], "B · cao": [], "B · thấp": []}
    for r in runs:
        labels.append(r.label)
        s = r.summary
        vals["A · cao"].append(s["disc_rate_a_high"] * 100)
        vals["A · thấp"].append(s["disc_rate_a_low"] * 100)
        vals["B · cao"].append(s["disc_rate_b_high"] * 100)
        vals["B · thấp"].append(s["disc_rate_b_low"] * 100)
    colors = {"A · cao": COLOR_A, "A · thấp": "#7FB5EA", "B · cao": COLOR_B, "B · thấp": "#F59C9C"}
    fig = go.Figure()
    for k, v in vals.items():
        fig.add_bar(name=k, x=labels, y=v, marker_color=colors[k])
    fig.update_layout(barmode="group", title="Tỷ lệ giảm giá theo mức nhu cầu (%)", yaxis_title="%", **LAYOUT)
    return fig


def fig_learning_curve(runs, window=100):
    fig = go.Figure()
    if len(runs) == 1:
        r = runs[0]
        x = np.arange(1, len(r.train["profit_a"]) + 1)
        fig.add_scatter(x=x, y=_roll(r.train["profit_a"], window), name="Agent A (Q-Learning)",
                        line=dict(color=COLOR_A))
        fig.add_scatter(x=x, y=_roll(r.train["profit_b"], window), name="Agent B (Rule-Based)",
                        line=dict(color=COLOR_B))
    else:
        for r in runs:
            x = np.arange(1, len(r.train["profit_a"]) + 1)
            fig.add_scatter(x=x, y=_roll(r.train["profit_a"], window), name=f"A · {r.label}")
        x = np.arange(1, len(runs[0].train["profit_b"]) + 1)
        b_mean = np.mean([r.train["profit_b"] for r in runs], axis=0)
        fig.add_scatter(x=x, y=_roll(b_mean, window), name="B (trung bình)",
                        line=dict(color="black", dash="dash"))
    fig.update_layout(title=f"Đường học: lợi nhuận huấn luyện (trung bình trượt {window} episode)",
                      xaxis_title="Episode", yaxis_title="Lợi nhuận", **LAYOUT)
    return fig


def fig_gap_curve(runs, window=100):
    fig = go.Figure()
    for r in runs:
        gap = r.train["profit_a"] - r.train["profit_b"]
        fig.add_scatter(x=np.arange(1, len(gap) + 1), y=_roll(gap, window), name=r.label)
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(title="Chênh lệch lợi nhuận A − B trong huấn luyện",
                      xaxis_title="Episode", yaxis_title="A − B", **LAYOUT)
    return fig


def fig_train_discount(runs, window=100):
    fig = go.Figure()
    for r in runs:
        x = np.arange(1, len(r.train["disc_rate_a"]) + 1)
        fig.add_scatter(x=x, y=_roll(r.train["disc_rate_a"], window) * 100, name=f"A · {r.label}")
    r0 = runs[0]
    fig.add_scatter(x=np.arange(1, len(r0.train["disc_rate_b"]) + 1),
                    y=_roll(r0.train["disc_rate_b"], window) * 100, name="B",
                    line=dict(color="black", dash="dash"))
    fig.update_layout(title="Tỷ lệ giảm giá trong huấn luyện (%)", xaxis_title="Episode",
                      yaxis_title="%", **LAYOUT)
    return fig


def fig_checkpoints(run):
    ck = run.checkpoints
    if not ck:
        return None
    fig = go.Figure()
    x = [c["episode"] for c in ck]
    fig.add_scatter(x=x, y=[c["profit_a"] for c in ck], name="Agent A (đánh giá greedy)",
                    mode="lines+markers", line=dict(color=COLOR_A))
    fig.add_scatter(x=x, y=[c["profit_b"] for c in ck], name="Agent B",
                    mode="lines+markers", line=dict(color=COLOR_B))
    fig.update_layout(title="Diễn biến học: lợi nhuận đánh giá tại các mốc episode",
                      xaxis_title="Episode huấn luyện", yaxis_title="Lợi nhuận", **LAYOUT)
    return fig


def fig_stability_box(runs):
    fig = go.Figure()
    for r in runs:
        fig.add_box(y=r.eval["profit_a"], name=f"A · {r.label}", marker_color=COLOR_A, boxmean=True)
        fig.add_box(y=r.eval["profit_b"], name=f"B · {r.label}", marker_color=COLOR_B, boxmean=True)
    fig.update_layout(title="Độ ổn định: phân phối lợi nhuận từng episode đánh giá",
                      yaxis_title="Lợi nhuận", showlegend=False, **LAYOUT)
    return fig


def fig_stability_cv(runs):
    labels = [r.label for r in runs]
    fig = go.Figure()
    fig.add_bar(name="A", x=labels, y=[r.summary["cv_a"] for r in runs], marker_color=COLOR_A)
    fig.add_bar(name="B", x=labels, y=[r.summary["cv_b"] for r in runs], marker_color=COLOR_B)
    fig.update_layout(barmode="group", title="Hệ số biến thiên CV = σ / μ (càng thấp càng ổn định)",
                      yaxis_title="CV", **LAYOUT)
    return fig


def fig_qtable(run):
    q = run.q_table
    fig = go.Figure(go.Heatmap(
        z=q, x=ACTION_NAMES, y=[f"S{s}: {state_label(s)}" for s in range(q.shape[0])],
        colorscale="Blues", text=np.round(q, 0), texttemplate="%{text}", showscale=True))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=f"Q-Table — {run.label}", height=420, **LAYOUT)
    return fig


def fig_compare(df: pd.DataFrame, metric: str, title: str):
    fig = go.Figure()
    for entry, g in df.groupby("entry", sort=False):
        fig.add_bar(name=entry, x=g["label"], y=g[metric])
    fig.update_layout(barmode="group", title=title, **LAYOUT)
    return fig
