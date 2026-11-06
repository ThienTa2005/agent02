"""Pricing Arena - Dashboard Streamlit: Q-Learning vs Rule-Based cạnh tranh giá."""
import json
import time
from dataclasses import fields

import pandas as pd
import streamlit as st

from arena import plots
from arena.agents import RULES
from arena.config import Config
from arena.environment import (ACTION_NAMES, DEMAND_NAMES, HIGH, LOW, PricingEnv,
                               state_label)
from arena.experiments import (EXPERIMENTS, curves_to_dataframe, results_to_json,
                               runs_to_dataframe, seed_stability)

st.set_page_config(page_title="Pricing Arena", page_icon="🛒", layout="wide")

COLUMN_LABELS = {
    "experiment": "Thí nghiệm", "label": "Cấu hình", "seed": "Seed",
    "profit_a_mean": "LN A (TB)", "profit_b_mean": "LN B (TB)",
    "profit_gap_mean": "Chênh lệch A−B", "profit_a_std": "σ LN A", "profit_b_std": "σ LN B",
    "cv_a": "CV A", "cv_b": "CV B", "win_rate_a": "Tỷ lệ A thắng",
    "sales_a_mean": "Doanh số A", "sales_b_mean": "Doanh số B",
    "disc_rate_a": "Giảm giá A", "disc_rate_b": "Giảm giá B",
    "disc_rate_a_high": "Giảm giá A (cầu cao)", "disc_rate_a_low": "Giảm giá A (cầu thấp)",
    "policy": "Chính sách A (S0..S7)",
}

DEFAULTS = {
    "base_price": 100.0, "cost": 60.0, "discount_pct": 20,
    "alpha": 0.10, "gamma": 0.90, "epsilon": 0.50, "eps_decay": 0.999, "eps_min": 0.01,
    "train_ep": 5000, "eval_ep": 500, "steps": 30, "seed": 42, "rule": "demand",
    "quality": 8.0, "beta": 8.0, "out_high": 0.0, "out_low": 1.0,
    "n_high": 100, "n_low": 40, "p_high": 0.5, "persistence": 0.7,
}


# --------------------------------------------------------------------------
# Tiện ích giao diện
# --------------------------------------------------------------------------
def show_fig(fig, key):
    st.plotly_chart(fig, width="stretch", key=key)


def show_df(df, **kw):
    st.dataframe(df, width="stretch", hide_index=True, **kw)


def pretty(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns=COLUMN_LABELS).copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(3 if "CV" in c or "Tỷ lệ" in c or "Giảm giá" in c else 1)
    return out


def reset_defaults():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("results", {})
st.session_state.setdefault("log", [])
st.session_state.setdefault("log_counter", 0)


def sidebar_config() -> Config:
    sb = st.sidebar
    sb.title("⚙️ Tham số")
    sb.button("↺ Khôi phục mặc định", on_click=reset_defaults, width="stretch")
    with sb.expander("🏪 Thị trường", expanded=True):
        base = st.number_input("Giá bán niêm yết (nghìn đồng)", 1.0, 100000.0, step=5.0, key="base_price")
        cost = st.number_input("Giá vốn (nghìn đồng)", 0.0, 100000.0, step=5.0, key="cost")
        disc = st.slider("Mức giảm giá (%)", 0, 60, key="discount_pct")
        rule = st.selectbox("Luật của Agent B", list(RULES), format_func=RULES.get, key="rule")
    with sb.expander("🧠 Q-Learning (Agent A)", expanded=True):
        alpha = st.slider("Alpha (tốc độ học)", 0.001, 1.0, step=0.001, format="%.3f", key="alpha")
        gamma = st.slider("Gamma (chiết khấu)", 0.0, 0.99, step=0.01, key="gamma")
        eps = st.slider("Epsilon ban đầu", 0.0, 1.0, step=0.01, key="epsilon")
        decay = st.slider("Epsilon decay / episode", 0.990, 1.0, step=0.0001, format="%.4f", key="eps_decay")
        eps_min = st.slider("Epsilon tối thiểu", 0.0, 0.3, step=0.01, key="eps_min")
    with sb.expander("🏋️ Huấn luyện & đánh giá", expanded=True):
        n_train = st.number_input("Số episode huấn luyện", 100, 100000, step=500, key="train_ep")
        n_eval = st.number_input("Số episode đánh giá", 50, 10000, step=50, key="eval_ep")
        steps = st.number_input("Số vòng mỗi episode", 5, 365, step=5, key="steps")
        seed = st.number_input("Random seed", 0, 10**6, step=1, key="seed")
    with sb.expander("🔬 Tham số thị trường nâng cao"):
        quality = st.number_input("Độ hấp dẫn sản phẩm (a)", 0.0, 50.0, step=0.5, key="quality")
        beta = st.number_input("Độ nhạy giá (b)", 0.0, 50.0, step=0.5, key="beta")
        out_h = st.number_input("Tiện ích 'không mua' – cầu cao", -10.0, 10.0, step=0.5, key="out_high")
        out_l = st.number_input("Tiện ích 'không mua' – cầu thấp", -10.0, 10.0, step=0.5, key="out_low")
        n_h = st.number_input("Số khách / vòng – cầu cao", 1, 10000, step=10, key="n_high")
        n_l = st.number_input("Số khách / vòng – cầu thấp", 1, 10000, step=10, key="n_low")
        p_high = st.slider("Xác suất mức cầu cao", 0.0, 1.0, step=0.05, key="p_high")
        pers = st.slider("Độ bền mức cầu (giữ nguyên vòng sau)", 0.0, 1.0, step=0.05, key="persistence")
    cfg = Config(base_price=base, cost=cost, discount=disc / 100, quality=quality, beta=beta,
                 outside_high=out_h, outside_low=out_l, n_high=int(n_h), n_low=int(n_l),
                 p_high=p_high, persistence=pers, steps_per_episode=int(steps), rule=rule,
                 alpha=alpha, gamma=gamma, epsilon=eps, epsilon_decay=decay, epsilon_min=eps_min,
                 train_episodes=int(n_train), eval_episodes=int(n_eval), seed=int(seed))
    if cost >= base * (1 - disc / 100):
        sb.warning("Giá vốn ≥ giá sau giảm → mỗi đơn giảm giá bị lỗ.")
    return cfg


# --------------------------------------------------------------------------
# Chạy thí nghiệm
# --------------------------------------------------------------------------
def run_experiment(key: str, cfg: Config, bar=None, offset=0.0, span=1.0):
    spec = EXPERIMENTS[key]

    def progress(i, n, label):
        if bar is not None:
            bar.progress(min(1.0, offset + span * i / max(n, 1)), text=f"{spec.title}: {label}")

    runs = spec.func(cfg, progress)
    st.session_state.results[key] = runs
    st.session_state.log_counter += 1
    st.session_state.log.append({
        "id": st.session_state.log_counter, "exp": key, "time": time.strftime("%H:%M:%S"),
        "cfg": cfg.to_dict(), "runs": runs})
    return runs


def cfg_diff(a: dict, b: dict) -> dict:
    return {k: (a[k], b[k]) for k in a if a[k] != b.get(k)}


cfg = sidebar_config()

st.title("🛒 Pricing Arena")
st.caption("Hai người bán cạnh tranh giá: **Agent A (Q-Learning)** vs **Agent B (Rule-Based)**.")

if not st.session_state.results:  # chạy Baseline mặc định cho lần mở đầu tiên
    with st.spinner("Đang chạy Baseline lần đầu…"):
        run_experiment("baseline", cfg)

tabs = st.tabs(["📘 Mô hình", "🧪 Thí nghiệm", "📊 Biểu đồ", "🧠 Q-Table & Chính sách",
                "📋 Bảng kết quả", "⚖️ So sánh", "💾 Xuất dữ liệu"])

# ============================ TAB 1: MÔ HÌNH ================================
with tabs[0]:
    st.subheader("Môi trường thương mại điện tử")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "- **Hai người bán** bán cùng một sản phẩm, mỗi vòng chọn **Giữ giá** hoặc **Giảm giá**.\n"
            "- **Hai mức nhu cầu** (thấp/cao) chuyển đổi theo chuỗi Markov: số khách ghé thăm mỗi vòng "
            "khác nhau và khách ở nhu cầu thấp dễ bỏ đi hơn.\n"
            "- **Agent A** quan sát trạng thái *(mức nhu cầu, hành động vòng trước của A, của B)* → 8 trạng thái.\n"
            f"- **Agent B** chạy luật cố định: *{RULES[cfg.rule]}*.")
    with c2:
        st.markdown("**Công thức mua hàng, doanh số, lợi nhuận**")
        st.latex(r"U_i = a - b\,\frac{p_i}{p_0},\quad i\in\{A,B\}")
        st.latex(r"P_i=\frac{e^{U_i}}{e^{U_A}+e^{U_B}+e^{U_0(d)}}")
        st.latex(r"Q_i \approx N(d)\,P_i,\qquad \pi_i = Q_i\,(p_i-c)")
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]")
        st.caption("a: độ hấp dẫn, b: độ nhạy giá, p₀: giá niêm yết, c: giá vốn, N(d): số khách theo mức nhu cầu, "
                   "U₀(d): tiện ích 'không mua', r = lợi nhuận vòng đó của A. Doanh số thực tế = kỳ vọng + nhiễu Gauss.")

    st.subheader("Ma trận lợi nhuận kỳ vọng mỗi vòng (cấu hình hiện tại)")
    env = PricingEnv(cfg)
    cols = st.columns(2)
    for col, d in zip(cols, (HIGH, LOW)):
        rows = []
        for a in (0, 1):
            row = {"A \\ B": ACTION_NAMES[a]}
            for b in (0, 1):
                pa, pb, sa, sb = env.expected_payoff(d, a, b)
                row[f"B: {ACTION_NAMES[b]}"] = f"A {pa:,.0f} | B {pb:,.0f}"
            rows.append(row)
        with col:
            st.markdown(f"**{DEMAND_NAMES[d]}** (N = {env.n_customers[d]} khách/vòng)")
            show_df(pd.DataFrame(rows))
    prow = []
    for d in (HIGH, LOW):
        for a in (0, 1):
            for b in (0, 1):
                pa, pb = env.purchase_probs(d, a, b)
                prow.append({"Nhu cầu": DEMAND_NAMES[d], "A": ACTION_NAMES[a], "B": ACTION_NAMES[b],
                             "P(mua của A)": round(pa, 3), "P(mua của B)": round(pb, 3),
                             "P(không mua)": round(1 - pa - pb, 3)})
    with st.expander("Xác suất mua hàng chi tiết"):
        show_df(pd.DataFrame(prow))

# ============================ TAB 2: THÍ NGHIỆM =============================
with tabs[1]:
    st.subheader("Chạy / chạy lại thí nghiệm")
    st.caption("Mỗi lần chạy dùng **tham số đang đặt ở thanh bên**. Kết quả cũ vẫn được giữ trong tab *So sánh*.")
    if st.button("▶️ Chạy tất cả 6 thí nghiệm", type="primary"):
        bar = st.progress(0.0)
        keys = list(EXPERIMENTS)
        for i, k in enumerate(keys):
            run_experiment(k, cfg, bar, i / len(keys), 1 / len(keys))
        bar.progress(1.0, text="Hoàn tất")
        st.success("Đã chạy xong tất cả thí nghiệm.")
    st.divider()
    for k, spec in EXPERIMENTS.items():
        c1, c2, c3 = st.columns([4, 2, 1.4])
        c1.markdown(f"**{spec.title}**  \n{spec.description}")
        done = st.session_state.results.get(k)
        if done:
            last = [e for e in st.session_state.log if e["exp"] == k][-1]
            c2.markdown(f"✅ {len(done)} lần chạy · {last['time']}")
        else:
            c2.markdown("⏳ chưa chạy")
        if c3.button("Chạy lại" if done else "Chạy", key=f"run_{k}", width="stretch"):
            bar = st.progress(0.0)
            run_experiment(k, cfg, bar)
            bar.progress(1.0, text="Hoàn tất")
            st.rerun()

# ============================ TAB 3: BIỂU ĐỒ ================================
with tabs[2]:
    done_keys = [k for k in EXPERIMENTS if k in st.session_state.results]
    if not done_keys:
        st.info("Chưa có kết quả. Hãy chạy thí nghiệm ở tab *Thí nghiệm*.")
    else:
        k = st.selectbox("Thí nghiệm", done_keys, format_func=lambda x: EXPERIMENTS[x].title, key="chart_exp")
        runs = st.session_state.results[k]
        win = st.slider("Cửa sổ trung bình trượt (episode)", 10, 500, 100, 10, key="chart_win")
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plots.fig_profit(runs), f"p_{k}")
        with c2:
            show_fig(plots.fig_discount_rate(runs), f"d_{k}")
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plots.fig_learning_curve(runs, win), f"l_{k}")
        with c2:
            show_fig(plots.fig_gap_curve(runs, win), f"g_{k}")
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plots.fig_stability_box(runs), f"b_{k}")
        with c2:
            show_fig(plots.fig_stability_cv(runs), f"cv_{k}")
        c1, c2 = st.columns(2)
        with c1:
            show_fig(plots.fig_discount_by_demand(runs), f"dd_{k}")
        with c2:
            show_fig(plots.fig_train_discount(runs, win), f"td_{k}")
        if k == "learning":
            fig = plots.fig_checkpoints(runs[0])
            if fig is not None:
                show_fig(fig, "ckpt")
                show_df(pd.DataFrame(runs[0].checkpoints).rename(columns={
                    "episode": "Episode", "profit_a": "LN A", "profit_b": "LN B", "epsilon": "Epsilon"}).round(3))
        if k == "seeds":
            st.markdown("**Độ ổn định giữa 5 seed**")
            st.dataframe(seed_stability(runs).round(3), width="stretch")

# ============================ TAB 4: Q-TABLE ================================
with tabs[3]:
    done_keys = [k for k in EXPERIMENTS if k in st.session_state.results]
    if done_keys:
        c1, c2 = st.columns(2)
        k = c1.selectbox("Thí nghiệm", done_keys, format_func=lambda x: EXPERIMENTS[x].title, key="q_exp")
        runs = st.session_state.results[k]
        idx = c2.selectbox("Lần chạy", range(len(runs)), format_func=lambda i, r=runs: r[i].label, key="q_run")
        run = runs[idx]
        c1, c2 = st.columns([3, 2])
        with c1:
            show_fig(plots.fig_qtable(run), f"q_{k}_{idx}")
        with c2:
            st.markdown("**Chính sách Agent A (greedy)**")
            pol = pd.DataFrame({
                "Trạng thái": [f"S{s}" for s in range(8)],
                "Mô tả": [state_label(s) for s in range(8)],
                "Q(Giữ)": run.q_table[:, 0].round(0), "Q(Giảm)": run.q_table[:, 1].round(0),
                "Hành động": [ACTION_NAMES[a] for a in run.policy]})
            show_df(pol)
        st.markdown("**Chính sách Agent B (luật cố định)**")
        from arena.agents import RuleBasedAgent
        rb = RuleBasedAgent(run.config["rule"])
        brows = [{"Nhu cầu": DEMAND_NAMES[d], "A vòng trước": ACTION_NAMES[a],
                  "B chọn": ACTION_NAMES[rb.act(d, a)]} for d in (LOW, HIGH) for a in (0, 1)]
        show_df(pd.DataFrame(brows))
        st.caption(f"Epsilon cuối huấn luyện: {run.final_epsilon:.4f}")

# ============================ TAB 5: BẢNG KẾT QUẢ ===========================
with tabs[4]:
    done_keys = [k for k in EXPERIMENTS if k in st.session_state.results]
    if not done_keys:
        st.info("Chưa có kết quả.")
    for k in done_keys:
        st.markdown(f"#### {EXPERIMENTS[k].title}")
        df = runs_to_dataframe(k, st.session_state.results[k]).drop(columns=["experiment"])
        show_df(pretty(df))

# ============================ TAB 6: SO SÁNH ================================
with tabs[5]:
    log = st.session_state.log
    if len(log) < 2:
        st.info("Cần ít nhất 2 lần chạy. Hãy đổi tham số ở thanh bên rồi chạy lại một thí nghiệm.")
    else:
        def entry_name(e):
            return f"#{e['id']} · {EXPERIMENTS[e['exp']].title} · {e['time']}"
        by_id = {e["id"]: e for e in log}
        ids = [e["id"] for e in log]
        c1, c2 = st.columns(2)
        ia = c1.selectbox("Kết quả 1", ids, index=max(0, len(ids) - 2), format_func=lambda i: entry_name(by_id[i]))
        ib = c2.selectbox("Kết quả 2", ids, index=len(ids) - 1, format_func=lambda i: entry_name(by_id[i]))
        ea, eb = by_id[ia], by_id[ib]
        frames = []
        for e in (ea, eb):
            d = runs_to_dataframe(e["exp"], e["runs"])
            d.insert(0, "entry", entry_name(e))
            frames.append(d)
        both = pd.concat(frames, ignore_index=True)
        diff = cfg_diff(ea["cfg"], eb["cfg"])
        st.markdown("**Tham số khác nhau giữa hai lần chạy**")
        if diff:
            show_df(pd.DataFrame([{"Tham số": k, "Kết quả 1": v[0], "Kết quả 2": v[1]} for k, v in diff.items()]))
        else:
            st.caption("Cấu hình giống hệt nhau.")
        metric = st.selectbox("Chỉ số so sánh", [
            "profit_a_mean", "profit_b_mean", "profit_gap_mean", "win_rate_a", "disc_rate_a", "cv_a"],
            format_func=lambda m: COLUMN_LABELS[m])
        show_fig(plots.fig_compare(both, metric, f"So sánh: {COLUMN_LABELS[metric]}"), "cmp")
        show_df(pretty(both.drop(columns=["experiment"]).rename(columns={"entry": "Lần chạy"}), ).rename(
            columns={"entry": "Lần chạy"}))
        if ea["exp"] == eb["exp"] and len(ea["runs"]) == len(eb["runs"]):
            m = frames[0].merge(frames[1], on="label", suffixes=(" 1", " 2"))
            m["Δ (2 − 1)"] = m[f"{metric} 2"] - m[f"{metric} 1"]
            st.markdown("**Chênh lệch theo từng cấu hình**")
            show_df(m[["label", f"{metric} 1", f"{metric} 2", "Δ (2 − 1)"]].round(3).rename(columns={"label": "Cấu hình"}))

# ============================ TAB 7: XUẤT DỮ LIỆU ===========================
with tabs[6]:
    res = st.session_state.results
    if not res:
        st.info("Chưa có kết quả để xuất.")
    else:
        summary = pd.concat([runs_to_dataframe(k, r) for k, r in res.items()], ignore_index=True)
        curves = curves_to_dataframe(res)
        payload = {"config_current": cfg.to_dict(), "experiments": results_to_json(res)}
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Bảng kết quả (CSV)", summary.to_csv(index=False).encode("utf-8-sig"),
                           "ket_qua_thi_nghiem.csv", "text/csv", width="stretch")
        c2.download_button("⬇️ Đường học (CSV)", curves.to_csv(index=False).encode("utf-8-sig"),
                           "duong_hoc.csv", "text/csv", width="stretch")
        c3.download_button("⬇️ Toàn bộ kết quả (JSON)", json.dumps(payload, ensure_ascii=False, indent=2),
                           "ket_qua_day_du.json", "application/json", width="stretch")
        st.markdown("**Xuất từng thí nghiệm**")
        for k, runs in res.items():
            a, b = st.columns([3, 1])
            a.write(EXPERIMENTS[k].title)
            b.download_button("CSV", runs_to_dataframe(k, runs).to_csv(index=False).encode("utf-8-sig"),
                              f"{k}.csv", "text/csv", key=f"dl_{k}")
        with st.expander("Xem trước bảng tổng hợp"):
            show_df(pretty(summary))
