# 🛒 Pricing Arena — Q-Learning vs Rule-Based

Ứng dụng web mô phỏng **hai người bán cạnh tranh giá** trên sàn thương mại điện tử:

- **Agent A** — học tăng cường (**Q-Learning**, bảng Q 8 trạng thái × 2 hành động)
- **Agent B** — **Rule-Based** (luật cố định, không học)

Dashboard tương tác (Streamlit + Plotly), backend Python thuần (NumPy/Pandas).

> 🔗 **Demo công khai:** `https://<ten-app-cua-ban>.streamlit.app` _(điền URL sau khi deploy)_
> 📦 **Source code:** `https://github.com/<tai-khoan>/pricing-arena` _(điền sau khi push)_

---

## 1. Mô hình

| Thành phần         | Mô tả                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Hành động          | `Giữ giá` (giá niêm yết p₀) hoặc `Giảm giá` (p₀·(1 − mức giảm))                                    |
| Nhu cầu thị trường | 2 mức **thấp/cao**, chuyển đổi theo chuỗi Markov (độ bền `persistence`, xác suất cầu cao `p_high`) |
| Trạng thái của A   | (mức nhu cầu, hành động vòng trước của A, hành động vòng trước của B) → 8 trạng thái               |
| Phần thưởng        | Lợi nhuận của A trong vòng đó                                                                      |
| Episode            | `steps_per_episode` vòng (mặc định 30)                                                             |

**Xác suất mua hàng (multinomial logit)**

```
U_i = a − b · (p_i / p₀)            i ∈ {A, B}
P_i = exp(U_i) / (exp(U_A) + exp(U_B) + exp(U₀(d)))
Doanh số_i  ≈ N(d) · P_i            (cộng nhiễu Gauss quanh kỳ vọng, xấp xỉ nhị thức)
Lợi nhuận_i = Doanh số_i · (p_i − giá vốn)
```

Trong đó `a` = độ hấp dẫn sản phẩm, `b` = độ nhạy giá, `N(d)` = số khách ghé thăm theo mức nhu cầu,
`U₀(d)` = tiện ích "không mua" (khách ở nhu cầu thấp dễ bỏ đi hơn).

**Cập nhật Q-Learning**

```
Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
```

Khám phá ε-greedy; ε giảm dần: `ε ← max(ε_min, ε · decay)` sau mỗi episode.

**Các luật cho Agent B** (chọn ở thanh bên)

| Luật                  | Hành vi                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| `demand` _(mặc định)_ | Nhu cầu thấp → giảm giá; nhu cầu cao → giữ giá                                 |
| `match`               | Như trên, nhưng ở nhu cầu cao sẽ giảm giá theo nếu A vừa giảm giá ở vòng trước |
| `always_hold`         | Luôn giữ giá                                                                   |
| `always_discount`     | Luôn giảm giá                                                                  |

## 2. Quy trình huấn luyện & đánh giá

1. **Huấn luyện** Agent A `5.000` episode (có khám phá, có cập nhật Q).
2. **Đánh giá riêng** `500` episode: ε = 0 (chọn tham lam), **không cập nhật Q**, dùng seed môi trường khác với lúc huấn luyện.
3. Mọi bảng/biểu đồ "kết quả" lấy từ giai đoạn **đánh giá**; "đường học" lấy từ giai đoạn huấn luyện.

## 3. Sáu thí nghiệm

| #   | Thí nghiệm       | Nội dung                                                    |
| --- | ---------------- | ----------------------------------------------------------- |
| 1   | Baseline         | Cấu hình hiện tại ở thanh bên                               |
| 2   | Diễn biến học    | Đánh giá Agent A tại các mốc 50, 100, 250, … episode        |
| 3   | Thay đổi Alpha   | α ∈ {0.01, 0.05, 0.1, 0.3, 0.5}                             |
| 4   | Thay đổi Epsilon | ε₀ ∈ {0.05, 0.1, 0.3, 0.5, 0.9}                             |
| 5   | 5 random seed    | seed, seed+1, … seed+4 → độ ổn định giữa các lần huấn luyện |
| 6   | Mức giảm giá     | 10%, 15%, 20%, 25%, 30%                                     |

Dashboard cho phép **chạy lại từng thí nghiệm** (dùng tham số hiện tại), **so sánh** hai lần chạy bất kỳ (tab _So sánh_),
và **xuất CSV/JSON**. Các tham số tùy chỉnh: giá bán, giá vốn, mức giảm giá, Alpha, Gamma, Epsilon (+ decay, min),
số episode huấn luyện/đánh giá, số vòng/episode, seed, luật của B và các tham số thị trường nâng cao.

> ⚠️ Kết quả phụ thuộc seed: Q-Learning với hàm thưởng nhiễu có thể hội tụ về các chính sách khác nhau giữa các seed —
> đó chính là lý do có thí nghiệm 5 seed. Hãy báo cáo trung bình ± độ lệch chuẩn, đừng chỉ dựa vào một lần chạy.

## 4. Cấu trúc thư mục

```
pricing-arena/
├── app.py                 # Dashboard Streamlit
├── run_experiments.py     # Chạy toàn bộ thí nghiệm bằng CLI, xuất CSV/JSON
├── arena/
│   ├── config.py          # Config (dataclass)
│   ├── environment.py     # Môi trường: xác suất mua, doanh số, lợi nhuận
│   ├── agents.py          # QLearningAgent, RuleBasedAgent
│   ├── simulation.py      # Huấn luyện / đánh giá, thống kê
│   ├── experiments.py     # 6 thí nghiệm, bảng kết quả, xuất dữ liệu
│   └── plots.py           # Biểu đồ Plotly
├── tests/                 # pytest
├── requirements.txt
├── render.yaml            # Blueprint deploy Render
└── .streamlit/config.toml
```

## 5. Chạy trên máy

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # mở http://localhost:8501
```

Chạy bằng dòng lệnh (không cần giao diện):

```bash
python run_experiments.py --alpha 0.05 --discount 0.2 --out outputs
```

Kiểm thử:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 6. Đưa lên GitHub

```bash
git init
git add .
git commit -m "Pricing Arena: Q-Learning vs Rule-Based"
git branch -M main
git remote add origin https://github.com/ThienTa2005/agent02.git
git push -u origin main
```

> Dùng **repo riêng** cho dự án này để deploy **độc lập với Agent 01** (không import code chung, không dùng chung app/URL).

## 7. Deploy

### Cách A — Streamlit Community Cloud (đơn giản nhất)

1. Vào <https://share.streamlit.io> → đăng nhập bằng GitHub → **Create app**.
2. Chọn repo `pricing-arena`, branch `main`, **Main file path:** `app.py`.
3. (Tùy chọn) _Advanced settings_ → chọn Python 3.11 hoặc 3.12.
4. Bấm **Deploy**. Sau 1–3 phút bạn có URL dạng `https://<ten-app>.streamlit.app`.
5. Dán URL vào đầu README này.

### Cách B — Render

1. Vào <https://render.com> → **New +** → **Blueprint** → chọn repo (Render đọc file `render.yaml`).
   Hoặc **New + → Web Service** và nhập tay:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
2. Chọn gói Free, bấm **Create**. URL có dạng `https://pricing-arena.onrender.com`.
3. Gói Free của Render sẽ "ngủ" khi không có truy cập, lần mở đầu có thể mất ~30–60 giây.

## 8. Ghi chú kỹ thuật

- Một lần huấn luyện 5.000 episode × 30 vòng chạy dưới ~1 giây trên máy thường; chạy cả 6 thí nghiệm khoảng vài giây.
- Doanh số mỗi vòng dùng xấp xỉ chuẩn của phân phối nhị thức để mô phỏng nhanh (có kẹp trong [0, N]).
- Trạng thái của A gồm cả hành động vòng trước của chính A vì luật `match` của B phản ứng với hành động đó.
