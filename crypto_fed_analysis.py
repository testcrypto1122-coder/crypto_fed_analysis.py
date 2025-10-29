# crypto_fed_analysis.py
# Chạy: python crypto_fed_analysis.py
# Yêu cầu: pip install yfinance pandas matplotlib numpy seaborn tabulate

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

sns.set(style="darkgrid", rc={"figure.dpi":150})

# --------- CẤU HÌNH ---------
TICKERS = ["BTC-USD", "ETH-USD"]
START = "2015-01-01"
END = datetime.today().strftime("%Y-%m-%d")
OUTDIR = "crypto_fed_output"
os.makedirs(OUTDIR, exist_ok=True)

# ---- DANH SÁCH NGÀY SỰ KIỆN (FED RATE CUT / ANNOUNCE) ----
# Thay bằng các ngày bạn muốn phân tích (YYYY-MM-DD).
# Ví dụ giả định: các ngày FED cắt giảm lãi trong lịch sử
# Bạn có thể thêm ngày thực tế (ví dụ '2024-07-31', '2020-03-15', ...)
FED_DATES = [
    # ví dụ mẫu - bạn hãy chỉnh hoặc thêm ngày thực tế
    "2020-03-15",
    "2020-03-03",
    "2023-06-15"
]

# RỔ WINDOW (số ngày trước/sau mỗi event để vẽ)
WINDOW_PRE = 30   # số ngày trước event
WINDOW_POST = 180 # số ngày sau event

# --------- TẢI DỮ LIỆU ---------
def download_data(tickers, start, end):
    print(f"Downloading {tickers} from {start} to {end} ...")
    data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
    # Nếu chỉ 1 ticker yfinance trả Series; chuyển thành DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all")
    return data

df = download_data(TICKERS, START, END)

# --------- BIỂU ĐỒ 1: PRICE (NORMALIZED) ---------
norm = df / df.iloc[0]  # normalized to 1 at start
plt.figure(figsize=(10,5))
for col in norm.columns:
    plt.plot(norm.index, norm[col], label=col)
plt.title("Crypto price (normalized) — BTC vs ETH")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Normalized price (index)")
plt.tight_layout()
fn1 = os.path.join(OUTDIR, "crypto_normalized.png")
plt.savefig(fn1)
plt.close()
print(f"Saved: {fn1}")

# --------- PHÂN TÍCH VÒNG NGÀY XUNG QUANH SỰ KIỆN ---------
def compute_event_returns(series, event_date, pre_days=30, post_days=180):
    ev = pd.to_datetime(event_date)
    start = ev - pd.Timedelta(days=pre_days)
    end = ev + pd.Timedelta(days=post_days)
    window = series.loc[(series.index >= start) & (series.index <= end)].copy()
    if window.empty:
        return None
    # convert to returns relative to event day close (or nearest previous)
    # find event day price (closest on/after event)
    if ev in series.index:
        base = series.loc[ev]
    else:
        # choose last available price on/before event
        earlier = series.index[series.index <= ev]
        if len(earlier)==0:
            return None
        base = series.loc[earlier[-1]]
    rel = window / base - 1.0
    rel.index = (rel.index - ev).days  # index = days from event (can be negative)
    return rel

# Build combined plot: for each event, plot BTC & ETH returns vs days-from-event
plt.figure(figsize=(12,8))
colors = ["C0","C1"]
for i, ev in enumerate(FED_DATES):
    # BTC
    btc_rel = compute_event_returns(df["BTC-USD"], ev, WINDOW_PRE, WINDOW_POST)
    eth_rel = compute_event_returns(df["ETH-USD"], ev, WINDOW_PRE, WINDOW_POST)
    if btc_rel is None or eth_rel is None:
        print(f"Warning: no data around {ev}")
        continue
    # smooth (optional)
    days = btc_rel.index
    plt.plot(days, btc_rel.values, label=f"BTC rel ({ev})", alpha=0.9)
    plt.plot(days, eth_rel.values, label=f"ETH rel ({ev})", alpha=0.6, linestyle="--")
plt.axvline(0, color="k", linewidth=0.8, linestyle=":")
plt.xlabel("Days since event")
plt.ylabel("Return vs event day (decimal, e.g. 0.2 = +20%)")
plt.title("Crypto returns around FED event dates")
plt.legend()
plt.tight_layout()
fn2 = os.path.join(OUTDIR, "crypto_event_returns.png")
plt.savefig(fn2)
plt.close()
print(f"Saved: {fn2}")

# --------- BẢNG TÓM TẮT LỢI SUẤT SAU 1M/3M/6M ---------
summary_rows = []
periods = {"1M":30, "3M":90, "6M":180}
for ev in FED_DATES:
    ev_dt = pd.to_datetime(ev)
    for ticker in TICKERS:
        s = df[ticker]
        # find base price at event (or previous available)
        if ev_dt in s.index:
            base = s.loc[ev_dt]
        else:
            earlier = s.index[s.index <= ev_dt]
            if len(earlier)==0:
                base = np.nan
            else:
                base = s.loc[earlier[-1]]
        row = {"event": ev, "ticker": ticker, "base_price": base}
        for pname, days in periods.items():
            target_date = ev_dt + pd.Timedelta(days=days)
            later = s.index[s.index <= target_date]
            if len(later)==0:
                ret = np.nan
            else:
                # pick the last available on/before target_date
                price_later = s.loc[later[-1]]
                ret = price_later / base - 1 if base and not np.isnan(base) else np.nan
            row[pname] = ret
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
# Save CSV and pretty-print
csv_path = os.path.join(OUTDIR, "crypto_event_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"Saved summary CSV: {csv_path}")

# Pretty print table to console
print("\nSummary table (returns):")
print(tabulate(summary_df, headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

# End
print("\nHoàn tất. Mở thư mục:", OUTDIR)
