import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import numpy as np
import pytz
import vnstock as vs
import requests  # Thư viện mới để gọi API
from statsmodels.tsa.arima.model import ARIMA  # Thư viện mới cho mô hình dự đoán

# --- Cấu hình trang Streamlit và CSS
st.set_page_config(page_title="Phân tích cổ phiếu nâng cao", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1, h2, h3, h4, h5, h6 { color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# --- Giao diện chọn mã cổ phiếu và loại thị trường
st.title("📊 Phân tích cổ phiếu nâng cao")
stock_type = st.radio("Chọn thị trường", ["Việt Nam", "Quốc tế"])

if stock_type == "Việt Nam":
    stock_code = st.text_input("Nhập mã cổ phiếu Việt Nam (VD: VNM, FPT, SSI):", "FPT")
else:
    stock_code = st.text_input("Nhập mã quốc tế (VD: AAPL, NVDA, MSFT):", "AAPL")

# --- Chọn mốc thời gian bằng lịch
st.subheader("📅 Chọn khoảng thời gian")
today = datetime.today().date()
default_start = today - timedelta(days=180)

start_date, end_date = st.date_input(
    "Chọn từ ngày đến ngày:",
    value=(default_start, today),
    min_value=date(2000, 1, 1),
    max_value=today,
    format="DD/MM/YYYY"
)

# Kiểm tra logic thời gian
if start_date > end_date:
    st.error("❌ Ngày bắt đầu không thể sau ngày kết thúc.")
    st.stop()

# Chuyển sang định dạng datetime
start = pd.to_datetime(start_date)
end = pd.to_datetime(end_date)

st.markdown(f"🕒 Dữ liệu từ **{start.strftime('%d/%m/%Y')}** đến **{end.strftime('%d/%m/%Y')}**")

# --- Tải dữ liệu
with st.spinner("📈 Đang tải dữ liệu..."):
    if stock_type == "Việt Nam":
        df = vs.stock_historical_data(
            symbol=stock_code,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            resolution="1D"
        )
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        df = yf.download(stock_code, start=start, end=end)

# --- Các phần tiếp theo (chỉ báo kỹ thuật, biểu đồ, ARIMA...)
# Giữ nguyên cấu trúc cũ phía dưới
# Nếu bạn cần mình tích hợp phần phân tích kỹ thuật, MACD, RSI, Bollinger,... hoặc ARIMA dự đoán nữa thì hãy gửi mình đoạn đầy đủ hoặc nói rõ
