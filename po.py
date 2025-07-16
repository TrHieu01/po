import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import vnstock as vs
import requests
from statsmodels.tsa.arima.model import ARIMA
import logging
import os

# --- Cấu hình logging ---
logging.basicConfig(filename='stock_dashboard.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Cấu hình Streamlit ---
st.set_page_config(layout="wide", page_title="Bảng Điều Khiển Phân Tích Cổ Phiếu")

# --- CSS tùy chỉnh ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #333;
    }
    .main .block-container {
        padding: 2rem; max-width: 1400px; margin: 0 auto;
        background: rgba(255, 255, 255, 0.95); border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    .css-1cpxqw2.e16z0g0x1 {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white;
        padding: 30px; text-align: center; border-radius: 20px 20px 0 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background: #f8f9fa; border-bottom: 2px solid #dee2e6; border-radius: 10px 10px 0 0; overflow: hidden; margin: 0 30px; }
    .stTabs [data-baseweb="tab-list"] button { padding: 20px; text-align: center; cursor: pointer; transition: all 0.3s; font-weight: 600; border-bottom: 3px solid transparent; color: #495057; min-width: 120px; }
    .stTabs [data-baseweb="tab-list"] button:hover { background: #e9ecef; }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { background: white; border-bottom-color: #667eea; color: #667eea; }
    .stButton>button { padding: 15px 25px; border: none; border-radius: 10px; font-size: 16px; cursor: pointer; transition: all 0.3s; font-weight: 600; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; width: 100%; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
    .stTextInput>div>div>input, .stSelectbox>div>div>select { padding: 15px; border: 2px solid #ddd; border-radius: 10px; font-size: 16px; width: 100%; }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus { border-color: #667eea; box-shadow: 0 0 10px rgba(102, 126, 234, 0.3); }
    .stRadio > label { font-weight: 600; margin-bottom: 10px; }
    .stRadio > div { display: flex; flex-wrap: wrap; gap: 10px; margin-left: 15px; }
    .stRadio > div > label { padding: 8px 16px; border: 2px solid #667eea; background: white; color: #667eea; border-radius: 5px; cursor: pointer; transition: all 0.3s; font-size: 14px; font-weight: 500; }
    .stRadio > div > label[data-baseweb="radio"] span:first-child { display: none; }
    .stRadio > div > label[data-baseweb="radio"][aria-checked="true"] { background: #667eea; color: white; box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3); }
    .st-emotion-cache-1r6dm1x { background: rgba(255, 255, 255, 0.95); border-radius: 20px; box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1); padding: 25px; margin-bottom: 20px; border: 1px solid #e9ecef; }
    .st-emotion-cache-1r6dm1x h3 { color: #2c3e50; margin-bottom: 15px; font-size: 1.4em; border-bottom: 1px solid #eee; padding-bottom: 10px; }
    .indicator-signal { font-size: 0.9em; padding: 5px 10px; border-radius: 5px; display: inline-block; margin-top: 5px; font-weight: 600; }
    .signal-buy { background: #d4edda; color: #155724; }
    .signal-sell { background: #f8d7da; color: #721c24; }
    .signal-hold { background: #fff3cd; color: #856404; }
    @media (max-width: 600px) {
        .stTabs [data-baseweb="tab-list"] { flex-direction: column; }
        .stButton>button { font-size: 14px; padding: 10px; }
        .stSelectbox, .stTextInput { font-size: 14px; }
    }
</style>
""", unsafe_allow_html=True)

# --- Tiêu đề ---
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; text-align: center; border-radius: 20px 20px 0 0;">
    <h1>📈 Bảng Điều Khiển Phân Tích Cổ Phiếu Nâng Cao</h1>
    <p>Phân tích và dự đoán cổ phiếu chuyên sâu</p>
</div>
""", unsafe_allow_html=True)

# --- Hàm hỗ trợ ---
def is_valid_symbol(symbol):
    """Kiểm tra tính hợp lệ của mã cổ phiếu."""
    try:
        vn_stock_symbols = vs.listing_companies()['ticker'].tolist()
        if symbol.upper() in vn_stock_symbols or symbol.upper().endswith('.VN'):
            return True
        yf.Ticker(symbol).info
        return True
    except Exception as e:
        logging.error(f"Invalid symbol {symbol}: {e}")
        return False

@st.cache_data(ttl=900)  # Cache 15 phút
def fetch_stock_data(symbol, time_range):
    """Lấy dữ liệu lịch sử và các chỉ số tài chính."""
    logging.info(f"Fetching data for {symbol} with time range {time_range}")
    if not is_valid_symbol(symbol):
        return None

    data = pd.DataFrame()
    is_vietnamese_stock = False
    vn_stock_symbols = vs.listing_companies()['ticker'].tolist()

    if symbol.upper() in vn_stock_symbols or symbol.upper().endswith('.VN'):
        is_vietnamese_stock = True
        end_date = datetime.now()
        days_map = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '6M': 180, '1Y': 365, '3Y': 3 * 365}
        start_date = end_date - timedelta(days=days_map.get(time_range, 365))
        try:
            data_vn = vs.stock_historical_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if not data_vn.empty:
                data = data_vn.rename(columns={'TradingDate': 'Date'}).set_index('Date')
                data.index = pd.to_datetime(data.index)
        except Exception as e:
            logging.error(f"Error fetching vnstock data for {symbol}: {e}")
            st.error(f"Lỗi khi tải dữ liệu từ vnstock cho {symbol}: {e}")
            return None
    else:
        period_map = {'1D': '1d', '1W': '1mo', '1M': '3mo', '3M': '6mo', '6M': '1y', '1Y': '2y', '3Y': '5y'}
        ticker = yf.Ticker(symbol)
        try:
            interval = '1m' if time_range == '1D' else '1d'
            data = ticker.history(period=period_map.get(time_range, '1y'), interval=interval)
        except Exception as e:
            logging.error(f"Error fetching yfinance data for {symbol}: {e}")
            st.error(f"Lỗi khi tải dữ liệu từ yfinance cho {symbol}: {e}")
            return None

    if data.empty:
        logging.error(f"No data retrieved for {symbol}")
        st.error(f"Không thể tải dữ liệu cho mã cổ phiếu: {symbol}")
        return None

    data.columns = [col.replace(' ', '') for col in data.columns]
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.index = pd.to_datetime(data.index.date)

    current_price = data['Close'].iloc[-1]
    previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
    change = current_price - previous_close
    change_percent = (change / previous_close) * 100 if previous_close != 0 else 0

    market_cap, pe_ratio, high_52w, low_52w, beta = (None,) * 5
    try:
        if is_vietnamese_stock:
            overview = vs.company_overview(symbol)
            market_cap = overview['marketCap'].iloc[0] * 1_000_000
            pe_ratio = overview['priceToEarningsRatio'].iloc[0]
            high_52w = overview['weekHigh52'].iloc[0]
            low_52w = overview['weekLow52'].iloc[0]
            beta = overview['beta'].iloc[0]
        else:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap')
            pe_ratio = info.get('trailingPE')
            high_52w = info.get('fiftyTwoWeekHigh')
            low_52w = info.get('fiftyTwoWeekLow')
            beta = info.get('beta')
    except Exception as e:
        logging.warning(f"Could not fetch financial details for {symbol}: {e}")

    return {
        'symbol': symbol, 'is_vietnamese': is_vietnamese_stock,
        'currentPrice': current_price, 'change': change, 'changePercent': change_percent,
        'volume': data['Volume'].iloc[-1],
        'marketCap': market_cap, 'peRatio': pe_ratio, 'high52w': high_52w, 'low52w': low_52w, 'beta': beta,
        'df': data
    }

def update_stock_info(stock_data):
    """Hiển thị thông tin cổ phiếu."""
    st.markdown("<h3>Thông tin cổ phiếu</h3>", unsafe_allow_html=True)
    if not stock_data:
        st.info("Nhấn \"Phân tích\" để xem thông tin cổ phiếu.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<h4>{stock_data['symbol']}</h4>", unsafe_allow_html=True)
        st.write(f"**Giá hiện tại:** ${stock_data['currentPrice']:.2f}")
        change_color = "green" if stock_data['change'] >= 0 else "red"
        change_symbol = "+" if stock_data['change'] >= 0 else ""
        st.markdown(f"**Thay đổi:** <span style='color:{change_color}'>{change_symbol}{stock_data['change']:.2f} ({stock_data['changePercent']:.2f}%)</span>", unsafe_allow_html=True)

    with col2:
        st.write(f"**Khối lượng:** {stock_data['volume']:,}")
        market_cap_str = f"${stock_data['marketCap'] / 1e9:.2f}B" if stock_data.get('marketCap') else "N/A"
        pe_ratio_str = f"{stock_data['peRatio']:.2f}" if stock_data.get('peRatio') else "N/A"
        st.write(f"**Vốn hóa:** {market_cap_str}")
        st.write(f"**P/E Ratio:** {pe_ratio_str}")

    st.markdown("<h3>Các chỉ số chính</h3>", unsafe_allow_html=True)
    stats_cols = st.columns(4)
    stats = [
        ("Giá hiện tại", f"${stock_data['currentPrice']:.2f}"),
        ("Thay đổi (%)", f"{change_symbol}{stock_data['changePercent']:.2f}%"),
        ("Khối lượng", f"{stock_data['volume'] / 1e6:.2f}M"),
        ("Vốn hóa thị trường", f"${stock_data['marketCap'] / 1e9:.2f}B" if stock_data.get('marketCap') else "N/A"),
        ("P/E Ratio", f"{stock_data['peRatio']:.2f}" if stock_data.get('peRatio') else "N/A"),
        ("Cao nhất 52W", f"${stock_data['high52w']:.2f}" if stock_data.get('high52w') else "N/A"),
        ("Thấp nhất 52W", f"${stock_data['low52w']:.2f}" if stock_data.get('low52w') else "N/A"),
        ("Beta", f"{stock_data['beta']:.2f}" if stock_data.get('beta') else "N/A")
    ]
    for i, (label, value) in enumerate(stats):
        with stats_cols[i % 4]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #dee2e6; height: 100%;">
                <div style="font-size: 1.5em; font-weight: bold; color: #2c3e50;">{value}</div>
                <div style="color: #6c757d; margin-top: 5px; font-size: 0.9em;">{label}</div>
            </div>
            """, unsafe_allow_html=True)

def update_charts(stock_data):
    """Hiển thị các biểu đồ giá, khối lượng, nến, RSI, MACD, Bollinger Bands."""
    st.markdown("<h3>Biểu đồ giá với các chỉ báo</h3>", unsafe_allow_html=True)
    if stock_data is None or stock_data['df'].empty:
        st.info("Không có dữ liệu biểu đồ để hiển thị.")
        return

    df = stock_data['df'].copy()
    dates = df.index

    # Tùy chọn bật/tắt chỉ báo
    st.markdown("**Tùy chọn hiển thị chỉ báo**")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_ma5 = st.checkbox("MA5", value=True)
    with col2:
        show_ma20 = st.checkbox("MA20", value=True)
    with col3:
        show_vwap = st.checkbox("VWAP", value=False)

    # Tính toán chỉ báo
    if show_ma5:
        df['MA5'] = ta.sma(df['Close'], length=5)
    if show_ma20:
        df['MA20'] = ta.sma(df['Close'], length=20)
    if show_vwap:
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

    # Biểu đồ giá
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=dates, y=df['Close'], mode='lines', name='Giá đóng cửa', line=dict(color='#2c3e50')))
    if show_ma5:
        fig_price.add_trace(go.Scatter(x=dates, y=df['MA5'], mode='lines', name='MA 5', line=dict(color='#28a745', dash='dash')))
    if show_ma20:
        fig_price.add_trace(go.Scatter(x=dates, y=df['MA20'], mode='lines', name='MA 20', line=dict(color='#ff9f40', dash='dash')))
    if show_vwap:
        fig_price.add_trace(go.Scatter(x=dates, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='#ff4500')))
    fig_price.update_layout(title='Biểu đồ giá và đường trung bình', xaxis_title='Ngày', yaxis_title='Giá', xaxis_rangeslider_visible=False, height=400, template="plotly_white")
    st.plotly_chart(fig_price, use_container_width=True)

    # Biểu đồ khối lượng
    st.markdown("<h3>Biểu đồ khối lượng</h3>", unsafe_allow_html=True)
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=dates, y=df['Volume'], name='Khối lượng', marker_color='#764ba2'))
    fig_volume.update_layout(title='Biểu đồ khối lượng giao dịch', xaxis_title='Ngày', yaxis_title='Khối lượng', xaxis_rangeslider_visible=False, height=300, template="plotly_white")
    st.plotly_chart(fig_volume, use_container_width=True)

    # Biểu đồ nến
    st.markdown("<h3>Biểu đồ nến (Candlestick)</h3>", unsafe_allow_html=True)
    fig_candle = go.Figure(data=[go.Candlestick(x=dates, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], increasing_line_color='#28a745', decreasing_line_color='#dc3545')])
    fig_candle.update_layout(title='Biểu đồ nến', xaxis_title='Ngày', yaxis_title='Giá', xaxis_rangeslider_visible=False, height=400, template="plotly_white")
    st.plotly_chart(fig_candle, use_container_width=True)

    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    st.markdown("<h3>RSI (Relative Strength Index)</h3>", unsafe_allow_html=True)
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=dates, y=df['RSI'], mode='lines', name='RSI (14)', line=dict(color='#007bff')))
    fig_rsi.add_trace(go.Scatter(x=dates, y=[70] * len(dates), mode='lines', name='Quá mua (70)', line=dict(color='#dc3545', dash='dash')))
    fig_rsi.add_trace(go.Scatter(x=dates, y=[30] * len(dates), mode='lines', name='Quá bán (30)', line=dict(color='#28a745', dash='dash')))
    fig_rsi.update_layout(title='Biểu đồ RSI', yaxis_range=[0, 100], height=300, template="plotly_white")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        df = df.join(macd_df)
    st.markdown("<h3>MACD</h3>", unsafe_allow_html=True)
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=dates, y=df.get('MACD_12_26_9'), mode='lines', name='MACD Line', line=dict(color='#007bff')))
    fig_macd.add_trace(go.Scatter(x=dates, y=df.get('MACDs_12_26_9'), mode='lines', name='Signal Line', line=dict(color='#ffc107', dash='dash')))
    fig_macd.add_trace(go.Bar(x=dates, y=df.get('MACDh_12_26_9'), name='Histogram', marker_color=['#28a745' if val >= 0 else '#dc3545' for val in df.get('MACDh_12_26_9', [])]))
    fig_macd.update_layout(title='Biểu đồ MACD', height=300, template="plotly_white")
    st.plotly_chart(fig_macd, use_container_width=True)

    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    if bbands is not None and not bbands.empty:
        df = df.join(bbands)
    st.markdown("<h3>Bollinger Bands</h3>", unsafe_allow_html=True)
    fig_bbands = go.Figure()
    fig_bbands.add_trace(go.Scatter(x=dates, y=df['Close'], mode='lines', name='Giá đóng cửa', line=dict(color='#2c3e50')))
    fig_bbands.add_trace(go.Scatter(x=dates, y=df.get('BBU_20_2.0'), mode='lines', name='Upper Band', line=dict(color='#ff7f0e', dash='dash')))
    fig_bbands.add_trace(go.Scatter(x=dates, y=df.get('BBM_20_2.0'), mode='lines', name='Middle Band', line=dict(color='#1f77b4', dash='dash')))
    fig_bbands.add_trace(go.Scatter(x=dates, y=df.get('BBL_20_2.0'), mode='lines', name='Lower Band', line=dict(color='#2ca02c', dash='dash')))
    fig_bbands.update_layout(title='Biểu đồ Bollinger Bands', height=400, template="plotly_white")
    st.plotly_chart(fig_bbands, use_container_width=True)

    # Xuất dữ liệu
    if st.button("Tải dữ liệu CSV"):
        csv_file = f"{stock_data['symbol']}_data.csv"
        df.to_csv(csv_file)
        with open(csv_file, "rb") as file:
            st.download_button("Tải file CSV", data=file, file_name=csv_file)
        logging.info(f"Exported data for {stock_data['symbol']} to {csv_file}")

def update_technical_indicators(stock_data):
    """Hiển thị các chỉ báo kỹ thuật."""
    st.markdown("<h3>Các chỉ báo kỹ thuật</h3>", unsafe_allow_html=True)
    if stock_data is None or stock_data['df'].empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    df = stock_data['df'].copy()
    prices = df['Close']

    def create_indicator_card(title, value, signal, signal_class):
        return f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); margin-bottom: 10px; height: 100%;">
            <div style="font-weight: bold; color: #2c3e50;">{title}</div>
            <div style="font-size: 1.2em; margin: 5px 0; color: #34495e;">{value}</div>
            <div class="indicator-signal {signal_class}">{signal}</div>
        </div>
        """

    # Moving Averages
    st.markdown("<h4>Moving Averages (MA)</h4>", unsafe_allow_html=True)
    ma_periods = [5, 10, 20, 50, 100, 200]
    ma_cols = st.columns(3)
    for i, period in enumerate(ma_periods):
        ma_values = ta.sma(prices, length=period)
        if ma_values is not None and not ma_values.empty:
            latest_ma = ma_values.iloc[-1]
            signal = "Giữ"
            signal_class = "signal-hold"
            if not np.isnan(latest_ma):
                if prices.iloc[-1] > latest_ma: signal, signal_class = "Tín hiệu: Mua", "signal-buy"
                elif prices.iloc[-1] < latest_ma: signal, signal_class = "Tín hiệu: Bán", "signal-sell"
            ma_cols[i % 3].markdown(create_indicator_card(f"MA {period}", f"{latest_ma:.2f}", signal, signal_class), unsafe_allow_html=True)

    # Exponential Moving Averages
    st.markdown("<h4>Exponential Moving Averages (EMA)</h4>", unsafe_allow_html=True)
    ema_periods = [12, 26, 50, 100]
    ema_cols = st.columns(4)
    for i, period in enumerate(ema_periods):
        ema_values = ta.ema(prices, length=period)
        if ema_values is not None and not ema_values.empty:
            latest_ema = ema_values.iloc[-1]
            signal, signal_class = "Giữ", "signal-hold"
            if not np.isnan(latest_ema):
                if prices.iloc[-1] > latest_ema: signal, signal_class = "Tín hiệu: Mua", "signal-buy"
                elif prices.iloc[-1] < latest_ema: signal, signal_class = "Tín hiệu: Bán", "signal-sell"
            ema_cols[i % 4].markdown(create_indicator_card(f"EMA {period}", f"{latest_ema:.2f}", signal, signal_class), unsafe_allow_html=True)

    # Other Indicators
    st.markdown("<h4>Các chỉ báo khác</h4>", unsafe_allow_html=True)
    other_cols = st.columns(3)
    rsi_series = ta.rsi(df['Close'], length=14)
    if rsi_series is not None and not rsi_series.empty:
        latest_rsi = rsi_series.iloc[-1]
        rsi_signal, rsi_signal_class = "Trung lập", "signal-hold"
        if not np.isnan(latest_rsi):
            if latest_rsi > 70: rsi_signal, rsi_signal_class = "Quá mua", "signal-sell"
            elif latest_rsi < 30: rsi_signal, rsi_signal_class = "Quá bán", "signal-buy"
        other_cols[0].markdown(create_indicator_card("RSI (14)", f"{latest_rsi:.2f}", rsi_signal, rsi_signal_class), unsafe_allow_html=True)

    macd_df = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        latest_macd = macd_df['MACD_12_26_9'].iloc[-1]
        latest_signal = macd_df['MACDs_12_26_9'].iloc[-1]
        macd_signal, macd_class = "Bán", "signal-sell"
        if not np.isnan(latest_macd) and not np.isnan(latest_signal) and latest_macd > latest_signal:
            macd_signal, macd_class = "Mua", "signal-buy"
        other_cols[1].markdown(create_indicator_card("MACD (12,26,9)", f"MACD: {latest_macd:.2f}", macd_signal, macd_class), unsafe_allow_html=True)

    adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx_df is not None and not adx_df.empty and 'ADX_14' in adx_df:
        latest_adx = adx_df['ADX_14'].iloc[-1]
        adx_signal, adx_class = "Xu hướng yếu", "signal-hold"
        if not np.isnan(latest_adx):
            if latest_adx > 25: adx_signal, adx_class = "Xu hướng mạnh", "signal-buy"
        other_cols[2].markdown(create_indicator_card("ADX (14)", f"{latest_adx:.2f}", adx_signal, adx_class), unsafe_allow_html=True)

def update_oscillators(stock_data):
    """Hiển thị các chỉ báo dao động."""
    st.markdown("<h3>Các chỉ báo dao động</h3>", unsafe_allow_html=True)
    if stock_data is None or stock_data['df'].empty:
        st.info("Không có dữ liệu để hiển thị.")
        return

    df = stock_data['df'].copy()
    dates = df.index

    stoch_df = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        df = df.join(stoch_df)
    st.markdown("<h4>Stochastic Oscillator</h4>", unsafe_allow_html=True)
    fig_stoch = go.Figure()
    fig_stoch.add_trace(go.Scatter(x=dates, y=df.get('STOCHk_14_3_3'), mode='lines', name='%K'))
    fig_stoch.add_trace(go.Scatter(x=dates, y=df.get('STOCHd_14_3_3'), mode='lines', name='%D'))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green")
    fig_stoch.update_layout(title='Stochastic Oscillator (14,3,3)', height=300, template="plotly_white", yaxis_range=[0,100])
    st.plotly_chart(fig_stoch, use_container_width=True)

    df['WPR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    st.markdown("<h4>Williams %R</h4>", unsafe_allow_html=True)
    fig_wpr = go.Figure()
    fig_wpr.add_trace(go.Scatter(x=dates, y=df.get('WPR'), mode='lines', name='W%R'))
    fig_wpr.add_hline(y=-20, line_dash="dash", line_color="red")
    fig_wpr.add_hline(y=-80, line_dash="dash", line_color="green")
    fig_wpr.update_layout(title='Williams %R (14)', height=300, template="plotly_white", yaxis_range=[-100,0])
    st.plotly_chart(fig_wpr, use_container_width=True)

def update_fundamentals(stock_data):
    """Hiển thị phân tích cơ bản."""
    st.markdown("<h3>Phân tích cơ bản</h3>", unsafe_allow_html=True)
    if not stock_data:
        st.info("Không có dữ liệu.")
        return

    symbol = stock_data['symbol']
    try:
        with st.spinner(f"Đang tải dữ liệu tài chính cho {symbol}..."):
            if stock_data['is_vietnamese']:
                st.subheader("Chỉ số tài chính (Theo Quý)")
                df_ratios = vs.financial_ratio(symbol, 'quarterly', True)
                st.dataframe(df_ratios)
                st.subheader("Báo cáo kết quả kinh doanh (Theo Quý)")
                df_income = vs.financial_flow(symbol, 'incomestatement', 'quarterly')
                st.dataframe(df_income)
            else:
                ticker = yf.Ticker(symbol)
                st.subheader("Báo cáo tài chính (Hàng năm)")
                st.dataframe(ticker.financials)
                st.subheader("Bảng cân đối kế toán (Hàng năm)")
                st.dataframe(ticker.balance_sheet)
    except Exception as e:
        logging.error(f"Error fetching fundamentals for {symbol}: {e}")
        st.error(f"Không thể tải dữ liệu tài chính: {e}")

def update_comparison(symbols_string, time_range):
    """So sánh hiệu suất các cổ phiếu."""
    st.markdown("<h3>So sánh hiệu suất các cổ phiếu</h3>", unsafe_allow_html=True)
    symbols = [s.strip().upper() for s in symbols_string.split(',') if s.strip()]
    if len(symbols) < 2:
        st.info("Nhập ít nhất 2 mã cổ phiếu (cách nhau bởi dấu phẩy) để so sánh.")
        return

    fig = go.Figure()
    all_data = {}
    with st.spinner("Đang tải dữ liệu so sánh..."):
        for symbol in symbols:
            data = fetch_stock_data(symbol, time_range)
            if data and not data['df'].empty:
                all_data[symbol] = data['df']
            else:
                st.warning(f"Không thể tải dữ liệu cho {symbol}")

    if not all_data:
        st.warning("Không tải được dữ liệu cho bất kỳ mã nào được nhập.")
        return

    for symbol, df in all_data.items():
        normalized_price = (df['Close'] / df['Close'].iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=normalized_price.index, y=normalized_price, mode='lines', name=symbol))
    fig.update_layout(title='So sánh hiệu suất giá (Chuẩn hóa theo %)', xaxis_title='Ngày', yaxis_title='Thay đổi (%)', template="plotly_white", height=500)
    st.plotly_chart(fig, use_container_width=True)

def update_predictions(stock_data):
    """Dự đoán giá bằng ARIMA."""
    st.markdown("<h3>Dự đoán giá (Mô hình ARIMA)</h3>", unsafe_allow_html=True)
    if stock_data is None or stock_data['df'].empty or len(stock_data['df']) < 50:
        st.info("Cần ít nhất 50 ngày dữ liệu để tạo dự đoán bằng mô hình ARIMA.")
        return

    df_close = stock_data['df']['Close']
    try:
        with st.spinner("Đang huấn luyện mô hình dự đoán..."):
            model = ARIMA(df_close, order=(5, 1, 0))
            model_fit = model.fit()
            forecast_result = model_fit.forecast(steps=30)
        pred1day = forecast_result.iloc[0]
        pred7day = forecast_result.iloc[6]
        pred30day = forecast_result.iloc[29]
        pred_cols = st.columns(3)
        predictions = [("1 ngày", pred1day), ("7 ngày", pred7day), ("30 ngày", pred30day)]
        for i, (label, value) in enumerate(predictions):
            with pred_cols[i]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4); height: 100%;">
                    <h4>Dự đoán {label}</h4>
                    <div style="font-size: 2.5em; font-weight: bold; margin: 10px 0;">${value:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<p style='font-size: 0.9em; color: #6c757d; margin-top: 15px;'><em>Lưu ý: Đây là dự đoán từ mô hình toán học, không phải lời khuyên tài chính.</em></p>", unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error in ARIMA prediction for {stock_data['symbol']}: {e}")
        st.error(f"Lỗi khi xây dựng mô hình dự đoán: {e}")

def update_news(symbol):
    """Hiển thị tin tức từ Finnhub."""
    st.markdown("<h3>Tin tức và Phân tích</h3>", unsafe_allow_html=True)
    try:
        API_KEY = st.secrets["FINNHUB_API_KEY"]
    except KeyError:
        st.error("Vui lòng cấu hình FINNHUB_API_KEY trong file secrets.toml")
        return

    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url = f'https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date}&to={end_date}&token={API_KEY}'
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        news_list = r.json()
        if not news_list:
            st.info(f"Không tìm thấy tin tức gần đây cho mã {symbol}.")
            return
        for news_item in news_list[:5]:
            news_date = datetime.fromtimestamp(news_item['datetime']).strftime('%d-%m-%Y')
            st.markdown(f"""
            <div style="background: white; border-radius: 15px; padding: 20px; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1); border: 1px solid #e9ecef; margin-bottom: 20px;">
                <div style="font-weight: bold; color: #2c3e50; margin-bottom: 10px; font-size: 1.1em;">
                    <a href="{news_item['url']}" target="_blank" style="text-decoration: none; color: inherit;">{news_item['headline']}</a>
                </div>
                <div style="color: #6c757d; font-size: 0.9em; line-height: 1.4;">{news_item['summary']}</div>
                <div style="color: #adb5bd; font-size: 0.8em; margin-top: 10px; text-align: right;">{news_date} - Nguồn: {news_item['source']}</div>
            </div>
            """, unsafe_allow_html=True)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching news for {symbol}: {e}")
        st.error(f"Lỗi kết nối đến API tin tức: {e}")
    except Exception as e:
        logging.error(f"Error processing news for {symbol}: {e}")
        st.error(f"Lỗi khi xử lý dữ liệu tin tức: {e}")

# --- Bố cục giao diện ---
vn_stock_symbols = vs.listing_companies()['ticker'].tolist()
popular_symbols = vn_stock_symbols + ['AAPL', 'NVDA', 'TSLA']
col1, col2, col3 = st.columns([3, 4, 1])
with col1:
    symbol_input = st.selectbox("Chọn hoặc nhập mã cổ phiếu (VD: FPT, VCB, NVDA)", popular_symbols, format_func=lambda x: x, key='symbol_input')
with col2:
    time_range_options = ['1D', '1W', '1M', '3M', '6M', '1Y', '3Y']
    if 'selected_time_range' not in st.session_state:
        st.session_state.selected_time_range = '1Y'
    selected_time_range = st.radio("Chọn phạm vi thời gian:", time_range_options, index=time_range_options.index(st.session_state.selected_time_range), horizontal=True, key='time_range_radio')
    st.session_state.selected_time_range = selected_time_range
with col3:
    st.write("")
    st.write("")
    analyze_button = st.button("Phân tích", key="analyze_button")

main_symbol = symbol_input.strip().upper()
if analyze_button:
    if not is_valid_symbol(main_symbol):
        st.error(f"Mã cổ phiếu {main_symbol} không hợp lệ. Vui lòng kiểm tra lại.")
    else:
        with st.spinner(f'Đang tải và phân tích {main_symbol}...'):
            st.session_state.stock_data = fetch_stock_data(main_symbol, st.session_state.selected_time_range)
            logging.info(f"Analyzed symbol {main_symbol} at {datetime.now()}")

# --- Tạo các tab ---
tab_list = ["Tổng quan", "Biểu đồ", "Chỉ báo kỹ thuật", "Oscillators", "Phân tích cơ bản", "So sánh", "Dự đoán", "Tin tức"]
tabs = st.tabs(tab_list)

with tabs[0]:
    try:
        update_stock_info(st.session_state.get('stock_data'))
    except Exception as e:
        logging.error(f"Error in stock info tab: {e}")
        st.error(f"Lỗi khi hiển thị thông tin cổ phiếu: {e}")
with tabs[1]:
    try:
        update_charts(st.session_state.get('stock_data'))
    except Exception as e:
        logging.error(f"Error in charts tab: {e}")
        st.error(f"Lỗi khi hiển thị biểu đồ: {e}")
with tabs[2]:
    try:
        update_technical_indicators(st.session_state.get('stock_data'))
    except Exception as e:
        logging.error(f"Error in technical indicators tab: {e}")
        st.error(f"Lỗi khi hiển thị chỉ báo kỹ thuật: {e}")
with tabs[3]:
    try:
        update_oscillators(st.session_state.get('stock_data'))
    except Exception as e:
        logging.error(f"Error in oscillators tab: {e}")
        st.error(f"Lỗi khi hiển thị chỉ báo dao động: {e}")
with tabs[4]:
    try:
        update_fundamentals(st.session_state.get('stock_data'))
    except Exception as e:
        logging.error(f"Error in fundamentals tab: {e}")
        st.error(f"Lỗi khi hiển thị phân tích cơ bản: {e}")
with tabs[5]:
    try:
        update_comparison(symbol_input, st.session_state.get('selected_time_range', '1Y'))
    except Exception as e:
        logging.error(f"Error in comparison tab: {e}")
        st.error(f"Lỗi khi hiển thị so sánh: {e}")
with tabs[6]:
    try:
        update_predictions(st.session_state.get('stock_data'))
    except Exception as e:
        logging.error(f"Error in predictions tab: {e}")
        st.error(f"Lỗi khi hiển thị dự đoán: {e}")
with tabs[7]:
    try:
        update_news(main_symbol)
    except Exception as e:
        logging.error(f"Error in news tab: {e}")
        st.error(f"Lỗi khi hiển thị tin tức: {e}")
