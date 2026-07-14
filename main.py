import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

st.set_page_config(
    page_title="Interactive Stock Analyzer",
    page_icon="📈",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=60 * 30)
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.dropna()
    return data


@st.cache_data(ttl=60 * 60)
def load_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        return stock.info
    except Exception:
        return {}


def calculate_indicators(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA60"] = df["Close"].rolling(window=60).mean()
    df["MA120"] = df["Close"].rolling(window=120).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    df["Daily Return"] = df["Close"].pct_change()
    df["Cumulative Return"] = (1 + df["Daily Return"]).cumprod() - 1

    return df


def format_large_number(value):
    if value is None or pd.isna(value):
        return "N/A"

    try:
        value = float(value)
    except Exception:
        return "N/A"

    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Stock Analyzer")

ticker = st.sidebar.text_input(
    "Ticker",
    value="AAPL",
    help="예: AAPL, MSFT, TSLA, NVDA, GOOGL, 005930.KS",
).upper().strip()

period_option = st.sidebar.selectbox(
    "Analysis Period",
    ["6개월", "1년", "2년", "5년", "직접 선택"],
    index=1,
)

today = date.today()

if period_option == "6개월":
    start_date = today - timedelta(days=182)
    end_date = today
elif period_option == "1년":
    start_date = today - timedelta(days=365)
    end_date = today
elif period_option == "2년":
    start_date = today - timedelta(days=365 * 2)
    end_date = today
elif period_option == "5년":
    start_date = today - timedelta(days=365 * 5)
    end_date = today
else:
    start_date = st.sidebar.date_input("Start Date", today - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", today)

show_ma20 = st.sidebar.checkbox("20일 이동평균", True)
show_ma60 = st.sidebar.checkbox("60일 이동평균", True)
show_ma120 = st.sidebar.checkbox("120일 이동평균", False)

chart_type = st.sidebar.radio(
    "Chart Type",
    ["Candlestick", "Line"],
    horizontal=True,
)

# -----------------------------
# Main
# -----------------------------
st.title("Interactive Stock Data Analysis App")
st.caption("Yahoo Finance 데이터를 활용한 Plotly 기반 주식 분석 대시보드")

if not ticker:
    st.warning("티커를 입력해주세요.")
    st.stop()

with st.spinner(f"{ticker} 데이터를 불러오는 중..."):
    raw_df = load_stock_data(ticker, start_date, end_date)
    info = load_company_info(ticker)

if raw_df.empty:
    st.error("데이터를 불러오지 못했습니다. 티커 또는 날짜 범위를 확인해주세요.")
    st.stop()

df = calculate_indicators(raw_df)

company_name = info.get("longName") or info.get("shortName") or ticker
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")

st.subheader(f"{company_name} ({ticker})")
st.caption(f"{sector} · {industry}")

latest = df.iloc[-1]
previous = df.iloc[-2] if len(df) > 1 else latest

current_price = latest["Close"]
price_change = current_price - previous["Close"]
price_change_pct = price_change / previous["Close"] * 100 if previous["Close"] else 0

period_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
volatility = df["Daily Return"].std() * np.sqrt(252) * 100
highest_price = df["High"].max()
lowest_price = df["Low"].min()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "현재가",
    f"${current_price:,.2f}",
    f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
)

col2.metric("기간 수익률", f"{period_return:+.2f}%")
col3.metric("연환산 변동성", f"{volatility:.2f}%")
col4.metric("기간 최고가", f"${highest_price:,.2f}")
col5.metric("기간 최저가", f"${lowest_price:,.2f}")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)

info_col1.metric("시가총액", format_large_number(info.get("marketCap")))
info_col2.metric("PER", f"{info.get('trailingPE', 'N/A')}")
info_col3.metric("배당수익률", f"{(info.get('dividendYield') or 0) * 100:.2f}%")
info_col4.metric("52주 최고가", f"${info.get('fiftyTwoWeekHigh', 0):,.2f}" if info.get("fiftyTwoWeekHigh") else "N/A")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Price Chart", "Technical Indicators", "Returns", "Raw Data"]
)

# -----------------------------
# Price Chart
# -----------------------------
with tab1:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
        subplot_titles=("Price", "Volume"),
    )

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close",
                line=dict(color="#2563eb", width=2),
            ),
            row=1,
            col=1,
        )

    if show_ma20:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                mode="lines",
                name="MA20",
                line=dict(color="#f59e0b", width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_ma60:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA60"],
                mode="lines",
                name="MA60",
                line=dict(color="#7c3aed", width=1.5),
            ),
            row=1,
            col=1,
        )

    if show_ma120:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA120"],
                mode="lines",
                name="MA120",
                line=dict(color="#0891b2", width=1.5),
            ),
            row=1,
            col=1,
        )

    volume_colors = np.where(df["Close"] >= df["Open"], "#16a34a", "#dc2626")

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.45,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=760,
        template="plotly_white",
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Technical Indicators
# -----------------------------
with tab2:
    fig2 = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.25, 0.30],
        subplot_titles=("Close Price", "RSI", "MACD"),
    )

    fig2.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(color="#111827", width=2),
        ),
        row=1,
        col=1,
    )

    fig2.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            name="RSI",
            line=dict(color="#2563eb", width=2),
        ),
        row=2,
        col=1,
    )

    fig2.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=2, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=2, col=1)

    macd_colors = np.where(df["MACD_Hist"] >= 0, "#16a34a", "#dc2626")

    fig2.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_Hist"],
            name="MACD Histogram",
            marker_color=macd_colors,
            opacity=0.5,
        ),
        row=3,
        col=1,
    )

    fig2.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color="#7c3aed", width=2),
        ),
        row=3,
        col=1,
    )

    fig2.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Signal"],
            mode="lines",
            name="Signal",
            line=dict(color="#f59e0b", width=2),
        ),
        row=3,
        col=1,
    )

    fig2.update_layout(
        height=820,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig2.update_yaxes(title_text="Price", row=1, col=1)
    fig2.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig2.update_yaxes(title_text="MACD", row=3, col=1)

    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Returns
# -----------------------------
with tab3:
    left, right = st.columns(2)

    with left:
        fig3 = go.Figure()

        fig3.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Cumulative Return"] * 100,
                mode="lines",
                fill="tozeroy",
                name="Cumulative Return",
                line=dict(color="#2563eb", width=2),
            )
        )

        fig3.update_layout(
            title="Cumulative Return",
            yaxis_title="Return (%)",
            template="plotly_white",
            height=430,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
        )

        st.plotly_chart(fig3, use_container_width=True)

    with right:
        fig4 = go.Figure()

        fig4.add_trace(
            go.Histogram(
                x=df["Daily Return"].dropna() * 100,
                nbinsx=60,
                marker_color="#7c3aed",
                opacity=0.75,
                name="Daily Return",
            )
        )

        fig4.update_layout(
            title="Daily Return Distribution",
            xaxis_title="Daily Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
            height=430,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Return Statistics")

    stat_df = pd.DataFrame(
        {
            "Metric": [
                "Average Daily Return",
                "Median Daily Return",
                "Best Day",
                "Worst Day",
                "Annualized Volatility",
                "Cumulative Return",
            ],
            "Value": [
                f"{df['Daily Return'].mean() * 100:.3f}%",
                f"{df['Daily Return'].median() * 100:.3f}%",
                f"{df['Daily Return'].max() * 100:.3f}%",
                f"{df['Daily Return'].min() * 100:.3f}%",
                f"{volatility:.3f}%",
                f"{period_return:.3f}%",
            ],
        }
    )

    st.dataframe(stat_df, use_container_width=True, hide_index=True)

# -----------------------------
# Raw Data
# -----------------------------
with tab4:
    st.subheader("Downloaded Price Data")

    display_df = df.copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")

    st.dataframe(
        display_df.sort_index(ascending=False),
        use_container_width=True,
    )

    csv = display_df.to_csv().encode("utf-8")

    st.download_button(
        label="CSV 다운로드",
        data=csv,
        file_name=f"{ticker}_stock_data.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Data source: Yahoo Finance via yfinance. 이 앱은 투자 조언이 아닌 데이터 분석용 예시입니다."
)
