import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

st.set_page_config(page_title="Korea AI & Semiconductor Stock Lab", layout="wide")

STOCKS = {
    "메모리/HBM": {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
    },
    "반도체 장비": {
        "한미반도체": "042700.KS",
        "HPSP": "403870.KQ",
        "원익IPS": "240810.KQ",
    },
    "소부장/테스트/PCB": {
        "리노공업": "058470.KQ",
        "ISC": "095340.KQ",
        "솔브레인": "357780.KQ",
        "이수페타시스": "007660.KS",
        "DB하이텍": "000990.KS",
    },
    "AI 플랫폼/인터넷": {
        "NAVER": "035420.KS",
        "카카오": "035720.KS",
    },
}

ALL_STOCKS = {name: ticker for group in STOCKS.values() for name, ticker in group.items()}


@st.cache_data(ttl=1800)
def download_prices(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
    )
    return data


@st.cache_data(ttl=3600)
def get_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


def get_single_price(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(0):
            df = data[ticker].copy()
        else:
            return pd.DataFrame()
    else:
        df = data.copy()

    df = df.dropna()
    return df


def add_indicators(df):
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Return"] = df["Close"].pct_change()
    df["CumReturn"] = (1 + df["Return"]).cumprod() - 1
    return df


def money(v):
    if v is None or pd.isna(v):
        return "N/A"
    if abs(v) >= 1_0000_0000_0000:
        return f"{v / 1_0000_0000_0000:.1f}조"
    if abs(v) >= 1_0000_0000:
        return f"{v / 1_0000_0000:.1f}억"
    return f"{v:,.0f}"


st.sidebar.title("Korea AI/Semi Lab")

groups = st.sidebar.multiselect(
    "분석할 테마",
    list(STOCKS.keys()),
    default=["메모리/HBM", "반도체 장비", "AI 플랫폼/인터넷"],
)

selected_names = []
for g in groups:
    selected_names.extend(STOCKS[g].keys())

selected_names = st.sidebar.multiselect(
    "종목 선택",
    list(ALL_STOCKS.keys()),
    default=selected_names,
)

custom = st.sidebar.text_input(
    "직접 티커 추가",
    placeholder="예: 005930.KS, 042700.KS",
)

period = st.sidebar.selectbox("기간", ["3개월", "6개월", "1년", "2년", "5년"], index=2)
days = {"3개월": 90, "6개월": 180, "1년": 365, "2년": 730, "5년": 1825}[period]

end_date = date.today()
start_date = end_date - timedelta(days=days)

tickers = [ALL_STOCKS[n] for n in selected_names]
if custom.strip():
    tickers += [x.strip().upper() for x in custom.split(",") if x.strip()]

tickers = sorted(list(set(tickers)))

st.title("한국 AI · 반도체 대표주 분석 대시보드")
st.caption("Yahoo Finance 데이터 기반. 투자 조언이 아닌 데이터 분석용 예시 앱입니다.")

if not tickers:
    st.warning("분석할 종목을 하나 이상 선택해주세요.")
    st.stop()

with st.spinner("주가 데이터를 불러오는 중..."):
    raw = download_prices(tickers, start_date, end_date)

price_map = {}
for ticker in tickers:
    df = get_single_price(raw, ticker)
    if not df.empty and "Close" in df.columns:
        price_map[ticker] = add_indicators(df)

if not price_map:
    st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")
    st.stop()

name_by_ticker = {v: k for k, v in ALL_STOCKS.items()}

summary = []
close_table = pd.DataFrame()

for ticker, df in price_map.items():
    name = name_by_ticker.get(ticker, ticker)
    latest = df.iloc[-1]
    first = df.iloc[0]
    ret = latest["Close"] / first["Close"] - 1
    vol = df["Return"].std() * np.sqrt(252)
    mdd = (df["Close"] / df["Close"].cummax() - 1).min()

    summary.append(
        {
            "종목": name,
            "티커": ticker,
            "현재가": latest["Close"],
            "기간수익률": ret,
            "연환산변동성": vol,
            "최대낙폭": mdd,
            "거래대금": latest["Close"] * latest["Volume"],
            "RSI": latest["RSI"],
        }
    )
    close_table[name] = df["Close"]

summary_df = pd.DataFrame(summary).sort_values("기간수익률", ascending=False)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
best = summary_df.iloc[0]
worst = summary_df.iloc[-1]

kpi1.metric("분석 종목 수", f"{len(summary_df)}개")
kpi2.metric("최고 수익률", best["종목"], f"{best['기간수익률'] * 100:.2f}%")
kpi3.metric("최저 수익률", worst["종목"], f"{worst['기간수익률'] * 100:.2f}%")
kpi4.metric("평균 수익률", f"{summary_df['기간수익률'].mean() * 100:.2f}%")

tab1, tab2, tab3, tab4 = st.tabs(["종목 비교", "개별 분석", "리스크/상관관계", "데이터"])

with tab1:
    normalized = close_table / close_table.iloc[0] * 100

    fig = go.Figure()
    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=col,
                line=dict(width=2),
            )
        )

    fig.update_layout(
        title="상대 수익률 비교: 시작일 = 100",
        template="plotly_white",
        height=560,
        hovermode="x unified",
        yaxis_title="Indexed Price",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    rank = summary_df.copy()
    rank["현재가"] = rank["현재가"].map(lambda x: f"{x:,.0f}원")
    rank["기간수익률"] = rank["기간수익률"].map(lambda x: f"{x * 100:.2f}%")
    rank["연환산변동성"] = rank["연환산변동성"].map(lambda x: f"{x * 100:.2f}%")
    rank["최대낙폭"] = rank["최대낙폭"].map(lambda x: f"{x * 100:.2f}%")
    rank["거래대금"] = rank["거래대금"].map(money)
    rank["RSI"] = rank["RSI"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    st.dataframe(rank, use_container_width=True, hide_index=True)

with tab2:
    selected = st.selectbox("개별 분석 종목", list(price_map.keys()), format_func=lambda x: name_by_ticker.get(x, x))
    df = price_map[selected]
    info = get_info(selected)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"{df['Close'].iloc[-1]:,.0f}원")
    c2.metric("기간수익률", f"{(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:.2f}%")
    c3.metric("시가총액", money(info.get("marketCap")))
    c4.metric("PER", info.get("trailingPE", "N/A"))

    fig2 = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=("가격/이동평균", "거래량", "RSI"),
    )

    fig2.add_trace(
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
    for ma, color in [("MA20", "#f59e0b"), ("MA60", "#2563eb"), ("MA120", "#7c3aed")]:
        fig2.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=color, width=1.5)), row=1, col=1)

    colors = np.where(df["Close"] >= df["Open"], "#16a34a", "#dc2626")
    fig2.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Volume", opacity=0.45), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#111827", width=2)), row=3, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=3, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="#16a34a", row=3, col=1)

    fig2.update_layout(
        template="plotly_white",
        height=820,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    left, right = st.columns(2)

    with left:
        risk = summary_df.copy()
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=risk["연환산변동성"] * 100,
                y=risk["기간수익률"] * 100,
                mode="markers+text",
                text=risk["종목"],
                textposition="top center",
                marker=dict(size=16, color=risk["기간수익률"] * 100, colorscale="RdYlGn", showscale=True),
                customdata=risk[["티커", "최대낙폭"]],
                hovertemplate="<b>%{text}</b><br>변동성 %{x:.2f}%<br>수익률 %{y:.2f}%<br>최대낙폭 %{customdata[1]:.2%}<extra></extra>",
            )
        )
        fig3.update_layout(
            title="리스크-수익률 맵",
            xaxis_title="연환산 변동성 (%)",
            yaxis_title="기간 수익률 (%)",
            template="plotly_white",
            height=520,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with right:
        returns = close_table.pct_change().dropna()
        corr = returns.corr()

        fig4 = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig4.update_layout(title="일간 수익률 상관관계", template="plotly_white", height=520)
        st.plotly_chart(fig4, use_container_width=True)

with tab4:
    st.subheader("원본 및 계산 데이터")
    selected_data = st.selectbox("데이터 확인 종목", list(price_map.keys()), format_func=lambda x: name_by_ticker.get(x, x), key="data_select")
    data_view = price_map[selected_data].copy()
    data_view.index = data_view.index.strftime("%Y-%m-%d")
    st.dataframe(data_view.sort_index(ascending=False), use_container_width=True)

    st.download_button(
        "CSV 다운로드",
        data=data_view.to_csv().encode("utf-8-sig"),
        file_name=f"{selected_data}_korea_ai_semiconductor.csv",
        mime="text/csv",
    )

st.caption("Data: Yahoo Finance via yfinance. 종목 리스트는 예시이며, 투자 판단은 별도 검토가 필요합니다.")
