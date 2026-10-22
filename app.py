import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Nạp module tin tức Tầng 3
from src.backtest import prepare_walk_forward_backtest_data, run_backtest
from src.data_quality import assess_market_data
from src.forecast_tracking import summarize_forecast_history, update_forecast_history
from src.market_refresh import market_data_refresh_lock
from src.auto_refresh import read_refresh_status, run_refresh_if_due
from src.refresh_pipeline import refresh_market_data as run_market_refresh_pipeline

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:  # The dashboard still works; install requirements for automatic browser refresh.
    st_autorefresh = None
from src.atomic_files import atomic_output_path
from src.sjc_prices import quote_comparison, validate_sjc_quotes
from src.news_sentiment import (
    NewsFetchError,
    NewsTranslationError,
    get_latest_news_sentiment,
    translate_headline_to_vietnamese,
)

st.set_page_config(
    page_title="Gold Market Desk | AI Analytics",
    page_icon="◈",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'gold_data_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gold_model.pkl')
FORECAST_HISTORY_PATH = os.path.join(BASE_DIR, 'data', 'forecast_history.sqlite3')
SJC_DATA_PATH = os.path.join(BASE_DIR, 'data', 'sjc_prices.csv')


@st.cache_data(ttl=300)
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date').reset_index(drop=True)


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


@st.cache_data(ttl=900)
def load_news(refresh_key):
    return get_latest_news_sentiment()


@st.cache_data(ttl=86400, max_entries=256, show_spinner=False)
def translate_news_title(title):
    return translate_headline_to_vietnamese(title)


def refresh_market_data():
    return run_market_refresh_pipeline(BASE_DIR)


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(family="Inter, sans-serif", color="#B8C4D0", size=12),
        margin=dict(l=18, r=18, t=36, b=18),
        hoverlabel=dict(bgcolor="#18232D", bordercolor="#344451", font_color="#F4F7FA"),
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.10)", zerolinecolor="rgba(148, 163, 184, 0.15)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.10)", zerolinecolor="rgba(148, 163, 184, 0.15)")
    return fig


# Refresh the browser periodically while the dashboard is open. A separate
# worker/OS scheduler can refresh data even when no browser is connected.
if st_autorefresh is not None:
    st_autorefresh(interval=15 * 60 * 1000, key="gold_market_page_refresh")

try:
    AUTO_REFRESH_INTERVAL_MINUTES = max(
        1, int(os.getenv("GOLD_AUTO_REFRESH_MINUTES", "60"))
    )
except ValueError:
    AUTO_REFRESH_INTERVAL_MINUTES = 60

try:
    auto_refresh_status = run_refresh_if_due(
        BASE_DIR,
        refresh_market_data,
        interval_minutes=AUTO_REFRESH_INTERVAL_MINUTES,
    )
    if auto_refresh_status.get("changed"):
        st.cache_data.clear()
except (OSError, RuntimeError, TimeoutError, ValueError) as auto_refresh_error:
    auto_refresh_status = {
        "status": "error",
        "message": str(auto_refresh_error),
    }

df = load_data()
model_payload = load_model()

if df is None or df.empty:
    st.error("Chưa tìm thấy dữ liệu đã xử lý. Hãy tải dữ liệu thị trường để tạo bộ dữ liệu dashboard.")
    if st.button("Tải và chuẩn bị dữ liệu thị trường"):
        with st.spinner("Đang tải dữ liệu, tính chỉ báo và kiểm định lại mô hình..."):
            refresh_market_data()
        st.cache_data.clear()
        st.rerun()
    st.stop()
st.html("""
<style>
[data-testid="stHeader"] { background: rgba(9, 15, 21, 0.94); }
[data-testid="stSidebar"] {
    background: #0C141B;
    border-right: 1px solid #22303A;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #EEF3F7; letter-spacing: -0.02em; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: #B6C3CE; }
.block-container { max-width: 1440px; padding-top: 3rem; padding-bottom: 3.5rem; }
h1, h2, h3 { color: #F1F5F8; letter-spacing: -0.025em; }
h2 { font-size: 1.3rem !important; margin-top: 1.5rem !important; }
[data-testid="stCaptionContainer"] { color: #94A4B1; }
[data-testid="stMarkdownContainer"] p { line-height: 1.65; }
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #121D26 0%, #0F1921 100%);
    border: 1px solid #263743;
    border-radius: 14px;
    padding: 17px 19px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.12);
    transition: border-color 160ms ease, transform 160ms ease;
}
[data-testid="stMetric"]:hover { border-color: #52616B; transform: translateY(-1px); }
[data-testid="stMetricLabel"] {
    color: #AAB8C3;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.035em;
}
[data-testid="stMetricValue"] {
    color: #F3F6F8;
    font-family: 'Cascadia Code', Consolas, monospace;
    font-size: clamp(1.25rem, 2vw, 1.85rem);
}
[data-testid="stMetricDelta"] { font-size: 0.78rem; font-weight: 500; }
[data-testid="stTabs"] [role="tablist"] {
    gap: 0.45rem;
    border-bottom: 1px solid #25343F;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #AEBBC6;
    padding: 0.85rem 1.1rem;
    border-radius: 8px 8px 0 0;
    transition: background 150ms ease, color 150ms ease;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: #F4F6F8; background: #131F28; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #F4CC68;
    border-bottom-color: #F2C14E;
}
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid #2A3A45; }
.stButton > button,
[data-testid="stDownloadButton"] > button {
    border-radius: 9px;
    border: 1px solid #34444F;
    transition: border-color 150ms ease, background 150ms ease;
}
.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover { border-color: #F2C14E; }
.market-header {
    position: relative;
    display: flex; justify-content: space-between; align-items: center; gap: 1rem;
    padding: 1.55rem 1.55rem; margin: .5rem 0 1.5rem;
    background: linear-gradient(112deg, #14212A 0%, #111A22 68%, #172019 100%);
    border: 1px solid #2B3A44; border-radius: 16px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.16);
}
.market-header::before {
    position: absolute;
    inset: 0 auto 0 0;
    width: 3px;
    border-radius: 16px 0 0 16px;
    background: linear-gradient(180deg, #F2C14E, #B58A2C);
    content: "";
}
.market-brand { display: flex; align-items: center; gap: 1rem; min-width: 0; flex: 1 1 auto; }
.market-brand > div:last-child { min-width: 0; }
.market-logo {
    display: grid; place-items: center; width: 42px; height: 42px;
    flex: 0 0 42px;
    border: 1px solid rgba(242, 193, 78, 0.24);
    border-radius: 13px; background: rgba(242, 193, 78, 0.1);
    color: #F2C14E; font-weight: 800; font-family: 'Cascadia Code', Consolas, monospace;
}
.market-kicker {
    color: #B7C4CE; text-transform: uppercase; letter-spacing: .12em;
    font-size: .72rem; font-weight: 700; line-height: 1.6;
    white-space: nowrap;
}
.market-title { color: #F4F7F9; font-weight: 700; font-size: 1.2rem; letter-spacing: -0.02em; margin-top: .16rem; white-space: nowrap; }
.market-subtitle { color: #AAB8C3; font-size: .78rem; margin-top: .22rem; }
.market-meta { color: #B8C4CD; text-align: right; font-size: .74rem; line-height: 1.8; flex: 0 1 auto; }
.market-live { color: #66D19E; font-family: 'Cascadia Code', Consolas, monospace; }
.market-stale { color: #F2C14E; font-family: 'Cascadia Code', Consolas, monospace; }
div[data-testid="stPlotlyChart"] {
    border: 1px solid #24343E;
    border-radius: 14px;
    background: #0E161E;
    padding: 0.4rem;
}
@media (max-width: 760px) {
    .market-header { align-items: flex-start; flex-direction: column; }
    .market-meta { text-align: left; }
    [data-testid="stMetric"] { padding: 14px 15px; }
}
</style>
""")

latest_row = df.iloc[-1]
data_health = assess_market_data(df)
data_status_label = {
    'ready': 'DATA READY',
    'stale': 'DỮ LIỆU CHẬM',
    'error': 'DỮ LIỆU CÓ LỖI',
}[data_health['status']]

data_status_class = (
    'market-live'
    if data_health['status'] == 'ready'
    else 'market-stale'
)

st.html(f"""
<div class="market-header">
  <div class="market-brand">
    <div class="market-logo">Au</div>
    <div>
      <div class="market-kicker">Gold market intelligence</div>
      <div class="market-title">Gold Market Desk</div>
      <div class="market-subtitle">XAU · AI signal · risk monitoring</div>
    </div>
  </div>
  <div class="market-meta">
    <div><span class="{data_status_class}">● {data_status_label}</span></div>
    <div>Phiên dữ liệu mới nhất &nbsp; <strong>{latest_row['Date']:%d/%m/%Y}</strong></div>
    <div>{data_health['row_count']:,} dòng · trễ {data_health['business_days_lag']} ngày làm việc</div>
  </div>
</div>
""")

if data_health['status'] == 'error':
    st.error(
        "Dữ liệu có vấn đề về cấu trúc hoặc giá trị. "
        "Hãy cập nhật/kiểm tra dữ liệu trước khi dựa vào tín hiệu."
    )

elif data_health['status'] == 'stale':
    st.warning(
        "Dữ liệu đã trễ hơn 5 ngày làm việc. "
        "Hãy cập nhật trước khi dùng tín hiệu."
    )

elif (
    data_health['calendar_age_days']
    and data_health['calendar_age_days'] >= 3
):
    st.info(
        "Thị trường có thể nghỉ cuối tuần/ngày lễ; "
        "hãy đối chiếu ngày phiên mới nhất."
    )

# SIDEBAR
st.sidebar.markdown("### MARKET DESK")
st.sidebar.caption("GOLD · QUANT ANALYTICS")
st.sidebar.divider()
st.sidebar.subheader("Bảng điều khiển")

if st.sidebar.button(
    "↻ Cập nhật dữ liệu thị trường",
    use_container_width=True,
):
    try:
        with st.spinner(
            "Đang tải dữ liệu, kiểm tra thay đổi "
            "và cập nhật mô hình nếu có dữ liệu mới..."
        ):
            manual_status = run_refresh_if_due(
                BASE_DIR,
                refresh_market_data,
                interval_minutes=AUTO_REFRESH_INTERVAL_MINUTES,
                force=True,
            )

    except (
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        st.error(
            f"Không thể cập nhật thị trường: {error}"
        )

    else:
        if manual_status.get("status") == "error":
            st.error(
                "Cập nhật thất bại: "
                f"{manual_status.get('message', 'Lỗi không rõ')}"
            )
        else:
            st.cache_data.clear()
            st.rerun()

st.sidebar.caption(
    f"Phiên dữ liệu mới nhất: {latest_row['Date']:%d/%m/%Y}"
)

st.sidebar.caption(
    "Tự cập nhật: mỗi "
    f"{AUTO_REFRESH_INTERVAL_MINUTES} phút khi dashboard mở"
)

if auto_refresh_status.get("status") == "error":
    st.sidebar.warning(
        "Lần cập nhật tự động gần nhất thất bại: "
        f"{auto_refresh_status.get('message', '')}"
    )

elif auto_refresh_status.get("last_success_utc"):
    st.sidebar.caption(
        "Lần kiểm tra tự động thành công gần nhất (UTC): "
        f"{auto_refresh_status['last_success_utc']}"
    )

with st.sidebar.expander("Chất lượng và độ mới dữ liệu"):
    st.write(
        f"Trạng thái: **{data_status_label}**"
    )
    st.write(
        f"Số bản ghi: {data_health['row_count']:,}"
    )
    st.write(
        f"Trễ: {data_health['business_days_lag']} ngày làm việc"
    )
    st.write(
        "Ngày lịch từ phiên gần nhất: "
        f"{data_health['calendar_age_days']}"
    )
    if data_health['issues']:
        for issue in data_health['issues']:
            st.warning(issue)
    else:
        st.success("Không phát hiện ngày trùng, giá đóng cửa lỗi hoặc ngày tương lai.")

num_days = st.sidebar.slider(
    "Khoảng thời gian biểu đồ (phiên)",
    min_value=1,
    max_value=len(df),
    value=min(180, len(df)),
)

st.sidebar.subheader("Theo dõi rủi ro")
enable_layer1 = st.sidebar.checkbox("Cảnh báo biến động", value=True)
enable_layer2 = st.sidebar.checkbox("Điều chỉnh vị thế theo biến động", value=True)
enable_layer3 = st.sidebar.checkbox("Phân tích tin tức", value=True)

st.sidebar.subheader("Cảnh báo Telegram")
telegram_direction_filter = st.sidebar.selectbox(
    "Cho phép gửi hướng",
    ['Tất cả', 'Chỉ tín hiệu tăng', 'Chỉ tín hiệu giảm'],
    key='telegram_direction_filter',
)
telegram_min_confidence = st.sidebar.slider(
    "Tin cậy tối thiểu để gửi (%)",
    min_value=50,
    max_value=95,
    value=60,
    step=5,
    key='telegram_min_confidence',
)
st.sidebar.caption("Chỉ gửi khi bạn chủ động bấm nút; ứng dụng không tự gửi tin.")

st.sidebar.subheader("Chỉ báo trên biểu đồ")
show_sma = st.sidebar.checkbox("Đường trung bình SMA", value=True)
show_bb = st.sidebar.checkbox("Dải Bollinger", value=True)
show_daily_vol = st.sidebar.checkbox("Lợi suất hàng ngày", value=True)

df_view = df.tail(num_days).copy()
forecast_history = pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs([
    "Thị trường & tín hiệu",
    "Hiệu suất chiến lược",
    "Chất lượng mô hình",
    "Giá vàng SJC",
])

# ================= TAB 1 =================
with tab1:
    st.caption("MARKET SNAPSHOT")
    col1, col2 = st.columns(2)

    prev_close = (
        df.iloc[-2]['Close']
        if len(df) > 1
        else latest_row['Close']
    )

    diff = latest_row['Close'] - prev_close
    pct = (diff / prev_close) * 100 if prev_close else 0.0

    col1.metric(
        "Vàng · USD/oz",
        f"${latest_row['Close']:,.2f}",
        f"{diff:+.2f} ({pct:+.2f}%)",
    )

    if 'USD_Index' in latest_row:
        dxy_ret = latest_row.get(
            'USD_Index_Return',
            0.0,
        ) * 100

        col2.metric(
            "USD Index",
            f"{latest_row['USD_Index']:.2f}",
            f"{dxy_ret:+.2f}%",
        )

    col3, col4 = st.columns(2)

    if 'US10Y_Yield' in latest_row:
        col3.metric(
            "Lợi suất 10Y",
            f"{latest_row['US10Y_Yield']:.2f}%",
        )

    daily_range = latest_row.get(
        'Daily_Range_USD',
        0.0,
    )

    col4.metric(
        "Biên độ phiên",
        f"${daily_range:.2f}",
        f"{latest_row.get('Daily_Range_Pct', 0.0):.2f}%",
    )

    if (
        model_payload is not None
        and data_health['status'] != 'error'
    ):
        model = model_payload['model']
        feature_cols = model_payload['feature_cols']
        input_vector = df.iloc[[-1]][feature_cols]

        pred = model.predict(input_vector)[0]

        confidence = float(
            model.predict_proba(input_vector)[0][pred]
        )

        proba = confidence * 100

        return_model = model_payload.get('return_model')

        predicted_return_pct = (
            float(return_model.predict(input_vector)[0])
            if return_model is not None
            else 0.0
        )

        interval_half_width_pct = float(
            model_payload.get(
                'prediction_interval_pct',
                0.0,
            )
        )

        expected_close = (
            latest_row['Close']
            * (1 + predicted_return_pct / 100)
        )

        lower_close = (
            latest_row['Close']
            * (
                1
                + (
                    predicted_return_pct
                    - interval_half_width_pct
                ) / 100
            )
        )

        upper_close = (
            latest_row['Close']
            * (
                1
                + (
                    predicted_return_pct
                    + interval_half_width_pct
                ) / 100
            )
        )

        forecast_history = update_forecast_history(
            FORECAST_HISTORY_PATH,
            df[['Date', 'Target', 'Return', 'Close']],
            {
                'date': latest_row['Date'],
                'prediction': int(pred),
                'confidence': confidence,
                'model_name': model_payload['model_name'],
                'previous_direction': int(
                    latest_row['Return'] > 0
                ),
                'predicted_return_pct': predicted_return_pct,
                'reference_close': float(
                    latest_row['Close']
                ),
                'prediction_interval_pct': interval_half_width_pct,
                'model_version': model_payload.get(
                    'trained_at',
                    (
                        f"{model_payload['model_name']}:"
                        f"{model_payload.get('walk_forward', {}).get('test_end_date', 'legacy')}"
                    ),
                ),
            },
        )

        signal_text = (
            f"Tín hiệu mô hình: "
            f"{'TĂNG' if pred == 1 else 'GIẢM'} "
            f"· xác suất {proba:.1f}%"
        )

        if pred == 1:
            st.success(signal_text)
        else:
            st.error(signal_text)

        st.metric(
            "Giá đóng cửa ước tính phiên tới",
            f"${expected_close:,.2f}",
            f"{predicted_return_pct:+.3f}%",
        )

        interval_level = model_payload.get(
            'return_metrics',
            {},
        ).get(
            'prediction_interval_level',
            0.90,
        )

        st.caption(
            f"Khoảng ước lượng {(interval_level * 100):.0f}%: "
            f"${lower_close:,.2f} – ${upper_close:,.2f}. "
            "Độ rộng lấy từ sai số validation; không phải mức giá đảm bảo."
        )

        st.caption(
            "Xác suất tăng/giảm là điểm predict_proba chưa hiệu chuẩn; "
            "không nên hiểu là xác suất chính xác ngoài thực tế."
        )

        direction_allowed = (
            telegram_direction_filter == 'Tất cả'
            or (
                telegram_direction_filter == 'Chỉ tín hiệu tăng'
                and pred == 1
            )
            or (
                telegram_direction_filter == 'Chỉ tín hiệu giảm'
                and pred == 0
            )
        )

        confidence_allowed = (
            proba >= telegram_min_confidence
        )

        if not direction_allowed:
            st.caption(
                "Chưa gửi: hướng tín hiệu không khớp "
                "bộ lọc Telegram."
            )

        elif not confidence_allowed:
            st.caption(
                f"Chưa gửi: độ tin cậy {proba:.1f}% thấp hơn "
                f"ngưỡng {telegram_min_confidence}%."
            )

        if st.button(
            "Gửi tín hiệu này qua Telegram",
            key="send_telegram_signal",
            disabled=not (
                direction_allowed and confidence_allowed
            ),
        ):
            from src.telegram_bot import send_telegram_message

            direction = (
                "TĂNG" if pred == 1 else "GIẢM"
            )

            telegram_text = (
                "TÍN HIỆU DỰ BÁO GIÁ VÀNG AI\n"
                f"Giá hiện tại: ${latest_row['Close']:,.2f} USD/Oz\n"
                f"Dự báo phiên tới: {direction}\n"
                f"Độ tin cậy của mô hình: {proba:.1f}%\n"
                f"Giá đóng cửa ước tính: ${expected_close:,.2f} USD/Oz\n"
                f"Khoảng ước lượng: ${lower_close:,.2f} - "
                f"${upper_close:,.2f} USD/Oz\n"
                "Xác suất phân lớp chưa hiệu chuẩn; "
                "khoảng giá không đảm bảo.\n"
                "Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."
            )

            try:
                send_telegram_message(telegram_text)

            except (
                RuntimeError,
                OSError,
                ValueError,
            ) as error:
                st.error(str(error))

            else:
                st.success("Đã gửi tín hiệu Telegram.")

    elif model_payload is not None:
        st.error(
            "Đã tạm ẩn tín hiệu vì dữ liệu có lỗi kiểm tra. "
            "Hãy cập nhật dữ liệu rồi thử lại."
        )

    else:
        st.warning("Chưa có mô hình đã huấn luyện.")

    st.divider()

    if enable_layer1:
        current_vol = latest_row.get(
            'Volatility_20_Ann',
            0.0,
        )

        vol_spike = latest_row.get(
            'Volatility_Spike_Flag',
            0,
        )

        if vol_spike == 1 or current_vol > 20.0:
            st.error(
                "**RỦI RO BIẾN ĐỘNG CAO**\n\n"
                f"Biến động thường niên: `{current_vol:.1f}%`. "
                "Cân nhắc giảm quy mô vị thế và đánh giá lại "
                "rủi ro trước khi giao dịch."
            )

        else:
            st.success(
                "**BIẾN ĐỘNG TRONG NGƯỠNG THEO DÕI**\n\n"
                f"Biến động thường niên hiện tại: `{current_vol:.1f}%`. "
                "Đây là chỉ báo tham khảo, không phải khuyến nghị đầu tư."
            )

    if enable_layer3:
        st.subheader("Tin tức & rủi ro địa chính trị")

        if st.button(
            "Làm mới tin tức",
            key="refresh_news",
        ):
            st.session_state['news_refresh_key'] = (
                st.session_state.get('news_refresh_key', 0) + 1
            )

        refresh_key = st.session_state.get(
            'news_refresh_key',
            0,
        )

        try:
            news_data = load_news(refresh_key)

        except NewsFetchError as error:
            st.error(str(error))

        else:
            n_col1, n_col2 = st.columns([1, 2])
            with n_col1:      
            with n_col1:
                st.metric(
                    "Điểm cảm xúc tiêu đề",
                    f"{news_data['avg_sentiment']:.2f}",
                    (
                        "Có từ khóa rủi ro"
                        if news_data['crisis_flag']
                        else "Không phát hiện từ khóa rủi ro"
                    ),
                )

                if news_data['crisis_flag']:
                    st.warning(
                        "Có tiêu đề chứa từ khóa rủi ro "
                        f"({news_data['crisis_count']})."
                    )
                else:
                    st.info(
                        "Không phát hiện từ khóa rủi ro trong tiêu đề."
                    )

            with n_col2:
                st.markdown(
                    "**Tin thật mới nhất · Google News RSS**"
                )

                translation_available = True

                for item in news_data['news_list']:
                    icon = (
                        "🔴"
                        if item['is_crisis']
                        else "·"
                    )

                    article_url = item.get('url', '')
                    original_title = item.get('title', '')
                    translated_title = original_title

                    if translation_available:
                        try:
                            translated_title = translate_news_title(
                                original_title
                            )
                        except NewsTranslationError:
                            translation_available = False

                    if article_url:
                        st.markdown(
                            f"{icon} **{item['source']}** · "
                            f"[{translated_title}]({article_url})"
                        )
                    else:
                        st.write(
                            f"{icon} **{item['source']}** · "
                            f"{translated_title}"
                        )

                    if translated_title != original_title:
                        st.caption(
                            f"Nguyên văn: {original_title} · "
                            f"{item['published_at']}"
                        )
                    else:
                        st.caption(item['published_at'])

                if not translation_available:
                    st.warning(
                        "Không thể dịch một số tiêu đề; các tin đó "
                        "đang hiển thị bằng ngôn ngữ gốc."
                    )

            st.caption(
                "Tin và liên kết bài gốc lấy từ Google News RSS; "
                "tiêu đề được dịch sang tiếng Việt, kèm nguyên văn "
                "để đối chiếu. Bản dịch máy có thể chưa chính xác "
                "tuyệt đối. Bản dịch được lưu tối đa 24 giờ; "
                "tin RSS tối đa 15 phút. Điểm cảm xúc chỉ dựa trên "
                "từ khóa tiêu đề gốc."
            )

    st.subheader("Giá vàng và chỉ báo kỹ thuật")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_view['Date'],
            y=df_view['Close'],
            mode='lines',
            name='Giá đóng cửa',
            line=dict(
                color='#F2C14E',
                width=2.4,
            ),
        )
    )

    if show_sma:
        for col_name, color in [
            ('SMA_7', '#3B82F6'),
            ('SMA_14', '#8B5CF6'),
            ('SMA_50', '#EF4444'),
            ('SMA_200', '#10B981'),
        ]:
            if col_name in df_view.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_view['Date'],
                        y=df_view[col_name],
                        mode='lines',
                        name=col_name,
                        line=dict(
                            color=color,
                            width=1,
                        ),
                    )
                )

    if (
        show_bb
        and 'BB_Upper' in df_view.columns
        and 'BB_Lower' in df_view.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=df_view['Date'],
                y=df_view['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(
                    color='#9CA3AF',
                    dash='dash',
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_view['Date'],
                y=df_view['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(
                    color='#9CA3AF',
                    dash='dash',
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Ngày giao dịch",
        yaxis_title="USD / oz",
        hovermode="x unified",
        height=440,
    )

    style_chart(fig)

    st.plotly_chart(
        fig,
        width="stretch",
    )

    if (
        show_daily_vol
        and 'Daily_Return_Pct' in df_view.columns
    ):
        st.subheader("Lợi suất theo phiên")

        colors = [
            '#54C99A' if val >= 0 else '#F07178'
            for val in df_view['Daily_Return_Pct']
        ]

        fig_d = go.Figure()

        fig_d.add_trace(
            go.Bar(
                x=df_view['Date'],
                y=df_view['Daily_Return_Pct'],
                marker_color=colors,
                name="Lợi suất (%)",
            )
        )

        fig_d.update_layout(
            xaxis_title="Ngày",
            yaxis_title="Lợi suất (%)",
            height=300,
            hovermode="x unified",
        )

        style_chart(fig_d)

        st.plotly_chart(
            fig_d,
            width="stretch",
        )

    st.download_button(
        "Tải CSV dữ liệu đang xem",
        data=df_view.to_csv(
            index=False
        ).encode('utf-8-sig'),
        file_name="gold_market_data.csv",
        mime="text/csv",
    )


# ================= TAB 4: VIETNAM SJC QUOTES =================
with tab4:
    st.subheader("Giá vàng miếng SJC Việt Nam")

    st.caption(
        "Giá SJC được nhập từ tệp CSV do bạn cung cấp. "
        "Ứng dụng không tự giả lập giá SJC và chưa kết nối "
        "nguồn báo giá trực tiếp. Hãy xác minh nguồn và thời điểm cập nhật."
    )

    st.markdown(
        "Tệp CSV cần các cột "
        "`Date`, `Buy_VND_per_luong`, `Sell_VND_per_luong`, `Source`. "
        "Ngày dùng định dạng YYYY-MM-DD; giá tính bằng VND/lượng."
    )

    template_csv = (
        "Date,Buy_VND_per_luong,Sell_VND_per_luong,Source\n"
    )

    st.download_button(
        "Tải mẫu CSV giá SJC",
        data=template_csv.encode("utf-8-sig"),
        file_name="sjc_prices_template.csv",
        mime="text/csv",
        key="download_sjc_template",
    )

    sjc_upload = st.file_uploader(
        "Chọn CSV báo giá SJC",
        type=["csv"],
        key="sjc_csv_upload",
    )

    sjc_frame = None

    if sjc_upload is not None:
        try:
            sjc_frame = validate_sjc_quotes(
                pd.read_csv(sjc_upload)
            )
        except (
            ValueError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            st.error(
                f"Không thể đọc báo giá SJC: {error}"
            )

    elif os.path.exists(SJC_DATA_PATH):
        try:
            sjc_frame = validate_sjc_quotes(
                pd.read_csv(SJC_DATA_PATH)
            )
        except (
            ValueError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            st.error(
                f"Tệp SJC đã lưu không hợp lệ: {error}"
            )

    if sjc_frame is not None and not sjc_frame.empty:
        if (
            sjc_upload is not None
            and st.button(
                "Lưu dữ liệu SJC vào ứng dụng",
                key="save_sjc_quotes",
            )
        ):
            try:
                with atomic_output_path(
                    SJC_DATA_PATH
                ) as temporary_path:
                    sjc_frame.to_csv(
                        temporary_path,
                        index=False,
                        date_format="%Y-%m-%d",
                    )

            except OSError as error:
                st.error(
                    f"Không thể lưu dữ liệu SJC: {error}"
                )

            else:
                st.success(
                    "Đã lưu báo giá SJC đã kiểm tra "
                    "vào data/sjc_prices.csv."
                )

        latest_sjc = sjc_frame.iloc[-1]

        quote_date = pd.Timestamp(
            latest_sjc["Date"]
        ).date()

        age_days = max(
            (
                pd.Timestamp.now().date()
                - quote_date
            ).days,
            0,
        )

        if age_days > 3:
            st.warning(
                f"Báo giá mới nhất đã {age_days} ngày tuổi; "
                "hãy kiểm tra lại nguồn trước khi sử dụng."
            )
        else:
            st.info(
                f"Ngày báo giá mới nhất: {quote_date:%d/%m/%Y} "
                f"· Nguồn: {latest_sjc['Source']}"
            )

        sjc_buy, sjc_sell, sjc_spread = st.columns(3)

        sjc_buy.metric(
            "SJC mua vào",
            f"{latest_sjc['Buy_VND_per_luong']:,.0f} ₫/lượng",
        )

        sjc_sell.metric(
            "SJC bán ra",
            f"{latest_sjc['Sell_VND_per_luong']:,.0f} ₫/lượng",
        )

        sjc_spread.metric(
            "Chênh lệch mua–bán",
            f"{latest_sjc['Sell_VND_per_luong'] - latest_sjc['Buy_VND_per_luong']:,.0f} ₫",
        )

        sjc_chart = go.Figure()

        sjc_chart.add_trace(
            go.Scatter(
                x=sjc_frame["Date"],
                y=sjc_frame["Buy_VND_per_luong"],
                mode="lines+markers",
                name="Mua vào",
            )
        )

        sjc_chart.add_trace(
            go.Scatter(
                x=sjc_frame["Date"],
                y=sjc_frame["Sell_VND_per_luong"],
                mode="lines+markers",
                name="Bán ra",
            )
        )

        sjc_chart.update_layout(
            title="Lịch sử báo giá SJC",
            xaxis_title="Ngày",
            yaxis_title="VND/lượng",
            height=390,
        )

        style_chart(sjc_chart)

        st.plotly_chart(
            sjc_chart,
            width="stretch",
            key="sjc_history_chart",
        )

        st.markdown(
            "**So sánh tham khảo với vàng thế giới**"
        )

        fx = st.number_input(
            "Tỷ giá USD/VND do bạn nhập",
            min_value=0.0,
            value=0.0,
            step=100.0,
            help=(
                "Nhập tỷ giá tham chiếu từ nguồn bạn tin cậy. "
                "Không tự động cập nhật và không phải báo giá "
                "ngân hàng trực tiếp."
            ),
            key="sjc_usd_vnd_rate",
        )

        if fx > 0 and float(latest_row["Close"]) > 0:
            try:
                comparison = quote_comparison(
                    sjc_frame,
                    float(latest_row["Close"]),
                    fx,
                )
            except ValueError as error:
                st.error(str(error))

            else:
                eq_col, premium_col, pct_col = st.columns(3)

                eq_col.metric(
                    "Giá quốc tế quy đổi*",
                    f"{comparison['international_equivalent']:,.0f} ₫/lượng",
                )

                premium_col.metric(
                    "Chênh lệch so với giá SJC giữa mua–bán",
                    f"{comparison['midpoint_premium_vnd']:+,.0f} ₫",
                )

                pct_col.metric(
                    "Chênh lệch tương đối",
                    f"{comparison['midpoint_premium_pct']:+.2f}%",
                )

                st.caption(
                    "*Quy đổi lý thuyết từ GC=F (USD/ounce troy) "
                    "và tỷ giá bạn nhập, chưa gồm thuế, phí, chi phí "
                    "nhập khẩu, khác biệt sản phẩm hay điều kiện thị trường. "
                    "Không dùng làm giá giao dịch hoặc khuyến nghị đầu tư."
                )

        else:
            st.info(
                "Nhập tỷ giá USD/VND để xem phép quy đổi tham khảo; "
                "ứng dụng không tự đoán tỷ giá hiện hành."
            )

        st.dataframe(
            sjc_frame.sort_values(
                "Date",
                ascending=False,
            ),
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Tải dữ liệu SJC đã kiểm tra",
            data=sjc_frame.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="sjc_prices_validated.csv",
            mime="text/csv",
            key="download_sjc_validated",
        )

    else:
        st.info(
            "Chưa có dữ liệu SJC. Tải mẫu CSV, điền báo giá "
            "có nguồn rõ ràng rồi tải lên. Các giá trị mẫu "
            "không được tự tạo để tránh hiển thị giá sai."
        )  