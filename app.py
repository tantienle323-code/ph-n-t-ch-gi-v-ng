import os
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Nạp module tin tức Tầng 3
from src.news_sentiment import get_latest_news_sentiment

st.set_page_config(
    page_title="Hệ Thống AI Phân Tích & Dự Báo Giá Vàng (Enterprise Quant)",
    page_icon="🪙",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'gold_data_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gold_model.pkl')

@st.cache_data
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

df = load_data()
model_payload = load_model()

if df is None:
    st.error("❌ Chưa tìm thấy dữ liệu `gold_data_processed.csv`. Hãy chạy `python data/get_data.py` và `python data/preprocess_data.py`!")
    st.stop()

st.title("🪙 HỆ THỐNG PHÂN TÍCH & DỰ BÁO GIÁ VÀNG AI (3-LAYER CRISIS PROTECTED)")
st.caption("Hệ thống Quant Trading Tích hợp Phân tích Vĩ mô, Rada Tin tức Địa chính trị & Bảo vệ Rủi ro 3 Tầng")
st.divider()

# SIDEBAR
st.sidebar.header("⚙️ Cấu Hình Hệ Thống")
num_days = st.sidebar.slider("Khung thời gian xem (Phiên gần nhất):", min_value=30, max_value=len(df), value=180)

st.sidebar.subheader("🛡️ Cấu Hình Ứng Phó Rủi Ro 3 Tầng:")
enable_layer1 = st.sidebar.checkbox("Tầng 1: Visual Volatility / Crisis Radar", value=True)
enable_layer2 = st.sidebar.checkbox("Tầng 2: Dynamic Vol-Sizing Backtest", value=True)
enable_layer3 = st.sidebar.checkbox("Tầng 3: News Geopolitical Sentiment Radar", value=True)

st.sidebar.subheader("📈 Tùy Chọn Chỉ Báo Kỹ Thuật:")
show_sma = st.sidebar.checkbox("Đường Trung Bình SMA (7, 14, 50, 200)", value=True)
show_bb = st.sidebar.checkbox("Dải Bollinger Bands", value=True)
show_daily_vol = st.sidebar.checkbox("📊 Biến Động Thị Trường Theo Từng Ngày", value=True)

df_view = df.tail(num_days).copy()
latest_row = df.iloc[-1]

tab1, tab2, tab3 = st.tabs(["🔮 Dự Báo & Bảo Vệ 3 Tầng", "💼 Mô Phỏng Đầu Tư Quant (Backtest)", "🔬 Phân Tích Mô Hình AI"])

# ================= TAB 1 =================
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    prev_close = df.iloc[-2]['Close']
    diff = latest_row['Close'] - prev_close
    pct = (diff / prev_close) * 100
    
    col1.metric("Giá Vàng (USD/Oz)", f"${latest_row['Close']:,.2f}", f"{diff:+.2f} ({pct:+.2f}%)")
    
    if 'USD_Index' in latest_row:
        dxy_ret = latest_row.get('USD_Index_Return', 0.0)*100
        col2.metric("Chỉ Số USD (DXY)", f"{latest_row['USD_Index']:.2f}", f"{dxy_ret:+.2f}%")
    if 'US10Y_Yield' in latest_row:
        col3.metric("Lãi Suất Mỹ 10Y", f"{latest_row['US10Y_Yield']:.2f}%")
        
    daily_range = latest_row.get('Daily_Range_USD', 0.0)
    col4.metric("Biên Độ Giá Hôm Nay", f"${daily_range:.2f}", f"{latest_row.get('Daily_Range_Pct', 0.0):.2f}% Dao động")
        
    with col5:
        if model_payload is not None:
            model = model_payload['model']
            feature_cols = model_payload['feature_cols']
            input_vector = df.iloc[[-1]][feature_cols]
            
            pred = model.predict(input_vector)[0]
            proba = model.predict_proba(input_vector)[0][pred] * 100
            
            if pred == 1:
                st.success(f"📈 **NGÀY MAI: TĂNG**\n\nĐộ tin cậy: **{proba:.1f}%**")
            else:
                st.error(f"📉 **NGÀY MAI: GIẢM**\n\nĐộ tin cậy: **{proba:.1f}%**")
        else:
            st.warning("⚠️ Chưa nạp mô hình!")

    # 🔴 TẦNG 1: visual VOLATILITY / CRISIS RADAR BANNER
    if enable_layer1:
        current_vol = latest_row.get('Volatility_20_Ann', 0.0)
        vol_spike = latest_row.get('Volatility_Spike_Flag', 0)
        
        if vol_spike == 1 or current_vol > 20.0:
            st.error("🚨 **TẦNG 1 KÍCH HOẠT: CẢNH BÁO THIÊN NGA ĐEN / BÃI GIÁ (CRISIS ALERT)**\n\n"
                     f"Biên độ giá ATR bùng nổ vượt 1.8x trung bình! Độ biến động hàng năm: `{current_vol:.1f}%`. "
                     "Khuyên nghị: Tạm ngưng mở vị thế mới hoặc giảm 50% quy mô vốn giao dịch!")
        else:
            st.success("🟢 **TẦNG 1: TRẠNG THÁI THỊ TRƯỜNG ỔN ĐỊNH**\n\nBiên độ biến động nằm trong ngưỡng an toàn. Tín hiệu dự báo AI đạt độ tin cậy chuẩn định lượng.")

    # 🌐 TẦNG 3: RADA PHÂN TÍCH TIN TỨC ĐỊA CHÍNH TRỊ REAL-TIME
    if enable_layer3:
        st.subheader("🌐 Tầng 3: Rada Phân Tích Cảm Sức Tin Tức Địa Chính Trị (News Geopolitical Radar)")
        news_data = get_latest_news_sentiment()
        
        n_col1, n_col2 = st.columns([1, 2])
        with n_col1:
            st.metric("Điểm Cảm Sức Tin Tức", f"{news_data['avg_sentiment']:.2f}", 
                      "Rủi ro Địa chính trị" if news_data['crisis_flag'] else "Bình thường")
            if news_data['crisis_flag']:
                st.warning("⚠️ Phát hiện từ khóa Xung đột / Chiến tranh / Trừng phạt trong dòng tin tức quốc tế!")
            else:
                st.info("ℹ️ Không phát hiện biến cố tin tức bất ngờ.")
                
        with n_col2:
            st.markdown("**📰 Tiêu Đề Tin Tức Quốc Tế Mới Nhất Quét Được:**")
            for item in news_data['news_list']:
                icon = "🔴" if item['is_crisis'] else "🔹"
                st.write(f"{icon} **[{item['source']}]** {item['title']}")

    st.subheader("📈 Biểu Đồ Diễn Biến Giá Vàng & Chỉ Báo Kỹ Thuật")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['Close'], mode='lines', name='Giá Đóng Cửa', line=dict(color='#EAB308', width=2)))
    
    if show_sma:
        for col_name, color in [('SMA_7', '#3B82F6'), ('SMA_14', '#8B5CF6'), ('SMA_50', '#EF4444'), ('SMA_200', '#10B981')]:
            if col_name in df_view.columns:
                fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view[col_name], mode='lines', name=col_name, line=dict(color=color, width=1)))
                
    if show_bb and 'BB_Upper' in df_view.columns and 'BB_Lower' in df_view.columns:
        fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='#9CA3AF', dash='dash')))
        fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='#9CA3AF', dash='dash')))

    fig.update_layout(xaxis_title="Ngày Giao Dịch", yaxis_title="USD/Oz", hovermode="x unified", height=420)
    st.plotly_chart(fig, use_container_width=True)

    if show_daily_vol and 'Daily_Return_Pct' in df_view.columns:
        st.subheader("📊 Biểu Đồ Biến Động Thị Trường Theo Từng Ngày")
        colors = ['#10B981' if val >= 0 else '#EF4444' for val in df_view['Daily_Return_Pct']]
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=df_view['Date'], y=df_view['Daily_Return_Pct'], marker_color=colors, name='% Thay Đổi Ngày'))
        fig_d.update_layout(xaxis_title="Ngày", yaxis_title="% Thay Đổi", height=300, hovermode="x unified")
        st.plotly_chart(fig_d, use_container_width=True)

# ================= TAB 2: BACKTEST QUANT =================
with tab2:
    st.subheader("💼 Tầng 2: Mô Phỏng Đầu Tư Quant & Quản Trị Vốn Theo Biến Động (Vol-Sizing Backtest)")
    
    if model_payload is not None:
        col_set1, col_set2, col_set3 = st.columns(3)
        initial_capital = col_set1.number_input("Số vốn ban đầu (USD):", value=10000, step=1000)
        stop_loss_pct = col_set2.slider("Cắt lỗ tự động (Stop-Loss %):", min_value=0.5, max_value=5.0, value=1.5, step=0.1) / 100
        take_profit_pct = col_set3.slider("Chốt lời tự động (Take-Profit %):", min_value=1.0, max_value=10.0, value=3.0, step=0.5) / 100
        
        model = model_payload['model']
        feature_cols = model_payload['feature_cols']
        
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        preds = model.predict(test_df[feature_cols])
        test_df['Signal'] = preds
        
        capital = initial_capital
        portfolio = [capital]
        win_trades, loss_trades = 0, 0
        
        for i in range(len(test_df) - 1):
            price_today = test_df.loc[i, 'Close']
            price_next = test_df.loc[i+1, 'Close']
            sig = test_df.loc[i, 'Signal']
            vol_spike = test_df.loc[i, 'Volatility_Spike_Flag'] if 'Volatility_Spike_Flag' in test_df.columns else 0
            
            if sig == 1:
                # 🟡 TẦNG 2: Volatility Position Sizing - Giảm 50% vốn vào lệnh nếu bão giá
                position_scale = 0.5 if (enable_layer2 and vol_spike == 1) else 1.0
                
                raw_ret = (price_next - price_today) / price_today
                
                # Quản trị rủi ro Stop-Loss & Take-Profit
                if raw_ret <= -stop_loss_pct:
                    actual_ret = -stop_loss_pct * position_scale
                    loss_trades += 1
                elif raw_ret >= take_profit_pct:
                    actual_ret = take_profit_pct * position_scale
                    win_trades += 1
                else:
                    actual_ret = raw_ret * position_scale
                    if actual_ret > 0: win_trades += 1
                    elif actual_ret < 0: loss_trades += 1
                    
                capital = capital * (1 + actual_ret)
            portfolio.append(capital)
            
        test_df['AI_Portfolio'] = portfolio
        
        ai_final = test_df['AI_Portfolio'].iloc[-1]
        ai_return_pct = ((ai_final - initial_capital) / initial_capital) * 100
        
        portfolio_series = pd.Series(portfolio)
        rolling_max = portfolio_series.cummax()
        drawdowns = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        daily_returns = portfolio_series.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if (len(daily_returns) > 0 and daily_returns.std() != 0) else 0.0
        
        total_trades = win_trades + loss_trades
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0.0

        buy_hold_ret = (test_df['Close'] - test_df['Close'].iloc[0]) / test_df['Close'].iloc[0]
        test_df['Buy_Hold_Portfolio'] = initial_capital * (1 + buy_hold_ret)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tài Sản Cuối Kỳ (AI)", f"${ai_final:,.2f}", f"{ai_return_pct:+.2f}%")
        c2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Hiệu quả lợi nhuận/rủi ro")
        c3.metric("Max Drawdown (MDD)", f"{max_drawdown:.2f}%", "Mức sụt giảm tối đa")
        c4.metric("Win Rate (%)", f"{win_rate:.1f}%", f"{win_trades}/{total_trades} Lệnh thắng")
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=test_df['Date'], y=test_df['AI_Portfolio'], name='Chiến Lược Quant AI (Bảo vệ 3 Tầng)', line=dict(color='#10B981', width=2.5)))
        fig_bt.add_trace(go.Scatter(x=test_df['Date'], y=test_df['Buy_Hold_Portfolio'], name='Chiến Lược Buy & Hold', line=dict(color='#6B7280', dash='dash')))
        fig_bt.update_layout(xaxis_title="Ngày", yaxis_title="USD", height=400)
        st.plotly_chart(fig_bt, use_container_width=True)

# ================= TAB 3: MODEL DIAGNOSTICS =================
with tab3:
    st.subheader("🔬 Đánh Giá Mô Hình & Độ Quan Trọng Của Các Đặc Trưng")
    if model_payload is not None:
        metrics = model_payload['metrics']
        st.write(f"🏆 **Mô hình xuất sắc nhất:** `{model_payload['model_name']}`")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        m3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        
        st.divider()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**📊 Top 10 Đặc Trưng Quan Trọng Nhất:**")
            importances = model_payload.get('feature_importances', {})
            if importances:
                top_imp = dict(list(importances.items())[:10])
                fig_imp = px.bar(x=list(top_imp.values()), y=list(top_imp.keys()), orientation='h', color=list(top_imp.values()), color_continuous_scale='Viridis')
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)
        with col_m2:
            st.markdown("**🧩 Ma Trận Nhầm Lẫn (Confusion Matrix):**")
            cm = np.array(metrics['confusion_matrix'])
            fig_cm = px.imshow(cm, text_auto=True, x=['Dự báo GIẢM', 'Dự báo TĂNG'], y=['Thực tế GIẢM', 'Thực tế TĂNG'], color_continuous_scale='Blues')
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)