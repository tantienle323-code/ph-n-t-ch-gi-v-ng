import os
import pickle
import requests
import pandas as pd

# Điền Token Bot và Chat ID Telegram của bạn
TELEGRAM_BOT_TOKEN = "7123456789:AAE-xxxxxxxxx_xxxxxxxxxxxxxxx"
TELEGRAM_CHAT_ID = "123456789"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'gold_data_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gold_model.pkl')

def send_telegram_alert():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("❌ Chưa có file data hoặc model!")
        return

    df = pd.read_csv(DATA_PATH)
    with open(MODEL_PATH, 'rb') as f:
        model_payload = pickle.load(f)

    model = model_payload['model']
    feature_cols = model_payload['feature_cols']

    latest_row = df.iloc[-1]
    input_data = df.iloc[[-1]][feature_cols]

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][pred] * 100

    signal_text = "📈 TĂNG MẠNH" if pred == 1 else "📉 GIẢM MẠNH"
    price = latest_row['Close']
    rsi = latest_row['RSI_14']

    message = (
        f"🚨 **TÍN HIỆU DỰ BÁO GIÁ VÀNG AI** 🚨\n\n"
        f"💵 Giá hiện tại: **${price:,.2f} USD/Oz**\n"
        f"📊 Chỉ số RSI(14): **{rsi:.1f}**\n"
        f"🎯 Dự báo phiên tới: **{signal_text}**\n"
        f"🔥 Độ tự tin cậy: **{proba:.1f}%**\n\n"
        f"💡 *Khuyến nghị: Kết hợp quản trị vốn chặt chẽ!*"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Đã gửi thông báo Telegram thành công!")
    else:
        print(f"❌ Lỗi gửi Telegram: {response.text}")

if __name__ == '__main__':
    send_telegram_alert()