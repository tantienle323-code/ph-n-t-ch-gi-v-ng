import os
import pickle

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'gold_data_processed.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gold_model.pkl')


def send_telegram_message(message):
    """Send a Telegram message using credentials provided by the environment."""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        raise RuntimeError(
            'Thiếu cấu hình Telegram. Hãy đặt TELEGRAM_BOT_TOKEN và TELEGRAM_CHAT_ID.'
        )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={'chat_id': chat_id, 'text': message},
            timeout=(5, 15),
        )
    except requests.RequestException:
        raise RuntimeError('Không thể kết nối Telegram. Vui lòng thử lại sau.') from None

    if not response.ok:
        raise RuntimeError(f"Telegram trả về lỗi HTTP {response.status_code}.")
    try:
        response_data = response.json()
    except ValueError:
        raise RuntimeError('Telegram trả về phản hồi không hợp lệ.') from None
    if not response_data.get('ok'):
        raise RuntimeError(f"Telegram từ chối gửi tin nhắn: {response_data.get('description', 'lỗi không xác định')}")


def build_signal_message(data_path=DATA_PATH, model_path=MODEL_PATH):
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Chưa có dữ liệu hoặc mô hình để tạo tín hiệu Telegram.")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Dữ liệu chưa có phiên giao dịch để tạo tín hiệu.")
    with open(model_path, 'rb') as model_file:
        model_payload = pickle.load(model_file)

    model = model_payload['model']
    feature_cols = model_payload['feature_cols']
    input_data = df.iloc[[-1]][feature_cols]
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction] * 100
    latest_row = df.iloc[-1]
    direction = "TĂNG" if prediction == 1 else "GIẢM"
    rsi = latest_row.get('RSI_14')
    rsi_text = f"{rsi:.1f}" if pd.notna(rsi) else "N/A"

    return (
        "TÍN HIỆU DỰ BÁO GIÁ VÀNG AI\n\n"
        f"Giá hiện tại: ${latest_row['Close']:,.2f} USD/Oz\n"
        f"RSI(14): {rsi_text}\n"
        f"Dự báo phiên tới: {direction}\n"
        f"Độ tin cậy của mô hình: {probability:.1f}%\n\n"
        "Chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."
    )


def send_telegram_alert():
    send_telegram_message(build_signal_message())


if __name__ == '__main__':
    send_telegram_alert()
