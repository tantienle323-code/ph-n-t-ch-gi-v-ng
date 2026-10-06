import os
import pandas as pd
import numpy as np

# Tự động định vị đường dẫn tuyệt đối tới thư mục chứa file script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'gold_data_5y.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'gold_data_processed.csv')

def preprocess_gold_data():
    print("==================================================")
    print("   TIỀN XỬ LÝ & NÂNG CẤP ĐẶC TRƯNG TÀI CHÍNH  ")
    print("==================================================")
    
    # 1. Kiểm tra sự tồn tại của file dữ liệu thô
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_FILE}'!")
        print("   Hãy chạy lệnh 'python data/get_data.py' trước!")
        return

    # 2. Đọc và sắp xếp dữ liệu theo chuỗi thời gian
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 3. Biến động giá hàng ngày (Daily Return)
    df['Return'] = df['Close'].pct_change()

    # 4. Các đường trung bình động (SMA 7, 14, 50)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 5. Độ biến động giá 20 ngày (Volatility)
    df['Volatility_20'] = df['Return'].rolling(window=20).std()

    # 6. Chỉ số sức mạnh tương đối RSI (14 ngày)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 7. Chỉ báo phân kỳ hội tụ MACD & Signal
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 8. Dải Bollinger Bands (20 ngày)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)

    # 9. Biến trễ giá Vàng (Lag Features: t-1, t-2, t-3)
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag3'] = df['Return'].shift(3)

    # 10. Biến vĩ mô Chỉ số USD (nếu có dữ liệu USD_Index)
    if 'USD_Index' in df.columns:
        df['USD_Return'] = df['USD_Index'].pct_change()
        df['USD_Lag1'] = df['USD_Return'].shift(1)

    # 11. Biến mục tiêu Target (1: Ngày mai giá tăng, 0: Ngày mai giá giảm/bằng)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 12. Kiểm soát an toàn dữ liệu: Xử lý giá trị vô cực (inf) & làm sạch NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().reset_index(drop=True)

    # 13. Lưu kết quả ra file gold_data_processed.csv
    df_clean.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Đã xử lý & nâng cấp dữ liệu thành công!")
    print(f"-> Vị trí file đầu ra:\n   {OUTPUT_FILE}")
    print(f"-> Tổng số dòng hợp lệ: {len(df_clean)} phiên")
    print(f"-> Tỷ lệ xu hướng tăng (Target = 1): {(df_clean['Target'].mean() * 100):.2f}%")
    print(f"-> Danh sách các cột đặc trưng cung cấp cho TV2 ({len(df_clean.columns)} cột):")
    print(list(df_clean.columns))

if __name__ == '__main__':
    preprocess_gold_data()