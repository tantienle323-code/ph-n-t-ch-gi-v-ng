import pandas as pd
import numpy as np

def preprocess_gold_data(input_file='gold_data_5y.csv', output_file='gold_data_processed.csv'):
    print("==========================================")
    print("  TUẦN 3: TIỀN XỬ LÝ & TẠO ĐẶC TRƯNG    ")
    print("==========================================")
    
    # 1. Đọc dữ liệu
    df = pd.read_csv(input_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Tính phần trăm thay đổi giá hàng ngày (Daily Return)
    df['Return'] = df['Close'].pct_change()
    
    # 3. Tạo các đường trung bình động (SMA)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # 4. Độ biến động giá 20 ngày (Volatility)
    df['Volatility_20'] = df['Return'].rolling(window=20).std()
    
    # 5. Chỉ số sức mạnh tương đối RSI (14 ngày)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 6. Tạo biến mục tiêu Target (1 = Ngày mai giá tăng, 0 = Ngày mai giá giảm/bằng)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # 7. Loạt bỏ các dòng chứa NaN phát sinh do cửa sổ tính toán (Rolling Window / Shift)
    df_clean = df.dropna().reset_index(drop=True)
    
    # 8. Lưu kết quả
    df_clean.to_csv(output_file, index=False)
    
    print(f"\n---> Đã lưu dữ liệu làm sạch tại: {output_file}")
    print(f"---> Tổng số quan sát hợp lệ: {len(df_clean)} dòng")
    print(f"---> Tỷ lệ ngày giá tăng (Target = 1): {(df_clean['Target'].mean() * 100):.2f}%")
    print("\n5 dòng dữ liệu đầu tiên sau xử lý:")
    print(df_clean[['Date', 'Close', 'SMA_7', 'RSI_14', 'Target']].head())

if __name__ == '__main__':
    preprocess_gold_data()