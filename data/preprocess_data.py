import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'gold_data_5y.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'gold_data_processed.csv')

def preprocess_gold_data():
    print("==================================================")
    print("  TIỀN XỬ LÝ: TƯƠNG QUAN VĨ MÔ & BẢO VỆ RỦI RO 3 TẦNG ")
    print("==================================================")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Lỗi: Không tìm thấy file '{INPUT_FILE}'!")
        return

    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # 1. Calendar Features
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    # 2. Daily Fluctuation Metrics
    df['Return'] = df['Close'].pct_change()
    df['Daily_Return_Pct'] = df['Return'] * 100
    df['Daily_Range_USD'] = df['High'] - df['Low']
    df['Daily_Range_Pct'] = (df['Daily_Range_USD'] / df['Low']) * 100

    # 3. CHUỖI TƯƠNG QUAN VĨ MÔ NÂNG CAO (MACRO CROSS-ASSET)
    if 'Crude_Oil' in df.columns and 'USD_Index' in df.columns:
        df['Gold_Oil_Ratio'] = df['Close'] / df['Crude_Oil']
        df['Gold_DXY_Ratio'] = df['Close'] / df['USD_Index']
        df['Corr_Gold_DXY_30d'] = df['Close'].rolling(30).corr(df['USD_Index'])
        
    if 'US10Y_Yield' in df.columns and 'Crude_Oil' in df.columns:
        df['Real_Yield_Proxy'] = df['US10Y_Yield'] - df['Crude_Oil'].pct_change(20)*100

    # 4. TẦNG 1: BIẾN CHẨN ĐOÁN THIÊN NGA ĐEN / BÃO GIÁ (CRISIS & VOLATILITY SPIKE)
    df['Volatility_20_Ann'] = df['Return'].rolling(20).std() * np.sqrt(252) * 100
    
    # ATR & ATR Spike Detection
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()
    df['ATR_Pct'] = (df['ATR_14'] / df['Close']) * 100
    
    # Flag biến động bất thường (ATR bùng nổ > 1.8x trung bình 50 phiên)
    df['Volatility_Spike_Flag'] = (df['ATR_14'] > 1.8 * df['ATR_14'].rolling(50).mean()).astype(int)

    # 5. Technical Indicators (SMA, EMA, RSI, MACD, BB, Stochastic)
    df['SMA_7'] = df['Close'].rolling(7).mean()
    df['SMA_14'] = df['Close'].rolling(14).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    sma_20 = df['Close'].rolling(20).mean()
    std_20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / sma_20) * 100

    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # 6. Lag Features
    for lag in [1, 2, 3, 5]:
        df[f'Return_Lag{lag}'] = df['Return'].shift(lag)

    # Macro Returns
    macro_cols = ['USD_Index', 'US10Y_Yield', 'Crude_Oil', 'SP500']
    for col in macro_cols:
        if col in df.columns:
            df[f'{col}_Return'] = df[col].pct_change()
            df[f'{col}_Lag1'] = df[f'{col}_Return'].shift(1)

    # Target
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df.dropna().reset_index(drop=True)

    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Đã tiền xử lý thành công! Đã tạo {len(df_clean.columns)} đặc trưng kỹ thuật & vĩ mô.")

if __name__ == '__main__':
    preprocess_gold_data()