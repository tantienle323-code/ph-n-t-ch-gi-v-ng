import os
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'gold_data_5y.csv')

def fetch_comprehensive_financial_data():
    print("==================================================")
    print("   TV1: TẢI DỮ LIỆU ĐA TÀI SẢN & VĨ MÔ TOÀN CẦU   ")
    print("==================================================")
    
    # Danh sách mã tài chính & chỉ số vĩ mô toàn cầu
    tickers = {
        'Gold': 'GC=F',          # Giá Vàng tương lai
        'USD_Index': 'DX-Y.NYB', # Chỉ số USD Index (DXY)
        'US10Y_Yield': '^TNX',   # Lãi suất Trái phiếu Mỹ 10 năm
        'Crude_Oil': 'CL=F',     # Giá Dầu thô WTI
        'SP500': '^GSPC'         # Chỉ số Chứng khoán S&P 500
    }
    
    df_list = []
    
    for name, symbol in tickers.items():
        print(f"-> Đang tải dữ liệu {name} ({symbol})...")
        t = yf.Ticker(symbol)
        data = t.history(period="5y").reset_index()
        
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        
        if name == 'Gold':
            # Giữ đủ các cột OHLCV cho Vàng
            cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            data = data[[c for c in cols if c in data.columns]]
        else:
            # Các tài sản vĩ mô lấy giá Close
            data = data[['Date', 'Close']].rename(columns={'Close': name})
            
        df_list.append(data)
        
    print("-> Đang gộp dữ liệu đa tài sản vĩ mô...")
    merged_df = df_list[0]
    for df_sub in df_list[1:]:
        merged_df = pd.merge(merged_df, df_sub, on='Date', how='left')
        
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)
    
    # Lấp khoảng trống dữ liệu do lệch ngày nghỉ giữa các thị trường
    merged_df = merged_df.ffill().bfill()
    
    os.makedirs(BASE_DIR, exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Đã hoàn thành! Dữ liệu đã lưu tại:\n   {OUTPUT_FILE}")
    print(f"-> Tổng số phiên giao dịch: {len(merged_df)} dòng")
    print(f"-> Danh sách các cột thô: {list(merged_df.columns)}")

if __name__ == '__main__':
    fetch_comprehensive_financial_data()