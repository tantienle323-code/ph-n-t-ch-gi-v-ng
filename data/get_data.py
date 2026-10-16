import os
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'gold_data_5y.csv')

def fetch_comprehensive_financial_data():
    print("==================================================")
    print("   FETCHING MULTI-ASSET FINANCIAL DATA")
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
        print(f"-> Downloading {name} ({symbol})...")
        t = yf.Ticker(symbol)
        data = t.history(period="5y").reset_index()

        if data.empty or 'Date' not in data.columns or 'Close' not in data.columns:
            raise ValueError(f"Không nhận được dữ liệu hợp lệ cho {name} ({symbol}). File cũ được giữ nguyên.")

        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)

        if name == 'Gold':
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [column for column in required_columns if column not in data.columns]
            if missing_columns:
                raise ValueError(f"Dữ liệu vàng thiếu các cột: {', '.join(missing_columns)}.")
            data = data[required_columns]
        else:
            # Các tài sản vĩ mô lấy giá Close
            data = data[['Date', 'Close']].rename(columns={'Close': name})
            
        df_list.append(data)
        
    print("-> Merging macroeconomic asset data...")
    merged_df = df_list[0]
    for df_sub in df_list[1:]:
        merged_df = pd.merge(merged_df, df_sub, on='Date', how='left')
        
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    # Carry the most recent known macro value forward; never backfill from future dates.
    merged_df = merged_df.ffill()
    if merged_df.empty or merged_df['Close'].isna().all():
        raise ValueError("Không có dữ liệu giá vàng hợp lệ sau khi ghép dữ liệu.")

    os.makedirs(BASE_DIR, exist_ok=True)
    temporary_file = OUTPUT_FILE + '.tmp'
    merged_df.to_csv(temporary_file, index=False)
    os.replace(temporary_file, OUTPUT_FILE)
    
    print(f"\nCompleted. Data saved to:\n   {OUTPUT_FILE}")
    print(f"-> Trading sessions: {len(merged_df)} rows")
    print(f"-> Raw columns: {list(merged_df.columns)}")
    return OUTPUT_FILE

if __name__ == '__main__':
    fetch_comprehensive_financial_data()