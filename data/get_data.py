import os
import pandas as pd
import yfinance as yf

# Tự động định vị đường dẫn tuyệt đối tới thư mục chứa file script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, 'gold_data_5y.csv')

def fetch_gold_and_usd_data():
    print("==================================================")
    print("    TẢI DỮ LIỆU GIÁ VÀNG & CHỈ SỐ USD (DXY)   ")
    print("==================================================")
    
    # 1. Tải giá Vàng (GC=F) 5 năm
    print("-> Đang tải dữ liệu Giá Vàng thế giới (GC=F)...")
    gold_ticker = yf.Ticker("GC=F")
    gold_df = gold_ticker.history(period="5y").reset_index()
    
    # Chuẩn hóa múi giờ cột Date
    if 'Date' in gold_df.columns:
        gold_df['Date'] = pd.to_datetime(gold_df['Date']).dt.tz_localize(None)

    # Giữ các cột dữ liệu thô chuẩn
    cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    gold_df = gold_df[[col for col in cols_to_keep if col in gold_df.columns]]

    # 2. Tải Chỉ số Sức mạnh Đồng USD (DX-Y.NYB) 5 năm (Nâng cấp Vĩ mô)
    print("-> Đang tải dữ liệu Chỉ số USD Index (DX-Y.NYB)...")
    usd_ticker = yf.Ticker("DX-Y.NYB")
    usd_df = usd_ticker.history(period="5y").reset_index()
    
    if 'Date' in usd_df.columns:
        usd_df['Date'] = pd.to_datetime(usd_df['Date']).dt.tz_localize(None)
        
    usd_df = usd_df[['Date', 'Close']].rename(columns={'Close': 'USD_Index'})

    # 3. Gộp 2 bảng dữ liệu Vàng và USD theo Ngày giao dịch
    print("-> Đang gộp dữ liệu Vàng và USD...")
    merged_df = pd.merge(gold_df, usd_df, on='Date', how='inner')
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    # 4. Lưu ra file gold_data_5y.csv
    os.makedirs(BASE_DIR, exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Đã hoàn thành! Dữ liệu đã lưu tại:\n   {OUTPUT_FILE}")
    print(f"-> Tổng số phiên giao dịch: {len(merged_df)} dòng")
    print(f"-> Danh sách cột thô: {list(merged_df.columns)}")

if __name__ == '__main__':
    fetch_gold_and_usd_data()