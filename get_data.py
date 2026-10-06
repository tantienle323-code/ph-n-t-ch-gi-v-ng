import yfinance as yf
import pandas as pd
import os

print("========================================")
print("   TUẦN 2: THU THẬP DỮ LIỆU GIÁ VÀNG   ")
print("========================================")

# 1. Định nghĩa mã tài sản (GC=F là Hợp đồng tương lai Giá vàng thế giới)
ticker = "GC=F"
print(f"Đang tải dữ liệu {ticker} trong 5 năm gần đây từ Yahoo Finance...")

# Tải dữ liệu 5 năm
df = yf.download(ticker, period="5y")

# Xử lý chuẩn hóa bảng dữ liệu (xóa MultiIndex nếu có)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Đưa cột Date từ dạng Index thành cột dữ liệu chính
df = df.reset_index()

# Chỉ giữ lại 6 cột tiêu chuẩn trong phân tích tài chính
cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
df = df[cols_to_keep]

# 2. Hiển thị thông tin kiểm tra dữ liệu
print("\n--- THÔNG TIN TẬP DỮ LIỆU THU THẬP ĐƯỢC ---")
print(f"- Tổng số ngày giao dịch: {len(df)} dòng")
print(f"- Khoảng thời gian: Từ {df['Date'].min().strftime('%Y-%m-%d')} đến {df['Date'].max().strftime('%Y-%m-%d')}")

print("\n5 dòng dữ liệu đầu tiên (Head):")
print(df.head())

# Kiểm tra xem có giá trị nào bị thiếu (NaN) không
missing_values = df.isnull().sum().sum()
print(f"\n- Tổng số ô bị thiếu dữ liệu (Null/NaN): {missing_values}")

# 3. Lưu tập dữ liệu ra file CSV
output_file = "gold_data_5y.csv"
df.to_csv(output_file, index=False)

print(f"\n[SUCCESS] Đã lưu thành công dữ liệu vào file '{output_file}'!")