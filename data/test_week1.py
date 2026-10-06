import yfinance as yf

print("========================================")
print("   HỆ THỐNG DỰ BÁO GIÁ VÀNG - TUẦN 1   ")
print("========================================")
print("Đang kết nối API để tải dữ liệu giá vàng thế giới (GC=F)...")

# Tải dữ liệu giá vàng 1 tháng gần nhất
data = yf.download('GC=F', period='1mo')

print("\n--> Dữ liệu 5 ngày giao dịch gần nhất:")
print(data.tail())

print("\n[SUCCESS] Môi trường Python & API yfinance đã hoạt động hoàn hảo!")