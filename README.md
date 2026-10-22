# Hệ thống phân tích & dự báo giá vàng AI

Mô tả ngắn gọn:
- Thu thập dữ liệu giá vàng và các chỉ số vĩ mô quan trọng như USD Index, lãi suất 10Y, dầu thô và S&P 500.
- Tiền xử lý và tạo đặc trưng kỹ thuật, tương quan vĩ mô và nhãn mục tiêu `Target`.
- Huấn luyện mô hình học máy để dự báo xu hướng ngày tiếp theo của giá vàng.
- Hiển thị báo cáo, biểu đồ và mô phỏng backtest bằng Streamlit.

## Cấu trúc dự án

- `app.py`: giao diện dashboard Streamlit.
- `data/get_data.py`: tải dữ liệu từ Yahoo Finance.
- `data/preprocess_data.py`: tạo các chỉ báo kỹ thuật và tính nhãn mục tiêu.
- `src/train.py`: chọn và huấn luyện mô hình bằng chia dữ liệu theo thời gian.
- `src/forecast_tracking.py`: lưu tín hiệu dự báo và đối chiếu với kết quả thực tế.
- `src/telegram_bot.py`: gửi tín hiệu dự báo qua Telegram.
- `src/sjc_prices.py`: kiểm tra báo giá SJC do người dùng cung cấp và quy đổi tham khảo vàng quốc tế.
- `src/backtest.py`: mô phỏng chiến lược có phí giao dịch và trượt giá.
- `models/`: lưu mô hình đã huấn luyện.
- `data/`: lưu dữ liệu thô và dữ liệu đã xử lý.
- `tests/`: kiểm tra smoke test cơ bản.

## Môi trường cần có

- Python 3.11+ (khuyến nghị dùng đúng phiên bản thư viện trong `requirements.txt`)
- Virtual environment (khuyến nghị)
- Thư viện: pandas, numpy, scikit-learn, plotly, streamlit, yfinance, requests

## Cài đặt

PowerShell:

```powershell
cd "D:\doan phan tich gia vang nhom 2\phan_tich_gia_vang"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt