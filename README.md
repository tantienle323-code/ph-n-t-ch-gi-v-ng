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
- `src/train.py`: huấn luyện mô hình Random Forest / ensemble.
- `src/telegram_bot.py`: mẫu gửi tín hiệu dự báo qua Telegram.
- `models/`: lưu mô hình đã huấn luyện.
- `data/`: lưu dữ liệu thô và dữ liệu đã xử lý.
- `tests/`: kiểm tra smoke test cơ bản.

## Môi trường cần có

- Python 3.10+
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
```

## Dòng chạy chuẩn

1. Tải dữ liệu gốc:

```powershell
python data/get_data.py
```

2. Tiền xử lý dữ liệu:

```powershell
python data/preprocess_data.py
```

3. Huấn luyện mô hình:

```powershell
python src/train.py
```

4. Khởi động dashboard:

```powershell
streamlit run app.py
```

## Kiểm tra dữ liệu

Sau khi chạy tiền xử lý, file dự kiến được sinh ra:

- `data/gold_data_5y.csv`
- `data/gold_data_processed.csv`

File `gold_data_processed.csv` nên có các cột chính như:
- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `USD_Index`
- `RSI_14`
- `MACD`
- `Target`

## Lưu ý quan trọng

- `Target` là nhãn mục tiêu: `0` = giảm/đứng, `1` = tăng.
- Dữ liệu được vô hiệu hóa nếu file quá cũ hoặc không tương thích với định dạng. Nên chạy lại `get_data.py` trước khi huấn luyện.
- Đối với bot Telegram, cần điền token và chat ID thật trong `src/telegram_bot.py` trước khi gửi thông báo.

## Chạy kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## Kết luận

Dự án này xây dựng một pipeline hoàn chỉnh từ tài liệu dữ liệu đến mô hình dự báo và dashboard trực quan. Nhiệm vụ tiếp theo có thể là nâng cấp mô hình, thêm backtest chuyên nghiệp hơn, và kết nối bot cảnh báo theo thời gian thực.
