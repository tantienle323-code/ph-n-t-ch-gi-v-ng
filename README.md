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
- `src/backtest.py`: mô phỏng chiến lược có phí giao dịch và trượt giá.
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
- Dashboard hiển thị ngày phiên mới nhất, độ trễ theo ngày làm việc và kiểm tra ngày trùng, giá đóng cửa/nhãn lỗi, ngày tương lai, cột bắt buộc. Dữ liệu trễ hơn 5 ngày làm việc hoặc có lỗi được cảnh báo; tín hiệu bị ẩn nếu phát hiện lỗi dữ liệu. Nút **Cập nhật dữ liệu thị trường** tải dữ liệu Yahoo Finance và chạy lại tiền xử lý/huấn luyện; các lần cập nhật từ nhiều dashboard được xếp hàng bằng khóa tiến trình, file kết quả được thay thế nguyên tử và dữ liệu cũ không bị ghi đè nếu nguồn tải rỗng/không hợp lệ. Bản dữ liệu được làm mới cache tối đa mỗi 5 phút.
- Tin tức được lấy trực tiếp từ Google News RSS, không cần API key. Dashboard hiển thị tối đa 10 tin mới nhất, đổi thời gian xuất bản sang giờ Việt Nam (ICT, UTC+7), bỏ qua tiêu đề ghi ngày trong tương lai theo giờ Việt Nam, dịch tiêu đề sang tiếng Việt qua MyMemory và giữ nguyên tiêu đề gốc để đối chiếu; liên kết mở bài gốc, bản dịch được lưu cache tối đa 24 giờ, tin RSS tối đa 15 phút và có nút làm mới. Nếu dịch vụ dịch không khả dụng, giao diện báo rõ và hiển thị tiêu đề gốc. Điểm cảm xúc chỉ phân tích từ khóa trên tiêu đề gốc, không đánh giá nội dung bài viết.
- Quy trình kiểm định mô hình chia chuỗi theo thời gian thành train (70%), validation (15%) và test (15%). Mô hình được chọn bằng F1 trên validation; test được dự báo theo walk-forward từng khối tối đa 20 phiên, chỉ huấn luyện trên nhãn đã biết trước mỗi khối. Báo cáo test được đối chiếu với baseline luôn dự báo tăng và baseline theo hướng phiên trước. Dashboard backtest dùng chính dự báo ngoài mẫu này. Chạy lại `python src/train.py` sau khi cập nhật dữ liệu để tạo lại mô hình và kết quả kiểm định.
- Dashboard ghi lại hướng dự báo, lợi suất dự báo và giá ước tính vào `data/forecast_history.sqlite3`; khi giá phiên kế tiếp có sẵn, tự chấm hướng, lợi suất, giá thực tế và sai số. Tab **Chất lượng mô hình** lọc được 30/90 ngày hoặc toàn bộ lịch sử, so sánh sai số hồi quy với baseline 0%, hiển thị chuỗi dự báo-thực tế và tải CSV. Dữ liệu cũ được bổ sung cột khi cần, không xóa lịch sử. Lưu trữ cục bộ trên máy chạy ứng dụng; không đồng bộ và không gắn với tài khoản người dùng.
- Ngoài hướng tăng/giảm, mô hình hồi quy ước lượng phần trăm lợi suất và giá đóng cửa phiên tới. Khoảng giá tham khảo lấy từ phân vị 90% sai số tuyệt đối trên validation; dashboard báo riêng độ phủ quan sát được trên test walk-forward. Đây là khoảng ước lượng thực nghiệm, không phải bảo đảm xác suất. Độ tin cậy phân lớp lấy từ `predict_proba` và chưa được hiệu chuẩn.
- Dòng dữ liệu mới nhất chưa có giá đóng cửa phiên kế tiếp nên `Target` để trống, không gán giả là giảm; dòng này vẫn được giữ để tạo tín hiệu dự báo phiên tới. Dữ liệu vĩ mô được forward-fill theo thời gian và không backfill từ ngày tương lai.
- Backtest giả định lệnh mua theo tín hiệu được giữ tối đa một phiên, giới hạn lợi nhuận/thua lỗ theo stop-loss/take-profit, và áp dụng phí/trượt giá mỗi chiều theo bps (basis point; 1 bps = 0,01%). Đây là mô phỏng lịch sử, không cam kết lợi nhuận tương lai.
- CSV dữ liệu đang xem và kết quả backtest có thể tải trực tiếp từ dashboard.
- Để gửi tín hiệu Telegram, tạo bot/chat rồi đặt `TELEGRAM_BOT_TOKEN` và `TELEGRAM_CHAT_ID` trong môi trường chạy Streamlit. Không lưu token trong mã nguồn hoặc repository. Ví dụ PowerShell:

```powershell
$env:TELEGRAM_BOT_TOKEN = "token-cua-ban"
$env:TELEGRAM_CHAT_ID = "chat-id-cua-ban"
streamlit run app.py
```

- Trong sidebar, có thể lọc Telegram theo hướng và đặt ngưỡng độ tin cậy tối thiểu; ứng dụng chỉ gửi khi người dùng bấm nút, không tự động phát tin. Điểm `predict_proba` chưa hiệu chuẩn nên ngưỡng này chỉ là bộ lọc, không phải xác suất đúng đã được đảm bảo.

## Chạy kiểm thử

```powershell
python -m unittest discover -s tests -v
```

## Kết luận

Dự án này xây dựng một pipeline hoàn chỉnh từ tài liệu dữ liệu đến mô hình dự báo và dashboard trực quan. Nhiệm vụ tiếp theo có thể là nâng cấp mô hình, thêm backtest chuyên nghiệp hơn, và kết nối bot cảnh báo theo thời gian thực.
