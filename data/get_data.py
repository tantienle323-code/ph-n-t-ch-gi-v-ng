import os
import sys
import time

import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:
    # Giúp các bài kiểm thử vẫn import được khi chưa cài yfinance.
    class _YFinanceUnavailable:
        @staticmethod
        def Ticker(_symbol):
            raise RuntimeError(
                "Thiếu thư viện yfinance. Hãy cài dependencies bằng: "
                "python -m pip install -r requirements.txt"
            )

        @staticmethod
        def download(_symbol, **_kwargs):
            raise RuntimeError(
                "Thiếu thư viện yfinance. Hãy cài dependencies bằng: "
                "python -m pip install -r requirements.txt"
            )

    yf = _YFinanceUnavailable()

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.atomic_files import atomic_output_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "gold_data_5y.csv")

# Yahoo Finance thỉnh thoảng timeout hoặc trả về dữ liệu rỗng
# trong chốc lát (lỗi mạng, giới hạn truy cập, lấy cookie/crumb
# thất bại). Trước đây code chỉ tải đúng 1 lần rồi bỏ cuộc ngay,
# nên một lần trượt mạng là cả lần cập nhật bị báo thất bại.
# Nay tải lại tối đa 4 lần, chờ tăng dần giữa các lần thử,
# và nếu Ticker.history vẫn rỗng thì thử tiếp bằng yf.download.
MAX_ATTEMPTS = 4
RETRY_DELAYS_SECONDS = (2, 5, 10)


def _normalize_history(data):
    """Đưa dữ liệu Yahoo về dạng bảng có cột Date và Close."""
    if data is None or len(data) == 0:
        return None

    data = data.copy()

    # yf.download có thể trả cột MultiIndex (Price, Ticker);
    # chỉ giữ tầng tên cột đầu tiên: Open, High, Low, Close...
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()

    # Tên cột ngày có thể là "Date", "Datetime" hoặc "index".
    if "Date" not in data.columns:
        for candidate in ("Datetime", "index"):
            if candidate in data.columns:
                data = data.rename(columns={candidate: "Date"})
                break

    if (
        "Date" not in data.columns
        or "Close" not in data.columns
        or data["Close"].isna().all()
    ):
        return None

    return data


def _download_once(symbol):
    """Tải một mã một lần: thử history trước, download sau."""
    try:
        data = _normalize_history(
            yf.Ticker(symbol).history(period="5y")
        )
        if data is not None:
            return data
    except Exception:
        pass

    try:
        data = _normalize_history(
            yf.download(
                symbol,
                period="5y",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        )
        if data is not None:
            return data
    except Exception:
        pass

    return None


def fetch_ticker_history(name, symbol):
    """Tải lịch sử giá của một mã, tự thử lại khi Yahoo trượt.

    Chỉ khi đã thử đủ số lần mà vẫn không có dữ liệu hợp lệ
    mới báo lỗi và giữ nguyên file cũ như trước.
    """
    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            data = _download_once(symbol)
        except Exception as error:
            data = None
            last_error = error

        if data is not None:
            if attempt > 1:
                print(
                    f"   -> {name} ({symbol}): thành công "
                    f"ở lần thử {attempt}."
                )
            return data

        if attempt < MAX_ATTEMPTS:
            delay = RETRY_DELAYS_SECONDS[
                min(attempt - 1, len(RETRY_DELAYS_SECONDS) - 1)
            ]
            print(
                f"   -> {name} ({symbol}): lần thử {attempt} "
                f"không có dữ liệu, thử lại sau {delay} giây..."
            )
            time.sleep(delay)

    message = (
        f"Không nhận được dữ liệu hợp lệ cho "
        f"{name} ({symbol}) sau {MAX_ATTEMPTS} lần thử. "
        f"File cũ được giữ nguyên."
    )
    if last_error is not None:
        message += f" Lỗi cuối cùng: {last_error}"
    raise ValueError(message)


def fetch_comprehensive_financial_data():
    print("==================================================")
    print("   FETCHING MULTI-ASSET FINANCIAL DATA")
    print("==================================================")

    # Các mã tài chính và chỉ số vĩ mô toàn cầu.
    tickers = {
        "Gold": "GC=F",
        "USD_Index": "DX-Y.NYB",
        "US10Y_Yield": "^TNX",
        "Crude_Oil": "CL=F",
        "SP500": "^GSPC",
    }

    df_list = []

    for name, symbol in tickers.items():
        print(f"-> Downloading {name} ({symbol})...")

        data = fetch_ticker_history(name, symbol)

        data["Date"] = (
            pd.to_datetime(data["Date"])
            .dt.tz_localize(None)
        )

        if name == "Gold":
            required_columns = [
                "Date",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Dữ liệu vàng thiếu các cột: "
                    + ", ".join(missing_columns)
                    + "."
                )

            data = data[required_columns]

        else:
            # Với các tài sản vĩ mô, lấy giá đóng cửa.
            data = data[["Date", "Close"]].rename(
                columns={"Close": name}
            )

        df_list.append(data)

    print("-> Merging macroeconomic asset data...")

    merged_df = df_list[0]

    for df_sub in df_list[1:]:
        merged_df = pd.merge(
            merged_df,
            df_sub,
            on="Date",
            how="left",
        )

    merged_df = (
        merged_df.sort_values("Date")
        .reset_index(drop=True)
    )

    # Chỉ điền tiếp giá trị đã biết từ quá khứ,
    # không lấy dữ liệu tương lai điền ngược về trước.
    merged_df = merged_df.ffill()

    if (
        merged_df.empty
        or merged_df["Close"].isna().all()
    ):
        raise ValueError(
            "Không có dữ liệu giá vàng hợp lệ sau khi ghép dữ liệu."
        )

    os.makedirs(BASE_DIR, exist_ok=True)

    # Ghi file theo cách nguyên tử để tránh file bị ghi dở.
    with atomic_output_path(OUTPUT_FILE) as temporary_file:
        merged_df.to_csv(
            temporary_file,
            index=False,
        )

    print(f"\nCompleted. Data saved to:\n   {OUTPUT_FILE}")
    print(f"-> Trading sessions: {len(merged_df)} rows")
    print(f"-> Raw columns: {list(merged_df.columns)}")

    return OUTPUT_FILE


if __name__ == "__main__":
    fetch_comprehensive_financial_data()
