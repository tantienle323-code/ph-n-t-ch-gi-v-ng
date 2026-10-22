
import os
import sys

import pandas as pd

try:
    import yfinance as yf
except ModuleNotFoundError:
    class _YFinanceUnavailable:
        @staticmethod
        def Ticker(_symbol):
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


def fetch_comprehensive_financial_data():
    print("=" * 50)
    print("   FETCHING MULTI-ASSET FINANCIAL DATA")
    print("=" * 50)

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

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="5y").reset_index()

        if (
            data.empty
            or "Date" not in data.columns
            or "Close" not in data.columns
        ):
            raise ValueError(
                f"Không nhận được dữ liệu hợp lệ cho {name} ({symbol}). "
                "File cũ được giữ nguyên."
            )

        data["Date"] = pd.to_datetime(
            data["Date"]
        ).dt.tz_localize(None)

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
                )

            data = data[required_columns]
        else:
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

    # Chỉ điền tiếp giá trị đã biết từ quá khứ.
    merged_df = merged_df.ffill()

    if merged_df.empty or merged_df["Close"].isna().all():
        raise ValueError(
            "Không có dữ liệu giá vàng hợp lệ sau khi ghép dữ liệu."
        )

    os.makedirs(BASE_DIR, exist_ok=True)

    with atomic_output_path(OUTPUT_FILE) as temporary_file:
        merged_df.to_csv(temporary_file, index=False)

    print(f"\nCompleted. Data saved to:\n   {OUTPUT_FILE}")
    print(f"-> Trading sessions: {len(merged_df)} rows")
    print(f"-> Raw columns: {list(merged_df.columns)}")

    return OUTPUT_FILE


if __name__ == "__main__":
    fetch_comprehensive_financial_data()
