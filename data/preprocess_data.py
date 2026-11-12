import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.atomic_files import atomic_output_path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "gold_data_5y.csv",
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "gold_data_processed.csv",
)


def preprocess_gold_frame(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    # 1. Đặc trưng lịch.
    df["DayOfWeek"] = df["Date"].dt.day_of_week
    df["Month"] = df["Date"].dt.month

    # 2. Biến động giá hằng ngày.
    df["Return"] = df["Close"].pct_change()
    df["Daily_Return_Pct"] = df["Return"] * 100

    df["Daily_Range_USD"] = (
        df["High"] - df["Low"]
    )

    df["Daily_Range_Pct"] = (
        df["Daily_Range_USD"]
        / df["Low"].replace(0, np.nan)
    ) * 100

    # 3. Quan hệ giữa vàng và các tài sản vĩ mô.
    if (
        "Crude_Oil" in df.columns
        and "USD_Index" in df.columns
    ):
        df["Gold_Oil_Ratio"] = (
            df["Close"] / df["Crude_Oil"]
        )

        df["Gold_DXY_Ratio"] = (
            df["Close"] / df["USD_Index"]
        )

        df["Corr_Gold_DXY_30d"] = (
            df["Close"]
            .rolling(30)
            .corr(df["USD_Index"])
        )

    if (
        "US10Y_Yield" in df.columns
        and "Crude_Oil" in df.columns
    ):
        df["Real_Yield_Proxy"] = (
            df["US10Y_Yield"]
            - df["Crude_Oil"].pct_change(20) * 100
        )

    # 4. Độ biến động và ATR.
    df["Volatility_20_Ann"] = (
        df["Return"].rolling(20).std()
        * np.sqrt(252)
        * 100
    )

    high_low = df["High"] - df["Low"]
    high_close = np.abs(
        df["High"] - df["Close"].shift(1)
    )
    low_close = np.abs(
        df["Low"] - df["Close"].shift(1)
    )

    true_range = pd.concat(
        [high_low, high_close, low_close],
        axis=1,
    ).max(axis=1)

    df["ATR_14"] = true_range.rolling(14).mean()

    df["ATR_Pct"] = (
        df["ATR_14"] / df["Close"]
    ) * 100

    df["Volatility_Spike_Flag"] = (
        df["ATR_14"]
        > 1.8 * df["ATR_14"].rolling(50).mean()
    ).astype(int)

    # 5. Các chỉ báo kỹ thuật.
    df["SMA_7"] = df["Close"].rolling(7).mean()
    df["SMA_14"] = df["Close"].rolling(14).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()

    gain = (
        delta.where(delta > 0, 0)
        .rolling(14)
        .mean()
    )

    loss = (
        -delta.where(delta < 0, 0)
        .rolling(14)
        .mean()
    )

    rs = gain / loss

    df["RSI_14"] = (
        100 - (100 / (1 + rs))
    )

    df["EMA_12"] = df["Close"].ewm(
        span=12,
        adjust=False,
    ).mean()

    df["EMA_26"] = df["Close"].ewm(
        span=26,
        adjust=False,
    ).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False,
    ).mean()

    sma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()

    df["BB_Upper"] = sma_20 + (std_20 * 2)
    df["BB_Lower"] = sma_20 - (std_20 * 2)

    df["BB_Width"] = (
        (df["BB_Upper"] - df["BB_Lower"])
        / sma_20
    ) * 100

    low_14 = df["Low"].rolling(14).min()
    high_14 = df["High"].rolling(14).max()

    df["Stoch_K"] = (
        100
        * (
            (df["Close"] - low_14)
            / (high_14 - low_14)
        )
    )

    df["Stoch_D"] = (
        df["Stoch_K"].rolling(3).mean()
    )

    # 6. Đặc trưng trễ.
    for lag in [1, 2, 3, 5]:
        df[f"Return_Lag{lag}"] = (
            df["Return"].shift(lag)
        )

    macro_cols = [
        "USD_Index",
        "US10Y_Yield",
        "Crude_Oil",
        "SP500",
    ]

    for col in macro_cols:
        if col in df.columns:
            df[f"{col}_Return"] = (
                df[col].pct_change()
            )

            df[f"{col}_Lag1"] = (
                df[f"{col}_Return"].shift(1)
            )

    # Nhãn: giá đóng cửa phiên tiếp theo tăng hay không.
    next_close = df["Close"].shift(-1)

    df["Target"] = (
        next_close > df["Close"]
    ).where(next_close.notna()).astype(float)

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_columns = [
        column
        for column in df.columns
        if column != "Target"
    ]

    df_clean = (
        df.dropna(subset=feature_columns)
        .reset_index(drop=True)
    )

    if df_clean.empty:
        return (
            df.dropna(subset=["Close"])
            .reset_index(drop=True)
        )

    return df_clean


def preprocess_gold_data(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
):
    print("==================================================")
    print("  PREPROCESSING: MACRO FEATURES AND RISK INDICATORS")
    print("==================================================")

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu nguồn: {input_file}"
        )

    df = pd.read_csv(input_file)
    df_clean = preprocess_gold_frame(df)

    if df_clean.empty:
        raise ValueError(
            "Tiền xử lý không tạo được dữ liệu hợp lệ; "
            "file kết quả cũ được giữ nguyên."
        )

    with atomic_output_path(output_file) as temporary_file:
        df_clean.to_csv(
            temporary_file,
            index=False,
        )

    print(
        "Preprocessing complete. "
        f"Created {len(df_clean.columns)} technical and macro features."
    )

    return output_file


if __name__ == "__main__":
    preprocess_gold_data()