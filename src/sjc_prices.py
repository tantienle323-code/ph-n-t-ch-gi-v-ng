
"""Validation and comparison helpers for user-supplied Vietnam SJC quotes.

No unofficial web scraping or synthetic prices are used. Quotes must come from
an explicitly identified source and include a quote date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SJC_COLUMNS = [
    "Date",
    "Buy_VND_per_luong",
    "Sell_VND_per_luong",
    "Source",
]

GRAMS_PER_TROY_OUNCE = 31.1034768
GRAMS_PER_LUONG = 37.5


def validate_sjc_quotes(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate an SJC quote table; return quotes by date."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("File SJC chưa có dữ liệu báo giá.")

    required = {
        "Date",
        "Buy_VND_per_luong",
        "Sell_VND_per_luong",
    }
    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            "Thiếu cột bắt buộc: " + ", ".join(missing)
        )

    result = frame.copy()
    result["Date"] = pd.to_datetime(
        result["Date"], errors="coerce"
    )

    for column in (
        "Buy_VND_per_luong",
        "Sell_VND_per_luong",
    ):
        result[column] = pd.to_numeric(
            result[column], errors="coerce"
        )

    if "Source" not in result.columns:
        result["Source"] = "Nguồn do người dùng cung cấp"

    result["Source"] = (
        result["Source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if result["Date"].isna().any():
        raise ValueError("Có ngày báo giá SJC không hợp lệ.")

    values = result[
        ["Buy_VND_per_luong", "Sell_VND_per_luong"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Giá mua/bán phải là số hữu hạn lớn hơn 0."
        )

    if (
        result["Sell_VND_per_luong"]
        < result["Buy_VND_per_luong"]
    ).any():
        raise ValueError(
            "Giá bán ra không được thấp hơn giá mua vào."
        )

    if (
        result["Date"] > pd.Timestamp.now().normalize()
    ).any():
        raise ValueError(
            "Ngày báo giá không được nằm trong tương lai."
        )

    if result["Source"].eq("").any():
        raise ValueError(
            "Mỗi báo giá cần ghi rõ nguồn tham chiếu."
        )

    result = (
        result[SJC_COLUMNS]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
    )

    return result.reset_index(drop=True)


def international_equivalent_vnd_per_luong(
    gold_usd_per_troy_ounce: float,
    usd_vnd: float,
) -> float:
    """Convert USD per troy ounce to theoretical VND per luong."""
    gold = float(gold_usd_per_troy_ounce)
    fx = float(usd_vnd)

    if not np.isfinite(gold) or gold <= 0:
        raise ValueError(
            "Giá vàng thế giới phải lớn hơn 0."
        )

    if not np.isfinite(fx) or fx <= 0:
        raise ValueError(
            "Tỷ giá USD/VND phải lớn hơn 0."
        )

    return (
        gold
        * fx
        * GRAMS_PER_LUONG
        / GRAMS_PER_TROY_OUNCE
    )


def quote_comparison(
    frame: pd.DataFrame,
    gold_usd_per_ounce: float,
    usd_vnd: float,
) -> dict:
    """Compare the latest SJC quote with a theoretical equivalent."""
    quotes = validate_sjc_quotes(frame)
    latest = quotes.iloc[-1]

    equivalent = international_equivalent_vnd_per_luong(
        gold_usd_per_ounce,
        usd_vnd,
    )

    buy = float(latest["Buy_VND_per_luong"])
    sell = float(latest["Sell_VND_per_luong"])
    midpoint = (buy + sell) / 2

    return {
        "date": latest["Date"],
        "buy": buy,
        "sell": sell,
        "source": str(latest["Source"]),
        "international_equivalent": equivalent,
        "midpoint_premium_vnd": midpoint - equivalent,
        "midpoint_premium_pct": (
            midpoint / equivalent - 1
        ) * 100,
        "spread_vnd": sell - buy,
    }
