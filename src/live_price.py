"""Giá vàng trực tiếp (intraday) cho dải hiển thị ở Tab 1.

Dữ liệu theo ngày (gold_data_5y.csv) chỉ thay đổi khi có
phiên mới, nên Tab 1 cần một nguồn riêng nhẹ hơn để hiện
giá gần thời gian thực: nến 1 phút của GC=F lấy qua Yahoo
Chart API (không cần cookie/crumb như yfinance).

Lưu ý: dữ liệu Yahoo miễn phí vốn đã trễ khoảng 10 phút
so với sàn COMEX, nên đây là "gần thời gian thực" —
giao diện phải ghi rõ điều này cho người xem.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}
_VN_TZ = ZoneInfo("Asia/Bangkok")


def fetch_live_gold() -> dict | None:
    """Lấy giá GC=F mới nhất theo phút.

    Trả về dict gồm giá hiện tại, giá đóng cửa phiên trước,
    thay đổi tuyệt đối/phần trăm, thời điểm báo giá (giờ VN),
    chuỗi giá trong ngày để vẽ biểu đồ nhỏ và tên nguồn.
    Không lấy được thì trả None để giao diện tự lùi về
    hiển thị giá phiên gần nhất như cũ.
    """
    try:
        response = requests.get(
            _CHART_URL,
            params={"range": "1d", "interval": "1m"},
            headers=_HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
    except Exception:
        return _fetch_live_gold_yfinance()

    meta = result.get("meta", {})
    quote = result["indicators"]["quote"][0]

    points = [
        (ts, close)
        for ts, close in zip(
            result.get("timestamp") or [],
            quote.get("close") or [],
        )
        if close is not None
    ]

    if not points and meta.get("regularMarketPrice") is None:
        return None

    series = pd.DataFrame(points, columns=["timestamp", "Close"])
    series["Time"] = (
        pd.to_datetime(series["timestamp"], unit="s", utc=True)
        .dt.tz_convert(_VN_TZ)
    )

    price = float(meta.get("regularMarketPrice") or points[-1][1])
    previous_close = float(meta.get("chartPreviousClose") or price)

    market_time = meta.get("regularMarketTime")
    if market_time:
        quote_time = datetime.fromtimestamp(
            market_time, tz=timezone.utc
        ).astimezone(_VN_TZ)
    elif points:
        quote_time = series["Time"].iloc[-1].to_pydatetime()
    else:
        quote_time = None

    return {
        "price": price,
        "previous_close": previous_close,
        "change": price - previous_close,
        "change_pct": (
            (price - previous_close) / previous_close * 100
            if previous_close
            else 0.0
        ),
        "quote_time": quote_time,
        "series": series[["Time", "Close"]],
        "source": "Yahoo Chart API (nến 1 phút)",
    }


def _fetch_live_gold_yfinance() -> dict | None:
    """Dự phòng bằng yfinance khi Chart API không vào được."""
    try:
        import yfinance as yf

        frame = yf.Ticker("GC=F").history(
            period="1d", interval="1m"
        )
    except Exception:
        return None

    if frame is None or frame.empty or "Close" not in frame:
        return None

    frame = frame.dropna(subset=["Close"])
    if frame.empty:
        return None

    series = pd.DataFrame(
        {
            "Time": frame.index.tz_convert(_VN_TZ),
            "Close": frame["Close"].to_numpy(),
        }
    )

    price = float(frame["Close"].iloc[-1])
    reference = float(frame["Close"].iloc[0])

    return {
        "price": price,
        "previous_close": reference,
        "change": price - reference,
        "change_pct": (
            (price - reference) / reference * 100
            if reference
            else 0.0
        ),
        "quote_time": series["Time"].iloc[-1].to_pydatetime(),
        "series": series,
        "source": "yfinance (nến 1 phút)",
    }
