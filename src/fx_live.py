"""Lấy tỷ giá USD/VND trực tiếp cho khối so sánh ở Tab 4.

Nguồn chính: Yahoo Finance (mã USDVND=X) qua Chart API — cùng
cơ chế với src/live_price.py, tỷ giá thị trường cập nhật trong
ngày. Dự phòng 1: Vietcombank (XML công khai, lấy giá mua
chuyển khoản). Dự phòng 2: open.er-api.com (tỷ giá ngày, miễn
phí, không cần khóa).

Mọi hàm đều không ném ngoại lệ ra ngoài: lấy không được ở cả
ba nguồn thì trả None để giao diện lùi về nhập tỷ giá tay.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
YAHOO_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/USDVND%3DX"
)
VCB_XML_URL = (
    "https://portal.vietcombank.com.vn/Usercontrols/"
    "TVPortal.TyGia/pXML.aspx"
)
ER_API_URL = "https://open.er-api.com/v6/latest/USD"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gold-dashboard/1.0)"}
TIMEOUT = 10


def _from_yahoo():
    resp = requests.get(
        YAHOO_CHART_URL,
        params={"range": "1d", "interval": "5m"},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    meta = resp.json()["chart"]["result"][0]["meta"]
    rate = float(meta["regularMarketPrice"])
    ts = meta.get("regularMarketTime")
    quote_time = (
        datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(VN_TZ)
        if ts
        else None
    )
    return {
        "rate": rate,
        "source": "Yahoo Finance (USDVND=X)",
        "quote_time": quote_time,
    }


def _from_vietcombank():
    resp = requests.get(VCB_XML_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    for node in root.iter("Exrate"):
        if node.get("CurrencyCode") == "USD":
            rate = float(node.get("Transfer").replace(",", ""))
            quote_time = None
            dt_text = root.findtext("DateTime")
            if dt_text:
                try:
                    quote_time = datetime.strptime(
                        dt_text.strip(), "%m/%d/%Y %I:%M:%S %p"
                    ).replace(tzinfo=VN_TZ)
                except ValueError:
                    quote_time = None
            return {
                "rate": rate,
                "source": "Vietcombank (mua chuyển khoản)",
                "quote_time": quote_time,
            }
    raise ValueError("Không thấy USD trong bảng tỷ giá Vietcombank")


def _from_er_api():
    resp = requests.get(ER_API_URL, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    rate = float(resp.json()["rates"]["VND"])
    return {
        "rate": rate,
        "source": "ExchangeRate-API (tỷ giá ngày)",
        "quote_time": None,
    }


def fetch_live_usd_vnd():
    """Trả dict {rate, source, quote_time, fetched_at} hoặc None."""
    for getter in (_from_yahoo, _from_vietcombank, _from_er_api):
        try:
            result = getter()
        except Exception:
            continue
        if result and result["rate"] > 0:
            result["fetched_at"] = datetime.now(VN_TZ)
            return result
    return None
