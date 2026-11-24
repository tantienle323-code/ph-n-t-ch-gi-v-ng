"""Giá vàng miếng SJC trực tiếp cho Tab 4.

Nguồn chính: API bảng giá công khai của PNJ (mã "SJC" —
Vàng miếng SJC 999.9), cùng hệ nguồn với dữ liệu CSV
lịch sử đang lưu trong ứng dụng. PNJ niêm yết theo
nghìn đồng/chỉ nên module quy đổi về VND/lượng
(1 lượng = 10 chỉ, hệ số nhân 10.000).

Nguồn dự phòng: vang.today (mã SJL1L10 — SJC 9999),
giá ở đó đã là VND/lượng.

Giá SJC là giá niêm yết của doanh nghiệp, chỉ thay đổi
khi doanh nghiệp cập nhật bảng giá (thường vài lần mỗi
ngày) — không phải giá khớp lệnh từng giây như sàn
quốc tế. Vì vậy kết quả luôn kèm nguồn và thời điểm
báo giá để giao diện hiển thị minh bạch.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import requests

_PNJ_URL = (
    "https://edge-api.pnj.io/ecom-frontend/v1/get-gold-price"
)
_VANG_TODAY_URL = "https://vang.today/api/prices"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}
_VN_TZ = ZoneInfo("Asia/Bangkok")


def fetch_live_sjc() -> dict | None:
    """Lấy báo giá SJC mới nhất: PNJ trước, vang.today sau.

    Trả về dict gồm buy, sell, spread (VND/lượng), source,
    quote_time (thời điểm báo giá nếu nguồn công bố, ngược
    lại None) và fetched_at (thời điểm lấy, giờ VN).
    Cả hai nguồn đều không lấy được thì trả None để giao
    diện lùi về dữ liệu CSV như cũ.
    """
    return _fetch_from_pnj() or _fetch_from_vang_today()


def _fetch_from_pnj() -> dict | None:
    try:
        response = requests.get(
            _PNJ_URL, headers=_HEADERS, timeout=15
        )
        response.raise_for_status()
        rows = response.json().get("data") or []
    except Exception:
        return None

    for row in rows:
        if row.get("masp") != "SJC":
            continue

        # PNJ niêm yết nghìn đồng/chỉ:
        # VND/lượng = giá x 1.000 (đồng) x 10 (chỉ/lượng).
        buy = float(row["giamua"]) * 10_000
        sell = float(row["giaban"]) * 10_000
        if buy <= 0 or sell <= 0:
            return None

        return {
            "buy": buy,
            "sell": sell,
            "spread": sell - buy,
            "source": (
                "PNJ — Vàng miếng SJC 999.9"
                " (bảng giá trực tuyến)"
            ),
            "quote_time": None,
            "fetched_at": datetime.now(_VN_TZ),
        }

    return None


def _fetch_from_vang_today() -> dict | None:
    try:
        response = requests.get(
            _VANG_TODAY_URL, headers=_HEADERS, timeout=15
        )
        response.raise_for_status()
        payload = response.json()
        entry = (payload.get("prices") or {}).get("SJL1L10")
        if not entry:
            return None

        buy = float(entry["buy"])
        sell = float(entry["sell"])
        if buy <= 0 or sell <= 0:
            return None

        quote_time = None
        try:
            quote_time = datetime.strptime(
                f"{payload['date']} {payload['time']}",
                "%Y-%m-%d %H:%M",
            ).replace(tzinfo=_VN_TZ)
        except (KeyError, ValueError):
            pass

        return {
            "buy": buy,
            "sell": sell,
            "spread": sell - buy,
            "source": "vang.today — SJC 9999 (tổng hợp)",
            "quote_time": quote_time,
            "fetched_at": datetime.now(_VN_TZ),
        }
    except Exception:
        return None
