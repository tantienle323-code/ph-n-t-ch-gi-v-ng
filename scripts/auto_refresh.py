"""Automatic market refresh worker.

Run once (for Task Scheduler/cron): python scripts/auto_refresh.py --once
Run continuously: python scripts/auto_refresh.py --interval-minutes 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auto_refresh import run_refresh_if_due, read_refresh_status
from src.refresh_pipeline import refresh_market_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tự động cập nhật dữ liệu vàng."
    )
    parser.add_argument("--interval-minutes", type=int, default=60)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Chạy một lần nếu đã đến hạn.",
    )
    args = parser.parse_args()

    if args.interval_minutes < 1:
        parser.error("--interval-minutes phải lớn hơn hoặc bằng 1")

    print(
        "Gold Market Desk auto-refresh | "
        f"interval={args.interval_minutes} phút"
    )

    while True:
        status = run_refresh_if_due(
            ROOT,
            lambda: refresh_market_data(ROOT),
            interval_minutes=args.interval_minutes,
        )

        print(
            f"[{status.get('status', 'unknown')}] "
            f"{status.get('message', '')}"
        )

        if args.once:
            return (
                0
                if status.get("status") in {"success", "no_change"}
                else 1
            )

        time.sleep(min(args.interval_minutes, 5) * 60)


if __name__ == "__main__":
    raise SystemExit(main())