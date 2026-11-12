"""Safe, persistent scheduling for automatic market-data refreshes."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Any

from src.atomic_files import atomic_output_path
from src.market_refresh import market_data_refresh_lock

DEFAULT_REFRESH_INTERVAL_MINUTES = 24 * 60


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed.astimezone(timezone.utc)

    except (TypeError, ValueError):
        return None


def read_refresh_status(
    status_path: str | Path,
) -> dict[str, Any]:
    path = Path(status_path)

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
        return value if isinstance(value, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def _write_status(
    status_path: str | Path,
    status: dict[str, Any],
) -> None:
    with atomic_output_path(status_path) as temporary_path:
        Path(temporary_path).write_text(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def refresh_due(
    status: dict[str, Any],
    interval_minutes: int,
    now: datetime | None = None,
) -> bool:
    """Kiểm tra đã đến hạn cập nhật dữ liệu chưa."""
    if interval_minutes < 1:
        raise ValueError(
            "interval_minutes phải lớn hơn hoặc bằng 1."
        )

    now = now or _now()

    retry_after = _parse_time(
        status.get("retry_after_utc")
    )

    if (
        status.get("status") == "error"
        and retry_after is not None
    ):
        return now >= retry_after

    last_attempt = _parse_time(
        status.get("last_attempt_utc")
    )

    if last_attempt is None:
        return True

    return (
        now - last_attempt
        >= timedelta(minutes=interval_minutes)
    )


def run_refresh_if_due(
    project_root: str | Path,
    refresh_callback: Callable[
        [], dict[str, Any] | None
    ],
    interval_minutes: int = DEFAULT_REFRESH_INTERVAL_MINUTES,
    force: bool = False,
) -> dict[str, Any]:
    """Chỉ cập nhật khi đến hạn, dùng khóa chung giữa các tiến trình.

    Nếu cập nhật thất bại, thử lại sau tối đa 10 phút.
    """
    root = Path(project_root)
    data_dir = root / "data"

    status_path = data_dir / "auto_refresh_status.json"
    lock_path = data_dir / ".market_refresh.lock"

    data_dir.mkdir(parents=True, exist_ok=True)

    current = read_refresh_status(status_path)

    if not force and not refresh_due(
        current,
        interval_minutes,
    ):
        return current

    # Dùng chung khóa để giao diện và tiến trình nền
    # không đồng thời tải/xử lý cùng một bộ dữ liệu.
    with market_data_refresh_lock(lock_path):
        current = read_refresh_status(status_path)

        if not force and not refresh_due(
            current,
            interval_minutes,
        ):
            return current

        attempted_at = _now().isoformat()

        running = {
            **current,
            "status": "running",
            "last_attempt_utc": attempted_at,
            "message": (
                "Đang tải và kiểm tra dữ liệu thị trường."
            ),
        }

        _write_status(status_path, running)

        try:
            result = refresh_callback() or {}

        except Exception as exc:
            failed_at = _now().isoformat()

            failed = {
                **running,
                "status": "error",
                "last_attempt_utc": failed_at,
                "message": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

            # Thử lại sớm hơn thay vì chờ đủ một giờ.
            failed["retry_after_utc"] = (
                _now()
                + timedelta(
                    minutes=min(interval_minutes, 10)
                )
            ).isoformat()

            _write_status(status_path, failed)
            return failed

        finished_at = _now().isoformat()
        changed = bool(result.get("changed", True))

        completed = {
            **running,
            "status": (
                "success" if changed else "no_change"
            ),
            "last_attempt_utc": finished_at,
            "last_success_utc": finished_at,
            "message": result.get(
                "message",
                (
                    "Đã cập nhật dữ liệu và mô hình."
                    if changed
                    else
                    "Nguồn chưa có dữ liệu mới; "
                    "giữ nguyên mô hình hiện tại."
                ),
            ),
            "changed": changed,
        }

        completed.pop("retry_after_utc", None)

        _write_status(status_path, completed)

        return completed