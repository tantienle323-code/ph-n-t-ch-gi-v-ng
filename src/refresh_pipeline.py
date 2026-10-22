"""Shared market refresh pipeline used by the dashboard and scheduler."""
from __future__ import annotations

import hashlib
from pathlib import Path


def _file_digest(path: Path) -> str | None:
    """Tạo mã SHA-256 để nhận biết nội dung file có thay đổi không."""
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def refresh_market_data(
    project_root: str | Path,
) -> dict[str, object]:
    """Tải dữ liệu thị trường và cập nhật các tệp dẫn xuất."""
    root = Path(project_root)

    raw_path = root / "data" / "gold_data_5y.csv"
    processed_digest_path = (
        root / "data" / ".processed_source_sha256"
    )

    try:
        processed_digest = (
            processed_digest_path.read_text(
                encoding="utf-8"
            ).strip()
        )
    except OSError:
        processed_digest = ""

    # Import trễ để tránh tải dữ liệu khi chỉ khởi động giao diện.
    from data.get_data import (
        fetch_comprehensive_financial_data,
    )

    fetch_comprehensive_financial_data()

    after_digest = _file_digest(raw_path)

    if after_digest is None:
        raise ValueError(
            "Nguồn tải về không tạo được tệp dữ liệu thô."
        )

    # Nếu dữ liệu không thay đổi thì không cần xử lý
    # và huấn luyện lại mô hình.
    if processed_digest and after_digest == processed_digest:
        return {
            "changed": False,
            "message": (
                "Nguồn chưa có dữ liệu mới; giữ nguyên "
                "dữ liệu đã xử lý và mô hình."
            ),
        }

    from data.preprocess_data import preprocess_gold_data
    from src.train import train_and_evaluate

    preprocess_gold_data()
    train_and_evaluate()

    # Chỉ ghi dấu dữ liệu nguồn sau khi xử lý và huấn luyện
    # hoàn tất. Nếu có lỗi, lần cập nhật tiếp theo sẽ thử lại.
    from src.atomic_files import atomic_output_path

    with atomic_output_path(
        processed_digest_path
    ) as temporary_path:
        Path(temporary_path).write_text(
            after_digest,
            encoding="utf-8",
        )

    return {
        "changed": True,
        "message": (
            "Đã tải dữ liệu mới, tính lại chỉ báo "
            "và huấn luyện lại mô hình."
        ),
    }