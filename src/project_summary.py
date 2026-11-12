from collections.abc import Mapping
from typing import Any


def build_project_summary(
    latest_data_date: str,
    session_count: int,
    model_payload: Mapping[str, Any] | None,
) -> str:
    """Build a concise, data-backed summary for project presentations."""
    lines = [
        "# Tóm tắt đồ án: Phân tích và dự báo giá vàng",
        "",
        f"- Ngày dữ liệu mới nhất: {latest_data_date}",
        f"- Số phiên dữ liệu: {session_count:,}",
        "- Nguồn: Yahoo Finance (GC=F, DX-Y.NYB, ^TNX, CL=F, ^GSPC).",
        "",
        "## Mục tiêu",
        "Xây dựng dashboard thu thập và phân tích dữ liệu vàng, dự báo hướng "
        "biến động của phiên kế tiếp, theo dõi kết quả thực tế và mô phỏng "
        "chiến lược trên dữ liệu ngoài mẫu.",
        "",
        "## Dữ liệu và đặc trưng",
        "Dữ liệu vĩ mô được ghép theo ngày giao dịch của vàng và chỉ điền xuôi "
        "từ giá trị đã biết. Đặc trưng gồm lợi suất, độ biến động, ATR, RSI, "
        "MACD, SMA/EMA, Bollinger Bands, độ trễ và biến động các chỉ số vĩ mô.",
        "",
        "Nhãn **TĂNG** khi giá đóng cửa phiên kế tiếp cao hơn phiên hiện tại; "
        "ngược lại là **KHÔNG TĂNG** (bằng hoặc thấp hơn).",
        "",
        "## Huấn luyện và đánh giá",
        "Dữ liệu được chia theo thời gian, xấp xỉ 70% train, 15% validation và "
        "15% test. "
        "Mô hình được chọn trên validation; test chỉ dùng đánh giá walk-forward "
        "ngoài mẫu theo các khối tối đa 20 phiên. Mô hình cuối được huấn luyện "
        "lại trên toàn bộ dữ liệu có nhãn sau đánh giá.",
        "",
        "Các ứng viên phân loại gồm Extra Trees, Gradient Boosting và Random "
        "Forest. Mô hình lợi suất được so sánh với baseline dự báo 0%.",
        "",
        "Dữ liệu được kiểm tra cập nhật theo lịch 24 giờ. Nếu dashboard không "
        "mở, cần chạy worker nền hoặc cấu hình Task Scheduler để lịch tiếp tục hoạt động.",
    ]

    if model_payload is None:
        lines.extend([
            "",
            "Chưa có mô hình đã huấn luyện để đưa kết quả kiểm định vào báo cáo.",
        ])
    else:
        model_metrics = model_payload.get("metrics") or {}
        return_metrics = model_payload.get("return_metrics") or {}
        walk_forward = model_payload.get("walk_forward") or {}

        lines.extend([
            "",
            "### Kết quả mô hình hiện tại",
            f"- Mô hình phân loại: {model_payload.get('model_name', 'N/A')}.",
            f"- Mô hình lợi suất: {model_payload.get('return_model_name', 'N/A')}.",
        ])

        if walk_forward:
            lines.append(
                f"- Train: {walk_forward.get('train_size', 0)} phiên đến "
                f"{walk_forward.get('train_end_date', 'N/A')}; validation: "
                f"{walk_forward.get('validation_size', 0)} phiên đến "
                f"{walk_forward.get('validation_end_date', 'N/A')}; test: "
                f"{walk_forward.get('test_size', 0)} phiên từ "
                f"{walk_forward.get('test_start_date', 'N/A')} đến "
                f"{walk_forward.get('test_end_date', 'N/A')}."
            )
            lines.append(
                f"- Walk-forward gồm {walk_forward.get('fold_count', 0)} đợt, "
                f"mỗi đợt tối đa {walk_forward.get('step_size', 0)} phiên."
            )

        classification_metric_names = (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1"),
        )
        available_classification_metrics = [
            f"{label} {float(model_metrics[key]) * 100:.2f}%"
            for key, label in classification_metric_names
            if key in model_metrics
        ]
        if available_classification_metrics:
            lines.append(
                "- Chỉ số phân loại ngoài mẫu: "
                + ", ".join(available_classification_metrics)
                + "."
            )

        available_return_metrics = [
            f"{label} {float(return_metrics[key]):.4f}%"
            for key, label in (
                ("mae_pct", "MAE"),
                ("rmse_pct", "RMSE"),
            )
            if key in return_metrics
        ]
        if "directional_accuracy" in return_metrics:
            available_return_metrics.append(
                "Đúng hướng "
                f"{float(return_metrics['directional_accuracy']) * 100:.2f}%"
            )
        if available_return_metrics:
            lines.append(
                "- Chỉ số dự báo lợi suất ngoài mẫu: "
                + ", ".join(available_return_metrics)
                + "."
            )
        if {
            "prediction_interval_coverage",
            "prediction_interval_level",
        }.issubset(return_metrics):
            lines.append(
                "- Độ phủ khoảng ước lượng: "
                f"{float(return_metrics['prediction_interval_coverage']) * 100:.2f}% "
                f"(mục tiêu {float(return_metrics['prediction_interval_level']) * 100:.0f}%)."
            )

    lines.extend([
        "",
        "## Giới hạn",
        "Dữ liệu phụ thuộc nguồn bên ngoài và có thể trễ hoặc thiếu. Điểm "
        "`predict_proba` chưa được hiệu chuẩn; tin tức được phân tích theo từ "
        "khóa tiêu đề; báo giá SJC và tỷ giá quy đổi do người dùng cung cấp. "
        "Backtest là mô phỏng theo giả định, không đảm bảo hiệu suất tương lai. "
        "Ứng dụng phục vụ học tập/nghiên cứu, không phải khuyến nghị đầu tư.",
    ])

    return "\n".join(lines)
