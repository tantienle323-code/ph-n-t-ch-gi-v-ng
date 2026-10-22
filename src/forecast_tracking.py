
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def update_forecast_history(database_path, market_data, forecast):
    """Store a live forecast and resolve earlier outcomes when labels arrive."""
    required_market_columns = {"Date", "Target"}

    if not required_market_columns.issubset(market_data.columns):
        raise ValueError(
            "Dữ liệu theo dõi dự báo cần có cột Date và Target."
        )

    required_forecast_fields = {
        "date",
        "prediction",
        "confidence",
        "model_name",
        "model_version",
        "previous_direction",
    }

    if not required_forecast_fields.issubset(forecast):
        raise ValueError("Thông tin dự báo thiếu trường bắt buộc.")

    prediction = forecast["prediction"]

    if prediction not in (0, 1):
        raise ValueError("Hướng dự báo phải là 0 hoặc 1.")

    confidence = float(forecast["confidence"])

    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(
            "Độ tin cậy dự báo phải nằm trong khoảng từ 0 đến 1."
        )

    forecast_date = (
        pd.Timestamp(forecast["date"])
        .normalize()
        .strftime("%Y-%m-%d")
    )

    model_name = str(forecast["model_name"]).strip()
    model_version = str(forecast["model_version"]).strip()

    if not model_name or not model_version:
        raise ValueError(
            "Tên và phiên bản mô hình không được để trống."
        )

    ordered_market_data = (
        market_data.assign(
            _forecast_date=pd.to_datetime(
                market_data["Date"]
            ).dt.normalize()
        )
        .sort_values("_forecast_date")
        .reset_index(drop=True)
    )

    market_dates = (
        ordered_market_data["_forecast_date"]
        .dt.strftime("%Y-%m-%d")
    )

    actual_by_date = {}
    previous_direction_by_date = {}
    actual_return_by_date = {}
    actual_close_by_date = {}

    future_close = (
        pd.to_numeric(
            ordered_market_data["Close"], errors="coerce"
        ).shift(-1)
        if "Close" in ordered_market_data
        else None
    )

    returns = (
        ordered_market_data["Return"]
        if "Return" in ordered_market_data.columns
        else None
    )

    for index, (current_date, target) in enumerate(
        zip(market_dates, ordered_market_data["Target"])
    ):
        if pd.notna(target):
            if target not in (0, 1):
                raise ValueError(
                    "Nhãn Target thực tế phải là 0 hoặc 1."
                )

            actual_by_date[current_date] = int(target)

            if (
                future_close is not None
                and pd.notna(future_close.iloc[index])
            ):
                current_close = float(
                    ordered_market_data.iloc[index]["Close"]
                )

                if current_close > 0:
                    actual_close_by_date[current_date] = float(
                        future_close.iloc[index]
                    )

                    actual_return_by_date[current_date] = (
                        float(future_close.iloc[index])
                        / current_close - 1
                    ) * 100

        if returns is not None and pd.notna(returns.iloc[index]):
            previous_direction_by_date[current_date] = int(
                returns.iloc[index] > 0
            )

    previous_direction = forecast["previous_direction"]

    if previous_direction not in (0, 1):
        raise ValueError(
            "Baseline theo hướng phiên trước phải là 0 hoặc 1."
        )

    predicted_return_pct = forecast.get("predicted_return_pct")
    reference_close = forecast.get("reference_close")

    if predicted_return_pct is not None:
        predicted_return_pct = float(predicted_return_pct)

        if not math.isfinite(predicted_return_pct):
            raise ValueError(
                "Lợi suất dự báo phải là một số hữu hạn."
            )

    if reference_close is not None:
        reference_close = float(reference_close)

        if (
            not math.isfinite(reference_close)
            or reference_close <= 0
        ):
            raise ValueError(
                "Giá tham chiếu phải là số hữu hạn lớn hơn 0."
            )

    predicted_close = (
        reference_close * (1 + predicted_return_pct / 100)
        if (
            reference_close is not None
            and predicted_return_pct is not None
        )
        else None
    )

    database_path = Path(database_path)
    database_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with closing(
        sqlite3.connect(database_path, timeout=30)
    ) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_history (
                    forecast_date TEXT NOT NULL,
                    predicted_direction INTEGER NOT NULL
                        CHECK (predicted_direction IN (0, 1)),
                    confidence REAL NOT NULL
                        CHECK (confidence >= 0 AND confidence <= 1),
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    actual_direction INTEGER CHECK (
                        actual_direction IS NULL
                        OR actual_direction IN (0, 1)
                    ),
                    previous_direction INTEGER CHECK (
                        previous_direction IS NULL
                        OR previous_direction IN (0, 1)
                    ),
                    predicted_return_pct REAL,
                    reference_close REAL,
                    predicted_close REAL,
                    actual_return_pct REAL,
                    actual_close REAL,
                    PRIMARY KEY (forecast_date, model_version)
                )
                """
            )

            table_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(forecast_history)"
                ).fetchall()
            }

            if "previous_direction" not in table_columns:
                connection.execute(
                    """
                    ALTER TABLE forecast_history
                    ADD COLUMN previous_direction INTEGER CHECK (
                        previous_direction IS NULL
                        OR previous_direction IN (0, 1)
                    )
                    """
                )

            for column_name in (
                "predicted_return_pct",
                "reference_close",
                "predicted_close",
                "actual_return_pct",
                "actual_close",
            ):
                if column_name not in table_columns:
                    connection.execute(
                        "ALTER TABLE forecast_history "
                        f"ADD COLUMN {column_name} REAL"
                    )

            connection.executemany(
                """
                UPDATE forecast_history
                SET actual_direction = ?
                WHERE forecast_date = ?
                """,
                [
                    (actual_direction, current_date)
                    for current_date, actual_direction
                    in actual_by_date.items()
                ],
            )

            connection.executemany(
                """
                UPDATE forecast_history
                SET previous_direction = ?
                WHERE forecast_date = ?
                """,
                [
                    (direction, current_date)
                    for current_date, direction
                    in previous_direction_by_date.items()
                ],
            )

            connection.executemany(
                """
                UPDATE forecast_history
                SET actual_return_pct = ?, actual_close = ?
                WHERE forecast_date = ?
                """,
                [
                    (
                        actual_return_by_date[current_date],
                        actual_close_by_date[current_date],
                        current_date,
                    )
                    for current_date in actual_return_by_date
                ],
            )


            connection.execute(
                """
                INSERT INTO forecast_history (
                    forecast_date,
                    predicted_direction,
                    confidence,
                    model_name,
                    model_version,
                    recorded_at,
                    previous_direction,
                    actual_direction,
                    predicted_return_pct,
                    reference_close,
                    predicted_close
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (forecast_date, model_version) DO UPDATE SET
                    predicted_direction = excluded.predicted_direction,
                    confidence = excluded.confidence,
                    model_name = excluded.model_name,
                    previous_direction = excluded.previous_direction,
                    predicted_return_pct = excluded.predicted_return_pct,
                    reference_close = excluded.reference_close,
                    predicted_close = excluded.predicted_close
                """,
                (
                    forecast_date,
                    int(prediction),
                    confidence,
                    model_name,
                    model_version,
                    datetime.now(timezone.utc).isoformat(),
                    int(previous_direction),
                    actual_by_date.get(forecast_date),
                    predicted_return_pct,
                    reference_close,
                    predicted_close,
                ),
            )

            rows = connection.execute(
                """
                SELECT forecast_date, predicted_direction, confidence,
                       model_name, model_version, recorded_at,
                       actual_direction, previous_direction,
                       predicted_return_pct, reference_close,
                       predicted_close, actual_return_pct, actual_close
                FROM forecast_history
                ORDER BY forecast_date DESC, recorded_at DESC
                """
            ).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            "forecast_date",
            "predicted_direction",
            "confidence",
            "model_name",
            "model_version",
            "recorded_at",
            "actual_direction",
            "previous_direction",
            "predicted_return_pct",
            "reference_close",
            "predicted_close",
            "actual_return_pct",
            "actual_close",
        ],
    )


def summarize_forecast_history(history):
    """Summarize the latest recorded forecast for each market date."""
    if history.empty:
        return {
            "forecast_count": 0,
            "resolved_count": 0,
            "pending_count": 0,
            "accuracy": None,
            "f1_score": None,
            "average_confidence": None,
            "baseline_metrics": {},
            "return_metrics": {},
            "latest_history": history.copy(),
            "resolved_history": history.copy(),
        }

    for column in (
        "predicted_return_pct",
        "actual_return_pct",
    ):
        if column not in history:
            history = history.assign(**{column: np.nan})

    latest_by_date = (
        history.sort_values(
            ["recorded_at", "model_version"]
        )
        .drop_duplicates("forecast_date", keep="last")
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    resolved = latest_by_date.dropna(
        subset=["actual_direction"]
    ).copy()

    if resolved.empty:
        accuracy = None
        f1 = None
    else:
        actual = resolved["actual_direction"].astype(int)
        predicted = resolved["predicted_direction"].astype(int)

        accuracy = accuracy_score(actual, predicted)
        f1 = f1_score(
            actual, predicted, zero_division=0
        )

    baseline_history = resolved.dropna(
        subset=["previous_direction"]
    )

    if baseline_history.empty:
        baseline_metrics = {}
    else:
        baseline_actual = baseline_history[
            "actual_direction"
        ].astype(int)

        baseline_predicted = baseline_history[
            "predicted_direction"
        ].astype(int)

        baseline_metrics = {
            "model": {
                "accuracy": accuracy_score(
                    baseline_actual, baseline_predicted
                ),
                "f1_score": f1_score(
                    baseline_actual,
                    baseline_predicted,
                    zero_division=0,
                ),
            },
            "sample_count": len(baseline_history),
        }

        baseline_metrics.update({
            "always_up": {
                "accuracy": accuracy_score(
                    baseline_actual,
                    [1] * len(baseline_history),
                ),
                "f1_score": f1_score(
                    baseline_actual,
                    [1] * len(baseline_history),
                    zero_division=0,
                ),
            },
            "previous_day_direction": {
                "accuracy": accuracy_score(
                    baseline_actual,
                    baseline_history[
                        "previous_direction"
                    ].astype(int),
                ),
                "f1_score": f1_score(
                    baseline_actual,
                    baseline_history[
                        "previous_direction"
                    ].astype(int),
                    zero_division=0,
                ),
            },
        })

    return_history = resolved.dropna(
        subset=[
            "predicted_return_pct",
            "actual_return_pct",
        ]
    ).copy()

    if return_history.empty:
        return_metrics = {}
    else:
        actual_returns = return_history[
            "actual_return_pct"
        ].to_numpy(dtype=float)

        predicted_returns = return_history[
            "predicted_return_pct"
        ].to_numpy(dtype=float)

        errors = predicted_returns - actual_returns

        return_metrics = {
            "sample_count": len(return_history),
            "mae_pct": float(
                np.mean(np.abs(errors))
            ),
            "rmse_pct": float(
                np.sqrt(np.mean(np.square(errors)))
            ),
            "directional_accuracy": float(
                np.mean(
                    (predicted_returns > 0)
                    == (actual_returns > 0)
                )
            ),
            "zero_baseline_mae_pct": float(
                np.mean(np.abs(actual_returns))
            ),
            "zero_baseline_rmse_pct": float(
                np.sqrt(
                    np.mean(np.square(actual_returns))
                )
            ),
        }

    return {
        "forecast_count": len(latest_by_date),
        "resolved_count": len(resolved),
        "pending_count": int(
            latest_by_date["actual_direction"].isna().sum()
        ),
        "accuracy": accuracy,
        "f1_score": f1,
        "average_confidence": (
            float(resolved["confidence"].mean())
            if not resolved.empty
            else None
        ),
        "baseline_metrics": baseline_metrics,
        "return_metrics": return_metrics,
        "latest_history": latest_by_date,
        "resolved_history": resolved,
    }
