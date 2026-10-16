import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def update_forecast_history(database_path, market_data, forecast):
    """Store one live forecast and resolve earlier outcomes when labels arrive."""
    required_market_columns = {'Date', 'Target'}
    if not required_market_columns.issubset(market_data.columns):
        raise ValueError("Dữ liệu theo dõi dự báo cần có cột Date và Target.")

    required_forecast_fields = {
        'date',
        'prediction',
        'confidence',
        'model_name',
        'model_version',
        'previous_direction',
    }
    if not required_forecast_fields.issubset(forecast):
        raise ValueError("Thông tin dự báo thiếu trường bắt buộc.")

    prediction = forecast['prediction']
    if prediction not in (0, 1):
        raise ValueError("Hướng dự báo phải là 0 hoặc 1.")
    confidence = float(forecast['confidence'])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Độ tin cậy dự báo phải nằm trong khoảng từ 0 đến 1.")

    forecast_date = pd.Timestamp(forecast['date']).normalize().strftime('%Y-%m-%d')
    model_name = str(forecast['model_name']).strip()
    model_version = str(forecast['model_version']).strip()
    if not model_name or not model_version:
        raise ValueError("Tên và phiên bản mô hình không được để trống.")

    market_dates = pd.to_datetime(market_data['Date']).dt.normalize().dt.strftime('%Y-%m-%d')
    actual_by_date = {}
    previous_direction_by_date = {}
    returns = market_data['Return'] if 'Return' in market_data.columns else None
    for index, (current_date, target) in enumerate(zip(market_dates, market_data['Target'])):
        if pd.notna(target):
            if target not in (0, 1):
                raise ValueError("Nhãn Target thực tế phải là 0 hoặc 1.")
            actual_by_date[current_date] = int(target)
        if returns is not None and pd.notna(returns.iloc[index]):
            previous_direction_by_date[current_date] = int(returns.iloc[index] > 0)

    previous_direction = forecast['previous_direction']
    if previous_direction not in (0, 1):
        raise ValueError("Baseline theo hướng phiên trước phải là 0 hoặc 1.")

    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database_path, timeout=30)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_history (
                    forecast_date TEXT NOT NULL,
                    predicted_direction INTEGER NOT NULL CHECK (predicted_direction IN (0, 1)),
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    actual_direction INTEGER CHECK (
                        actual_direction IS NULL OR actual_direction IN (0, 1)
                    ),
                    previous_direction INTEGER CHECK (
                        previous_direction IS NULL OR previous_direction IN (0, 1)
                    ),
                    PRIMARY KEY (forecast_date, model_version)
                )
                """
            )
            table_columns = {
                row[1] for row in connection.execute(
                    "PRAGMA table_info(forecast_history)"
                ).fetchall()
            }
            if 'previous_direction' not in table_columns:
                connection.execute(
                    """
                    ALTER TABLE forecast_history
                    ADD COLUMN previous_direction INTEGER CHECK (
                        previous_direction IS NULL OR previous_direction IN (0, 1)
                    )
                    """
                )
            connection.executemany(
                """
                UPDATE forecast_history
                SET actual_direction = ?
                WHERE forecast_date = ?
                """,
                [
                    (actual_direction, current_date)
                    for current_date, actual_direction in actual_by_date.items()
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
                    for current_date, direction in previous_direction_by_date.items()
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
                    actual_direction
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (forecast_date, model_version) DO UPDATE SET
                    predicted_direction = excluded.predicted_direction,
                    confidence = excluded.confidence,
                    model_name = excluded.model_name,
                    previous_direction = excluded.previous_direction
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
                ),
            )
            rows = connection.execute(
                """
                SELECT forecast_date, predicted_direction, confidence,
                       model_name, model_version, recorded_at, actual_direction,
                       previous_direction
                FROM forecast_history
                ORDER BY forecast_date DESC, recorded_at DESC
                """
            ).fetchall()

    return pd.DataFrame(
        rows,
        columns=[
            'forecast_date',
            'predicted_direction',
            'confidence',
            'model_name',
            'model_version',
            'recorded_at',
            'actual_direction',
            'previous_direction',
        ],
    )


def summarize_forecast_history(history):
    """Summarize the latest recorded forecast for each market date."""
    if history.empty:
        return {
            'forecast_count': 0,
            'resolved_count': 0,
            'pending_count': 0,
            'accuracy': None,
            'f1_score': None,
            'average_confidence': None,
            'baseline_metrics': {},
            'latest_history': history.copy(),
            'resolved_history': history.copy(),
        }

    latest_by_date = (
        history.sort_values(['recorded_at', 'model_version'])
        .drop_duplicates('forecast_date', keep='last')
        .sort_values('forecast_date')
        .reset_index(drop=True)
    )
    resolved = latest_by_date.dropna(subset=['actual_direction']).copy()
    if resolved.empty:
        accuracy = None
        f1 = None
    else:
        actual = resolved['actual_direction'].astype(int)
        predicted = resolved['predicted_direction'].astype(int)
        accuracy = accuracy_score(actual, predicted)
        f1 = f1_score(actual, predicted, zero_division=0)

    baseline_history = resolved.dropna(subset=['previous_direction'])
    if baseline_history.empty:
        baseline_metrics = {}
    else:
        baseline_actual = baseline_history['actual_direction'].astype(int)
        baseline_predicted = baseline_history['predicted_direction'].astype(int)
        baseline_metrics = {
            'model': {
                'accuracy': accuracy_score(baseline_actual, baseline_predicted),
                'f1_score': f1_score(
                    baseline_actual,
                    baseline_predicted,
                    zero_division=0,
                ),
            },
            'sample_count': len(baseline_history),
        }
        baseline_metrics.update({
            'always_up': {
                'accuracy': accuracy_score(baseline_actual, [1] * len(baseline_history)),
                'f1_score': f1_score(
                    baseline_actual,
                    [1] * len(baseline_history),
                    zero_division=0,
                ),
            },
            'previous_day_direction': {
                'accuracy': accuracy_score(
                    baseline_actual,
                    baseline_history['previous_direction'].astype(int),
                ),
                'f1_score': f1_score(
                    baseline_actual,
                    baseline_history['previous_direction'].astype(int),
                    zero_division=0,
                ),
            },
        })

    return {
        'forecast_count': len(latest_by_date),
        'resolved_count': len(resolved),
        'pending_count': int(latest_by_date['actual_direction'].isna().sum()),
        'accuracy': accuracy,
        'f1_score': f1,
        'average_confidence': (
            float(resolved['confidence'].mean()) if not resolved.empty else None
        ),
        'baseline_metrics': baseline_metrics,
        'latest_history': latest_by_date,
        'resolved_history': resolved,
    }
