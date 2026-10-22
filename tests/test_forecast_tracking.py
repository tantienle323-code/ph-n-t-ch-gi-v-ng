import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.forecast_tracking import (
    summarize_forecast_history,
    update_forecast_history,
)


class TestForecastTracking(unittest.TestCase):
    def test_forecast_is_saved_once_and_resolved_when_actual_target_arrives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir)
                / "forecast_history.sqlite3"
            )

            first_data = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-10-15",
                    "2026-10-16",
                ]),
                "Target": [1.0, float("nan")],
                "Return": [0.01, -0.02],
            })

            forecast = {
                "date": "2026-10-16",
                "prediction": 0,
                "confidence": 0.72,
                "model_name": "Gradient Boosting",
                "model_version": "build-1",
                "previous_direction": 0,
            }

            pending_history = update_forecast_history(
                database_path,
                first_data,
                forecast,
            )

            duplicate_history = update_forecast_history(
                database_path,
                first_data,
                forecast,
            )

            self.assertEqual(
                len(pending_history),
                1,
            )

            self.assertEqual(
                len(duplicate_history),
                1,
            )

            self.assertTrue(
                pd.isna(
                    duplicate_history.iloc[0][
                        "actual_direction"
                    ]
                )
            )

            updated_data = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-10-15",
                    "2026-10-16",
                    "2026-10-19",
                ]),
                "Target": [
                    1.0,
                    1.0,
                    float("nan"),
                ],
                "Return": [
                    0.01,
                    -0.02,
                    0.03,
                ],
            })

            resolved_history = update_forecast_history(
                database_path,
                updated_data,
                {
                    **forecast,
                    "date": "2026-10-19",
                    "prediction": 1,
                    "model_version": "build-1",
                },
            )

            resolved_row = resolved_history.loc[
                resolved_history["forecast_date"]
                == "2026-10-16"
            ].iloc[0]

            self.assertEqual(
                resolved_row["actual_direction"],
                1,
            )

            self.assertEqual(
                len(resolved_history),
                2,
            )

    def test_return_forecast_is_scored_after_following_close_arrives(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir)
                / "forecast_history.sqlite3"
            )

            first_data = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-10-16",
                ]),
                "Target": [float("nan")],
                "Close": [100.0],
                "Return": [0.02],
            })

            forecast = {
                "date": "2026-10-16",
                "prediction": 1,
                "confidence": 0.7,
                "model_name": "Gradient Boosting",
                "model_version": "build-2",
                "previous_direction": 1,
                "predicted_return_pct": 2.0,
                "reference_close": 100.0,
            }

            update_forecast_history(
                database_path,
                first_data,
                forecast,
            )

            updated_data = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-10-16",
                    "2026-10-19",
                ]),
                "Target": [
                    1.0,
                    float("nan"),
                ],
                "Close": [
                    100.0,
                    103.0,
                ],
                "Return": [
                    0.02,
                    0.03,
                ],
            })

            history = update_forecast_history(
                database_path,
                updated_data,
                {
                    **forecast,
                    "date": "2026-10-19",
                    "model_version": "build-3",
                },
            )

            settled = history.loc[
                history["forecast_date"] == "2026-10-16"
            ].iloc[0]

            summary = summarize_forecast_history(
                history
            )

            self.assertEqual(
                settled["predicted_close"],
                102.0,
            )

            self.assertEqual(
                settled["actual_close"],
                103.0,
            )

            self.assertAlmostEqual(
                settled["actual_return_pct"],
                3.0,
            )

            self.assertAlmostEqual(
                summary["return_metrics"]["mae_pct"],
                1.0,
            )

            self.assertAlmostEqual(
                summary["return_metrics"]["zero_baseline_mae_pct"],
                3.0,
            )

    def test_summary_uses_latest_model_forecast_for_each_date(self):
        history = pd.DataFrame([
            {
                "forecast_date": "2026-10-15",
                "predicted_direction": 0,
                "confidence": 0.7,
                "model_name": "Older model",
                "model_version": "build-1",
                "recorded_at": "2026-10-15T10:00:00+00:00",
                "actual_direction": 1,
                "previous_direction": 0,
            },
            {
                "forecast_date": "2026-10-15",
                "predicted_direction": 1,
                "confidence": 0.8,
                "model_name": "Newer model",
                "model_version": "build-2",
                "recorded_at": "2026-10-15T11:00:00+00:00",
                "actual_direction": 1,
                "previous_direction": 1,
            },
            {
                "forecast_date": "2026-10-16",
                "predicted_direction": 0,
                "confidence": 0.6,
                "model_name": "Newer model",
                "model_version": "build-2",
                "recorded_at": "2026-10-16T10:00:00+00:00",
                "actual_direction": float("nan"),
                "previous_direction": 0,
            },
        ])

        summary = summarize_forecast_history(
            history
        )

        self.assertEqual(
            summary["forecast_count"],
            2,
        )

        self.assertEqual(
            summary["resolved_count"],
            1,
        )

        self.assertEqual(
            summary["pending_count"],
            1,
        )

        self.assertEqual(
            summary["accuracy"],
            1.0,
        )

        self.assertEqual(
            summary["f1_score"],
            1.0,
        )

        self.assertEqual(
            summary["average_confidence"],
            0.8,
        )

        self.assertEqual(
            summary["baseline_metrics"][
                "previous_day_direction"
            ]["f1_score"],
            1.0,
        )

        self.assertEqual(
            summary["baseline_metrics"]["model"]["f1_score"],
            1.0,
        )

    def test_invalid_forecast_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            market_data = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-10-16",
                ]),
                "Target": [float("nan")],
                "Return": [0.0],
            })

            forecast = {
                "date": "2026-10-16",
                "prediction": 2,
                "confidence": 0.72,
                "model_name": "Gradient Boosting",
                "model_version": "build-1",
                "previous_direction": 0,
            }

            with self.assertRaisesRegex(
                ValueError,
                "0 hoặc 1",
            ):
                update_forecast_history(
                    Path(temp_dir)
                    / "forecast_history.sqlite3",
                    market_data,
                    forecast,
                )

    def test_existing_ledger_is_migrated_with_previous_direction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = (
                Path(temp_dir)
                / "forecast_history.sqlite3"
            )

            with closing(
                sqlite3.connect(database_path)
            ) as connection:
                with connection:
                    connection.execute(
                        """
                        CREATE TABLE forecast_history (
                            forecast_date TEXT NOT NULL,
                            predicted_direction INTEGER NOT NULL,
                            confidence REAL NOT NULL,
                            model_name TEXT NOT NULL,
                            model_version TEXT NOT NULL,
                            recorded_at TEXT NOT NULL
                                DEFAULT CURRENT_TIMESTAMP,
                            actual_direction INTEGER,
                            PRIMARY KEY (
                                forecast_date,
                                model_version
                            )
                        )
                        """
                    )

                    connection.execute(
                        """
                        INSERT INTO forecast_history (
                            forecast_date,
                            predicted_direction,
                            confidence,
                            model_name,
                            model_version,
                            actual_direction
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            "2026-10-15",
                            1,
                            0.75,
                            "Older model",
                            "build-0",
                            None,
                        ),
                    )

            market_data = pd.DataFrame({
                "Date": pd.to_datetime([
                    "2026-10-15",
                    "2026-10-16",
                ]),
                "Target": [
                    1.0,
                    float("nan"),
                ],
                "Return": [
                    0.01,
                    -0.02,
                ],
            })

            history = update_forecast_history(
                database_path,
                market_data,
                {
                    "date": "2026-10-16",
                    "prediction": 0,
                    "confidence": 0.72,
                    "model_name": "Gradient Boosting",
                    "model_version": "build-1",
                    "previous_direction": 0,
                },
            )

            old_forecast = history.loc[
                history["forecast_date"] == "2026-10-15"
            ].iloc[0]

            self.assertEqual(
                old_forecast["actual_direction"],
                1,
            )

            self.assertEqual(
                old_forecast["previous_direction"],
                1,
            )


if __name__ == "__main__":
    unittest.main()