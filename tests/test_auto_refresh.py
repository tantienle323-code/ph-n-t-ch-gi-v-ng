import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.auto_refresh import (
    refresh_due,
    run_refresh_if_due,
    read_refresh_status,
)


class TestAutomaticRefresh(unittest.TestCase):
    def test_refresh_is_due_without_previous_status(self):
        self.assertTrue(refresh_due({}, 60))

    def test_refresh_is_not_due_inside_interval(self):
        now = datetime(
            2026, 10, 21, 0, 0,
            tzinfo=timezone.utc,
        )

        status = {
            "last_attempt_utc": (
                now - timedelta(minutes=20)
            ).isoformat()
        }

        self.assertFalse(
            refresh_due(status, 60, now=now)
        )

    def test_failed_refresh_uses_early_retry_time(self):
        now = datetime(
            2026, 10, 21, 0, 0,
            tzinfo=timezone.utc,
        )

        status = {
            "status": "error",
            "last_attempt_utc": (
                now - timedelta(minutes=2)
            ).isoformat(),
            "retry_after_utc": (
                now - timedelta(seconds=1)
            ).isoformat(),
        }

        self.assertTrue(
            refresh_due(status, 60, now=now)
        )

    def test_refresh_runs_once_and_persists_status(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            calls = []

            def callback():
                calls.append(True)
                return {
                    "changed": True,
                    "message": "test success",
                }

            first = run_refresh_if_due(
                root,
                callback,
                interval_minutes=60,
            )

            second = run_refresh_if_due(
                root,
                callback,
                interval_minutes=60,
            )

            self.assertEqual(
                first["status"],
                "success",
            )

            self.assertEqual(
                second["status"],
                "success",
            )

            self.assertEqual(len(calls), 1)

            saved_status = read_refresh_status(
                root / "data" / "auto_refresh_status.json"
            )

            self.assertEqual(
                saved_status["message"],
                "test success",
            )

    def test_refresh_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)

            def callback():
                raise RuntimeError("network down")

            result = run_refresh_if_due(
                root,
                callback,
                interval_minutes=60,
            )

            self.assertEqual(
                result["status"],
                "error",
            )

            self.assertIn(
                "network down",
                result["message"],
            )

            self.assertIn(
                "retry_after_utc",
                result,
            )


if __name__ == "__main__":
    unittest.main()