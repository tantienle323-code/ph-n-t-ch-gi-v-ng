import os
import sys
import unittest
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_quality import assess_market_data


class TestMarketDataQuality(unittest.TestCase):
    def test_recent_friday_data_is_fresh_on_saturday(self):
        data = pd.DataFrame({
            'Date': ['2026-10-16'],
            'Close': [5000.0],
            'Target': [float('nan')],
        })

        health = assess_market_data(
            data,
            now=datetime(2026, 10, 17, tzinfo=timezone.utc),
        )

        self.assertEqual(health['status'], 'ready')
        self.assertEqual(health['business_days_lag'], 0)
        self.assertEqual(health['latest_date'].isoformat(), '2026-10-16')

    def test_old_or_corrupt_rows_are_flagged(self):
        data = pd.DataFrame({
            'Date': ['2026-10-01', '2026-10-01', '2026-10-20'],
            'Close': [5000.0, 0.0, 5100.0],
            'Target': [0.0, 1.0, float('nan')],
        })

        health = assess_market_data(
            data,
            now=datetime(2026, 10, 17, tzinfo=timezone.utc),
        )

        self.assertEqual(health['status'], 'error')
        self.assertEqual(health['duplicate_dates'], 1)
        self.assertEqual(health['invalid_close_count'], 1)
        self.assertEqual(health['future_date_count'], 1)

    def test_market_data_delayed_by_more_than_five_business_days_is_stale(self):
        data = pd.DataFrame({
            'Date': ['2026-10-01'],
            'Close': [5000.0],
            'Target': [1.0],
        })

        health = assess_market_data(
            data,
            now=datetime(2026, 10, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(health['status'], 'stale')
        self.assertGreater(health['business_days_lag'], 5)

    def test_missing_required_columns_are_reported(self):
        health = assess_market_data(pd.DataFrame({'Date': ['2026-10-16']}))

        self.assertEqual(health['status'], 'error')
        self.assertIn('Close', health['missing_columns'])


if __name__ == '__main__':
    unittest.main()
