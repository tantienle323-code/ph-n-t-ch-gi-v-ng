import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.preprocess_data import preprocess_gold_frame
from src.news_sentiment import analyze_headline_sentiment, get_latest_news_sentiment


class TestGoldPipeline(unittest.TestCase):
    def setUp(self):
        self.sample_df = {
            'Date': ['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07'],
            'Open': [100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105],
            'Low': [99, 100, 101, 102, 103],
            'Close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'USD_Index': [100, 100.5, 101, 101.5, 102],
            'US10Y_Yield': [1.0, 1.1, 1.2, 1.3, 1.4],
            'Crude_Oil': [60, 61, 62, 63, 64],
            'SP500': [3000, 3005, 3010, 3015, 3020],
        }

    def test_preprocess_creates_target_and_technical_columns(self):
        import pandas as pd

        df = pd.DataFrame(self.sample_df)
        processed = preprocess_gold_frame(df)

        self.assertIn('Target', processed.columns)
        self.assertIn('RSI_14', processed.columns)
        self.assertIn('MACD', processed.columns)
        self.assertIn('Volatility_20_Ann', processed.columns)
        self.assertFalse(processed.empty)

    def test_news_sentiment_detects_risk_keywords(self):
        sentiment, crisis_flag = analyze_headline_sentiment('War tensions escalate near oil facilities and sanctions intensify')
        self.assertLess(sentiment, 0)
        self.assertEqual(crisis_flag, 1)

    def test_news_sentiment_summary_returns_expected_keys(self):
        result = get_latest_news_sentiment()
        self.assertIn('avg_sentiment', result)
        self.assertIn('crisis_flag', result)
        self.assertIn('news_list', result)
        self.assertGreaterEqual(len(result['news_list']), 1)


if __name__ == '__main__':
    unittest.main()
