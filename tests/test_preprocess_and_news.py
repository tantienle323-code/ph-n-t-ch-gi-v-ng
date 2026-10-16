import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import get_data
from data.preprocess_data import preprocess_gold_frame
from src.backtest import prepare_walk_forward_backtest_data, run_backtest
from src.train import (
    regression_metrics,
    temporal_split_indices,
    walk_forward_predict,
    walk_forward_regression_predict,
)
from src.news_sentiment import (
    NewsTranslationError,
    analyze_headline_sentiment,
    parse_google_news_rss,
    translate_headline_to_vietnamese,
)
from src.telegram_bot import send_telegram_message


RSS_FIXTURE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <item>
    <title>Gold rises as markets await policy deal</title>
    <link>https://example.com/gold-news</link>
    <source url="https://example.com">Example News</source>
    <pubDate>Wed, 02 Oct 2002 13:00:00 GMT</pubDate>
  </item>
  <item>
    <title>War tensions lift gold prices</title>
    <source>Market Desk</source>
    <pubDate>Wed, 02 Oct 2002 14:00:00 GMT</pubDate>
  </item>
</channel></rss>"""


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

    def test_preprocess_keeps_latest_unlabeled_row_for_live_prediction(self):
        import pandas as pd

        rows = 240
        close = pd.Series(range(100, 100 + rows), dtype=float)
        df = pd.DataFrame({
            'Date': pd.date_range('2025-01-01', periods=rows, freq='D'),
            'Open': close - 0.5,
            'High': close + 1,
            'Low': close - 1,
            'Close': close,
            'Volume': 1000,
            'USD_Index': close / 10,
            'US10Y_Yield': close / 100,
            'Crude_Oil': close / 2,
            'SP500': close * 10,
        })

        processed = preprocess_gold_frame(df)

        self.assertEqual(len(processed), rows - 199)
        self.assertTrue(pd.isna(processed.iloc[-1]['Target']))
        self.assertTrue(processed['Target'].iloc[:-1].notna().all())

    def test_temporal_split_keeps_train_validation_test_in_order(self):
        train_end, validation_end = temporal_split_indices(1058)

        self.assertEqual((train_end, validation_end), (740, 899))
        self.assertLess(train_end, validation_end)
        self.assertLess(validation_end, 1058)

    def test_walk_forward_fits_only_on_rows_before_each_test_block(self):
        import pandas as pd

        fitted_sizes = []

        class ConstantClassifier:
            def fit(self, features, targets):
                fitted_sizes.append(len(targets))
                return self

            def predict(self, features):
                return [0] * len(features)

        features = pd.DataFrame({'feature': range(10)})
        targets = pd.Series([0, 1] * 5)
        predictions, folds = walk_forward_predict(
            ConstantClassifier,
            features,
            targets,
            start_index=4,
            step_size=2,
        )

        self.assertEqual(fitted_sizes, [4, 6, 8])
        self.assertEqual(predictions.tolist(), [0] * 6)
        self.assertEqual([fold['test_size'] for fold in folds], [2, 2, 2])

    def test_walk_forward_regression_fits_only_on_known_rows(self):
        import pandas as pd

        fitted_sizes = []

        class MeanRegressor:
            def fit(self, features, targets):
                fitted_sizes.append(len(targets))
                self.mean = float(targets.mean())
                return self

            def predict(self, features):
                return [self.mean] * len(features)

        features = pd.DataFrame({'feature': range(10)})
        targets = pd.Series([float(value) for value in range(10)])
        predictions, folds = walk_forward_regression_predict(
            MeanRegressor,
            features,
            targets,
            start_index=4,
            step_size=2,
        )

        self.assertEqual(fitted_sizes, [4, 6, 8])
        self.assertEqual(predictions.tolist(), [1.5, 1.5, 2.5, 2.5, 3.5, 3.5])
        self.assertEqual([fold['test_size'] for fold in folds], [2, 2, 2])

    def test_regression_metrics_report_return_error_and_direction(self):
        metrics = regression_metrics(
            np.array([1.0, -1.0, 0.5]),
            np.array([0.5, -0.5, -0.5]),
        )

        self.assertAlmostEqual(metrics['mae_pct'], 2 / 3)
        self.assertAlmostEqual(metrics['rmse_pct'], (0.5) ** 0.5)
        self.assertAlmostEqual(metrics['directional_accuracy'], 2 / 3)

    def test_backtest_uses_out_of_sample_signals_and_following_price(self):
        import pandas as pd

        data = pd.DataFrame({
            'Date': pd.date_range('2026-01-01', periods=5, freq='D'),
            'Close': [100, 101, 99, 102, 103],
        })
        predictions = [
            {'date': '2026-01-02', 'prediction': 0, 'actual': 0},
            {'date': '2026-01-03', 'prediction': 1, 'actual': 1},
        ]

        backtest_data, signals = prepare_walk_forward_backtest_data(data, predictions)

        self.assertEqual(backtest_data['Close'].tolist(), [101, 99, 102])
        self.assertEqual(signals, [0, 1, 0])

    def test_backtest_rejects_out_of_sample_dates_missing_from_current_data(self):
        import pandas as pd

        data = pd.DataFrame({
            'Date': pd.date_range('2026-01-01', periods=3, freq='D'),
            'Close': [100, 101, 102],
        })

        with self.assertRaisesRegex(ValueError, 'hãy huấn luyện lại'):
            prepare_walk_forward_backtest_data(
                data,
                [{'date': '2025-12-31', 'prediction': 1, 'actual': 1}],
            )

    def test_empty_market_response_does_not_overwrite_existing_data(self):
        class EmptyTicker:
            def history(self, period):
                import pandas as pd

                return pd.DataFrame()

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / 'gold_data_5y.csv'
            output_file.write_text('existing-data', encoding='utf-8')
            with patch.object(get_data, 'OUTPUT_FILE', str(output_file)), \
                    patch.object(get_data.yf, 'Ticker', return_value=EmptyTicker()):
                with self.assertRaisesRegex(ValueError, 'dữ liệu hợp lệ'):
                    get_data.fetch_comprehensive_financial_data()
            self.assertEqual(output_file.read_text(encoding='utf-8'), 'existing-data')

    def test_news_sentiment_detects_risk_keywords(self):
        sentiment, crisis_flag = analyze_headline_sentiment('War tensions escalate near oil facilities and sanctions intensify')
        self.assertLess(sentiment, 0)
        self.assertEqual(crisis_flag, 1)

    @patch('src.news_sentiment.requests.get')
    def test_headline_translation_returns_vietnamese_text(self, get):
        response = Mock()
        response.json.return_value = {
            'responseStatus': 200,
            'responseData': {'translatedText': 'Vàng tăng giá'},
        }
        get.return_value = response

        translated = translate_headline_to_vietnamese('Gold prices rise')

        self.assertEqual(translated, 'Vàng tăng giá')
        get.assert_called_once_with(
            'https://api.mymemory.translated.net/get',
            params={'q': 'Gold prices rise', 'langpair': 'en|vi'},
            timeout=(5, 15),
        )

    @patch('src.news_sentiment.requests.get')
    def test_headline_translation_reports_invalid_service_response(self, get):
        response = Mock()
        response.json.return_value = {
            'responseStatus': 429,
            'responseData': {'translatedText': ''},
        }
        get.return_value = response

        with self.assertRaises(NewsTranslationError):
            translate_headline_to_vietnamese('Gold prices rise')

    def test_google_news_rss_parser_returns_scored_articles(self):
        result = parse_google_news_rss(RSS_FIXTURE)
        self.assertIn('avg_sentiment', result)
        self.assertIn('crisis_flag', result)
        self.assertIn('news_list', result)
        self.assertEqual(result['source'], 'Google News RSS')
        self.assertEqual(len(result['news_list']), 2)
        self.assertEqual(result['news_list'][0]['source'], 'Market Desk')
        self.assertEqual(result['news_list'][1]['source'], 'Example News')
        self.assertEqual(result['news_list'][1]['url'], 'https://example.com/gold-news')
        self.assertEqual(
            result['news_list'][1]['published_at'],
            '2002-10-02 20:00 ICT (UTC+7)',
        )
        self.assertEqual(result['crisis_flag'], 1)

    def test_google_news_rss_excludes_headlines_dated_after_today(self):
        rss = b"""<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0"><channel>
          <item>
            <title>Gold prices today, October 17, 2026: market update</title>
            <link>https://example.com/future-gold-news</link>
            <source>Example News</source>
            <pubDate>Fri, 16 Oct 2026 15:39:38 GMT</pubDate>
          </item>
          <item>
            <title>Gold market closes higher</title>
            <link>https://example.com/current-gold-news</link>
            <source>Market Desk</source>
            <pubDate>Fri, 16 Oct 2026 14:00:00 GMT</pubDate>
          </item>
        </channel></rss>"""

        result = parse_google_news_rss(rss, reference_date=date(2026, 10, 16))

        self.assertEqual(len(result['news_list']), 1)
        self.assertEqual(result['news_list'][0]['title'], 'Gold market closes higher')

    def test_backtest_applies_round_trip_fees_and_slippage(self):
        import pandas as pd

        data = pd.DataFrame({
            'Close': [100.0, 102.0, 101.0],
            'Volatility_Spike_Flag': [0, 0, 0],
        })
        result = run_backtest(
            data,
            signals=[1, 1, 1],
            initial_capital=1000,
            stop_loss_pct=0.05,
            take_profit_pct=0.05,
            transaction_cost_bps=10,
            slippage_bps=5,
        )
        self.assertAlmostEqual(result.loc[0, 'Trading_Cost_Return'], 0.003)
        self.assertAlmostEqual(result.loc[0, 'Net_Return'], 0.017)
        self.assertAlmostEqual(result.loc[1, 'Net_Return'], -0.01280392, places=6)
        self.assertEqual(len(result['AI_Portfolio']), len(data))

    @patch.dict(os.environ, {}, clear=True)
    def test_telegram_requires_environment_credentials(self):
        with self.assertRaisesRegex(RuntimeError, 'TELEGRAM_BOT_TOKEN'):
            send_telegram_message('test')

    @patch('src.telegram_bot.requests.post')
    @patch.dict(os.environ, {'TELEGRAM_BOT_TOKEN': 'test-token', 'TELEGRAM_CHAT_ID': '123'})
    def test_telegram_sends_message_with_timeout(self, post):
        response = Mock()
        response.ok = True
        response.json.return_value = {'ok': True}
        post.return_value = response

        send_telegram_message('hello')

        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs['timeout'], (5, 15))


if __name__ == '__main__':
    unittest.main()
