import unittest

import pandas as pd

from src.sjc_prices import (
    international_equivalent_vnd_per_luong,
    quote_comparison,
    validate_sjc_quotes,
)


class TestSJCPrices(unittest.TestCase):
    def setUp(self):
        self.quotes = pd.DataFrame({
            "Date": [
                "2026-10-18",
                "2026-10-19",
            ],
            "Buy_VND_per_luong": [
                100_000_000,
                101_000_000,
            ],
            "Sell_VND_per_luong": [
                102_000_000,
                103_000_000,
            ],
            "Source": [
                "Nguồn kiểm thử",
                "Nguồn kiểm thử",
            ],
        })

    def test_valid_quotes_are_sorted_and_normalized(self):
        result = validate_sjc_quotes(
            self.quotes.iloc[::-1]
        )

        self.assertEqual(
            result.iloc[-1]["Date"],
            pd.Timestamp("2026-10-19"),
        )

        self.assertEqual(
            list(result.columns),
            [
                "Date",
                "Buy_VND_per_luong",
                "Sell_VND_per_luong",
                "Source",
            ],
        )

    def test_rejects_sell_price_below_buy_price(self):
        invalid = self.quotes.copy()

        invalid.loc[
            0,
            "Sell_VND_per_luong",
        ] = 99_000_000

        with self.assertRaisesRegex(
            ValueError,
            "không được thấp hơn",
        ):
            validate_sjc_quotes(invalid)

    def test_rejects_missing_source(self):
        invalid = self.quotes.copy()
        invalid.loc[0, "Source"] = ""

        with self.assertRaisesRegex(
            ValueError,
            "ghi rõ nguồn",
        ):
            validate_sjc_quotes(invalid)

    def test_international_conversion_uses_troy_ounce_and_luong_grams(self):
        result = international_equivalent_vnd_per_luong(
            2000,
            25000,
        )

        self.assertAlmostEqual(
            result,
            2000 * 25000 * 37.5 / 31.1034768,
        )

    def test_quote_comparison_reports_latest_quote_and_premium(self):
        result = quote_comparison(
            self.quotes,
            2000,
            25000,
        )

        self.assertEqual(
            result["date"],
            pd.Timestamp("2026-10-19"),
        )

        self.assertEqual(
            result["buy"],
            101_000_000,
        )

        self.assertGreater(
            result["midpoint_premium_vnd"],
            0,
        )

    def test_rejects_future_date(self):
        invalid = self.quotes.copy()

        invalid.loc[0, "Date"] = (
            pd.Timestamp.now().normalize()
            + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        with self.assertRaisesRegex(
            ValueError,
            "tương lai",
        ):
            validate_sjc_quotes(invalid)


if __name__ == "__main__":
    unittest.main()