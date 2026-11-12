import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.telegram_bot import build_signal_message


class _ConstantModel:
    def predict(self, features):
        return [0 for _ in features.index]

    def predict_proba(self, features):
        return [[0.7, 0.3] for _ in features.index]


class TestTelegramSignalMessage(unittest.TestCase):
    def test_non_increase_label_and_probability_caveat_are_clear(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_path = root / "data.csv"
            model_path = root / "model.pkl"
            data_path.write_text(
                "Date,Close,RSI_14\n2026-11-12,4000,55\n",
                encoding="utf-8",
            )
            model_path.write_bytes(b"test model")
            model_payload = {
                "model": _ConstantModel(),
                "feature_cols": ["Close"],
            }

            with patch(
                "src.telegram_bot.pickle.load",
                return_value=model_payload,
            ):
                message = build_signal_message(
                    data_path=data_path,
                    model_path=model_path,
                )

        self.assertIn("Dự báo phiên tới: KHÔNG TĂNG", message)
        self.assertIn("predict_proba chưa được hiệu chuẩn", message)


if __name__ == "__main__":
    unittest.main()
