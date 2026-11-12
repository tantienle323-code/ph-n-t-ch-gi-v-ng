import unittest

from src.project_summary import build_project_summary


class TestProjectSummary(unittest.TestCase):
    def test_summary_includes_current_metrics_and_precise_target_labels(self):
        summary = build_project_summary(
            "12/11/2026",
            1250,
            {
                "model_name": "Gradient Boosting",
                "return_model_name": "No change",
                "metrics": {
                    "accuracy": 0.65,
                    "precision": 0.61,
                    "recall": 0.63,
                    "f1_score": 0.62,
                },
                "return_metrics": {
                    "mae_pct": 0.85,
                    "rmse_pct": 1.12,
                    "directional_accuracy": 0.58,
                    "prediction_interval_coverage": 0.89,
                    "prediction_interval_level": 0.90,
                },
                "walk_forward": {
                    "train_size": 700,
                    "train_end_date": "2023-12-31",
                    "validation_size": 150,
                    "validation_end_date": "2024-06-30",
                    "test_start_date": "2025-01-01",
                    "test_end_date": "2026-01-01",
                    "test_size": 250,
                    "fold_count": 13,
                    "step_size": 20,
                },
            },
        )

        self.assertIn("12/11/2026", summary)
        self.assertIn("1,250", summary)
        self.assertIn("KHÔNG TĂNG", summary)
        self.assertIn("Accuracy 65.00%, Precision 61.00%", summary)
        self.assertIn("F1 62.00%", summary)
        self.assertIn("MAE 0.8500%, RMSE 1.1200%", summary)
        self.assertIn("Độ phủ khoảng ước lượng: 89.00%", summary)
        self.assertIn("mỗi đợt tối đa 20 phiên", summary)

    def test_summary_explains_when_model_results_are_unavailable(self):
        summary = build_project_summary(
            "12/11/2026",
            1250,
            None,
        )

        self.assertIn("Chưa có mô hình đã huấn luyện", summary)
        self.assertIn("## Giới hạn", summary)


if __name__ == "__main__":
    unittest.main()
