
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.atomic_files import atomic_output_path

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "gold_data_processed.csv"
)
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gold_model.pkl")

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
WALK_FORWARD_STEP_SIZE = 20
PREDICTION_INTERVAL_LEVEL = 0.90


def temporal_split_indices(
    sample_count,
    train_fraction=TRAIN_FRACTION,
    validation_fraction=VALIDATION_FRACTION,
):
    """Return chronological train/validation boundaries; remainder is test."""
    if sample_count < 3:
        raise ValueError(
            "Cần ít nhất ba quan sát để chia dữ liệu theo thời gian."
        )

    if (
        not 0 < train_fraction < 1
        or not 0 < validation_fraction < 1
    ):
        raise ValueError(
            "Tỷ lệ train và validation phải nằm trong khoảng (0, 1)."
        )

    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            "Tổng tỷ lệ train và validation phải nhỏ hơn 1."
        )

    train_end = int(sample_count * train_fraction)
    validation_end = int(
        sample_count * (train_fraction + validation_fraction)
    )

    if (
        train_end < 1
        or validation_end <= train_end
        or validation_end >= sample_count
    ):
        raise ValueError(
            "Không đủ quan sát để tạo các tập train, validation và test."
        )

    return train_end, validation_end


def classification_metrics(y_true, y_pred):
    """Calculate consistent binary-classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }


def walk_forward_predict(
    model_factory,
    features,
    targets,
    start_index,
    step_size=WALK_FORWARD_STEP_SIZE,
):
    """Predict time blocks using only labels known before each block."""
    if not 0 < start_index < len(features):
        raise ValueError(
            "Điểm bắt đầu walk-forward phải nằm giữa tập dữ liệu."
        )

    if step_size < 1:
        raise ValueError(
            "Kích thước mỗi bước walk-forward phải lớn hơn 0."
        )

    predictions = np.empty(
        len(features) - start_index,
        dtype=int,
    )
    folds = []

    for block_start in range(
        start_index, len(features), step_size
    ):
        block_end = min(
            block_start + step_size, len(features)
        )

        model = model_factory()
        model.fit(
            features.iloc[:block_start],
            targets.iloc[:block_start],
        )

        predictions[
            block_start - start_index:block_end - start_index
        ] = model.predict(
            features.iloc[block_start:block_end]
        )

        folds.append({
            "train_end_index": block_start - 1,
            "test_start_index": block_start,
            "test_end_index": block_end - 1,
            "test_size": block_end - block_start,
        })

    return predictions, folds


def walk_forward_regression_predict(
    model_factory,
    features,
    targets,
    start_index,
    step_size=WALK_FORWARD_STEP_SIZE,
):
    """Predict numeric targets in chronological blocks."""
    if not 0 < start_index < len(features):
        raise ValueError(
            "Điểm bắt đầu walk-forward phải nằm giữa tập dữ liệu."
        )

    if step_size < 1:
        raise ValueError(
            "Kích thước mỗi bước walk-forward phải lớn hơn 0."
        )

    predictions = np.empty(
        len(features) - start_index,
        dtype=float,
    )
    folds = []

    for block_start in range(
        start_index, len(features), step_size
    ):
        block_end = min(
            block_start + step_size, len(features)
        )

        model = model_factory()
        model.fit(
            features.iloc[:block_start],
            targets.iloc[:block_start],
        )

        predictions[
            block_start - start_index:block_end - start_index
        ] = model.predict(
            features.iloc[block_start:block_end]
        )

        folds.append({
            "train_end_index": block_start - 1,
            "test_start_index": block_start,
            "test_end_index": block_end - 1,
            "test_size": block_end - block_start,
        })

    return predictions, folds


def regression_metrics(actual, predicted):
    """Summarize next-session return errors in percentage points."""
    return {
        "mae_pct": mean_absolute_error(actual, predicted),
        "rmse_pct": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
        "directional_accuracy": accuracy_score(
            np.asarray(actual) > 0,
            np.asarray(predicted) > 0,
        ),
    }


def _next_session_returns(dataframe):
    """Align known Target rows with their next-session percentage return."""
    target_is_known = dataframe["Target"].notna()

    next_returns = (
        (
            dataframe["Close"].shift(-1)
            / dataframe["Close"] - 1
        ) * 100
    ).loc[target_is_known].reset_index(drop=True)

    if next_returns.isna().any():
        raise ValueError(
            "Không thể căn chỉnh lợi suất phiên kế tiếp "
            "với các nhãn Target."
        )

    return next_returns


def _model_factories():
    return {
        "Extra Trees": lambda: ExtraTreesClassifier(
            n_estimators=150,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            random_state=42,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=150,
            max_depth=7,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def _return_model_factories():
    return {
        "Extra Trees Regressor": lambda: ExtraTreesRegressor(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": lambda: GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.03,
            loss="huber",
            random_state=42,
        ),
        "Random Forest Regressor": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=7,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_and_evaluate():
    print("=" * 50)
    print("  CHRONOLOGICAL MODEL VALIDATION")
    print("=" * 50)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu đã xử lý: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    if "Target" not in df.columns:
        raise ValueError("Dữ liệu không có cột Target.")

    ignore_cols = [
        "Date", "Target", "Open", "High",
        "Low", "Close", "Volume",
    ]
    feature_cols = [
        col for col in df.columns
        if col not in ignore_cols
    ]

    labeled_df = (
        df.dropna(subset=["Target"])
        .reset_index(drop=True)
    )

    if labeled_df.empty:
        raise ValueError(
            "Không có nhãn Target hợp lệ để huấn luyện."
        )

    X = labeled_df[feature_cols]
    y = labeled_df["Target"].astype(int)
    next_return = _next_session_returns(df)

    if len(next_return) != len(labeled_df):
        raise ValueError(
            "Không thể căn chỉnh lợi suất phiên kế tiếp "
            "với các nhãn Target."
        )

    train_end, validation_end = temporal_split_indices(
        len(labeled_df)
    )
    model_factories = _model_factories()

    validation_metrics = {}
    best_name = ""
    best_f1 = -1.0

    for name, model_factory in model_factories.items():
        model = model_factory()
        model.fit(X.iloc[:train_end], y.iloc[:train_end])

        predictions = model.predict(
            X.iloc[train_end:validation_end]
        )

        metrics = classification_metrics(
            y.iloc[train_end:validation_end],
            predictions,
        )
        validation_metrics[name] = metrics

        print(
            f"{name:20s} | validation F1: "
            f"{metrics['f1_score'] * 100:.2f}% "
            f"| accuracy: {metrics['accuracy'] * 100:.2f}%"
        )

        if metrics["f1_score"] > best_f1:
            best_name = name
            best_f1 = metrics["f1_score"]

    selected_factory = model_factories[best_name]
    print(f"\nSelected model on validation: {best_name}")

    return_factories = _return_model_factories()

    return_validation_metrics = {
        "No change": regression_metrics(
            next_return.iloc[train_end:validation_end],
            np.zeros(validation_end - train_end),
        )
    }

    best_return_name = "No change"
    best_return_mae = return_validation_metrics[
        "No change"
    ]["mae_pct"]

    for name, model_factory in return_factories.items():
        model = model_factory()
        model.fit(
            X.iloc[:train_end],
            next_return.iloc[:train_end],
        )

        predictions = model.predict(
            X.iloc[train_end:validation_end]
        )

        metrics = regression_metrics(
            next_return.iloc[train_end:validation_end],
            predictions,
        )

        return_validation_metrics[name] = metrics

        print(
            f"{name:28s} | validation MAE: "
            f"{metrics['mae_pct']:.4f}% "
            f"| direction accuracy: "
            f"{metrics['directional_accuracy'] * 100:.2f}%"
        )

        if metrics["mae_pct"] < best_return_mae:
            best_return_name = name
            best_return_mae = metrics["mae_pct"]

    return_factory = return_factories.get(best_return_name)

    if return_factory is None:
        return_oos_predictions = np.zeros(
            len(next_return) - validation_end
        )
        return_folds = []
        validation_return_predictions = np.zeros(
            validation_end - train_end
        )
        return_validation_model = None

    else:
        return_validation_model = return_factory()
        return_validation_model.fit(
            X.iloc[:train_end],
            next_return.iloc[:train_end],
        )

        validation_return_predictions = (
            return_validation_model.predict(
                X.iloc[train_end:validation_end]
            )
        )

        return_oos_predictions, return_folds = (
            walk_forward_regression_predict(
                return_factory,
                X,
                next_return,
                start_index=validation_end,
            )
        )

    validation_residuals = np.abs(
        next_return.iloc[train_end:validation_end].to_numpy()
        - validation_return_predictions
    )

    prediction_interval_pct = float(
        np.quantile(
            validation_residuals,
            PREDICTION_INTERVAL_LEVEL,
        )
    )

    return_test_metrics = regression_metrics(
        next_return.iloc[validation_end:],
        return_oos_predictions,
    )

    return_baseline_metrics = regression_metrics(
        next_return.iloc[validation_end:],
        np.zeros(len(next_return) - validation_end),
    )

    return_test_metrics["prediction_interval_level"] = (
        PREDICTION_INTERVAL_LEVEL
    )
    return_test_metrics["prediction_interval_half_width_pct"] = (
        prediction_interval_pct
    )
    return_test_metrics["prediction_interval_coverage"] = float(
        np.mean(
            np.abs(
                next_return.iloc[validation_end:].to_numpy()
                - return_oos_predictions
            ) <= prediction_interval_pct
        )
    )

    print(
        f"Walk-forward return | MAE: "
        f"{return_test_metrics['mae_pct']:.4f}% "
        f"| direction accuracy: "
        f"{return_test_metrics['directional_accuracy'] * 100:.2f}% "
        f"| interval coverage: "
        f"{return_test_metrics['prediction_interval_coverage'] * 100:.2f}%"
    )

    y_test = y.iloc[validation_end:]

    oos_predictions, folds = walk_forward_predict(
        selected_factory,
        X,
        y,
        start_index=validation_end,
    )

    test_metrics = classification_metrics(
        y_test, oos_predictions
    )

    baselines = {
        "always_up": classification_metrics(
            y_test,
            np.ones(len(y_test), dtype=int),
        ),
        "previous_day_direction": classification_metrics(
            y_test,
            (
                X.iloc[validation_end:]["Return"].to_numpy() > 0
            ).astype(int),
        ),
    }

    print(
        f"Walk-forward test | F1: "
        f"{test_metrics['f1_score'] * 100:.2f}% "
        f"| accuracy: {test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Baseline always up | F1: "
        f"{baselines['always_up']['f1_score'] * 100:.2f}% "
        f"| accuracy: "
        f"{baselines['always_up']['accuracy'] * 100:.2f}%"
    )

    print(
        "Baseline previous direction | F1: "
        f"{baselines['previous_day_direction']['f1_score'] * 100:.2f}% "
        f"| accuracy: "
        f"{baselines['previous_day_direction']['accuracy'] * 100:.2f}%"
    )

    oos_rows = labeled_df.iloc[validation_end:][["Date"]].copy()
    oos_rows["prediction"] = oos_predictions
    oos_rows["actual"] = y_test.to_numpy()

    final_model = selected_factory()
    final_model.fit(X, y)

    final_return_model = (
        return_factory()
        if return_factory is not None
        else None
    )

    if final_return_model is not None:
        final_return_model.fit(X, next_return)

    feature_importances = {}

    if hasattr(final_model, "feature_importances_"):
        feature_importances = dict(
            sorted(
                zip(
                    feature_cols,
                    final_model.feature_importances_,
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    model_payload = {
        "model": final_model,
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "return_model": final_return_model,
        "return_model_name": best_return_name,
        "return_validation_metrics": return_validation_metrics,
        "return_metrics": return_test_metrics,
        "return_baseline_metrics": return_baseline_metrics,
        "prediction_interval_pct": prediction_interval_pct,
        "metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "baseline_metrics": baselines,
        "walk_forward": {
            "fold_count": len(folds),
            "step_size": WALK_FORWARD_STEP_SIZE,
            "train_end_date": labeled_df.iloc[
                train_end - 1
            ]["Date"].strftime("%Y-%m-%d"),
            "validation_end_date": labeled_df.iloc[
                validation_end - 1
            ]["Date"].strftime("%Y-%m-%d"),
            "test_start_date": labeled_df.iloc[
                validation_end
            ]["Date"].strftime("%Y-%m-%d"),
            "test_end_date": labeled_df.iloc[-1][
                "Date"
            ].strftime("%Y-%m-%d"),
            "train_size": train_end,
            "validation_size": validation_end - train_end,
            "test_size": len(y_test),
        },
        "walk_forward_folds": folds,
        "return_walk_forward_folds": return_folds,
        "oos_predictions": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "prediction": int(row.prediction),
                "actual": int(row.actual),
            }
            for row in oos_rows.itertuples(index=False)
        ],
        "oos_return_predictions": [
            {
                "date": row.Date.strftime("%Y-%m-%d"),
                "actual_return_pct": float(row.actual_return_pct),
                "predicted_return_pct": float(
                    row.predicted_return_pct
                ),
            }
            for row in (
                labeled_df.iloc[validation_end:][["Date"]]
                .assign(
                    actual_return_pct=(
                        next_return.iloc[validation_end:].to_numpy()
                    ),
                    predicted_return_pct=return_oos_predictions,
                )
                .itertuples(index=False)
            )
        ],
        "feature_importances": feature_importances,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)

    with atomic_output_path(MODEL_PATH) as temporary_path:
        with open(temporary_path, "wb") as model_file:
            pickle.dump(model_payload, model_file)

    print(f"\nSaved current prediction model to: {MODEL_PATH}")
    print("Test metrics are out-of-sample walk-forward results.")


if __name__ == "__main__":
    train_and_evaluate()
