import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'gold_data_processed.csv')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'gold_model.pkl')

def train_and_evaluate():
    print("==================================================")
    print("  HUẤN LUYỆN MÔ HÌNH ENSEMBLE STACKING CẤP ĐỘ QUANT ")
    print("==================================================")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Lỗi: Không tìm thấy file '{DATA_PATH}'!")
        return

    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    ignore_cols = ['Date', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    X = df[feature_cols]
    y = df['Target']

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    estimators = [
        ('et', ExtraTreesClassifier(n_estimators=150, max_depth=8, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.03, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, class_weight='balanced'))
    ]

    candidate_models = {
        'Extra Trees': estimators[0][1],
        'Gradient Boosting': estimators[1][1],
        'Random Forest': estimators[2][1],
        'Stacking Ensemble (Advanced)': StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
    }

    best_model = None
    best_name = ""
    best_f1 = -1.0
    best_metrics = {}

    print("\n--------------------------------------------------")
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"🔹 {name:30s} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            best_metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }

    print("--------------------------------------------------")
    print(f"🏆 MÔ HÌNH XUẤT SẮC NHẤT: {best_name} (F1-Score: {best_f1*100:.2f}%)")

    feature_importances = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importances = dict(sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True))
    elif hasattr(best_model, 'estimators_'):
        importances = best_model.estimators_[0].feature_importances_
        feature_importances = dict(sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_payload = {
        'model': best_model,
        'model_name': best_name,
        'feature_cols': feature_cols,
        'metrics': best_metrics,
        'feature_importances': feature_importances
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_payload, f)

    print(f"\n✅ Đã lưu file mô hình tại: {MODEL_PATH}")

if __name__ == '__main__':
    train_and_evaluate()