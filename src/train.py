import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# Lấy đường dẫn của thư mục gốc dự án (ph-n-t-ch-gi-v-ng)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đặt đường dẫn tuyệt đối đến file dữ liệu
DATA_PATH = os.path.join(BASE_DIR, "data", "gold_data_process.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
def main():
    # 1. Đọc dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"❌ Chưa tìm thấy file dữ liệu tại {DATA_PATH}. Hãy kiểm tra lại thư mục data!")
        return

    print("📊 Đang đọc dữ liệu...")
    df = pd.read_csv(DATA_PATH)
    print("Các cột có trong dữ liệu:", df.columns.tolist())

    # 2. Xử lý dữ liệu & Chọn Feature (Thay đổi tên cột cho khớp với file của TV1)
    # Tạm thời giả định các cột đặc trưng và cột Target (1: Tăng, 0: Giảm)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Target', 'Close_Next']]
    
    X = df[feature_cols]
    y = df['Target']

    # 3. Chia tập dữ liệu (Train/Test) theo thứ tự thời gian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. Huấn luyện mô hình Random Forest
    print("🤖 Đang huấn luyện mô hình AI...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Đánh giá mô hình
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n================ KẾT QUẢ ĐÁNH GIÁ ================")
    print(f" Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    print(f" ROC-AUC Score:  {roc_auc_score(y_test, y_proba):.4f}")
    print("\nChi tiết Báo cáo Phân loại:")
    print(classification_report(y_test, y_pred))

    # 6. Lưu file mô hình vào thư mục models/
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ Đã xuất và lưu mô hình thành công tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()