# ============================================================
#  STEP 0. 데이터 & 모델 준비
#  - Iris 데이터 로드
#  - RandomForest 모델 학습
#  - iris_model.pkl 저장
# ============================================================

from iris_common import PKL_PATH, CLASS_NAMES
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data():
    """Iris 데이터셋 로드 & 분리"""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   - 학습 샘플: {len(X_train)}개 / 테스트 샘플: {len(X_test)}개")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """RandomForest 모델 학습"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """모델 정확도 평가"""
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"   - 정확도: {acc * 100:.1f}%")
    print(f"   - 클래스: {CLASS_NAMES}")
    return acc


def save_model(model):
    """모델 pkl로 저장"""
    with open(PKL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"   - 저장 경로: {PKL_PATH}")


if __name__ == "__main__":
    print("=" * 55)
    print("  📌 [STEP 0] 데이터 & 모델 준비")
    print("=" * 55)

    print("\n▶ 데이터 로드 중...")
    X_train, X_test, y_train, y_test = load_data()

    print("\n▶ 모델 학습 중...")
    model = train_model(X_train, y_train)

    print("\n▶ 모델 평가 중...")
    evaluate_model(model, X_test, y_test)

    print("\n▶ 모델 저장 중...")
    save_model(model)

    print("\n✅ STEP 0 완료! → 다음: python iris_step1_onnx.py")