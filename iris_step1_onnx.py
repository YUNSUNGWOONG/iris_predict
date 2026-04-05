# ============================================================
#  STEP 1. 모델 경량화 — ONNX 변환
#  - iris_model.pkl 로드
#  - ONNX 포맷으로 변환
#  - iris_model.onnx 저장
#  - 크기 비교 출력
# ============================================================

from iris_common import PKL_PATH, ONNX_PATH, load_pkl_model
import os
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def convert_to_onnx(model):
    """sklearn 모델 → ONNX 변환 & 저장"""
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())


def compare_size():
    """pkl vs onnx 파일 크기 비교"""
    pkl_size  = os.path.getsize(PKL_PATH)  / 1024
    onnx_size = os.path.getsize(ONNX_PATH) / 1024
    diff      = ((onnx_size - pkl_size) / pkl_size) * 100

    print(f"   - 원본  (.pkl) : {pkl_size:.1f} KB")
    print(f"   - 변환후(.onnx): {onnx_size:.1f} KB")
    print(f"   - 크기 변화    : {diff:+.1f}%")


if __name__ == "__main__":
    print("=" * 55)
    print("  📌 [STEP 1] 모델 경량화 — ONNX 변환")
    print("=" * 55)

    print("\n▶ pkl 모델 로드 중...")
    model = load_pkl_model()
    print(f"   - {PKL_PATH} 로드 완료")

    print("\n▶ ONNX 변환 중...")
    convert_to_onnx(model)
    print(f"   - {ONNX_PATH} 저장 완료")

    print("\n▶ 파일 크기 비교:")
    compare_size()

    print("\n✅ STEP 1 완료! → 다음: python iris_step2_serve.py")