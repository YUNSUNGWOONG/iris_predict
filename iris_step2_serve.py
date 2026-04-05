# ============================================================
#  STEP 2. 모델 서빙 — ONNX 추론
#  - iris_model.onnx 로드
#  - 샘플 데이터 3개 추론 테스트
#  - 예측 결과 출력
# ============================================================

from iris_common import CLASS_NAMES, load_onnx_session
import numpy as np


# 테스트 샘플 (입력값, 정답)
TEST_CASES = [
    ([5.1, 3.5, 1.4, 0.2], "setosa"),
    ([6.7, 3.1, 4.7, 1.5], "versicolor"),
    ([6.3, 3.3, 6.0, 2.5], "virginica"),
]


def run_inference(session, input_name, features):
    """ONNX 세션으로 추론 실행"""
    inp  = np.array([features], dtype=np.float32)
    pred = session.run(None, {input_name: inp})[0][0]
    return CLASS_NAMES[pred]


def test_all(session, input_name):
    """전체 샘플 테스트 & 결과 출력"""
    passed = 0
    for features, expected in TEST_CASES:
        predicted = run_inference(session, input_name, features)
        ok = "✅" if predicted == expected else "❌"
        print(f"   {ok} 입력{features}")
        print(f"      예측: {predicted}  /  정답: {expected}")
        if predicted == expected:
            passed += 1

    print(f"\n   결과: {passed}/{len(TEST_CASES)} 정답")
    return passed


if __name__ == "__main__":
    print("=" * 55)
    print("  📌 [STEP 2] 모델 서빙 — ONNX 추론 테스트")
    print("=" * 55)

    print("\n▶ ONNX 모델 로드 중...")
    session    = load_onnx_session()
    input_name = session.get_inputs()[0].name
    print("   - 로드 완료")

    print("\n▶ 추론 테스트 실행 중...")
    passed = test_all(session, input_name)

    if passed == len(TEST_CASES):
        print("\n✅ STEP 2 완료! → 다음: uvicorn iris_step3_app:app --reload")
    else:
        print("\n⚠️  일부 예측 실패 — STEP 1부터 다시 확인해보세요")