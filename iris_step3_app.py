# ============================================================
#  STEP 3. FastAPI 서버
#  - iris_model.onnx 로드
#  - POST /predict 엔드포인트 제공
#  - 실행: uvicorn iris_step3_app:app --reload
#  - 테스트: http://localhost:8000/docs
# ============================================================

from iris_common import CLASS_NAMES, load_onnx_session
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# ── 앱 초기화
app = FastAPI(title="🌸 붓꽃 분류 API", version="1.0")

# ── 모델 로드 (서버 시작 시 1회만 실행)
session    = load_onnx_session()
input_name = session.get_inputs()[0].name


# ── 입력 스키마
class IrisInput(BaseModel):
    sepal_length: float   # 꽃받침 길이
    sepal_width:  float   # 꽃받침 너비
    petal_length: float   # 꽃잎 길이
    petal_width:  float   # 꽃잎 너비


# ── 엔드포인트
@app.get("/")
def root():
    return {"status": "서버 정상 작동 중 🌸"}


@app.post("/predict")
def predict(data: IrisInput):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], dtype=np.float32)

    pred_idx  = session.run(None, {input_name: features})[0][0]
    pred_name = CLASS_NAMES[pred_idx]

    return {
        "prediction": pred_name,
        "class_id":   int(pred_idx),
        "input":      data.dict()
    }