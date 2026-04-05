# iris_common.py — 공통 설정 & 유틸리티
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# 클래스 이름 (전 파일 공통 사용)
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# 모델 파일 경로
PKL_PATH  = "iris_model.pkl"
ONNX_PATH = "iris_model.onnx"

def load_pkl_model():
    """pkl 모델 불러오기"""
    with open(PKL_PATH, "rb") as f:
        return pickle.load(f)

def load_onnx_session():
    """ONNX 세션 불러오기"""
    import onnxruntime as ort
    return ort.InferenceSession(ONNX_PATH)


    