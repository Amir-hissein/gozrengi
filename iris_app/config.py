from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "face_landmarker.task"

IRIS_SOL_IDX = [468, 469, 470, 471, 472]
IRIS_SAG_IDX = [473, 474, 475, 476, 477]
