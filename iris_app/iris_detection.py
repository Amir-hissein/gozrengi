from __future__ import annotations

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions

from iris_app.config import MODEL_PATH, MODEL_URL


def model_indir() -> None:
    if not MODEL_PATH.exists():
        print("  Model indiriliyor (ilk çalıştırma ~30 MB)...")
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        print("  Model indirildi.\n")
    else:
        print("  Model mevcut.\n")


class IrisLandmarker:
    def __init__(self) -> None:
        model_indir()
        base_opts = mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes())
        opts = FaceLandmarkerOptions(
            base_options=base_opts,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    def detect(self, frame_bgr, timestamp_ms: int):
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )
        return self.landmarker.detect_for_video(mp_img, timestamp_ms)

    def close(self) -> None:
        self.landmarker.close()
