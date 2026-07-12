from __future__ import annotations

import cv2

from iris_app.color_analysis import rgb_to_hex
from iris_app.models import GozBilgisi, RenkSonucu

YESIL = (0, 220, 120)
BEYAZ = (240, 240, 240)
GRI = (140, 140, 140)

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

def yuz_maskesi_ciz(frame, landmarks, h_f: int, w_f: int) -> None:
    # Örümcek ağı (tessellation) stili - ince ve profesyonel
    ag_stili = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=1, circle_radius=0)
    nokta_stili = mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=0, circle_radius=0) # Noktaları gizle
    
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=nokta_stili,
        connection_drawing_spec=ag_stili)


def fps_goster(frame, fps: float) -> None:
    w = frame.shape[1]
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YESIL, 1, cv2.LINE_AA)


def durum_cubugu(frame, mesaj: str, renk=YESIL) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 32), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, mesaj, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, renk, 1, cv2.LINE_AA)


def iris_panel(frame, iris_img, bilgi: GozBilgisi, px: int, py: int) -> None:
    pw, boyut = 120, 70
    cv2.rectangle(frame, (px, py), (px + pw, py + 125), (22, 22, 22), -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + 125), (65, 65, 65), 1)

    try:
        kucuk = cv2.resize(iris_img, (boyut, boyut))
        cx_off = px + (pw - boyut) // 2
        frame[py + 6 : py + 6 + boyut, cx_off : cx_off + boyut] = kucuk
    except Exception:
        pass

    r, g, b = bilgi.ortalama_rgb
    h, s, v = bilgi.renk_sonucu.hsv

    satirlar = [
        (f"{bilgi.taraf} Goz", GRI, py + boyut + 20),
        (bilgi.renk_sonucu.ad, YESIL, py + boyut + 36),
        (f"RGB({r},{g},{b})", GRI, py + boyut + 52),
        (f"H:{h:.0f} S:{s:.0f} V:{v:.0f}", GRI, py + boyut + 68),
    ]
    for metin, renk_c, y_pos in satirlar:
        cv2.putText(frame, metin, (px + 6, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.32, renk_c, 1, cv2.LINE_AA)


def orta_sonuc_kutusu(frame, en_iyi: RenkSonucu, ort_r: int, ort_g: int, ort_b: int) -> None:
    h_f, w_f = frame.shape[:2]
    
    # HEX kodunu ve OpenCV'nin renk formatı olan BGR'yi ayarlıyoruz (Eksik olan satır buydu)
    hex_kodu = rgb_to_hex(ort_r, ort_g, ort_b)
    bgr = (int(ort_b), int(ort_g), int(ort_r)) 

    # Ekran genişliğine göre dinamik boyutlandırma
    kw = int(w_f * 0.35)
    kh = 65
    kx = (w_f - kw) // 2
    ky = int(h_f * 0.85)

    # Arka plan ve çerçeveler
    cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (25, 25, 25), -1)
    cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), (75, 75, 75), 1)
    
    # Rengin gösterildiği küçük kare (İşte burada bgr değişkeni kullanılıyor)
    cv2.rectangle(frame, (kx + 8, ky + 8), (kx + 46, ky + 52), bgr, -1)
    cv2.rectangle(frame, (kx + 8, ky + 8), (kx + 46, ky + 52), (110, 110, 110), 1)

    # Yazılar
    cv2.putText(frame, f"Goz Rengi: {en_iyi.ad}", (kx + 54, ky + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, BEYAZ, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{hex_kodu}  |  RGB({ort_r},{ort_g},{ort_b})  |  Skor: {en_iyi.skor:.0f}%",
        (kx + 54, ky + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        GRI,
        1,
        cv2.LINE_AA,
    )

def iris_cember_ciz(frame, landmarks, iris_idx: list[int], h_f: int, w_f: int) -> None:
    if iris_idx[0] >= len(landmarks):
        return

    mx = int(landmarks[iris_idx[0]].x * w_f)
    my = int(landmarks[iris_idx[0]].y * h_f)

    yaricaplar = []
    for idx in iris_idx[1:]:
        if idx < len(landmarks):
            px = int(landmarks[idx].x * w_f)
            py = int(landmarks[idx].y * h_f)
            yaricaplar.append(((px - mx) ** 2 + (py - my) ** 2) ** 0.5)

    r = int(sum(yaricaplar) / len(yaricaplar)) if yaricaplar else 10
    cv2.circle(frame, (mx, my), r, YESIL, 1)
    cv2.circle(frame, (mx, my), 3, YESIL, -1)
