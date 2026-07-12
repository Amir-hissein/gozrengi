from __future__ import annotations

import cv2

from iris_app.color_analysis import rgb_to_hex
from iris_app.models import GozBilgisi, RenkSonucu

YESIL = (0, 220, 120)
BEYAZ = (240, 240, 240)
GRI = (140, 140, 140)

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing

_mask_anim_progress = 0.0

def yuz_maskesi_reset():
    global _mask_anim_progress
    _mask_anim_progress = 0.0

def yuz_maskesi_ciz(frame, landmarks, h_f: int, w_f: int, mod: str = "canli") -> None:
    global _mask_anim_progress
    
    if mod == "resim":
        progress = 1.0
    else:
        _mask_anim_progress = min(_mask_anim_progress + 0.025, 1.0) # ~1.3 saniyede tamamlanır
        progress = _mask_anim_progress
    
    # #0097b2 rengi - BGR formatında (178, 151, 0)
    ag_stili = mp_drawing.DrawingSpec(color=(178, 151, 0), thickness=1, circle_radius=0)
    nokta_stili = mp_drawing.DrawingSpec(color=(178, 151, 0), thickness=0, circle_radius=0)
    
    # Göz çevresinde boşluk bırakmak için göz ve kaş noktalarını dışla
    dislanacak_noktalar = set()
    for kategori in [
        FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE,
        FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE,
        FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
        FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
        FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW,
        FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
    ]:
        for conn in kategori:
            dislanacak_noktalar.add(conn.start)
            dislanacak_noktalar.add(conn.end)

    tum_baglantilar = [conn for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION 
                       if conn.start not in dislanacak_noktalar and conn.end not in dislanacak_noktalar]
    
    # Animasyon için bağlantıları yukarıdan aşağıya sırala
    tum_baglantilar.sort(key=lambda conn: (landmarks[conn.start].y + landmarks[conn.end].y) / 2)
    
    limit = int(len(tum_baglantilar) * progress)
    
    if limit > 0:
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=tum_baglantilar[:limit],
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

def yuz_boyutu_goster(frame, width_px: int, height_px: int) -> None:
    h_f, w_f = frame.shape[:2]
    metin = f"Yuz Boyutu: {width_px} x {height_px} px"
    
    (tw, th), _ = cv2.getTextSize(metin, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    px, py = w_f - tw - 20, 30
    
    # Yarı saydam arka plan için overlay kullanımı daha profesyonel olur, ancak basit rectangle da iş görür
    cv2.rectangle(frame, (px - 10, py - 10), (px + tw + 10, py + th + 10), (25, 25, 25), -1)
    cv2.rectangle(frame, (px - 10, py - 10), (px + tw + 10, py + th + 10), (100, 100, 100), 1)
    
    # Yazı (Farmasötik Mavi ile)
    cv2.putText(frame, metin, (px, py + th), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (178, 151, 0), 1, cv2.LINE_AA)

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
