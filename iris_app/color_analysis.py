from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from iris_app.models import RenkSonucu

GOZ_RENKLERI = [
    {"ad": "Koyu/Siyah", "hex": "#1A0F00", "h_min": 0, "h_max": 360, "s_min": 0, "s_max": 45, "v_min": 0, "v_max": 30},
    {"ad": "Kahverengi", "hex": "#6B3D1E", "h_min": 10, "h_max": 40, "s_min": 30, "s_max": 100, "v_min": 20, "v_max": 55},
    {"ad": "Acik Kahve", "hex": "#A0622A", "h_min": 18, "h_max": 45, "s_min": 25, "s_max": 85, "v_min": 50, "v_max": 85},
    {"ad": "Ela (Findik)", "hex": "#7B5E3A", "h_min": 25, "h_max": 55, "s_min": 20, "s_max": 70, "v_min": 35, "v_max": 70},
    {"ad": "Amber", "hex": "#C87A20", "h_min": 28, "h_max": 55, "s_min": 55, "s_max": 100, "v_min": 50, "v_max": 90},
    {"ad": "Yesil", "hex": "#4A7A3A", "h_min": 65, "h_max": 155, "s_min": 10, "s_max": 100, "v_min": 25, "v_max": 85},
    {"ad": "Mavi-Yesil", "hex": "#3A7A7A", "h_min": 150, "h_max": 200, "s_min": 10, "s_max": 100, "v_min": 25, "v_max": 85},
    {"ad": "Mavi", "hex": "#2A4A9A", "h_min": 195, "h_max": 260, "s_min": 8, "s_max": 100, "v_min": 25, "v_max": 90},
    {"ad": "Gri", "hex": "#7A8A9A", "h_min": 0, "h_max": 360, "s_min": 0, "s_max": 12, "v_min": 30, "v_max": 72},
]


class RenkYumusatici:
    def __init__(self, pencere: int = 15):
        self.pencere = pencere
        self.gecmis: list[str] = []

    def guncelle(self, renk_adi: str) -> str:
        self.gecmis.append(renk_adi)
        if len(self.gecmis) > self.pencere:
            self.gecmis.pop(0)
        return Counter(self.gecmis).most_common(1)[0][0]


def rgb_to_hsv(r: int, g: int, b: int) -> tuple:
    r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r_n, g_n, b_n)
    min_c = min(r_n, g_n, b_n)
    delta = max_c - min_c

    if delta == 0:
        h = 0.0
    elif max_c == r_n:
        h = 60 * (((g_n - b_n) / delta) % 6)
    elif max_c == g_n:
        h = 60 * (((b_n - r_n) / delta) + 2)
    else:
        h = 60 * (((r_n - g_n) / delta) + 4)

    s = 0.0 if max_c == 0 else (delta / max_c) * 100
    v = max_c * 100
    return round(h, 1), round(s, 1), round(v, 1)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def renk_esles(r: int, g: int, b: int) -> list[RenkSonucu]:
    h, s, v = rgb_to_hsv(r, g, b)
    sonuclar = []

    for renk in GOZ_RENKLERI:
        puan = 0
        
        # Gri ve Siyah için Hue 0-360 olduğundan onlara haksız puan vermiyoruz.
        # Bu renkler için sadece Doygunluk (S) ve Parlaklık (V) değerleri önemlidir.
        if renk["ad"] in ["Gri", "Koyu/Siyah"]:
            if renk["s_min"] <= s <= renk["s_max"]:
                puan += 50
            if renk["v_min"] <= v <= renk["v_max"]:
                puan += 50
        else:
            # Gerçek renkler için Hue (Renk Özü) en belirleyici faktördür.
            h_min = renk["h_min"]
            h_max = renk["h_max"]
            
            if h_min <= h_max:
                if h_min <= h <= h_max:
                    puan += 50
            else:
                if h >= h_min or h <= h_max:
                    puan += 50
                    
            if renk["s_min"] <= s <= renk["s_max"]:
                puan += 25
            if renk["v_min"] <= v <= renk["v_max"]:
                puan += 25

        sonuclar.append(
            RenkSonucu(
                ad=renk["ad"],
                hex_kodu=renk["hex"],
                skor=float(puan),
                rgb=(r, g, b),
                hsv=(h, s, v),
            )
        )

    sonuclar.sort(key=lambda sonuc: sonuc.skor, reverse=True)
    return sonuclar


def iris_ortalama_renk(iris_img: np.ndarray) -> tuple[int, int, int]:
    # CLAHE ile ışık normalizasyonu (Lab renk uzayında L kanalına uygulama)
    lab = cv2.cvtColor(iris_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    n_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    h, w = n_img.shape[:2]
    maske = np.zeros((h, w), dtype=np.uint8)
    dis_r = int(min(w, h) * 0.28)  # 0.45 yerine 0.28 yaptık (Ten rengi dışarıda kaldı)
    ic_r = int(dis_r * 0.45)       # 0.3 yerine 0.45 yaptık (Siyah gözbebeği dışarıda kaldı)
    
    cv2.circle(maske, (w // 2, h // 2), dis_r, 255, -1)
    cv2.circle(maske, (w // 2, h // 2), ic_r, 0, -1)

    b_ort = cv2.mean(n_img[:, :, 0], mask=maske)[0]
    g_ort = cv2.mean(n_img[:, :, 1], mask=maske)[0]
    r_ort = cv2.mean(n_img[:, :, 2], mask=maske)[0]

    return int(r_ort), int(g_ort), int(b_ort)