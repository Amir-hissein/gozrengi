from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from iris_app.models import RenkSonucu

# Referans Renkler - RGB Uzayında
STANDART_RENKLER = [
    {"ad": "Koyu/Siyah", "hex": "#1A0F00", "rgb": (26, 15, 0)},
    {"ad": "Kahverengi", "hex": "#6B3D1E", "rgb": (107, 61, 30)},
    {"ad": "Acik Kahve", "hex": "#A0622A", "rgb": (160, 98, 42)},
    {"ad": "Ela (Findik)", "hex": "#7B5E3A", "rgb": (123, 94, 58)},
    {"ad": "Amber", "hex": "#C87A20", "rgb": (200, 122, 32)},
    {"ad": "Yesil", "hex": "#4A7A3A", "rgb": (74, 122, 58)},
    {"ad": "Mavi-Yesil", "hex": "#3A7A7A", "rgb": (58, 122, 122)},
    {"ad": "Mavi", "hex": "#2A4A9A", "rgb": (42, 74, 154)},
    {"ad": "Gri", "hex": "#7A8A9A", "rgb": (122, 138, 154)},
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

    # Tespit edilen rengin LAB uzayına dönüşümü (Daha profesyonel algısal mesafe)
    c_detected = np.uint8([[[b, g, r]]])
    lab_detected = cv2.cvtColor(c_detected, cv2.COLOR_BGR2LAB)[0][0].astype(float)

    for renk in STANDART_RENKLER:
        rr, gg, bb = renk["rgb"]
        c_target = np.uint8([[[bb, gg, rr]]])
        lab_target = cv2.cvtColor(c_target, cv2.COLOR_BGR2LAB)[0][0].astype(float)
        
        # LAB uzayında Öklid mesafesi
        mesafe = np.sqrt(np.sum((lab_detected - lab_target)**2))
        
        # Daha yumuşak bir skor hesaplaması
        skor = max(0.0, 100.0 - (mesafe * 0.8))

        sonuclar.append(
            RenkSonucu(
                ad=renk["ad"],
                hex_kodu=renk["hex"],
                skor=float(skor),
                rgb=(r, g, b),
                hsv=(h, s, v),
            )
        )

    sonuclar.sort(key=lambda sonuc: sonuc.skor, reverse=True)
    return sonuclar


def iris_ortalama_renk(iris_img: np.ndarray) -> tuple[int, int, int]:
    # CLAHE ile ışık normalizasyonu (Renkleri çok yıkamamak için clipLimit düşürüldü)
    lab = cv2.cvtColor(iris_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
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