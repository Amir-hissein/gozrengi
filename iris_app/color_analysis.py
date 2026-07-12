from __future__ import annotations
from collections import Counter
import cv2
import numpy as np
from iris_app.models import RenkSonucu

# Referans Renkler - Gerçekçi RGB Uzayında (Kamera için kalibre edilmiş)
STANDART_RENKLER = [
    {"ad": "Koyu/Siyah", "hex": "#1A0F00", "rgb": (35, 25, 20)},
    {"ad": "Kahverengi", "hex": "#6B3D1E", "rgb": (85, 50, 30)},
    {"ad": "Acik Kahve", "hex": "#A0622A", "rgb": (135, 85, 45)},
    {"ad": "Ela (Findik)", "hex": "#7B5E3A", "rgb": (120, 105, 60)},
    {"ad": "Amber", "hex": "#C87A20", "rgb": (170, 115, 35)},
    {"ad": "Yesil", "hex": "#4A7A3A", "rgb": (65, 115, 55)},
    {"ad": "Mavi-Yesil", "hex": "#3A7A7A", "rgb": (55, 105, 115)},
    {"ad": "Mavi", "hex": "#2A4A9A", "rgb": (45, 65, 135)},
    {"ad": "Gri", "hex": "#7A8A9A", "rgb": (115, 125, 135)},
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

    c_detected = np.uint8([[[b, g, r]]])
    lab_detected = cv2.cvtColor(c_detected, cv2.COLOR_BGR2LAB)[0][0].astype(float)

    for renk in STANDART_RENKLER:
        rr, gg, bb = renk["rgb"]
        c_target = np.uint8([[[bb, gg, rr]]])
        lab_target = cv2.cvtColor(c_target, cv2.COLOR_BGR2LAB)[0][0].astype(float)
        
        mesafe = np.sqrt(np.sum((lab_detected - lab_target)**2))
        skor = max(0.0, 100.0 - (mesafe * 1.5))

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
    h, w = iris_img.shape[:2]
    if h < 5 or w < 5:
        return (0, 0, 0)
        
    maske = np.zeros((h, w), dtype=np.uint8)
    dis_r = int(min(w, h) * 0.35)
    ic_r = int(dis_r * 0.45)
    cv2.circle(maske, (w // 2, h // 2), dis_r, 255, -1)
    cv2.circle(maske, (w // 2, h // 2), ic_r, 0, -1)
    
    piksel_indeksleri = np.where(maske == 255)
    pikseller = iris_img[piksel_indeksleri[0], piksel_indeksleri[1]]
    
    if len(pikseller) == 0:
        return (0, 0, 0)
        
    # Akıllı Filtreleme: Parlama (Reflections) ve gölgeleri (Shadows) dışla
    gecerli_pikseller = []
    for p in pikseller:
        b_val, g_val, r_val = p
        parlaklik = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
        if 25 < parlaklik < 210:
            gecerli_pikseller.append(p)
            
    if not gecerli_pikseller:
        gecerli_pikseller = pikseller
        
    gecerli_pikseller = np.float32(gecerli_pikseller)
    
    # Yapay Zeka (K-Means) ile en baskın asıl rengi bul
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    try:
        _, labels, centers = cv2.kmeans(gecerli_pikseller, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten())
        dominant_renk = centers[np.argmax(counts)]
        b_ort, g_ort, r_ort = dominant_renk
    except Exception:
        # Hata olursa basit ortalama al
        b_ort = np.mean([p[0] for p in gecerli_pikseller])
        g_ort = np.mean([p[1] for p in gecerli_pikseller])
        r_ort = np.mean([p[2] for p in gecerli_pikseller])
    
    # Doygunluğu hafif artır
    hsv_renk = cv2.cvtColor(np.uint8([[[b_ort, g_ort, r_ort]]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv_renk[1] = min(255, int(hsv_renk[1] * 1.25))
    canli_bgr = cv2.cvtColor(np.uint8([[hsv_renk]]), cv2.COLOR_HSV2BGR)[0][0]
    
    return int(canli_bgr[2]), int(canli_bgr[1]), int(canli_bgr[0])