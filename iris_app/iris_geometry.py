import cv2
import numpy as np
from typing import Optional

def iris_kirp(frame: np.ndarray, landmarks: list, iris_idx: list[int], padding: float = 1.5) -> Optional[tuple]:
    h_f, w_f = frame.shape[:2]
    noktalar = []
    for idx in iris_idx:
        if idx >= len(landmarks): return None
        lm = landmarks[idx]
        noktalar.append((int(lm.x * w_f), int(lm.y * h_f)))

    if not noktalar: return None

    mx, my = noktalar[0]
    yaricaplar = [np.sqrt((px - mx) ** 2 + (py - my) ** 2) for px, py in noktalar[1:]]
    r = max(int((np.mean(yaricaplar) if yaricaplar else 12) * padding), 20)

    x1, y1 = max(0, mx - r), max(0, my - r)
    x2, y2 = min(w_f, mx + r), min(h_f, my + r)
    
    kirpilmis = frame[y1:y2, x1:x2].copy()
    hk, wk = kirpilmis.shape[:2]

    if hk == 0 or wk == 0:
        return None

    # Halka Maskesi: Gözbebeğini (merkezi %20) dışarıda bırakır
    maske = np.zeros((hk, wk), dtype=np.uint8)
    dis_r = max(min(wk, hk) // 2 - 2, 1)
    ic_r = int(dis_r * 0.25) # Gözbebeği alanı tahmini
    
    cv2.circle(maske, (wk // 2, hk // 2), dis_r, 255, -1)
    cv2.circle(maske, (wk // 2, hk // 2), ic_r, 0, -1) # İçini boşalt
    
    sonuc = cv2.bitwise_and(kirpilmis, kirpilmis, mask=maske)
    return sonuc, (mx, my)