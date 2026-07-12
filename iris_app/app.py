from __future__ import annotations

import argparse
import time
import sys
import cv2

from iris_app.color_analysis import RenkYumusatici, iris_ortalama_renk, renk_esles
from iris_app.config import IRIS_SAG_IDX, IRIS_SOL_IDX
from iris_app.iris_detection import IrisLandmarker
from iris_app.iris_geometry import iris_kirp
from iris_app.models import GozBilgisi
from iris_app.visualization import (
    durum_cubugu, 
    fps_goster, 
    iris_cember_ciz, 
    iris_panel, 
    orta_sonuc_kutusu,
    yuz_maskesi_ciz
)

def analiz_et(kare, landmarker, ms_sayaci, sol_yum, sag_yum, mod="canli"):
    """Hem resim hem de video karesi için ortak analiz mantığı."""
    h_k, w_k = kare.shape[:2]
    sonuc = landmarker.detect(kare, ms_sayaci)

    yuz_var = bool(
        sonuc.face_landmarks
        and len(sonuc.face_landmarks) > 0
        and len(sonuc.face_landmarks[0]) > 477
    )

    if yuz_var:
        landmarks = sonuc.face_landmarks[0]
        
        # Yüz boyutunu hesapla
        min_x = min(lm.x for lm in landmarks)
        max_x = max(lm.x for lm in landmarks)
        min_y = min(lm.y for lm in landmarks)
        max_y = max(lm.y for lm in landmarks)
        
        yuz_w = int((max_x - min_x) * w_k)
        yuz_h = int((max_y - min_y) * h_k)
        
        from iris_app.visualization import yuz_boyutu_goster
        yuz_boyutu_goster(kare, yuz_w, yuz_h)

        # Yüz maskesi (örümcek ağı) çizimi
        yuz_maskesi_ciz(kare, landmarks, h_k, w_k, mod)

        # İrisleri görselleştir
        iris_cember_ciz(kare, landmarks, IRIS_SOL_IDX, h_k, w_k)
        iris_cember_ciz(kare, landmarks, IRIS_SAG_IDX, h_k, w_k)

        # İris bölgelerini kırp
        sol = iris_kirp(kare, landmarks, IRIS_SOL_IDX)
        sag = iris_kirp(kare, landmarks, IRIS_SAG_IDX)

        sr, sg, sb = 0, 0, 0
        ar, ag, ab = 0, 0, 0

        if sol:
            sol_img, _ = sol
            sr, sg, sb = iris_ortalama_renk(sol_img)
            sol_renkler = renk_esles(sr, sg, sb)
            # Yumuşatma sadece canlı modda aktif
            final_ad = sol_yum.guncelle(sol_renkler[0].ad) if mod == "canli" else sol_renkler[0].ad
            
            iris_panel(
                kare,
                sol_img,
                GozBilgisi("Sol", sol_img, (sr, sg, sb), sol_renkler[0], (0, 0)),
                px=20,
                py=20,
            )

        if sag:
            sag_img, _ = sag
            ar, ag, ab = iris_ortalama_renk(sag_img)
            sag_renkler = renk_esles(ar, ag, ab)
            final_ad = sag_yum.guncelle(sag_renkler[0].ad) if mod == "canli" else sag_renkler[0].ad
            
            iris_panel(
                kare,
                sag_img,
                GozBilgisi("Sag", sag_img, (ar, ag, ab), sag_renkler[0], (0, 0)),
                px=w_k - 140,
                py=20,
            )

        if sol and sag:
            ort_r, ort_g, ort_b = (sr + ar) // 2, (sg + ag) // 2, (sb + ab) // 2
            en_iyi = renk_esles(ort_r, ort_g, ort_b)[0]
            orta_sonuc_kutusu(kare, en_iyi, ort_r, ort_g, ort_b)

        durum_cubugu(kare, "Analiz tamamlandi | q: Cikis | s: Kaydet")
    else:
        from iris_app.visualization import yuz_maskesi_reset
        yuz_maskesi_reset()
        durum_cubugu(kare, "Yuz bulunamadi - Kameraya yaklasin", renk=(80, 80, 200))
    
    return kare

def main() -> None:
    # Argüman kontrolü (Resim modu için)
    parser = argparse.ArgumentParser(description="Iris Renk Tespit Sistemi")
    parser.add_argument("--resim", type=str, help="Analiz edilecek resim dosyasının yolu")
    args = parser.parse_args()

    print("=" * 55)
    print("  IRIS RENK TESPİT SİSTEMİ (Gelişmiş Versiyon)")
    print("=" * 55)

    landmarker = IrisLandmarker()
    sol_yum = RenkYumusatici(15)
    sag_yum = RenkYumusatici(15)

    if args.resim:
        # --- RESİM ANALİZ MODU ---
        print(f"  Resim analiz ediliyor: {args.resim}")
        kare = cv2.imread(args.resim)
        if kare is None:
            print(f"[HATA] Resim yüklenemedi: {args.resim}")
            landmarker.close()
            return
        
        # Ekran boyutuna göre ölçekleme (Gerekirse)
        h, w = kare.shape[:2]
        if w > 1280 or h > 720:
            scale = min(1280/w, 720/h)
            kare = cv2.resize(kare, (int(w*scale), int(h*scale)))

        sonuc_kare = analiz_et(kare, landmarker, 100, sol_yum, sag_yum, mod="resim")
        cv2.imshow("Iris Analiz Sonucu (Resim)", sonuc_kare)
        print("  Kaydetmek için 's', çıkmak için 'q' veya başka bir tuşa basın...")
        while True:
            tus = cv2.waitKey(0) & 0xFF
            if tus == ord("s"):
                dosya = f"iris_analiz_resim_{int(time.time())}.png"
                cv2.imwrite(dosya, sonuc_kare)
                print(f"  Kaydedildi: {dosya}")
            else:
                break
    
    else:
        # --- CANLI KAMERA MODU ---
        kamera = cv2.VideoCapture(0)
        if not kamera.isOpened():
            print("[HATA] Kamera açılamadı!")
            landmarker.close()
            return

        kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("  Canlı analiz başlatılıyor... Çıkmak için 'q' tuşuna basın.")
        
        onceki_zaman = time.time()
        kare_sayaci = 0
        fps_degeri = 0.0
        ms_sayaci = 0

        while True:
            ret, kare = kamera.read()
            if not ret: break

            kare = cv2.flip(kare, 1)
            kare_sayaci += 1
            ms_sayaci += 33

            if kare_sayaci % 10 == 0:
                simdi = time.time()
                fps_degeri = 10 / max(simdi - onceki_zaman, 0.001)
                onceki_zaman = simdi

            # Analiz işlemini gerçekleştir
            kare = analiz_et(kare, landmarker, ms_sayaci, sol_yum, sag_yum, mod="canli")
            
            fps_goster(kare, fps_degeri)
            cv2.imshow("Iris Renk Tespiti | Canli", kare)

            tus = cv2.waitKey(1) & 0xFF
            if tus == ord("q"): break
            if tus == ord("s"):
                dosya = f"iris_analiz_{int(time.time())}.png"
                cv2.imwrite(dosya, kare)
                print(f"  Kaydedildi: {dosya}")

        kamera.release()

    cv2.destroyAllWindows()
    landmarker.close()
    print("  Uygulama kapatıldı.")

if __name__ == "__main__":
    main()