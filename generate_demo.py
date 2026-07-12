import cv2
from iris_app.app import analiz_et
from iris_app.iris_detection import IrisLandmarker
from iris_app.color_analysis import RenkYumusatici

def main():
    landmarker = IrisLandmarker()
    sol_yum = RenkYumusatici(15)
    sag_yum = RenkYumusatici(15)
    kare = cv2.imread("yuz2.jpg")
    if kare is None:
        print("Image not found")
        return
    h, w = kare.shape[:2]
    if w > 1280 or h > 720:
        scale = min(1280/w, 720/h)
        kare = cv2.resize(kare, (int(w*scale), int(h*scale)))

    sonuc = analiz_et(kare, landmarker, 100, sol_yum, sag_yum, mod="resim")
    cv2.imwrite("demo.jpg", sonuc)
    print("Saved demo.jpg")

if __name__ == "__main__":
    main()
