# 👁️ Göz Rengi Tespiti Uygulaması (Iris Color Detector)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-orange.svg)

Bu proje, görüntü işleme ve yapay zeka tekniklerini kullanarak yüz üzerindeki irisleri tespit eden ve göz rengini analiz eden gelişmiş bir uygulamadır. Kamera üzerinden canlı olarak veya statik görseller üzerinde çalışabilir.

## 🌟 Özellikler

- **Yüksek Doğruluk:** MediaPipe Face Mesh altyapısı ile hassas göz tespiti.
- **Renk Analizi:** Gelişmiş algoritmalar ile irisin merkez bölgesindeki gerçek rengi saptama.
- **Canlı Mod:** Webcam üzerinden eşzamanlı ve akıcı analiz imkanı (FPS göstergesi dahil).
- **Fotoğraf Modu:** Yüklenen görseller üzerinden detaylı analiz yapabilme.
- **Ekran Görüntüsü Alma:** Canlı modda analiz sonuçlarını anında kaydedebilme.

## 📸 Örnek Analiz Çıktısı

Aşağıda uygulamanın gerçek zamanlı kamera ve statik fotoğraflar üzerindeki analizine örnekler verilmiştir. Göz bebekleri algılanmış, yüz maskesi uygulanmış ve renk tespiti yapılarak ekrana yansıtılmıştır:

### 🎥 Canlı Kamera Modu
![Canlı Mod](iris_analiz_1783888915.png)

### 🖼️ Fotoğraf Analiz Modu
![Fotoğraf Modu](iris_analiz_resim_1783889392.png)

## 🛠️ Kurulum

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin:

1. **Gereksinimleri Yükleyin:**
   Kullanmakta olduğunuz ortama göre bağımlılıkları yükleyin. macOS ve bazı Linux dağıtımlarında `pip3` kullanmanız gerekebilir:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Projeyi Başlatın:**
   Kamera modunda (Canlı analiz) projeyi başlatmak için:
   ```bash
   python3 main.py
   ```

## 💻 Kullanım Kılavuzu

Uygulama başlatıldığında aşağıdaki seçeneklerle etkileşime geçebilirsiniz:

- **🖼️ Belirli Bir Fotoğraf ile Test Etmek İçin:**
  Projeyi bir görsel yolu belirterek çalıştırabilirsiniz:
  ```bash
  python3 main.py --resim yuz2.jpg
  ```
  *(Çıkmak için sonuç penceresindeyken herhangi bir tuşa basınız.)*

- **🛑 Projeden Çıkış Yapmak İçin (Canlı Mod):**  
  Klavyenizden `Q` tuşuna basın.

- **💾 Ekran Görüntüsü Almak İçin (Canlı Mod):**  
  Klavyenizden `S` tuşuna basarak anlık görüntüyü diskinize kaydedebilirsiniz.

---
*Geliştirici: Amir-Hissein*
