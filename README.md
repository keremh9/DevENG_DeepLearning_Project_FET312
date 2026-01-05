# Çelik Yüzey Kusuru Sınıflandırması (Steel Surface Defect Classification) - Final Projesi

**Ders:** Derin Öğrenme (Deep Learning) – Güz 2025-2026  
**Ekip Adı:** DevENG

Bu proje, **NEU-DET (Northeastern University Surface Defect Database)** veri setini kullanarak çelik yüzeyindeki üretim hatalarını tespit etmek ve sınıflandırmak amacıyla geliştirilmiştir. Projede hem sıfırdan tasarlanan **Baseline CNN** modelleri hem de **Transfer Learning** (ResNet18, EfficientNet, MobileNet) yöntemleri kullanılarak kapsamlı bir performans karşılaştırması yapılmıştır.

## 👥 Ekip Üyeleri

* **Mustafa ÖZBEZEK** - 23040301067
* **Mehmet Kerem HAKAN** - 23040301045

---

## 📂 Depo ve Dosya İçeriği (Repository Structure)

Final teslimi kapsamında yüklenen dosyaların açıklamaları aşağıdadır:

| Dosya Adı | Açıklama |
| :--- | :--- |
| `MEHMET_KEREM_HAKAN_23040301045_DevENG_ProjectReport.pdf` | **Final Proje Raporu** (Projenin tüm detaylarını, yöntemlerini ve sonuçlarını içeren ana rapor). |
| `MEHMET_KEREM_HAKAN_23040301045_DevENG_ProjectSunum.pdf` | **Proje Sunumu** (Sunum slaytları). |
| `MEHMET_KEREM_HAKAN_23040301045_DevENG.ipynb` | **Kerem'in Kod Dosyası** (SimpleCNN Baseline + ResNet18 Frozen & Fine-Tune modellerini içerir). |
| `MUSTAFA_OZBEZEK_23040301067_DevENG.ipynb` | **Mustafa'nın Kod Dosyası** (SimpleCNN Baseline + EfficientNet-B0 + MobileNetV3 modellerini içerir). |
| `MUSTAFA_OZBEZEK_23040301067_DevENG_ProjectReport.pdf` | (Yedek) Final Proje Raporu kopyası. |

---

## 🚀 Kullanılan Modeller ve Yöntemler

Projede farklı karmaşıklık seviyelerine sahip aşağıdaki mimariler test edilmiştir:

### 1. Baseline Modeller (Özgün Mimariler)
* **SimpleCNN (Kerem):** Düşük parametreli, 3 bloklu CNN yapısı.
* **SimpleCNN (Mustafa):** 5x5 kernel boyutuna sahip, geniş alanda doku analizi yapan CNN yapısı.

### 2. Gelişmiş Modeller (Transfer Learning)
* **ResNet18 (Kerem):** Hem "Frozen" (sadece son katman eğitimi) hem de "Fine-Tuning" (tüm ağın eğitimi) stratejileri ile denenmiştir.
* **EfficientNet-B0 (Mustafa):** Model parametre ve doğruluk dengesi için kullanılmıştır.
* **MobileNetV3-Small (Mustafa):** Hız ve verimlilik odaklı hafif mimari.

---

## 🏆 Performans Sonuçları (Test Seti)

Elde edilen en iyi sonuçlar aşağıda özetlenmiştir:

| Model | Doğruluk (Accuracy) | Macro F1-Score |
| :--- | :--- | :--- |
| **ResNet18 (Fine-Tune)** | **%100.00** | **1.0000** |
| EfficientNet-B0 | %97.78 | 0.9775 |
| ResNet18 (Frozen) | %95.83 | 0.9575 |
| MobileNetV3-Small | %93.33 | 0.9337 |
| SimpleCNN (Ortalama) | ~%82.50 | ~0.5600 |

> **Sonuç:** Transfer Learning yöntemleri, özellikle `ResNet18 (Fine-Tune)` stratejisi, veri setindeki sınıfları ayırt etmede %100 başarı sağlayarak en iyi performansı göstermiştir.

---

## 🎥 Proje Videosu

Projenin detaylı anlatım videosuna aşağıdaki linkten ulaşabilirsiniz:

[👉 **YouTube Video Linki İçin Tıklayın**](https://youtu.be/ICsRilibHCc?si=TV0-5lYGvXSjEVBN)

---

## 🛠️ Kurulum ve Gereksinimler

Kodları çalıştırmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:

```bash
pip install torch torchvision matplotlib scikit-learn pandas numpy seaborn
