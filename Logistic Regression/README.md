# Öğrenci Depresyon Tahmini Projesi

Bu proje, öğrenci depresyon tahmini için iki farklı lojistik regresyon implementasyonunu içermektedir. Proje, Scikit-learn kütüphanesi ve sıfırdan NumPy implementasyonu olmak üzere iki farklı yaklaşımı karşılaştırmaktadır.

## Proje Yapısı

Proje iki ana Python dosyasından oluşmaktadır:

1. `Yzm212lab2_skitlearn.py`: Scikit-learn implementasyonu
2. `Yzm212lab2_scratch.py`: Sıfırdan NumPy implementasyonu

## Gereksinimler

```
numpy
pandas
matplotlib
scikit-learn
```

## Kullanım

1. Veri setini projenin ana dizinine yerleştirin:
   - `Student Depression Dataset.csv`

2. İlgili Python dosyalarını çalıştırın:
   ```bash
   python Yzm212lab2_skitlearn.py  # Scikit-learn versiyonu
   python Yzm212lab2_scratch.py  # Sıfırdan implementasyon
   ```

## Teknik Detaylar

### Veri Ön İşleme
- Eksik verilerin doldurulması
  - Kategorik veriler için mod
  - Sayısal veriler için ortalama
- Kategorik değişkenlerin sayısallaştırılması
- Özellik ölçeklendirme (standardizasyon)
- 80-20 eğitim-test bölünmesi

### Model Özellikleri

#### Scikit-learn Uygulaması
- Hazır LogisticRegression sınıfı
- LBFGS optimizasyonu
- L2 regularizasyonu
- Otomatik hiperparametre ayarı

#### Sıfırdan Uygulama
- Gradient Descent optimizasyonu
- Öğrenme oranı: 0.01
- İterasyon sayısı: 1000
- Sigmoid aktivasyon fonksiyonu

## Çıktılar

Her iki uygulama da aşağıdaki çıktıları üretir:
- Confusion Matrix görselleştirmesi
- Doğruluk skoru
- Eğitim ve test süreleri

## Karşılaştırma Sonuçları

### Avantajlar ve Dezavantajlar

**Scikit-learn:**
- ✓ Optimize edilmiş performans
- ✓ Daha az kod yazımı
- ✓ Güvenilir ve test edilmiş
- ✗ "Kara kutu" yaklaşımı
- ✗ Özelleştirme sınırlılığı

**Sıfırdan Uygulama:**
- ✓ Algoritmanın tam kontrolü
- ✓ Eğitim sürecinin şeffaflığı
- ✓ Öğrenme amaçlı değer daha yüksektir
- ✗ Daha düşük performans
- ✗ Optimizasyon eksikliği

