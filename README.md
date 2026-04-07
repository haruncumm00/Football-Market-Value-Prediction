# Football Market Value Prediction ⚽💰

Bu proje, Avrupa'nın Top 5 ligindeki futbolcuların performans verilerini kullanarak piyasa değerlerini tahmin eden bir yapay zeka modelidir.

## Teknik Detaylar
- **Algoritma:** Çoklu Doğrusal Regresyon (Linear Regression)
- **Optimizasyon:** Adam Optimizer (Sıfırdan koda döküldü)
- **Kütüphaneler:** NumPy, Pandas
- **Hata Oranı (MSE Loss):** 1.01

## Nasıl Çalışır?
Model; yaş, gol, asist ve oynanan dakika gibi verileri normalize eder ve Adam Optimizer kullanarak en doğru ağırlıkları hesaplar. Örnek bir tahminde 25 yaşında, 
iyi istatistikli bir oyuncu için yaklaşık **4 Milyon €** değer biçmektedir.
## Veri Seti (Dataset)
Bu projede kullanılan veriler Kaggle üzerinden alınmıştır:
[Kaggle - Football Data from Transfermarkt](https://www.kaggle.com/datasets/davidcarelo/football-data-from-transfermarkt)
