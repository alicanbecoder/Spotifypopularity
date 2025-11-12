# 🎵 Spotify Song Popularity Prediction

Bu proje, Spotify şarkı verilerini kullanarak bir parçanın **ne kadar popüler olacağını (Low / Medium / High)** tahmin etmeyi amaçlayan bir makine öğrenmesi çalışmasıdır.  
Model, **ses, sanatçı, zaman ve playlist tabanlı** özellikleri bir araya getirerek, XGBoost algoritmasıyla **yaklaşık %81 doğruluk (Macro F1: 0.816)** elde etmiştir.

---

## 📘 İçerik
- 🎯 **Proje Amacı**
- ⚙️ **Veri Seti**
- 🧠 **Modelleme Süreci**
- 📊 **Sonuçlar**
- 💻 **GUI Prototipi (Streamlit)**
- 🔍 **Gelecek Çalışmalar**

---

## 🎯 Proje Amacı
Spotify ekosistemindeki parçaların popülerlik seviyelerini tahmin etmek için;  
yalnızca ses özelliklerini değil, **sanatçı üretkenliği**, **yayın yılı**, **playlist etkisi** ve **müzikal profil bileşenlerini (PCA)** de içeren kapsamlı bir yaklaşım geliştirilmiştir.

---

## ⚙️ Veri Seti
Kaynak: [TidyTuesday Spotify Dataset (2020)](https://github.com/rfordatascience/tidytuesday/blob/main/data/2020/2020-01-21/readme.md)  
Toplam **32.000+ şarkı**, aşağıdaki temel sütunları içerir:

| Değişken | Açıklama |
|-----------|-----------|
| `track_name`, `track_artist` | Şarkı ve sanatçı bilgileri |
| `playlist_genre`, `playlist_subgenre` | Tür ve alt tür |
| `danceability`, `energy`, `valence` | Ses/müzikal özellikler |
| `track_popularity` | Spotify popülerlik skoru (0–100) |

---

## 🧠 Modelleme Süreci

**Kullanılan Özellik Grupları:**
- 🎵 *Audio*: danceability, energy, valence, tempo, acousticness  
- 🕒 *Temporal*: release_year, is_2010s, is_recent_era  
- 🎧 *Artist Intelligence*: artist_track_count, genre_diversity, career_length, exposure_score  
- 📜 *Playlist Sinyalleri*: playlist_size, playlist_count, is_editorial  
- 🧩 *Müzikal PCA Bileşenleri*: 3 bileşen ile boyut indirgeme  

**Model:**  
XGBoost Classifier (n_estimators=900, max_depth=9, learning_rate=0.04)

---

## 📊 Sonuçlar

| Metric | Score |
|--------|--------|
| Accuracy | **0.809** |
| Macro F1 | **0.816** |

**Confusion Matrix & SHAP Analizi** → Model, özellikle “Medium” sınıfında yüksek genelleme başarısı göstermektedir.

---

## 💻 Streamlit GUI (Prototip)
Proje kapsamında, kullanıcıların şarkı özelliklerini girerek anında popülerlik sınıfı tahmini alabileceği bir **Streamlit tabanlı arayüz** geliştirilmiştir.  
Arayüz, modelin gerçek zamanlı yorumlanabilirliğini göstermeyi amaçlamaktadır.

```bash
streamlit run app.py
```

---

## 🔍 Gelecek Çalışmalar
- 🎤 Sanatçı popülerliği ve Spotify takipçi sayısı entegrasyonu  
- 🌍 Google Trends tabanlı müzik ilgisi eklemesi  
- 🧮 Müzikal embedding temelli “genre similarity” metrikleri  
- 📱 Tam entegre web arayüzü (Spotify API bağlantısı)

---

## 👨‍💻 Geliştirici
**Alican Tunç**  
📧 alicanbecoder@gmail.com  
🔗 [GitHub](https://github.com/alicanbecoder) • [LinkedIn](https://linkedin.com/in/alican-tunc-776178165)
