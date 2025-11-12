# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 10:01:50 2025

@author: Alican
"""

# ================================================================
# Audio Feature Correlation Map (Spotify)
# ================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Veri yükleme
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# İlgili sütunlar
audio_features = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms", "track_popularity"
]

# Dakika cinsinden süre ekle (isteğe bağlı, görseli sadeleştirir)
data["duration_min"] = data["duration_ms"] / 60000

# Korelasyon matrisi oluşturma
corr = data[[
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "duration_min", "track_popularity"
]].corr()

# 🎨 Görsel stilleri
plt.figure(figsize=(8, 6))
sns.set(style="white", font_scale=1.0)

# Korelasyon ısı haritası
heatmap = sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.6,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "Correlation"},
    annot_kws={"size": 9}
)

# Başlık ve düzen
plt.title("Audio Feature Correlation Map", fontsize=13, weight="bold", pad=15)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

