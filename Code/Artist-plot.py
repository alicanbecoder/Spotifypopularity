# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:24:03 2025

@author: Alican
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1️⃣ Veri yükleme ---
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# --- 2️⃣ Artist bazlı özet istatistikler ---
# Her sanatçı için: toplam şarkı sayısı, ortalama popülerlik, tür çeşitliliği, playlist sayısı
artist_stats = (
    data.groupby("track_artist")
    .agg({
        "track_id": "count",                 # Kaç şarkısı var?
        "track_popularity": "mean",          # Ortalama popülerlik
        "playlist_genre": "nunique",         # Kaç farklı türde yer almış?
        "playlist_id": "nunique"             # Kaç farklı playlist'e girmiş?
    })
    .reset_index()
)

# Kolon isimlerini sadeleştir
artist_stats.columns = [
    "artist", 
    "artist_track_count", 
    "artist_mean_popularity", 
    "artist_genre_diversity", 
    "artist_playlist_count"
]

# --- 3️⃣ Görselleştirme ---
plt.figure(figsize=(12, 6))
sns.scatterplot(
    data=artist_stats,
    x="artist_track_count",
    y="artist_mean_popularity",
    size="artist_genre_diversity",
    hue="artist_playlist_count",
    palette="viridis",
    alpha=0.7,
    sizes=(40, 300)
)

# --- 4️⃣ Biçimlendirme ---
plt.title("Artist-Level Popularity Landscape", fontsize=14, weight="bold")
plt.xlabel("Number of Tracks (Üretilen Şarkı Sayısı)", fontsize=12)
plt.ylabel("Average Popularity (Ortalama Popülerlik)", fontsize=12)
plt.legend(
    title="Playlist Count (Liste Sayısı)",
    bbox_to_anchor=(1.05, 1), loc='upper left'
)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

