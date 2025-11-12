# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:36:37 2025

@author: Alican
"""

# ==============================================
# 🎧 Genre Dataset Builder (Album-ID Leak-Free)
# ==============================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# === 1. Ana veriyi yükle ===
DATA_PATH = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
OUT_PATH = r"C:\Users\Alican\Desktop\Projects\Spotify\data\genre_datasets"

os.makedirs(OUT_PATH, exist_ok=True)
data = pd.read_csv(DATA_PATH)

# === 2. Gereksiz kolonları kontrol et ===
required_cols = ["playlist_genre", "track_album_id", "track_popularity"]
for c in required_cols:
    if c not in data.columns:
        raise ValueError(f"❌ '{c}' sütunu bulunamadı — veri yapını kontrol et.")

# === 3. Popularity sınıflarını oluştur (3 sınıf)
if "popularity_class" not in data.columns:
    data["popularity_class"] = pd.cut(
        data["track_popularity"],
        bins=[-1, 33, 66, 100],
        labels=["low", "medium", "high"]
    )

# === 4. Genre listesi ===
genres = data["playlist_genre"].dropna().unique()
print(f"🎵 Toplam {len(genres)} tür bulundu: {genres}\n")

# === 5. Tür bazlı leak-free split ===
for genre in genres:
    subset = data[data["playlist_genre"] == genre].copy()
    print(f"🎧 Tür: {genre} ({len(subset)} şarkı)")

    if len(subset) < 300:
        print("⚠️ Yetersiz veri, atlanıyor.\n")
        continue

    # Albüm bazlı grup (leak-free)
    album_ids = subset["track_album_id"].dropna().unique()
    album_train, album_test = train_test_split(
        album_ids,
        test_size=0.2,
        random_state=42
    )

    train_df = subset[subset["track_album_id"].isin(album_train)]
    test_df  = subset[subset["track_album_id"].isin(album_test)]

    print(f"  → Train: {train_df.shape}, Test: {test_df.shape}")

    # CSV olarak kaydet
    train_df.to_csv(os.path.join(OUT_PATH, f"{genre}_train.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_PATH, f"{genre}_test.csv"), index=False)

print("\n✅ Tür bazlı album-id split tamamlandı!")
print(f"📁 Kaydedilen dosyalar: {OUT_PATH}")
