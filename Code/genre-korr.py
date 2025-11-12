# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:42:57 2025

@author: Alican
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === 1️⃣ Veri yükle
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# === 2️⃣ Popularity sınıfı yoksa oluştur
if "popularity_class" not in data.columns:
    data["popularity_class"] = pd.cut(
        data["track_popularity"],
        bins=[-1, 33, 66, 100],
        labels=["low", "medium", "high"]
    )

# === 3️⃣ Zaman temelli feature engineering
def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except:
        return None

data["release_year"] = data["track_album_release_date"].apply(extract_year)
current_year = datetime.now().year
data["song_age"] = current_year - data["release_year"]

# On yıllık aralık (örneğin 1990s, 2000s)
def get_decade(year):
    if pd.isna(year):
        return "Unknown"
    decade_start = int(year // 10 * 10)
    return f"{decade_start}s"

data["song_decade"] = data["release_year"].apply(get_decade)

# 🎯 Hedef sayısallaştır
target_map = {'low': 0, 'medium': 1, 'high': 2}
data["popularity_num"] = data["popularity_class"].map(target_map)

# === 4️⃣ Tür listesi
genres = ["pop", "rap", "rock", "r&b", "edm"]

# === 5️⃣ Korelasyon analizi fonksiyonu
def correlation_analysis(df, genre_name):
    df = df.copy()
    df = df.dropna(subset=["popularity_class"])
    df["popularity_num"] = df["popularity_class"].map(target_map)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["popularity_num", "track_popularity"]]

    # 🎯 Feature vs Target korelasyon
    corr_target = df[numeric_cols].corrwith(df["popularity_num"]).sort_values(ascending=False)
    print(f"\n==============================")
    print(f"🎵 Genre: {genre_name.upper()} — Feature vs Target Correlation")
    print("==============================")
    print(corr_target.round(3))

    # 🔥 Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation Heatmap — {genre_name.upper()}")
    plt.tight_layout()
    plt.show()

    # 🔝 En anlamlı 10 feature
    top_corr = corr_target.abs().sort_values(ascending=False).head(10)
    top_corr.plot(kind="barh", color="teal")
    plt.title(f"Top 10 Correlated Features — {genre_name.upper()}")
    plt.xlabel("Correlation with Popularity Class")
    plt.show()

    return corr_target

# === 6️⃣ Tür bazlı analiz
results = {}
for g in genres:
    df_g = data[data["playlist_genre"].str.lower() == g]
    if len(df_g) < 100:
        print(f"⚠️ {g.upper()} için yeterli veri yok, atlandı.")
        continue
    results[g] = correlation_analysis(df_g, g)

print("\n✅ Yeni zaman temelli feature'lar eklendi:")
print(data[["track_album_release_date", "release_year", "song_age", "song_decade"]].head())


