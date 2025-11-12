# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:23:49 2025

@author: Alican
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 1️⃣ Klasör ve tür listesi ===
base_dir = r"C:\Users\Alican\Desktop\Projects\Spotify\data"
genres = ["pop", "rap", "rock", "r&b", "edm"]

# === 2️⃣ Popularity sınıfını sayıya dönüştürme
target_map = {'low': 0, 'medium': 1, 'high': 2}

# === 3️⃣ Korelasyon analizi fonksiyonu ===
def correlation_analysis(df, genre_name):
    df = df.copy()
    if "popularity_class" not in df.columns and "track_popularity" in df.columns:
        df["popularity_class"] = pd.cut(
            df["track_popularity"], bins=[-1, 33, 66, 100],
            labels=["low", "medium", "high"]
        )

    df["popularity_num"] = df["popularity_class"].map(target_map)

    # sadece sayısal kolonlar
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["track_popularity"]]

    # 🎯 Feature vs Target korelasyonu
    corr_target = df[numeric_cols].corrwith(df["popularity_num"]).sort_values(ascending=False)
    print(f"\n==============================")
    print(f"🎵 Genre: {genre_name.upper()} — Feature vs Target Correlation")
    print("==============================")
    print(corr_target.round(3))

    # 🔥 Korelasyon heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0)
    plt.title(f"Feature Correlation Heatmap — {genre_name.upper()}", fontsize=14)
    plt.tight_layout()
    plt.show()

    # 🔝 En anlamlı 10 feature
    top_corr = corr_target.abs().sort_values(ascending=False).head(10)
    top_corr.plot(kind="barh", color="teal")
    plt.title(f"Top 10 Correlated Features — {genre_name.upper()}", fontsize=12)
    plt.xlabel("Correlation with Popularity")
    plt.tight_layout()
    plt.show()

    return corr_target

# === 4️⃣ Tür bazlı analiz döngüsü ===
results = {}
for g in genres:
    path = os.path.join(base_dir, f"spotify_{g.replace('&','and').replace(' ', '_')}_features.csv")
    if not os.path.exists(path):
        print(f"⚠️ {g.upper()} dosyası bulunamadı, atlanıyor -> {path}")
        continue

    df_g = pd.read_csv(path)
    results[g] = correlation_analysis(df_g, g)

print("\n✅ Tüm türler için korelasyon analizleri tamamlandı.")
