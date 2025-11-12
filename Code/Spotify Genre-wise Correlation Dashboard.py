# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:06:00 2025

@author: Alican
"""

# ================================================================
# 🎧 Spotify Genre-wise Correlation Dashboard
# ================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === 1️⃣ Veri yükle
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# === 2️⃣ Popularity sınıfı oluştur (eğer yoksa)
if "popularity_class" not in data.columns:
    data["popularity_class"] = pd.cut(
        data["track_popularity"],
        bins=[-1, 50, 75, 100],
        labels=["low", "medium", "high"]
    )

# === 3️⃣ Zaman bazlı feature’lar
def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except:
        return None

data["release_year"] = data["track_album_release_date"].apply(extract_year)
current_year = datetime.now().year
data["song_age"] = current_year - data["release_year"]

def get_decade(year):
    if pd.isna(year):
        return "Unknown"
    decade_start = int(year // 10 * 10)
    return f"{decade_start}s"

data["song_decade"] = data["release_year"].apply(get_decade)

# 🎯 Target’ı sayısal hale getir
target_map = {'low': 0, 'medium': 1, 'high': 2}
data["popularity_num"] = data["popularity_class"].map(target_map)

# === 4️⃣ Analiz yapılacak türler
genres = ["pop", "rap", "rock", "r&b", "edm"]

# === 5️⃣ Tür bazlı korelasyon analiz fonksiyonu
def correlation_analysis(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["track_popularity", "popularity_num"]]
    corr = df[numeric_cols].corrwith(df["track_popularity"]).sort_values(ascending=False)
    return corr, df[numeric_cols].corr()

# === 6️⃣ Grafik: çoklu heatmap + özet bar chart
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
results = {}

for i, g in enumerate(genres):
    df_g = data[data["playlist_genre"].str.lower() == g]
    if len(df_g) < 100:
        axes[i].set_visible(False)
        continue
    
    corr_target, corr_matrix = correlation_analysis(df_g)
    results[g] = corr_target

    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=axes[i], cbar=False)
    axes[i].set_title(f"{g.upper()} Genre", fontsize=12, pad=10)

# === 7️⃣ Özet bar chart (her tür için en anlamlı özellik)
corr_summary = []
for g, corr in results.items():
    top_feat = corr.abs().sort_values(ascending=False).head(1)
    corr_summary.append({
        "Genre": g,
        "Top_Feature": top_feat.index[0],
        "Correlation": top_feat.values[0]
    })

corr_df = pd.DataFrame(corr_summary).sort_values("Correlation", ascending=False)

# 7️⃣ bar chart altta (son panel)
sns.barplot(data=corr_df, x="Correlation", y="Genre", palette="crest", ax=axes[-1])
for i, row in corr_df.iterrows():
    axes[-1].text(row["Correlation"] + 0.005, i, f"{row['Top_Feature']}", fontsize=9, va='center')
axes[-1].set_title("Top Feature Correlation with Popularity by Genre")

# === 8️⃣ Genel başlık ve düzen
plt.suptitle("Genre-wise Audio Feature Correlations & Top Influencers", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()
