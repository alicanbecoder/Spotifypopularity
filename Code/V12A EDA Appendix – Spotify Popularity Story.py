# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:47:14 2025

@author: Alican
"""

# ================================================================
# 🎨 V12A EDA Appendix – Spotify Popularity Story
# ================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1️⃣ Veri yükleme ve ön hazırlık
# ------------------------------------------------
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

data["duration_min"] = data["duration_ms"] / 60000.0
data["release_year"] = data["track_album_release_date"].astype(str).str[:4].astype(float)
data["playlist_genre"] = data["playlist_genre"].str.lower()
data["popularity_class"] = pd.cut(
    data["track_popularity"], bins=[-1,50,75,100], labels=["low","medium","high"]
).astype(str)

sns.set(style="whitegrid", palette="crest")
plt.rcParams["axes.titlesize"] = 13

# ================================================================
# 2️⃣ Genre–Year–Popularity Heatmap
# ================================================================
plt.figure(figsize=(10,6))
pivot = data.pivot_table(
    values="track_popularity",
    index="playlist_genre",
    columns="release_year",
    aggfunc="mean"
)
sns.heatmap(pivot, cmap="crest", linewidths=0.5)
plt.title("🔥 Average Popularity by Genre and Release Year")
plt.xlabel("Release Year")
plt.ylabel("Playlist Genre")
plt.tight_layout()
plt.show()

# ================================================================
# 3️⃣ Valence vs Danceability by Genre
# ================================================================
top_genres = data["playlist_genre"].value_counts().index[:6]
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=data[data["playlist_genre"].isin(top_genres)],
    x="danceability", y="valence",
    hue="playlist_genre", alpha=0.6
)
plt.title("💃 Valence vs Danceability by Genre")
plt.xlabel("Danceability")
plt.ylabel("Valence (Positivity)")
plt.legend(title="Genre", bbox_to_anchor=(1.05,1), loc="upper left")
plt.tight_layout()
plt.show()

# ================================================================
# 4️⃣ Artist Track Count Distribution
# ================================================================
data["track_artist"] = data["track_artist"].astype(str).str.lower().str.strip()
artist_counts = data["track_artist"].value_counts()

plt.figure(figsize=(8,4))
sns.histplot(artist_counts, bins=40, color="teal", kde=False)
plt.title("🎤 Artist Productivity Distribution")
plt.xlabel("Number of Tracks per Artist")
plt.ylabel("Artist Count")
plt.xlim(0,50)
plt.tight_layout()
plt.show()

# ================================================================
# 5️⃣ Genre Audio Profile (Radar-style via line plot)
# ================================================================
audio_feats = ["danceability","energy","speechiness","acousticness",
               "instrumentalness","liveness","valence"]
genre_profile = data.groupby("playlist_genre")[audio_feats].mean().reset_index()
top5 = genre_profile.sort_values("energy", ascending=False).head(5)

plt.figure(figsize=(9,5))
for _, row in top5.iterrows():
    plt.plot(audio_feats, row[audio_feats], marker="o", label=row["playlist_genre"])
plt.title("Genre Audio Profiles")
plt.ylabel("Average Feature Value")
plt.xticks(rotation=30)
plt.legend(title="Genre")
plt.tight_layout()
plt.show()

# ================================================================
# 6️⃣ Popularity Class Distribution by Genre
# ================================================================
plt.figure(figsize=(10,5))
order = data["playlist_genre"].value_counts().index[:10]
sns.countplot(
    data=data[data["playlist_genre"].isin(order)],
    x="playlist_genre", hue="popularity_class", palette="crest"
)
plt.title("📊 Popularity Class Distribution by Genre (Top 10 Genres)")
plt.xlabel("Playlist Genre")
plt.ylabel("Track Count")
plt.xticks(rotation=30)
plt.legend(title="Popularity Class")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7,4))
genre_pop = data.groupby("playlist_genre")["track_popularity"].mean().sort_values(ascending=False)
sns.barplot(x=genre_pop.values, y=genre_pop.index, palette="viridis")
plt.title("Average Popularity by Genre")
plt.xlabel("Mean Track Popularity")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# ================================================================
# 🎨 Genre Similarity Map (PCA Based on Audio Features)
# ================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Tür bazında ortalama ses özellikleri
audio_feats = ["danceability","energy","speechiness","acousticness",
               "instrumentalness","liveness","valence","tempo"]
genre_means = data.groupby("playlist_genre")[audio_feats].mean()

# Normalize et
scaler = StandardScaler()
scaled = scaler.fit_transform(genre_means)

# PCA (2 boyuta indirgeme)
pca = PCA(n_components=2, random_state=42)
genre_pca = pca.fit_transform(scaled)

# DataFrame oluştur
genre_pca_df = pd.DataFrame(genre_pca, index=genre_means.index, columns=["PC1","PC2"])

# Plot
plt.figure(figsize=(7,6))
plt.scatter(genre_pca_df["PC1"], genre_pca_df["PC2"], s=100, alpha=0.8)

for genre, (x, y) in genre_pca_df.iterrows():
    plt.text(x+0.03, y, genre, fontsize=10, weight='bold')

plt.title("Genre Similarity Map (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("✅ V12A EDA Appendix tamamlandı – 6 görselleştirme üretildi.")
