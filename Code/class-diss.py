# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:31:24 2025

@author: Alican
"""

# ================================================================
# 🎧 Spotify Songs Dataset – Exploratory Data Analysis (EDA)
# ================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1️⃣ Veri yükleme ve temel hazırlık
# ------------------------------------------------
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

data["duration_min"] = data["duration_ms"] / 60000.0

data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[-1, 50, 75, 100],
    labels=["low", "medium", "high"]
).astype(str)

data["playlist_genre"] = data["playlist_genre"].str.lower()

sns.set(style="whitegrid", palette="crest")
plt.rcParams["axes.titlesize"] = 13

# ------------------------------------------------
# 2️⃣ Popularity distribution
# ------------------------------------------------
plt.figure(figsize=(8,4))
sns.histplot(data["track_popularity"], bins=40, kde=True, color="teal")
plt.axvline(50, color="red", linestyle="--", label="Low / Medium sınırı (50)")
plt.axvline(75, color="orange", linestyle="--", label="Medium / High sınırı (75)")
plt.title("Track Popularity Distribution")
plt.xlabel("Track Popularity (0–100)")
plt.ylabel("Count")
plt.legend()
plt.show()

# ------------------------------------------------
# 3️⃣ Genre-wise popularity means
# ------------------------------------------------
plt.figure(figsize=(10,5))
genre_mean = data.groupby("playlist_genre")["track_popularity"].mean().sort_values(ascending=False)
sns.barplot(x=genre_mean.values, y=genre_mean.index, palette="crest")
plt.title("Average Popularity by Playlist Genre")
plt.xlabel("Mean Track Popularity")
plt.ylabel("Genre")
plt.show()

# ------------------------------------------------
# 4️⃣ Correlation among audio features
# ------------------------------------------------
audio_feats = ["danceability", "energy", "speechiness", "acousticness",
               "instrumentalness", "liveness", "valence", "tempo", "duration_min","track_popularity"]
plt.figure(figsize=(10,7))
sns.heatmap(data[audio_feats].corr(), cmap="crest", annot=True, fmt=".2f")
plt.title("🎛 Audio Feature Correlation Map")
plt.show()

# ------------------------------------------------
# 5️⃣ Boxplots by popularity class
# ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.boxplot(x="popularity_class", y="danceability", data=data, palette="crest", ax=axes[0])
axes[0].set_title("Danceability by Popularity Class")
axes[0].set_xlabel("Popularity Class")
axes[0].set_ylabel("Danceability")

sns.boxplot(x="popularity_class", y="energy", data=data, palette="crest", ax=axes[1])
axes[1].set_title("Energy by Popularity Class")
axes[1].set_xlabel("Popularity Class")
axes[1].set_ylabel("Energy")

plt.tight_layout()
plt.show()

# ------------------------------------------------
# 6️⃣ Popularity vs Release Year trend
# ------------------------------------------------
data["release_year"] = data["track_album_release_date"].astype(str).str[:4].astype(float)
plt.figure(figsize=(9,5))
sns.lineplot(x="release_year", y="track_popularity", data=data, ci=None, color="seagreen")
plt.title("Average Track Popularity by Release Year")
plt.ylabel("Mean Popularity")
plt.xlabel("Release Year")
plt.show()

print("✅ EDA tamamlandı – 6 görselleştirme üretildi.")
