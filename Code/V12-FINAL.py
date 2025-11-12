# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 09:53:59 2025

@author: Alican
"""

# ================================================================
# 🎵 Spotify Popularity Prediction (V12-Final: Trend + PCA + Exposure Log + Report)
# ================================================================
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ------------------------------------------------
# 1️⃣ Veri yükleme
# ------------------------------------------------
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# ------------------------------------------------
# 2️⃣ Temel hazırlık
# ------------------------------------------------
data["duration_min"] = data["duration_ms"] / 60000.0
data["playlist_genre"] = data["playlist_genre"].str.lower()
data["track_artist"] = data["track_artist"].astype(str).str.lower().str.strip()

# Popularity sınıfı
data["popularity_class"] = pd.cut(
    data["track_popularity"], bins=[-1, 50, 75, 100],
    labels=["low", "medium", "high"]
).astype(str)

# ------------------------------------------------
# 3️⃣ Temporal özellikler
# ------------------------------------------------
data["release_year"] = data["track_album_release_date"].astype(str).str[:4].astype(float)
data["song_age"] = 2025 - data["release_year"]

data["is_pre_spotify_era"] = (data["release_year"] < 2010).astype(int)
data["is_2010s"] = ((data["release_year"] >= 2010) & (data["release_year"] < 2020)).astype(int)
data["is_post_2020"] = (data["release_year"] >= 2020).astype(int)

# ------------------------------------------------
# 4️⃣ Playlist bazlı sinyaller
# ------------------------------------------------
if "playlist_id" in data.columns:
    data["playlist_size"] = data.groupby("playlist_id")["track_id"].transform("count")
    data["playlist_count"] = data.groupby("track_id")["playlist_id"].transform("nunique")
else:
    data["playlist_size"] = data.groupby("playlist_genre")["track_name"].transform("count")
    data["playlist_count"] = data.groupby("track_name")["playlist_genre"].transform("nunique")

if "playlist_name" in data.columns:
    data["is_editorial"] = data["playlist_name"].str.contains("editorial|spotify", case=False, na=False).astype(int)
else:
    data["is_editorial"] = 0

# ------------------------------------------------
# 5️⃣ Artist Intelligence Features (popularity-free)
# ------------------------------------------------
data["artist_track_count"] = data.groupby("track_artist")["track_id"].transform("count")

feature_cols = ["danceability", "energy", "valence", "tempo", "acousticness"]
artist_means = data.groupby("track_artist")[feature_cols].mean().add_prefix("artist_mean_")
data = data.merge(artist_means, on="track_artist", how="left")

data["artist_genre_diversity"] = data.groupby("track_artist")["playlist_genre"].transform("nunique")

artist_year_stats = data.groupby("track_artist")["release_year"].agg(["min", "max"]).reset_index()
artist_year_stats.columns = ["track_artist", "artist_first_year", "artist_last_year"]
data = data.merge(artist_year_stats, on="track_artist", how="left")
data["artist_career_length"] = data["artist_last_year"] - data["artist_first_year"]

if "playlist_id" in data.columns:
    data["artist_playlist_count"] = data.groupby("track_artist")["playlist_id"].transform("nunique")
else:
    data["artist_playlist_count"] = data.groupby("track_artist")["playlist_genre"].transform("nunique")

# Exposure Score + Log normalize
exposure_feats = ["artist_track_count", "artist_genre_diversity", "artist_career_length", "artist_playlist_count"]
scaler = MinMaxScaler()
data["artist_exposure_score"] = scaler.fit_transform(data[exposure_feats].fillna(0)).mean(axis=1)
data["artist_exposure_score_log"] = np.log1p(data["artist_exposure_score"] * 100)

# ------------------------------------------------
# 6️⃣ Artist recent activity
# ------------------------------------------------
data["artist_recent_activity"] = data.groupby("track_artist")["release_year"].transform(
    lambda x: np.mean(x >= 2020)
)

# ------------------------------------------------
# 7️⃣ PCA: Müzikal profil bileşenleri
# ------------------------------------------------
music_feats = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
pca = PCA(n_components=3, random_state=42)
music_pca = pca.fit_transform(data[music_feats].fillna(0))
for i in range(3):
    data[f"music_pca_{i+1}"] = music_pca[:, i]

# ------------------------------------------------
# 8️⃣ Etkileşim feature'ları
# ------------------------------------------------
data["energy_dance"] = data["energy"] * data["danceability"]
data["valence_energy"] = data["valence"] * data["energy"]
data["speech_live"] = data["speechiness"] * data["liveness"]
data["acoustic_energy"] = data["acousticness"] * data["energy"]

# ------------------------------------------------
# 9️⃣ Veri hazırlama
# ------------------------------------------------
drop_cols = ["track_popularity", "popularity_class"]
num_cols = data.select_dtypes(include="number").columns
X = data[num_cols.difference(drop_cols, sort=False)].fillna(0)
y = data["popularity_class"]

le = LabelEncoder()
y_enc = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.1, stratify=y_enc, random_state=42
)

# ------------------------------------------------
# 🔟 Model
# ------------------------------------------------
model = XGBClassifier(
    n_estimators=900,
    learning_rate=0.04,
    max_depth=9,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.1,
    random_state=42,
    eval_metric="mlogloss"
)

print("\n🚀 Training Final V12A XGBoost...")
model.fit(X_train, y_train)

# ------------------------------------------------
# 🏁 Tahmin & Değerlendirme
# ------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\n=== 🧭 V12A Final Results ===")
print(f"Accuracy: {acc:.3f} | Macro F1: {f1:.3f}\n")

# --- Classification Report ---
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='crest',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("🎯 Confusion Matrix - Spotify Popularity Classes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 💾 Model kaydet
# ------------------------------------------------
model.save_model("spotify_v12a_final.json")
print("\n✅ Model kaydedildi: spotify_v12a_final.json")





