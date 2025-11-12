# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:31:26 2025

@author: Alican
"""

# ================================================================
# 🎵 Spotify Popularity Prediction (V12A-FULL + SHAP Analysis)
# ================================================================
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import shap  # 🔍 SHAP eklendi

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

# Popularity sınıfları
data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[-1, 50, 75, 100],
    labels=["low", "medium", "high"]
).astype(str)

# ------------------------------------------------
# 3️⃣ Temporal (zaman bazlı) özellikler
# ------------------------------------------------
data["release_year"] = data["track_album_release_date"].astype(str).str[:4].astype(float)
data["is_pre_spotify_era"] = (data["release_year"] < 2010).astype(int)
data["is_2010s"] = ((data["release_year"] >= 2010) & (data["release_year"] < 2019)).astype(int)
data["is_recent_era"] = ((data["release_year"] >= 2019) & (data["release_year"] <= 2020)).astype(int)

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
# 5️⃣ Artist Intelligence (popularity-free)
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
# 6️⃣ PCA: Müzikal profil bileşenleri
# ------------------------------------------------
music_feats = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
pca = PCA(n_components=3, random_state=42)
music_pca = pca.fit_transform(data[music_feats].fillna(0))
for i in range(3):
    data[f"music_pca_{i+1}"] = music_pca[:, i]

# ------------------------------------------------
# 7️⃣ Etkileşim feature'ları
# ------------------------------------------------
data["energy_dance"] = data["energy"] * data["danceability"]
data["valence_energy"] = data["valence"] * data["energy"]
data["speech_live"] = data["speechiness"] * data["liveness"]
data["acoustic_energy"] = data["acousticness"] * data["energy"]

# ------------------------------------------------
# 8️⃣ Train/Test Split
# ------------------------------------------------
train_df, test_df = train_test_split(
    data, test_size=0.1, stratify=data["popularity_class"], random_state=42
)

# ------------------------------------------------
# 9️⃣ Veri hazırlama
# ------------------------------------------------
def prepare_data(train_df, test_df):
    drop_cols = ["track_popularity", "popularity_class"]
    num_cols = train_df.select_dtypes(include="number").columns
    X_train = train_df[num_cols.difference(drop_cols, sort=False)].fillna(0)
    X_test  = test_df[num_cols.difference(drop_cols, sort=False)].fillna(0)

    label_map = {"low": 0, "medium": 1, "high": 2}
    y_train = train_df["popularity_class"].map(label_map)
    y_test  = test_df["popularity_class"].map(label_map)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns

# ------------------------------------------------
# 🔟 Model ve SHAP analizi
# ------------------------------------------------
def run_xgboost(X_train, X_test, y_train, y_test, feature_names):
    model = XGBClassifier(
        n_estimators=900,
        max_depth=9,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.1,
        random_state=42,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["low", "medium", "high"]))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest",
                xticklabels=["low", "medium", "high"], yticklabels=["low", "medium", "high"])
    plt.title("Confusion Matrix - V12A Full")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # 🎯 SHAP Analysis
    print("\n🧠 Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance (summary)
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names, plot_type="bar")
    shap.summary_plot(shap_values, features=X_test, feature_names=feature_names)

    return acc, f1

# ------------------------------------------------
# 🏁 Çalıştır
# ------------------------------------------------
X_train, X_test, y_train, y_test, feature_names = prepare_data(train_df, test_df)
acc, f1 = run_xgboost(X_train, X_test, y_train, y_test, feature_names)
print(f"\n=== 🧭 V12-FULL + SHAP ===\nAccuracy: {acc:.3f} | Macro F1: {f1:.3f}")




