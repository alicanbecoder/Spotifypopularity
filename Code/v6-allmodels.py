# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 19:08:20 2025

@author: Alican
"""

# ================================================================
# 🎵 Spotify Genre-Aware Popularity Prediction (V6.2-TRACKCOUNT-ONLY)
# ================================================================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# 0️⃣ Veri yükleme
# ------------------------------------------------
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# Süreyi dakika cinsine çevir
data["duration_min"] = data["duration_ms"] / 60000.0

# Popularity class (low, medium, high)
if "popularity_class" not in data.columns:
    data["popularity_class"] = pd.cut(
        data["track_popularity"],
        bins=[-1, 50, 75, 100],
        labels=["low", "medium", "high"]
    ).astype(str)

data["playlist_genre"] = data["playlist_genre"].str.lower()
data["track_artist"] = data["track_artist"].astype(str).str.lower().str.strip()

# ------------------------------------------------
# 1️⃣ release_year & song_age (2020 tabanlı)
# ------------------------------------------------
def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except Exception:
        return np.nan

data["release_year"] = data["track_album_release_date"].apply(extract_year)
data["song_age"] = 2020 - data["release_year"]

# ------------------------------------------------
# 2️⃣ Train/Test Split
# ------------------------------------------------
target = "popularity_class"
train_idx, test_idx, y_train_raw, y_test_raw = train_test_split(
    data.index,
    data[target],
    test_size=0.1,
    stratify=data[target],
    random_state=42
)

train_df = data.loc[train_idx].reset_index(drop=True)
test_df  = data.loc[test_idx].reset_index(drop=True)

# ------------------------------------------------
# 3️⃣ Artist-based Feature (only track_count, no leakage)
# ------------------------------------------------
artist_stats = (
    train_df
    .groupby("track_artist")["track_popularity"]
    .agg(artist_track_count="size")
    .reset_index()
)

train_df = train_df.merge(artist_stats, on="track_artist", how="left")
test_df  = test_df.merge(artist_stats, on="track_artist", how="left")

for col in ["artist_track_count"]:
    train_df[col] = train_df[col].fillna(0)
    test_df[col]  = test_df[col].fillna(0)

# ------------------------------------------------
# 4️⃣ Ek Feature Engineering
# ------------------------------------------------
for df in [train_df, test_df]:
    df["energy_dance"]    = df["energy"] * df["danceability"]
    df["acoustic_energy"] = df["acousticness"] * df["energy"]
    df["speech_live"]     = df["speechiness"] * df["liveness"]
    df["valence_energy"]  = df["valence"] * df["energy"]
    df["tempo_norm"]      = (df["tempo"] - df["tempo"].mean()) / (df["tempo"].std() + 1e-6)
    df["is_recent"]       = (df["release_year"] >= 2015).astype(int)

# ------------------------------------------------
# 5️⃣ Genre Distance + Mapping
# ------------------------------------------------
numeric_feats = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "release_year", "song_age", "duration_min",
    "energy_dance", "acoustic_energy", "speech_live",
    "valence_energy", "tempo_norm",
    "artist_track_count"   # 👈 sadece bu eklendi
]

genre_vectors = train_df.groupby("playlist_genre")[numeric_feats].mean()

scaler = StandardScaler()
genre_vectors_scaled = pd.DataFrame(
    scaler.fit_transform(genre_vectors),
    index=genre_vectors.index,
    columns=genre_vectors.columns
)

genre_distance = pd.DataFrame(
    cosine_distances(genre_vectors_scaled),
    index=genre_vectors_scaled.index,
    columns=genre_vectors_scaled.index
)

def add_genre_distance_features(df):
    df = df.copy()
    for g in genre_distance.columns:
        df[f"dist_to_{g}"] = df["playlist_genre"].apply(
            lambda x: genre_distance.loc[x, g] if x in genre_distance.index else np.nan
        )
    return df

train_df = add_genre_distance_features(train_df)
test_df  = add_genre_distance_features(test_df)

# PCA embedding (2D genre map)
pca = PCA(n_components=2, random_state=42)
genre_map = pd.DataFrame(
    pca.fit_transform(genre_vectors_scaled),
    index=genre_vectors_scaled.index,
    columns=["genre_map_x", "genre_map_y"]
)

def add_genre_mapping(df):
    df = df.copy()
    df["genre_map_x"] = df["playlist_genre"].map(genre_map["genre_map_x"])
    df["genre_map_y"] = df["playlist_genre"].map(genre_map["genre_map_y"])
    return df

train_df = add_genre_mapping(train_df)
test_df  = add_genre_mapping(test_df)

# ------------------------------------------------
# 6️⃣ Feature matrix
# ------------------------------------------------
drop_cols = ["track_popularity", "popularity_class"]
num_cols = train_df.select_dtypes(include="number").columns

X_train = train_df[num_cols.difference(drop_cols, sort=False)].fillna(0)
X_test  = test_df[num_cols.difference(drop_cols, sort=False)].fillna(0)
y_train = train_df[target]
y_test  = test_df[target]

# ------------------------------------------------
# 7️⃣ Modeller
# ------------------------------------------------
models = {
    "CatBoost": CatBoostClassifier(
        iterations=700,
        depth=8,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
        auto_class_weights="Balanced"
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=42,
        class_weight="balanced"
    )
}

# ------------------------------------------------
# 8️⃣ Benchmark (TrackCount Only)
# ------------------------------------------------
results = []
trained_models = {}

for name, model in models.items():
    print(f"\n🚀 Training {name} (ARTIST_TRACK_COUNT only)...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"{name} Accuracy: {acc:.3f} | Macro F1: {f1:.3f}")
    results.append({"Model": name, "Accuracy": acc, "MacroF1": f1})
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n=== 🏁 Multi-Model Benchmark Results (TrackCount Only) ===")
print(results_df)

# ------------------------------------------------
# 9️⃣ Feature Importance (LightGBM)
# ------------------------------------------------
lgb_model = trained_models["LightGBM"]
imp = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(8, 7))
sns.barplot(y="feature", x="importance", data=imp.head(25))
plt.title("LightGBM Feature Importances (Top 25, TrackCount Only)")
plt.tight_layout()
plt.show()

print("\n🔍 Top 20 Important Features:")
print(imp.head(20))





