# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:30:28 2025

@author: Alican
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# --- 1️⃣ Veri yükleme ---
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")

# --- 2️⃣ Feature engineering ---
data["year"] = pd.to_datetime(data["track_album_release_date"], errors="coerce").dt.year
data["song_age"] = datetime.now().year - data["year"]
data["is_recent"] = (data["year"] >= (datetime.now().year - 5)).astype(int)
data["energy_danceability"] = data["energy"] * data["danceability"]
data["duration_min"] = (data["duration_ms"] / 60000).round(1)

bins = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030]
labels = [f"{i}s" for i in range(1920,2030,10)][:len(bins) - 1]
data["Song Decade"] = pd.cut(data["year"], bins=bins, labels=labels, right=False)

data = data.dropna(subset=["track_popularity", "year"])

# --- 3️⃣ Popularity sınıfları ---
data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[0, 40, 70, 100],
    labels=["low", "medium", "high"],
    include_lowest=True
)

# --- 4️⃣ Özellik ve hedef ---
features = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness",
    "duration_min", "energy_danceability", "year", "song_age",
    "is_recent", "playlist_genre", "Song Decade"
]
target = "popularity_class"

X = data[features]
y = data[target]

# --- 5️⃣ Label encoding ---
le = LabelEncoder()
y_enc = le.fit_transform(y)

# --- 6️⃣ Train/Val/Test split (0.8 / 0.1 / 0.1) ---
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_enc, test_size=0.1, stratify=y_enc, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1111, stratify=y_train_full, random_state=42
)  # 0.1111 * 0.9 ≈ 0.1

# --- 7️⃣ Encoding ---
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = ["playlist_genre", "Song Decade"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

X_train_enc = preprocessor.fit_transform(X_train)
X_val_enc = preprocessor.transform(X_val)
X_test_enc = preprocessor.transform(X_test)

# --- 8️⃣ SMOTE (train verisine uygulanır) ---
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_enc, y_train)

print("🎯 SMOTE sonrası sınıf dağılımı:\n", pd.Series(y_train_res).value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

max_depth_values = [3, 5, 8, 10, 15, 20, 30, 40]
min_leaf_values = [1, 2, 5, 10]

results = []

for depth in max_depth_values:
    for leaf in min_leaf_values:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)

        train_acc = accuracy_score(y_train, model.predict(X_train_enc))
        val_acc = accuracy_score(y_val, model.predict(X_val_enc))
        test_acc = accuracy_score(y_test, model.predict(X_test_enc))

        results.append((depth, leaf, train_acc, val_acc, test_acc))

# Sonuçları DataFrame'e al
results_df = pd.DataFrame(results, columns=["max_depth", "min_samples_leaf", "train_acc", "val_acc", "test_acc"])
print(results_df.sort_values("val_acc", ascending=False).head(10))

# Grafik (max_depth'e göre)
plt.figure(figsize=(8,5))
for leaf in min_leaf_values:
    subset = results_df[results_df["min_samples_leaf"] == leaf]
    plt.plot(subset["max_depth"], subset["val_acc"], marker="o", label=f"min_leaf={leaf}")
plt.title("🎯 Validation Accuracy vs max_depth")
plt.xlabel("max_depth")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


