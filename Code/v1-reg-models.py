# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:57:19 2025

@author: Alican
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1️⃣ Veri yükleme ---
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")

# --- 2️⃣ Feature engineering ---
data["year"] = pd.to_datetime(data["track_album_release_date"], errors="coerce").dt.year
data["song_age"] = datetime.now().year - data["year"]
data["is_recent"] = (data["year"] >= (datetime.now().year - 5)).astype(int)
data["energy_danceability"] = data["energy"] * data["danceability"]
data["duration_min"] = (data["duration_ms"] / 60000).round(1)

# Onyıl bilgisi
bins = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030]
labels = [f"{i}s" for i in range(1920,2030,10)][:len(bins) - 1]
data["Song Decade"] = pd.cut(data["year"], bins=bins, labels=labels, right=False)

# Eksik verileri temizle
data = data.dropna(subset=["track_popularity", "year", "playlist_genre"])

# --- 3️⃣ Yeni feature'lar (leakage olmayanlar) ---
data["mood_mix"] = data["energy"] * data["valence"] * data["danceability"]
data["log_age"] = np.log1p(data["song_age"])
data["energy_acoustic_contrast"] = data["energy"] - data["acousticness"]
data["valence_energy_ratio"] = data["valence"] / (data["energy"] + 0.01)

# --- 4️⃣ Özellikler ve hedef ---
features = [
    "danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "loudness", "duration_min", "energy_danceability",
    "song_age", "is_recent", "mood_mix", "log_age",
    "energy_acoustic_contrast", "valence_energy_ratio",
    "playlist_genre", "Song Decade"
]
target = "track_popularity"

X = data[features]
y = data[target]

# --- 5️⃣ Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6️⃣ Preprocessing ---
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = ["playlist_genre", "Song Decade"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# --- 7️⃣ Random Forest Pipeline ---
model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

# --- 8️⃣ Eğitim ---
model.fit(X_train, y_train)

# --- 9️⃣ Tahmin ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# --- 🔟 Performans ---
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"🏋️‍♂️ Train R²: {r2_train:.3f} | MAE: {mae_train:.2f}")
print(f"🧩 Test  R²: {r2_test:.3f} | MAE: {mae_test:.2f}")

# --- 1️⃣1️⃣ Feature Importance ---
rf = model.named_steps["rf"]
pre = model.named_steps["preprocess"]

feature_names = pre.get_feature_names_out()
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n🔝 En Önemli 15 Özellik:\n")
print(importance_df.head(15))

# --- 🎨 Görselleştirme ---
plt.figure(figsize=(10,6))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df.head(15),
    palette="viridis"
)
plt.title("🎧 Random Forest Feature Importance (Leakage'siz Model)")
plt.xlabel("Önem Skoru")
plt.tight_layout()
plt.show()
