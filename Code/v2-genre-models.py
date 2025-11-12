# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:54:54 2025

@author: Alican
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- 1️⃣ Veri yükleme ---
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")

# --- 2️⃣ Feature engineering ---
data["year"] = pd.to_datetime(data["track_album_release_date"], errors="coerce").dt.year
data["song_age"] = datetime.now().year - data["year"]
data["duration_min"] = (data["duration_ms"] / 60000).round(1)
data["mood_mix"] = data["energy"] * data["valence"] * data["danceability"]
data["speech_music_ratio"] = data["speechiness"] / (data["instrumentalness"] + 0.001)
data["dance_tempo_product"] = data["danceability"] * data["tempo"]
data["energy_acoustic_contrast"] = data["energy"] - data["acousticness"]
data["log_song_age"] = np.log1p(data["song_age"])

# --- 3️⃣ Popularity sınıfları ---
data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[0, 40, 70, 100],
    labels=["low", "medium", "high"],
    include_lowest=True
)

# --- 4️⃣ Özellik seti ---
features = [
    "acousticness", "danceability", "energy", "loudness",
    "speechiness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_min", "song_age", "mood_mix",
    "speech_music_ratio", "dance_tempo_product", "energy_acoustic_contrast", "log_song_age"
]

target = "popularity_class"

# --- 5️⃣ Sonuç listesi ---
results_scaled = []

# --- 6️⃣ Her genre için ayrı model ---
genres = data["playlist_genre"].dropna().unique()

for genre in genres:
    subset = data[data["playlist_genre"] == genre].dropna(subset=features + [target])
    
    if subset[target].nunique() < 3:
        continue
    
    X = subset[features]
    y = subset[target]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    # StandardScaler (her tür için ayrı uygulanıyor)
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), features)
    ])
    
    model = Pipeline([
        ("pre", preprocessor),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    f1_macro = report["macro avg"]["f1-score"]
    
    results_scaled.append({"genre": genre, "accuracy": acc, "f1_macro": f1_macro, "n_samples": len(subset)})

# --- 7️⃣ Sonuçları göster ---
results_scaled_df = pd.DataFrame(results_scaled).sort_values(by="accuracy", ascending=False)
print(results_scaled_df)

# --- 8️⃣ Görselleştirme ---
plt.figure(figsize=(8,5))
plt.bar(results_scaled_df["genre"], results_scaled_df["accuracy"], color="slateblue")
plt.title("🎧 Genre Bazlı Model Doğruluk Oranları (StandardScaler Uygulandı)")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


