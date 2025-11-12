# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 16:10:15 2025

@author: Alican
"""


import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- 1️⃣ Veri yükleme ---
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")

# --- 2️⃣ Feature engineering ---
data["year"] = pd.to_datetime(data["track_album_release_date"], errors="coerce").dt.year
data["song_age"] = datetime.now().year - data["year"]
data["is_recent"] = (data["year"] >= (datetime.now().year - 5)).astype(int)
data["energy_danceability"] = data["energy"] * data["danceability"]
data["duration_min"] = (data["duration_ms"] / 60000).round(1)

# Onyıl
bins = [1920,1930,1940,1950,1960,1970,1980,1990,2000,2010,2020,2030]
labels = [f"{i}s" for i in range(1920,2030,10)][:len(bins) - 1]
data["Song Decade"] = pd.cut(data["year"], bins=bins, labels=labels, right=False)

# --- 3️⃣ Eksik verileri temizle ---
data = data.dropna(subset=["track_popularity", "year"])

# --- 4️⃣ Hedefi kategorileştir ---
data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[0, 40, 70, 100],
    labels=["low", "medium", "high"],
    include_lowest=True
)

# --- 5️⃣ Özellik ve hedef ---
features = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "loudness",
    "duration_min", "energy_danceability", "year", "song_age",
    "is_recent", "playlist_genre", "Song Decade"
]
target = "popularity_class"

X = data[features]
y = data[target]

# --- 6️⃣ Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# --- 7️⃣ Preprocessing ---
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = ["playlist_genre", "Song Decade"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# --- 8️⃣ Modeller ---
models = {
    "Logistic Regression": LogisticRegression(max_iter=300, multi_class="multinomial"),
    "KNN": KNeighborsClassifier(n_neighbors=10),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
    "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.1, random_state=42)
}

# --- 9️⃣ Eğitim ve test ---
results = []
for name, model in models.items():
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"\n{name} Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

# --- 🔟 Sonuç tablosu ---
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
print("\n📊 Model Accuracy Karşılaştırması:\n", results_df)
