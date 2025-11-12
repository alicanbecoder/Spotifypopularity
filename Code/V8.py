# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 18:29:43 2025

@author: Alican
"""

# ================================================================
# 🎵 V8 — Genre Embedding Neural Network (Spotify Popularity Prediction)
# ================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1️⃣ Veri yükleme
# ------------------------------------------------
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

data["playlist_genre"] = data["playlist_genre"].str.lower()
data["duration_min"] = data["duration_ms"] / 60000.0

# Popularity class oluştur
if "popularity_class" not in data.columns:
    data["popularity_class"] = pd.cut(
        data["track_popularity"], bins=[-1, 30, 70, 100],
        labels=["low", "medium", "high"]
    ).astype(str)

# Yıl / yaş
def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except:
        return np.nan
data["release_year"] = data["track_album_release_date"].apply(extract_year)
data["song_age"] = 2020 - data["release_year"]

# ------------------------------------------------
# 2️⃣ Feature engineering
# ------------------------------------------------
data["mood_intensity"] = data["energy"] * data["valence"]
data["acoustic_purity"] = data["acousticness"] * (1 - data["instrumentalness"])
data["rhythm_drive"] = data["danceability"] * data["tempo"]

# ------------------------------------------------
# 3️⃣ Encoding
# ------------------------------------------------
genre_le = LabelEncoder()
data["genre_id"] = genre_le.fit_transform(data["playlist_genre"])

target_le = LabelEncoder()
y = target_le.fit_transform(data["popularity_class"])

numeric_feats = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
    "release_year", "song_age", "duration_min",
    "mood_intensity", "acoustic_purity", "rhythm_drive"
]
data = data.dropna(subset=numeric_feats + ["genre_id"])

X_num = data[numeric_feats].values
X_genre = data["genre_id"].values
scaler = RobustScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Train/Val/Test
X_train_num, X_test_num, X_train_genre, X_test_genre, y_train, y_test = train_test_split(
    X_num_scaled, X_genre, y, test_size=0.1, stratify=y, random_state=42
)
X_train_num, X_val_num, X_train_genre, X_val_genre, y_train, y_val = train_test_split(
    X_train_num, X_train_genre, y_train, test_size=0.1111, stratify=y_train, random_state=42
)

y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# ------------------------------------------------
# 4️⃣ Model — Genre Embedding + Numeric Inputs
# ------------------------------------------------
n_genres = data["genre_id"].nunique()
embed_dim = 6  # genre embedding boyutu

# Girişler
input_num = Input(shape=(X_train_num.shape[1],), name="numeric_input")
input_genre = Input(shape=(1,), name="genre_input")

# Embedding + Flatten
genre_emb = Embedding(input_dim=n_genres, output_dim=embed_dim, name="genre_embedding")(input_genre)
genre_emb_flat = Flatten()(genre_emb)

# Concatenate + MLP
merged = Concatenate()([input_num, genre_emb_flat])
x = Dense(512, activation="relu")(merged)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.25)(x)
output = Dense(3, activation="softmax")(x)

model = Model(inputs=[input_num, input_genre], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.0003), loss="categorical_crossentropy", metrics=["accuracy"])

es = EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True)
lr_sched = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=1)

# ------------------------------------------------
# 5️⃣ Eğitim
# ------------------------------------------------
history = model.fit(
    [X_train_num, X_train_genre], y_train_cat,
    validation_data=([X_val_num, X_val_genre], y_val_cat),
    epochs=250,
    batch_size=64,
    callbacks=[es, lr_sched],
    verbose=1
)

# ------------------------------------------------
# 6️⃣ Test değerlendirme
# ------------------------------------------------
test_loss, test_acc = model.evaluate([X_test_num, X_test_genre], y_test_cat, verbose=0)
print(f"\n🎯 V8 Genre Embedding Model Test Accuracy: {test_acc:.3f}")

# ------------------------------------------------
# 7️⃣ Grafik
# ------------------------------------------------
plt.figure(figsize=(10,6))
plt.plot(history.history["accuracy"], label="Train Acc", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Acc", linewidth=2)
plt.axhline(y=test_acc, color='r', linestyle='--', label=f"Test Acc = {test_acc:.3f}")
plt.title("🎧 V8 - Genre Embedding Neural Network Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


