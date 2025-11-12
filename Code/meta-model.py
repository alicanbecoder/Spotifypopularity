# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 14:35:17 2025

@author: Alican
"""
# ==============================================
# 🎧 Spotify Smart Stacking v2 (Leak-Free + Tuned)
# ==============================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# -----------------------------
# 1️⃣ Veri yükle + yeni feature'lar
# -----------------------------
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")

# --- Genre tabanlı haritalar
genre_cluster_map = {
    "edm": "energy_cluster", "rock": "energy_cluster",
    "pop": "mid_cluster",
    "rap": "rhythm_cluster", "r&b": "rhythm_cluster", "latin": "rhythm_cluster"
}
genre_distance_map = {"edm": 0.16, "rock": 0.15, "rap": 0.10,
                      "latin": 0.09, "r&b": 0.08, "pop": 0.05}

data["genre_cluster"]  = data["playlist_genre"].map(genre_cluster_map)
data["genre_distance"] = data["playlist_genre"].map(genre_distance_map)

# --- Popularity sınıfı
data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[-1, 33, 66, 100],
    labels=["low", "medium", "high"]
)

# --- Ek feature engineering
data["album_year"] = pd.to_datetime(data["track_album_release_date"], errors="coerce").dt.year
data["album_age"]  = 2025 - data["album_year"]
data["duration_min"] = data["duration_ms"] / 60000
data["movement_score"] = data["energy"] * data["danceability"]
data["emotion_score"]  = data["acousticness"] * data["valence"]
data["genre_energy_combo"] = data["energy"] * data["genre_distance"]
data["genre_dance_combo"]  = data["danceability"] * data["genre_distance"]

# Subgenre trend feature (proxy popularity)
trend_map = data.groupby("playlist_subgenre")["track_popularity"].mean()
data["subgenre_trend"] = data["playlist_subgenre"].map(trend_map)

# -----------------------------
# 2️⃣ Album bazlı split (leak-free)
# -----------------------------
album_ids = data["track_album_id"].unique()
train_albums, test_albums = train_test_split(album_ids, test_size=0.25, random_state=42)

mask_train = data["track_album_id"].isin(train_albums)
mask_test  = data["track_album_id"].isin(test_albums)
train_df = data.loc[mask_train].copy()
test_df  = data.loc[mask_test].copy()

# -----------------------------
# 3️⃣ Gereksiz / leak oluşturan kolonlar
# -----------------------------
drop_cols = [
    "track_id","track_name","track_artist","track_album_id","track_album_name",
    "track_album_release_date","playlist_name","playlist_id","track_popularity"
]
train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

# -----------------------------
# 4️⃣ Base layer (CatBoost genre modelleri, 5-fold OOF)
# -----------------------------
genres = data["playlist_genre"].dropna().unique()

meta_train_blocks, meta_test_blocks = [], []
base_acc = {}

for g in genres:
    train_g = train_df[train_df["playlist_genre"] == g]
    test_g  = test_df[test_df["playlist_genre"] == g]
    if len(train_g) < 400 or len(test_g) < 150:
        print(f"⚠️ {g} türü atlandı (train={len(train_g)}, test={len(test_g)})")
        continue

    print(f"\n🎧 Genre: {g} (train={len(train_g)}, test={len(test_g)})")

    X_train_full = train_g.drop(columns=["popularity_class","playlist_genre","playlist_subgenre"], errors="ignore")
    y_train_full = train_g["popularity_class"]
    X_test_full  = test_g.drop(columns=["popularity_class","playlist_genre","playlist_subgenre"], errors="ignore")
    y_test_full  = test_g["popularity_class"]

    all_g = pd.concat([X_train_full, X_test_full], axis=0)
    all_g_dum = pd.get_dummies(all_g, drop_first=True, dtype=np.float32)
    X_train_g = all_g_dum.iloc[:len(X_train_full)]
    X_test_g  = all_g_dum.iloc[len(X_train_full):]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X_train_g), 3), dtype=np.float32)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_g, y_train_full)):
        X_tr, X_val = X_train_g.iloc[tr_idx], X_train_g.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]

        cb = CatBoostClassifier(
            iterations=300,
            depth=9,
            learning_rate=0.05,
            l2_leaf_reg=3,
            loss_function="MultiClass",
            eval_metric="Accuracy",
            random_state=42,
            verbose=False
        )
        cb.fit(X_tr, y_tr)
        oof_proba[val_idx, :] = cb.predict_proba(X_val)
        print(f"   Fold {fold+1}/5 tamam.")

    proba_cols = [f"{g}_proba_{cls}" for cls in cb.classes_]
    meta_train_blocks.append(pd.DataFrame(oof_proba, index=train_g.index, columns=proba_cols))

    final_cb = CatBoostClassifier(
        iterations=400,
        depth=10,
        learning_rate=0.04,
        l2_leaf_reg=2,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        random_state=42,
        verbose=False
    )
    final_cb.fit(X_train_g, y_train_full)
    test_proba = final_cb.predict_proba(X_test_g)
    meta_test_blocks.append(pd.DataFrame(test_proba, index=test_g.index, columns=proba_cols))

    acc_g = accuracy_score(y_test_full, final_cb.predict(X_test_g))
    base_acc[g] = acc_g
    print(f"   🎯 Base {g} accuracy: {acc_g:.3f}")

# -----------------------------
# 5️⃣ Meta feature matrisi
# -----------------------------
meta_train_df = pd.concat(meta_train_blocks, axis=0).sort_index().fillna(0)
meta_test_df  = pd.concat(meta_test_blocks, axis=0).sort_index().fillna(0)
y_train_meta = train_df.loc[meta_train_df.index, "popularity_class"]
y_test_meta  = test_df.loc[meta_test_df.index, "popularity_class"]

# Ek audio & combo feature'ları meta'ya ekle
extra_feats = [
    "energy","valence","tempo","danceability","loudness","genre_distance",
    "album_age","movement_score","emotion_score","genre_energy_combo",
    "genre_dance_combo","subgenre_trend"
]
for f in extra_feats:
    if f in train_df.columns:
        meta_train_df[f] = train_df.loc[meta_train_df.index, f].astype(np.float32)
        meta_test_df[f]  = test_df.loc[meta_test_df.index,  f].astype(np.float32)

print(f"\n✅ Meta feature matrisi hazır: train={meta_train_df.shape}, test={meta_test_df.shape}")

# -----------------------------
# 6️⃣ Meta model (LightGBM tuned)
# -----------------------------
meta_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=0.8,
    reg_alpha=0.4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
meta_model.fit(meta_train_df, y_train_meta)
meta_pred = meta_model.predict(meta_test_df)

acc_meta = accuracy_score(y_test_meta, meta_pred)
print(f"\n🏁 Meta-Classifier (v2 Tuned, leak-free) Accuracy: {acc_meta:.3f}")
print("\n", classification_report(y_test_meta, meta_pred, zero_division=0))

