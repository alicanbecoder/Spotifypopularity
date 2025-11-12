# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 16:35:26 2025

@author: Alican
"""

# ====================================================
# 🎧 Spotify Genre Popularity — V3 Stacking Ensemble
# Author: Alican Tunç
# ====================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")
data["playlist_genre_lower"] = data["playlist_genre"].str.lower()

# Create popularity class if missing
if "popularity_class" not in data.columns:
    data["popularity_class"] = pd.cut(
        data["track_popularity"],
        bins=[-1, 33, 66, 100],
        labels=["low", "medium", "high"]
    )

# Time-based features
def extract_year(date_str):
    try:
        return int(str(date_str)[:4])
    except:
        return np.nan

data["release_year"] = data["track_album_release_date"].apply(extract_year)
current_year = datetime.now().year
data["song_age"] = current_year - data["release_year"]

def get_decade(year):
    if pd.isna(year):
        return "Unknown"
    decade_start = int(year // 10 * 10)
    return f"{decade_start}s"

data["song_decade"] = data["release_year"].apply(get_decade)

target_map = {"low": 0, "medium": 1, "high": 2}
genres = ["pop", "rap", "rock", "r&b", "edm"]

# -----------------------------
# 2️⃣ Feature Engineering V3
# -----------------------------
def add_genre_features(df):
    df = df.copy()
    df["pop_emotion"] = df["valence"] * df["energy"] * df["danceability"]
    df["rap_performance"] = df["tempo"] * df["speechiness"] * (1 - df["acousticness"])
    df["rock_power"] = df["loudness"] * df["energy"] * (1 + df["song_age"] / 100)
    df["rnb_smooth"] = df["acousticness"] * (1 - df["tempo"] / 200)
    df["edm_drop_intensity"] = df["energy"] * df["tempo"] * (1 - df["acousticness"])
    return df

data = add_genre_features(data)

# -----------------------------
# 3️⃣ Train & Evaluate per Genre
# -----------------------------
results = []
reports = []

for g in genres:
    print(f"\n==============================")
    print(f"🎵 Genre: {g.upper()}")
    print("==============================")

    df_g = data[data["playlist_genre_lower"] == g].dropna(subset=["popularity_class"])
    if len(df_g) < 500:
        print(f"⚠️ {g} skipped (too few samples)")
        continue

    drop_cols = [
        "track_id", "track_name", "track_artist",
        "track_album_id", "track_album_name", "track_album_release_date",
        "playlist_name", "playlist_id", "playlist_genre", "playlist_subgenre",
        "playlist_genre_lower", "track_popularity", "song_decade"
    ]

    X = df_g.drop(columns=[c for c in drop_cols if c in df_g.columns])
    y = df_g["popularity_class"].map(target_map)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    X = X[numeric_cols].fillna(0)

    groups = df_g["track_album_id"] if "track_album_id" in df_g.columns else np.arange(len(df_g))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ⚙️ Scale + Balance
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"🧩 After SMOTE: {np.bincount(y_train_res)}")

    # -----------------------------
    # 4️⃣ Base Models
    # -----------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=42, class_weight="balanced_subsample"
    )

    cat = CatBoostClassifier(
        iterations=400, depth=8, learning_rate=0.05,
        loss_function="MultiClass", eval_metric="Accuracy",
        auto_class_weights="Balanced", verbose=False, random_seed=42
    )

    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        objective="multiclass", subsample=0.8, colsample_bytree=0.8, random_state=42
    )

    # -----------------------------
    # 5️⃣ Stacking Ensemble
    # -----------------------------
    meta_model = LGBMClassifier(
    n_estimators=200, learning_rate=0.05, num_leaves=16, max_depth=4,
    class_weight="balanced", random_state=42)

    stack_model = StackingClassifier(
        estimators=[("rf", rf), ("cat", cat), ("lgbm", lgbm)],
        final_estimator=meta_model,
        cv=3, n_jobs=-1, passthrough=True
    )

    stack_model.fit(X_train_res, y_train_res)
    y_pred = stack_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Stacking Accuracy ({g}): {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=["low", "medium", "high"]))

    results.append({"genre": g, "accuracy": acc, "samples": len(df_g)})
    reports.append(
        pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.assign(genre=g)
    )

# -----------------------------
# 6️⃣ Results Summary
# -----------------------------
results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
reports_df = pd.concat(reports, ignore_index=True)

results_df.to_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\results_v3_stacking.csv", index=False)
reports_df.to_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\classification_reports_v3.csv", index=False)

print("\n=== 🏁 Final Results (V3 Stacking Ensemble) ===")
print(results_df)

weighted_acc = np.average(results_df["accuracy"], weights=results_df["samples"])
print(f"\n🎯 Weighted Average Accuracy: {weighted_acc:.3f}")
