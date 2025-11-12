# ================================================================
# 🎓 Train & Export V12A Model for Streamlit GUI
# ================================================================
import pandas as pd, numpy as np, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier
from sklearn.decomposition import PCA

# ------------------------------------------------
# 1️⃣ Veri yükleme
# ------------------------------------------------
data = pd.read_csv(r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv")

# ------------------------------------------------
# 2️⃣ Hazırlık ve Feature Engineering
# ------------------------------------------------
data["duration_min"] = data["duration_ms"] / 60000.0
data["playlist_genre"] = data["playlist_genre"].str.lower()
data["track_artist"] = data["track_artist"].astype(str).str.lower().str.strip()

# Popularity sınıfları
data["popularity_class"] = pd.cut(
    data["track_popularity"], bins=[-1, 50, 75, 100],
    labels=["low", "medium", "high"]
).astype(str)

# Zaman bazlı
data["release_year"] = data["track_album_release_date"].astype(str).str[:4].astype(float)
data["is_pre_spotify_era"] = (data["release_year"] < 2010).astype(int)
data["is_2010s"] = ((data["release_year"] >= 2010) & (data["release_year"] < 2019)).astype(int)
data["is_recent_era"] = ((data["release_year"] >= 2019) & (data["release_year"] <= 2020)).astype(int)

# Playlist bazlı
data["playlist_size"] = data.groupby("playlist_id")["track_id"].transform("count")
data["playlist_count"] = data.groupby("track_id")["playlist_id"].transform("nunique")
data["is_editorial"] = data["playlist_name"].str.contains("spotify|editorial", case=False, na=False).astype(int)

# Artist intelligence (popularity-free)
data["artist_track_count"] = data.groupby("track_artist")["track_id"].transform("count")
data["artist_genre_diversity"] = data.groupby("track_artist")["playlist_genre"].transform("nunique")
artist_stats = data.groupby("track_artist")["release_year"].agg(["min", "max"]).reset_index()
artist_stats.columns = ["track_artist", "artist_first_year", "artist_last_year"]
data = data.merge(artist_stats, on="track_artist", how="left")
data["artist_career_length"] = data["artist_last_year"] - data["artist_first_year"]
data["artist_playlist_count"] = data.groupby("track_artist")["playlist_id"].transform("nunique")

exposure_feats = ["artist_track_count", "artist_genre_diversity", "artist_career_length", "artist_playlist_count"]
scaler_exp = MinMaxScaler()
data["artist_exposure_score"] = scaler_exp.fit_transform(data[exposure_feats].fillna(0)).mean(axis=1)
data["artist_exposure_score_log"] = np.log1p(data["artist_exposure_score"] * 100)

# PCA (müzikal özellikler)
music_feats = ["danceability","energy","speechiness","acousticness","instrumentalness","liveness","valence","tempo"]
pca = PCA(n_components=3, random_state=42)
music_pca = pca.fit_transform(data[music_feats].fillna(0))
for i in range(3):
    data[f"music_pca_{i+1}"] = music_pca[:, i]

# Etkileşim feature'ları
data["energy_dance"] = data["energy"] * data["danceability"]
data["valence_energy"] = data["valence"] * data["energy"]

# ------------------------------------------------
# 3️⃣ Train/Test Split ve Model Eğitimi
# ------------------------------------------------
train_df, test_df = train_test_split(data, test_size=0.1, stratify=data["popularity_class"], random_state=42)
num_cols = train_df.select_dtypes(include="number").columns
drop_cols = ["track_popularity"]
X_train = train_df[num_cols.difference(drop_cols, sort=False)].fillna(0)
X_test  = test_df[num_cols.difference(drop_cols, sort=False)].fillna(0)
y_train = train_df["popularity_class"].map({"low":0, "medium":1, "high":2})
y_test  = test_df["popularity_class"].map({"low":0, "medium":1, "high":2})

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = XGBClassifier(
    n_estimators=900, max_depth=9, learning_rate=0.04,
    subsample=0.9, colsample_bytree=0.8,
    reg_lambda=1.1, random_state=42, eval_metric="mlogloss"
)
model.fit(X_train_scaled, y_train)

# ------------------------------------------------
# 4️⃣ Model Export
# ------------------------------------------------
joblib.dump(model, "model_v12a.pkl")
joblib.dump(scaler, "scaler_v12a.pkl")
json.dump(list(X_train.columns), open("feature_list.json", "w"))

print("✅ Model, scaler ve feature list başarıyla kaydedildi!")
