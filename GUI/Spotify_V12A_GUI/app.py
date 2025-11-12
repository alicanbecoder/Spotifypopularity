# ================================================================
# 🎧 Spotify Popularity Predictor (V12A Showcase)
# ================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go

model = joblib.load("model_v12a.pkl")
scaler = joblib.load("scaler_v12a.pkl")
feature_list = json.load(open("feature_list.json"))

st.set_page_config(page_title="Spotify Popularity Predictor", page_icon="🎵", layout="centered")

st.title("🎧 Spotify Popularity Predictor (V12A)")
st.markdown("Bu uygulama, Spotify şarkı özelliklerine göre popülerlik sınıfını tahmin eder.")

st.sidebar.header("🔧 Özellikleri Gir")

danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5)
valence = st.sidebar.slider("Valence (Pozitiflik)", 0.0, 1.0, 0.5)
tempo = st.sidebar.slider("Tempo (BPM)", 50, 200, 120)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.3)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.3)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.1)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.2)
artist_track_count = st.sidebar.number_input("Artist Track Count", 1, 500, 20)
artist_genre_diversity = st.sidebar.number_input("Artist Genre Diversity", 1, 10, 3)
artist_career_length = st.sidebar.slider("Artist Career Length (years)", 0, 40, 10)
artist_playlist_count = st.sidebar.number_input("Artist Playlist Count", 1, 100, 5)
playlist_size = st.sidebar.number_input("Playlist Size", 1, 500, 50)
is_editorial = st.sidebar.selectbox("Spotify Editorial Playlist?", ["No", "Yes"])

input_data = pd.DataFrame([
    [danceability, energy, valence, tempo, speechiness, acousticness,
     instrumentalness, liveness, artist_track_count, artist_genre_diversity,
     artist_career_length, artist_playlist_count, playlist_size,
     1 if is_editorial == "Yes" else 0]
], columns=["danceability","energy","valence","tempo","speechiness","acousticness",
            "instrumentalness","liveness","artist_track_count","artist_genre_diversity",
            "artist_career_length","artist_playlist_count","playlist_size","is_editorial"])

X_scaled = scaler.transform(input_data.reindex(columns=feature_list, fill_value=0))
prediction = model.predict(X_scaled)[0]
pred_label = ["Low", "Medium", "High"][int(prediction)]

st.markdown("---")
st.subheader("🎯 Model Tahmini")
color_map = {"Low": "#FF6B6B", "Medium": "#FFD93D", "High": "#6BCB77"}
st.markdown(f"<div style='text-align:center; font-size:28px; background-color:{color_map[pred_label]}; color:white; padding:10px; border-radius:10px;'>🔊 Tahmin Edilen Popülerlik: <b>{pred_label}</b></div>", unsafe_allow_html=True)

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction * 50 + 25,
    title={"text": "Popularity Level", "font": {"size": 20}},
    gauge={"axis": {"range": [0, 150]},
           "bar": {"color": color_map[pred_label]},
           "steps": [{"range": [0, 50], "color": "#FFB6B6"},
                     {"range": [50, 100], "color": "#FFF3B0"},
                     {"range": [100, 150], "color": "#B6E2A1"}]}
))
st.plotly_chart(fig, use_container_width=True)
