# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:32:16 2025

@author: Alican
"""

# ==============================================
# 🎵 Genre-Specific Feature Engineering Module
# ==============================================

import pandas as pd
import numpy as np

def create_genre_features(df: pd.DataFrame, genre: str) -> pd.DataFrame:
    """
    Her müzik türü için özel feature mühendisliği uygular.
    Genre bazında popülerlik belirleyen dinamikleri ortaya çıkarır.

    Parametreler:
        df (pd.DataFrame): Orijinal tür datası
        genre (str): Tür adı (örnek: "pop", "rap", "rock", "r&b", "edm")

    Dönüş:
        pd.DataFrame: Yeni feature'lar eklenmiş dataframe
    """
    df = df.copy()

    # --- Ortak güvenlik kontrolü ---
    for col in ["energy", "valence", "danceability", "tempo", "loudness", 
                "speechiness", "acousticness", "instrumentalness"]:
        if col not in df.columns:
            df[col] = 0.0

    # --- Tür bazlı özel feature'lar ---
    if genre == "pop":
        df["pop_energy_valence"] = df["energy"] * df["valence"]
        df["dance_energy_combo"] = df["danceability"] * df["energy"]
        df["happy_tempo"] = df["valence"] * df["tempo"]
        df["smoothness"] = (1 - df["acousticness"]) * df["danceability"]

    elif genre == "rap":
        df["speech_energy"] = df["speechiness"] * df["energy"]
        df["rhythm_power"] = df["loudness"] * df["tempo"]
        df["word_beat_sync"] = df["speechiness"] * df["tempo"]
        df["aggressiveness"] = df["energy"] * (1 - df["acousticness"])

    elif genre == "rock":
        df["guitar_intensity"] = df["instrumentalness"] * df["energy"]
        df["rock_drive"] = df["loudness"] * df["tempo"]
        df["electric_balance"] = (1 - df["acousticness"]) * df["energy"]
        df["headbang_factor"] = df["energy"] * df["valence"] * df["tempo"]

    elif genre == "r&b":
        df["smooth_factor"] = df["acousticness"] * (1 - df["valence"])
        df["groove"] = df["danceability"] * df["energy"]
        df["soul_vibe"] = df["valence"] * (1 - df["energy"])
        df["chillness"] = df["acousticness"] * df["danceability"]

    elif genre == "edm":
        df["beat_intensity"] = df["energy"] * df["tempo"]
        df["drop_power"] = df["loudness"] * df["energy"]
        df["festival_factor"] = df["energy"] * df["valence"] * df["danceability"]
        df["electro_drive"] = (1 - df["acousticness"]) * df["energy"]

    else:
        # Diğer türler için genel feature set
        df["energy_valence_combo"] = df["energy"] * df["valence"]
        df["dance_energy_mix"] = df["danceability"] * df["energy"]

    return df
