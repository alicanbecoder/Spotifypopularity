# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 15:05:26 2025

@author: Alican
"""

import pandas as pd
import os
from datetime import datetime

# === 1️⃣ Dosya dizinleri ve tür listesi ===
base_dir = r"C:\Users\Alican\Desktop\Projects\Spotify\data"

# Buradaki isimleri kendi CSV isimlerine göre ayarla
genre_file_map = {
    "pop":  "spotify_pop.csv",
    "rap":  "spotify_rap.csv",
    "rock": "spotify_rock.csv",
    "r&b":  "spotify_r&b.csv",      # eğer farklıysa değiştir
    "edm":  "spotify_edm.csv",
}

current_year = datetime.now().year

def safe_extract_year(x):
    """Yılı güvenli şekilde çek, parse edemezse NaN döndür."""
    s = str(x)
    if len(s) < 4:
        return None
    try:
        y = int(s[:4])
        # çok uçuk bir yıl gelirse filtrele (örn. 0000)
        if y < 1900 or y > current_year:
            return None
        return y
    except:
        return None

def get_decade(year):
    if pd.isna(year):
        return "Unknown"
    decade_start = int(year // 10 * 10)
    return f"{decade_start}s"

for genre, fname in genre_file_map.items():
    path_in = os.path.join(base_dir, fname)
    if not os.path.exists(path_in):
        print(f"⚠️ {genre.upper()} dosyası bulunamadı, atlanıyor -> {path_in}")
        continue

    print(f"\n==============================")
    print(f"🎵 Tür işleniyor: {genre.upper()}")
    print("==============================")

    df = pd.read_csv(path_in)

    # --- 1) Zaman feature'ları: release_year, song_age, song_decade ---
    if "track_album_release_date" not in df.columns:
        print(f"⚠️ 'track_album_release_date' sütunu yok, zaman feature'ları eklenemedi.")
    else:
        df["release_year"] = df["track_album_release_date"].apply(safe_extract_year)

        # NaN yıl’ları medyan/ortalamayla doldur
        if df["release_year"].isna().any():
            median_year = df["release_year"].dropna().median()
            df["release_year"].fillna(median_year, inplace=True)

        df["song_age"] = current_year - df["release_year"]
        df["song_decade"] = df["release_year"].apply(get_decade)

    # --- 2) Genre-özel feature'ları Sadece o tür için ekle ---
    # Ortak kullanılacak kolonları olduğundan emin ol
    for col in ["energy", "valence", "danceability", "loudness",
                "instrumentalness", "speechiness", "tempo",
                "acousticness", "liveness"]:
        if col not in df.columns:
            print(f"⚠️ {genre.upper()}: '{col}' sütunu yok, bazı feature'lar NaN olabilir.")

    if genre == "pop":
        df["pop_energy_valence"] = df["energy"] * df["valence"]
        df["pop_vocal_energy"]   = df["loudness"] * (1 - df["instrumentalness"])

    elif genre == "rap":
        df["rap_flow"]         = df["speechiness"] * df["tempo"]
        df["rap_dance_energy"] = df["danceability"] * df["energy"]

    elif genre == "rock":
        # rock_legacy: eski ve gürültülü rock
        if "song_age" in df.columns:
            df["rock_legacy"] = df["song_age"] * df["loudness"]
        else:
            df["rock_legacy"] = df["loudness"]
        df["rock_live_feel"] = df["energy"] * df["liveness"]

    elif genre == "r&b":
        # yumuşaklık: acoustic + (düşük loudness)
        loud_max = df["loudness"].abs().max() if "loudness" in df.columns else 1
        if loud_max == 0:
            loud_max = 1
        df["rnb_softness"] = df["acousticness"] * (1 - df["loudness"].abs() / loud_max)
        df["rnb_emotion"]  = df["valence"] * df["danceability"]

    elif genre == "edm":
        df["edm_bpm_feel"] = df["tempo"] * df["energy"]
        df["edm_vibe"]     = df["valence"] * df["danceability"]

    # --- 3) Yeni isimle kaydet ---
    out_name = fname.replace(".csv", "_features.csv")
    path_out = os.path.join(base_dir, out_name)
    df.to_csv(path_out, index=False, encoding="utf-8-sig")

    print(f"✅ {genre.upper()}: Feature'lar eklendi ve kaydedildi -> {path_out}")
    print("   Eklenen örnek kolonlar:", [c for c in df.columns if genre.split("&")[0] in c][:5])

print("\n🎯 Tüm türler için GENRE-ÖZEL feature'lı CSV'ler oluşturuldu.")



