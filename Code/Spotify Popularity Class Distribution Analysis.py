# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:09:13 2025

@author: Alican
"""

# ================================================================
# 🎵 Spotify Popularity Class Distribution Analysis
# ================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# 1️⃣ Veri yükleme
# ------------------------------------------------
data_path = r"C:\Users\Alican\Desktop\Projects\Spotify\data\spotify_songs.csv"
data = pd.read_csv(data_path)

# ------------------------------------------------
# 2️⃣ Sınıflandırma (popularity_class)
# ------------------------------------------------
data["popularity_class"] = pd.cut(
    data["track_popularity"],
    bins=[-1, 50, 75, 100],
    labels=["low", "medium", "high"]
).astype(str)

# ------------------------------------------------
# 3️⃣ Train-Test Split
# ------------------------------------------------
train_df, test_df = train_test_split(
    data,
    test_size=0.1,
    stratify=data["popularity_class"],
    random_state=42
)

# ------------------------------------------------
# 4️⃣ Dağılım Analizi
# ------------------------------------------------
def show_class_distribution(train_df, test_df, data):
    summary = pd.DataFrame({
        "Dataset": ["Train", "Test", "Total"],
        "Low (count)": [
            train_df["popularity_class"].value_counts().get("low", 0),
            test_df["popularity_class"].value_counts().get("low", 0),
            data["popularity_class"].value_counts().get("low", 0)
        ],
        "Medium (count)": [
            train_df["popularity_class"].value_counts().get("medium", 0),
            test_df["popularity_class"].value_counts().get("medium", 0),
            data["popularity_class"].value_counts().get("medium", 0)
        ],
        "High (count)": [
            train_df["popularity_class"].value_counts().get("high", 0),
            test_df["popularity_class"].value_counts().get("high", 0),
            data["popularity_class"].value_counts().get("high", 0)
        ]
    })

    total_counts = summary.loc[2, ["Low (count)", "Medium (count)", "High (count)"]]
    summary["Low (%)"] = (summary["Low (count)"] / total_counts["Low (count)"] * 100).round(2)
    summary["Medium (%)"] = (summary["Medium (count)"] / total_counts["Medium (count)"] * 100).round(2)
    summary["High (%)"] = (summary["High (count)"] / total_counts["High (count)"] * 100).round(2)

    print("\n📊 Class Distribution Summary:")
    print(summary.to_string(index=False))

    # Grafikler
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    datasets = [train_df, test_df, data]
    titles = ["Train Set", "Test Set", "Total Dataset"]

    for ax, df, title in zip(axes, datasets, titles):
        sns.countplot(x=df["popularity_class"], order=["low", "medium", "high"], palette="crest", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Popularity Class")
        ax.set_ylabel("Count" if title == "Train Set" else "")
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 100, f'{height:.0f}', ha="center")

    plt.suptitle("🎶 Popularity Class Distribution (Train–Test–Total)", fontsize=13)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 5️⃣ Fonksiyonu çalıştır
# ------------------------------------------------
show_class_distribution(train_df, test_df, data)


