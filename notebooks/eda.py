import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from pathlib import Path

# Paths 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST     = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
PLOTS_DIR    = PROJECT_ROOT / "notebooks" / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(MANIFEST)

# AQI Distribution 
plt.figure(figsize=(10, 4))
sns.histplot(df["aqi"], bins=30, kde=True, color="steelblue")
plt.title("AQI Distribution Across Dataset")
plt.xlabel("AQI Value")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "aqi_distribution.png", dpi=150)
plt.show()
print("Saved: aqi_distribution.png")

#  Category Count 
plt.figure(figsize=(10, 4))
order = df["category"].value_counts().index
sns.countplot(data=df, y="category", order=order, palette="RdYlGn_r")
plt.title("Images per AQI Category")
plt.xlabel("Count")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_distribution.png", dpi=150)
plt.show()
print("Saved: category_distribution.png")

#  Sample Images per Category 
categories = df["category"].unique()
fig, axes = plt.subplots(len(categories), 3, figsize=(12, 4 * len(categories)))

for row_idx, cat in enumerate(sorted(categories)):
    samples = df[df["category"] == cat].sample(
        min(3, len(df[df["category"] == cat])), random_state=42
    )
    for col_idx, (_, sample) in enumerate(samples.iterrows()):
        ax = axes[row_idx][col_idx]
        try:
            img = mpimg.imread(sample["filepath"])
            ax.imshow(img)
            ax.set_title(f"AQI: {sample['aqi']:.0f}", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, "Load Error", ha="center")
        ax.axis("off")
    axes[row_idx][0].set_ylabel(cat, fontsize=9, rotation=0,
                                labelpad=120, va="center")

plt.suptitle("Sample Images per AQI Category", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "sample_images.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: sample_images.png")

# Image Size Distribution ────
import cv2

print("\nChecking image dimensions (sample of 100)...")
widths, heights = [], []
sample_paths = df["filepath"].sample(min(100, len(df)), random_state=42)

for path in sample_paths:
    img = cv2.imread(path)
    if img is not None:
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)

plt.figure(figsize=(10, 4))
plt.scatter(widths, heights, alpha=0.5, color="coral")
plt.axvline(224, color="blue", linestyle="--", label="Target width (224)")
plt.axhline(224, color="green", linestyle="--", label="Target height (224)")
plt.title("Image Dimensions in Dataset")
plt.xlabel("Width (px)")
plt.ylabel("Height (px)")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "image_dimensions.png", dpi=150)
plt.show()
print("Saved: image_dimensions.png")

print(f"\nWidth  — min: {min(widths)}, max: {max(widths)}, mean: {np.mean(widths):.0f}")
print(f"Height — min: {min(heights)}, max: {max(heights)}, mean: {np.mean(heights):.0f}")
print("\nEDA complete. Check notebooks/plots/ for saved figures.")