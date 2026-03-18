import os
import pandas as pd
from pathlib import Path
import logging

#Logging 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

#Paths 
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
RAW_DIR        = PROJECT_ROOT / "data" / "raw"
PROC_DIR       = PROJECT_ROOT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

PM25_BREAKPOINTS = [
    (0.0,   12.0,   0,   50,  "Good"),
    (12.1,  35.4,  51,  100, "Moderate"),
    (35.5,  55.4, 101,  150, "Unhealthy for Sensitive Groups"),
    (55.5, 150.4, 151,  200, "Unhealthy"),
    (150.5, 250.4, 201, 300, "Very Unhealthy"),
    (250.5, 500.4, 301, 500, "Hazardous"),
]
# Based on visibility research and EPA AQI standards
DAWN_AQI_MAP = {
    "mist":           (75,  20),
    "foggy":         (125,  25),
    "haze":          (175,  30),
    "rain_storm":     (80,  20),
    "snow_storm":     (85,  20),
    "sand_storm_g2_": (225, 35),
    "sand_storm_g2":  (260, 35),
    "sand_storm":     (300, 40),
    "dusttornado":    (375, 50),
}

DENSE_HAZE_AQI = (280, 40)

def aqi_to_category(aqi: float) -> str:
    for _, _, i_lo, i_hi, cat in PM25_BREAKPOINTS:
        if i_lo <= aqi <= i_hi:
            return cat
    return "Unknown"


def build_dense_haze_manifest() -> pd.DataFrame:
    """Assigns AQI 280 to all dense haze images."""
    hazy_dir = RAW_DIR / "dense_haze" / "hazy"
    records = []

    for img_file in sorted(hazy_dir.glob("*.png")):
        records.append({
            "filename": img_file.name,
            "filepath": str(img_file),
            "aqi":      float(DENSE_HAZE_AQI),
            "category": aqi_to_category(DENSE_HAZE_AQI),
            "source":   "dense_haze"
        })

    log.info(f"Dense Haze: {len(records)} images")
    return pd.DataFrame(records)


def build_dawn_manifest() -> pd.DataFrame:
    images_dir = RAW_DIR / "dawn" / "images"
    records = []
    skipped = 0
    rng = np.random.default_rng(42)

    for img_file in sorted(images_dir.glob("*.[jJpP][pPnN][gG]*")):
        stem     = img_file.stem
        category = stem.rsplit("-", 1)[0]

        aqi_params = None
        for key in sorted(DAWN_AQI_MAP.keys(), key=len, reverse=True):
            if category == key:
                aqi_params = DAWN_AQI_MAP[key]
                break

        if aqi_params is None:
            skipped += 1
            continue

        mean, std = aqi_params
        # Sample AQI from gaussian, clamp to valid EPA range
        aqi_val = float(np.clip(rng.normal(mean, std), 0, 500))

        records.append({
            "filename": img_file.name,
            "filepath": str(img_file),
            "aqi":      round(aqi_val, 1),
            "category": aqi_to_category(aqi_val),
            "source":   "dawn"
        })

    log.info(f"DAWN: {len(records)} images mapped, {skipped} skipped")
    return pd.DataFrame(records)

def build_dense_haze_manifest() -> pd.DataFrame:
    hazy_dir = RAW_DIR / "dense_haze" / "hazy"
    records  = []
    rng      = np.random.default_rng(99)
    mean, std = DENSE_HAZE_AQI

    for img_file in sorted(hazy_dir.glob("*.png")):
        aqi_val = float(np.clip(rng.normal(mean, std), 0, 500))
        records.append({
            "filename": img_file.name,
            "filepath": str(img_file),
            "aqi":      round(aqi_val, 1),
            "category": aqi_to_category(aqi_val),
            "source":   "dense_haze"
        })

    log.info(f"Dense Haze: {len(records)} images")
    return pd.DataFrame(records)


def print_stats(df: pd.DataFrame):
    print("\n" + "═" * 55)
    print(f"  Total images     : {len(df)}")
    print(f"  AQI range        : {df['aqi'].min():.0f} – {df['aqi'].max():.0f}")
    print(f"  AQI mean ± std   : {df['aqi'].mean():.1f} ± {df['aqi'].std():.1f}")
    print(f"\n  {'Source':<20} {'Count'}")
    print(f"  {'-'*35}")
    for src, count in df["source"].value_counts().items():
        print(f"  {src:<20} {count}")
    print(f"\n  {'AQI Category':<42} {'Count'}")
    print(f"  {'-'*50}")
    for cat, count in df["category"].value_counts().items():
        bar = "█" * (count // 10)
        print(f"  {cat:<42} {count:>4}  {bar}")
    print("═" * 55 + "\n")
# ── Preprocessing Pipeline ─────────────────────────────────────────────────────
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE    = 224
BATCH_SIZE  = 32
AQI_MIN     = 0.0
AQI_MAX     = 500.0


def normalize_aqi(aqi: float) -> float:
    """Scale AQI to [0, 1] for regression stability."""
    return (aqi - AQI_MIN) / (AQI_MAX - AQI_MIN)


def denormalize_aqi(val: float) -> float:
    """Convert normalized prediction back to AQI scale."""
    return val * (AQI_MAX - AQI_MIN) + AQI_MIN


def split_dataset(df: pd.DataFrame, val_size=0.15, test_size=0.15, random_state=42):
    """
    Stratified train/val/test split based on AQI category.
    Ensures each split has proportional representation of all categories.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["category"],
        random_state=random_state
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df["category"],
        random_state=random_state
    )

    log.info(f"Split — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)


def load_and_preprocess_image(filepath: str, label: float,
                               augment: bool = False):
    """
    Loads, resizes, and preprocesses a single image.
    Uses EfficientNet-specific preprocessing (scales to [-1, 1]).
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)

    if augment:
        # Pollution-realistic augmentations only
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 255.0)

    # EfficientNet preprocessing: scale pixels to [-1, 1]
    img = (img / 127.5) - 1.0

    return img, label


def make_tf_dataset(df: pd.DataFrame,
                    augment: bool = False,
                    shuffle: bool = True,
                    batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """
    Builds a tf.data.Dataset from a manifest DataFrame.
    """
    # Normalize AQI labels to [0, 1]
    filepaths = df["filepath"].values
    labels    = df["aqi"].apply(normalize_aqi).values.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)

    dataset = dataset.map(
        lambda fp, lbl: load_and_preprocess_image(fp, lbl, augment=augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    log.info("Building manifests...")

    df_dense = build_dense_haze_manifest()
    df_dawn  = build_dawn_manifest()
    df       = pd.concat([df_dense, df_dawn], ignore_index=True)

    out_path = PROC_DIR / "manifest.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Manifest saved → {out_path}")

    print_stats(df)

    # ── Split ──────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_dataset(df)

    # Save splits
    train_df.to_csv(PROC_DIR / "train.csv", index=False)
    val_df.to_csv(PROC_DIR  / "val.csv",   index=False)
    test_df.to_csv(PROC_DIR / "test.csv",  index=False)
    log.info("Split CSVs saved → data/processed/")

    # ── Build tf.data datasets ─────────────────────────────────────────────────
    log.info("Building tf.data datasets...")
    train_ds = make_tf_dataset(train_df, augment=True,  shuffle=True)
    val_ds   = make_tf_dataset(val_df,   augment=False, shuffle=False)
    test_ds  = make_tf_dataset(test_df,  augment=False, shuffle=False)

    # ── Verify one batch ───────────────────────────────────────────────────────
    for images, labels in train_ds.take(1):
        print(f"\nBatch verification:")
        print(f"  Image batch shape : {images.shape}")
        print(f"  Label batch shape : {labels.shape}")
        print(f"  Image value range : {images.numpy().min():.3f} to {images.numpy().max():.3f}")
        print(f"  Label range       : {labels.numpy().min():.3f} to {labels.numpy().max():.3f}")
        print(f"  Sample AQI (denorm): {denormalize_aqi(float(labels.numpy()[0])):.1f}")

    log.info("Pipeline ready ✅")