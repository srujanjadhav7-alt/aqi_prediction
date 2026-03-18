# src/train.py

import sys
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from model import build_model, unfreeze_backbone, model_summary
from data_pipeline import make_tf_dataset, denormalize_aqi

PROC_DIR     = PROJECT_ROOT / "data" / "processed"
SAVED_MODELS = PROJECT_ROOT / "saved_models"
LOGS_DIR     = PROJECT_ROOT / "logs"
SAVED_MODELS.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── Hyperparameters ────────────────────────────────────────────────────────────
STAGE1_EPOCHS = 30
STAGE2_EPOCHS = 50
BATCH_SIZE    = 32
STAGE1_LR     = 1e-3
STAGE2_LR     = 1e-5
PATIENCE      = 10


def get_callbacks(stage: int, model_name: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = SAVED_MODELS / f"{model_name}_stage{stage}.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(LOGS_DIR / f"stage{stage}_{timestamp}"),
            histogram_freq=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(LOGS_DIR / f"stage{stage}_{timestamp}.csv")
        ),
    ]

    return callbacks, ckpt_path


def evaluate_model(model, test_ds, label="Final"):
    results  = model.evaluate(test_ds, verbose=0)
    loss, mae, rmse = results
    mae_aqi  = mae  * 500
    rmse_aqi = rmse * 500

    print("\n" + "═" * 50)
    print(f"  [{label}] Test Results")
    print(f"  Huber Loss  : {loss:.4f}")
    print(f"  MAE         : {mae_aqi:.1f} AQI points")
    print(f"  RMSE        : {rmse_aqi:.1f} AQI points")
    print("═" * 50 + "\n")

    return mae_aqi, rmse_aqi


def train():
    # ── Load Data ──────────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_df = pd.read_csv(PROC_DIR / "train.csv")
    val_df   = pd.read_csv(PROC_DIR / "val.csv")
    test_df  = pd.read_csv(PROC_DIR / "test.csv")

    val_ds  = make_tf_dataset(val_df,  augment=False, shuffle=False, batch_size=BATCH_SIZE)
    test_ds = make_tf_dataset(test_df, augment=False, shuffle=False, batch_size=BATCH_SIZE)

    log.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Stage 1: Train Head Only ───────────────────────────────────────────────
    print("\n" + "━" * 50)
    print("  STAGE 1 — Training regression head (frozen backbone)")
    print("━" * 50)

    model = build_model(freeze_backbone=True)
    model_summary(model)

    train_ds_s1      = make_tf_dataset(train_df, augment=True, shuffle=True, batch_size=BATCH_SIZE)
    callbacks_s1, ckpt_s1 = get_callbacks(stage=1, model_name="aqi_model")

    history_s1 = model.fit(
        train_ds_s1,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=callbacks_s1,
        verbose=1
    )

    log.info(f"Stage 1 complete. Best model saved → {ckpt_s1}")

    # Load best Stage 1 weights before fine-tuning
    model.load_weights(str(ckpt_s1))

    # ── Stage 2: Fine-tune Top Backbone Layers ─────────────────────────────────
    print("\n" + "━" * 50)
    print("  STAGE 2 — Fine-tuning top backbone layers")
    print("━" * 50)

    model = unfreeze_backbone(model, learning_rate=STAGE2_LR)
    model_summary(model)

    train_ds_s2           = make_tf_dataset(train_df, augment=True, shuffle=True, batch_size=BATCH_SIZE)
    callbacks_s2, ckpt_s2 = get_callbacks(stage=2, model_name="aqi_model")

    history_s2 = model.fit(
        train_ds_s2,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=callbacks_s2,
        verbose=1
    )

    log.info(f"Stage 2 complete. Best model saved → {ckpt_s2}")

    # ── Final Evaluation ───────────────────────────────────────────────────────
    evaluate_model(model, test_ds, label="Stage 2 Fine-tuned")

    # Save final model
    final_path = SAVED_MODELS / "aqi_model_final.h5"
    model.save(str(final_path))
    log.info(f"Final model saved → {final_path}")

    return model, history_s1, history_s2


if __name__ == "__main__":
    train()