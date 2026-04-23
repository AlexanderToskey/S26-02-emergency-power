"""
train_autoencoder_from_pipeline.py

Trains an autoencoder using the SAME preprocessing pipeline as the occurrence model.
This ensures perfect alignment with run_inference().
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Match pipeline structure
# Resolve project root (2 levels up from this file)
BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Add relevant subdirectories to Python path
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "outage_occurrence"))
sys.path.append(str(BASE_DIR / "outage_scope"))
sys.path.append(str(BASE_DIR / "outage_duration"))

from outage_occurrence.data_loader_occurrence import (
    build_occurrence_labels,
    merge_occurrence_with_weather
)
from outage_occurrence.preprocessor_occurrence import run_full_pipeline as run_occ_pipeline
from outage_occurrence.occurrence_model import OutageOccurrenceModel
from outage_scope.src.data_loader import load_eagle_outages
from autoencoder import Autoencoder

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
TEST_YEAR = 2022
SAVE_PATH = MODELS_DIR / "autoencoder.pt"

# ─────────────────────────────────────────────────────────────
# Helper: normalization
# ─────────────────────────────────────────────────────────────
def normalize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std

# ─────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("[AE] TRAINING AUTOENCODER FROM PIPELINE")
    print("=" * 60)

    # ── Load occurrence model (for feature alignment)
    print("[AE] Loading occurrence model...")
    occ_model = OutageOccurrenceModel.load(MODELS_DIR / "occurrence_model.joblib")

    # ── Load raw data (same as pipeline)
    print("[AE] Loading raw data...")
    eagle_files = sorted(DATA_DIR.glob("eaglei_outages_*.csv"))
    ghcnd = pd.read_csv(DATA_DIR / "ghcnd_va_daily.csv")

    outages = load_eagle_outages(eagle_files)

    # ── Build occurrence dataset
    print("[AE] Building occurrence dataset...")
    occurrence_labels = build_occurrence_labels(outages)
    occurrence_labels["fips_code"] = occurrence_labels["fips_code"].astype(str).str.zfill(5)
    ghcnd["fips_code"] = ghcnd["fips_code"].astype(str).str.zfill(5)
    merged_occ = merge_occurrence_with_weather(occurrence_labels, ghcnd)

    X_occ_full, y_occ_full = run_occ_pipeline(merged_occ)

    # ── Train ONLY on pre-2022 data
    print(f"[AE] Filtering training data (excluding {TEST_YEAR})...")
    train_mask = X_occ_full["year"] < TEST_YEAR
    X_train_df = X_occ_full.loc[train_mask].copy()

    print(f"[AE] Training samples: {len(X_train_df):,}")

    # ── Align features EXACTLY like inference
    print("[AE] Aligning feature columns...")
    feature_cols = occ_model.feature_columns

    for col in feature_cols:
        if col not in X_train_df.columns:
            X_train_df[col] = 0.0

    X_train = (
        X_train_df[feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values
    )

    # ── Normalize
    print("[AE] Normalizing features...")
    X_train_norm, mean, std = normalize(X_train)

    X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)

    # ── Build model
    print("[AE] Building model...")
    model = Autoencoder(input_dim=X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    # ── Train
    print("[AE] Training...")
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        recon = model(X_tensor)
        loss = loss_fn(recon, X_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"[AE] Epoch {epoch}, Loss: {loss.item():.6f}")

    # ── Compute threshold
    print("[AE] Computing anomaly threshold...")
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    threshold = np.percentile(errors, 95)

    # ── Save everything
    print("[AE] Saving model...")
    torch.save({
        "model_state": model.state_dict(),
        "threshold": float(threshold),
        "mean": torch.tensor(mean),
        "std": torch.tensor(std),
        "input_dim": X_tensor.shape[1],
        "feature_columns": feature_cols,
    }, SAVE_PATH)

    print(f"[AE] Saved → {SAVE_PATH}")
    print(f"[AE] Threshold: {threshold:.6f}")
    print("[AE] Done.")


if __name__ == "__main__":
    main()