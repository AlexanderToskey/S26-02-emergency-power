#Trains an autoencoder on historical weather data to detect unusual conditions at inference time
#Uses the same preprocessing pipeline as the occurrence model so the features line up exactly

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

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

#Hold out 2022 so the anomaly threshold isn't inflated by the test year.
TEST_YEAR = 2022
SAVE_PATH = MODELS_DIR / "autoencoder.pt"


def normalize(X: np.ndarray):
    #Standard z-score normalization
    # Clamp std to 1 for constant features so we don't divide by zero
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def main():
    """Load occurrence data, train the autoencoder on pre-2022 samples, and save to models/."""
    print("=" * 60)
    print("[AE] TRAINING AUTOENCODER FROM PIPELINE")
    print("=" * 60)

    #Load the occurrence model just to borrow its feature column list
    #This guarantees the autoencoder sees the same features as the live pipeline
    print("[AE] Loading occurrence model...")
    occ_model = OutageOccurrenceModel.load(MODELS_DIR / "occurrence_model.joblib")

    print("[AE] Loading raw data...")
    eagle_files = sorted(DATA_DIR.glob("eaglei_outages_*.csv"))
    ghcnd = pd.read_csv(DATA_DIR / "ghcnd_va_daily.csv")
    outages = load_eagle_outages(eagle_files)

    #Build the same county-day dataset the occurrence model was trained on
    print("[AE] Building occurrence dataset...")
    occurrence_labels = build_occurrence_labels(outages)
    occurrence_labels["fips_code"] = occurrence_labels["fips_code"].astype(str).str.zfill(5)
    ghcnd["fips_code"] = ghcnd["fips_code"].astype(str).str.zfill(5)
    merged_occ = merge_occurrence_with_weather(occurrence_labels, ghcnd)
    X_occ_full, y_occ_full = run_occ_pipeline(merged_occ)

    #Only train on pre-2022 data so the threshold reflects truly "normal" conditions
    print(f"[AE] Filtering training data (excluding {TEST_YEAR})...")
    train_mask = X_occ_full["year"] < TEST_YEAR
    X_train_df = X_occ_full.loc[train_mask].copy()
    print(f"[AE] Training samples: {len(X_train_df):,}")

    #Zero-fill any features the occurrence model expects but the dataset doesn't have
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

    #Normalize and save the stats.
    print("[AE] Normalizing features...")
    X_train_norm, mean, std = normalize(X_train)
    X_tensor = torch.tensor(X_train_norm, dtype=torch.float32)

    print("[AE] Building model...")
    model = Autoencoder(input_dim=X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    #Full-batch training — the dataset fits in memory so no DataLoader needed
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

    #Set the anomaly threshold at the 95th percentile of training errors
    #Anything above this at inference is flagged as unusual.
    print("[AE] Computing anomaly threshold...")
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    threshold = np.percentile(errors, 95)

    #Save the model weights alongside the normalization stats and feature list
    #so inference can reconstruct everything from a single file.
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