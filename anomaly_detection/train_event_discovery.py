"""
train_iforest_from_pipeline.py

Trains an Isolation Forest using the SAME preprocessing pipeline
as the occurrence model for perfect alignment with inference.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────
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

# 🔹 IMPORT YOUR EXISTING PIPELINE
from eventdiscovery import (
    DataPreprocessor,
    IsolationForestModel,
    ModelTrainer
)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
TEST_YEAR = 2022

SCALER_PATH = MODELS_DIR / "if_scaler.joblib"
MODEL_PATH = MODELS_DIR / "if_model.joblib"

# ─────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("[IF] TRAINING ISOLATION FOREST FROM PIPELINE")
    print("=" * 60)

    # ── Load occurrence model (for feature alignment)
    print("[IF] Loading occurrence model...")
    occ_model = OutageOccurrenceModel.load(MODELS_DIR / "occurrence_model.joblib")

    # ── Load raw data
    print("[IF] Loading raw data...")
    eagle_files = sorted(DATA_DIR.glob("eaglei_outages_*.csv"))
    ghcnd = pd.read_csv(DATA_DIR / "ghcnd_va_daily.csv")

    outages = load_eagle_outages(eagle_files)

    # ── Build dataset
    print("[IF] Building dataset...")
    occurrence_labels = build_occurrence_labels(outages)

    occurrence_labels["fips_code"] = occurrence_labels["fips_code"].astype(str).str.zfill(5)
    ghcnd["fips_code"] = ghcnd["fips_code"].astype(str).str.zfill(5)

    merged_occ = merge_occurrence_with_weather(occurrence_labels, ghcnd)

    X_occ_full, _ = run_occ_pipeline(merged_occ)

    # ── Train ONLY on pre-2022 data
    print(f"[IF] Filtering training data (excluding {TEST_YEAR})...")
    train_mask = X_occ_full["year"] < TEST_YEAR
    X_train_df = X_occ_full.loc[train_mask].copy()

    print(f"[IF] Training samples: {len(X_train_df):,}")

    # ── Align features EXACTLY like inference
    print("[IF] Aligning feature columns...")
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

    # ── Initialize pipeline
    print("[IF] Initializing model pipeline...")
    preprocessor = DataPreprocessor()
    model = IsolationForestModel(
        contamination=0.05,
        n_estimators=100,
        random_state=42
    )

    trainer = ModelTrainer(preprocessor, model)

    # ── Train
    print("[IF] Training Isolation Forest...")
    trainer.train(X_train)

    # ── Save pipeline
    print("[IF] Saving model...")
    trainer.save_pipeline(SCALER_PATH, MODEL_PATH)

    print(f"[IF] Saved scaler → {SCALER_PATH}")
    print(f"[IF] Saved model  → {MODEL_PATH}")
    print("[IF] Done.")


if __name__ == "__main__":
    main()