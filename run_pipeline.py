"""
run_pipeline.py - Centralized Execution for the Predictive Intelligence Suite
Runs the complete data loading, preprocessing, and model cascade.
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# --- 1. Dynamic Pathing for Subfolders ---
OCCURRENCE_FOLDER = "outage_occurrence"
DURATION_FOLDER = "outage_duration"
sys.path.append(OCCURRENCE_FOLDER)
sys.path.append(DURATION_FOLDER)

# --- Occurrence Imports ---
from outage_occurrence.data_loader_occurrence import build_occurrence_labels, merge_occurrence_with_weather
from outage_occurrence.preprocessor_occurrence import run_full_pipeline as run_occ_pipeline
from outage_occurrence.evaluator_occurrence import evaluateModel as evaluate_occ
from outage_occurrence.evaluator_occurrence import printEvaluationReport as print_occ_report
from outage_occurrence.occurrence_model import OutageOccurrenceModel

# --- Event (Scope) Imports ---
from outage_scope.src.data_loader import (
    load_eagle_outages, load_noaa_weather, merge_weather_outages, 
    load_ghcnd_weather, merge_ghcnd_weather
)
from outage_scope.src.preprocessor import run_full_pipeline as run_scope_pipeline
from outage_scope.src.evaluator import evaluateModel as evaluate_scope
from outage_scope.src.evaluator import printEvaluationReport as print_scope_report
from outage_scope.src.two_stage_model import TwoStageScopeModel

# --- Event (Duration) Imports ---
from outage_duration.src.preprocessor import run_full_pipeline as run_dur_pipeline
from outage_duration.src.evaluator import evaluateModel as evaluate_dur
from outage_duration.src.evaluator import printEvaluationReport as print_dur_report
from outage_duration.src.two_stage_model import TwoStageOutageModel

def main():
    print("="*60)
    print("INITIALIZING PREDICTIVE INTELLIGENCE SUITE")
    print("="*60)

    # Get the base directory and get the data and model directories
    BASE_DIR = Path(__file__).resolve().parent
    data_dir = BASE_DIR / "data"
    models_dir = BASE_DIR / "models"

    # =========================================================================
    # STEP 1: LOAD ALL MODELS
    # =========================================================================
    print("\n[1] Loading Serialized Models from /models...")
    try:
        occ_model = OutageOccurrenceModel.load(models_dir / "occurrence_model.joblib")
        scope_model = TwoStageScopeModel.load(models_dir / "scope_model.joblib")
        dur_model = TwoStageOutageModel.load(models_dir / "duration_model.joblib")
        print("  -> Models loaded successfully.")
    except Exception as e:
        print(f"  [ERROR] Could not load models: {e}")
        return

    # =========================================================================
    # STEP 2: LOAD AND PREPROCESS RAW DATA
    # =========================================================================
    print("\n[2] Loading Raw Data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")
    weather = load_noaa_weather(data_dir / "noaa_storm_events_va_2014_2022.csv")

    # Occurrence dataset
    print("\n--- Building Occurrence (County-Day) Dataset ---")
    #print(outages.columns)
    occurrence_labels = build_occurrence_labels(outages)
    merged_occ = merge_occurrence_with_weather(occurrence_labels, ghcnd)

    X_occ_full, y_occ_full = run_occ_pipeline(merged_occ)
    

    # Scope and duration datasets
    print("\n--- Building Event (Scope & Duration) Datasets ---")
    merged_scope = merge_weather_outages(outages, weather)
    merged_scope = merge_ghcnd_weather(merged_scope, ghcnd)

    # Run pipelines to get the specific target variables for Scope and Duration
    X_scope_full, y_scope_full = run_scope_pipeline(merged_scope)
    X_dur_full, y_dur_full = run_dur_pipeline(merged_scope)

    # =========================================================================
    # STEP 3: TEMPORAL SPLIT (TESTING ON 2022 ONLY)
    # =========================================================================
    TEST_YEAR = 2022
    print(f"\n[3] Isolating {TEST_YEAR} Test Data...")

    # Occurrence Split
    occ_test_mask = X_occ_full['year'] == TEST_YEAR
    X_occ_test = X_occ_full[occ_test_mask].copy()
    y_occ_test = y_occ_full[occ_test_mask].copy()

    # Scope Split
    scope_test_mask = X_scope_full['year'] == TEST_YEAR
    X_scope_test = X_scope_full[scope_test_mask].copy()
    y_scope_test = y_scope_full[scope_test_mask].copy()

    # Duration Split
    dur_test_mask = X_dur_full['year'] == TEST_YEAR
    X_dur_test = X_dur_full[dur_test_mask].copy()
    y_dur_test = y_dur_full[dur_test_mask].copy()

    print(scope_model.featureNames)

    print(f"  Occurrence Test Samples: {len(X_occ_test):,}")
    print(f"  Event Test Samples:      {len(X_scope_test):,}")

    # Before Step 4, ensure the columns exist. 
    # If they were dropped, you need to re-add them or stop dropping them.
    print(X_scope_test.columns) # Run this to see what is actually there.

    # =========================================================================
    # STEP 4: ORACLE EVALUATION (MODELS IN A VACUUM)
    # =========================================================================
    print("\n" + "="*60)
    print("PART A: ORACLE EVALUATION (PERFECT ROUTING)")
    print("="*60)

    # --- 1. Occurrence Model Oracle ---
    print("\n--- 1. Occurrence Model (Standalone) ---")
    preds_occ, prob_occ = occ_model.predict(X_occ_test)
    y_pred_occ = (prob_occ >= 0.5).astype(int)

    occ_metrics = evaluate_occ(y_occ_test.values, y_pred_occ, prob_occ)
    print_occ_report(occ_metrics)

    # --- 2. Scope Model Oracle ---
    print("\n--- 2. Scope Model (Standalone - Perfect Gatekeeper) ---")
    SCOPE_THRESHOLD = 500
    oracle_scope_preds = np.zeros(len(X_scope_test))
    s_large_mask = y_scope_test >= SCOPE_THRESHOLD
    s_small_mask = y_scope_test < SCOPE_THRESHOLD
    
    if s_small_mask.sum() > 0:
        p_log_s_small = scope_model.shortRegressor.predict(X_scope_test.loc[s_small_mask, scope_model.featureNames])
        oracle_scope_preds[s_small_mask] = np.expm1(np.clip(p_log_s_small, -20, 20))
    if s_large_mask.sum() > 0:
        p_log_s_large = scope_model.longRegressor.predict(X_scope_test.loc[s_large_mask, scope_model.featureNames])
        oracle_scope_preds[s_large_mask] = np.expm1(np.clip(p_log_s_large, -20, 20))

    scope_oracle_metrics = evaluate_scope(y_scope_test.values, oracle_scope_preds)
    print_scope_report(scope_oracle_metrics)

    # --- 3. Duration Model Oracle ---
    print("\n--- 3. Duration Model (Standalone - Perfect Gatekeeper) ---")
    LONG_THRESHOLD_MIN = 240.0
    oracle_dur_preds = np.zeros(len(X_dur_test))
    d_large_mask = y_dur_test >= LONG_THRESHOLD_MIN
    d_small_mask = y_dur_test < LONG_THRESHOLD_MIN
    
    if d_small_mask.sum() > 0:
        p_log_d_small = dur_model.shortRegressor.predict(X_dur_test.loc[d_small_mask, dur_model.featureNames])
        oracle_dur_preds[d_small_mask] = np.expm1(np.clip(p_log_d_small, -20, 20))
    if d_large_mask.sum() > 0:
        p_log_d_large = dur_model.longRegressor.predict(X_dur_test.loc[d_large_mask, dur_model.featureNames])
        oracle_dur_preds[d_large_mask] = np.expm1(np.clip(p_log_d_large, -20, 20))

    dur_oracle_metrics = evaluate_dur(y_dur_test.values, oracle_dur_preds)
    print_dur_report(dur_oracle_metrics)

    # =========================================================================
    # STEP 5: THE CASCADE PIPELINE
    # =========================================================================
    print("\n" + "="*60)
    print("PART B: CASCADING PIPELINE (REAL-WORLD ERROR PROPAGATION)")
    print("="*60)
    
    #DEBUG: REMOVE =====================
    print(f"Occurrence has 'day': {'day' in X_occ_test.columns}")
    print(f"Scope has 'day': {'day' in X_scope_test.columns}")
    # ==================

    # 5a. STAGE 1 (Occurrence) maps to STAGE 2 (Event)
    occ_preds_df = X_occ_test[['fips_code', 'year', 'month', 'day']].copy()
    occ_preds_df['occ_prediction'] = y_pred_occ

    join_cols = ['fips_code', 'year', 'month', 'day']

    # Map the occurrence predictions to the scope dataset
    X_scope_mapped = X_scope_test.reset_index().merge(
        occ_preds_df, on=join_cols, how='left'
    ).set_index('index')
    X_dur_mapped = X_dur_test.reset_index().merge(
        occ_preds_df, on=join_cols, how='left'
    ).set_index('index')

    pipeline_event_mask = X_scope_mapped['occ_prediction'] == 1
    missed_events = (~pipeline_event_mask).sum()
    print(f"  -> Occurrence Model missed {missed_events:,} actual events (These will be predicted as 0).")

    # Initialize final pipeline prediction arrays
    pipeline_scope_preds = np.zeros(len(X_scope_test))
    pipeline_dur_preds = np.zeros(len(X_dur_test))

    # 5b. STAGE 2 & 3 (Gatekeepers & Specialists)
    # --- SCOPE PREDICTION ---
    scope_mask = (X_scope_mapped['occ_prediction'] == 1).values
    if scope_mask.sum() > 0:
        X_input_scope = X_scope_mapped.loc[scope_mask, scope_model.featureNames]
        pipeline_scope_preds[scope_mask] = scope_model.predict(X_input_scope)

    # --- DURATION PREDICTION ---
    # We create a NEW mask specifically for the Duration dataframe's length
    dur_mask = (X_dur_mapped['occ_prediction'] == 1).values
    if dur_mask.sum() > 0:
        X_input_dur = X_dur_mapped.loc[dur_mask, dur_model.featureNames]
        pipeline_dur_preds[dur_mask] = dur_model.predict(X_input_dur)

    # 5c. Final Pipeline Evaluation
    print("\n--- End-to-End Cascaded Pipeline Evaluation (SCOPE) ---")
    final_scope_metrics = evaluate_scope(y_scope_test.values, pipeline_scope_preds)
    print_scope_report(final_scope_metrics)

    print("\n--- End-to-End Cascaded Pipeline Evaluation (DURATION) ---")
    final_dur_metrics = evaluate_dur(y_dur_test.values, pipeline_dur_preds)
    print_dur_report(final_dur_metrics)

if __name__ == "__main__":
    main()