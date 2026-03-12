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
from outage_occurrence.occurrence_model import OutageOccurrenceModel
from outage_occurrence.data_loader_occurrence import build_occurrence_labels, merge_occurrence_with_weather
from outage_occurrence.preprocessor_occurrence import run_full_pipeline as run_occ_pipeline
from outage_occurrence.evaluator_occurrence import evaluateModel as evaluate_occ
from outage_occurrence.evaluator_occurrence import printEvaluationReport as print_occ_report

# --- Event (Scope) Imports ---
from outage_scope.src.data_loader import (
    load_eagle_outages, load_noaa_weather, merge_weather_outages, 
    load_ghcnd_weather, merge_ghcnd_weather
)
from outage_scope.src.preprocessor import run_full_pipeline as run_event_pipeline
from outage_scope.src.evaluator import evaluateModel as evaluate_event
from outage_scope.src.evaluator import printEvaluationReport as print_event_report
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

    data_dir = Path("data")
    models_dir = Path("models")

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
    # STEP 2: LOAD AND UNIFY DATA TYPES
    # =========================================================================
    print("\n[2] Loading Raw Data...")
    eagle_files = sorted(data_dir.glob("eaglei_outages_*.csv"))
    outages = load_eagle_outages(eagle_files)
    ghcnd = load_ghcnd_weather(data_dir / "ghcnd_va_daily.csv")
    weather = load_noaa_weather(data_dir / "noaa_storm_events_va_2014_2022.csv")

    # MANDATORY UNIFICATION: Force every key to the same type for ALL models
    for df in [outages, ghcnd, weather]:
        if 'fips_code' in df.columns:
            df['fips_code'] = df['fips_code'].astype(str)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'run_start_time' in df.columns:
            df['date'] = pd.to_datetime(df['run_start_time']).dt.normalize()

    # --- Occurrence Path ---
    print("\n--- Building Occurrence (County-Day) Dataset ---")
    # All the 'duct tape' is gone here. The updated functions handle it perfectly.
    occurrence_labels = build_occurrence_labels(outages)
    merged_occ = merge_occurrence_with_weather(occurrence_labels, ghcnd)
    X_occ_full, y_occ_full = run_occ_pipeline(merged_occ)

    # --- Event Path ---
    print("\n--- Building Event (Scope & Duration) Datasets ---")
    merged_event = merge_weather_outages(outages, weather)
    
    # Re-verify types for the event-level merge
    merged_event['date'] = pd.to_datetime(merged_event['run_start_time']).dt.normalize()
    merged_event['fips_code'] = merged_event['fips_code'].astype(str)

    merged_event = pd.merge(merged_event, ghcnd, on=["fips_code", "date"], how="left") # <-- ADD THIS
    
    # Assigning the event results
    X_event_full, y_scope_full = run_event_pipeline(merged_event)
    X_dur_full, y_dur_full = run_dur_pipeline(merged_event)

    print("  -> Data types unified across all datasets.")

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
    scope_test_mask = X_event_full['year'] == TEST_YEAR
    X_scope_test = X_event_full[scope_test_mask].copy()
    y_scope_test = y_scope_full[scope_test_mask].copy()

    # Duration Split
    dur_test_mask = X_dur_full['year'] == TEST_YEAR
    X_dur_test = X_dur_full[dur_test_mask].copy()
    y_dur_test = y_dur_full[dur_test_mask].copy()

    # =========================================================================
    # Generalization: Drop future-leaking features from events
    # =========================================================================
    cols_to_drop = ['year', 'initial_customers_affected', 'delta_customers_affected_15m', 'pct_growth_15m']
    
    # Scope model was trained WITHOUT these, so we must drop them
    X_scope_test = X_scope_test.drop(columns=[c for c in cols_to_drop if c in X_scope_test.columns])
    
    # THE FIX: Duration model was trained WITH these, so we CANNOT drop them!
    # X_dur_test = X_dur_test.drop(columns=[c for c in cols_to_drop if c in X_dur_test.columns])
    
    # Occurrence model doesn't use year
    X_occ_test = X_occ_test.drop(columns=['year'], errors='ignore')

    print(f"  Occurrence Test Samples: {len(X_occ_test):,}")
    print(f"  Event Test Samples:      {len(X_scope_test):,}")

    # =========================================================================
    # STEP 4: ORACLE EVALUATION (MODELS IN A VACUUM)
    # =========================================================================
    print("\n" + "="*60)
    print("PART A: ORACLE EVALUATION (PERFECT ROUTING)")
    print("="*60)

    # --- 1. Occurrence Model Oracle ---
    print("\n--- 1. Occurrence Model (Standalone) ---")
    # Clean unpacking of the updated .predict() method
    preds_occ_hard, preds_occ_prob = occ_model.predict(X_occ_test)
    occ_metrics = evaluate_occ(y_occ_test.values, preds_occ_hard, preds_occ_prob)
    print_occ_report(occ_metrics)

    # --- 2. Scope Model Oracle ---
    print("\n--- 2. Scope Model (Standalone - Perfect Gatekeeper) ---")
    oracle_scope_preds = np.zeros(len(X_scope_test))
    s_large_mask = y_scope_test >= scope_model.thresholdMin
    s_small_mask = ~s_large_mask
    
    if s_small_mask.sum() > 0:
        p_log_s_small = scope_model.shortRegressor.predict(X_scope_test[s_small_mask])
        oracle_scope_preds[s_small_mask] = np.expm1(np.clip(p_log_s_small, -20, 20))
    if s_large_mask.sum() > 0:
        p_log_s_large = scope_model.longRegressor.predict(X_scope_test[s_large_mask])
        oracle_scope_preds[s_large_mask] = np.expm1(np.clip(p_log_s_large, -20, 20))

    scope_oracle_metrics = evaluate_event(y_scope_test.values, oracle_scope_preds)
    print_event_report(scope_oracle_metrics)

    # --- 3. Duration Model Oracle ---
    print("\n--- 3. Duration Model (Standalone - Perfect Gatekeeper) ---")
    oracle_dur_preds = np.zeros(len(X_dur_test))
    d_large_mask = y_dur_test >= dur_model.thresholdMin
    d_small_mask = ~d_large_mask
    
    if d_small_mask.sum() > 0:
        p_log_d_small = dur_model.shortRegressor.predict(X_dur_test[d_small_mask])
        oracle_dur_preds[d_small_mask] = np.expm1(np.clip(p_log_d_small, -20, 20))
    if d_large_mask.sum() > 0:
        p_log_d_large = dur_model.longRegressor.predict(X_dur_test[d_large_mask])
        oracle_dur_preds[d_large_mask] = np.expm1(np.clip(p_log_d_large, -20, 20))

    dur_oracle_metrics = evaluate_dur(y_dur_test.values, oracle_dur_preds)
    print_dur_report(dur_oracle_metrics)

    # =========================================================================
    # STEP 5: THE CASCADE PIPELINE
    # =========================================================================
    print("\n" + "="*60)
    print("PART B: CASCADING PIPELINE (REAL-WORLD ERROR PROPAGATION)")
    print("="*60)
    
    # 5a. STAGE 1 (Occurrence) maps to STAGE 2 (Event)
    # Reconstruct the exact date for mapping using the original DataFrames' preserved indexes
    occ_preds_df = merged_occ.loc[X_occ_test.index, ['fips_code', 'date']].copy()
    occ_preds_df['occ_prediction'] = preds_occ_hard

    scope_dates_df = merged_event.loc[X_scope_test.index, ['fips_code', 'date']].copy()
    dur_dates_df = merged_event.loc[X_dur_test.index, ['fips_code', 'date']].copy()

    # Merge occurrence predictions onto the events based on Date and FIPS
    mapped_scope = pd.merge(scope_dates_df.reset_index(), occ_preds_df, on=['fips_code', 'date'], how='left').set_index('index')
    mapped_dur = pd.merge(dur_dates_df.reset_index(), occ_preds_df, on=['fips_code', 'date'], how='left').set_index('index')

    pipeline_event_mask = mapped_scope['occ_prediction'] == 1
    missed_events = (~pipeline_event_mask).sum()
    print(f"  -> Occurrence Model missed {missed_events:,} actual events (These will be predicted as 0).")

    # Initialize final pipeline prediction arrays
    pipeline_scope_preds = np.zeros(len(X_scope_test))
    pipeline_dur_preds = np.zeros(len(X_dur_test))

    # 5b. STAGE 2 & 3 (Gatekeepers & Specialists)
    if pipeline_event_mask.sum() > 0:
        X_predicted_scope = X_scope_test[pipeline_event_mask]
        X_predicted_dur = X_dur_test[pipeline_event_mask]
        
        print(f"  -> Routing {pipeline_event_mask.sum():,} predicted outages through Gatekeepers...")
        pipeline_scope_preds[pipeline_event_mask] = scope_model.predict(X_predicted_scope)
        pipeline_dur_preds[pipeline_event_mask] = dur_model.predict(X_predicted_dur)

    # 5c. Final Pipeline Evaluation
    print("\n--- End-to-End Cascaded Pipeline Evaluation (SCOPE) ---")
    final_scope_metrics = evaluate_event(y_scope_test.values, pipeline_scope_preds)
    print_event_report(final_scope_metrics)

    print("\n--- End-to-End Cascaded Pipeline Evaluation (DURATION) ---")
    final_dur_metrics = evaluate_dur(y_dur_test.values, pipeline_dur_preds)
    print_dur_report(final_dur_metrics)

if __name__ == "__main__":
    main()