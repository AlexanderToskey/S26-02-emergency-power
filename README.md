# AI Predictive System for Emergency Power Response




## Quick Start

1. Make sure you have Python installed.

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

## Running the Dashboard Application

1. Run the Flask application:

```bash
python app/outage_app.py
```

2. Open a browser and go to:

```bash
http://127.0.0.1:5000/
```

You should see the Virginia county map. Hover over a county to see the predicted outage information.

Click on a county to see the feature importance for the prediction.

## Project Structure

```
S26-02-emergency-power/
├── anomaly_detection/
│   ├── autoencoder.py                      # Autoencoder for anomaly detection
│   ├── train_anomaly_detector.py           # Train isolation forest
│   └── train_autoencoder_from_pipeline.py  # Train autoencoder
├── app/
│   ├── static/
│   │   ├── Images/          # Images for dashboard
│   │   ├── counties.json    # County geometry shapes
│   │   ├── script.js        # JavaScript for dashboard website
│   │   └── style.css        # Stylesheet for dashboard website
│   ├── templates/
│   │   └── index.html       # HTML for dashboard website
│   └── outage_app.py        # Dashboard application script
├── data_download/
│   ├── duration/
│   │   ├── download_eaglei_data.py       # Eagle-I dataset download
│   │   ├── download_ghcnd_data.py        # GHCNd dataset download
│   │   └── download_noaa_data.py         # NOAA dataset download
│   ├── occurrence/
│   │   ├── download_eaglei_data.py       # Eagle-I dataset download
│   │   ├── download_ghcnd_data.py        # GHCNd dataset download
│   │   └── download_noaa_data.py         # NOAA dataset download
│   ├── scope/
│   │   ├── download_eaglei_data.py       # Eagle-I dataset download
│   │   ├── download_ghcnd_data.py        # GHCNd dataset download
│   │   └── download_noaa_data.py         # NOAA dataset download
│   ├── download_openmeteo_historical.py  # Open-Meteo historical dataset download
│   ├── export_county_stats.py            # Compute historical outage stats from EAGLE-I
│   └── generate_virginia_geo.py          # Generate latitude/longitude data for counties
├── inference/
│   ├── realtime_data_inventory.py        # Map real-time data types to models
│   ├── realtime_inference.py             # Real-time prediction pipeline for Virginia outages
│   └── weatherapi.py                     # Fetch real-time weather via Open-Meteo.
├── models/
│   ├── autoencoder.pt
│   ├── duration_forecast.joblib          
│   ├── duration_model.joblib
│   ├── isolation_forest.joblib
│   ├── occurrence_model.joblib           
│   ├── scope_forecast.joblib
│   └── scope_model.joblib                
├── outage_duration/
│   ├── src/
│   │   ├── data_loader.py                # Load and merge EAGLE-I and NOAA data
│   │   ├── evaluator.py                  # Compute MAE, RMSE, and MAPE metrics
│   │   ├── explainer.py                  # Duration SHAP analysis
│   │   ├── model.py                      # One-stage duration model
│   │   ├── preprocessor.py               # Feature engineering
│   │   └── two_stage_model.py            # Two-stage duration model
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt
│   └── train_eval.py                     # Train and evaluate duration model
├── outage_occurrence/
│   ├── data_loader_occurrence.py         # Load and merge EAGLE-I, NOAA, and GHCNd data
│   ├── evaluator_occurrence.py           # Compute accuracy, precision, recall, F1, ROC_AUC, PR_AUC
│   ├── occurrence_explainer_model.py     # Occurrence SHAP analysis
│   ├── occurrence_model.py               # Occurrence model
│   ├── occurrence_test.py                # Basic model and SHAP test
│   ├── preprocessor_occurrence.py        # Extract and prepare features for training
│   ├── test_pipeline_occurrence.py       # Smoke test for data loading and preprocessing pipline
│   └── train_eval_occurrence.py          # Train and evaluate occurrence model
├── outage_scope/
│   ├── models/
│   │   ├── .gitkeep
│   │   ├── autoencoder.pt
│   │   ├── duration_model.joblib
│   │   └── scope_model.joblib
│   ├── src/
│   │   ├── data_loader.py                # Load and merge EAGLE-I, NOAA, and GHCNd data
│   │   ├── evaluator.py                  # Compute MAE, RMSE, and MAPE metrics
│   │   ├── explainer.py                  # Scope SHAP analysis
│   │   ├── model.py                      # One-stage scope model
│   │   ├── preprocessor.py               # Extract and prepare features for training
│   │   └── two_stage_model.py            # Two-stage scope model
│   ├── utils/
│   │   ├── .gitkeep
│   ├── README.md
│   ├── test_pipeline.py                  # Smoke test for data loading and preprocessing pipeline
│   └── train_eval_scope.py               # Train and evaluate scope model
├── tests/
│   ├── api_test.py    # Test real-time data APIs
├── .gitignore         # Files for Git to ignore
├── README.md          # Main project landing page
├── requirements.txt   # Required packages and dependencies
└── run_pipeline.py    # Run full cascade system tests
```

