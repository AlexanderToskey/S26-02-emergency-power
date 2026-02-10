# Outage Duration Prediction Model

XGBoost regression model that predicts power outage duration in minutes. Uses EAGLE-I outage records and NOAA weather data for Virginia (2014-2022).

## Setup

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the data:
```bash
python download_noaa_data.py       # NOAA storm events (~2 min)
python download_eaglei_data.py     # EAGLE-I outages (~15-20 min)
```

4. Verify the pipeline:
```bash
python test_pipeline.py
```

## Project Structure

```
OutageDurationModel/
├── data/                       # datasets (downloaded via scripts, gitignored)
├── src/
│   ├── data_loader.py          # load EAGLE-I and NOAA data, merge, validate
│   ├── preprocessor.py         # clean, feature engineering, train-ready output
│   ├── model.py                # XGBoost model
│   ├── evaluator.py            # MAE, RMSE, MAPE metrics
│   └── explainer.py            # SHAP analysis
├── notebooks/                  # jupyter notebooks for exploration
├── download_eaglei_data.py     # download EAGLE-I data from Figshare
├── download_noaa_data.py       # download NOAA storm events from NCEI
├── test_pipeline.py            # end-to-end pipeline smoke test
├── requirements.txt
└── README.md
```

## Quick Start

```python
from pathlib import Path
from src.data_loader import load_eagle_outages, load_noaa_weather, merge_weather_outages
from src.preprocessor import run_full_pipeline
from src.model import OutageDurationModel
from src.evaluator import evaluateModel, printEvaluationReport

# load data
eagle_files = sorted(Path('data').glob('eaglei_outages_*.csv'))
outages = load_eagle_outages(eagle_files)
weather = load_noaa_weather('data/noaa_storm_events_va_2014_2022.csv')
merged = merge_weather_outages(outages, weather)

# preprocess
X, y = run_full_pipeline(merged)

# train model
model = OutageDurationModel()
model.train(X, y)

# evaluate
preds = model.predict(X)
metrics = evaluateModel(y, preds)
printEvaluationReport(metrics)
```

## Data Sources

- **EAGLE-I**: County-level power outage snapshots at 15-min intervals ([Figshare](https://doi.org/10.6084/m9.figshare.24237376))
- **NOAA Storm Events**: Weather event records by county/zone ([NCEI](https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/))
- **NWS Zone-County Mapping**: Forecast zone to county FIPS crosswalk ([NWS](https://www.weather.gov/gis/ZoneCounty))

## Features

- **Temporal**: year, month, day, hour, dayofweek
- **Spatial**: fips_code
- **Weather**: event_type
- **Outage**: peak_customers_affected

## Performance Targets

| Metric | Marginal | Target |
|--------|----------|--------|
| Accuracy (within tolerance) | ≥70% | ≥90% |
| MAPE | <25% | <15% |

Tolerance: ±30min for short outages (<4hr), ±2hr for long outages (≥4hr)
