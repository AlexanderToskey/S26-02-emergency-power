# Outage Duration Prediction Model

XGBoost regression model that predicts power outage duration in minutes. Uses EAGLE-I outage records and NOAA weather data.

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

3. Put your data files in the `/data` folder:
   - EAGLE-I outage records (CSV)
   - NOAA weather data (CSV)

## Project Structure

```
OutageDurationModel/
├── data/               # datasets go here
├── src/
│   ├── data_loader.py  # load EAGLE-I and NOAA data
│   ├── preprocessor.py # clean and transform data
│   ├── model.py        # XGBoost model
│   ├── evaluator.py    # MAE, RMSE, MAPE metrics
│   └── explainer.py    # SHAP analysis
├── notebooks/          # jupyter notebooks for exploration
├── requirements.txt
└── README.md
```

## Quick Start

```python
from src.data_loader import loadBothDatasets
from src.preprocessor import runFullPipeline
from src.model import OutageDurationModel
from src.evaluator import evaluateModel, printEvaluationReport
from src.explainer import OutageExplainer

# load data
eagleDf, noaaDf = loadBothDatasets('data/eagle_i.csv', 'data/noaa.csv')

# preprocess
X, y = runFullPipeline(eagleDf, noaaDf)

# train model
model = OutageDurationModel()
model.train(X, y)

# evaluate
preds = model.predict(X)
metrics = evaluateModel(y, preds)
printEvaluationReport(metrics)

# explain
explainer = OutageExplainer(model, X)
explainer.plotSummary(X)
```

## Features

- **Temporal**: year, month, day, hour, dayofweek
- **Spatial**: fips_code, state_code
- **Weather**: weather_code

## Performance Targets

| Metric | Marginal | Target |
|--------|----------|--------|
| Accuracy (within tolerance) | ≥70% | ≥90% |
| MAPE | <25% | <15% |

Tolerance: ±30min for short outages (<4hr), ±2hr for long outages (≥4hr)
