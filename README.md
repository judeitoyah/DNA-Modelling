# DNA-Modelling — NHS GP Appointment DNA Rate Forecasting

A modular Python pipeline that forecasts monthly **Did Not Attend (DNA) rates** for NHS England GP appointments using five complementary time-series and machine-learning models.

---

## Overview

| Item | Detail |
|---|---|
| **Dataset** | NHS England GP Appointment Publication Summary, March 2026 |
| **Tables used** | Table 1a (GP practice systems), 1b (PCN systems) |
| **Target variable** | Monthly GP DNA rate (%) |
| **Time period** | October 2023 – March 2026 (30 months) |
| **Models** | SARIMA · Prophet · Random Forest · XGBoost · MLP |
| **Pipeline stages** | Ingestion → Cleaning → EDA → Feature Engineering → Feature Selection → Modelling → Evaluation |

---

## Repository Structure

```
DNA-Modelling/
├── dna_forecast/             # Core Python package
│   ├── __init__.py
│   ├── config.py             # Paths, constants, plot style
│   ├── ingest.py             # Stage 1 — Data ingestion
│   ├── clean.py              # Stage 2 — Data cleaning
│   ├── features.py           # Stages 4 & 5 — Feature engineering & selection
│   ├── models.py             # Stage 6 — Five forecasting models
│   ├── evaluate.py           # Stage 7 — Metrics & cross-validation
│   └── visualise.py          # Stage 8 — All plots
├── notebooks/
│   └── Study1_DNA_Forecasting_Pipeline_RUN.ipynb   # Original exploratory notebook
├── data/                     # Place NHS CSV files here (not tracked by git)
├── outputs/                  # Generated metrics CSVs and plots
├── run_pipeline.py           # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## Methodology

The pipeline follows a seven-stage approach:

### Stage 1 — Data Ingestion
Reads three NHS England GP Appointment Publication Summary CSV tables. Extracts monthly time series for appointment volumes, DNA counts, modality mix (face-to-face, telephone, video), booking lead times, and staffing type.

### Stage 2 — Data Cleaning
Imputes missing values using linear interpolation (both directions) with forward/backward fill as fallback. Zero data loss after cleaning.

### Stage 3 — Exploratory Data Analysis
- GP vs PCN DNA rate time series with October peak annotation
- Appointment mode mix stacked area chart
- Seasonal distribution boxplot (monthly)
- Booking lead time vs DNA rate dual-axis chart
- Seasonal decomposition (trend, seasonal, residual)
- Augmented Dickey-Fuller stationarity test

### Stage 4 — Feature Engineering
22 engineered features across five categories:

| Category | Features |
|---|---|
| Autoregressive lags | `dna_lag_1/2/3/6/12` |
| Rolling statistics | `dna_roll3_mean`, `dna_roll6_mean`, `dna_roll3_std`, `dna_ewm3` |
| Temporal encoding | `month_sin`, `month_cos`, `is_october` |
| Momentum | `dna_diff1`, `dna_diff12`, `video_mom` |
| Derived ratios | `remote_index`, `gp_pcn_gap`, `appts_norm` |

### Stage 5 — Feature Selection
Three complementary methods run in parallel:
- **LassoCV** — penalised regression coefficient magnitudes
- **Random Forest importance** — mean decrease in impurity
- **Pearson correlation** — linear association with target

### Stage 6 — Model Training

| Model | Implementation | Key hyperparameters |
|---|---|---|
| SARIMA | statsmodels | (1,1,1)(1,1,0,12) |
| Prophet | prophet | yearly + quarterly seasonality, changepoint_prior=0.3 |
| Random Forest | scikit-learn | 500 trees, max_depth=5, max_features=0.6 |
| XGBoost | xgboost | 400 rounds, lr=0.04, max_depth=3, L1+L2 reg |
| MLP | scikit-learn | (64,32,16) ReLU, Adam, early stopping |

Prediction intervals: 90% confidence intervals via SARIMA analytical CI and Prophet built-in CI; 90% bootstrap intervals (200 resamples) for RF, XGBoost, and MLP.

### Stage 7 — Evaluation
- **Held-out test set**: last 6 months
- **Metrics**: RMSE, MAE, MAPE (%), R²
- **Cross-validation**: 5-fold time-series CV (test_size=2) for ML models

---

## Installation

```bash
# Clone the repository
git clone https://github.com/judeitoyah/DNA-Modelling.git
cd DNA-Modelling

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Data Setup

Place the three NHS GP Appointment Publication Summary CSV files in the `data/` folder:

```
data/
├── GP_Appointment_Publication_Summary_March_2026(Table 1a).csv
├── GP_Appointment_Publication_Summary_March_2026(Table 1b).csv
└── GP_Appointment_Publication_Summary_March_2026(Table 1c).csv
```

The data is available from [NHS England Statistics](https://www.england.nhs.uk/statistics/statistical-work-areas/appointment-in-general-practice/).

---

## Usage

### Run the full pipeline

```bash
python run_pipeline.py
```

### Custom paths and test window

```bash
python run_pipeline.py \
    --data-dir path/to/csv/files \
    --output-dir results \
    --test-months 6
```

### Import individual modules

```python
from dna_forecast.ingest import load_raw
from dna_forecast.clean import clean
from dna_forecast.features import engineer, make_ml_matrix
from dna_forecast.models import fit_random_forest
from dna_forecast.evaluate import score

df = clean(load_raw())
```

---

## Output Files

| File | Description |
|---|---|
| `outputs/metrics.csv` | RMSE, MAE, MAPE%, R², Train RMSE for all five models |
| `outputs/cross_validation.csv` | Mean and std RMSE from 5-fold time-series CV |

All charts are displayed interactively during the run.

---

## Results Summary

Evaluated on a 6-month held-out test set (October 2025 – March 2026).

| Model | RMSE (pp) | MAE (pp) | MAPE% | R² |
|---|---|---|---|---|
| XGBoost | best | — | — | — |
| Random Forest | — | — | — | — |
| MLP | — | — | — | — |
| Prophet | — | — | — | — |
| SARIMA | — | — | — | — |

*Run the pipeline to generate your own results table.*

---

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt)

---

## Author

**Jude Isememe Itoyah**  
Data Scientist & Analyst  
[github.com/judeitoyah](https://github.com/judeitoyah)

---

## Licence

MIT — see [LICENSE](LICENSE) for details.
