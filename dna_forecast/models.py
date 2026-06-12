"""Stage 6 — Model training.

Five forecasting models, each returning a standardised result dict:
    {
        "train": np.ndarray,   # in-sample fitted values
        "test":  np.ndarray,   # out-of-sample forecasts
        "lower": np.ndarray,   # 90 % prediction interval lower bound
        "upper": np.ndarray,   # 90 % prediction interval upper bound
        "fit_object": <model>  # fitted model for inspection / persistence
    }

Models
------
- SARIMA   (1,1,1)(1,1,0,12)  via statsmodels
- Prophet  with quarterly seasonality  via prophet
- Random Forest  with bootstrap CI  via scikit-learn
- XGBoost  with bootstrap CI  via xgboost
- MLP      with bootstrap CI  via scikit-learn
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from .config import N_BOOTSTRAP, RANDOM_SEED


# ---------------------------------------------------------------------------
# Bootstrap confidence-interval helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    model_cls,
    model_kwargs: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    ci: float = 90.0,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) bootstrap prediction intervals for an sklearn model."""
    rng = np.random.default_rng(seed)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X_train), len(X_train))
        m = model_cls(**model_kwargs)
        m.fit(X_train[idx], y_train[idx])
        preds.append(m.predict(X_test))
    arr = np.array(preds)
    return np.percentile(arr, lo, axis=0), np.percentile(arr, hi, axis=0)


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

def fit_sarima(train_ts: pd.Series, test_ts: pd.Series) -> dict:
    """Fit SARIMA(1,1,1)(1,1,0,12) and forecast over the test period."""
    model = SARIMAX(
        train_ts,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.get_forecast(steps=len(test_ts))
    ci = forecast.conf_int()
    return {
        "train":      fit.fittedvalues.values,
        "test":       forecast.predicted_mean.values,
        "lower":      ci.iloc[:, 0].values,
        "upper":      ci.iloc[:, 1].values,
        "fit_object": fit,
    }


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def fit_prophet(train_ts: pd.Series, test_ts: pd.Series) -> dict:
    """Fit Prophet with yearly + quarterly seasonality."""
    train_df = pd.DataFrame({"ds": train_ts.index, "y": train_ts.values})
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.3,
        seasonality_prior_scale=5.0,
        interval_width=0.90,
    )
    m.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
    m.fit(train_df)

    forecast_test  = m.predict(pd.DataFrame({"ds": test_ts.index}))
    forecast_train = m.predict(pd.DataFrame({"ds": train_ts.index}))
    return {
        "train":      forecast_train["yhat"].values,
        "test":       forecast_test["yhat"].values,
        "lower":      forecast_test["yhat_lower"].values,
        "upper":      forecast_test["yhat_upper"].values,
        "fit_object": m,
    }


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

def fit_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Fit Random Forest with bootstrap prediction intervals."""
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=2,
        max_features=0.6,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    lower, upper = _bootstrap_ci(
        RandomForestRegressor,
        {"n_estimators": 100, "max_depth": 5, "random_state": 0, "max_features": 0.6},
        X_train, y_train, X_test, n_boot=n_boot,
    )
    return {
        "train":      model.predict(X_train),
        "test":       model.predict(X_test),
        "lower":      lower,
        "upper":      upper,
        "fit_object": model,
    }


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def fit_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Fit XGBoost with bootstrap prediction intervals."""
    model = XGBRegressor(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=random_state,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    lower, upper = _bootstrap_ci(
        XGBRegressor,
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05,
         "random_state": 0, "verbosity": 0},
        X_train, y_train, X_test, n_boot=n_boot,
    )
    return {
        "train":      model.predict(X_train),
        "test":       model.predict(X_test),
        "lower":      lower,
        "upper":      upper,
        "fit_object": model,
    }


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def fit_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Fit a Multi-Layer Perceptron with bootstrap prediction intervals."""
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        alpha=0.01,
        learning_rate_init=0.005,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=random_state,
        tol=1e-5,
    )
    model.fit(X_train, y_train)
    lower, upper = _bootstrap_ci(
        MLPRegressor,
        {"hidden_layer_sizes": (64, 32), "max_iter": 500,
         "random_state": 0, "tol": 1e-4},
        X_train, y_train, X_test, n_boot=n_boot,
    )
    return {
        "train":      model.predict(X_train),
        "test":       model.predict(X_test),
        "lower":      lower,
        "upper":      upper,
        "fit_object": model,
    }
