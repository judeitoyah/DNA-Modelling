"""Stage 7 — Evaluation metrics and time-series cross-validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(100 * np.mean(np.abs((y_true - y_pred) / y_true)))


def score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    y_train_pred: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE, R2 for a single model.

    Parameters
    ----------
    y_true, y_pred : array-like
        Test actuals and forecasts.
    y_train, y_train_pred : array-like, optional
        When supplied, train RMSE is included.

    Returns
    -------
    dict[str, float]
    """
    metrics: dict[str, float] = {
        "RMSE":  round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE":   round(float(mean_absolute_error(y_true, y_pred)), 4),
        "MAPE%": round(_mape(y_true, y_pred), 3),
        "R2":    round(float(r2_score(y_true, y_pred)), 4),
    }
    if y_train is not None and y_train_pred is not None:
        metrics["Train_RMSE"] = round(
            float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 4
        )
    return metrics


def build_metrics_table(
    results: dict,
    test_actuals: dict[str, np.ndarray],
    train_actuals: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compile evaluation metrics for all models into a ranked DataFrame.

    Parameters
    ----------
    results : dict
        Model name -> result dict from ``dna_forecast.models``.
    test_actuals, train_actuals : dict
        Model name -> ground-truth array.

    Returns
    -------
    pd.DataFrame
        One row per model, sorted by RMSE ascending.
    """
    rows = {
        name: score(
            test_actuals[name], res["test"],
            train_actuals[name], res["train"],
        )
        for name, res in results.items()
    }
    return pd.DataFrame(rows).T.sort_values("RMSE")


def cross_validate(
    models: dict,
    X_full: np.ndarray,
    y_full: np.ndarray,
    n_splits: int = 5,
    test_size: int = 2,
) -> pd.DataFrame:
    """Time-series cross-validation for ML models.

    Parameters
    ----------
    models : dict
        Model name -> fitted sklearn estimator.
    X_full, y_full : np.ndarray
        Full feature matrix and target vector.
    n_splits : int
        Number of CV folds.
    test_size : int
        Months per test fold.

    Returns
    -------
    pd.DataFrame
        Mean and std RMSE per model.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    cv_results = {}
    for name, model in models.items():
        fold_rmses = []
        for tr_idx, te_idx in tscv.split(X_full):
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_full[tr_idx])
            X_te = sc.transform(X_full[te_idx])
            model.fit(X_tr, y_full[tr_idx])
            fold_rmses.append(
                float(np.sqrt(mean_squared_error(y_full[te_idx], model.predict(X_te))))
            )
        cv_results[name] = {
            "mean_RMSE": round(np.mean(fold_rmses), 4),
            "std_RMSE":  round(np.std(fold_rmses), 4),
        }
    return pd.DataFrame(cv_results).T
