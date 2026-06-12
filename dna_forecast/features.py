"""Stages 4 & 5 — Feature engineering and selection.

engineer()       — adds lag, rolling, momentum and temporal features
make_ml_matrix() — builds the ML feature matrix used by RF / XGBoost / MLP
select_features()— runs LassoCV, RF importance and Pearson correlation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED

# Features passed to the selection step
FEATURE_COLS: list[str] = [
    "f2f_pct", "video_pct", "phone_pct", "same_day_pct", "d28p_pct",
    "gp_pct", "nurse_pct", "working_days", "remote_index", "appts_norm",
    "month_sin", "month_cos", "is_october",
    "dna_lag_1", "dna_lag_2", "dna_lag_3",
    "dna_roll3_mean", "dna_roll6_mean", "dna_ewm3",
    "dna_diff1", "video_mom", "gp_pcn_gap",
]

# Auxiliary columns used to build the ML matrix
ML_AUX_COLS: list[str] = [
    "video_pct", "f2f_pct", "phone_pct", "same_day_pct",
    "d28p_pct", "working_days", "gp_pct", "remote_index",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, momentum and temporal features to a clean DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned monthly DataFrame with at minimum the columns produced by
        :func:`dna_forecast.clean.clean`.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with engineered features.
    """
    d = df.copy()
    d["remote_index"] = d["video_pct"] + d["phone_pct"]

    for lag in (1, 2, 3, 6, 12):
        d[f"dna_lag_{lag}"] = d["dna_pct"].shift(lag)

    d["dna_roll3_mean"] = d["dna_pct"].shift(1).rolling(3).mean()
    d["dna_roll6_mean"] = d["dna_pct"].shift(1).rolling(6).mean()
    d["dna_roll3_std"]  = d["dna_pct"].shift(1).rolling(3).std()
    d["dna_ewm3"]       = d["dna_pct"].shift(1).ewm(span=3).mean()

    d["month_sin"]  = np.sin(2 * np.pi * d.index.month / 12)
    d["month_cos"]  = np.cos(2 * np.pi * d.index.month / 12)
    d["is_october"] = (d.index.month == 10).astype(int)

    d["video_mom"]  = d["video_pct"].diff()
    d["dna_diff1"]  = d["dna_pct"].diff(1)
    d["dna_diff12"] = d["dna_pct"].diff(12)
    d["gp_pcn_gap"] = d["pcn_dna_pct"] - d["dna_pct"]
    d["appts_norm"] = d["appts_per_day"] / d["appts_per_day"].max()

    return d


# ---------------------------------------------------------------------------
# ML feature matrix
# ---------------------------------------------------------------------------

def make_ml_matrix(
    series: pd.Series,
    extra_df: pd.DataFrame,
    n_lags: int = 3,
) -> pd.DataFrame:
    """Build a supervised feature matrix from a time series + auxiliary cols.

    Parameters
    ----------
    series : pd.Series
        Target time series (e.g. ``df_clean["dna_pct"]``).
    extra_df : pd.DataFrame
        Auxiliary columns to include alongside the lag features.
    n_lags : int
        Number of autoregressive lags to include (default 3).

    Returns
    -------
    pd.DataFrame
        Rows with complete data only; last column is ``"target"``.
    """
    d = extra_df.copy()
    for lag in range(1, n_lags + 1):
        d[f"lag_{lag}"] = series.shift(lag)
    d["roll3"]  = series.shift(1).rolling(3).mean()
    d["roll6"]  = series.shift(1).rolling(6).mean()
    d["ewm3"]   = series.shift(1).ewm(span=3).mean()
    d["is_oct"] = (d.index.month == 10).astype(int)
    d["m_sin"]  = np.sin(2 * np.pi * d.index.month / 12)
    d["m_cos"]  = np.cos(2 * np.pi * d.index.month / 12)
    d["target"] = series
    return d.dropna()


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    df_feat: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    random_state: int = RANDOM_SEED,
) -> dict[str, pd.Series]:
    """Run three complementary feature-selection methods.

    Parameters
    ----------
    df_feat : pd.DataFrame
        DataFrame containing ``feature_cols`` and ``"dna_pct"``.
    feature_cols : list[str]
        Feature columns to evaluate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``"lasso"``, ``"rf_importance"``, ``"correlation"`` —
    each a :class:`pd.Series` sorted by absolute importance descending.
    """
    df_model = df_feat[feature_cols + ["dna_pct"]].dropna()
    X = df_model[feature_cols].values
    y = df_model["dna_pct"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    lasso = LassoCV(cv=5, random_state=random_state, max_iter=5000)
    lasso.fit(X_sc, y)
    lasso_coefs = (
        pd.Series(np.abs(lasso.coef_), index=feature_cols)
        .sort_values(ascending=False)
    )

    rf_fs = RandomForestRegressor(
        n_estimators=300, random_state=random_state, max_features="sqrt"
    )
    rf_fs.fit(X_sc, y)
    rf_imp = (
        pd.Series(rf_fs.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )

    corr = (
        df_model[feature_cols + ["dna_pct"]]
        .corr()["dna_pct"]
        .drop("dna_pct")
        .sort_values(key=abs, ascending=False)
    )

    return {"lasso": lasso_coefs, "rf_importance": rf_imp, "correlation": corr}
