"""Stage 2 — Data cleaning.

Fills missing values with linear interpolation plus forward/backward fill.
"""

from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values and return a clean copy.

    Strategy: linear interpolation (both directions) then ffill/bfill for
    any remaining NaNs at series boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Raw monthly DataFrame from :func:`dna_forecast.ingest.load_raw`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with no missing values.
    """
    return (
        df.copy()
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )


def summarise_target(df: pd.DataFrame, col: str = "dna_pct") -> None:
    """Print descriptive statistics for the target variable."""
    s = df[col]
    print(f"  Target range  : {s.min():.2f}% - {s.max():.2f}%")
    print(f"  Mean / Std    : {s.mean():.3f}% / {s.std():.3f}%")
    print(f"  Skewness      : {s.skew():.3f}")
    print(f"  Missing after : {s.isna().sum()}")
