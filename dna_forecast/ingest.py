"""Stage 1 — Data ingestion.

Parses the three NHS England GP Appointment Publication Summary CSV tables
and returns a tidy monthly DataFrame indexed by period.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PATH_1A, PATH_1B, PATH_1C


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _parse_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="latin1", header=None)


def _extract_series(df: pd.DataFrame, row_idx: int,
                    col_start: int = 2, col_end: int = 32) -> list[float]:
    """Extract a numeric row from a raw NHS summary table."""
    cleaned = []
    for v in df.iloc[row_idx, col_start:col_end].tolist():
        if pd.isna(v) or str(v).strip() in ("", ".z", "—"):
            cleaned.append(np.nan)
        else:
            s = str(v).strip().replace(",", "").replace("%", "")
            try:
                cleaned.append(float(s))
            except ValueError:
                cleaned.append(np.nan)
    return cleaned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(
    path_1a: Path = PATH_1A,
    path_1b: Path = PATH_1B,
    path_1c: Path = PATH_1C,
) -> pd.DataFrame:
    """Load NHS GP appointment tables and return a tidy monthly DataFrame.

    Parameters
    ----------
    path_1a, path_1b, path_1c:
        Paths to the three NHS publication summary CSV files.

    Returns
    -------
    pd.DataFrame
        Monthly DataFrame indexed by ``pd.DatetimeIndex`` with columns:
        working_days, total_appts, dna_pct (target), attended_pct,
        f2f_pct, phone_pct, video_pct, same_day_pct, d28p_pct,
        gp_pct, nurse_pct, pcn_dna_pct, appts_per_day.
    """
    t1a = _parse_table(path_1a)
    t1b = _parse_table(path_1b)

    months_raw = t1a.iloc[9, 2:32].tolist()
    months = pd.to_datetime(
        [m for m in months_raw if pd.notna(m)], format="%b-%y"
    )
    n = len(months)

    df = pd.DataFrame(
        {
            "month":        months,
            "working_days": _extract_series(t1a, 18)[:n],
            "total_appts":  _extract_series(t1a, 20)[:n],
            "dna_pct":      _extract_series(t1a, 86)[:n],   # TARGET
            "attended_pct": _extract_series(t1a, 85)[:n],
            "f2f_pct":      _extract_series(t1a, 95)[:n],
            "phone_pct":    _extract_series(t1a, 97)[:n],
            "video_pct":    _extract_series(t1a, 98)[:n],
            "same_day_pct": _extract_series(t1a, 101)[:n],
            "d28p_pct":     _extract_series(t1a, 107)[:n],
            "gp_pct":       _extract_series(t1a, 89)[:n],
            "nurse_pct":    _extract_series(t1a, 90)[:n],
            "pcn_dna_pct":  _extract_series(t1b, 85)[:n],
        }
    ).set_index("month").sort_index()

    df["appts_per_day"] = df["total_appts"] / df["working_days"].replace(0, np.nan)
    return df
