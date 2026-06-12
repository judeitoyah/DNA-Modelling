"""Visualisation functions for every pipeline stage."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from .config import PALETTE, MODEL_ORDER, MODEL_COLORS


# ---------------------------------------------------------------------------
# Stage 3 — EDA
# ---------------------------------------------------------------------------

def plot_eda(df_clean) -> None:
    """Four-panel EDA overview plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))

    # Panel 1: DNA time series GP vs PCN
    ax = axes[0, 0]
    ax.fill_between(df_clean.index, df_clean.dna_pct, alpha=0.15, color=PALETTE["SARIMA"])
    ax.plot(df_clean.index, df_clean.dna_pct,
            color=PALETTE["SARIMA"], lw=2.2, marker="o", ms=4, label="GP DNA%")
    ax.plot(df_clean.index, df_clean.pcn_dna_pct,
            color=PALETTE["Prophet"], lw=2, ls="--", marker="s", ms=3.5, label="PCN DNA%")
    oct_mask = df_clean.index.month == 10
    ax.scatter(df_clean.index[oct_mask], df_clean.dna_pct[oct_mask],
               color="#E05C2E", s=90, zorder=6, label="October peak")
    ax.axhline(df_clean.dna_pct.mean(), color=PALETTE["SARIMA"], ls=":", lw=1.2, alpha=0.6)
    ax.set_title("GP vs PCN DNA rate (%) — Oct 2023 to Mar 2026", fontweight="bold")
    ax.legend(fontsize=9)

    # Panel 2: Appointment mode mix
    ax = axes[0, 1]
    ax.stackplot(
        df_clean.index,
        df_clean.f2f_pct, df_clean.phone_pct, df_clean.video_pct,
        labels=["Face-to-Face", "Telephone", "Video/Online"],
        colors=["#1B4F8A", "#2E8B57", "#C9840A"], alpha=0.75,
    )
    ax.set_title("Appointment mode mix (%) — stacked", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 105)

    # Panel 3: Seasonal boxplot
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    groups = [
        df_clean.dna_pct[df_clean.index.month == m].values
        for m in range(1, 13)
        if len(df_clean.dna_pct[df_clean.index.month == m]) > 0
    ]
    labels_present = [
        month_names[m - 1] for m in range(1, 13)
        if len(df_clean.dna_pct[df_clean.index.month == m]) > 0
    ]
    ax = axes[1, 0]
    bp = ax.boxplot(groups, labels=labels_present, patch_artist=True,
                    medianprops=dict(color="white", lw=2))
    for patch, lbl in zip(bp["boxes"], labels_present):
        patch.set_facecolor("#E05C2E" if lbl == "Oct" else PALETTE["SARIMA"])
        patch.set_alpha(0.75)
    ax.set_title("DNA rate seasonal distribution", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("DNA%")

    # Panel 4: Lead time vs DNA
    ax = axes[1, 1]
    ax.fill_between(df_clean.index, df_clean.same_day_pct, alpha=0.2, color=PALETTE["RF"])
    ax.plot(df_clean.index, df_clean.same_day_pct,
            color=PALETTE["RF"], lw=2, label="Same-day %")
    ax.plot(df_clean.index, df_clean.d28p_pct,
            color=PALETTE["Prophet"], lw=2, ls="--", label=">28 days %")
    ax_r = ax.twinx()
    ax_r.plot(df_clean.index, df_clean.dna_pct,
              color=PALETTE["SARIMA"], lw=1.5, ls=":", alpha=0.8, label="DNA%")
    ax_r.set_ylabel("DNA%", color=PALETTE["SARIMA"])
    ax.set_title("Booking lead time vs DNA rate", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("EDA Overview — NHS GP Appointments DNA Rate Study",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage 3 — Decomposition & stationarity
# ---------------------------------------------------------------------------

def plot_decomposition(ts) -> None:
    """Seasonal decomposition + ADF stationarity test."""
    adf = adfuller(ts)
    print(f"  ADF Statistic : {adf[0]:.4f}")
    print(f"  p-value       : {adf[1]:.4f}")
    print(f"  Result        : {'Stationary' if adf[1] < 0.05 else 'Non-stationary'}")

    decomp = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    axes[0].plot(ts.index, ts.values, color=PALETTE["SARIMA"], lw=2, marker="o", ms=3)
    axes[0].set_title("Original series", fontweight="bold")
    axes[1].plot(ts.index, decomp.trend, color=PALETTE["RF"], lw=2.5)
    axes[1].set_title("Trend", fontweight="bold")
    axes[2].bar(ts.index, decomp.seasonal, color=PALETTE["Prophet"], alpha=0.8, width=20)
    axes[2].axhline(0, color="gray", lw=0.8, ls="--")
    axes[2].set_title("Seasonal (period=12)", fontweight="bold")
    axes[3].scatter(ts.index, decomp.resid, color=PALETTE["XGBoost"], s=50, alpha=0.85)
    axes[3].axhline(0, color="gray", lw=0.8, ls="--")
    axes[3].set_title("Residual", fontweight="bold")
    plt.suptitle("Seasonal Decomposition — GP DNA Rate", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage 5 — Feature selection
# ---------------------------------------------------------------------------

def plot_feature_selection(selection: dict) -> None:
    """Three-panel feature importance plot (correlation, Lasso, RF)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    specs = [
        (selection["correlation"].sort_values(),
         "Correlation with DNA%",  PALETTE["SARIMA"]),
        (selection["lasso"].sort_values().tail(15),
         "LassoCV |coefficient|",   PALETTE["XGBoost"]),
        (selection["rf_importance"].sort_values().tail(15),
         "RF feature importance",   PALETTE["RF"]),
    ]
    for ax, (vals, title, color) in zip(axes, specs):
        if "Correlation" in title:
            bar_c = [PALETTE["SARIMA"] if v >= 0 else PALETTE["Prophet"]
                     for v in vals.values]
            ax.axvline(0, color="gray", lw=0.8)
        else:
            bar_c = [color] * len(vals)
        ax.barh(vals.index, vals.values, color=bar_c,
                alpha=0.82, edgecolor="white", height=0.7)
        ax.set_title(title, fontweight="bold", fontsize=10)

    plt.suptitle("Feature Selection", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Stage 8 — Results
# ---------------------------------------------------------------------------

def plot_forecasts(results, rows_metrics, test_dates, train_dates,
                   test_actuals, train_actuals) -> None:
    """Individual forecast plot for each model (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (mname, color) in enumerate(zip(MODEL_ORDER, MODEL_COLORS)):
        ax = axes[i]
        ax.plot(train_dates[mname], train_actuals[mname],
                color="#aaa", lw=1.5, alpha=0.7, label="Train actual")
        ax.plot(train_dates[mname], results[mname]["train"],
                color=color, lw=1.8, ls="--", alpha=0.7, label="Train fit")
        ax.plot(test_dates[mname], test_actuals[mname],
                color="#111", lw=2.2, marker="o", ms=5, label="Test actual")
        ax.plot(test_dates[mname], results[mname]["test"],
                color=color, lw=2.2, marker="s", ms=5, label="Forecast")
        ax.fill_between(test_dates[mname],
                        results[mname]["lower"], results[mname]["upper"],
                        color=color, alpha=0.18)
        ax.axvline(train_dates[mname][-1], color="gray", lw=1, ls=":")
        r2   = rows_metrics[mname]["R2"]
        rmse = rows_metrics[mname]["RMSE"]
        ax.set_title(f"{mname}  |  RMSE={rmse:.4f}  R2={r2:.3f}",
                     fontweight="bold", fontsize=10, color=color)
        ax.set_ylabel("DNA%")
        ax.legend(fontsize=7.5)
        ax.set_ylim(3.5, 9)

    axes[5].axis("off")
    plt.suptitle("5 Model Forecasts vs Actual — GP DNA Rate",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_comparison(results, rows_metrics, ts_full, test_dates, test_actuals) -> None:
    """Head-to-head overlay and actual-vs-predicted scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.plot(ts_full.index, ts_full.values,
            color="#111", lw=2.5, marker="o", ms=5, label="Actual DNA%", zorder=10)
    for mname, color in zip(MODEL_ORDER, MODEL_COLORS):
        ax.plot(test_dates[mname], results[mname]["test"],
                color=color, lw=2.2, marker="D", ms=5,
                label=f"{mname} (RMSE={rows_metrics[mname]['RMSE']:.4f})")
    ax.axvline(list(test_dates.values())[0][0], color="gray", lw=1.5, ls="--", alpha=0.7)
    ax.set_title("All Models — Test-Period Forecast", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1]
    for mname, color in zip(MODEL_ORDER, MODEL_COLORS):
        ax.scatter(test_actuals[mname], results[mname]["test"],
                   color=color, s=90, alpha=0.9, edgecolors="white", label=mname)
    mn, mx = 4.0, 5.5
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, alpha=0.5, label="Perfect")
    ax.set_xlabel("Actual DNA%")
    ax.set_ylabel("Predicted DNA%")
    ax.set_title("Actual vs Predicted (test period)", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Model Comparison", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_metrics_bar(rows_metrics) -> None:
    """Bar chart of RMSE with MAPE% overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(MODEL_ORDER))
    rmse_vals = [rows_metrics[m]["RMSE"] for m in MODEL_ORDER]
    bars = ax.bar(x_pos, rmse_vals, color=MODEL_COLORS,
                  alpha=0.85, edgecolor="white", width=0.6)
    ax2 = ax.twinx()
    ax2.plot(x_pos, [rows_metrics[m]["MAPE%"] for m in MODEL_ORDER],
             "kD--", ms=8, lw=2, label="MAPE%")
    ax2.set_ylabel("MAPE (%)")
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylabel("RMSE (pp)")
    ax.set_title("RMSE & MAPE% by Model", fontweight="bold", fontsize=12)
    plt.tight_layout()
    plt.show()
