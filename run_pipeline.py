#!/usr/bin/env python
"""
NHS GP Appointment DNA Rate Forecasting — Pipeline Entry Point
==============================================================

Usage
-----
    python run_pipeline.py
    python run_pipeline.py --data-dir /path/to/data --output-dir results --test-months 6

All outputs (metrics CSV, cross-validation CSV) are written to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from dna_forecast import config
from dna_forecast.ingest import load_raw
from dna_forecast.clean import clean, summarise_target
from dna_forecast.features import engineer, make_ml_matrix, ML_AUX_COLS, select_features
from dna_forecast.models import (
    fit_sarima, fit_prophet, fit_random_forest, fit_xgboost, fit_mlp,
)
from dna_forecast.evaluate import build_metrics_table, cross_validate
from dna_forecast import visualise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NHS GP DNA Rate Forecasting Pipeline")
    p.add_argument("--data-dir",    type=Path, default=config.DATA_DIR,
                   help="Folder containing the three NHS CSV tables")
    p.add_argument("--output-dir",  type=Path, default=config.OUTPUT_DIR,
                   help="Folder to write results (created if absent)")
    p.add_argument("--test-months", type=int,  default=config.TEST_MONTHS,
                   help="Number of months held out for evaluation (default 6)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 65
    print(sep)
    print("  NHS GP DNA Rate Forecasting  |  Modular Pipeline")
    print(sep)

    # ------------------------------------------------------------------
    # Stage 1: Ingest
    # ------------------------------------------------------------------
    print("\n[1/7] Loading data ...")
    df_raw = load_raw(
        args.data_dir / config.PATH_1A.name,
        args.data_dir / config.PATH_1B.name,
        args.data_dir / config.PATH_1C.name,
    )
    print(f"      Shape: {df_raw.shape}  |  "
          f"{df_raw.index[0].strftime('%b %Y')} - {df_raw.index[-1].strftime('%b %Y')}")

    # ------------------------------------------------------------------
    # Stage 2: Clean
    # ------------------------------------------------------------------
    print("\n[2/7] Cleaning ...")
    df_clean = clean(df_raw)
    summarise_target(df_clean)

    # ------------------------------------------------------------------
    # Stage 3: EDA
    # ------------------------------------------------------------------
    print("\n[3/7] Exploratory data analysis ...")
    visualise.plot_eda(df_clean)
    visualise.plot_decomposition(df_clean["dna_pct"])

    # ------------------------------------------------------------------
    # Stage 4: Feature engineering
    # ------------------------------------------------------------------
    print("\n[4/7] Feature engineering ...")
    df_feat = engineer(df_clean)
    print(f"      Total columns after engineering: {df_feat.shape[1]}")

    # ------------------------------------------------------------------
    # Stage 5: Feature selection
    # ------------------------------------------------------------------
    print("\n[5/7] Feature selection ...")
    selection = select_features(df_feat)
    print("      Top-5 Lasso :", selection["lasso"].head(5).index.tolist())
    print("      Top-5 RF    :", selection["rf_importance"].head(5).index.tolist())
    visualise.plot_feature_selection(selection)

    # ------------------------------------------------------------------
    # Train / test split
    # ------------------------------------------------------------------
    ts_full   = df_clean["dna_pct"].sort_index()
    split_idx = -args.test_months
    train_ts, test_ts = ts_full.iloc[:split_idx], ts_full.iloc[split_idx:]
    print(f"\n      Train : {train_ts.index[0].strftime('%b-%y')} -> "
          f"{train_ts.index[-1].strftime('%b-%y')}  ({len(train_ts)} months)")
    print(f"      Test  : {test_ts.index[0].strftime('%b-%y')} -> "
          f"{test_ts.index[-1].strftime('%b-%y')}  ({len(test_ts)} months)")

    # Build ML feature matrix
    df_clean_aug = df_clean.copy()
    df_clean_aug["remote_index"] = df_clean_aug["video_pct"] + df_clean_aug["phone_pct"]
    df_ml = make_ml_matrix(ts_full, df_clean_aug[ML_AUX_COLS])
    ML_FEATS  = [c for c in df_ml.columns if c != "target"]
    X_full    = df_ml[ML_FEATS].values
    y_full    = df_ml["target"].values
    ml_split  = len(df_ml) - args.test_months
    X_tr, X_te = X_full[:ml_split], X_full[ml_split:]
    y_tr, y_te = y_full[:ml_split], y_full[ml_split:]
    scaler     = StandardScaler()
    X_tr_sc    = scaler.fit_transform(X_tr)
    X_te_sc    = scaler.transform(X_te)
    test_idx_ml  = df_ml.index[ml_split:]
    train_idx_ml = df_ml.index[:ml_split]

    # ------------------------------------------------------------------
    # Stage 6: Train models
    # ------------------------------------------------------------------
    print("\n[6/7] Training models ...")
    results: dict = {}
    results["SARIMA"]  = fit_sarima(train_ts, test_ts)
    print("      SARIMA  done")
    results["Prophet"] = fit_prophet(train_ts, test_ts)
    print("      Prophet done")
    results["RF"]      = fit_random_forest(X_tr_sc, y_tr, X_te_sc)
    print("      RF      done")
    results["XGBoost"] = fit_xgboost(X_tr_sc, y_tr, X_te_sc)
    print("      XGBoost done")
    results["MLP"]     = fit_mlp(X_tr_sc, y_tr, X_te_sc)
    print("      MLP     done")

    # Align actuals and date indices per model
    test_actuals  = {m: (test_ts.values if m in ("SARIMA", "Prophet") else y_te)
                     for m in config.MODEL_ORDER}
    train_actuals = {m: (train_ts.values if m in ("SARIMA", "Prophet") else y_tr)
                     for m in config.MODEL_ORDER}
    test_dates    = {m: (test_ts.index if m in ("SARIMA", "Prophet") else test_idx_ml)
                     for m in config.MODEL_ORDER}
    train_dates   = {m: (train_ts.index if m in ("SARIMA", "Prophet") else train_idx_ml)
                     for m in config.MODEL_ORDER}

    # ------------------------------------------------------------------
    # Stage 7: Evaluate
    # ------------------------------------------------------------------
    print("\n[7/7] Evaluating ...")
    metrics_df   = build_metrics_table(results, test_actuals, train_actuals)
    rows_metrics = metrics_df.to_dict(orient="index")
    print(metrics_df.to_string())

    ml_models = {
        "RF":      results["RF"]["fit_object"],
        "XGBoost": results["XGBoost"]["fit_object"],
        "MLP":     results["MLP"]["fit_object"],
    }
    cv_df = cross_validate(ml_models, X_full, y_full)
    print("\nTime-series cross-validation:")
    print(cv_df.to_string())

    # ------------------------------------------------------------------
    # Visualise results
    # ------------------------------------------------------------------
    visualise.plot_forecasts(
        results, rows_metrics, test_dates, train_dates, test_actuals, train_actuals
    )
    visualise.plot_comparison(results, rows_metrics, ts_full, test_dates, test_actuals)
    visualise.plot_metrics_bar(rows_metrics)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    metrics_path = args.output_dir / "metrics.csv"
    cv_path      = args.output_dir / "cross_validation.csv"
    metrics_df.to_csv(metrics_path)
    cv_df.to_csv(cv_path)
    print(f"\nOutputs saved:")
    print(f"  {metrics_path}")
    print(f"  {cv_path}")
    print(f"\n{sep}")


if __name__ == "__main__":
    main()
