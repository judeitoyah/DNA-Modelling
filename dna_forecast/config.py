"""Central configuration — paths, constants, plot style."""

from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data paths  (override via CLI args or environment variables)
# ---------------------------------------------------------------------------
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")

PATH_1A = DATA_DIR / "GP_Appointment_Publication_Summary_March_2026(Table 1a).csv"
PATH_1B = DATA_DIR / "GP_Appointment_Publication_Summary_March_2026(Table 1b).csv"
PATH_1C = DATA_DIR / "GP_Appointment_Publication_Summary_March_2026(Table 1c).csv"

# ---------------------------------------------------------------------------
# Modelling
# ---------------------------------------------------------------------------
TEST_MONTHS = 6       # months held out for evaluation
RANDOM_SEED = 42
N_BOOTSTRAP = 200     # bootstrap iterations for ML confidence intervals

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
PALETTE = {
    "SARIMA":  "#1B4F8A",
    "Prophet": "#E05C2E",
    "RF":      "#2E8B57",
    "XGBoost": "#8B4EA6",
    "MLP":     "#C9840A",
    "actual":  "#1A1A2E",
}
MODEL_ORDER  = ["SARIMA", "Prophet", "RF", "XGBoost", "MLP"]
MODEL_COLORS = [PALETTE[m] for m in MODEL_ORDER]

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "figure.dpi":         110,
})
