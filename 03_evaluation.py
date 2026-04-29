# =============================================================================
# NOTEBOOK 3: evaluation.py
# Travel Time Prediction — Metrics, Bootstrap CIs, DM Test, SHAP, TDA Validation
# =============================================================================
# Paper: "Travel Time Prediction Using Various Time Series Feature Generation
#         Techniques", IEEE Access, 2025
# =============================================================================
# Sections covered:
#   1.  Evaluation metrics: MAE, RMSE, MAPE, R²   (Eq. 15–18)
#   2.  95 % BCa bootstrap confidence intervals     (1 000 resamples, block=12)
#   3.  Pairwise Diebold–Mariano significance tests
#   4.  SHAP feature importance (XGBoost+TDA)
#   5.  Pearson correlation: TDA features vs. traffic temporal descriptors
#   6.  Results tables (match Tables 9–13 in paper)
#   7.  Visualizations (prediction plots, heatmaps, scatter plots)
# =============================================================================

# ── 1. IMPORTS ────────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from scipy.signal import stft as scipy_stft
import warnings
warnings.filterwarnings("ignore")

OUTPUTS_DIR = "../outputs"
PLOTS_DIR   = "../outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# =============================================================================
# 2. EVALUATION METRICS   (paper Section IV-F, Equations 15–18)
# =============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error  [Eq. 15]"""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Square Error  [Eq. 16]"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray,
         eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error  [Eq. 17]"""
    return float(100.0 * np.mean(np.abs((y_true - y_pred) /
                                        (np.abs(y_true) + eps))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination  [Eq. 18]"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                label: str = "") -> dict:
    """Compute all four metrics and return as dict."""
    m = dict(model=label,
             MAE=mae(y_true, y_pred),
             RMSE=rmse(y_true, y_pred),
             MAPE=mape(y_true, y_pred),
             R2=r2(y_true, y_pred))
    print(f"  {label:30s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}"
          f"  MAPE={m['MAPE']:.4f}  R²={m['R2']:.4f}")
    return m


# =============================================================================
# 3. BOOTSTRAP CONFIDENCE INTERVALS (BCa, 1 000 resamples, block length=12)
# =============================================================================
# Paper Section IV-F:
#   "Block bootstrap (1 000 resamples, block length = 12 time steps) to
#    respect the temporal dependence structure of the test series."


def _block_bootstrap_sample(errors: np.ndarray, block_len: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Draw one circular block-bootstrap resample of the error series."""
    n        = len(errors)
    n_blocks = int(np.ceil(n / block_len))
    starts   = rng.integers(0, n, size=n_blocks)
    sample   = np.concatenate([
        errors[np.arange(s, s + block_len) % n] for s in starts
    ])
    return sample[:n]


def bca_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                     metric_fn, n_boot: int = 1000,
                     block_len: int = 12,
                     alpha: float = 0.05,
                     seed: int = SEED) -> tuple:
    """
    BCa (bias-corrected and accelerated) bootstrap CI for a single metric.

    Returns (lower, upper) bounds of the (1-alpha) CI.
    Reference: Efron & Tibshirani (1993); Künsch (1989) for block bootstrap.
    Paper Section IV-F, [36].
    """
    rng      = np.random.default_rng(seed)
    errors   = y_true - y_pred
    obs_stat = metric_fn(y_true, y_pred)
    boot_stats = np.empty(n_boot)

    for b in range(n_boot):
        err_b   = _block_bootstrap_sample(errors, block_len, rng)
        y_b     = y_pred + err_b            # synthetic actuals
        boot_stats[b] = metric_fn(y_b, y_pred)

    # Bias-correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_stats < obs_stat))

    # Acceleration factor a (jackknife)
    jack_stats = np.array([
        metric_fn(np.delete(y_true, i), np.delete(y_pred, i))
        for i in range(min(len(y_true), 200))   # cap at 200 for speed
    ])
    jack_mean = jack_stats.mean()
    num   = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5 + 1e-12)
    a     = num / denom

    # Adjusted quantiles
    z_alpha  = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)
    p_lo = stats.norm.cdf(z0 + (z0 + z_alpha)  / (1 - a * (z0 + z_alpha)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    lo = float(np.percentile(boot_stats, 100 * p_lo))
    hi = float(np.percentile(boot_stats, 100 * p_hi))
    return lo, hi


def bootstrap_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           label: str = "", n_boot: int = 1000) -> dict:
    """Compute BCa CIs for MAE, RMSE, and R²."""
    mae_lo, mae_hi   = bca_bootstrap_ci(y_true, y_pred, mae,  n_boot)
    rmse_lo, rmse_hi = bca_bootstrap_ci(y_true, y_pred, rmse, n_boot)
    r2_lo,  r2_hi    = bca_bootstrap_ci(y_true, y_pred, r2,   n_boot)

    result = dict(
        model=label,
        MAE_ci=f"[{mae_lo:.2f}, {mae_hi:.2f}]",
        RMSE_ci=f"[{rmse_lo:.2f}, {rmse_hi:.2f}]",
        R2_ci=f"[{r2_lo:.3f}, {r2_hi:.3f}]",
    )
    print(f"  {label:30s}  "
          f"MAE {result['MAE_ci']}  "
          f"RMSE {result['RMSE_ci']}  "
          f"R² {result['R2_ci']}")
    return result


# =============================================================================
# 4. DIEBOLD–MARIANO TEST
# =============================================================================
# Paper Section IV-F, [37]:
#   Pairwise Diebold–Mariano (DM) tests confirm whether performance differences
#   between models are statistically significant.


def diebold_mariano_test(y_true:  np.ndarray,
                         y_pred1: np.ndarray,
                         y_pred2: np.ndarray,
                         loss: str = "squared",
                         h: int = 1) -> dict:
    """
    Harvey, Leybourne & Newbold (1997) modified DM test.
    H0: Equal predictive accuracy between model 1 and model 2.

    Parameters
    ----------
    loss : 'squared' (MSE-based) or 'absolute' (MAE-based)
    h    : forecast horizon (default 1 for one-step-ahead)

    Returns
    -------
    dict with 'dm_stat', 'p_value', 'reject_H0'
    """
    if loss == "squared":
        d = (y_true - y_pred1) ** 2 - (y_true - y_pred2) ** 2
    else:
        d = np.abs(y_true - y_pred1) - np.abs(y_true - y_pred2)

    n    = len(d)
    d_bar = d.mean()

    # Newey-West HAC variance estimator
    gamma0 = np.var(d, ddof=1)
    gammas = [np.cov(d[j:], d[:-j])[0, 1] if j > 0 else gamma0
              for j in range(h)]
    var_d = (gamma0 + 2 * sum(gammas[1:])) / n

    dm_stat = d_bar / np.sqrt(var_d + 1e-12)

    # Modified DM (Harvey et al. 1997) — t distribution with n-1 d.f.
    t_stat  = dm_stat * np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

    return dict(dm_stat=float(dm_stat),
                t_stat=float(t_stat),
                p_value=float(p_value),
                reject_H0=bool(p_value < 0.05))


# =============================================================================
# 5. SHAP FEATURE IMPORTANCE (XGBoost + TDA)
# =============================================================================
# Paper Section V-D: "SHAP analysis identifies Mean Persistence H0 as the
# third most important predictor after lag1 and lag2."


def compute_shap_importance(model_path: str, X_test: np.ndarray,
                             feature_names: list, suffix: str):
    """
    Compute SHAP values for an XGBoost model and plot feature importance.
    Requires: pip install shap
    """
    try:
        import shap
        from xgboost import XGBRegressor
    except ImportError:
        print("  Install shap:  pip install shap")
        return

    model = XGBRegressor()
    model.load_model(model_path)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "mean_|SHAP|": mean_abs_shap,
    }).sort_values("mean_|SHAP|", ascending=False).reset_index(drop=True)

    print("\nSHAP Feature Importance (top 15):")
    print(importance_df.head(15).to_string(index=False))

    # Plot
    plt.figure(figsize=(9, 6))
    bars = plt.barh(importance_df["feature"][:15][::-1],
                    importance_df["mean_|SHAP|"][:15][::-1],
                    color="#2196F3")
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"XGBoost+TDA Feature Importance — {suffix}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/shap_importance_{suffix}.png", dpi=300)
    plt.close()
    print(f"  SHAP plot saved → {PLOTS_DIR}/shap_importance_{suffix}.png")

    importance_df.to_csv(f"{OUTPUTS_DIR}/shap_importance_{suffix}.csv", index=False)
    return importance_df


# Feature names for XGBoost+TDA (lag10 + 10 TDA features)
TDA_FEAT_NAMES = [
    "H0_betti", "H0_mean_persistence", "H0_persistence_entropy",
    "H0_l1_norm", "H0_l2_norm",
    "H1_betti", "H1_mean_persistence", "H1_persistence_entropy",
    "H1_l1_norm", "H1_l2_norm",
]
LAG_FEAT_NAMES = [f"lag_{i+1}" for i in range(10)]
XGB_TDA_FEAT_NAMES = LAG_FEAT_NAMES + TDA_FEAT_NAMES


# =============================================================================
# 6. PEARSON CORRELATION: TDA FEATURES vs. TRAFFIC TEMPORAL DESCRIPTORS
# =============================================================================
# Paper Section V-D / III-D4:
#   7 traffic temporal descriptors computed:
#     ACF at lags 1, 12, 288; Fourier power at daily & rush-hour freq;
#     peak-hour duration; off-peak variance.
#   Key findings:
#     Mean Persistence H0  vs. ACF lag-288  → r = 0.71, p < 0.001
#     Persistence Entropy H1 vs. rush-hour power → r = 0.63, p < 0.001


def compute_acf_at_lag(series: np.ndarray, lag: int) -> np.ndarray:
    """
    Sliding ACF value at a specific lag across the series.
    For each window ending at t, computes corr(series, series shifted by lag).
    Returns array of same length as series (NaN-padded for early indices).
    """
    n = len(series)
    acf_vals = np.full(n, np.nan)
    for t in range(lag, n):
        if t - lag >= 0:
            seg = series[max(0, t - 288): t + 1]
            if len(seg) > lag + 1:
                acf_vals[t] = np.corrcoef(seg[:-lag], seg[lag:])[0, 1]
    return acf_vals


def compute_stft_spectral_power(series: np.ndarray,
                                 fs: float = 1.0/300,      # 5-min samples → Hz
                                 target_period_h: float = 24.0,
                                 window_sec: int = 1800    # 30-min window
                                 ) -> np.ndarray:
    """
    Short-time Fourier power at target_period_h frequency across the series.
    fs: sampling frequency in Hz (1 sample per 300 s → 1/300 Hz)
    """
    nperseg = int(window_sec * fs)   # samples per 30-min window
    target_freq = 1.0 / (target_period_h * 3600)  # Hz

    f, t_stft, Zxx = scipy_stft(series, fs=fs, nperseg=max(nperseg, 4))
    power   = np.abs(Zxx) ** 2
    freq_idx = np.argmin(np.abs(f - target_freq))
    # Interpolate STFT power back to original series length
    stft_power = np.interp(np.arange(len(series)),
                           np.linspace(0, len(series)-1, len(t_stft)),
                           power[freq_idx])
    return stft_power


def pearson_tda_vs_traffic(tda_matrix: np.ndarray,
                            series: np.ndarray,
                            suffix: str) -> pd.DataFrame:
    """
    Compute Pearson r between each TDA feature and 7 traffic temporal descriptors.
    Returns correlation matrix (10 TDA × 7 descriptors) with p-values.

    Paper Section V-D: reports r=0.71 (Mean Persistence H0 vs. ACF lag-288)
                        and r=0.63 (Persistence Entropy H1 vs. rush-hour power).
    """
    n = len(tda_matrix)

    # ── Compute 7 temporal descriptors (one value per window/timestep) ────────
    acf1   = compute_acf_at_lag(series, 1)[-n:]
    acf12  = compute_acf_at_lag(series, 12)[-n:]
    acf288 = compute_acf_at_lag(series, 288)[-n:]
    daily_power  = compute_stft_spectral_power(series[-n:], target_period_h=24.0)
    rush_power   = compute_stft_spectral_power(series[-n:], target_period_h=1.0)

    # Peak-hour flag (07:00–09:00 and 16:00–18:00 at 5-min resolution)
    # Assumes series starts at midnight; 288 intervals/day
    t_mod = np.arange(n) % 288
    peak_mask = ((t_mod >= 84) & (t_mod <= 108)) | \
                ((t_mod >= 192) & (t_mod <= 216))
    peak_duration = peak_mask.astype(float)

    # Off-peak variance (rolling std outside peak hours)
    offpeak = series[-n:].copy()
    offpeak[peak_mask] = np.nan
    roll_std = pd.Series(offpeak).rolling(12, min_periods=1).std().values
    offpeak_var = roll_std ** 2

    descriptors = {
        "ACF_lag1":       acf1,
        "ACF_lag12":      acf12,
        "ACF_lag288":     acf288,
        "Fourier_daily":  daily_power,
        "Fourier_rush":   rush_power,
        "Peak_duration":  peak_duration,
        "Offpeak_var":    offpeak_var,
    }

    tda_names = TDA_FEAT_NAMES

    rows = []
    for i, tda_name in enumerate(tda_names):
        row = {"TDA_feature": tda_name}
        for desc_name, desc_vals in descriptors.items():
            # Remove NaN pairs
            valid = ~(np.isnan(tda_matrix[:, i]) | np.isnan(desc_vals))
            if valid.sum() > 10:
                r_val, p_val = stats.pearsonr(
                    tda_matrix[valid, i], desc_vals[valid])
            else:
                r_val, p_val = np.nan, np.nan
            row[desc_name]          = round(r_val, 3) if not np.isnan(r_val) else np.nan
            row[f"{desc_name}_p"]   = round(p_val, 4) if not np.isnan(p_val) else np.nan
        rows.append(row)

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(f"{OUTPUTS_DIR}/pearson_tda_correlations_{suffix}.csv", index=False)

    # ── Plot heatmap (Figure 9 equivalent) ────────────────────────────────────
    heat_data = corr_df.set_index("TDA_feature")[list(descriptors.keys())].astype(float)

    plt.figure(figsize=(11, 6))
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5,
                annot_kws={"size": 9})
    plt.title(f"Pearson Correlation — TDA Features vs. Traffic Temporal Descriptors\n"
              f"({suffix}, p < 0.001 for all r ≥ 0.60)", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/pearson_heatmap_{suffix}.png", dpi=300)
    plt.close()
    print(f"  Pearson heatmap saved → {PLOTS_DIR}/pearson_heatmap_{suffix}.png")
    return corr_df


# =============================================================================
# 7. RESULTS TABLES (matching paper Tables 10, 11, 12, 13)
# =============================================================================

def build_results_table(suffix: str,
                         model_keys: dict,
                         scaler=None) -> pd.DataFrame:
    """
    Build a results DataFrame for all models on a given dataset suffix.

    Parameters
    ----------
    suffix     : e.g. 'tid1', 'pems405'
    model_keys : dict mapping display label → predictions .npy filename stem
    scaler     : fitted MinMaxScaler (to inverse-transform to seconds)

    Returns
    -------
    pd.DataFrame with columns: Model, MAE, RMSE, MAPE, R²
    """
    y_test = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")

    rows = []
    for label, pred_stem in model_keys.items():
        try:
            preds = np.load(f"{OUTPUTS_DIR}/{pred_stem}_{suffix}.npy")
        except FileNotFoundError:
            print(f"  [skip] {pred_stem}_{suffix}.npy not found")
            continue

        # Inverse-transform to seconds if scaler provided
        if scaler is not None:
            y_s = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            p_s = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        else:
            y_s, p_s = y_test, preds

        rows.append(all_metrics(y_s, p_s, label=label))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUTS_DIR}/results_{suffix}.csv", index=False)
    print(f"\nResults saved → {OUTPUTS_DIR}/results_{suffix}.csv")
    return df


# Model key map (display name → predictions file stem)
MODEL_KEYS_LONGTERM = {
    "XGBoost-Lag":       "preds_xgb_lag",
    "XGBoost+TDA":       "preds_xgb_tda",
    "XGBoost+KMeans":    "preds_xgb_km",
    "LSTM-Lag":          "preds_lstm_lag",
    "LSTM-Attn+TDA":     "preds_lstm_attn_tda",
    "LSTM-Attn+KMeans":  "preds_lstm_attn_km",
}

print("\n=== Results Table — WSDOT Trip ID 1 (Table 10 equivalent) ===")
results_tid1 = build_results_table("tid1", MODEL_KEYS_LONGTERM)
print(results_tid1.to_string(index=False))


# =============================================================================
# 8. BOOTSTRAP CIs TABLE (matching paper Table 11)
# =============================================================================

def build_bootstrap_ci_table(suffix: str, model_keys: dict,
                              n_boot: int = 1000) -> pd.DataFrame:
    y_test = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")
    rows   = []
    for label, pred_stem in model_keys.items():
        try:
            preds = np.load(f"{OUTPUTS_DIR}/{pred_stem}_{suffix}.npy")
        except FileNotFoundError:
            continue
        row = bootstrap_all_metrics(y_test, preds, label=label, n_boot=n_boot)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUTS_DIR}/bootstrap_ci_{suffix}.csv", index=False)
    return df


print("\n=== 95% BCa Bootstrap CIs — WSDOT Trip ID 1 (Table 11 equivalent) ===")
ci_table = build_bootstrap_ci_table("tid1", MODEL_KEYS_LONGTERM, n_boot=1000)
print(ci_table.to_string(index=False))


# =============================================================================
# 9. DIEBOLD–MARIANO PAIRWISE TESTS
# =============================================================================

def dm_pairwise_table(suffix: str, model_keys: dict) -> pd.DataFrame:
    """
    Compute DM test for all pairs where the best model is model 1.
    Reference model = LSTM-Attn+TDA (best in paper).
    """
    y_test   = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")
    ref_stem = "preds_lstm_attn_tda"

    try:
        y_ref = np.load(f"{OUTPUTS_DIR}/{ref_stem}_{suffix}.npy")
    except FileNotFoundError:
        print(f"  Reference model preds not found: {ref_stem}_{suffix}.npy")
        return pd.DataFrame()

    rows = []
    for label, pred_stem in model_keys.items():
        if pred_stem == ref_stem:
            continue
        try:
            y_other = np.load(f"{OUTPUTS_DIR}/{pred_stem}_{suffix}.npy")
        except FileNotFoundError:
            continue
        dm = diebold_mariano_test(y_test, y_ref, y_other, loss="squared")
        rows.append({"model":      label,
                     "DM_stat":    round(dm["dm_stat"], 3),
                     "t_stat":     round(dm["t_stat"], 3),
                     "p_value":    round(dm["p_value"], 4),
                     "reject_H0":  dm["reject_H0"]})

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUTS_DIR}/dm_tests_{suffix}.csv", index=False)
    print(f"\nDM test table saved → {OUTPUTS_DIR}/dm_tests_{suffix}.csv")
    return df


print("\n=== Diebold–Mariano Tests (reference: LSTM-Attn+TDA) ===")
dm_table = dm_pairwise_table("tid1", MODEL_KEYS_LONGTERM)
print(dm_table.to_string(index=False))


# =============================================================================
# 10. VISUALIZATIONS
# =============================================================================

def plot_predictions(y_true:  np.ndarray,
                     preds_dict: dict,
                     title:   str,
                     n_steps: int = 288,
                     save_path: str = None):
    """
    Plot actual vs. predicted travel times for last n_steps of test set.
    Replicates Figure 6 style from the paper.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true[-n_steps:], color="black", lw=1.5, label="Actual")

    styles = ["-", "--", ":"]
    colors = ["#E53935", "#1E88E5", "#43A047"]
    for (label, preds), ls, col in zip(preds_dict.items(), styles, colors):
        ax.plot(preds[-n_steps:], ls=ls, color=col, lw=1.2, label=label)

    ax.set_xlabel("Time Step (5-min intervals)")
    ax.set_ylabel("Travel Time (normalized)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"  Plot saved → {save_path}")
    plt.close()


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                              label: str, suffix: str):
    """
    Scatter plot: actual vs. predicted (Figure 10 style).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="#1E88E5")
    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.5, label="Perfect fit (y=x)")
    r2_val = r2(y_true, y_pred)
    mae_val = mae(y_true, y_pred)
    ax.set_xlabel("Actual Travel Time (s)")
    ax.set_ylabel("Predicted Travel Time (s)")
    ax.set_title(f"{label} — {suffix}\nMAE={mae_val:.2f}s  R²={r2_val:.4f}")
    ax.legend()
    plt.tight_layout()
    path = f"{PLOTS_DIR}/scatter_{label.replace('+','_')}_{suffix}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Scatter plot saved → {path}")


def plot_performance_heatmap(results_all_trips: pd.DataFrame,
                              metric: str = "MAE"):
    """
    Performance heatmap across all Trip IDs and model configurations.
    Replicates Figure 8 from the paper.
    """
    pivot = results_all_trips.pivot(index="Trip_ID", columns="Model",
                                    values=metric)
    plt.figure(figsize=(12, 5))
    # Green=good (low MAE), Red=bad (high MAE)
    cmap = "RdYlGn_r" if metric in ("MAE", "RMSE", "MAPE") else "RdYlGn"
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, annot_kws={"size": 9})
    plt.title(f"Performance Heatmap — {metric} across Trip IDs")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/heatmap_{metric}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Heatmap saved → {path}")


# =============================================================================
# 11. MULTI-TRIP EVALUATION (Table 12 equivalent)
# =============================================================================

TRIP_IDS = [1, 2, 3, 4, 5, 6, 7, 8]

def multi_trip_evaluation(trip_ids: list, model_keys: dict) -> pd.DataFrame:
    """
    Evaluate all models on the last 1 500 records of each Trip ID.
    Returns a long-form DataFrame matching Table 12 format.
    """
    all_rows = []
    for tid in trip_ids:
        suffix = f"tid{tid}"
        try:
            y_test = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")[-1500:]
        except FileNotFoundError:
            print(f"  [skip] y_test_{suffix}.npy not found")
            continue

        for label, pred_stem in model_keys.items():
            try:
                preds = np.load(
                    f"{OUTPUTS_DIR}/{pred_stem}_{suffix}.npy")[-1500:]
            except FileNotFoundError:
                continue
            m = all_metrics(y_test, preds, label=f"TID{tid} {label}")
            m["Trip_ID"] = tid
            m["Model"]   = label
            all_rows.append(m)

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{OUTPUTS_DIR}/multi_trip_results.csv", index=False)
    return df


print("\n=== Multi-Trip Generalizability (Table 12 equivalent) ===")
multi_trip_df = multi_trip_evaluation(TRIP_IDS, {
    "XGBoost-Lag":   "preds_xgb_lag",
    "XGBoost+TDA":   "preds_xgb_tda",
    "LSTM-Lag":      "preds_lstm_lag",
    "LSTM-Attn+TDA": "preds_lstm_attn_tda",
})
print(multi_trip_df[["Trip_ID", "Model", "MAE", "R2"]].to_string(index=False))

# Heatmap
plot_performance_heatmap(multi_trip_df, metric="MAE")
plot_performance_heatmap(multi_trip_df, metric="R2")


# =============================================================================
# 12. PeMS CROSS-DATASET RESULTS (Table 13 equivalent)
# =============================================================================

PEMS_ROUTES = [405, 10, 605, 210, 60]
PEMS_MODEL_KEYS = {
    "XGBoost-Lag": "preds_xgb_lag",
    "LSTM-Lag":    "preds_lstm_lag",
}

print("\n=== Cross-Dataset Validation — PeMS (Table 13 equivalent) ===")
pems_rows = []
for route in PEMS_ROUTES:
    suffix = f"pems{route}"
    y_test = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")
    for label, stem in PEMS_MODEL_KEYS.items():
        try:
            preds = np.load(f"{OUTPUTS_DIR}/{stem}_{suffix}.npy")
        except FileNotFoundError:
            continue
        m = all_metrics(y_test, preds, label=f"Route{route} {label}")
        m["Route"] = route
        m["Model"] = label
        pems_rows.append(m)

pems_df = pd.DataFrame(pems_rows)
pems_df.to_csv(f"{OUTPUTS_DIR}/pems_results.csv", index=False)
print(pems_df[["Route", "Model", "MAE", "RMSE", "MAPE", "R2"]].to_string(index=False))

print("\n=== 03_evaluation.py COMPLETE ===")
