# =============================================================================
# NOTEBOOK 2: model_training.py
# Travel Time Prediction — ARIMA / ARIMAX / XGBoost / LSTM / LSTM-Attention
# =============================================================================
# Paper: "Travel Time Prediction Using Various Time Series Feature Generation
#         Techniques", IEEE Access, 2025
# =============================================================================
# Models trained in this notebook:
#   A. ARIMA  (rolling re-estimation, short-horizon)
#   B. ARIMAX (rolling re-estimation, TDA or K-Means exogenous, short-horizon)
#   C. XGBoost (fixed model, lag / TDA / KMeans features, long-horizon)
#   D. LSTM-Lag             (standard LSTM, lag-only, long-horizon)
#   E. LSTM-Attn+TDA        (attention-LSTM, TDA context, long-horizon)
#   F. LSTM-Attn+KMeans     (attention-LSTM, KMeans context, long-horizon)
# =============================================================================

# ── 1. IMPORTS & SEEDS ────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

# Statsmodels for ARIMA/ARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# XGBoost
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# TensorFlow / Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout,
                                     Concatenate, Activation, Lambda,
                                     Multiply, RepeatVector, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K

# Reproducibility (paper Section III-G, Section IV-A)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

OUTPUTS_DIR = "../outputs"
MODELS_DIR  = "../outputs/models"
os.makedirs(MODELS_DIR, exist_ok=True)

WINDOW_SIZE  = 10
N_TDA_FEATS  = 10   # 5 per H0 + 5 per H1
N_KM_CLASSES = 3    # one-hot K-means

# ── helper: load saved arrays ─────────────────────────────────────────────────
def load_arrays(tag: str, suffix: str):
    """Load pre-computed feature arrays from 01_feature_generation.py."""
    return (np.load(f"{OUTPUTS_DIR}/X_lag_{s}_{suffix}.npy") for s in
            ("train", "val", "test")), \
           (np.load(f"{OUTPUTS_DIR}/X_{tag}_{s}_{suffix}.npy") for s in
            ("train", "val", "test")), \
           (np.load(f"{OUTPUTS_DIR}/y_{s}_{suffix}.npy") for s in
            ("train", "val", "test"))


# =============================================================================
# A. ARIMA — SHORT-HORIZON ROLLING FORECAST
# =============================================================================
# Paper Section III-E1 / IV-D1 / Algorithm 1
#
# Two configurations:
#   Config 1 (Fixed-p):   p = 10, q selected via AIC over [0,5], d via ADF
#   Config 2 (AIC-opt):   (p,q) grid search over [0,5]², d via ADF
#
# Re-estimation at every prediction step (rolling forecast).
# Evaluated on 5-day window (May 2–6) and 26-day window (May 2–27).

def adf_differencing_order(series: np.ndarray,
                            significance: float = 0.05) -> int:
    """
    Determine differencing order d via Augmented Dickey-Fuller test.
    Returns d = 0 if series is already stationary.
    """
    result = adfuller(series, autolag="AIC")
    return 0 if result[1] <= significance else 1


def select_arima_order_aic(series: np.ndarray, d: int,
                            p_range=(0, 5), q_range=(0, 5),
                            fixed_p: int = None) -> tuple:
    """
    Grid search over (p, q) ∈ [0,5]² using AIC.
    If fixed_p is set, only q is searched (Config 1).
    """
    best_aic, best_pq = np.inf, (1, 1)
    p_vals = [fixed_p] if fixed_p is not None else range(p_range[0], p_range[1] + 1)

    for p in p_vals:
        for q in range(q_range[0], q_range[1] + 1):
            try:
                m = ARIMA(series, order=(p, d, q)).fit()
                if m.aic < best_aic:
                    best_aic, best_pq = m.aic, (p, q)
            except Exception:
                continue
    return best_pq


def ljung_box_check(residuals: np.ndarray,
                    lags=(10, 20, 30)) -> bool:
    """
    Ljung–Box test: returns True if residuals pass (white-noise, p > 0.05).
    Paper Section III-E1.
    """
    lb = acorr_ljungbox(residuals, lags=list(lags), return_df=True)
    return bool((lb["lb_pvalue"] > 0.05).all())


def run_arima_rolling(series: np.ndarray,
                      test_start_idx: int,
                      n_steps: int,
                      config: str = "aic_opt",
                      exog_series: np.ndarray = None) -> np.ndarray:
    """
    Rolling one-step-ahead ARIMA / ARIMAX forecast with re-estimation
    at every step. Implements Algorithm 1 from the paper.

    Parameters
    ----------
    series        : full normalized travel-time array (train+test)
    test_start_idx: index of the first test observation
    n_steps       : number of steps to forecast
    config        : 'fixed_p' or 'aic_opt'
    exog_series   : exogenous feature array (for ARIMAX), shape (N, n_feats)
                    or None for pure ARIMA

    Returns
    -------
    forecasts : ndarray of shape (n_steps,)
    """
    forecasts = []

    for i in range(n_steps):
        idx = test_start_idx + i
        # growing history: all observations up to current step
        history = series[:idx]

        # Step: determine differencing order (ADF)
        d = adf_differencing_order(history)

        # Step: select (p, q)
        if config == "fixed_p":
            p, q = select_arima_order_aic(history, d, fixed_p=10)
        else:   # aic_opt
            p, q = select_arima_order_aic(history, d)

        # Fit model
        try:
            if exog_series is not None:
                exog_hist = exog_series[:idx]
                exog_next = exog_series[idx: idx + 1]
                model = ARIMA(history, order=(p, d, q),
                              exog=exog_hist).fit()
                fc = model.forecast(steps=1, exog=exog_next)[0]
            else:
                model = ARIMA(history, order=(p, d, q)).fit()
                fc    = model.forecast(steps=1)[0]
        except Exception:
            fc = history[-1]   # fallback: persist last value

        forecasts.append(fc)

    return np.array(forecasts)


# ── Example usage for ARIMA on WSDOT Trip ID 1 ───────────────────────────────
# (Uncomment to run; requires the cleaned & normalized series from Notebook 1)
#
# df1    = pd.read_csv("../data/wsdot/trip_1.csv", parse_dates=["timestamp"])
# series = df1["travel_time_s_norm"].values   # pre-normalized full series
#
# 5-day window = 5 × 288 steps = 1440 steps
# test_start   = index of 2011-05-02 in series
# n_5day       = 1440
# n_26day      = 26 * 288
#
# preds_arima_5d  = run_arima_rolling(series, test_start, n_5day, "aic_opt")
# preds_arimax_5d = run_arima_rolling(series, test_start, n_5day, "aic_opt",
#                                     exog_series=tda_full_series)


# =============================================================================
# B. XGBOOST — LONG-HORIZON FIXED MODEL (Algorithm 2)
# =============================================================================
# Paper Section III-E2 / IV-D2
# Hyperparameters (Table 8): n_estimators=200, max_depth=4, lr=0.05,
#                             subsample=0.8, colsample_bytree=0.8
# 5-fold TimeSeriesSplit cross-validation on training partition.

XGBOOST_PARAMS = dict(
    n_estimators      = 200,
    max_depth         = 4,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = SEED,
    verbosity         = 0,
    tree_method       = "hist",   # fast CPU/GPU compatible
)


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray,   y_val: np.ndarray,
                  feature_tag: str, dataset_suffix: str) -> XGBRegressor:
    """
    Train XGBoost with 5-fold time-series CV then refit on full training set.
    Saves the trained model to disk.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    cv_maes = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        xva, yva = X_train[va_idx], y_train[va_idx]
        m = XGBRegressor(**XGBOOST_PARAMS)
        m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        fold_mae = np.mean(np.abs(m.predict(xva) - yva))
        cv_maes.append(fold_mae)
        print(f"  XGBoost {feature_tag} | fold {fold+1}/5 | MAE={fold_mae:.4f}")

    print(f"  CV-MAE = {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")

    # Final model on full training partition
    final = XGBRegressor(**XGBOOST_PARAMS)
    final.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)

    path = f"{MODELS_DIR}/xgb_{feature_tag}_{dataset_suffix}.json"
    final.save_model(path)
    print(f"  Saved → {path}")
    return final


def run_xgboost_all_features(suffix: str):
    """Train XGBoost for lag-only, +TDA, +KMeans on a given dataset suffix."""
    results = {}
    for feat in ("lag", "tda", "km"):
        Xtr = np.load(f"{OUTPUTS_DIR}/X_{feat}_train_{suffix}.npy")
        Xva = np.load(f"{OUTPUTS_DIR}/X_{feat}_val_{suffix}.npy")
        Xte = np.load(f"{OUTPUTS_DIR}/X_{feat}_test_{suffix}.npy")
        ytr = np.load(f"{OUTPUTS_DIR}/y_train_{suffix}.npy")
        yva = np.load(f"{OUTPUTS_DIR}/y_val_{suffix}.npy")
        yte = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")

        model = train_xgboost(Xtr, ytr, Xva, yva, feat, suffix)
        preds = model.predict(Xte)
        results[f"xgb_{feat}"] = {"preds": preds, "targets": yte}
        np.save(f"{OUTPUTS_DIR}/preds_xgb_{feat}_{suffix}.npy", preds)

    return results


# ── Run XGBoost for primary benchmark (Trip ID 1) ────────────────────────────
print("=== XGBoost — WSDOT Trip ID 1 ===")
xgb_results_tid1 = run_xgboost_all_features("tid1")

# ── Run XGBoost for PeMS Dataset 2 ───────────────────────────────────────────
PEMS_ROUTES = [405, 10, 605, 210, 60]
for route in PEMS_ROUTES:
    print(f"\n=== XGBoost — PeMS Route {route} ===")
    run_xgboost_all_features(f"pems{route}")


# =============================================================================
# C. LSTM — STANDARD (LAG-ONLY) — Algorithm 3
# =============================================================================
# Paper Section III-E3 / IV-D3
# Architecture: LSTM(64) → Dropout(0.3) → Dense(1)
# Training: Adam (lr=1e-3), batch=64, max 100 epochs, early stopping (pat=10)


def build_lstm_lag(window_size: int = WINDOW_SIZE) -> Model:
    """
    Standard LSTM for lag-only input.
    Input shape: (batch, window_size, 1)
    """
    inp  = Input(shape=(window_size, 1), name="lag_input")
    x    = LSTM(64, name="lstm")(inp)
    x    = Dropout(0.3, name="dropout")(x)
    out  = Dense(1, name="output")(x)
    return Model(inputs=inp, outputs=out, name="LSTM_Lag")


def train_lstm(model: Model,
               X_train: np.ndarray, y_train: np.ndarray,
               X_val:   np.ndarray, y_val:   np.ndarray,
               model_name: str, suffix: str) -> Model:
    """
    Compile and train an LSTM model with early stopping.
    Returns trained model.
    """
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.summary()

    ckpt_path = f"{MODELS_DIR}/{model_name}_{suffix}.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    # Reshape for LSTM: (samples, timesteps, features)
    Xtr_r = X_train.reshape(-1, X_train.shape[1], 1) \
            if X_train.ndim == 2 else X_train
    Xva_r = X_val.reshape(-1, X_val.shape[1], 1) \
            if X_val.ndim == 2 else X_val

    model.fit(Xtr_r, y_train,
              validation_data=(Xva_r, y_val),
              epochs=100, batch_size=64,
              callbacks=callbacks, verbose=1)

    print(f"  Saved → {ckpt_path}")
    return model


def predict_lstm_lag(model: Model, X_test: np.ndarray) -> np.ndarray:
    Xte_r = X_test.reshape(-1, X_test.shape[1], 1)
    return model.predict(Xte_r, verbose=0).flatten()


# =============================================================================
# D. LSTM-ATTENTION with TDA / K-MEANS CONTEXT — Algorithm 3 (Attn branch)
# =============================================================================
# Paper Section III-E4 / IV-D3
# Architecture:
#   Lag input  (w=10) → LSTM(64) → hidden states
#   Aux input  (TDA 10-D or KM 3-D) → attention scores via softmax
#   Context vector z = Σ a_t * h_t   (Eq. 13, 14)
#   z || h_T → Dense(1)


def build_lstm_attention(window_size: int = WINDOW_SIZE,
                         aux_dim:    int = N_TDA_FEATS) -> Model:
    """
    Attention-based LSTM.
    Lag input  : (batch, window_size, 1)
    Aux input  : (batch, aux_dim)          — TDA or K-Means features
    Output     : (batch, 1)

    Attention mechanism (paper Eq. 13-14):
        (a1,...,aT) = softmax(f_att(aux_features))
        z           = Σ a_t * h_t
    """
    # ── lag sequence branch ──────────────────────────────────────────────────
    lag_in   = Input(shape=(window_size, 1), name="lag_input")
    lstm_out = LSTM(64, return_sequences=True, name="lstm")(lag_in)
    # lstm_out shape: (batch, timesteps, 64)

    # ── auxiliary feature branch → attention weights ──────────────────────────
    aux_in   = Input(shape=(aux_dim,), name="aux_input")
    # Project aux features to (batch, timesteps) attention logits
    attn_logits = Dense(window_size, activation="linear",
                        name="attn_logits")(aux_in)
    attn_weights = Activation("softmax", name="attn_weights")(attn_logits)
    # Expand to (batch, timesteps, 1) for broadcasting
    attn_exp = Lambda(lambda x: K.expand_dims(x, axis=-1),
                      name="attn_expand")(attn_weights)

    # Weighted sum of LSTM hidden states → context vector (batch, 64)
    context = Multiply(name="attn_apply")([lstm_out, attn_exp])
    context = Lambda(lambda x: K.sum(x, axis=1),
                     name="context_sum")(context)

    # Final hidden state (batch, 64)
    final_h = Lambda(lambda x: x[:, -1, :],
                     name="final_hidden")(lstm_out)

    # Concatenate context + final hidden state
    merged  = Concatenate(name="merge")([context, final_h])
    merged  = Dropout(0.3, name="dropout")(merged)
    out     = Dense(1, name="output")(merged)

    return Model(inputs=[lag_in, aux_in], outputs=out,
                 name=f"LSTM_Attn_aux{aux_dim}")


def train_lstm_attention(model: Model,
                         X_lag_tr: np.ndarray, X_aux_tr: np.ndarray,
                         y_tr:     np.ndarray,
                         X_lag_va: np.ndarray, X_aux_va: np.ndarray,
                         y_va:     np.ndarray,
                         model_name: str, suffix: str) -> Model:
    """Train attention-LSTM with dual inputs."""
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.summary()

    ckpt_path = f"{MODELS_DIR}/{model_name}_{suffix}.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss",
                        save_best_only=True, verbose=0),
    ]

    # reshape lag: (samples, timesteps, 1)
    Xtr_r = X_lag_tr.reshape(-1, WINDOW_SIZE, 1)
    Xva_r = X_lag_va.reshape(-1, WINDOW_SIZE, 1)

    model.fit([Xtr_r, X_aux_tr], y_tr,
              validation_data=([Xva_r, X_aux_va], y_va),
              epochs=100, batch_size=64,
              callbacks=callbacks, verbose=1)

    print(f"  Saved → {ckpt_path}")
    return model


def predict_lstm_attention(model: Model,
                           X_lag: np.ndarray,
                           X_aux: np.ndarray) -> np.ndarray:
    Xte_r = X_lag.reshape(-1, WINDOW_SIZE, 1)
    return model.predict([Xte_r, X_aux], verbose=0).flatten()


# ── Run LSTM models for Trip ID 1 ────────────────────────────────────────────
print("\n=== LSTM Models — WSDOT Trip ID 1 ===")

suffix = "tid1"

# Load arrays
Xlag_tr = np.load(f"{OUTPUTS_DIR}/X_lag_train_{suffix}.npy")
Xlag_va = np.load(f"{OUTPUTS_DIR}/X_lag_val_{suffix}.npy")
Xlag_te = np.load(f"{OUTPUTS_DIR}/X_lag_test_{suffix}.npy")
ytr     = np.load(f"{OUTPUTS_DIR}/y_train_{suffix}.npy")
yva     = np.load(f"{OUTPUTS_DIR}/y_val_{suffix}.npy")
yte     = np.load(f"{OUTPUTS_DIR}/y_test_{suffix}.npy")

Xtda_tr = np.load(f"{OUTPUTS_DIR}/X_tda_train_{suffix}.npy")[:, WINDOW_SIZE:]
Xtda_va = np.load(f"{OUTPUTS_DIR}/X_tda_val_{suffix}.npy")[:, WINDOW_SIZE:]
Xtda_te = np.load(f"{OUTPUTS_DIR}/X_tda_test_{suffix}.npy")[:, WINDOW_SIZE:]

Xkm_tr  = np.load(f"{OUTPUTS_DIR}/X_km_train_{suffix}.npy")[:, WINDOW_SIZE:]
Xkm_va  = np.load(f"{OUTPUTS_DIR}/X_km_val_{suffix}.npy")[:, WINDOW_SIZE:]
Xkm_te  = np.load(f"{OUTPUTS_DIR}/X_km_test_{suffix}.npy")[:, WINDOW_SIZE:]

# D. Standard LSTM (lag-only)
print("\n--- LSTM-Lag ---")
lstm_lag = build_lstm_lag()
lstm_lag = train_lstm(lstm_lag, Xlag_tr, ytr, Xlag_va, yva,
                      "lstm_lag", suffix)
preds_lstm_lag = predict_lstm_lag(lstm_lag, Xlag_te)
np.save(f"{OUTPUTS_DIR}/preds_lstm_lag_{suffix}.npy", preds_lstm_lag)

# E. LSTM-Attn + TDA
print("\n--- LSTM-Attn+TDA ---")
lstm_attn_tda = build_lstm_attention(aux_dim=N_TDA_FEATS)
lstm_attn_tda = train_lstm_attention(
    lstm_attn_tda,
    Xlag_tr, Xtda_tr, ytr,
    Xlag_va, Xtda_va, yva,
    "lstm_attn_tda", suffix)
preds_lstm_tda = predict_lstm_attention(lstm_attn_tda, Xlag_te, Xtda_te)
np.save(f"{OUTPUTS_DIR}/preds_lstm_attn_tda_{suffix}.npy", preds_lstm_tda)

# F. LSTM-Attn + KMeans
print("\n--- LSTM-Attn+KMeans ---")
lstm_attn_km = build_lstm_attention(aux_dim=N_KM_CLASSES)
lstm_attn_km = train_lstm_attention(
    lstm_attn_km,
    Xlag_tr, Xkm_tr, ytr,
    Xlag_va, Xkm_va, yva,
    "lstm_attn_km", suffix)
preds_lstm_km = predict_lstm_attention(lstm_attn_km, Xlag_te, Xkm_te)
np.save(f"{OUTPUTS_DIR}/preds_lstm_attn_km_{suffix}.npy", preds_lstm_km)

print("\n=== 02_model_training.py COMPLETE ===")
