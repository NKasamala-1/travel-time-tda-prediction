# =============================================================================
# NOTEBOOK 1: feature_generation.py
# Travel Time Prediction — Data Cleaning, Preprocessing & Feature Engineering
# =============================================================================
# Paper: "Travel Time Prediction Using Various Time Series Feature Generation
#         Techniques", IEEE Access, 2025
# Authors: Nancy Kasamala et al., South Carolina State University /
#          North Carolina A&T State University
# GitHub:  https://github.com/nkasamal-scsu/travel-time-tda-prediction
# =============================================================================
# Pipeline stages covered in this notebook:
#   1. Library imports & random seed setup
#   2. Dataset 1 (WSDOT) loading & cleaning
#   3. Dataset 2 (PeMS) loading, travel-time derivation & cleaning
#   4. MinMax normalization (fit on train only)
#   5. Temporal train / validation / test split
#   6. Sliding window construction (w = 10)
#   7. TDA feature extraction (ripser, Vietoris–Rips, persistent homology)
#   8. K-Means cluster label generation
#   9. Save feature matrices and split indices to disk
# =============================================================================

# ── 1. IMPORTS & SEEDS ────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
import ripser                          # pip install ripser  (v0.6.8)
from scipy.stats import entropy as sp_entropy
import warnings
warnings.filterwarnings("ignore")

# Reproducibility — all random seeds fixed at 42 (paper Section III-G)
SEED = 42
np.random.seed(SEED)

# Window size (chosen via 5-fold CV in paper Section III-A1)
WINDOW_SIZE = 10

# TDA parameters (paper Section III-D)
TAU          = 3       # time-delay embedding dimension (false-NN < 5 %)
MAXDIM       = 1       # compute H0 and H1
N_CLUSTERS   = 3       # K = 3 (elbow method, paper Section III-C3)

OUTPUTS_DIR = "../outputs"
SPLITS_DIR  = "../splits"
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR,  exist_ok=True)

print("=== Imports & seeds OK ===")

# ── 2. DATASET 1 — WSDOT TRAVEL TIME ARCHIVE ─────────────────────────────────
# Source : https://www.wsdot.wa.gov/mapsdata/travel/travelmonitoring.htm
# Archive: WSDOT-TT-2011-I5-5min
# 8 Trip IDs, May 2 – Oct 31 2011, 5-minute intervals, weekdays only
# Reference [31] in paper.

DATA_DIR_WSDOT = "../data/wsdot"   # place the raw CSV files here

# ── helper ──────────────────────────────────────────────────────────────────
def load_wsdot(trip_id: int, data_dir: str) -> pd.DataFrame:
    """
    Load a single WSDOT Trip-ID CSV.
    Expected columns: ['timestamp', 'travel_time_s']
    Adjust column names to match your downloaded file layout.
    """
    fpath = os.path.join(data_dir, f"trip_{trip_id}.csv")
    df = pd.read_csv(fpath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def clean_wsdot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Section III-B1 / IV-B3 preprocessing pipeline (6 steps).

    Step 1 — Missing value removal by listwise deletion.
              Timestamp-gap analysis: any gap > 5 min flags a missing record.
              MCAR confirmed by Little's test (p > 0.05) — not re-run here.
    Step 2 — Outlier removal: records outside [100 s, 5 000 s] removed
              (0.3 % of records, confirmed as sensor errors).
    Steps 3-6 are applied later (normalization / split / window / TDA).
    """
    # Step 1: drop rows with NaN travel time
    before = len(df)
    df = df.dropna(subset=["travel_time_s"]).copy()

    # Step 1: also drop where timestamp gap > 5 min (missing interval)
    df = df.sort_values("timestamp").reset_index(drop=True)
    time_diff = df["timestamp"].diff().dt.total_seconds()
    df = df[~((time_diff > 300) & (time_diff.notna()))].reset_index(drop=True)

    after_missing = len(df)
    print(f"  Step 1 — removed {before - after_missing} missing records "
          f"({(before - after_missing)/before*100:.2f} %)")

    # Step 2: outlier removal
    before2 = len(df)
    df = df[(df["travel_time_s"] >= 100) & (df["travel_time_s"] <= 5000)].reset_index(drop=True)
    print(f"  Step 2 — removed {before2 - len(df)} outliers "
          f"({(before2 - len(df))/before2*100:.2f} %)")

    return df


# ── process all 8 Trip IDs ───────────────────────────────────────────────────
TRIP_IDS = [1, 2, 3, 4, 5, 6, 7, 8]
wsdot_clean = {}

for tid in TRIP_IDS:
    print(f"\n--- WSDOT Trip ID {tid} ---")
    df = load_wsdot(tid, DATA_DIR_WSDOT)
    df = clean_wsdot(df)
    wsdot_clean[tid] = df
    print(f"  Usable records: {len(df):,}")

print("\n=== WSDOT cleaning done ===")

# ── 3. DATASET 2 — CALTRANS PeMS LONGITUDINAL SPEED DATA ─────────────────────
# Source : https://pems.dot.ca.gov  (District 7, station 5-min data)
# Routes : 405, 10, 605, 210, 60  —  30 days × 288 intervals = 8,640 records
# Reference [32] in paper.

DATA_DIR_PEMS = "../data/pems"

SEGMENT_LENGTHS_KM = {
    405: 2.8,
    10:  3.2,
    605: 2.1,
    210: 1.9,
    60:  2.5,
}

PEMS_ROUTES = [405, 10, 605, 210, 60]


def load_pems(route: int, data_dir: str) -> pd.DataFrame:
    """
    Load a PeMS route CSV.
    Expected columns: ['timestamp', 'speed_mph']
    """
    fpath = os.path.join(data_dir, f"route_{route}.csv")
    df = pd.read_csv(fpath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def derive_travel_time(df: pd.DataFrame, segment_km: float) -> pd.DataFrame:
    """
    Convert speed (mph → km/h) to travel time (seconds).
    Formula (Eq. 3 in paper):
        TravelTime (s) = (SegmentLength_km / Speed_kmh) × 3600
    """
    df = df.copy()
    df["speed_kmh"] = df["speed_mph"] * 1.60934
    # avoid division by zero
    df["speed_kmh"] = df["speed_kmh"].replace(0, np.nan)
    df["travel_time_s"] = (segment_km / df["speed_kmh"]) * 3600
    return df


def clean_pems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paper Section III-B2 preprocessing (6 steps, PeMS variant).

    Step 1 — Remove speed values outside [5, 75] mph (0.8 % of records).
    Step 2 — Clip derived travel times to [60 s, 800 s].
    Steps 3-6 applied later.
    """
    before = len(df)
    df = df[(df["speed_mph"] >= 5) & (df["speed_mph"] <= 75)].copy()
    print(f"  Step 1 — removed {before - len(df)} speed outliers "
          f"({(before - len(df))/before*100:.2f} %)")

    df["travel_time_s"] = df["travel_time_s"].clip(60, 800)
    df = df.dropna(subset=["travel_time_s"]).reset_index(drop=True)
    return df


pems_clean = {}
for route in PEMS_ROUTES:
    print(f"\n--- PeMS Route {route} ---")
    df = load_pems(route, DATA_DIR_PEMS)
    df = derive_travel_time(df, SEGMENT_LENGTHS_KM[route])
    df = clean_pems(df)
    pems_clean[route] = df
    print(f"  Usable records: {len(df):,}")

print("\n=== PeMS cleaning done ===")

# ── 4. TEMPORAL TRAIN / VALIDATION / TEST SPLIT ───────────────────────────────
# Dataset 1 (WSDOT, paper Section III-A4):
#   Train : May 2 – Aug 31 2011  (60 %)
#   Val   : September 2011        (20 %)
#   Test  : October 2011          (20 %)
#
# Dataset 2 (PeMS, paper Section III-B2):
#   Train : Days  1–18  (60 %)
#   Val   : Days 19–24  (20 %)
#   Test  : Days 25–30  (20 %)


def split_wsdot(df: pd.DataFrame):
    """Chronological split for WSDOT (date-based)."""
    train = df[df["timestamp"] < "2011-09-01"].copy()
    val   = df[(df["timestamp"] >= "2011-09-01") &
               (df["timestamp"] <  "2011-10-01")].copy()
    test  = df[df["timestamp"] >= "2011-10-01"].copy()
    return train, val, test


def split_pems(df: pd.DataFrame, train_frac=0.60, val_frac=0.20):
    """Chronological 60/20/20 split for PeMS."""
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train:n_train + n_val].copy()
    test  = df.iloc[n_train + n_val:].copy()
    return train, val, test


# ── 5. MINMAX NORMALIZATION (fit on train only) ───────────────────────────────
# Paper Section III-B: "MinMax normalization using training-partition statistics
# only, strictly preventing information leakage from validation or test data."


def normalize_split(train, val, test, col="travel_time_s"):
    """Fit scaler on train, apply to all splits; return arrays and scaler."""
    scaler = MinMaxScaler()
    train_arr = scaler.fit_transform(train[[col]]).flatten()
    val_arr   = scaler.transform(val[[col]]).flatten()
    test_arr  = scaler.transform(test[[col]]).flatten()
    return train_arr, val_arr, test_arr, scaler


# ── 6. SLIDING WINDOW CONSTRUCTION (w = 10) ──────────────────────────────────
# Paper Section III-A1, Eq. 1:
#   Wi = [Ti, Ti+1, ..., Ti+w-1],  w = 10
#   Target: Ti+w


def make_windows(series: np.ndarray, w: int = WINDOW_SIZE):
    """
    Convert a 1-D normalized time series into supervised (X, y) arrays.
    Window is constructed AFTER the split to prevent look-ahead bias.
    """
    X, y = [], []
    for i in range(len(series) - w):
        X.append(series[i: i + w])
        y.append(series[i + w])
    return np.array(X), np.array(y)


# ── 7. TDA FEATURE EXTRACTION ─────────────────────────────────────────────────
# Paper Section III-D, three-step pipeline:
#   Step 1 — Time-delay embedding (Takens' theorem), τ = 3  (Eq. 5)
#   Step 2 — Vietoris–Rips filtration via ripser (v0.6.8)   (Eq. 6)
#   Step 3 — Persistent homology → 10 descriptors           (Eq. 7, Table 1)
#
# Descriptors per homology group (H0, H1):
#   • Betti number   β_k = |PD_k|
#   • Mean Persistence  MP = (1/|PD_k|) Σ (d - b)
#   • Persistence Entropy  PE = -Σ (l_i / L_0) log(l_i / L_0)
#   • L1 Norm  Σ l_i
#   • L2 Norm  (Σ l_i²)^0.5
# → 5 descriptors × 2 groups = 10 TDA features per window


def time_delay_embed(window: np.ndarray, tau: int = TAU) -> np.ndarray:
    """
    Takens time-delay embedding.
    Zs = (Y(s), Y(s-1), ..., Y(s-τ+1))  ∈ R^τ   [Eq. 5]
    Returns point cloud of shape (n_points, tau).
    """
    n = len(window)
    n_points = n - tau + 1
    cloud = np.array([window[i: i + tau] for i in range(n_points)])
    return cloud


def persistence_descriptors(dgm: np.ndarray) -> dict:
    """
    Compute 5 topological descriptors from a single persistence diagram.
    Infinite bars (death == inf) are removed before computation.

    Parameters
    ----------
    dgm : ndarray, shape (n, 2)
        Persistence diagram with columns [birth, death].

    Returns
    -------
    dict with keys: betti, mean_persistence, persistence_entropy, l1_norm, l2_norm
    """
    # Remove infinite bars
    finite = dgm[np.isfinite(dgm[:, 1])]

    if len(finite) == 0:
        return dict(betti=0, mean_persistence=0.0,
                    persistence_entropy=0.0, l1_norm=0.0, l2_norm=0.0)

    lifespans = finite[:, 1] - finite[:, 0]   # l_i = d_i - b_i  [Eq. 7]
    L0        = lifespans.sum()

    betti              = len(finite)
    mean_persistence   = lifespans.mean()
    l1_norm            = L0
    l2_norm            = float(np.sqrt((lifespans ** 2).sum()))

    # Persistence entropy: -Σ (l_i/L0) log(l_i/L0)
    if L0 > 0:
        probs              = lifespans / L0
        persistence_entropy = float(sp_entropy(probs, base=np.e))
    else:
        persistence_entropy = 0.0

    return dict(betti=betti,
                mean_persistence=float(mean_persistence),
                persistence_entropy=persistence_entropy,
                l1_norm=l1_norm,
                l2_norm=l2_norm)


def extract_tda_features(window: np.ndarray, tau: int = TAU,
                          maxdim: int = MAXDIM) -> np.ndarray:
    """
    Full TDA pipeline for a single lag window → 10-D feature vector.

    Steps:
      1. Time-delay embedding (Eq. 5)
      2. Vietoris–Rips filtration via ripser   (Eq. 6)
      3. Persistent homology → descriptors     (Table 1)
    """
    cloud = time_delay_embed(window, tau=tau)

    # ripser returns list of persistence diagrams, one per dimension
    result = ripser.ripser(cloud, maxdim=maxdim)["dgms"]

    h0_desc = persistence_descriptors(result[0])   # H0: connected components
    h1_desc = persistence_descriptors(result[1])   # H1: loops

    features = np.array([
        h0_desc["betti"],
        h0_desc["mean_persistence"],
        h0_desc["persistence_entropy"],
        h0_desc["l1_norm"],
        h0_desc["l2_norm"],
        h1_desc["betti"],
        h1_desc["mean_persistence"],
        h1_desc["persistence_entropy"],
        h1_desc["l1_norm"],
        h1_desc["l2_norm"],
    ], dtype=np.float32)

    return features


def build_tda_matrix(X_windows: np.ndarray, tau: int = TAU,
                     maxdim: int = MAXDIM) -> np.ndarray:
    """
    Compute TDA features for every lag window.
    Returns array of shape (n_samples, 10).
    Average computation time: 0.8 ms per window (paper Section III-G).
    """
    tda_feats = []
    for i, win in enumerate(X_windows):
        tda_feats.append(extract_tda_features(win, tau=tau, maxdim=maxdim))
        if (i + 1) % 5000 == 0:
            print(f"  TDA: processed {i+1}/{len(X_windows)} windows")
    return np.vstack(tda_feats)


# ── 8. K-MEANS CLUSTER LABELS (one-hot, K = 3) ───────────────────────────────
# Paper Section III-C3 / III-D5:
#   K = 3 selected via elbow method on WCSS over K ∈ {2,3,4,5,6}
#   k-means++ initialization, n_init = 10, random_state = 42
#   Fitted on TRAINING partition only; applied to val/test without re-fitting.
#   One-hot encoded → 3-D binary vector concatenated to lag vector.


def fit_kmeans(X_train: np.ndarray, k: int = N_CLUSTERS,
               seed: int = SEED) -> KMeans:
    """Fit K-Means on training lag vectors only."""
    km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                random_state=seed)
    km.fit(X_train)
    return km


def kmeans_onehot(X: np.ndarray, km: KMeans) -> np.ndarray:
    """
    Predict cluster labels and one-hot encode.
    Returns array of shape (n_samples, K).
    """
    labels = km.predict(X)
    onehot = np.zeros((len(labels), km.n_clusters), dtype=np.float32)
    onehot[np.arange(len(labels)), labels] = 1.0
    return onehot


# ── 9. WINDOW SIZE SELECTION VIA 5-FOLD CV ────────────────────────────────────
# Paper Section III-A1: w chosen over {5, 10, 15, 20} by 5-fold TimeSeriesSplit
# This block shows the selection logic for Trip ID 1 only.


def select_window_size(series_train: np.ndarray,
                       candidates=(5, 10, 15, 20)) -> int:
    """
    Select optimal window size via 5-fold time-series CV on training data.
    Uses mean absolute error as the selection criterion.
    """
    from xgboost import XGBRegressor
    tscv = TimeSeriesSplit(n_splits=5)
    best_w, best_mae = None, np.inf

    for w in candidates:
        X_w, y_w = make_windows(series_train, w=w)
        fold_maes = []
        for tr_idx, va_idx in tscv.split(X_w):
            xtr, ytr = X_w[tr_idx], y_w[tr_idx]
            xva, yva = X_w[va_idx], y_w[va_idx]
            m = XGBRegressor(n_estimators=100, max_depth=3,
                             learning_rate=0.05, random_state=SEED,
                             verbosity=0)
            m.fit(xtr, ytr)
            preds = m.predict(xva)
            fold_maes.append(np.mean(np.abs(preds - yva)))

        mean_mae = np.mean(fold_maes)
        print(f"  w={w:2d}  CV-MAE = {mean_mae:.6f}")
        if mean_mae < best_mae:
            best_mae, best_w = mean_mae, w

    print(f"  → Best window size: w = {best_w}  (MAE={best_mae:.6f})")
    return best_w


# ── 10. FULL PIPELINE: WSDOT TRIP ID 1 (PRIMARY BENCHMARK) ───────────────────

print("\n=== Running full pipeline for WSDOT Trip ID 1 ===")

df1 = wsdot_clean[1]
train1, val1, test1 = split_wsdot(df1)

print(f"  Train: {len(train1):,} | Val: {len(val1):,} | Test: {len(test1):,}")

# Normalization
tr_norm, va_norm, te_norm, scaler1 = normalize_split(train1, val1, test1)

# Sliding windows
X_tr1, y_tr1 = make_windows(tr_norm, WINDOW_SIZE)
X_va1, y_va1 = make_windows(va_norm, WINDOW_SIZE)
X_te1, y_te1 = make_windows(te_norm, WINDOW_SIZE)

print(f"  Train windows: {X_tr1.shape} | Val: {X_va1.shape} | Test: {X_te1.shape}")

# TDA features
print("  Extracting TDA features (train) ...")
TDA_tr1 = build_tda_matrix(X_tr1)
print("  Extracting TDA features (val)   ...")
TDA_va1 = build_tda_matrix(X_va1)
print("  Extracting TDA features (test)  ...")
TDA_te1 = build_tda_matrix(X_te1)

# K-Means features
print("  Fitting K-Means on train ...")
km1 = fit_kmeans(X_tr1)
KM_tr1 = kmeans_onehot(X_tr1, km1)
KM_va1 = kmeans_onehot(X_va1, km1)
KM_te1 = kmeans_onehot(X_te1, km1)

# Concatenated feature matrices
X_tda_tr1  = np.hstack([X_tr1, TDA_tr1])
X_tda_va1  = np.hstack([X_va1, TDA_va1])
X_tda_te1  = np.hstack([X_te1, TDA_te1])

X_km_tr1   = np.hstack([X_tr1, KM_tr1])
X_km_va1   = np.hstack([X_va1, KM_va1])
X_km_te1   = np.hstack([X_te1, KM_te1])

# Save to disk
np.save(f"{OUTPUTS_DIR}/X_lag_train_tid1.npy",  X_tr1)
np.save(f"{OUTPUTS_DIR}/X_lag_val_tid1.npy",    X_va1)
np.save(f"{OUTPUTS_DIR}/X_lag_test_tid1.npy",   X_te1)
np.save(f"{OUTPUTS_DIR}/y_train_tid1.npy",      y_tr1)
np.save(f"{OUTPUTS_DIR}/y_val_tid1.npy",        y_va1)
np.save(f"{OUTPUTS_DIR}/y_test_tid1.npy",       y_te1)

np.save(f"{OUTPUTS_DIR}/X_tda_train_tid1.npy",  X_tda_tr1)
np.save(f"{OUTPUTS_DIR}/X_tda_val_tid1.npy",    X_tda_va1)
np.save(f"{OUTPUTS_DIR}/X_tda_test_tid1.npy",   X_tda_te1)

np.save(f"{OUTPUTS_DIR}/X_km_train_tid1.npy",   X_km_tr1)
np.save(f"{OUTPUTS_DIR}/X_km_val_tid1.npy",     X_km_va1)
np.save(f"{OUTPUTS_DIR}/X_km_test_tid1.npy",    X_km_te1)

# Save exact split indices (paper reproducibility requirement)
split_indices = {
    "train_start": 0,
    "train_end":   len(X_tr1),
    "val_start":   len(X_tr1),
    "val_end":     len(X_tr1) + len(X_va1),
    "test_start":  len(X_tr1) + len(X_va1),
    "test_end":    len(X_tr1) + len(X_va1) + len(X_te1),
}
pd.DataFrame([split_indices]).to_csv(
    f"{SPLITS_DIR}/split_indices_tid1.csv", index=False)

print("\n=== Feature generation complete — all arrays saved ===")

# ── 11. SAME PIPELINE FOR ALL 8 TRIP IDs (loop) ───────────────────────────────
print("\n=== Processing all 8 WSDOT Trip IDs ===")
for tid in TRIP_IDS:
    df   = wsdot_clean[tid]
    tr, va, te = split_wsdot(df)
    tr_n, va_n, te_n, _ = normalize_split(tr, va, te)
    Xtr, ytr = make_windows(tr_n, WINDOW_SIZE)
    Xva, yva = make_windows(va_n, WINDOW_SIZE)
    Xte, yte = make_windows(te_n, WINDOW_SIZE)

    tda_tr = build_tda_matrix(Xtr)
    tda_va = build_tda_matrix(Xva)
    tda_te = build_tda_matrix(Xte)

    km = fit_kmeans(Xtr)
    km_tr = kmeans_onehot(Xtr, km)
    km_va = kmeans_onehot(Xva, km)
    km_te = kmeans_onehot(Xte, km)

    for tag, arr in [("lag_train", Xtr), ("lag_val", Xva), ("lag_test", Xte),
                     ("tda_train", np.hstack([Xtr, tda_tr])),
                     ("tda_val",   np.hstack([Xva, tda_va])),
                     ("tda_test",  np.hstack([Xte, tda_te])),
                     ("km_train",  np.hstack([Xtr, km_tr])),
                     ("km_val",    np.hstack([Xva, km_va])),
                     ("km_test",   np.hstack([Xte, km_te])),
                     ("y_train",   ytr), ("y_val", yva), ("y_test", yte)]:
        np.save(f"{OUTPUTS_DIR}/X_{tag}_tid{tid}.npy", arr)

    print(f"  Trip ID {tid} done — train={len(Xtr):,} | val={len(Xva):,} | test={len(Xte):,}")

print("\n=== All Trip IDs processed ===")

# ── 12. PEMS PIPELINE ─────────────────────────────────────────────────────────
print("\n=== Processing PeMS Dataset 2 ===")
for route in PEMS_ROUTES:
    df   = pems_clean[route]
    tr, va, te = split_pems(df)
    tr_n, va_n, te_n, _ = normalize_split(tr, va, te)
    Xtr, ytr = make_windows(tr_n, WINDOW_SIZE)
    Xva, yva = make_windows(va_n, WINDOW_SIZE)
    Xte, yte = make_windows(te_n, WINDOW_SIZE)

    tda_tr = build_tda_matrix(Xtr)
    tda_va = build_tda_matrix(Xva)
    tda_te = build_tda_matrix(Xte)

    km = fit_kmeans(Xtr)
    km_tr = kmeans_onehot(Xtr, km)
    km_va = kmeans_onehot(Xva, km)
    km_te = kmeans_onehot(Xte, km)

    for tag, arr in [("lag_train", Xtr), ("lag_val", Xva), ("lag_test", Xte),
                     ("tda_train", np.hstack([Xtr, tda_tr])),
                     ("tda_val",   np.hstack([Xva, tda_va])),
                     ("tda_test",  np.hstack([Xte, tda_te])),
                     ("km_train",  np.hstack([Xtr, km_tr])),
                     ("km_val",    np.hstack([Xva, km_va])),
                     ("km_test",   np.hstack([Xte, km_te])),
                     ("y_train",   ytr), ("y_val", yva), ("y_test", yte)]:
        np.save(f"{OUTPUTS_DIR}/X_{tag}_pems{route}.npy", arr)

    print(f"  Route {route} done — train={len(Xtr):,} | val={len(Xva):,} | test={len(Xte):,}")

print("\n=== 01_feature_generation.py COMPLETE ===")
