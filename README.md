# Travel Time Prediction Using Various Time Series Feature Generation Techniques

**IEEE Access, 2025**  
Nancy Kasamala · South Carolina State University  


> DOI: *to be assigned upon publication*  
> GitHub: `https://github.com/nkasamal-scsu/travel-time-tda-prediction`

---

## Overview

This repository contains the complete, reproducible pipeline for the paper:

> *"Travel Time Prediction Using Various Time Series Feature Generation Techniques"*

The study presents a controlled comparative evaluation of **five forecasting
architectures** (ARIMA, ARIMAX, XGBoost, LSTM, Attention-LSTM) across **three
feature representations**:

| Feature Set | Description |
|---|---|
| **Lag-only** | 10 most recent normalized travel-time observations |
| **Lag + TDA** | Lag vector augmented with 10 topological descriptors (persistent homology) |
| **Lag + KMeans** | Lag vector augmented with K=3 one-hot cluster labels (traffic regime) |

### Key Results (Dataset 1, Trip ID 1 — 37,728 records)

| Model | MAE (s) | RMSE (s) | MAPE (%) | R² |
|---|---|---|---|---|
| XGBoost-Lag | 20.93 | 80.78 | 1.12 | 0.886 |
| **XGBoost+TDA** | **19.63** | 78.89 | 1.10 | 0.886 |
| LSTM-Lag | 17.43 | 40.31 | 1.09 | 0.943 |
| **LSTM-Attn+TDA** ✓ | **15.62** | **37.29** | **0.96** | **0.975** |

---

## Repository Structure

```
travel-time-tda-prediction/
│
├── notebooks/
│   ├── 01_feature_generation.py   # Data cleaning, TDA, K-Means, sliding windows
│   ├── 02_model_training.py       # ARIMA, ARIMAX, XGBoost, LSTM, LSTM-Attention
│   └── 03_evaluation.py           # Metrics, Bootstrap CIs, DM test, SHAP, Pearson
│
├── data/
│   ├── wsdot/                     # Place WSDOT Trip ID CSVs here (trip_1.csv … trip_8.csv)
│   └── pems/                      # Place PeMS route CSVs here  (route_405.csv … route_60.csv)
│
├── outputs/                       # Auto-created — feature arrays, predictions, plots
│   ├── models/                    # Saved model weights
│   └── plots/                     # Figures
│
├── splits/                        # Exact train/val/test index CSVs (reproducibility)
│
├── requirements.txt
└── README.md
```

---

## Datasets

### Dataset 1 — WSDOT Travel Time Archive
- **Source**: Washington State Department of Transportation  
  https://www.wsdot.wa.gov/mapsdata/travel/travelmonitoring.htm  
  Archive ID: `WSDOT-TT-2011-I5-5min`
- **Coverage**: May 2 – October 31, 2011 | 8 Trip IDs | I-5 Seattle corridor | 5-min intervals
- **Size**: 261,408 raw records → 255,487 usable (after cleaning)

Download and place files as `data/wsdot/trip_1.csv` … `data/wsdot/trip_8.csv`  
Expected columns: `timestamp, travel_time_s`

### Dataset 2 — Caltrans PeMS Longitudinal Speed
- **Source**: California Performance Measurement System (Caltrans)  
  https://pems.dot.ca.gov | District 7 | Station 5-minute data
- **Routes**: 405, 10, 605, 210, 60 | 30 days × 288 intervals = 8,640 records/route
- **Segment lengths (km)**: 405→2.8, 10→3.2, 605→2.1, 210→1.9, 60→2.5

Download and place files as `data/pems/route_405.csv` … `data/pems/route_60.csv`  
Expected columns: `timestamp, speed_mph`

---

## Preprocessing Pipeline

Documented in `01_feature_generation.py` and paper Section III-B / IV-B3.

**Dataset 1 (WSDOT) — 6 steps:**
1. Missing value removal by listwise deletion (853 records, 2.26%, MCAR by Little's test)
2. Outlier removal: records outside [100 s, 5 000 s] (0.3%)
3. MinMax normalization [0, 1] — fit on training partition only
4. Temporal split: May–Aug (60% train) | September (20% val) | October (20% test)
5. 10-step sliding window constructed after split (no leakage)
6. TDA features computed via `ripser v0.6.8`, τ=3, maxdim=1

**Dataset 2 (PeMS) — same 6 steps adapted:**
1. Remove speed outside [5, 75] mph (0.8%)
2. Clip derived travel times to [60 s, 800 s]
3–6. Same as Dataset 1

---

## TDA Feature Extraction (3-Step Pipeline)

Paper Section III-D, implemented in `01_feature_generation.py`.

**Step 1 — Time-Delay Embedding** (Takens' theorem, τ=3):
```
Zs = (Y(s), Y(s-1), Y(s-τ+1)) ∈ R^τ
```

**Step 2 — Vietoris–Rips Filtration** (`ripser v0.6.8`):
```
σ ∈ VRε(X) ⟺ d(xi, xj) ≤ ε  ∀xi, xj ∈ σ
```

**Step 3 — Persistent Homology → 10 Descriptors** (5 per H0, 5 per H1):

| Feature | Formula | Traffic Interpretation |
|---|---|---|
| Betti number | β_k = \|PD_k\| | Connected components / loops |
| Mean Persistence | (1/\|PD_k\|) Σ(d-b) | 24-hour periodicity |
| Persistence Entropy | -Σ(l/L₀) log(l/L₀) | Rush-hour dynamics |
| L1 Norm | Σ l_i | Total structural complexity |
| L2 Norm | (Σ l²_i)^0.5 | Dominant structural features |

Computation time: **0.8 ms/window** (Intel Ultra 7 155U, 16 GB RAM).

---

## Models

| Model | Type | Feature Input | Inference |
|---|---|---|---|
| ARIMA | Statistical | Lag-only | Rolling re-estimation |
| ARIMAX | Statistical | Lag + TDA or KMeans | Rolling re-estimation |
| XGBoost | ML | Lag / +TDA / +KMeans | Fixed |
| LSTM-Lag | Deep Learning | Lag-only | Fixed |
| LSTM-Attn+TDA | Deep Learning | Lag + TDA (attention context) | Fixed |
| LSTM-Attn+KMeans | Deep Learning | Lag + KMeans (attention context) | Fixed |

**Hyperparameters (Table 8 in paper):**

```python
# XGBoost
n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, colsample_bytree=0.8

# LSTM / LSTM-Attn
hidden_units=64, dropout=0.3, batch_size=64,
optimizer=Adam(lr=1e-3), early_stopping(patience=10)
```

---

## Reproducibility

All random seeds fixed at **42**:
```python
np.random.seed(42)
tf.random.set_seed(42)
# KMeans random_state=42
```

Exact train/validation/test split indices are saved as CSV in `splits/`.

---

## Installation

```bash
git clone https://github.com/nkasamal-scsu/travel-time-tda-prediction.git
cd travel-time-tda-prediction
pip install -r requirements.txt
```

**Tested environment**: Python 3.11.13, Google Colab (CPU), Windows 11 host.

---

## Usage

Run notebooks in order:

```bash
# Step 1: Data cleaning, feature engineering
python notebooks/01_feature_generation.py

# Step 2: Model training (ARIMA, XGBoost, LSTM, LSTM-Attention)
python notebooks/02_model_training.py

# Step 3: Evaluation, bootstrap CIs, SHAP, Pearson TDA validation
python notebooks/03_evaluation.py
```

---

## Citation

```bibtex
@article{kasamala2025travel,
  title   = {Travel Time Prediction Using Various Time Series Feature Generation Techniques},
  author  = {Kasamala, Nancy and Comert, Gurcan and others},
  journal = {IEEE Access},
  volume  = {13},
  year    = {2025},
  doi     = {xxx/xxxxx}
}
```

---

## Funding

This research was partly funded by:
- U.S. Department of Education HBCU Master's Program Grant (P120A210048)
- U.S. Department of Transportation University Transportation Centers Program (SCSU)
- NSF Grants: 2131080, 2242812, 2200457, 2234920, 2305470
