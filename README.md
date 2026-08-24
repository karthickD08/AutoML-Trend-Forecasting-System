# AutoML Trend Forecasting System

A research-grade AutoML system that profiles any dataset, extracts meta-features, recommends the best ML algorithm using a trained meta-learner, and explains every decision with SHAP attribution and confidence intervals.

Built as a portfolio project demonstrating meta-learning, AutoML internals, and production-quality ML engineering — not just sklearn wrappers.

---

## What makes this different from just using AutoML

Most AutoML tools (Auto-sklearn, H2O, TPOT) are black boxes. This system exposes the meta-learning layer that sits inside them:

- **Meta-feature extraction** — 41 numeric signals extracted from every dataset to characterise its structure
- **Trained meta-learner** — an XGBoost classifier trained on OpenML benchmark data that maps those 41 signals to the best algorithm
- **Explainability** — SHAP values, bootstrap confidence intervals, and evidence-backed reasoning strings per recommendation

---

## Architecture

```
Dataset (CSV)
     │
     ▼
┌─────────────────────────────┐
│   Phase 1 — profiler.py     │  Schema validation, type inference,
│   DatasetProfile            │  stationarity tests, correlation audit
└──────────────┬──────────────┘
               │ DatasetProfile dict
               ▼
┌─────────────────────────────┐
│ Phase 2 — meta_extractor.py │  41-feature meta-vector:
│ MetaFeatureVector           │  Statistical + Model-based + Landmarking
└──────────────┬──────────────┘
               │ np.ndarray (41,)
               ▼
┌──────────────────────────────────┐
│ Phase 3 — algorithm_recommender  │  XGBoost meta-classifier trained on
│ AlgorithmRecommendation[]        │  OpenML benchmarks → ranked algorithms
└──────────────┬───────────────────┘
               │ top-k algorithms + reasoning
               ▼
┌─────────────────────────────┐
│ Phase 4a — explainability   │  SHAP values, 5-fold CV, bootstrap CI
│ ExplainBundle               │  per recommended algorithm
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Phase 4b — app.py           │  4-tab Streamlit UI
│ Streamlit UI                │  Profile · Meta-features · Recommend · Explain
└─────────────────────────────┘
```

---

## Project structure

```
.
├── profiler.py                # Phase 1 — dataset profiler
├── meta_extractor.py          # Phase 2 — meta-feature extractor
├── algorithm_recommender.py   # Phase 3 — meta-learner + recommender
├── explainability.py          # Phase 4a — SHAP + CI engine
├── app.py                     # Phase 4b — Streamlit UI
├── meta_dataset.csv           # (generated) OpenML benchmark meta-dataset
├── recommender.pkl            # (generated) trained meta-learner bundle
└── README.md
```

---

## Installation

```bash
pip install pandas numpy scipy statsmodels scikit-learn xgboost shap \
            streamlit plotly openml
```

Python 3.9+ recommended.

---

## Quick start

### 1. Build the meta-dataset (one-time, ~20 min)

Downloads 60 datasets from OpenML, benchmarks 8 algorithms on each, extracts meta-features, saves a CSV.

```python
from algorithm_recommender import build_meta_dataset
build_meta_dataset(n_datasets=60, output_path="meta_dataset.csv")
```

### 2. Train the recommender (one-time, ~1 min)

```python
from algorithm_recommender import train_recommender
train_recommender(meta_csv="meta_dataset.csv", model_path="recommender.pkl")
```

### 3. Run the Streamlit UI

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Or use the Python API directly

```python
import pandas as pd
from profiler import profile_dataset
from meta_extractor import extract_meta_features
from algorithm_recommender import recommend
from explainability import ExplainabilityEngine

df = pd.read_csv("your_data.csv")

# Profile
profile = profile_dataset(df, target_col="sales", datetime_col="date")
print(profile.summary())

# Meta-features
mf = extract_meta_features(df, profile, target_col="sales")
print(mf.summary())

# Recommend
results, profile, mf = recommend(df, target_col="sales",
                                  model_path="recommender.pkl", top_k=5)
for r in results:
    print(r)

# Explain top recommendation
engine = ExplainabilityEngine(df, target_col="sales")
engine.fit(results[0].algorithm)
bundle = engine.explain()
print(bundle.summary())
```

---

## Demo mode (no OpenML required)

If you want to test without building the real meta-dataset:

```python
from algorithm_recommender import _build_synthetic_meta_dataset, train_recommender
_build_synthetic_meta_dataset(n=150, path="meta_dataset_synthetic.csv")
train_recommender(meta_csv="meta_dataset_synthetic.csv", model_path="recommender.pkl")
```

The Streamlit UI also has a one-click **"Train demo recommender"** button in Tab 3.

---

## The 41 meta-features

### Family 1 — Statistical (26 features)
Extracted directly from the dataset profile. Zero model training required.

| Group | Features |
|---|---|
| Size | n_rows, n_cols, log_n_rows, log_n_cols, rows_cols_ratio |
| Quality | missing_ratio, duplicate_ratio, numeric_ratio, categorical_ratio, temporal_present |
| Distribution | skewness_mean, skewness_std, kurtosis_mean, kurtosis_std, outlier_ratio_mean |
| Target | target_skewness, class_imbalance, n_classes |
| Correlation | corr_mean_abs, corr_max_abs, high_corr_pair_count, vif_max |
| Temporal | is_stationary, adf_pvalue, trend_strength, seasonality_strength |

### Family 2 — Model-based (7 features)
Trains a Decision Tree and a Linear model, records performance and structural signals.

| Feature | Meaning |
|---|---|
| dt_cv_score | Decision Tree 3-fold CV score |
| linear_cv_score | Linear model 3-fold CV score |
| dt_linear_gap | DT minus Linear score — positive = data rewards non-linearity |
| dt_depth_used | Actual depth of fitted DT — complexity signal |
| linear_coef_std | Std of linear coefficients — feature spread |

### Family 3 — Landmarking (8 features)
Trains 5 fast landmark algorithms, records performance relative to a dummy baseline.

| Landmark | What it measures |
|---|---|
| lm_baseline | Dummy majority/mean predictor — always 0 (anchor) |
| lm_1nn | 1-nearest neighbour — local structure in data |
| lm_naive_bayes | Gaussian NB — feature independence assumption |
| lm_tiny_dt | Decision Tree depth=2 — simple split signal |
| lm_linear | Logistic Regression / Ridge — linear separability |
| lm_nonlinear_advantage | lm_1nn minus lm_linear — key routing signal |
| lm_best | Highest relative landmark score |
| lm_spread | Std of all landmark scores — dataset difficulty |

> **Note:** "Landmarking" here is a meta-learning term. It has nothing to do with facial landmark detection in computer vision. These are reference algorithm scores used to characterise a dataset's behaviour.

---

## The 8 candidate algorithms

| Algorithm | Best when |
|---|---|
| **Random Forest** | General-purpose tabular data, robust to outliers and redundant features |
| **XGBoost** | Large structured datasets with complex non-linear interactions |
| **Gradient Boosting** | Small-to-medium datasets needing high accuracy |
| **Logistic Regression / Ridge** | Approximately linear relationships, fast inference needed |
| **Lasso** | High multicollinearity, automatic feature selection via L1 sparsity |
| **KNN** | Low-dimensional data with strong local structure |
| **Decision Tree** | High interpretability required, simple decision boundaries |
| **SVM** | High-dimensional data, smaller datasets |

---

## The meta-learning pipeline in detail

```
Phase 1: profile_dataset(df, target_col)
  └─ Runs ADF + KPSS stationarity tests
  └─ STL decomposition for trend/seasonality strength
  └─ VIF multicollinearity scores
  └─ IQR outlier detection per column
  └─ Returns: DatasetProfile dataclass

Phase 2: extract_meta_features(df, profile, target_col)
  └─ Statistical: reads from profile dict (free)
  └─ Model-based: trains DT + LinearModel, records CV scores
  └─ Landmarking: trains 5 fast models, computes relative scores
  └─ Returns: MetaFeatureVector (41 floats)

Phase 3: recommend(df, target_col, model_path)
  └─ Loads trained XGBoost meta-classifier from .pkl
  └─ Feeds 41-feature vector → predict_proba → ranked algorithms
  └─ Loads per-algorithm score regressors → estimated CV score
  └─ Builds evidence-backed reasoning strings per recommendation
  └─ Returns: List[AlgorithmRecommendation]

Phase 4: ExplainabilityEngine(df, target_col).fit(algo).explain()
  └─ SHAP: TreeExplainer for tree models, KernelExplainer for others
  └─ CV: 5-fold StratifiedKFold (classification) or KFold (regression)
  └─ CI: 50-iteration bootstrap resampling → 5th–95th percentile bands
  └─ Returns: ExplainBundle
```

---

## Interview answer: why not just use AutoML?

> "I built this to understand what's inside AutoML frameworks — specifically the meta-learning layer that most engineers treat as a black box. I implemented meta-feature extraction from scratch across three families (statistical, model-based, landmarking), trained a meta-learner on OpenML benchmark data, and added explainability that commercial AutoML tools strip out. The goal wasn't to replicate AutoML — it was to understand and expose its internals."

---

## Comparable systems

| System | Similarity |
|---|---|
| Auto-sklearn | Same meta-learning approach, uses OpenML meta-dataset |
| TPOT | Algorithm + pipeline selection via genetic programming |
| H2O AutoML | Production-scale stacked ensemble selection |
| Google Vertex AutoML | Cloud-scale version of the same problem |

This project sits in the same **problem space** at a smaller scope, with the advantage of full transparency into the selection logic.

---

## Extending the system

**Add more algorithms to the registry** — edit `ALGORITHMS` dict in `algorithm_recommender.py`. The meta-learner will include them in the benchmark and learn when to recommend them.

**Expand the meta-dataset** — increase `n_datasets` in `build_meta_dataset()`. More datasets = better meta-learner accuracy. 200+ datasets is production-quality.

**Add time-series specific algorithms** — ARIMA, Prophet, LSTM, TFT can be added to the forecasting engine as a Phase 5 layer that runs after the recommender routes toward temporal algorithms.

**Replace XGBoost meta-learner** — swap in a neural network or Gaussian process for the meta-classifier if you want uncertainty estimates on the recommendation itself.

---

## Tech stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy |
| Statistics | scipy, statsmodels |
| ML | scikit-learn, xgboost |
| Explainability | shap |
| Meta-dataset | openml |
| UI | streamlit, plotly |
