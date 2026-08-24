"""
meta_extractor.py
-----------------
Phase 2 of the AutoML Trend Forecasting system.

Takes a DatasetProfile (from Phase 1) + the raw DataFrame and produces
a flat MetaFeatureVector — a fixed-length numeric vector the algorithm
recommender (Phase 3) will train and predict on.

Three families of meta-features:
  1. Statistical   — cheap, from the profile dict directly
  2. Model-based   — fast DT + LR trained, extract accuracy & coef stats
  3. Landmarking   — 5 landmark algorithms, relative CV performance scores

Usage:
    from profiler import profile_dataset
    from meta_extractor import extract_meta_features

    profile = profile_dataset(df, target_col="sales")
    mf = extract_meta_features(df, profile, target_col="sales")
    print(mf.to_dict())
    print(mf.summary())
"""

from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Output schema  (fixed-length numeric vector)
# ──────────────────────────────────────────────────────────────

@dataclass
class MetaFeatureVector:
    # 1. Statistical
    n_rows: float
    n_cols: float
    log_n_rows: float
    log_n_cols: float
    rows_cols_ratio: float

    missing_ratio: float
    duplicate_ratio: float
    numeric_ratio: float
    categorical_ratio: float
    temporal_present: float

    skewness_mean: float
    skewness_std: float
    kurtosis_mean: float
    kurtosis_std: float
    outlier_ratio_mean: float

    target_skewness: float
    class_imbalance: float
    n_classes: float

    corr_mean_abs: float
    corr_max_abs: float
    high_corr_pair_count: float
    vif_max: float

    is_stationary: float        # 1=yes  -1=no  0=unknown/not temporal
    adf_pvalue: float
    trend_strength: float
    seasonality_strength: float

    # 2. Model-based
    dt_cv_score: float
    dt_train_time: float
    linear_cv_score: float
    linear_train_time: float
    dt_linear_gap: float        # positive = data rewards non-linearity
    dt_depth_used: float
    linear_coef_std: float

    # 3. Landmarking (all relative to dummy baseline)
    lm_1nn: float
    lm_naive_bayes: float
    lm_tiny_dt: float
    lm_linear: float
    lm_baseline: float          # always 0 (anchor)
    lm_best: float
    lm_nonlinear_advantage: float   # lm_1nn - lm_linear
    lm_spread: float                # std of landmark scores = difficulty signal

    def to_dict(self) -> dict:
        return asdict(self)

    def to_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=float)

    def feature_names(self) -> list:
        return list(asdict(self).keys())

    def summary(self) -> str:
        lines = ["── Meta-feature summary ──────────────────────────────────"]
        lines.append(f"Dataset      : {int(self.n_rows)} rows x {int(self.n_cols)} cols")
        lines.append(f"Missing      : {self.missing_ratio*100:.1f}%   Duplicates: {self.duplicate_ratio*100:.1f}%")
        lines.append(f"Numeric cols : {self.numeric_ratio*100:.0f}%   Categorical: {self.categorical_ratio*100:.0f}%   Temporal: {bool(self.temporal_present)}")
        lines.append(f"Skewness     : mean={self.skewness_mean:.2f}  std={self.skewness_std:.2f}   Outliers(mean): {self.outlier_ratio_mean*100:.1f}%")
        lines.append(f"Correlation  : mean|r|={self.corr_mean_abs:.2f}  max|r|={self.corr_max_abs:.2f}  high_pairs={int(self.high_corr_pair_count)}  VIF_max={self.vif_max:.1f}")
        lines.append(f"Temporal     : stationary={self.is_stationary:+.0f}  ADF_p={self.adf_pvalue:.3f}  trend={self.trend_strength:.2f}  season={self.seasonality_strength:.2f}")
        lines.append(f"Target       : skew={self.target_skewness:.2f}  n_classes={int(self.n_classes)}  minority_ratio={self.class_imbalance:.3f}")
        lines.append(f"Model-based  : DT={self.dt_cv_score:.3f} ({self.dt_train_time:.2f}s)  Linear={self.linear_cv_score:.3f} ({self.linear_train_time:.2f}s)  gap={self.dt_linear_gap:+.3f}")
        lines.append(f"Landmarking  : 1NN={self.lm_1nn:.3f}  NB={self.lm_naive_bayes:.3f}  tinyDT={self.lm_tiny_dt:.3f}  Linear={self.lm_linear:.3f}")
        lines.append(f"             : best={self.lm_best:.3f}  nonlinear_adv={self.lm_nonlinear_advantage:+.3f}  spread={self.lm_spread:.3f}")

        # Interpretive hints
        lines.append("")
        lines.append("── Signals for algorithm recommender ─────────────────────")
        if self.is_stationary == -1:
            lines.append("  ⚠  Non-stationary series  →  ARIMA/SARIMA not preferred; consider XGBoost+lags or LSTM")
        if self.is_stationary == 1:
            lines.append("  ✓  Stationary series  →  ARIMA/SARIMA are valid candidates")
        if self.seasonality_strength > 0.4:
            lines.append(f"  ✓  Strong seasonality ({self.seasonality_strength:.2f})  →  Prophet, SARIMA, or seasonal decomposition")
        if self.trend_strength > 0.5:
            lines.append(f"  ✓  Strong trend ({self.trend_strength:.2f})  →  differencing or detrending recommended")
        if self.lm_nonlinear_advantage > 0.05:
            lines.append(f"  ✓  Non-linear advantage={self.lm_nonlinear_advantage:+.3f}  →  tree/ensemble models preferred over linear")
        if self.lm_nonlinear_advantage < -0.05:
            lines.append(f"  ✓  Linear advantage={-self.lm_nonlinear_advantage:+.3f}  →  Ridge/ARIMA likely competitive")
        if self.class_imbalance > 0 and self.class_imbalance < 0.2:
            lines.append(f"  ⚠  Class imbalance (minority={self.class_imbalance:.2f})  →  use class_weight='balanced' or SMOTE")
        if self.high_corr_pair_count > 2:
            lines.append(f"  ⚠  {int(self.high_corr_pair_count)} high-corr pairs  →  consider PCA or regularised models")
        if self.vif_max > 10:
            lines.append(f"  ⚠  VIF_max={self.vif_max:.1f}  →  multicollinearity; Ridge/Lasso preferred over plain LR")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

def _prepare_xy(df: pd.DataFrame, target_col: str, max_rows: int = 5000):
    df = df.copy()
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    try:
        dt_cols += df.select_dtypes(include=["datetimetz"]).columns.tolist()
    except Exception:
        pass
    drop = list(set(dt_cols + [c for c in df.columns if df[c].nunique() <= 1 and c != target_col]))
    df = df.drop(columns=[c for c in drop if c in df.columns])

    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    for col in X_raw.columns:
        if X_raw[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_raw[col]):
                X_raw[col].fillna(X_raw[col].median(), inplace=True)
            else:
                X_raw[col].fillna(X_raw[col].mode()[0], inplace=True)

    for col in X_raw.select_dtypes(include=["object", "bool", "category"]).columns:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    y_raw = y_raw.fillna(y_raw.mode()[0] if not pd.api.types.is_numeric_dtype(y_raw) else y_raw.median())

    if y_raw.nunique() <= 20 and not pd.api.types.is_float_dtype(y_raw):
        task = "classification"
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        task = "regression"
        y = y_raw.values.astype(float)

    X = X_raw.values.astype(float)
    if len(X) > max_rows:
        idx = np.random.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]

    return X, y, task


def _cv(estimator, X, y, task, cv=3):
    scoring = "accuracy" if task == "classification" else "r2"
    # Wrap in an imputer pipeline to guard against any residual NaNs
    safe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("est", estimator)])
    t0 = time.time()
    scores = cross_val_score(safe, X, y, cv=cv, scoring=scoring, error_score=0.0)
    return float(np.mean(scores)), time.time() - t0


def _relative(score, baseline):
    return float(np.clip(score - baseline, -1.0, 1.0))


# ──────────────────────────────────────────────────────────────
# Family 1 — Statistical
# ──────────────────────────────────────────────────────────────

def _statistical(profile) -> dict:
    p = profile
    numeric_types = {"numeric_continuous", "numeric_discrete"}
    cat_types = {"categorical", "boolean"}
    dtypes = [c.dtype for c in p.columns]
    total = max(p.n_cols, 1)

    n_numeric = sum(1 for d in dtypes if d in numeric_types)
    n_cat     = sum(1 for d in dtypes if d in cat_types)

    skews   = [abs(c.skewness)    for c in p.columns if c.skewness   is not None]
    kurts   = [c.kurtosis         for c in p.columns if c.kurtosis   is not None]
    outliers= [c.outlier_ratio    for c in p.columns if c.outlier_ratio is not None]

    corr = p.correlation
    high_pairs = len(corr.high_correlation_pairs) if corr else 0
    abs_corrs  = [abs(r) for r in (corr.target_correlations.values() if corr else []) if not np.isnan(r)]
    vif_vals   = list(corr.vif_scores.values()) if corr else []

    t = p.temporal
    is_stat   = (1.0  if (t and t.is_stationary is True)
                 else -1.0 if (t and t.is_stationary is False)
                 else 0.0)
    adf_p     = float(t.adf_pvalue)        if (t and t.adf_pvalue       is not None) else 1.0
    trend_s   = float(t.trend_strength)    if (t and t.trend_strength    is not None) else 0.0
    season_s  = float(t.seasonality_strength) if (t and t.seasonality_strength is not None) else 0.0

    tgt = p.target
    tgt_skew  = float(tgt.skewness) if (tgt and tgt.skewness is not None) else 0.0
    n_classes = 0.0
    imbalance = 0.0
    if tgt and tgt.class_balance:
        n_classes = float(len(tgt.class_balance))
        imbalance = float(min(tgt.class_balance.values()))

    return dict(
        n_rows=float(p.n_rows),
        n_cols=float(p.n_cols),
        log_n_rows=float(np.log1p(p.n_rows)),
        log_n_cols=float(np.log1p(p.n_cols)),
        rows_cols_ratio=float(p.n_rows / total),
        missing_ratio=float(p.total_missing_ratio),
        duplicate_ratio=float(p.duplicate_row_ratio),
        numeric_ratio=float(n_numeric / total),
        categorical_ratio=float(n_cat / total),
        temporal_present=float(p.is_temporal),
        skewness_mean=float(np.mean(skews))    if skews    else 0.0,
        skewness_std=float(np.std(skews))      if skews    else 0.0,
        kurtosis_mean=float(np.mean(kurts))    if kurts    else 0.0,
        kurtosis_std=float(np.std(kurts))      if kurts    else 0.0,
        outlier_ratio_mean=float(np.mean(outliers)) if outliers else 0.0,
        target_skewness=tgt_skew,
        class_imbalance=imbalance,
        n_classes=n_classes,
        corr_mean_abs=float(np.mean(abs_corrs)) if abs_corrs else 0.0,
        corr_max_abs=float(np.max(abs_corrs))   if abs_corrs else 0.0,
        high_corr_pair_count=float(high_pairs),
        vif_max=float(min(max(vif_vals), 1000.0)) if vif_vals else 0.0,
        is_stationary=is_stat,
        adf_pvalue=adf_p,
        trend_strength=trend_s,
        seasonality_strength=season_s,
    )


# ──────────────────────────────────────────────────────────────
# Family 2 — Model-based
# ──────────────────────────────────────────────────────────────

def _model_based(X, y, task) -> dict:
    if task == "classification":
        dt     = DecisionTreeClassifier(max_depth=5, random_state=42)
        linear = Pipeline([("s", StandardScaler()),
                           ("m", LogisticRegression(max_iter=300, random_state=42))])
    else:
        dt     = DecisionTreeRegressor(max_depth=5, random_state=42)
        linear = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])

    dt_score,  dt_time  = _cv(dt,     X, y, task)
    lin_score, lin_time = _cv(linear, X, y, task)

    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X_clean = imp.fit_transform(X)
    dt.fit(X_clean, y)
    linear.fit(X_clean, y)
    dt_depth = float(dt.get_depth())
    coef = getattr(linear.named_steps["m"], "coef_", np.array([[0.0]])).flatten()
    coef_std = float(np.std(coef))

    return dict(
        dt_cv_score=dt_score,
        dt_train_time=dt_time,
        linear_cv_score=lin_score,
        linear_train_time=lin_time,
        dt_linear_gap=dt_score - lin_score,
        dt_depth_used=dt_depth,
        linear_coef_std=coef_std,
    )


# ──────────────────────────────────────────────────────────────
# Family 3 — Landmarking
# ──────────────────────────────────────────────────────────────

def _landmarking(X, y, task) -> dict:
    if task == "classification":
        models = {
            "baseline":    DummyClassifier(strategy="most_frequent"),
            "1nn":         KNeighborsClassifier(n_neighbors=1),
            "naive_bayes": GaussianNB(),
            "tiny_dt":     DecisionTreeClassifier(max_depth=2, random_state=42),
            "linear":      Pipeline([("s", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=200, random_state=42))]),
        }
    else:
        models = {
            "baseline":    DummyRegressor(strategy="mean"),
            "1nn":         KNeighborsRegressor(n_neighbors=1),
            "naive_bayes": KNeighborsRegressor(n_neighbors=3),  # NB N/A for regression
            "tiny_dt":     DecisionTreeRegressor(max_depth=2, random_state=42),
            "linear":      Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        }

    raw = {name: _cv(m, X, y, task, cv=3)[0] for name, m in models.items()}
    baseline = raw["baseline"]
    rel = {k: _relative(v, baseline) for k, v in raw.items()}

    scores = [rel["1nn"], rel["naive_bayes"], rel["tiny_dt"], rel["linear"]]
    return dict(
        lm_1nn=rel["1nn"],
        lm_naive_bayes=rel["naive_bayes"],
        lm_tiny_dt=rel["tiny_dt"],
        lm_linear=rel["linear"],
        lm_baseline=0.0,
        lm_best=float(max(scores)),
        lm_nonlinear_advantage=rel["1nn"] - rel["linear"],
        lm_spread=float(np.std(scores)),
    )


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def extract_meta_features(
    df: pd.DataFrame,
    profile,
    target_col: str,
    random_state: int = 42,
) -> MetaFeatureVector:
    """
    Extract all three meta-feature families and return a MetaFeatureVector.

    Parameters
    ----------
    df          : Raw DataFrame (same one used for profile_dataset)
    profile     : DatasetProfile returned by profile_dataset()
    target_col  : Column the model will predict
    random_state: Seed for reproducibility
    """
    np.random.seed(random_state)

    stat = _statistical(profile)
    X, y, task = _prepare_xy(df, target_col)
    mb  = _model_based(X, y, task)
    lm  = _landmarking(X, y, task) 

    return MetaFeatureVector(**stat, **mb, **lm)