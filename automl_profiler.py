"""
dataset_profiler.py
-------------------
Phase 1 of the AutoML Trend Forecasting system.

Takes any pandas DataFrame and returns a structured DatasetProfile
object that the meta-feature extractor (Phase 2) will consume.

Usage:
    from profiler import profile_dataset
    import pandas as pd

    df = pd.read_csv("your_data.csv")
    report = profile_dataset(df, target_col="sales")
    print(report)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────

@dataclass
class ColumnProfile:
    name: str
    dtype: str                        # "numeric_continuous" | "numeric_discrete" | "categorical" | "datetime" | "boolean" | "text" | "constant"
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    outlier_ratio: Optional[float] = None   # IQR-based
    top_values: Optional[list] = None       # for categoricals


@dataclass
class TemporalProfile:
    datetime_col: str
    inferred_frequency: Optional[str]       # e.g. "D", "MS", "H"
    n_periods: int
    has_gaps: bool
    gap_count: int
    adf_pvalue: Optional[float]             # Augmented Dickey-Fuller
    kpss_pvalue: Optional[float]            # KPSS test
    is_stationary: Optional[bool]           # True if ADF rejects unit root AND KPSS doesn't
    trend_strength: Optional[float]         # from STL decomposition (0-1)
    seasonality_strength: Optional[float]   # from STL decomposition (0-1)


@dataclass
class CorrelationProfile:
    high_correlation_pairs: list            # [(col_a, col_b, pearson_r), ...]  |r| > 0.85
    vif_scores: dict                        # {col: vif} — VIF > 5 = multicollinearity concern
    target_correlations: dict               # {col: pearson_r with target}


@dataclass
class TargetProfile:
    col: str
    type: str                               # "continuous" | "binary" | "multiclass" | "unknown"
    missing_ratio: float
    skewness: Optional[float] = None
    class_balance: Optional[dict] = None    # {label: ratio} for classification
    leakage_risk_cols: list = field(default_factory=list)  # cols with |r| > 0.95 with target


@dataclass
class DatasetProfile:
    # ── shape ──────────────────────────────
    n_rows: int
    n_cols: int
    memory_mb: float

    # ── column profiles ────────────────────
    columns: list[ColumnProfile]

    # ── global quality ─────────────────────
    total_missing_ratio: float
    duplicate_row_ratio: float
    constant_col_names: list[str]
    near_constant_col_names: list[str]      # >95% same value

    # ── temporal ──────────────────────────
    is_temporal: bool
    temporal: Optional[TemporalProfile]

    # ── correlations ──────────────────────
    correlation: Optional[CorrelationProfile]

    # ── target ────────────────────────────
    target: Optional[TargetProfile]

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Dataset: {self.n_rows} rows × {self.n_cols} cols  ({self.memory_mb:.1f} MB)",
            f"Missing: {self.total_missing_ratio*100:.1f}%   Duplicates: {self.duplicate_row_ratio*100:.1f}%",
        ]
        if self.constant_col_names:
            lines.append(f"Constant cols (drop these): {self.constant_col_names}")
        if self.near_constant_col_names:
            lines.append(f"Near-constant cols: {self.near_constant_col_names}")
        if self.is_temporal and self.temporal:
            t = self.temporal
            lines.append(
                f"Temporal: '{t.datetime_col}'  freq={t.inferred_frequency}  "
                f"stationary={t.is_stationary}  "
                f"ADF_p={t.adf_pvalue:.3f}  KPSS_p={t.kpss_pvalue:.3f}"
            )
        if self.target:
            tgt = self.target
            lines.append(f"Target: '{tgt.col}'  type={tgt.type}  missing={tgt.missing_ratio*100:.1f}%")
            if tgt.leakage_risk_cols:
                lines.append(f"  ⚠ Leakage risk: {tgt.leakage_risk_cols}")
        if self.correlation:
            high = self.correlation.high_correlation_pairs
            if high:
                pairs_str = ", ".join(f"{a}↔{b}({r:.2f})" for a, b, r in high[:5])
                lines.append(f"High correlations: {pairs_str}")
            risky_vif = {k: v for k, v in self.correlation.vif_scores.items() if v > 5}
            if risky_vif:
                lines.append(f"Multicollinearity (VIF>5): {risky_vif}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _infer_dtype(series: pd.Series) -> str:
    if series.nunique() <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        # heuristic: discrete if few uniques relative to n
        if series.nunique() / len(series) < 0.05 and series.nunique() < 20:
            return "numeric_discrete"
        return "numeric_continuous"
    # object columns
    if series.nunique() == 2:
        return "boolean"
    avg_len = series.dropna().astype(str).str.len().mean()
    if avg_len > 50 or series.nunique() / len(series) > 0.9:
        return "text"
    return "categorical"


def _outlier_ratio_iqr(series: pd.Series) -> float:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
    return float(mask.sum() / len(series))


def _profile_column(series: pd.Series) -> ColumnProfile:
    dtype = _infer_dtype(series)
    n = len(series)
    missing = series.isna().sum()
    unique = series.nunique()
    base = dict(
        name=series.name,
        dtype=dtype,
        missing_count=int(missing),
        missing_ratio=float(missing / n),
        unique_count=int(unique),
        unique_ratio=float(unique / n),
    )
    if dtype in ("numeric_continuous", "numeric_discrete"):
        num = series.dropna()
        base.update(
            skewness=float(stats.skew(num)) if len(num) > 2 else None,
            kurtosis=float(stats.kurtosis(num)) if len(num) > 2 else None,
            mean=float(num.mean()),
            std=float(num.std()),
            min=float(num.min()),
            max=float(num.max()),
            outlier_ratio=_outlier_ratio_iqr(num),
        )
    elif dtype in ("categorical", "boolean", "numeric_discrete"):
        vc = series.value_counts(normalize=True).head(5)
        base["top_values"] = list(zip(vc.index.tolist(), vc.values.round(3).tolist()))
    return ColumnProfile(**base)


def _detect_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    # try parsing object columns
    for col in df.select_dtypes(include="object").columns:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True)
            if parsed.notna().mean() > 0.9:
                return col
        except Exception:
            pass
    return None  


def _profile_temporal(df: pd.DataFrame, dt_col: str, target_col: Optional[str]) -> TemporalProfile:
    series_dt = pd.to_datetime(df[dt_col])
    series_dt_sorted = series_dt.sort_values()

    # Infer frequency
    try:
        freq = pd.infer_freq(series_dt_sorted)
    except Exception:
        freq = None

    # Gap detection
    diffs = series_dt_sorted.diff().dropna()
    mode_diff = diffs.mode()[0] if len(diffs) > 0 else None
    gap_count = int((diffs > mode_diff * 1.5).sum()) if mode_diff else 0

    # Stationarity tests on target (or first numeric col)
    adf_p = kpss_p = is_stat = None
    ts_col = target_col or df.select_dtypes(include=np.number).columns[0] if len(df.select_dtypes(include=np.number).columns) else None
    if ts_col and ts_col in df.columns:
        ts = df.set_index(dt_col)[ts_col].dropna().sort_index()
        if len(ts) >= 10:
            try:
                adf_p = float(adfuller(ts)[1])
            except Exception:
                pass
            try:
                kpss_p = float(kpss(ts, regression="c", nlags="auto")[1])
            except Exception:
                pass
            if adf_p is not None and kpss_p is not None:
                is_stat = bool(adf_p < 0.05 and kpss_p > 0.05)

    # STL decomposition for trend/seasonality strength
    trend_str = season_str = None
    if ts_col and ts_col in df.columns and freq in ("D", "MS", "M", "QS", "Q", "H", "W"):
        try:
            from statsmodels.tsa.seasonal import STL
            ts = df.set_index(dt_col)[ts_col].dropna().sort_index()
            period = {"D": 7, "MS": 12, "M": 12, "QS": 4, "Q": 4, "H": 24, "W": 52}.get(freq, 7)
            if len(ts) >= period * 2:
                stl = STL(ts, period=period, robust=True).fit()
                var_total = ts.var()
                trend_str = float(1 - stl.resid.var() / (stl.trend + stl.resid).var()) if var_total > 0 else 0.0
                season_str = float(1 - stl.resid.var() / (stl.seasonal + stl.resid).var()) if var_total > 0 else 0.0
                trend_str = max(0.0, min(1.0, trend_str))
                season_str = max(0.0, min(1.0, season_str))
        except Exception:
            pass

    return TemporalProfile(
        datetime_col=dt_col,
        inferred_frequency=freq,
        n_periods=len(series_dt),
        has_gaps=gap_count > 0,
        gap_count=gap_count,
        adf_pvalue=adf_p,
        kpss_pvalue=kpss_p,
        is_stationary=is_stat,
        trend_strength=trend_str,
        seasonality_strength=season_str,
    )


def _profile_correlations(df: pd.DataFrame, target_col: Optional[str]) -> CorrelationProfile:
    num_df = df.select_dtypes(include=np.number).dropna()
    if num_df.empty:
        return CorrelationProfile([], {}, {})

    corr = num_df.corr(method="pearson")

    # High correlation pairs (upper triangle, |r| > 0.85, excluding target)
    high_pairs = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            if a == target_col or b == target_col:
                continue
            r = corr.loc[a, b]
            if abs(r) > 0.85:
                high_pairs.append((a, b, round(float(r), 3)))

    # VIF (multicollinearity) — exclude target, need at least 2 feature cols
    feature_cols = [c for c in num_df.columns if c != target_col]
    vif_scores = {}
    if len(feature_cols) >= 2:
        X = num_df[feature_cols].dropna()
        if len(X) > len(feature_cols):
            try:
                for i, col in enumerate(feature_cols):
                    vif_scores[col] = round(float(variance_inflation_factor(X.values, i)), 2)
            except Exception:
                pass

    # Target correlations
    target_corr = {}
    if target_col and target_col in num_df.columns:
        tc = corr[target_col].drop(target_col)
        target_corr = {c: round(float(v), 3) for c, v in tc.items()}

    return CorrelationProfile(
        high_correlation_pairs=high_pairs,
        vif_scores=vif_scores,
        target_correlations=target_corr,
    )


def _profile_target(df: pd.DataFrame, target_col: str) -> TargetProfile:
    series = df[target_col]
    missing_ratio = float(series.isna().mean())
    dtype = _infer_dtype(series)

    # Numeric with only 2 unique non-null values → binary classification target
    is_binary_int = (
        dtype in ("numeric_discrete", "numeric_continuous")
        and series.nunique() == 2
    )

    if is_binary_int or dtype in ("boolean", "categorical"):
        vc = series.value_counts(normalize=True)
        t_type = "binary" if len(vc) == 2 else "multiclass"
        class_balance = {str(k): round(float(v), 3) for k, v in vc.items()}
        skewness = None
    elif dtype in ("numeric_continuous", "numeric_discrete"):
        t_type = "continuous"
        skewness = float(stats.skew(series.dropna())) if series.notna().sum() > 2 else None
        class_balance = None
    else:
        t_type = "unknown"
        skewness = None
        class_balance = None

    # Leakage risk: numeric cols with |pearson| > 0.95 with target
    leakage = []
    if pd.api.types.is_numeric_dtype(series):
        for col in df.select_dtypes(include=np.number).columns:
            if col == target_col:
                continue
            try:
                r, _ = stats.pearsonr(df[col].dropna(), series[df[col].notna()])
                if abs(r) > 0.95:
                    leakage.append(col)
            except Exception:
                pass

    return TargetProfile(
        col=target_col,
        type=t_type,
        missing_ratio=missing_ratio,
        skewness=skewness,
        class_balance=class_balance,
        leakage_risk_cols=leakage,
    )


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def profile_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    datetime_col: Optional[str] = None,
) -> DatasetProfile:
    """
    Profile a dataset and return a DatasetProfile.

    Parameters
    ----------
    df          : Input DataFrame
    target_col  : The column the model will predict (optional but recommended)
    datetime_col: Override auto-detection of the datetime column
    """
    n_rows, n_cols = df.shape
    memory_mb = round(df.memory_usage(deep=True).sum() / 1e6, 3)

    # Column profiles
    col_profiles = [_profile_column(df[c]) for c in df.columns]

    # Quality
    total_missing = float(df.isna().mean().mean())
    dup_ratio = float(df.duplicated().mean())
    constant_cols = [c.name for c in col_profiles if c.dtype == "constant"]
    near_constant = []
    for col in df.columns:
        vc = df[col].value_counts(normalize=True)
        if len(vc) > 0 and vc.iloc[0] > 0.95 and col not in constant_cols:
            near_constant.append(col)

    # Temporal
    dt_col = datetime_col or _detect_datetime_col(df)
    is_temporal = dt_col is not None
    temporal_profile = _profile_temporal(df, dt_col, target_col) if is_temporal else None

    # Correlations
    corr_profile = _profile_correlations(df, target_col)

    # Target
    target_profile = _profile_target(df, target_col) if target_col else None

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        memory_mb=memory_mb,
        columns=col_profiles,
        total_missing_ratio=total_missing,
        duplicate_row_ratio=dup_ratio,
        constant_col_names=constant_cols,
        near_constant_col_names=near_constant,
        is_temporal=is_temporal,
        temporal=temporal_profile,
        correlation=corr_profile,
        target=target_profile,
    )