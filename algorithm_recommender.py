"""
algorithm_recommender.py
------------------------
Phase 3 of the AutoML Trend Forecasting system.

Two responsibilities: 
  A. build_meta_dataset()  — downloads datasets from OpenML, runs a
     benchmark suite (8 algorithms, 5-fold CV), extracts meta-features
     for each dataset, saves a meta-dataset CSV.

  B. train_recommender()   — trains an XGBoost classifier on the
     meta-dataset. Input = 41-feature vector. Output = ranked list of
     algorithms with confidence scores.

  C. recommend()           — inference: given a new DatasetProfile +
     DataFrame, returns a ranked AlgorithmRecommendation list with
     reasoning strings.

Usage:
    # One-time setup (takes ~10-30 min depending on datasets)
    from algorithm_recommender import build_meta_dataset, train_recommender
    build_meta_dataset(n_datasets=60, output_path="meta_dataset.csv")
    train_recommender(meta_csv="meta_dataset.csv", model_path="recommender.pkl")

    # Inference (fast — seconds)
    from algorithm_recommender import recommend
    from profiler import profile_dataset
    results = recommend(df, target_col="sales", model_path="recommender.pkl")
    for r in results:
        print(r)
"""

from __future__ import annotations

import os, json, time, warnings, pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

import openml
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Algorithm registry
# Each entry: (name, clf_class, reg_class, needs_scaling)
# ──────────────────────────────────────────────────────────────

ALGORITHMS = {
    "random_forest": {
        "clf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "reg": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "scale": False,
        "description": "Ensemble of decision trees. Robust to outliers and irrelevant features. Good default for tabular data.",
        "tags": ["non-linear", "ensemble", "handles-missing", "interpretable"],
    },
    "xgboost": {
        "clf": XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"),
        "reg": XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "scale": False,
        "description": "Gradient boosted trees. Best on structured/tabular data with complex interactions.",
        "tags": ["non-linear", "ensemble", "handles-missing", "high-performance"],
    },
    "gradient_boosting": {
        "clf": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "reg": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "scale": False,
        "description": "Sklearn gradient boosting. Slower than XGBoost but sometimes more accurate on small datasets.",
        "tags": ["non-linear", "ensemble"],
    },
    "logistic_regression": {
        "clf": LogisticRegression(max_iter=500, random_state=42),
        "reg": Ridge(alpha=1.0),
        "scale": True,
        "description": "Linear model. Fast, interpretable, works well when relationship is approximately linear.",
        "tags": ["linear", "interpretable", "fast"],
    },
    "lasso": {
        "clf": LogisticRegression(penalty="l1", solver="saga", max_iter=500, random_state=42),
        "reg": Lasso(alpha=0.1, max_iter=1000),
        "scale": True,
        "description": "L1-regularised linear model. Performs automatic feature selection via sparsity.",
        "tags": ["linear", "feature-selection", "interpretable"],
    },
    "knn": {
        "clf": KNeighborsClassifier(n_neighbors=5),
        "reg": KNeighborsRegressor(n_neighbors=5),
        "scale": True,
        "description": "Instance-based learning. Works well on low-dimensional datasets with local structure.",
        "tags": ["non-linear", "instance-based", "no-training"],
    },
    "decision_tree": {
        "clf": DecisionTreeClassifier(max_depth=8, random_state=42),
        "reg": DecisionTreeRegressor(max_depth=8, random_state=42),
        "scale": False,
        "description": "Single decision tree. Highly interpretable. Prone to overfitting on complex data.",
        "tags": ["non-linear", "interpretable", "fast"],
    },
    "svm": {
        "clf": SVC(kernel="rbf", probability=True, random_state=42),
        "reg": SVR(kernel="rbf"),
        "scale": True,
        "description": "Support vector machine. Strong on high-dimensional data. Slow on large datasets.",
        "tags": ["non-linear", "kernel", "high-dimensional"],
    },
}

# ──────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────

@dataclass
class AlgorithmRecommendation:
    rank: int
    algorithm: str
    confidence: float        # probability from meta-learner (0-1)
    predicted_score: float   # estimated CV score on this dataset
    description: str
    reasoning: list          # list of strings explaining why
    tags: list

    def __str__(self):
        lines = [f"#{self.rank}  {self.algorithm.upper()}  (confidence={self.confidence:.2f}  est.score={self.predicted_score:.3f})"]
        lines.append(f"    {self.description}")
        for r in self.reasoning:
            lines.append(f"    → {r}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# A. Build meta-dataset from OpenML
# ──────────────────────────────────────────────────────────────

def _prepare_openml_xy(df: pd.DataFrame, target_col: str, max_rows: int = 3000):
    """Prepare X, y from an OpenML dataset for benchmarking."""
    df = df.copy()
    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # Drop constant and datetime cols
    for col in X_raw.columns:
        if X_raw[col].nunique() <= 1:
            X_raw = X_raw.drop(columns=[col])

    # Encode categoricals
    for col in X_raw.select_dtypes(include=["object", "category", "bool"]).columns:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    # Determine task
    if y_raw.dtype == object or y_raw.nunique() <= 20:
        task = "classification"
        y = LabelEncoder().fit_transform(y_raw.astype(str))
    else:
        task = "regression"
        y = y_raw.values.astype(float)

    X = X_raw.values.astype(float)

    # Subsample
    if len(X) > max_rows:
        idx = np.random.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]

    return X, y, task


def _benchmark_algorithms(X, y, task: str, cv: int = 5) -> dict:
    """
    Run all 8 algorithms with CV. Returns {algo_name: cv_score}.
    Uses a safe pipeline: Imputer → Scaler (if needed) → Model.
    """
    scores = {}
    scoring = "accuracy" if task == "classification" else "r2"
    cv_splitter = (StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                   if task == "classification" else
                   KFold(n_splits=cv, shuffle=True, random_state=42))

    for name, cfg in ALGORITHMS.items():
        model = cfg["clf"] if task == "classification" else cfg["reg"]
        steps = [("imp", SimpleImputer(strategy="median"))]
        if cfg["scale"]:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))
        pipe = Pipeline(steps)

        try:
            s = cross_val_score(pipe, X, y, cv=cv_splitter,
                                scoring=scoring, error_score=0.0)
            scores[name] = float(np.mean(s))
        except Exception as e:
            scores[name] = 0.0

    return scores


def build_meta_dataset(
    n_datasets: int = 60,
    output_path: str = "meta_dataset.csv",
    task_filter: str = "both",   # "classification" | "regression" | "both"
    random_state: int = 42,
):
    """
    Download n_datasets from OpenML, benchmark all 8 algorithms on each,
    extract meta-features, and save a meta-dataset CSV.

    Columns: [meta_feature_1, ..., meta_feature_41, best_algorithm, algo_scores_json]
    """
    from automl_profiler import profile_dataset
    from meta_extracter import extract_meta_features

    np.random.seed(random_state)
    openml.config.apikey = ""  # public datasets don't need a key

    print(f"Fetching dataset list from OpenML...")

    # Get classification datasets
    clf_datasets = openml.datasets.list_datasets(output_format="dataframe")
    clf_datasets = clf_datasets[
        (clf_datasets["NumberOfInstances"] >= 100) &
        (clf_datasets["NumberOfInstances"] <= 10000) &
        (clf_datasets["NumberOfFeatures"] >= 3) &
        (clf_datasets["NumberOfFeatures"] <= 50) &
        (clf_datasets["NumberOfMissingValues"] / (clf_datasets["NumberOfInstances"] * clf_datasets["NumberOfFeatures"]) < 0.3)
    ].sample(frac=1, random_state=random_state)

    dataset_ids = clf_datasets["did"].tolist()[:n_datasets * 2]  # fetch 2x to account for failures

    rows = []
    success = 0
    attempted = 0

    for did in dataset_ids:
        if success >= n_datasets:
            break
        attempted += 1

        try:
            dataset = openml.datasets.get_dataset(
                did,
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            df, y_series, cat_mask, attr_names = dataset.get_data(dataset_format="dataframe")

            if y_series is None or df is None:
                continue

            target_col = dataset.default_target_attribute
            if target_col not in df.columns:
                df[target_col] = y_series

            if len(df) < 80 or df[target_col].nunique() < 2:
                continue

            print(f"  [{success+1}/{n_datasets}] did={did}  '{dataset.name}'  "
                  f"shape={df.shape}  target='{target_col}'", end="  ")

            # Profile + meta-features
            try:
                profile = profile_dataset(df, target_col=target_col)
                mf = extract_meta_features(df, profile, target_col=target_col,
                                           random_state=random_state)
                mf_dict = mf.to_dict()
            except Exception as e:
                print(f"[meta-feature failed: {e}]")
                continue

            # Benchmark
            X, y, task = _prepare_openml_xy(df, target_col)
            algo_scores = _benchmark_algorithms(X, y, task)
            best_algo = max(algo_scores, key=algo_scores.get)

            print(f"best={best_algo} ({algo_scores[best_algo]:.3f})")

            row = mf_dict.copy()
            row["best_algorithm"] = best_algo
            row["task_type"] = task
            row["openml_did"] = did
            row["dataset_name"] = dataset.name
            row["algo_scores"] = json.dumps({k: round(v, 4) for k, v in algo_scores.items()})
            rows.append(row)
            success += 1

        except Exception as e:
            print(f"  did={did} failed: {e}")
            continue

    if not rows:
        raise RuntimeError("No datasets successfully processed. Check your internet connection.")

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(output_path, index=False)
    print(f"\nMeta-dataset saved: {output_path}  ({len(meta_df)} datasets)")
    return meta_df


# ──────────────────────────────────────────────────────────────
# B. Train the recommender
# ──────────────────────────────────────────────────────────────

def train_recommender(
    meta_csv: str = "meta_dataset.csv",
    model_path: str = "recommender.pkl",
):
    """
    Train an XGBoost classifier on the meta-dataset.
    Input  = 41-feature meta-feature vector
    Output = best_algorithm label (one of 8 classes)

    Also trains a score regressor per algorithm to estimate
    predicted CV score on unseen datasets.

    Saves a dict to model_path:
      {
        "clf":           XGBClassifier (predicts best algorithm),
        "score_regs":    {algo_name: XGBRegressor} (predicts CV score),
        "label_encoder": LabelEncoder,
        "feature_names": [list of 41 feature names],
        "classes":       [list of algorithm names],
        "train_accuracy": float,
      }
    """
    from meta_extracter import MetaFeatureVector
    from dataclasses import fields

    meta_df = pd.read_csv(meta_csv)
    print(f"Loaded meta-dataset: {len(meta_df)} rows")

    feature_names = [f.name for f in fields(MetaFeatureVector)]
    X = meta_df[feature_names].values.astype(float)
    y_raw = meta_df["best_algorithm"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Main classifier: predicts best algorithm
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        eval_metric="mlogloss",
    )

    # Cross-validate to report train accuracy
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"Meta-learner 5-fold CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    clf.fit(X, y)

    # Per-algorithm score regressors: predict expected CV score
    score_regs = {}
    for algo in ALGORITHMS:
        if algo not in meta_df.columns:
            # Parse from algo_scores JSON column
            scores_col = meta_df["algo_scores"].apply(
                lambda s: json.loads(s).get(algo, np.nan)
            )
        else:
            scores_col = meta_df[algo]

        y_scores = scores_col.values.astype(float)
        valid = ~np.isnan(y_scores)
        if valid.sum() < 10:
            continue

        reg = XGBRegressor(n_estimators=100, max_depth=3,
                           learning_rate=0.1, random_state=42, verbosity=0)
        reg.fit(X[valid], y_scores[valid])
        score_regs[algo] = reg

    # Feature importances
    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances),
                          key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 meta-features by importance:")
    for fname, imp in top_features:
        print(f"  {fname:30s}  {imp:.4f}")

    # Save
    bundle = {
        "clf": clf,
        "score_regs": score_regs,
        "label_encoder": le,
        "feature_names": feature_names,
        "classes": le.classes_.tolist(),
        "train_accuracy": float(np.mean(cv_scores)),
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nRecommender saved: {model_path}")
    return bundle


# ──────────────────────────────────────────────────────────────
# C. Inference — recommend algorithms for a new dataset
# ──────────────────────────────────────────────────────────────

def _build_reasoning(mf, algo_name: str) -> list:
    """
    Generate human-readable reasoning strings explaining why
    this algorithm was recommended based on meta-features.
    """
    reasons = []
    cfg = ALGORITHMS[algo_name]
    tags = cfg["tags"]

    # Temporal / stationarity
    if mf.temporal_present:
        if mf.is_stationary == 1:
            if "linear" in tags:
                reasons.append(f"Series is stationary (ADF p={mf.adf_pvalue:.3f}) — linear models are valid")
        if mf.is_stationary == -1:
            if "non-linear" in tags:
                reasons.append(f"Non-stationary series (ADF p={mf.adf_pvalue:.3f}) — tree models handle trends well")

    # Seasonality
    if mf.seasonality_strength > 0.4:
        reasons.append(f"Strong seasonality detected ({mf.seasonality_strength:.2f}) — consider adding seasonal features")

    # Linearity signal
    if mf.lm_nonlinear_advantage > 0.05 and "non-linear" in tags:
        reasons.append(f"Non-linear advantage={mf.lm_nonlinear_advantage:+.3f} — dataset rewards non-linear models")
    if mf.lm_nonlinear_advantage < -0.05 and "linear" in tags:
        reasons.append(f"Linear advantage={-mf.lm_nonlinear_advantage:+.3f} — linear models competitive here")

    # Multicollinearity
    if mf.vif_max > 10 and "feature-selection" in tags:
        reasons.append(f"VIF_max={mf.vif_max:.1f} — multicollinearity detected; L1 regularisation will help")
    if mf.high_corr_pair_count > 2 and "ensemble" in tags:
        reasons.append(f"{int(mf.high_corr_pair_count)} high-correlation pairs — ensemble models robust to redundancy")

    # Class imbalance
    if mf.class_imbalance > 0 and mf.class_imbalance < 0.25 and "ensemble" in tags:
        reasons.append(f"Class imbalance (minority={mf.class_imbalance:.2f}) — use class_weight='balanced'")

    # Dataset size
    if mf.n_rows < 500 and "fast" in tags:
        reasons.append(f"Small dataset ({int(mf.n_rows)} rows) — fast model avoids overfitting")
    if mf.n_rows > 3000 and "high-performance" in tags:
        reasons.append(f"Large dataset ({int(mf.n_rows)} rows) — gradient boosting scales well")

    # Landmarking
    if algo_name in ("random_forest", "xgboost", "gradient_boosting"):
        if mf.lm_tiny_dt > 0.1:
            reasons.append(f"Tiny DT landmark score={mf.lm_tiny_dt:.3f} — tree-based splitting works on this data")
    if algo_name in ("logistic_regression", "lasso"):
        if mf.lm_linear > 0.05:
            reasons.append(f"Linear landmark score={mf.lm_linear:.3f} — linear boundary captures signal")
    if algo_name == "knn":
        if mf.lm_1nn > 0.1:
            reasons.append(f"1-NN landmark score={mf.lm_1nn:.3f} — strong local structure in data")

    if not reasons:
        reasons.append("Meta-learner pattern match based on overall dataset profile")

    return reasons


def recommend(
    df: pd.DataFrame,
    target_col: str,
    model_path: str = "recommender.pkl",
    top_k: int = 5,
    datetime_col: Optional[str] = None,
) -> list:
    """
    Given a new dataset, return a ranked list of AlgorithmRecommendation objects.

    Parameters
    ----------
    df          : Raw DataFrame
    target_col  : Column to predict
    model_path  : Path to saved recommender bundle (.pkl)
    top_k       : Number of top algorithms to return
    datetime_col: Override datetime column detection
    """
    from automl_profiler import profile_dataset
    from meta_extracter import extract_meta_features

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Recommender model not found at '{model_path}'. "
            f"Run build_meta_dataset() then train_recommender() first."
        )

    # Load model
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    clf = bundle["clf"]
    score_regs = bundle["score_regs"]
    le = bundle["label_encoder"]
    feature_names = bundle["feature_names"]

    # Extract meta-features
    print("Profiling dataset...")
    profile = profile_dataset(df, target_col=target_col, datetime_col=datetime_col)
    print("Extracting meta-features...")
    mf = extract_meta_features(df, profile, target_col=target_col)

    x = mf.to_array().reshape(1, -1)

    # Predict class probabilities
    proba = clf.predict_proba(x)[0]
    classes = le.classes_

    # Predict estimated score per algorithm
    estimated_scores = {}
    for algo in classes:
        if algo in score_regs:
            est = score_regs[algo].predict(x)[0]
            estimated_scores[algo] = float(np.clip(est, 0.0, 1.0))
        else:
            estimated_scores[algo] = 0.0

    # Rank by probability
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    recommendations = []
    for rank, (algo, conf) in enumerate(ranked[:top_k], 1):
        reasoning = _build_reasoning(mf, algo)
        rec = AlgorithmRecommendation(
            rank=rank,
            algorithm=algo,
            confidence=round(float(conf), 4),
            predicted_score=round(estimated_scores.get(algo, 0.0), 4),
            description=ALGORITHMS[algo]["description"],
            reasoning=reasoning,
            tags=ALGORITHMS[algo]["tags"],
        )
        recommendations.append(rec)

    return recommendations, profile, mf


# ──────────────────────────────────────────────────────────────
# Quick demo (no OpenML needed) — uses a synthetic meta-dataset
# ──────────────────────────────────────────────────────────────

def _build_synthetic_meta_dataset(n: int = 120, path: str = "meta_dataset_synthetic.csv"):
    """
    Generates a synthetic meta-dataset for testing the recommender
    without needing OpenML access. Uses heuristic label assignment.
    """
    from meta_extracter import MetaFeatureVector
    from dataclasses import fields

    np.random.seed(42)
    feature_names = [f.name for f in fields(MetaFeatureVector)]
    rows = []

    for _ in range(n):
        # Randomly sample meta-feature values in realistic ranges
        nonlinear_adv = np.random.uniform(-0.3, 0.5)
        is_stat       = np.random.choice([-1, 0, 1], p=[0.4, 0.3, 0.3])
        n_rows        = np.random.randint(100, 8000)
        n_classes     = np.random.choice([0, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.1])
        imbalance     = np.random.uniform(0.05, 0.5) if n_classes > 0 else 0.0
        vif_max       = np.random.choice([1.0, 5.0, 15.0, 50.0], p=[0.4, 0.3, 0.2, 0.1])
        season_str    = np.random.uniform(0, 0.9) if is_stat != 1 else 0.0
        lm_linear     = np.random.uniform(0, 0.4)
        lm_1nn        = lm_linear + nonlinear_adv + np.random.normal(0, 0.05)
        lm_tiny_dt    = np.random.uniform(0, 0.35)

        row = {
            "n_rows": n_rows,
            "n_cols": np.random.randint(3, 40),
            "log_n_rows": np.log1p(n_rows),
            "log_n_cols": np.log1p(np.random.randint(3, 40)),
            "rows_cols_ratio": np.random.uniform(10, 500),
            "missing_ratio": np.random.uniform(0, 0.2),
            "duplicate_ratio": np.random.uniform(0, 0.05),
            "numeric_ratio": np.random.uniform(0.3, 1.0),
            "categorical_ratio": np.random.uniform(0, 0.4),
            "temporal_present": float(is_stat != 0),
            "skewness_mean": np.random.uniform(0, 2),
            "skewness_std": np.random.uniform(0, 1),
            "kurtosis_mean": np.random.uniform(-1, 5),
            "kurtosis_std": np.random.uniform(0, 2),
            "outlier_ratio_mean": np.random.uniform(0, 0.15),
            "target_skewness": np.random.uniform(-2, 2),
            "class_imbalance": imbalance,
            "n_classes": float(n_classes),
            "corr_mean_abs": np.random.uniform(0, 0.5),
            "corr_max_abs": np.random.uniform(0.3, 0.99),
            "high_corr_pair_count": float(np.random.randint(0, 5)),
            "vif_max": vif_max,
            "is_stationary": float(is_stat),
            "adf_pvalue": np.random.uniform(0, 1),
            "trend_strength": np.random.uniform(0, 0.9),
            "seasonality_strength": season_str,
            "dt_cv_score": np.random.uniform(0.5, 0.95),
            "dt_train_time": np.random.uniform(0.01, 2.0),
            "linear_cv_score": np.random.uniform(0.4, 0.9),
            "linear_train_time": np.random.uniform(0.01, 1.0),
            "dt_linear_gap": nonlinear_adv,
            "dt_depth_used": float(np.random.randint(2, 8)),
            "linear_coef_std": np.random.uniform(0, 2),
            "lm_1nn": float(np.clip(lm_1nn, -0.5, 0.8)),
            "lm_naive_bayes": np.random.uniform(0, 0.4),
            "lm_tiny_dt": float(np.clip(lm_tiny_dt, -0.2, 0.7)),
            "lm_linear": float(np.clip(lm_linear, -0.2, 0.6)),
            "lm_baseline": 0.0,
            "lm_best": float(np.clip(max(lm_1nn, lm_tiny_dt, lm_linear), 0, 0.9)),
            "lm_nonlinear_advantage": float(np.clip(nonlinear_adv, -0.5, 0.8)),
            "lm_spread": np.random.uniform(0, 0.3),
        }

        # Heuristic label assignment based on dataset characteristics
        if nonlinear_adv > 0.15 and n_rows > 500:
            best = np.random.choice(["xgboost", "random_forest"], p=[0.6, 0.4])
        elif nonlinear_adv > 0.0 and n_rows <= 500:
            best = np.random.choice(["random_forest", "decision_tree", "gradient_boosting"], p=[0.5, 0.3, 0.2])
        elif nonlinear_adv < -0.05 and vif_max > 10:
            best = np.random.choice(["lasso", "logistic_regression"], p=[0.6, 0.4])
        elif nonlinear_adv < 0.05:
            best = np.random.choice(["logistic_regression", "lasso", "knn"], p=[0.5, 0.3, 0.2])
        else:
            best = np.random.choice(list(ALGORITHMS.keys()))

        row["best_algorithm"] = best
        row["task_type"] = "regression" if n_classes == 0 else "classification"
        row["openml_did"] = -1
        row["dataset_name"] = f"synthetic_{_}"
        row["algo_scores"] = json.dumps({k: round(np.random.uniform(0.4, 0.95), 4) for k in ALGORITHMS})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Synthetic meta-dataset saved: {path}  ({len(df)} rows)")
    return df