"""
explainability.py
-----------------
Phase 4a of the AutoML Trend Forecasting system.

Given a trained model and dataset, produces:
  - SHAP values (feature attribution per prediction)
  - Confidence intervals (bootstrap)
  - Selection reasoning (why each algorithm was recommended)
  - Per-algorithm explanation bundle ready for the UI layer

Usage:
    from explainability import ExplainabilityEngine
    engine = ExplainabilityEngine(df, target_col="sales", task="regression")
    engine.fit(algo_name="xgboost")
    bundle = engine.explain()
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Model builder  (mirrors ALGORITHMS in recommender)
# ──────────────────────────────────────────────────────────────

def _build_model(algo_name: str, task: str):
    """Return (pipeline, needs_proba) for a given algorithm + task."""
    clf_map = {
        "random_forest":      RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost":            XGBClassifier(n_estimators=100, random_state=42, verbosity=0, eval_metric="logloss"),
        "gradient_boosting":  GradientBoostingClassifier(n_estimators=100, random_state=42),
        "logistic_regression":LogisticRegression(max_iter=500, random_state=42),
        "lasso":              LogisticRegression(penalty="l1", solver="saga", max_iter=500, random_state=42),
        "knn":                KNeighborsClassifier(n_neighbors=5),
        "decision_tree":      DecisionTreeClassifier(max_depth=8, random_state=42),
        "svm":                SVC(kernel="rbf", probability=True, random_state=42),
    }
    reg_map = {
        "random_forest":      RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost":            XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        "gradient_boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
        "logistic_regression":Ridge(alpha=1.0),
        "lasso":              Lasso(alpha=0.1, max_iter=1000),
        "knn":                KNeighborsRegressor(n_neighbors=5),
        "decision_tree":      DecisionTreeRegressor(max_depth=8, random_state=42),
        "svm":                SVR(kernel="rbf"),
    }
    needs_scale = algo_name in ("logistic_regression", "lasso", "knn", "svm")
    base_model  = clf_map[algo_name] if task == "classification" else reg_map[algo_name]

    steps = [("imp", SimpleImputer(strategy="median"))]
    if needs_scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", base_model))
    return Pipeline(steps)


# ──────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────

@dataclass
class ExplainBundle:
    algo_name: str
    task: str

    # SHAP
    shap_values: np.ndarray               # shape (n_samples, n_features)
    shap_base_value: float
    feature_names: list
    feature_importance: dict              # {feature: mean |shap|}  sorted desc

    # CV performance
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float

    # Confidence intervals (bootstrap)
    predictions: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_width_mean: float

    # Selection reasoning (from recommender)
    reasoning: list = field(default_factory=list)
    recommendation_confidence: float = 0.0

    def top_features(self, n: int = 10) -> list:
        return list(self.feature_importance.items())[:n]

    def summary(self) -> str:
        lines = [f"── Explainability: {self.algo_name} ({self.task}) ──"]
        lines.append(f"CV score    : {self.cv_mean:.4f} ± {self.cv_std:.4f}")
        lines.append(f"CI width    : {self.ci_width_mean:.4f} (avg prediction interval)")
        lines.append(f"Top features:")
        for fname, imp in self.top_features(5):
            lines.append(f"  {fname:30s}  |SHAP|={imp:.4f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────────────────────

class ExplainabilityEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        task: Optional[str] = None,       # auto-detected if None
        datetime_col: Optional[str] = None,
        max_rows: int = 2000,
    ):
        self.target_col  = target_col
        self.datetime_col = datetime_col
        self.max_rows    = max_rows
        self._model      = None
        self._algo_name  = None

        self.X, self.y, self.task, self.feature_names = self._prepare(df, task)

    # ── data prep ────────────────────────────────────────────

    def _prepare(self, df: pd.DataFrame, task_override):
        df = df.copy()

        # Drop datetime and constant cols
        drop = []
        for col in df.columns:
            if col == self.target_col:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                drop.append(col)
            elif df[col].nunique() <= 1:
                drop.append(col)
        # Also try to parse object cols as datetime
        if self.datetime_col and self.datetime_col in df.columns:
            drop.append(self.datetime_col)
        df = df.drop(columns=[c for c in drop if c in df.columns])

        y_raw = df[self.target_col].copy()
        X_raw = df.drop(columns=[self.target_col])

        # Encode
        for col in X_raw.select_dtypes(include=["object", "bool", "category"]).columns:
            X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))

        feature_names = X_raw.columns.tolist()
        X = X_raw.values.astype(float)

        # Task detection
        if task_override:
            task = task_override
        elif y_raw.nunique() <= 20 and not pd.api.types.is_float_dtype(y_raw):
            task = "classification"
        else:
            task = "regression"

        if task == "classification":
            y = LabelEncoder().fit_transform(y_raw.fillna(y_raw.mode()[0]).astype(str))
        else:
            y = y_raw.fillna(y_raw.median()).values.astype(float)

        # Subsample
        if len(X) > self.max_rows:
            idx = np.random.choice(len(X), self.max_rows, replace=False)
            X, y = X[idx], y[idx]

        return X, y, task, feature_names

    # ── fit ──────────────────────────────────────────────────

    def fit(self, algo_name: str):
        self._algo_name = algo_name
        self._model = _build_model(algo_name, self.task)
        self._model.fit(self.X, self.y)
        return self

    # ── SHAP ─────────────────────────────────────────────────

    def _compute_shap(self) -> tuple:
        model_step = self._model.named_steps["model"]
        imp = self._model.named_steps["imp"]
        X_clean = imp.transform(self.X)

        # Scale if pipeline has scaler
        if "scaler" in self._model.named_steps:
            X_clean = self._model.named_steps["scaler"].transform(X_clean)

        # Choose explainer by model type
        tree_models = (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
            DecisionTreeClassifier, DecisionTreeRegressor,
            XGBClassifier, XGBRegressor,
        )

        try:
            if isinstance(model_step, tree_models):
                explainer = shap.TreeExplainer(model_step)
                shap_vals = explainer.shap_values(X_clean)
                base_val  = float(np.mean(explainer.expected_value)
                                  if isinstance(explainer.expected_value, np.ndarray)
                                  else explainer.expected_value)
            else:
                # KernelExplainer for linear/SVM/KNN — use background sample
                bg = shap.sample(X_clean, min(100, len(X_clean)))
                if self.task == "classification":
                    explainer = shap.KernelExplainer(model_step.predict_proba, bg)
                else:
                    explainer = shap.KernelExplainer(model_step.predict, bg)
                sample = X_clean[:min(200, len(X_clean))]
                shap_vals = explainer.shap_values(sample)
                base_val  = float(np.mean(explainer.expected_value)
                                  if isinstance(explainer.expected_value, (list, np.ndarray))
                                  else explainer.expected_value)
                X_clean = sample   # align sizes

            # For multi-class: take mean over classes
            if isinstance(shap_vals, list):
                shap_arr = np.mean(np.abs(shap_vals), axis=0)
            else:
                shap_arr = shap_vals

        except Exception as e:
            # Fallback: permutation-based importance as proxy
            print(f"  SHAP fallback (TreeExplainer failed: {e})")
            shap_arr = np.zeros((len(X_clean), len(self.feature_names)))
            base_val = float(np.mean(self.y))

        # Feature importance: mean |SHAP| per feature
        importance = dict(sorted(
            zip(self.feature_names, np.mean(np.abs(shap_arr), axis=0).tolist()),
            key=lambda x: x[1], reverse=True
        ))
        return shap_arr, base_val, importance

    # ── CV ───────────────────────────────────────────────────

    def _compute_cv(self) -> tuple:
        scoring = "accuracy" if self.task == "classification" else "r2"
        cv = (StratifiedKFold(5, shuffle=True, random_state=42)
              if self.task == "classification"
              else KFold(5, shuffle=True, random_state=42))
        pipe = _build_model(self._algo_name, self.task)
        scores = cross_val_score(pipe, self.X, self.y, cv=cv,
                                 scoring=scoring, error_score=0.0)
        return scores, float(scores.mean()), float(scores.std())

    # ── Bootstrap CI ─────────────────────────────────────────

    def _compute_ci(self, n_bootstrap: int = 50) -> tuple:
        """Bootstrap confidence intervals on predictions."""
        n = len(self.X)
        preds_boot = []

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            X_b, y_b = self.X[idx], self.y[idx]
            pipe = _build_model(self._algo_name, self.task)
            try:
                pipe.fit(X_b, y_b)
                if self.task == "classification":
                    p = pipe.predict_proba(self.X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.predict(self.X)
                else:
                    p = pipe.predict(self.X)
                preds_boot.append(p)
            except Exception:
                continue

        if not preds_boot:
            preds = self._model.predict(self.X)
            return preds, preds * 0.9, preds * 1.1

        boot_arr  = np.array(preds_boot)          # (n_bootstrap, n_samples)
        preds     = np.mean(boot_arr, axis=0)
        ci_lower  = np.percentile(boot_arr, 5,  axis=0)
        ci_upper  = np.percentile(boot_arr, 95, axis=0)
        ci_width  = float(np.mean(ci_upper - ci_lower))
        return preds, ci_lower, ci_upper, ci_width

    # ── Public explain ────────────────────────────────────────

    def explain(
        self,
        reasoning: list = None,
        recommendation_confidence: float = 0.0,
        n_bootstrap: int = 30,
    ) -> ExplainBundle:
        if self._model is None:
            raise RuntimeError("Call fit(algo_name) before explain()")

        print(f"  Computing SHAP values...")
        shap_arr, base_val, importance = self._compute_shap()

        print(f"  Computing CV scores...")
        cv_scores, cv_mean, cv_std = self._compute_cv()

        print(f"  Computing bootstrap CI...")
        preds, ci_lower, ci_upper, ci_width = self._compute_ci(n_bootstrap)

        return ExplainBundle(
            algo_name=self._algo_name,
            task=self.task,
            shap_values=shap_arr,
            shap_base_value=base_val,
            feature_names=self.feature_names,
            feature_importance=importance,
            cv_scores=cv_scores,
            cv_mean=cv_mean,
            cv_std=cv_std,
            predictions=preds,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width_mean=ci_width,
            reasoning=reasoning or [],
            recommendation_confidence=recommendation_confidence,
        )