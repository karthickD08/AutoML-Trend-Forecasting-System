"""
app.py
------
Phase 4b — Streamlit UI for the AutoML Trend Forecasting system.

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Trend Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.algo-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    background: #e8f0fe;
    color: #1a56db;
    margin-right: 6px;
}
.reason-item {
    background: #f0faf4;
    border-left: 3px solid #22c55e;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
}
.warn-item {
    background: #fff7ed;
    border-left: 3px solid #f59e0b;
    padding: 6px 12px;
    margin: 4px 0;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_profiler(df_json, target_col, datetime_col):
    from automl_profiler import profile_dataset
    df = pd.read_json(StringIO(df_json))
    if datetime_col and datetime_col != "None":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    return profile_dataset(df, target_col=target_col,
                           datetime_col=datetime_col if datetime_col != "None" else None)

@st.cache_data(show_spinner=False)
def run_meta_extractor(df_json, profile_dict_json, target_col):
    from automl_profiler import profile_dataset
    from meta_extracter import extract_meta_features
    df = pd.read_json(StringIO(df_json))
    target_col_clean = target_col
    profile = profile_dataset(df, target_col=target_col_clean)
    return extract_meta_features(df, profile, target_col=target_col_clean)

@st.cache_data(show_spinner=False)
def run_recommend(df_json, target_col, datetime_col, model_path, top_k):
    from algorithm_recommender import recommend
    df = pd.read_json(StringIO(df_json))
    if datetime_col and datetime_col != "None":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    results, profile, mf = recommend(
        df, target_col=target_col,
        model_path=model_path, top_k=top_k,
        datetime_col=datetime_col if datetime_col != "None" else None,
    )
    return results, profile, mf

def run_explainer(df, target_col, datetime_col, algo_name, reasoning, confidence):
    from explainability import ExplainabilityEngine
    engine = ExplainabilityEngine(
        df, target_col=target_col,
        datetime_col=datetime_col if datetime_col != "None" else None,
    )
    engine.fit(algo_name)
    bundle = engine.explain(reasoning=reasoning,
                            recommendation_confidence=confidence,
                            n_bootstrap=20)
    return bundle

def color_conf(conf):
    if conf >= 0.5: return "#22c55e"
    if conf >= 0.25: return "#f59e0b"
    return "#ef4444"


# ── sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 AutoML Forecaster")
    st.markdown("---")

    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
    st.markdown("**or use a demo dataset**")
    use_demo = st.button("Load demo dataset")

    st.markdown("---")
    model_path = st.text_input("Recommender model path", value="recommender.pkl")
    top_k = st.slider("Top-K recommendations", 1, 8, 5)
    st.markdown("---")
    st.caption("AutoML Trend Forecasting System · Phase 4")


# ── load data ─────────────────────────────────────────────────

df = None

if use_demo or "demo_loaded" in st.session_state:
    st.session_state["demo_loaded"] = True
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    trend = np.linspace(50, 200, n)
    sales = trend + 18*np.sin(2*np.pi*np.arange(n)/365) + np.random.normal(0, 8, n)
    df = pd.DataFrame({
        "date": dates.astype(str),
        "sales": sales.round(2),
        "marketing_spend": (sales*0.4 + np.random.normal(0,10,n)).round(2),
        "temperature": (20 + 10*np.sin(2*np.pi*np.arange(n)/365) + np.random.normal(0,2,n)).round(2),
        "day_of_week": [d.weekday() for d in dates],
        "region": np.random.choice(["North","South","East","West"], n),
        "promo_flag": np.random.choice([0,1], n, p=[0.8,0.2]),
    })
    st.session_state["df"] = df

elif uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["df"] = df

if "df" in st.session_state:
    df = st.session_state["df"]


# ── main ──────────────────────────────────────────────────────

if df is None:
    st.markdown("## Welcome to AutoML Trend Forecaster")
    st.markdown("""
    Upload a CSV or load the demo dataset to get started.

    **What this system does:**
    1. **Profiles** your dataset — detects types, missing values, stationarity, correlations
    2. **Extracts meta-features** — 41 numeric signals describing your data's structure
    3. **Recommends algorithms** — a meta-learner trained on OpenML benchmarks ranks the best algorithms for your data
    4. **Explains everything** — SHAP feature attribution, confidence intervals, and reasoning per recommendation
    """)
    st.stop()

# ── tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Profile",
    "🧠 Meta-features",
    "⚡ Recommendations",
    "🔍 Explainability",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 — DATASET PROFILE
# ════════════════════════════════════════════════════════════════

with tab1:
    st.header("Dataset Profile")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Configuration")
        target_col = st.selectbox("Target column", df.columns.tolist())
        dt_candidates = ["None"] + [c for c in df.columns
                                     if "date" in c.lower() or "time" in c.lower()
                                     or "year" in c.lower() or "month" in c.lower()]
        datetime_col = st.selectbox("Datetime column (optional)", dt_candidates)
        run_profile_btn = st.button("Run profiler", type="primary")

    if run_profile_btn or "profile" in st.session_state:
        if run_profile_btn:
            with st.spinner("Profiling dataset..."):
                p = run_profiler(df.to_json(), target_col,
                                 datetime_col if datetime_col != "None" else None)
                st.session_state["profile"] = p
                st.session_state["target_col"] = target_col
                st.session_state["datetime_col"] = datetime_col

        p = st.session_state["profile"]

        with col_right:
            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Rows", f"{p.n_rows:,}")
            k2.metric("Columns", p.n_cols)
            k3.metric("Missing", f"{p.total_missing_ratio*100:.1f}%")
            k4.metric("Duplicates", f"{p.duplicate_row_ratio*100:.1f}%")

        st.markdown("---")

        # Column type chart
        from collections import Counter
        dtype_counts = Counter(c.dtype for c in p.columns)
        fig_types = px.bar(
            x=list(dtype_counts.keys()),
            y=list(dtype_counts.values()),
            labels={"x": "Type", "y": "Count"},
            title="Column type distribution",
            color=list(dtype_counts.keys()),
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_types.update_layout(showlegend=False, height=300)

        # Missing values
        missing_data = [(c.name, c.missing_ratio*100) for c in p.columns if c.missing_ratio > 0]
        if missing_data:
            fig_miss = px.bar(
                x=[m[0] for m in missing_data],
                y=[m[1] for m in missing_data],
                labels={"x": "Column", "y": "Missing %"},
                title="Missing values per column",
                color=[m[1] for m in missing_data],
                color_continuous_scale="Reds",
            )
            fig_miss.update_layout(showlegend=False, height=300)
            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_types, use_container_width=True)
            c2.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.plotly_chart(fig_types, use_container_width=True)

        # Temporal section
        if p.is_temporal and p.temporal:
            t = p.temporal
            st.markdown("#### ⏱ Temporal analysis")
            tc1, tc2, tc3, tc4 = st.columns(4)
            tc1.metric("Frequency", t.inferred_frequency or "unknown")
            tc2.metric("Stationary", "✓ Yes" if t.is_stationary else "✗ No")
            tc3.metric("Trend strength", f"{t.trend_strength:.2f}" if t.trend_strength else "—")
            tc4.metric("Season strength", f"{t.seasonality_strength:.2f}" if t.seasonality_strength else "—")

            # Plot the target time series
            dt_col_name = t.datetime_col
            if dt_col_name in df.columns and target_col in df.columns:
                plot_df = df[[dt_col_name, target_col]].copy()
                plot_df[dt_col_name] = pd.to_datetime(plot_df[dt_col_name])
                plot_df = plot_df.sort_values(dt_col_name)
                fig_ts = px.line(plot_df, x=dt_col_name, y=target_col,
                                 title=f"{target_col} over time")
                fig_ts.update_layout(height=320)
                st.plotly_chart(fig_ts, use_container_width=True)

        # Column details table
        st.markdown("#### Column details")
        col_data = [{
            "Column": c.name,
            "Type": c.dtype,
            "Missing %": f"{c.missing_ratio*100:.1f}%",
            "Unique": c.unique_count,
            "Mean": f"{c.mean:.2f}" if c.mean is not None else "—",
            "Skewness": f"{c.skewness:.2f}" if c.skewness is not None else "—",
            "Outlier %": f"{c.outlier_ratio*100:.1f}%" if c.outlier_ratio is not None else "—",
        } for c in p.columns]
        st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

        # Flags
        if p.constant_col_names:
            st.warning(f"⚠ Constant columns detected (drop before training): {p.constant_col_names}")
        if p.near_constant_col_names:
            st.warning(f"⚠ Near-constant columns: {p.near_constant_col_names}")
        if p.target and p.target.leakage_risk_cols:
            st.error(f"🚨 Leakage risk — columns with |r|>0.95 with target: {p.target.leakage_risk_cols}")


# ════════════════════════════════════════════════════════════════
# TAB 2 — META-FEATURES
# ════════════════════════════════════════════════════════════════

with tab2:
    st.header("Meta-feature Vector")

    if "profile" not in st.session_state:
        st.info("Run the profiler in Tab 1 first.")
        st.stop()

    if st.button("Extract meta-features", type="primary") or "mf" in st.session_state:
        if "mf" not in st.session_state or st.session_state.get("mf_target") != st.session_state.get("target_col"):
            with st.spinner("Extracting 41 meta-features..."):
                from automl_profiler import profile_dataset
                from meta_extracter import extract_meta_features
                target = st.session_state["target_col"]
                dt_col = st.session_state["datetime_col"]
                df_work = df.copy()
                if dt_col and dt_col != "None" and dt_col in df_work.columns:
                    df_work[dt_col] = pd.to_datetime(df_work[dt_col])
                p = profile_dataset(df_work, target_col=target,
                                    datetime_col=dt_col if dt_col != "None" else None)
                mf = extract_meta_features(df_work, p, target_col=target)
                st.session_state["mf"] = mf
                st.session_state["mf_target"] = target

        mf = st.session_state["mf"]
        mf_dict = mf.to_dict()

        # Group features
        groups = {
            "Statistical — Size": ["n_rows","n_cols","log_n_rows","log_n_cols","rows_cols_ratio"],
            "Statistical — Quality": ["missing_ratio","duplicate_ratio","numeric_ratio","categorical_ratio","temporal_present"],
            "Statistical — Distribution": ["skewness_mean","skewness_std","kurtosis_mean","kurtosis_std","outlier_ratio_mean"],
            "Statistical — Target": ["target_skewness","class_imbalance","n_classes"],
            "Statistical — Correlation": ["corr_mean_abs","corr_max_abs","high_corr_pair_count","vif_max"],
            "Statistical — Temporal": ["is_stationary","adf_pvalue","trend_strength","seasonality_strength"],
            "Model-based": ["dt_cv_score","dt_train_time","linear_cv_score","linear_train_time","dt_linear_gap","dt_depth_used","linear_coef_std"],
            "Landmarking": ["lm_1nn","lm_naive_bayes","lm_tiny_dt","lm_linear","lm_baseline","lm_best","lm_nonlinear_advantage","lm_spread"],
        }

        for group_name, feat_list in groups.items():
            with st.expander(group_name, expanded="Landmarking" in group_name or "Model-based" in group_name):
                rows = []
                for feat in feat_list:
                    val = mf_dict.get(feat, "—")
                    rows.append({"Feature": feat, "Value": round(val, 5) if isinstance(val, float) else val})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # Radar chart — landmarking scores
        lm_names  = ["1-NN", "Naive Bayes", "Tiny DT", "Linear"]
        lm_values = [mf.lm_1nn, mf.lm_naive_bayes, mf.lm_tiny_dt, mf.lm_linear]
        fig_radar = go.Figure(go.Scatterpolar(
            r=lm_values + [lm_values[0]],
            theta=lm_names + [lm_names[0]],
            fill="toself",
            line_color="#4f46e5",
            fillcolor="rgba(79,70,229,0.15)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-0.2, 1.0])),
            title="Landmarking scores (relative to baseline)",
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Summary signals
        st.markdown("#### Signals for algorithm selection")
        summary = mf.summary()
        signal_lines = [l for l in summary.split("\n") if l.strip().startswith(("✓","⚠"))]
        for line in signal_lines:
            cls = "reason-item" if "✓" in line else "warn-item"
            st.markdown(f'<div class="{cls}">{line.strip()}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════

with tab3:
    st.header("Algorithm Recommendations")

    if "profile" not in st.session_state:
        st.info("Run the profiler in Tab 1 first.")
        st.stop()

    model_exists = os.path.exists(model_path)
    if not model_exists:
        st.warning(f"Recommender model not found at `{model_path}`. "
                   f"Run `build_meta_dataset()` then `train_recommender()` first, "
                   f"or load the demo recommender.")
        if st.button("Train demo recommender (synthetic meta-dataset)"):
            with st.spinner("Building synthetic meta-dataset and training recommender..."):
                from algorithm_recommender import _build_synthetic_meta_dataset, train_recommender
                _build_synthetic_meta_dataset(n=150, path="meta_dataset_synthetic.csv")
                train_recommender(meta_csv="meta_dataset_synthetic.csv",
                                  model_path=model_path)
            st.success("Demo recommender trained!")
            st.rerun()
        st.stop()

    if st.button("Get recommendations", type="primary") or "recommendations" in st.session_state:
        if "recommendations" not in st.session_state:
            target = st.session_state["target_col"]
            dt_col = st.session_state["datetime_col"]
            with st.spinner("Running meta-learner..."):
                df_work = df.copy()
                if dt_col and dt_col != "None" and dt_col in df_work.columns:
                    df_work[dt_col] = pd.to_datetime(df_work[dt_col])
                results, profile, mf = run_recommend(
                    df_work.to_json(), target, dt_col, model_path, top_k
                )
                st.session_state["recommendations"] = results
                st.session_state["rec_mf"] = mf

        results = st.session_state["recommendations"]

        # Confidence bar chart
        algos  = [r.algorithm for r in results]
        confs  = [r.confidence for r in results]
        scores = [r.predicted_score for r in results]

        fig_bar = make_subplots(rows=1, cols=2,
                                subplot_titles=("Recommendation confidence", "Estimated CV score"))
        fig_bar.add_trace(go.Bar(x=algos, y=confs, marker_color="#4f46e5",
                                  name="Confidence"), row=1, col=1)
        fig_bar.add_trace(go.Bar(x=algos, y=scores, marker_color="#22c55e",
                                  name="Est. score"), row=1, col=2)
        fig_bar.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Recommendation cards
        st.markdown("#### Ranked recommendations")
        for r in results:
            c_col = color_conf(r.confidence)
            with st.expander(
                f"#{r.rank}  {r.algorithm.upper()}  —  confidence {r.confidence:.0%}  |  est. score {r.predicted_score:.3f}",
                expanded=r.rank == 1
            ):
                st.markdown(f"**{r.description}**")
                st.markdown("**Tags:** " + "  ".join(
                    f'<span class="algo-badge">{t}</span>' for t in r.tags
                ), unsafe_allow_html=True)
                st.markdown("**Why recommended:**")
                for reason in r.reasoning:
                    st.markdown(f'<div class="reason-item">→ {reason}</div>',
                                unsafe_allow_html=True)

        # Store top recommendation for Tab 4
        st.session_state["top_algo"] = results[0].algorithm
        st.session_state["top_reasoning"] = results[0].reasoning
        st.session_state["top_confidence"] = results[0].confidence


# ════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ════════════════════════════════════════════════════════════════

with tab4:
    st.header("Explainability")

    if "profile" not in st.session_state:
        st.info("Run the profiler in Tab 1 first.")
        st.stop()

    target  = st.session_state.get("target_col", df.columns[0])
    dt_col  = st.session_state.get("datetime_col", "None")
    top_algo = st.session_state.get("top_algo", "random_forest")

    from algorithm_recommender import ALGORITHMS
    algo_choice = st.selectbox(
        "Algorithm to explain",
        list(ALGORITHMS.keys()),
        index=list(ALGORITHMS.keys()).index(top_algo) if top_algo in ALGORITHMS else 0,
    )

    if st.button("Run explainability", type="primary"):
        reasoning   = st.session_state.get("top_reasoning", [])
        confidence  = st.session_state.get("top_confidence", 0.0)

        with st.spinner(f"Running SHAP + bootstrap CI for {algo_choice}..."):
            df_work = df.copy()
            if dt_col and dt_col != "None" and dt_col in df_work.columns:
                df_work[dt_col] = pd.to_datetime(df_work[dt_col])
            bundle = run_explainer(df_work, target, dt_col, algo_choice, reasoning, confidence)
            st.session_state["bundle"] = bundle

    if "bundle" in st.session_state:
        bundle = st.session_state["bundle"]

        # KPI row
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Algorithm", bundle.algo_name.replace("_"," ").title())
        e2.metric("CV score", f"{bundle.cv_mean:.4f} ± {bundle.cv_std:.4f}")
        e3.metric("Avg CI width", f"{bundle.ci_width_mean:.4f}")
        e4.metric("Task", bundle.task.title())

        st.markdown("---")

        # ── SHAP feature importance bar ───────────────────────
        st.subheader("SHAP Feature Importance")
        top_n = min(15, len(bundle.feature_importance))
        top_feats = bundle.top_features(top_n)
        feat_names_plot = [f[0] for f in top_feats]
        feat_vals_plot  = [f[1] for f in top_feats]

        fig_shap = go.Figure(go.Bar(
            x=feat_vals_plot[::-1],
            y=feat_names_plot[::-1],
            orientation="h",
            marker_color=px.colors.sequential.Viridis[:top_n][::-1],
        ))
        fig_shap.update_layout(
            title=f"Mean |SHAP| — top {top_n} features",
            xaxis_title="Mean |SHAP value|",
            height=max(300, top_n * 28),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        # ── CV score distribution ─────────────────────────────
        st.subheader("Cross-validation Scores")
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Bar(
            x=[f"Fold {i+1}" for i in range(len(bundle.cv_scores))],
            y=bundle.cv_scores,
            marker_color="#4f46e5",
            text=[f"{s:.3f}" for s in bundle.cv_scores],
            textposition="outside",
        ))
        fig_cv.add_hline(y=bundle.cv_mean, line_dash="dash",
                         line_color="red", annotation_text=f"Mean={bundle.cv_mean:.3f}")
        fig_cv.update_layout(
            title="5-fold CV scores",
            yaxis_title="Score",
            height=300,
        )
        st.plotly_chart(fig_cv, use_container_width=True)

        # ── Predictions + CI ──────────────────────────────────
        st.subheader("Predictions with Confidence Intervals (90%)")
        n_show = min(150, len(bundle.predictions))
        idx_show = np.arange(n_show)
        preds_show   = bundle.predictions[:n_show]
        lower_show   = bundle.ci_lower[:n_show]
        upper_show   = bundle.ci_upper[:n_show]

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=np.concatenate([idx_show, idx_show[::-1]]),
            y=np.concatenate([upper_show, lower_show[::-1]]),
            fill="toself",
            fillcolor="rgba(79,70,229,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% CI",
        ))
        fig_ci.add_trace(go.Scatter(
            x=idx_show, y=preds_show,
            mode="lines",
            line=dict(color="#4f46e5", width=2),
            name="Prediction",
        ))
        fig_ci.update_layout(
            title=f"Predictions ± 90% CI (first {n_show} samples)",
            xaxis_title="Sample index",
            yaxis_title="Predicted value",
            height=360,
        )
        st.plotly_chart(fig_ci, use_container_width=True)

        # ── Selection reasoning ───────────────────────────────
        st.subheader("Why this algorithm was recommended")
        if bundle.reasoning:
            for reason in bundle.reasoning:
                st.markdown(f'<div class="reason-item">→ {reason}</div>',
                            unsafe_allow_html=True)
        else:
            st.info("Run recommendations in Tab 3 first to see selection reasoning here.")

        st.markdown(f"**Recommender confidence:** {bundle.recommendation_confidence:.0%}")
