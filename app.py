"""Streamlit application for Calgary Neighborhood Livability Segmentation.

Provides an interactive dashboard for exploring community-level features
derived from four Calgary Open Data datasets, performing KMeans and
Agglomerative clustering, comparing communities via radar charts, and
visualising PCA results.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import build_feature_matrix, FEATURE_COLUMNS, load_all_datasets  # noqa: E402
from model import (  # noqa: E402
    compute_elbow,
    find_optimal_k,
    train_kmeans,
    train_agglomerative,
    fit_pca,
    get_pca_loadings,
    profile_clusters,
    save_model,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Calgary Neighborhood Livability Segmentation",
    page_icon="🏘️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading and preparing data ...")
def load_data():
    """Load datasets, build feature matrix, and return raw & scaled frames."""
    datasets = load_all_datasets(force_refresh=False)
    raw_df, scaled_df, scaler = build_feature_matrix(datasets=datasets)
    return raw_df, scaled_df, scaler


@st.cache_resource(show_spinner="Fitting clustering models ...")
def fit_models(scaled_values: np.ndarray, optimal_k: int):
    """Train KMeans, Agglomerative, and PCA and cache results."""
    km = train_kmeans(scaled_values, n_clusters=optimal_k)
    agg = train_agglomerative(scaled_values, n_clusters=optimal_k)
    pca, X_pca = fit_pca(scaled_values)

    # Persist
    save_model(km, "kmeans_model.joblib")
    save_model(agg, "agglomerative_model.joblib")
    save_model(pca, "pca_transformer.joblib")

    return km, agg, pca, X_pca


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _pretty_name(col: str) -> str:
    """Convert a snake_case column name to a Title Case label."""
    return col.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Community Explorer",
        "Cluster Analysis",
        "Community Comparison",
        "PCA Visualization",
        "About",
    ],
)

# ---------------------------------------------------------------------------
# Data loading (shared across pages)
# ---------------------------------------------------------------------------
raw_df, scaled_df, scaler = load_data()
X_scaled = scaled_df[FEATURE_COLUMNS].values

# Pre-compute elbow
elbow_df = compute_elbow(X_scaled)
optimal_k = find_optimal_k(elbow_df)

km_model, agg_model, pca_model, X_pca = fit_models(X_scaled, optimal_k)

# Attach labels to raw_df
raw_df = raw_df.copy()
raw_df["kmeans_cluster"] = km_model.labels_
raw_df["agglom_cluster"] = agg_model.labels_

# =========================================================================
# PAGE: Community Explorer
# =========================================================================
if page == "Community Explorer":
    st.title("Community Explorer")
    st.markdown(
        "Browse high-level statistics and feature distributions across "
        "Calgary communities."
    )

    # --- Overview metrics ---------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Communities", len(raw_df))
    col2.metric("Avg Population", f"{raw_df['total_population'].mean():,.0f}")
    col3.metric("Avg Businesses", f"{raw_df['business_count'].mean():,.0f}")
    col4.metric("Avg Crime Rate", f"{raw_df['crime_rate'].mean():,.1f}")

    st.divider()

    # --- Feature distributions ----------------------------------------------
    st.subheader("Feature Distributions")
    selected_feature = st.selectbox(
        "Select a feature",
        FEATURE_COLUMNS,
        format_func=_pretty_name,
    )
    fig_hist = px.histogram(
        raw_df,
        x=selected_feature,
        nbins=30,
        title=f"Distribution of {_pretty_name(selected_feature)}",
        labels={selected_feature: _pretty_name(selected_feature)},
    )
    fig_hist.update_layout(bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- Top / bottom communities -------------------------------------------
    st.subheader("Top and Bottom Communities")
    rank_feature = st.selectbox(
        "Rank by",
        FEATURE_COLUMNS,
        format_func=_pretty_name,
        key="rank_feature",
    )
    n_show = st.slider("Number to show", 5, 20, 10)

    top = raw_df.nlargest(n_show, rank_feature)[["community", rank_feature]]
    bottom = raw_df.nsmallest(n_show, rank_feature)[["community", rank_feature]]

    left, right = st.columns(2)
    with left:
        st.markdown(f"**Top {n_show} by {_pretty_name(rank_feature)}**")
        fig_top = px.bar(
            top,
            x=rank_feature,
            y="community",
            orientation="h",
            color=rank_feature,
            color_continuous_scale="Greens",
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
        st.plotly_chart(fig_top, use_container_width=True)
    with right:
        st.markdown(f"**Bottom {n_show} by {_pretty_name(rank_feature)}**")
        fig_bot = px.bar(
            bottom,
            x=rank_feature,
            y="community",
            orientation="h",
            color=rank_feature,
            color_continuous_scale="Reds_r",
        )
        fig_bot.update_layout(yaxis={"categoryorder": "total descending"}, height=400)
        st.plotly_chart(fig_bot, use_container_width=True)


# =========================================================================
# PAGE: Cluster Analysis
# =========================================================================
elif page == "Cluster Analysis":
    st.title("Cluster Analysis")
    st.markdown(
        "Determine the optimal number of clusters and inspect cluster "
        "assignments and profiles."
    )

    # --- Elbow & silhouette curves ------------------------------------------
    st.subheader("Elbow Curve and Silhouette Scores")
    left, right = st.columns(2)

    with left:
        fig_elbow = px.line(
            elbow_df,
            x="k",
            y="inertia",
            markers=True,
            title="Elbow Curve (Inertia vs. k)",
            labels={"k": "Number of Clusters (k)", "inertia": "Inertia"},
        )
        fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="red")
        st.plotly_chart(fig_elbow, use_container_width=True)

    with right:
        fig_sil = px.line(
            elbow_df,
            x="k",
            y="silhouette",
            markers=True,
            title="Silhouette Score vs. k",
            labels={"k": "Number of Clusters (k)", "silhouette": "Silhouette Score"},
        )
        fig_sil.add_vline(x=optimal_k, line_dash="dash", line_color="red")
        st.plotly_chart(fig_sil, use_container_width=True)

    st.info(f"Optimal k selected by highest silhouette score: **{optimal_k}**")

    st.divider()

    # --- Cluster assignment bar chart (KMeans) ------------------------------
    st.subheader("KMeans Cluster Assignments")
    cluster_counts = raw_df["kmeans_cluster"].value_counts().sort_index().reset_index()
    cluster_counts.columns = ["Cluster", "Count"]
    fig_counts = px.bar(
        cluster_counts,
        x="Cluster",
        y="Count",
        color="Cluster",
        title="Communities per KMeans Cluster",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig_counts, use_container_width=True)

    # --- Cluster profile table ----------------------------------------------
    st.subheader("Cluster Profiles (Mean Feature Values)")
    profile = profile_clusters(raw_df, km_model.labels_, FEATURE_COLUMNS)
    styled = profile.style.format(
        {col: "{:,.2f}" for col in FEATURE_COLUMNS},
    ).background_gradient(axis=0, subset=FEATURE_COLUMNS, cmap="YlGnBu")
    st.dataframe(styled, use_container_width=True)

    # --- Show community list per cluster ------------------------------------
    st.subheader("Communities by Cluster")
    selected_cluster = st.selectbox(
        "Select cluster",
        sorted(raw_df["kmeans_cluster"].unique()),
    )
    cluster_communities = raw_df.loc[
        raw_df["kmeans_cluster"] == selected_cluster,
        ["community"] + FEATURE_COLUMNS,
    ].sort_values("community")
    st.dataframe(cluster_communities.reset_index(drop=True), use_container_width=True)


# =========================================================================
# PAGE: Community Comparison
# =========================================================================
elif page == "Community Comparison":
    st.title("Community Comparison")
    st.markdown(
        "Select 2 to 4 communities and compare their feature profiles "
        "using a radar chart and side-by-side metrics."
    )

    all_communities = sorted(raw_df["community"].unique())
    selected = st.multiselect(
        "Select communities (2-4)",
        all_communities,
        default=all_communities[:2] if len(all_communities) >= 2 else all_communities,
        max_selections=4,
    )

    if len(selected) < 2:
        st.warning("Please select at least 2 communities to compare.")
    else:
        subset = raw_df[raw_df["community"].isin(selected)].copy()

        # Normalise features to 0-1 for radar chart
        radar_df = subset[FEATURE_COLUMNS].copy()
        for col in FEATURE_COLUMNS:
            col_min = raw_df[col].min()
            col_max = raw_df[col].max()
            denom = col_max - col_min if col_max != col_min else 1
            radar_df[col] = (radar_df[col] - col_min) / denom
        radar_df["community"] = subset["community"].values

        # --- Radar chart ----------------------------------------------------
        st.subheader("Radar Chart")
        fig_radar = go.Figure()
        for _, row in radar_df.iterrows():
            values = [row[c] for c in FEATURE_COLUMNS]
            values.append(values[0])  # close the polygon
            cats = [_pretty_name(c) for c in FEATURE_COLUMNS]
            cats.append(cats[0])
            fig_radar.add_trace(
                go.Scatterpolar(r=values, theta=cats, name=row["community"], fill="toself")
            )
        fig_radar.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 1]}},
            title="Feature Comparison (normalised 0-1)",
            height=550,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # --- Side-by-side metrics -------------------------------------------
        st.subheader("Side-by-Side Metrics")
        display_cols = ["community"] + FEATURE_COLUMNS
        display_df = subset[display_cols].set_index("community").T
        display_df.index = [_pretty_name(c) for c in display_df.index]
        st.dataframe(
            display_df.style.format("{:,.2f}").background_gradient(
                axis=1, cmap="RdYlGn"
            ),
            use_container_width=True,
        )


# =========================================================================
# PAGE: PCA Visualization
# =========================================================================
elif page == "PCA Visualization":
    st.title("PCA Visualization")
    st.markdown(
        "Explore the principal component space and understand which "
        "features drive the most variance."
    )

    pca_df = pd.DataFrame(
        X_pca[:, :2],
        columns=["PC1", "PC2"],
    )
    pca_df["community"] = raw_df["community"].values
    pca_df["cluster"] = raw_df["kmeans_cluster"].astype(str).values

    # --- 2-D scatter coloured by cluster ------------------------------------
    st.subheader("2-D PCA Scatter Plot (coloured by KMeans cluster)")
    fig_scatter = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_name="community",
        title="Communities in PCA Space",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        height=550,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Explained variance bar chart ---------------------------------------
    st.subheader("Explained Variance by Component")
    n_components = pca_model.n_components_
    var_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Explained Variance Ratio": pca_model.explained_variance_ratio_,
            "Cumulative": np.cumsum(pca_model.explained_variance_ratio_),
        }
    )
    fig_var = px.bar(
        var_df,
        x="Component",
        y="Explained Variance Ratio",
        title="Explained Variance per Principal Component",
        text_auto=".2%",
    )
    fig_var.add_scatter(
        x=var_df["Component"],
        y=var_df["Cumulative"],
        mode="lines+markers",
        name="Cumulative",
    )
    st.plotly_chart(fig_var, use_container_width=True)

    # --- Loadings plot -------------------------------------------------------
    st.subheader("PCA Loadings")
    loadings = get_pca_loadings(pca_model, FEATURE_COLUMNS)
    # Show loadings for PC1 and PC2
    load_plot = loadings.loc[["PC1", "PC2"]].T.reset_index()
    load_plot = load_plot.rename(columns={"index": "Feature"})
    load_plot["Feature"] = load_plot["Feature"].apply(_pretty_name)

    fig_load = go.Figure()
    fig_load.add_trace(
        go.Bar(name="PC1", x=load_plot["Feature"], y=load_plot["PC1"])
    )
    fig_load.add_trace(
        go.Bar(name="PC2", x=load_plot["Feature"], y=load_plot["PC2"])
    )
    fig_load.update_layout(
        barmode="group",
        title="Feature Loadings on PC1 and PC2",
        xaxis_title="Feature",
        yaxis_title="Loading",
        height=450,
    )
    st.plotly_chart(fig_load, use_container_width=True)


# =========================================================================
# PAGE: About
# =========================================================================
elif page == "About":
    st.title("About This Project")

    st.markdown(
        """
        ## Calgary Neighborhood Livability Segmentation

        **Goal:** Group Calgary communities into meaningful livability
        segments using unsupervised machine learning, helping residents,
        urban planners, and policymakers understand the distinct profiles
        of different neighbourhoods.

        ### Methodology

        1. **Multi-Dataset Integration**
           Four datasets from the [Calgary Open Data Portal](https://data.calgary.ca)
           are fetched via the Socrata API and merged at the community level:

           | Dataset | Resource ID | Key Features |
           |---------|-------------|--------------|
           | Civic Census (Age/Gender) | `vsk6-ghca` | Population, median age proxy, gender ratio |
           | Community Crime Statistics | `78gh-n26t` | Total crimes, crime rate |
           | Business Licences | `vdjc-pybd` | Business count, business diversity |
           | Building Permits | `c2es-76ed` | Avg building cost, permit count, housing mix |

        2. **Feature Engineering**
           Ten community-level features are derived, covering demographics,
           safety, economic activity, and built environment.  Missing values
           are imputed with medians and all features are z-score standardised.

        3. **Clustering**
           - *KMeans* with the elbow method and silhouette score to select
             the optimal *k*.
           - *Agglomerative (Ward linkage)* as an alternative hierarchical
             approach.

        4. **Dimensionality Reduction**
           PCA is used to project the 10-dimensional feature space onto 2-D
           for visualisation and to assess how much variance each component
           captures.

        ### How to Run

        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```

        ### Tech Stack
        Python, pandas, scikit-learn, Plotly, Streamlit, sodapy, joblib.

        ---
        *Data source: City of Calgary Open Data Portal (data.calgary.ca)*
        """
    )
