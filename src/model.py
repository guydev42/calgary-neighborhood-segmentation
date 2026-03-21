"""Clustering models for Calgary Neighborhood Livability Segmentation.

Provides KMeans and Agglomerative clustering, PCA for dimensionality
reduction, silhouette-score evaluation, cluster profiling, and
persistence utilities.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Elbow method
# ---------------------------------------------------------------------------

def compute_elbow(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Run KMeans for a range of *k* and return inertia values.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    k_range : range
        Range of cluster counts to evaluate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``inertia``, ``silhouette``.
    """
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        records.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.info("k=%d  inertia=%.2f  silhouette=%.4f", k, km.inertia_, sil)
    return pd.DataFrame(records)


def find_optimal_k(elbow_df: pd.DataFrame) -> int:
    """Select optimal *k* as the one with the highest silhouette score.

    Parameters
    ----------
    elbow_df : pd.DataFrame
        Output of :func:`compute_elbow`.

    Returns
    -------
    int
        Optimal number of clusters.
    """
    best_row = elbow_df.loc[elbow_df["silhouette"].idxmax()]
    return int(best_row["k"])


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def train_kmeans(
    X: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> KMeans:
    """Fit a KMeans model and return it.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed.

    Returns
    -------
    KMeans
        Fitted KMeans estimator.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    model.fit(X)
    logger.info(
        "KMeans (k=%d): inertia=%.2f, silhouette=%.4f",
        n_clusters,
        model.inertia_,
        silhouette_score(X, model.labels_),
    )
    return model


def train_agglomerative(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> AgglomerativeClustering:
    """Fit an Agglomerative Clustering model.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_clusters : int
        Number of clusters.
    linkage : str
        Linkage criterion (default ``"ward"``).

    Returns
    -------
    AgglomerativeClustering
        Fitted model.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(X)
    sil = silhouette_score(X, model.labels_)
    logger.info(
        "Agglomerative (k=%d, linkage=%s): silhouette=%.4f",
        n_clusters,
        linkage,
        sil,
    )
    return model


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def fit_pca(
    X: np.ndarray,
    n_components: Optional[int] = None,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA and return the transformer plus transformed data.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    n_components : int or None
        Number of principal components. Defaults to ``min(X.shape)``.

    Returns
    -------
    pca : PCA
        Fitted PCA transformer.
    X_pca : np.ndarray
        Transformed data in the PC space.
    """
    if n_components is None:
        n_components = min(X.shape)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)
    logger.info(
        "PCA: %d components explain %.1f%% variance",
        n_components,
        pca.explained_variance_ratio_.sum() * 100,
    )
    return pca, X_pca


def get_pca_loadings(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """Return a DataFrame of PCA loadings (components x features).

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    feature_names : list[str]
        Names of original features.

    Returns
    -------
    pd.DataFrame
        Loadings matrix with PC labels as index and feature names as columns.
    """
    n_components = pca.n_components_
    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(
        pca.components_,
        index=pc_labels,
        columns=feature_names,
    )


# ---------------------------------------------------------------------------
# Cluster profiling
# ---------------------------------------------------------------------------

def profile_clusters(
    raw_df: pd.DataFrame,
    labels: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Generate summary statistics per cluster.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Original-scale feature matrix (must include *feature_columns*).
    labels : np.ndarray
        Cluster assignments aligned with ``raw_df``.
    feature_columns : list[str]
        Feature column names to summarise.

    Returns
    -------
    pd.DataFrame
        Mean feature values per cluster plus a ``cluster_size`` column.
    """
    df = raw_df.copy()
    df["cluster"] = labels

    profile = df.groupby("cluster")[feature_columns].mean()
    profile["cluster_size"] = df.groupby("cluster")["cluster"].transform("size").drop_duplicates().values
    # Safer cluster size
    size_series = df.groupby("cluster").size()
    profile["cluster_size"] = size_series.values

    return profile.reset_index()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(obj: object, filename: str) -> Path:
    """Persist a model or transformer to the models/ directory.

    Parameters
    ----------
    obj : object
        Any picklable object (model, scaler, PCA, ...).
    filename : str
        Destination file name inside ``models/``.

    Returns
    -------
    Path
        Full path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    joblib.dump(obj, path)
    logger.info("Saved %s to %s", type(obj).__name__, path)
    return path


def load_model(filename: str) -> object:
    """Load a previously saved model from the models/ directory.

    Parameters
    ----------
    filename : str
        Name of the file inside ``models/``.

    Returns
    -------
    object
        The deserialized model.
    """
    path = MODELS_DIR / filename
    obj = joblib.load(path)
    logger.info("Loaded %s from %s", type(obj).__name__, path)
    return obj


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from data_loader import build_feature_matrix, FEATURE_COLUMNS

    raw_df, scaled_df, scaler = build_feature_matrix()
    X = scaled_df[FEATURE_COLUMNS].values

    # Elbow analysis
    elbow_df = compute_elbow(X)
    optimal_k = find_optimal_k(elbow_df)
    print(f"Optimal k: {optimal_k}")

    # KMeans
    km = train_kmeans(X, n_clusters=optimal_k)
    save_model(km, "kmeans_model.joblib")

    # Agglomerative
    agg = train_agglomerative(X, n_clusters=optimal_k)
    save_model(agg, "agglomerative_model.joblib")

    # PCA
    pca, X_pca = fit_pca(X)
    save_model(pca, "pca_transformer.joblib")
    save_model(scaler, "feature_scaler.joblib")

    # Profiling
    profile = profile_clusters(raw_df, km.labels_, FEATURE_COLUMNS)
    print("\nCluster profiles:")
    print(profile.to_string(index=False))
