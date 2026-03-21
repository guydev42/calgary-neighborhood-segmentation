"""Data loader for Calgary Neighborhood Livability Segmentation.

Fetches multiple datasets from Calgary Open Data via the Socrata API,
caches them locally, and constructs a unified community-level feature
matrix for clustering analysis.

Datasets
--------
- Civic Census by Age/Gender (vsk6-ghca)
- Community Crime Statistics (78gh-n26t)
- Business Licenses (vdjc-pybd)
- Building Permits (c2es-76ed)
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sodapy import Socrata
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOMAIN = "data.calgary.ca"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "census": {
        "resource_id": "vsk6-ghca",
        "filename": "civic_census_age_gender.csv",
        "limit": 50000,
    },
    "crime": {
        "resource_id": "78gh-n26t",
        "filename": "community_crime_stats.csv",
        "limit": 50000,
    },
    "business": {
        "resource_id": "vdjc-pybd",
        "filename": "business_licences.csv",
        "limit": 50000,
    },
    "permits": {
        "resource_id": "c2es-76ed",
        "filename": "building_permits.csv",
        "limit": 50000,
    },
}

FEATURE_COLUMNS = [
    "total_population",
    "median_age_proxy",
    "gender_ratio",
    "total_crimes",
    "crime_rate",
    "business_count",
    "business_diversity",
    "avg_building_cost",
    "permit_count",
    "housing_mix",
]


# ---------------------------------------------------------------------------
# Fetching helpers
# ---------------------------------------------------------------------------

def _fetch_dataset(resource_id: str, limit: int = 50000) -> pd.DataFrame:
    """Fetch a single dataset from the Calgary Open Data portal.

    Parameters
    ----------
    resource_id : str
        Socrata dataset identifier.
    limit : int, optional
        Maximum number of rows to retrieve (default 50 000).

    Returns
    -------
    pd.DataFrame
        Raw dataframe returned by the API.
    """
    client = Socrata(DOMAIN, app_token=None, timeout=60)
    results = client.get(resource_id, limit=limit)
    client.close()
    return pd.DataFrame.from_records(results)


def fetch_and_cache(name: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch a dataset and cache it as CSV in the data/ directory.

    Parameters
    ----------
    name : str
        Key into ``DATASETS`` (e.g. ``"census"``).
    force_refresh : bool, optional
        If True, re-download even when a cached file exists.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    meta = DATASETS[name]
    filepath = DATA_DIR / meta["filename"]

    if filepath.exists() and not force_refresh:
        logger.info("Loading cached %s from %s", name, filepath)
        return pd.read_csv(filepath)

    logger.info("Fetching %s (resource %s) from API ...", name, meta["resource_id"])
    try:
        df = _fetch_dataset(meta["resource_id"], limit=meta["limit"])
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info("Fetched and cached %s (%d rows) to %s", name, len(df), filepath)
        return df
    except Exception as exc:
        logger.error("Failed to fetch %s from Socrata API: %s", name, exc)
        if filepath.exists():
            logger.warning("Falling back to cached %s data.", name)
            return pd.read_csv(filepath)
        raise


def load_all_datasets(force_refresh: bool = False) -> dict:
    """Load all four source datasets.

    Parameters
    ----------
    force_refresh : bool, optional
        If True, re-download all datasets from the API.

    Returns
    -------
    dict
        Mapping of dataset name to its DataFrame.
    """
    return {name: fetch_and_cache(name, force_refresh) for name in DATASETS}


# ---------------------------------------------------------------------------
# Feature engineering per dataset
# ---------------------------------------------------------------------------

def _build_census_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate census data to community-level population features.

    Returns a DataFrame indexed by community with columns:
    total_population, median_age_proxy, gender_ratio.
    """
    df = df.copy()

    # Standardise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Coerce numeric columns
    for col in ["males", "females"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use the most recent year available per community
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        latest_year = df.groupby("code")["year"].transform("max")
        df = df[df["year"] == latest_year]

    # Parse a numeric mid-point from age_range for the age proxy
    def _age_midpoint(val):
        """Return a numeric midpoint from an age-range string."""
        try:
            s = str(val).replace("+", "").replace(" ", "")
            parts = s.split("-")
            nums = [float(p) for p in parts if p.replace(".", "").isdigit()]
            if nums:
                return float(np.mean(nums))
        except Exception:
            pass
        return np.nan

    if "age_range" in df.columns:
        df["age_mid"] = df["age_range"].apply(_age_midpoint)
    else:
        df["age_mid"] = np.nan

    df["total_persons"] = df["males"] + df["females"]

    # Community name column
    comm_col = "code"

    agg = df.groupby(comm_col).agg(
        total_population=("total_persons", "sum"),
        total_males=("males", "sum"),
        total_females=("females", "sum"),
        weighted_age_sum=("age_mid", lambda x: (x * df.loc[x.index, "total_persons"]).sum()),
        total_for_age=("total_persons", "sum"),
    ).reset_index()

    agg["median_age_proxy"] = np.where(
        agg["total_for_age"] > 0,
        agg["weighted_age_sum"] / agg["total_for_age"],
        np.nan,
    )
    agg["gender_ratio"] = np.where(
        agg["total_females"] > 0,
        agg["total_males"] / agg["total_females"],
        np.nan,
    )

    agg = agg.rename(columns={comm_col: "community"})
    agg["community"] = agg["community"].str.strip().str.upper()

    return agg[["community", "total_population", "median_age_proxy", "gender_ratio"]]


def _build_crime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate crime data to community-level crime features.

    Returns a DataFrame with columns: community, total_crimes.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Identify the community column
    comm_col = "community"
    if comm_col not in df.columns:
        for candidate in ["community_name", "communityname"]:
            if candidate in df.columns:
                comm_col = candidate
                break

    # Coerce crime count
    count_col = "crime_count"
    if count_col not in df.columns:
        for candidate in ["count", "crimecount"]:
            if candidate in df.columns:
                count_col = candidate
                break
    if count_col in df.columns:
        df[count_col] = pd.to_numeric(df[count_col], errors="coerce").fillna(0)
    else:
        df[count_col] = 1  # fallback: each row = 1 incident

    agg = df.groupby(comm_col).agg(
        total_crimes=(count_col, "sum"),
    ).reset_index()

    agg = agg.rename(columns={comm_col: "community"})
    agg["community"] = agg["community"].str.strip().str.upper()
    return agg


def _build_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate business licence data to community-level features.

    Returns: community, business_count, business_diversity.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    comm_col = "comdistnm"
    if comm_col not in df.columns:
        for candidate in ["communityname", "community"]:
            if candidate in df.columns:
                comm_col = candidate
                break

    licence_col = "licencetypes"
    if licence_col not in df.columns:
        for candidate in ["licencetype", "licence_type", "licensetype"]:
            if candidate in df.columns:
                licence_col = candidate
                break

    agg = df.groupby(comm_col).agg(
        business_count=(comm_col, "size"),
        business_diversity=(licence_col, "nunique") if licence_col in df.columns else (comm_col, "size"),
    ).reset_index()

    agg = agg.rename(columns={comm_col: "community"})
    agg["community"] = agg["community"].str.strip().str.upper()
    return agg


def _build_permit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate building permit data to community-level features.

    Returns: community, avg_building_cost, permit_count, housing_mix.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    comm_col = "communityname"
    if comm_col not in df.columns:
        for candidate in ["community", "community_name"]:
            if candidate in df.columns:
                comm_col = candidate
                break

    cost_col = "estprojectcost"
    if cost_col not in df.columns:
        for candidate in ["estimatedprojectcost", "est_project_cost"]:
            if candidate in df.columns:
                cost_col = candidate
                break
    if cost_col in df.columns:
        df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

    class_col = "permitclassgroup"
    if class_col not in df.columns:
        for candidate in ["permit_class_group", "permitclass"]:
            if candidate in df.columns:
                class_col = candidate
                break

    agg_dict = {
        "permit_count": (comm_col, "size"),
    }
    if cost_col in df.columns:
        agg_dict["avg_building_cost"] = (cost_col, "mean")
    if class_col in df.columns:
        agg_dict["housing_mix"] = (class_col, "nunique")

    agg = df.groupby(comm_col).agg(**agg_dict).reset_index()

    agg = agg.rename(columns={comm_col: "community"})
    agg["community"] = agg["community"].str.strip().str.upper()

    if "avg_building_cost" not in agg.columns:
        agg["avg_building_cost"] = np.nan
    if "housing_mix" not in agg.columns:
        agg["housing_mix"] = np.nan

    return agg


# ---------------------------------------------------------------------------
# Unified feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    datasets: dict | None = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Build the unified community-level feature matrix.

    Parameters
    ----------
    datasets : dict or None
        Pre-loaded datasets; if None they are fetched/cached automatically.
    force_refresh : bool
        Whether to re-download raw data.

    Returns
    -------
    raw_df : pd.DataFrame
        Feature matrix with community names and original-scale features.
    scaled_df : pd.DataFrame
        Feature matrix with standardised (z-score) features.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler instance.
    """
    if datasets is None:
        datasets = load_all_datasets(force_refresh=force_refresh)

    census_feat = _build_census_features(datasets["census"])
    crime_feat = _build_crime_features(datasets["crime"])
    business_feat = _build_business_features(datasets["business"])
    permit_feat = _build_permit_features(datasets["permits"])

    # Merge all on community
    merged = census_feat.copy()
    for right_df in [crime_feat, business_feat, permit_feat]:
        merged = merged.merge(right_df, on="community", how="outer")

    # Derive crime_rate
    merged["crime_rate"] = np.where(
        merged["total_population"] > 0,
        merged["total_crimes"] / merged["total_population"] * 1000,
        np.nan,
    )

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    # Remove any duplicate column names that may arise from outer merges
    merged = merged.loc[:, ~merged.columns.duplicated()]

    # Keep only communities that have a non-null name
    merged = merged.dropna(subset=["community"])
    merged = merged[merged["community"] != ""]
    merged = merged.drop_duplicates(subset=["community"]).reset_index(drop=True)

    # ---- Imputation (median) ------------------------------------------------
    # keep_empty_features=True prevents dropping all-NaN columns so the
    # output always has the same number of columns as FEATURE_COLUMNS.
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    feature_vals = imputer.fit_transform(merged[FEATURE_COLUMNS])
    raw_df = merged[["community"]].copy()
    # Assign each column individually to avoid shape mismatches
    for i, col in enumerate(FEATURE_COLUMNS):
        raw_df[col] = feature_vals[:, i]

    # ---- Standardisation -----------------------------------------------------
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(raw_df[FEATURE_COLUMNS])
    scaled_df = raw_df[["community"]].copy()
    for i, col in enumerate(FEATURE_COLUMNS):
        scaled_df[col] = scaled_vals[:, i]

    # Persist the ready-to-use matrix
    output_path = DATA_DIR / "community_feature_matrix.csv"
    raw_df.to_csv(output_path, index=False)
    logger.info("Saved feature matrix (%d communities) to %s", len(raw_df), output_path)

    return raw_df, scaled_df, scaler


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raw, scaled, sc = build_feature_matrix()
    print(f"Feature matrix shape: {raw.shape}")
    print(raw.head())
