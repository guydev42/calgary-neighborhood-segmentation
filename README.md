<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Neighborhood%20Livability%20Segmentation&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Unsupervised%20clustering%20of%20200%2B%20Calgary%20communities&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/K--Means-Silhouette_0.42-blue?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/PCA-Dimensionality_Reduction-228B22?style=for-the-badge" />
  <img src="https://img.shields.io/badge/scikit--learn-ML_Pipeline-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- Choosing where to live in Calgary involves weighing safety, population density, economic vitality, and housing mix across 200+ communities. No single data source captures the full livability picture.
>
> **Solution** -- This project integrates four Calgary Open Data sources (census, crime, business licences, building permits) and applies K-Means clustering with PCA to group communities into distinct livability segments.
>
> **Impact** -- Helps residents, urban planners, and policymakers compare neighbourhoods at a glance through data-driven livability profiles with radar charts and cluster maps.

---

## Results

| Metric | Value |
|--------|-------|
| Best method | K-Means |
| Silhouette score | ~0.42 |
| PCA variance explained (2D) | ~55--65% |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  Census + Crime  │────>│  10-feature      │────>│  K-Means       │────>│  Streamlit      │
│  Data (Socrata) │     │  + Business +    │     │  community       │     │  clustering    │     │  dashboard      │
│  4 datasets     │     │  Permits merge   │     │  matrix          │     │  PCA reduction │     │  Radar charts   │
│  200+ areas     │     │  Per-community   │     │  Standardized    │     │  Agglomerative │     │  Cluster map    │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_06_neighborhood_segmentation/
├── app.py                              # Streamlit dashboard
├── index.html                          # Static landing page
├── requirements.txt                    # Python dependencies
├── README.md
├── data/
│   ├── building_permits.csv            # Building permit data
│   ├── business_licences.csv           # Business licence data
│   ├── civic_census_age_gender.csv     # Census demographics
│   ├── community_crime_stats.csv       # Community crime statistics
│   └── community_feature_matrix.csv    # Engineered feature matrix
├── models/                             # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb                    # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py                  # Data fetching & integration
    └── model.py                        # Clustering & PCA
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/neighborhood-segmentation.git
cd neighborhood-segmentation

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data](https://data.calgary.ca/) -- Census, Crime, Business Licences, Building Permits |
| Communities | 200+ |
| Access method | Socrata API (sodapy) |
| Integrated features | Population, crime rate, business diversity, permit activity, demographics (10 features) |
| Output | Community livability segments (cluster labels) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data integration

- Integrated four Calgary Open Data sources: civic census, crime statistics, business licences, and building permits
- Aggregated all datasets at the community level to create a unified feature matrix
- Handled communities with missing data through imputation and filtering

### Feature matrix construction

- Built a 10-feature community-level matrix covering population, crime rate, business diversity, permit activity, age distribution, and housing mix
- Standardized all features with z-score normalization for clustering

### Clustering and dimensionality reduction

- Applied K-Means with k=2 to 10 and selected optimal k using silhouette score analysis
- Compared with Agglomerative clustering for validation
- Reduced dimensionality with PCA for 2D visualization and component interpretation
- PCA explained ~55--65% of variance in two components

### Cluster profiling

- Profiled each cluster with summary statistics and radar charts
- Identified distinct community archetypes: urban core, suburban family, established residential, etc.
- Silhouette score of ~0.42 indicates moderate-to-good cluster separation

### Interactive dashboard

- Built a Streamlit dashboard with radar charts, cluster profiles, and community comparison tools
- PCA scatter plot with interactive hover for community-level detail

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing census, crime, business, and permit datasets
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
