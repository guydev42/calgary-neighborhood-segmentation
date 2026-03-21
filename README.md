# Calgary neighborhood livability segmentation

## Problem statement

Choosing where to live in Calgary involves weighing safety, population density, economic vitality, and housing mix across 200+ communities. This project applies unsupervised machine learning to group communities into distinct livability segments, helping residents, urban planners, and policymakers compare neighbourhoods at a glance.

## Approach

- Integrated four Calgary Open Data sources: census, crime, business licences, and building permits
- Built a 10-feature community-level matrix (population, crime rate, business diversity, etc.)
- Applied KMeans (k=2..10) and Agglomerative clustering with silhouette score selection
- Reduced dimensionality with PCA for visualization and component interpretation
- Built an interactive Streamlit dashboard with radar charts and cluster profiles

## Key results

| Metric | Value |
|--------|-------|
| Best method | KMeans |
| Silhouette score | ~0.42 |
| PCA variance explained (2D) | ~55-65% |

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
project_06_neighborhood_segmentation/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── notebooks/
│   └── 01_eda.ipynb
└── src/
    ├── __init__.py
    ├── data_loader.py
    └── model.py
```
