# California Energy Conservation Assistance Act (ECAA): Loan Performance Analysis and Energy Savings Prediction for Public Sector Projects

**Author:** Sarah Akhtar  
**Affiliation:** University of the Pacific  
**Date:** April 20, 2025

---

## Overview

This repository provides a comprehensive analysis of the California Energy Conservation Assistance Act (ECAA) project dataset. The ECAA program, established under Public Resources Code §25410 et seq., funds energy efficiency and renewable energy projects across California. Loans are distributed via two main programs:

- **ECAA-Ed:** Zero-interest loans for K-12 schools
- **ECAA-Reg:** 1% interest loans for cities, counties, tribes, and public institutions

This study uses machine learning and data visualization to examine equity, efficiency, and effectiveness in ECAA funding, with a focus on project status, disadvantaged community (DAC) status, and utility partnerships.

---

## Features

- **Data Exploration:** Summary statistics and distribution analysis of loan amounts, project types, and DAC status.
- **Equity Analysis:** Visualization of funding differences between DAC and non-DAC communities.
- **Geospatial Mapping:** Interactive maps showing the geographic distribution of projects using Folium.
- **Machine Learning:** Predictive modeling of energy savings and project completion using neural networks and ensemble methods.
- **Custom Visualizations:** Boxen plots, strip plots, and interactive maps for stakeholder insights.

---

## Dataset

The main dataset, `energy.csv`, contains information on all ECAA-funded projects reported by the State of California.  
**Key Columns:**
- `Approved_Loan_Amount`: Dollar value of the loan
- `Annual_Electric_Savings`: Projected annual kWh savings
- `Project_Type`: ECAA-Reg or ECAA-Ed
- `DAC`: Whether the project is in a disadvantaged community (Yes/No)
- `Year`: Year of approval
- `Project_Status`: Completed, Active, or Cancelled
- `x`, `y`: Geographic coordinates for mapping

---

## Analysis Highlights

- **Loan Distribution:** Median loan is ~$185,000, with a maximum over $29 million.
- **Project Status:** 96.5% completed, 3.4% active, 0.1% cancelled.
- **Equity:** 22% of projects are in DACs, but funding variability is higher for DACs, especially in ECAA-Ed projects.
- **Geospatial Patterns:** Urban centers like Oakland, Los Angeles, and Sacramento are major hubs for both DAC and non-DAC projects.

---

## Usage

### Requirements

- Python 3.8+
- pandas, numpy, seaborn, matplotlib, folium
- scikit-learn, scipy

### Running the Analysis

1. Clone this repository.
2. Place `energy.csv` in the project root.
3. Run the Jupyter notebook or Python scripts to reproduce the analysis and visualizations.

### Example: Loading the Data

```python
import pandas as pd
energy = pd.read_csv('energy.csv')
print(energy.head())
```

---

## Machine Learning Workflow

- **Regression:** Predict annual energy savings using `MLPRegressor` and `RandomForestRegressor`.
- **Classification:** Predict project completion status with `MLPClassifier`.
- **Preprocessing:** Standardization, feature selection (PCA, SelectKBest), and hyperparameter tuning (GridSearchCV).
- **Validation:** RMSE, R², accuracy, confusion matrix, and classification report.

---

## Key Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.tree import plot_tree
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
```

---

## Acknowledgements

This study was created for Data Analytics Programming taught by Dr. Julia Olivieri. Without her support and guidance, this project would not have been possible.
