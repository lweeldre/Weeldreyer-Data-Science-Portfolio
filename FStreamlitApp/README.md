
# Unsupervised Machine Learning Streamlit App

## Project Overview

This project is an interactive unsupervised machine learning web application built using Python and Streamlit. The goal of the app is to allow users to explore hidden patterns, relationships, and structures within datasets through hands-on experimentation with unsupervised learning models.

Users can upload their own dataset or select from built-in sample datasets, apply multiple unsupervised machine learning techniques, adjust hyperparameters, and evaluate model performance using visualizations and clustering diagnostics.

This project demonstrates practical skills in unsupervised machine learning, exploratory data analysis, dimensionality reduction, clustering evaluation, and interactive application development.

---

## Libraries Used

- Python
- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

---

## App Features

# Dataset Selection

- Upload your own CSV dataset
- Use built-in sample datasets (Iris, Wine, or custom datasets)
- Automatic missing value removal
- Automatic numeric feature selection for clustering compatibility

---

# Model Selection

The app includes the following unsupervised machine learning models:

## K-Means Clustering

- Groups similar observations into K distinct clusters
- Identifies hidden structures in unlabeled data
- Uses centroid-based partitioning

### Hyperparameters:
- **Maximum K (Elbow Method):** Tests multiple cluster counts
- **Number of Clusters (K):** User selects optimal cluster count

### Performance Feedback:
- Elbow Plot
- Inertia
- Silhouette Score
- Cluster Visualization Scatterplot

---

## Hierarchical Clustering

- Builds nested clusters based on similarity
- Does not require pre-selecting a fixed number of clusters
- Uses agglomerative clustering structure

### Hyperparameters:
- Linkage Method (Ward method used for minimizing variance)

### Performance Feedback:
- Hierarchical Clustering Dendrogram
- Cluster merge distances
- Visual hierarchy of grouped observations

---

## Principal Component Analysis (PCA)

- Reduces dimensionality while preserving important variance
- Simplifies large datasets into principal components
- Supports pattern discovery and visualization

### Hyperparameters:
- Number of Principal Components

### Performance Feedback:
- Explained Variance Ratio
- PCA Explained Variance Graph
- PCA Projection (First Two Components)

---

## Model Evaluation

The app provides multiple performance metrics and visual outputs to help users interpret unsupervised learning effectiveness:

### K-Means:
- **Elbow Method:** Identifies optimal K by minimizing within-cluster variance
- **Silhouette Score:** Measures cluster cohesion and separation
- **Scatterplots:** Visualize cluster structure

### Hierarchical Clustering:
- **Dendrogram:** Shows hierarchical cluster merging
- **Distance Thresholds:** Helps identify natural grouping boundaries

### PCA:
- **Explained Variance Ratio:** Shows how much information each component retains
- **Projection Graphs:** Visualize high-dimensional data in reduced dimensions

---

## Hyperparameter Tuning

Users can interactively adjust model parameters using Streamlit widgets and sliders.

### Examples:
- Increasing K changes clustering granularity
- Different dendrogram structures reveal different cluster relationships
- Increasing PCA components preserves more variance but adds complexity

This interactive tuning allows users to observe how parameter adjustments impact clustering structure and dimensionality reduction in real time.

---

## Steps to Run the App Locally

### 1. Make sure Python is installed on your system

### 2. Install required dependencies:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn scipy

---

## Streamlit App URL


