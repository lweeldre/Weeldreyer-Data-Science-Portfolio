#Importing Libraries:
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage

# Title:
st.title("Unsupervised Machine Learning Explorer")

st.markdown("""
This application allows users to explore unsupervised machine learning techniques through interactive controls.

### Users Can:
- Upload their own dataset or use sample datasets
- Apply K-Means Clustering
- Perform Hierarchical Clustering
- Reduce dimensions with PCA
- Analyze model performance with visualizations
""")

# Dataset Selection:
dataset_option = st.selectbox(
    "Choose Dataset Source",
    ["Upload your own CSV", "Use sample dataset"]
)

# Sample datasets
if dataset_option == "Use sample dataset":
    sample_choice = st.selectbox(
        "Choose a sample dataset",
        ["Iris", "Wine"]
    )

    if sample_choice == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame

    elif sample_choice == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine(as_frame=True)
        df = data.frame

elif dataset_option == "Upload your own CSV":
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

# Main App
if 'df' in locals():

    # Drop missing values
    df = df.dropna()

    # Keep numeric columns only
    df_numeric = df.select_dtypes(include=np.number)

    st.subheader("Dataset Preview")
    st.write(df_numeric.head())

    # Standardize Data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # Technique Selection
    model_choice = st.selectbox(
        "Choose Unsupervised Learning Technique",
        ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis (PCA)"]
    )

    # K-MEANS
    if model_choice == "K-Means Clustering":

        st.markdown("### K-Means Clustering")

        st.markdown("""
        K-Means Clustering is an unsupervised machine learning algorithm used to group similar observations into distinct clusters based on their feature similarity.

        Unlike supervised learning, K-Means does not use labeled outcomes. Instead, it identifies hidden patterns by assigning data points to groups whose members are more similar to one another than to points in other groups.

        ### How K-Means Works:
        1. Choose a value for **K** (the number of clusters)
        2. Randomly initialize cluster centers (centroids)
        3. Assign each data point to the nearest centroid
        4. Recalculate centroid positions
        5. Repeat until clusters stabilize

        **K** represents the number of clusters you want the algorithm to identify in the dataset.

        - Small K → broader, fewer groups
        - Large K → more detailed, narrower groups

        Choosing the correct K is important because too few clusters may oversimplify the data, while too many may overfit random variation.
        """)

        max_k = st.slider("Select maximum K for Elbow Method", 2, 10, 6)

        inertia = []

        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)

        # Elbow Plot

        st.markdown("""
        ### Elbow Method

        The Elbow Method helps determine the optimal number of clusters by plotting the relationship between:

        - **X-axis:** Number of clusters (K)
        - **Y-axis:** Inertia (Within-Cluster Sum of Squares)

        Inertia measures how tightly grouped data points are within each cluster.

        - Lower inertia = tighter clusters
        - Higher inertia = more spread-out clusters

        ### How to Interpret the Graph:
        As K increases, inertia decreases because clusters become smaller and more specialized.

        The “elbow point” is where the rate of improvement sharply slows down.

        This point often represents the best balance between:
        - Model simplicity
        - Cluster quality
        """)

        fig, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), inertia, marker='o')
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia")
        ax.set_title("Elbow Method")
        st.pyplot(fig)

        st.markdown("""
        ### Number of Clusters

        After reviewing the Elbow Plot, you can choose the number of clusters that best represents your data structure.

        This selection directly affects:
        - Group separation
        - Interpretability
        - Model complexity
        """)

        k = st.slider("Choose Number of Clusters", 2, max_k, 3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_data)

        st.markdown("""
        ### Silhouette Score

        The Silhouette Score measures how well-separated and cohesive your clusters are.

        ### Score Range:
        - **+1** → Strong, well-separated clusters
        - **0** → Overlapping clusters
        - **-1** → Poor clustering

        ### Why It Matters:
        A high silhouette score suggests:
        - Data points are close to their own cluster
        - Data points are far from neighboring clusters

        This provides a more meaningful evaluation than inertia alone.
        """)

        silhouette = silhouette_score(scaled_data, clusters)

        st.write(f"Silhouette Score: {silhouette:.2f}")

        # PCA for 2D visualization

        st.markdown("""
        ### K-Means Cluster Visualization

        This scatterplot displays your clustered data in two dimensions.

        Because many datasets contain multiple variables, dimensionality reduction (such as PCA) is often used to project data into 2D for visualization.

        ### What the Graph Shows:
        - Each point = one observation
        - Each color = assigned cluster
        - Distance between groups = cluster separation

        ### Interpretation:
        Well-separated colored groups suggest stronger clustering performance.
        Overlapping groups may indicate weak or unclear segmentation.
        """)

        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(scaled_data)

        fig, ax = plt.subplots()
        scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters)
        ax.set_title("K-Means Cluster Visualization")
        st.pyplot(fig)


    # HIERARCHICAL
    elif model_choice == "Hierarchical Clustering":

        st.markdown("### Hierarchical Clustering")

        st.markdown("""
        Hierarchical Clustering is an unsupervised learning technique that builds clusters step-by-step by measuring similarity between observations.

        Unlike K-Means, Hierarchical Clustering does not require you to pre-select the number of clusters.

        ### Two Main Approaches:
        - **Agglomerative:** Start with individual points and merge them
        - **Divisive:** Start with one large cluster and split it

        This app typically uses agglomerative clustering.
        """)

        st.markdown("""
        ### Hierarchical Clustering Dendrogram

        A dendrogram is a tree-like diagram that visualizes how clusters merge over time.

        ### What the Graph Shows:
        - Bottom = individual observations
        - Branches = merges between points or groups
        - Height = distance (dissimilarity)

        ### Interpretation:
        - Short branches → highly similar groups
        - Tall branches → more distinct groups

        ### Why It's Useful:
        Dendrograms help identify natural groupings and determine where clusters should be separated based on vertical distance.
        """)

        st.markdown("""
        ### Adaptive Hierarchical Clustering Dendrogram

        This dendrogram automatically adjusts visualization settings based on dataset size so it remains readable across multiple datasets.

        ### Design Logic:
        - Small datasets → show full labels
        - Medium datasets → smaller font + rotation
        - Large datasets → truncate branches for readability
        """)

        # Create linkage matrix
        linked = linkage(scaled_data, method='ward')

        # Determine dataset size
        num_samples = len(df_numeric)

        # Optional label source:
        # Uses index by default, but if your dataset has a descriptive first column
        # like Country or CustomerID, you can substitute it.
        labels = df.index.astype(str).tolist()

        fig, ax = plt.subplots(figsize=(18, 8))

        # SMALL DATASETS
        if num_samples <= 40:
            dendrogram(
               linked,
               labels=labels,
               leaf_rotation=90,
               leaf_font_size=8,
               ax=ax
            )

        # MEDIUM DATASETS
        elif num_samples <= 150:
           dendrogram(
               linked,
               labels=labels,
               leaf_rotation=90,
               leaf_font_size=5,
               ax=ax
            )

        # LARGE DATASETS
        else:
            dendrogram(
              linked,
                truncate_mode='lastp',   # Show only final branches
                p=40,                    # Number of displayed groups
                leaf_rotation=90,
                leaf_font_size=8,
                show_contracted=True,
                ax=ax
            )

        #Titles and labels
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Data Points / Cluster Groupings")
        ax.set_ylabel("Euclidean Distance")

        st.pyplot(fig)

        #Interpretation Guide
        st.markdown("""
        ### How to Read This Dendrogram

        A dendrogram visualizes how observations are grouped based on similarity.

        ### Key Insights:
        - **Lower merges** → observations are highly similar
        - **Higher merges** → observations are more different
        - **Large vertical jumps** → may indicate natural cluster boundaries
        """)

    # PCA
    elif model_choice == "Principal Component Analysis (PCA)":

        st.markdown("### Principal Component Analysis")

        st.markdown("""
        Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique used to simplify complex datasets while preserving as much important information as possible.

        Many datasets contain numerous variables, which can make visualization and interpretation difficult.

        PCA transforms original variables into new variables called **principal components**.

        ### Principal Components:
        - PC1 = captures the most variance
        - PC2 = captures the second most variance
        - Additional PCs capture progressively less variance

        ### Why PCA Matters:
        - Reduces complexity
        - Improves visualization
        - Helps identify patterns
        - Removes redundancy
        """)

        st.markdown("""
        ### Number of Principal Components

        This setting determines how many transformed dimensions to retain.

        ### Tradeoff:
        - Fewer components → simpler representation, less information
        - More components → more detail, greater complexity

        The goal is usually to preserve most variance with fewer dimensions.
        """)


        num_components = st.slider(
            "Select Number of Principal Components",
            2,
            min(len(df_numeric.columns), 5),
            2
        )

        pca = PCA(n_components=num_components)
        pca_data = pca.fit_transform(scaled_data)

        explained_variance = pca.explained_variance_ratio_

        st.markdown("""
        ### Explained Variance Ratio

        The Explained Variance Ratio measures how much information each principal component retains from the original dataset.

        ### Example:
        If PC1 explains 70% variance:
        - PC1 alone captures most of the dataset’s structure

        ### Why It Matters:
        Higher explained variance means the component is more informative.
        """)

        st.write("Explained Variance Ratio:")
        st.write(explained_variance)

        st.markdown("""
        ### PCA Explained Variance Graph

        This bar chart shows how much variance each principal component contributes.

        ### What the Graph Shows:
        - Taller bars = more important components
        - Shorter bars = less useful components

        ### Interpretation:
        If the first few bars are large, the dataset can be simplified effectively with minimal information loss.
        """)

        fig, ax = plt.subplots()
        ax.bar(range(1, num_components + 1), explained_variance)
        ax.set_xlabel("Principal Components")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance")
        st.pyplot(fig)

        st.markdown("""
        ### PCA Projection (First Two Components)

        This scatterplot visualizes the dataset using the first two principal components.

        ### What the Graph Represents:
        - Each point = one observation
        - X-axis = Principal Component 1
        - Y-axis = Principal Component 2

        ### Why It’s Useful:
        This allows high-dimensional data to be visualized in two dimensions.

        ### Interpretation:
        - Clear groupings may suggest natural clusters
        - Overlap may suggest weak separation
        - Spread indicates variation within the data
        """)

        if num_components >= 2:
            fig, ax = plt.subplots()
            ax.scatter(pca_data[:, 0], pca_data[:, 1])
            ax.set_title("PCA Projection (First Two Components)")
            st.pyplot(fig)