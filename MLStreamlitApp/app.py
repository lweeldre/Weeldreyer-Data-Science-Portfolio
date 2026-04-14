# Importing important Libraries:
import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_curve, auc)

from sklearn.preprocessing import label_binarize

# Markdown Hashtags:
st.title("Machine Learning Model Explorer")

st.markdown("""
This application allows users to explore supervised machine learning by uploading their own dataset or selecting a sample dataset.

Users can:
- Select a target variable
- Train different machine learning models
- Adjust hyperparameters
- Evaluate model performance using metrics and visualizations

This interactive approach helps demonstrate how model choice and parameter tuning impact predictive performance.
""")

st.write("Upload your dataset and experiment with machine learning models:")

dataset_option = st.selectbox(
    "Choose Dataset Source",
    ["Upload your own CSV", "Use sample dataset"]
)

# Sample datasets
if dataset_option == "Use sample dataset":

    sample_choice = st.selectbox(
        "Choose a sample dataset",
        ["Penguins", "Iris", "Breast Cancer"]
    )

    if sample_choice == "Penguins":
        #pd.read_csv("MLStreamlitApp/data/penguins.csv")
        #df = pd.read_csv("data/penguins.csv")
        st.write("Penguins dataset: classify penguin species based on physical measurements.")


        import pandas as pd

        # Load dataset
        df = pd.read_csv("MLStreamlitApp/data/penguins.csv")

        # Drop missing values
        df = df.dropna()

        # Convert categorical columns to numeric
        df = pd.get_dummies(df, drop_first=True)

        # Show preview (so you know it's working)
        st.write("Penguins Dataset Preview:")
        st.write(df.head())






    elif sample_choice == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        df = data.frame
        st.write("Iris dataset: classify flowers into three species.")

    elif sample_choice == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        st.write("Breast cancer dataset: predict malignant vs benign tumors.")

# Uploading Options:
elif dataset_option == "Upload your own CSV":

    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if 'df' in locals():

    # Cleaning Data:
    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    st.markdown("""
    ### Dataset Overview

    Below is a preview of the dataset. Each row is an observation, and each column is a feature.

    You will select one column as the **target variable** for prediction.
    """)

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.markdown("""
    ### Target Variable Selection

    Choose the variable you want the model to predict. This is the **target variable**.

    All other columns will be used as input features.
    """)

    target = st.selectbox("Choose Target Variable", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Handling categorical Variables Automatically:
    X = pd.get_dummies(X)

    # Train/Test split slider:
    test_size = st.slider("Test Size Percentage", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Hyperparameter tuning:
    st.markdown("""
    ### Model Selection

    Choose a machine learning model:

    - **Logistic Regression**: Good for linear classification problems
    - **Decision Tree**: Splits data based on feature values
    - **KNN**: Classifies based on nearest neighbors
    """)
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Decision Tree", "KNN"])
    if model_choice == "Logistic Regression":
        st.markdown("""

        Logistic Regression is a supervised learning algorithm used for **classification problems**.

        It models the probability that an observation belongs to a particular class using a logistic (S-shaped) function. The model calculates a weighted combination of input features and transforms it into a probability between 0 and 1.

        A decision threshold (commonly 0.5) is then used to assign a final class label.

        This model works best when the relationship between the features and the log-odds of the outcome is approximately linear.

        **Key Strengths:**
        - Interpretable and simple
        - Performs well on linearly separable data

        **Limitations:**
        - Struggles with complex, non-linear relationships
        """)
        st.markdown("""
        **Regularization Strength (C):**  
        Controls how much the model avoids overfitting.
        """)

        C = st.slider("Regularization Strength", 0.01, 10.0, 1.0)

        model = LogisticRegression(C=C, max_iter=1000, class_weight = 'balanced')

    elif model_choice == "Decision Tree":
        st.markdown("""
        ### Decision Tree

        A Decision Tree is a supervised learning model that makes predictions by recursively splitting the dataset based on feature values.

        At each step, the model selects the feature and threshold that best separates the data into distinct classes. This process continues until a stopping condition is reached (such as maximum depth).

        The result is a tree-like structure where:
        - Internal nodes represent decisions based on features
        - Branches represent outcomes of those decisions
        - Leaf nodes represent final predictions

        **Key Strengths:**
        - Easy to interpret and visualize
        - Can capture non-linear relationships
        - Requires little data preprocessing

        **Limitations:**
        - Prone to overfitting if the tree is too deep
        - Can be unstable with small changes in data
        """)
        st.markdown("""
        **Max Depth:**  
        Controls how complex the tree can become.
        """)

        max_depth = st.slider("Max Depth", 1, 20, 5)

        model = DecisionTreeClassifier(max_depth=max_depth)

    elif model_choice == "KNN":
        st.markdown("""
        ### K-Nearest Neighbors (KNN)

        K-Nearest Neighbors is a non-parametric algorithm that makes predictions based on the similarity between data points.

        To classify a new observation, the model:
        1. Calculates the distance between the new point and all training points
        2. Identifies the K closest neighbors
        3. Assigns the most common class among those neighbors

        The choice of K controls the model’s behavior:
        - Small K → more sensitive to noise
        - Large K → smoother but less flexible

        **Key Strengths:**
        - Simple and intuitive
        - No training phase (lazy learner)
        - Works well with well-separated data

        **Limitations:**
        - Computationally expensive with large datasets
        - Sensitive to feature scaling and irrelevant features
        """)
        st.markdown("""
        **Number of Neighbors (K):**  
        Determines how many nearby points influence predictions.
        """)

        neighbors = st.slider("Number of Neighbors", 1, 15, 5)

        model = KNeighborsClassifier(n_neighbors=neighbors)

    # Train model
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.markdown("""
    ### Model Performance

    The model is evaluated on test data to measure how well it generalizes.
    """)

    st.write(f"Accuracy: {accuracy:.2f}")

    st.markdown("""
    - **Accuracy**: Overall correctness  
    - **Precision**: Correct positive predictions  
    - **Recall**: Ability to find all positives  
    - **F1-score**: Balance between precision and recall  
    """)

    st.text("Classification Report:")
    st.text(classification_report(y_test, predictions))

    st.markdown("""
    ### Confusion Matrix

    A confusion matrix is a table used to evaluate the performance of a classification model by comparing predicted labels to actual labels.

    Each row represents the actual class, and each column represents the predicted class.

    - Diagonal values represent correct predictions
    - Off-diagonal values represent misclassifications

    From the confusion matrix, we can derive key performance metrics such as:
    - Precision
    - Recall
    - F1-score

    The confusion matrix is especially useful for identifying **which specific classes the model struggles with**, rather than just looking at overall accuracy.
    """)

    if len(np.unique(y)) > 10:
        st.error("Too many unique target values detected. Please choose a categorical target variable for classification.")
        st.stop()

    import seaborn as sns
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.markdown("""
    ### ROC Curve

    The ROC curve is a graphical representation of a classification model’s performance across different decision thresholds.

    It plots:
    - **True Positive Rate (Recall)** on the y-axis
    - **False Positive Rate** on the x-axis

    Each point on the curve represents a different threshold used to classify observations.

    The **Area Under the Curve (AUC)** summarizes performance:
    - AUC = 1.0 → perfect model
    - AUC = 0.5 → random guessing

    For multiclass problems, a separate ROC curve is generated for each class using a one-vs-rest approach.

    The ROC curve is useful because it shows how well the model balances:
    - Detecting true positives
    - Avoiding false positives

    This provides a more complete evaluation than accuracy alone.
    """)

    import numpy as np
    classes = sorted(np.unique(y_test))

    y_probs = model.predict_proba(X_test)

    fig, ax = plt.subplots()

    if len(classes) == 2:

        fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")

    else:

        y_test_bin = label_binarize(y_test, classes=classes)

        for i in range(len(classes)):

            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, label=f"{classes[i]} AUC = {roc_auc:.2f}")

    ax.plot([0, 1], [0, 1], linestyle='--')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    st.pyplot(fig)

    st.markdown("""
    **Note:** Accuracy alone can be misleading, especially with imbalanced datasets. Metrics like precision, recall, and AUC provide a more nuanced evaluation of model performance.
    """)