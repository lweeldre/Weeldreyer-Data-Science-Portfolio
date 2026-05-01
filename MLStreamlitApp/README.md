# Machine Learning Streamlit App

## Project Overview

This project is an interactive machine learning web application built using Python and Streamlit. The goal of the app is to allow users to explore supervised machine learning models in a hands-on way.

Users can upload their own dataset or select from built-in sample datasets, choose a target variable, train different models, adjust hyperparameters, and evaluate model performance using multiple metrics and visualizations.

This project demonstrates practical skills in machine learning, data preprocessing, model evaluation, and interactive application development.

## Libraries Used

- Python
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## App Features

### Dataset Selection
- Upload your own CSV dataset
- Use built-in datasets (Penguins, Iris, Breast Cancer)

### Model Selection

The app includes the following supervised machine learning models:

#### Logistic Regression
- Used for classification problems
- Models probabilities using a logistic function
- Hyperparameter:
  - **C (Regularization Strength)**: Controls overfitting

#### Decision Tree
- Splits data based on feature values to make predictions
- Can capture non-linear relationships
- Hyperparameter:
  - **Max Depth**: Controls complexity of the tree

#### K-Nearest Neighbors (KNN)
- Classifies data based on similarity to nearby points
- Hyperparameter:
  - **K (Number of Neighbors)**: Controls how many neighbors influence predictions

---

## Model Evaluation

The app provides multiple performance metrics:

- **Accuracy**: Overall correctness of predictions  
- **Precision**: Accuracy of positive predictions  
- **Recall**: Ability to identify all positive cases  
- **F1-score**: Balance between precision and recall  

### Confusion Matrix
Displays correct vs incorrect classifications for each class.

### ROC Curve and AUC
- Shows model performance across classification thresholds  
- Supports both binary and multiclass classification (one-vs-rest approach)  
- AUC values closer to 1 indicate better performance  

---

## Hyperparameter Tuning

Users can interactively adjust model parameters using sliders in the app interface. This allows users to observe how parameter changes affect model performance in real time.

Examples:
- Increasing tree depth may improve accuracy but risk overfitting
- Changing K in KNN affects model sensitivity to noise
- Adjusting regularization in Logistic Regression controls model complexity

---

## References

- Streamlit Documentation: https://docs.streamlit.io  
- Scikit-learn Documentation: https://scikit-learn.org/stable/  
- ROC Curve Explanation: https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics  
- Pandas Documentation: https://pandas.pydata.org/docs/  

---

## Project Significance

This project demonstrates the ability to:
- Build interactive data science applications
- Implement and compare machine learning models
- Perform data preprocessing and feature engineering
- Evaluate models using multiple performance metrics
- Deploy applications to the cloud using Streamlit

It highlights practical, real-world data science skills and complements my complete data science portfolio.

---

# Streamlit App URL

[Streamlit App]( https://weeldreyer-ml-app.streamlit.app/)
