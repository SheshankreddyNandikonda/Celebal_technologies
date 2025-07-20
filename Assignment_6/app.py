# Install required libraries (if running locally, uncomment)
# !pip install streamlit scikit-learn matplotlib seaborn pandas

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
features = data.feature_names
target_names = data.target_names

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter grids
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    "Random Forest": {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5]
    },
    "SVM": {
        'C': uniform(0.1, 10),
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Training & tuning function
def tune_model(name):
    if name == "Logistic Regression":
        model = LogisticRegression()
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1')
    elif name == "Random Forest":
        model = RandomForestClassifier()
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1')
    elif name == "SVM":
        model = SVC(probability=True)
        grid = RandomizedSearchCV(model, param_distributions=param_grids[name], n_iter=20, cv=5, scoring='f1', random_state=42)
    elif name == "KNN":
        model = KNeighborsClassifier()
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1')
    else:
        return None

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    return {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'Best Params': grid.best_params_,
        'Best Estimator': best_model,
        'Predictions': y_pred
    }

# Run all models
model_names = ["Logistic Regression", "Random Forest", "SVM", "KNN"]
results = [tune_model(name) for name in model_names]

df_results = pd.DataFrame(results)

st.title("🔍 ML Model Comparison with Hyperparameter Tuning")

# Show metric results
st.subheader("Model Performance Table")
st.dataframe(df_results[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']])

# Heatmap of metrics
st.subheader(" Performance Heatmap")
metrics = df_results[['Accuracy', 'Precision', 'Recall', 'F1 Score']].set_index(df_results['Model'])
fig, ax = plt.subplots()
sns.heatmap(metrics, annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
st.pyplot(fig)

# Bar plot
st.subheader("Metric Comparison Bar Chart")
metrics.plot(kind='bar', figsize=(10, 5), colormap='viridis')
plt.title("Evaluation Metrics by Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.legend(loc='lower right')
st.pyplot(plt)

# Confusion matrix and ROC for best model
best_result = max(results, key=lambda x: x['F1 Score'])
best_model = best_result['Best Estimator']
y_pred = best_result['Predictions']
st.subheader(f"Best Model: {best_result['Model']}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
ax_cm.set_title("Confusion Matrix")
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ROC Curve
def plot_roc_curve(model, X_test, y_test, label):
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    st.pyplot(fig)

st.subheader("ROC Curve")
plot_roc_curve(best_model, X_test, y_test, best_result['Model'])

# Pie chart of predictions
st.subheader("Prediction Distribution")
labels, counts = np.unique(y_pred, return_counts=True)
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(counts, labels=[f"{label} ({count})" for label, count in zip(target_names, counts)],
           autopct='%1.1f%%', startangle=140)
st.pyplot(fig_pie)

# Show best hyperparameters
st.subheader("Best Hyperparameters")
for r in results:
    st.write(f"**{r['Model']}**: {r['Best Params']}")
