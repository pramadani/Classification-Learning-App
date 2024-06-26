import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
import streamlit as st
import numpy as np
import pandas as pd
import io
import sys
from contextlib import contextmanager
from sklearn.svm import SVC

@contextmanager
def capture_stdout():
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    try:
        yield new_stdout
    finally:
        sys.stdout = old_stdout

# Load dataset Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Visualize train-test split
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

scatter1 = ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis')
ax[0].set_title('Train set')
ax[0].set_xlabel('Sepal length')
ax[0].set_ylabel('Sepal width')
plt.colorbar(scatter1, ax=ax[0], ticks=[0, 1, 2], label='Classes')

scatter2 = ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')
ax[1].set_title('Test set')
ax[1].set_xlabel('Sepal length')
ax[1].set_ylabel('Sepal width')
plt.colorbar(scatter2, ax=ax[1], ticks=[0, 1, 2], label='Classes')

plt.tight_layout()

st.pyplot(fig)

param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001],  # Kernel coefficient
    'kernel': ['linear', 'rbf', 'poly']  # Kernel type
}

# Perform GridSearchCV with verbose output
grid_svm = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)

with capture_stdout() as verbose_output:
    grid_svm.fit(X_train, y_train)

# Retrieve verbose output
verbose_result = verbose_output.getvalue()

# Print best parameters
with st.expander("Show/Hide Best Parameters"):
    st.write("Best parameters found: ", grid_svm.best_params_)

with st.expander("GridSearchCV Verbose Output"):
    st.text_area('', verbose_result, height=400)
# Display verbose output in Streamlit