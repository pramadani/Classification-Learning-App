import streamlit as st
import numpy as np
import pandas as pd
import io
import sys
from contextlib import contextmanager
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

with st.echo():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm = SVC(C=0.1, gamma=0.1, kernel='poly')
    svm.fit(X_train, y_train)

with st.expander("Show/Hide SVM Parameters and Results"):
    st.write("SVM Parameters: C=0.1, gamma=0.1, kernel='poly'")
    st.write("Training Accuracy: ", svm.score(X_train, y_train))
    st.write("Test Accuracy: ", svm.score(X_test, y_test))


with st.echo():
    y_pred = svm.predict(X_test)
st.write("Classification Report:")
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write(report_df)

with st.echo():
    cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.tight_layout()

st.pyplot(fig)

misclassified_idx = np.where(y_test != y_pred)[0]

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1, edgecolor='k', label='True labels')
plt.scatter(X_test[misclassified_idx, 0], X_test[misclassified_idx, 1], marker='x', s=100, color='r', label='Misclassified')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Misclassified examples')
plt.legend()
plt.colorbar(ticks=[0, 1, 2], label='Classes')
st.pyplot(plt)