import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm = SVC(C=0.1, gamma=0.1, kernel='poly')
svm.fit(X_train, y_train)

# Streamlit app
st.title("Iris Species Prediction")

# Input fields for custom data
st.sidebar.header("Input Custom Data")

sepal_length = st.sidebar.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.sidebar.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.sidebar.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.sidebar.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.0)

custom_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Button to make prediction
if st.sidebar.button("Predict"):
    custom_prediction = svm.predict(custom_data)
    species = iris.target_names[custom_prediction][0]
    st.write(f"The predicted species is: {species}")

# Predictions and classification report
y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write(report_df)

# Visualizing all feature combinations with pairplot
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = iris.target_names[y]

# Add custom input to the dataframe for visualization
custom_df = pd.DataFrame(custom_data, columns=iris.feature_names)
custom_df['species'] = ['Custom Input']

# Combine custom input with original data
plot_df = pd.concat([df, custom_df], ignore_index=True)

# Pairplot with hue based on species
pairplot = sns.pairplot(plot_df, hue='species', palette='Set1', markers=['o', 's', 'D', 'X'], height=2.5)

# Set custom marker size for the custom input point
for i, j in zip(*np.triu_indices_from(pairplot.axes, 1)):
    pairplot.axes[i, j].scatter(custom_df.iloc[:, j], custom_df.iloc[:, i], marker='X', color='red', edgecolor='k', s=100)

st.pyplot(pairplot)
