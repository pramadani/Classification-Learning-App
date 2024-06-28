import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

with st.expander("SVM Parameters and Results", icon=":material/visibility:"):
    st.title("Code:")
    with st.echo():
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        svm = SVC(C=0.1, gamma=0.1, kernel='poly')
        svm.fit(X_train, y_train)
    st.title("Parameter:")
    st.text("SVM Parameters: C=0.1, gamma=0.1, kernel='poly'")
    st.text("Training Accuracy: " + str(svm.score(X_train, y_train)))
    st.text("Test Accuracy: " + str(svm.score(X_test, y_test)))

    st.title("\nClassification Report:")
    y_pred = svm.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=iris.target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)
    
st.title("Iris Species Prediction")
st.text("Set new Iris data in sidebar panel.")

# Input fields for custom data
st.sidebar.header("Input Custom Data")

sepal_length = st.sidebar.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.sidebar.number_input("Petal Width", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

custom_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Button to make prediction
if st.sidebar.button("Predict"):
    custom_prediction = svm.predict(custom_data)
    species = iris.target_names[custom_prediction][0]
    st.markdown(f"<h2 style='color:orange;'>{species}</h2>", unsafe_allow_html=True)
    st.write("is the predicted species.")

with st.expander("Data Distribution", expanded=True, icon=":material/bar_chart:"):
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['species'] = iris.target_names[y]
    custom_df = pd.DataFrame(custom_data, columns=iris.feature_names)
    custom_df['species'] = ['Custom Input']
    plot_df = pd.concat([df, custom_df], ignore_index=True)
    pairplot = sns.pairplot(plot_df, hue='species', palette={'setosa': '#FF6969', 'versicolor': '#F9D689', 'virginica': '#37B7C3', 'Custom Input': 'b'}, markers=['^', 's', 'D', 'o'], height=2.5)
    st.pyplot(pairplot)
