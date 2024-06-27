import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
import streamlit as st
import pandas as pd

iris = datasets.load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Checking for missing values
miss = iris_df.isnull().sum()
miss_df = pd.DataFrame({'Column': miss.index, 'Missing Values': miss.values})
st.write(miss_df)

iris_df['class'] = iris.target
'''
sebelum
'''
st.write(iris_df)
'''
sesudah
'''

normalized = zscore(iris.data, axis=0)
normalized_df = pd.DataFrame(data=normalized, columns=iris.feature_names)
normalized_df['class'] = iris.target
st.write(normalized_df)