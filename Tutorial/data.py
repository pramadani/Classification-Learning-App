import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import pandas as pd


# Load dataset Iris
iris = datasets.load_iris()

# Konversi dataset menjadi dataframe
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Menambahkan kolom target ke dataframe
iris_df['class'] = iris.target

# Menampilkan dataframe menggunakan Streamlit
st.write(iris_df)
y = iris.target

# Count class instances
class_counts = np.bincount(y)

# Plot class distribution
plt.figure(figsize=(6, 4))
plt.bar(np.unique(y), class_counts, color=plt.cm.Set1.colors, tick_label=iris.target_names)
plt.xlabel('Classes')
plt.ylabel('Number of instances')
plt.title('Class distribution of Iris dataset')
st.pyplot(plt)

describe = iris_df.describe()

st.write(describe)


st.subheader('Histograms of Iris Dataset Features')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, feature in enumerate(iris.feature_names):
    sns.histplot(data=iris_df, x=feature, kde=True, bins=20, color='skyblue', ax=axes[i])


st.pyplot(fig)

st.subheader('Correlation Heatmap of Iris Dataset Features')
fig_heatmap, ax_heatmap = plt.subplots(figsize=(8, 6))
sns.heatmap(iris_df.corr(), annot=True, cmap='viridis', ax=ax_heatmap)

# Menampilkan plot heatmap menggunakan st.pyplot
st.pyplot(fig_heatmap)
