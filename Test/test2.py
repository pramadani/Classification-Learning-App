import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Judul aplikasi
st.title('Dashboard dengan Enam Grafik dengan Lebar Kolom Berbeda dan Tata Letak Fleksibel')

# Membuat data sample untuk grafik
data1 = pd.DataFrame({'x': range(1, 11), 'y': np.random.randn(10)})
data2 = pd.DataFrame({'x': range(1, 11), 'z': np.random.randn(10)})
data3 = pd.DataFrame({'x': range(1, 11), 'a': np.random.randn(10)})
data4 = pd.DataFrame({'x': range(1, 11), 'b': np.random.randn(10)})
data5 = pd.DataFrame({'x': range(1, 11), 'c': np.random.randn(10)})
data6 = pd.DataFrame({'x': range(1, 11), 'd': np.random.randn(10)})

# Fungsi untuk membuat grafik
def plot_graph(data, x_col, y_col, title, x_label, y_label, marker, color):
    fig, ax = plt.subplots()
    ax.plot(data[x_col], data[y_col], marker=marker, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    st.pyplot(fig)

# Membuat kolom dengan lebar berbeda
col1, col2, col3 = st.columns([1, 2, 1])

# Grafik pertama (kolom 1)
with col1:
    st.header("Grafik 1")
    plot_graph(data1, 'x', 'y', 'Random Data 1', 'X-axis', 'Y-axis', 'o', 'b')

# Grafik kedua (kolom 2)
with col2:
    st.header("Grafik 2")
    plot_graph(data2, 'x', 'z', 'Random Data 2', 'X-axis', 'Z-axis', 'x', 'r')

# Grafik ketiga (kolom 3)
with col3:
    st.header("Grafik 3")
    plot_graph(data3, 'x', 'a', 'Random Data 3', 'X-axis', 'A-axis', 's', 'g')

# Grafik keempat (kolom 1)
with col1:
    st.header("Grafik 4")
    plot_graph(data4, 'x', 'b', 'Random Data 4', 'X-axis', 'B-axis', 'd', 'm')

# Grafik kelima (kolom 2)
with col2:
    st.header("Grafik 5")
    plot_graph(data5, 'x', 'c', 'Random Data 5', 'X-axis', 'C-axis', '^', 'c')

# Grafik keenam (kolom 3)
with col3:
    st.header("Grafik 6")
    plot_graph(data6, 'x', 'd', 'Random Data 6', 'X-axis', 'D-axis', '*', 'y')
