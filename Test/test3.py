import streamlit as st
import pandas as pd
import numpy as np
import time

# Judul aplikasi
st.title('Grafik Real-Time')

# Membuat tempat kosong untuk grafik
placeholder = st.empty()

# Inisialisasi data
data = pd.DataFrame({
    # 'x': range(1, 11),
    'y': np.random.randn(10)
})

# Grafik Real-Time
while True:
    data['y'] = np.random.randn(10)  # Memperbarui data dengan nilai random baru

    # Menggambar grafik baru
    with placeholder.container():
        st.line_chart(data)

    time.sleep(1)  # Menunggu 1 detik sebelum memperbarui grafik
