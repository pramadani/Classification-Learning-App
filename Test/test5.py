import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

# Judul aplikasi
st.title('Visualisasi Meteran Batang Real-Time')

# Fungsi untuk membuat grafik meteran batang
def create_bar_meter(value, title, max_value, units):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+gauge+delta",
        value=value,
        title={'text': title},
        domain={'x': [0.1, 0.9], 'y': [0.2, 0.8]},
        gauge={
            'shape': "bullet",
            'axis': {'range': [None, max_value]},
            'bar': {'color': "purple"},
            'steps': [
                {'range': [0, max_value/3], 'color': "lightgray"},
                {'range': [max_value/3, 2*max_value/3], 'color': "gray"},
                {'range': [2*max_value/3, max_value], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        number={'suffix': units}
    ))

    fig.update_layout(width=400, height=100)
    st.plotly_chart(fig)

# Inisialisasi data awal
temperature = 20
humidity = 50
light = 70
co2 = 500

# Loop untuk update real-time
while True:
    # Update data dengan nilai acak
    temperature += np.random.randint(-1, 2)
    humidity += np.random.randint(-2, 3)
    light += np.random.randint(-5, 6)
    co2 += np.random.randint(-10, 11)

    # Menggambar meteran batang untuk masing-masing parameter
    st.header('Temperatur')
    create_bar_meter(temperature, 'Temperature', 30, '°C')

    st.header('Kelembaban')
    create_bar_meter(humidity, 'Humidity', 100, '%')

    st.header('Cahaya')
    create_bar_meter(light, 'Light Intensity', 100, '')

    st.header('CO2')
    create_bar_meter(co2, 'CO2 Level', 1000, 'ppm')

    time.sleep(2)  # Menunggu 2 detik sebelum memperbarui data
