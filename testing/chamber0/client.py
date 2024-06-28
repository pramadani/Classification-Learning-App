import streamlit as st
import requests
import time

# Fungsi untuk mengambil data dari API Flask
def fetch_data():
    response = requests.get('http://127.0.0.1:5000/api/random')
    if response.status_code == 200:
        data = response.json()
        return data['value']
    else:
        return None

st.title("Realtime Data Dashboard from API")

# Membuat elemen placeholder untuk data
placeholder = st.empty()

# Loop untuk memperbarui data secara periodik
while True:
    # Mengambil data dari API
    value = fetch_data()

    # Memperbarui elemen placeholder dengan data baru
    with placeholder.container():
        st.write(f"Nilai acak dari API: {value}")
    
    # Menunggu sebelum mengambil data baru
    time.sleep(1)  # Update setiap 1 detik
