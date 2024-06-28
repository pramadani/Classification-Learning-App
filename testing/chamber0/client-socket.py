import streamlit as st
import socketio
import time
from PIL import Image

# Inisialisasi Socket.IO Client
sio = socketio.Client()

# Fungsi untuk halaman Real-time Data
def page_realtime_data():
    st.title('Real-time Data Stream with Socket.IO')
    placeholder = st.empty()
    value = 0

    @sio.event
    def connect():
        print("Connected to the server")

    @sio.event
    def disconnect():
        print("Disconnected from server")

    @sio.on('data_response')
    def handle_data_response(data):
        nonlocal value
        value = data['value']
        print(value)

    sio.connect('http://127.0.0.1:5000')

    while True:
        with placeholder.container():
            st.write(f"Nilai acak dari Socket: {value}")
        time.sleep(1)

# Fungsi untuk halaman Halaman Lain
def page_other_content():
    st.title('Halaman Lain')
    st.write("Ini adalah halaman lain dengan konten yang berbeda.")
    st.write("Anda dapat menambahkan konten apa pun di sini.")

# Mengatur navigasi antara halaman
def main():
    st.set_page_config(page_title='Real-time Data Stream', page_icon=':chart_with_upwards_trend:')

    pages = {
        "Real-time Data": page_realtime_data,
        "Halaman Lain": page_other_content,
    }

    st.sidebar.title('Navigasi')
    import streamlit as st
import socketio
import time

# Inisialisasi Socket.IO Client
sio = socketio.Client()

# Fungsi untuk halaman Real-time Data
def page_realtime_data():
    st.title('Real-time Data Stream with Socket.IO')
    placeholder = st.empty()
    value = 0

    @sio.event
    def connect():
        print("Connected to the server")

    @sio.event
    def disconnect():
        print("Disconnected from server")

    @sio.on('data_response')
    def handle_data_response(data):
        nonlocal value
        value = data['value']
        print(value)

    sio.connect('http://127.0.0.1:5000')

    while True:
        with placeholder.container():
            st.write(f"Nilai acak dari Socket: {value}")
        time.sleep(1)

# Fungsi untuk halaman Halaman Lain
def page_other_content():
    st.title('Halaman Lain')
    st.write("Ini adalah halaman lain dengan konten yang berbeda.")
    st.write("Anda dapat menambahkan konten apa pun di sini.")

# Mengatur navigasi antara halaman
def main():
    st.set_page_config(page_title='Real-time Data Stream', page_icon=':chart_with_upwards_trend:')
       
    logo = Image.open('logo.png')
    st.sidebar.image(logo, width=200)
        
    st.sidebar.title('Navigasi')
    
    pages = {
        "Real-time Data": page_realtime_data,
        "Halaman Lain": page_other_content,
    }
            
    
    selection = st.sidebar.selectbox("Pilih Halaman", list(pages.keys()))

    current_page = pages[selection]
    current_page()

if __name__ == "__main__":
    main()
