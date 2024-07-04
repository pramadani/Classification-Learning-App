import streamlit as st
from tensorflow.keras.models import load_model

# temp = False

loaded_model = load_model('./resources/model.h5')

# st.switch_page("dashboard.py")