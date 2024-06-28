import streamlit as st

if 'button1_pressed' not in st.session_state:
    st.session_state.button1_pressed = False
if 'button2_pressed' not in st.session_state:
    st.session_state.button2_pressed = False

if st.button('Hit me 1'):
    st.session_state.button1_pressed = True
    
if st.session_state.button1_pressed:
    st.write("halo 1")
    
if st.button('Hit me 2'):
    st.session_state.button2_pressed = True
    
if st.session_state.button2_pressed:
    st.write("halo 2")
