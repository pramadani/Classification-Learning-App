import streamlit as st

# CSS untuk membuat kontainer khusus
css = """
<style>
.custom-container {
    background-color: #f0f0f0;
    padding: 20px;
    border-radius: 20px;
}
</style>
"""

# Menampilkan CSS menggunakan st.markdown
st.markdown(css, unsafe_allow_html=True)

# Menampilkan konten di dalam kontainer khusus
st.markdown('<div class="custom-container">Isi kontainer Anda di sini</div>', unsafe_allow_html=True)
