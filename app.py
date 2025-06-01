import streamlit as st
import pandas as pd

st.title('Hello Streamlit!')
st.write('Aplikasi Streamlit pertama saya')

if st.button('Klik saya!'):
    st.write('Berhasil!')
    st.balloons()
