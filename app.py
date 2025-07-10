import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import random

# Konfigurasi Halaman
st.set_page_config(
    page_title="AQI Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Fungsi Data Rekayasa
def generate_data():
    dates = pd.date_range(end=datetime.today(), periods=30, freq='D')
    aqi = [random.randint(0, 300) for _ in range(30)]
    return pd.DataFrame({'Tanggal': dates, 'AQI': aqi})

# Fungsi Notifikasi Telegram
def send_telegram_alert(message):
    bot_token = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, json={'chat_id': chat_id, 'text': message})

# UI Dashboard
st.title("🌫️ AQI Monitoring")
df = generate_data()
latest_aqi = df['AQI'].iloc[-1]

# Visualisasi
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.line(df, x='Tanggal', y='AQI', title='Trend AQI 30 Hari'))

with col2:
    st.metric("AQI Terkini", latest_aqi)
    if latest_aqi > 150:  # Threshold
        send_telegram_alert(f"🚨 AQI Berbahaya: {latest_aqi}")

# Tampilkan Data
st.dataframe(df)
