import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
import random

# Konfigurasi halaman
st.set_page_config(
    page_title="AQI Dashboard",
    page_icon="🌍",
    layout="wide"
)

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = "7754603821:AAEArAmBjCm8yI5vdsVkroY1g-DqOE5Bcjo"
TELEGRAM_CHAT_ID = " 1458169344"  # Ganti dengan chat ID Anda

def send_telegram_message(message):
    """Kirim pesan ke Telegram bot"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML"
        }
        response = requests.post(url, data=data)
        return response.json()
    except Exception as e:
        st.error(f"Error mengirim pesan Telegram: {e}")
        return None

def generate_aqi_data():
    """Generate data AQI rekayasa"""
    # Data untuk trend (3 data point seperti di gambar)
    base_time = datetime.now()
    times = [
        (base_time - timedelta(minutes=10)).strftime("%H:%M"),
        (base_time - timedelta(minutes=5)).strftime("%H:%M"),
        base_time.strftime("%H:%M")
    ]
    
    # Data AQI trend
    aqi_values = [66, 69, 75]  # Sesuai dengan gambar
    
    # Current readings
    current_data = {
        "PM2.5": round(random.uniform(30, 40), 1),
        "PM10": round(random.uniform(40, 50), 1),
        "Temperature": round(random.uniform(27, 30), 1),
        "Humidity": random.randint(60, 70),
        "CO2": random.randint(800, 900),
        "AQI": aqi_values[-1]
    }
    
    return times, aqi_values, current_data

def get_aqi_status(aqi_value):
    """Dapatkan status AQI berdasarkan nilai"""
    if aqi_value <= 50:
        return "Baik", "green"
    elif aqi_value <= 100:
        return "Sedang", "yellow"
    elif aqi_value <= 150:
        return "Tidak Sehat untuk Kelompok Sensitif", "orange"
    elif aqi_value <= 200:
        return "Tidak Sehat", "red"
    elif aqi_value <= 300:
        return "Sangat Tidak Sehat", "purple"
    else:
        return "Berbahaya", "maroon"

def create_aqi_trend_chart(times, aqi_values):
    """Buat grafik trend AQI"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=times,
        y=aqi_values,
        mode='lines+markers',
        name='AQI',
        line=dict(color='#FF8C00', width=2),
        marker=dict(size=8, color='#FF8C00')
    ))
    
    fig.update_layout(
        title="AQI Trend",
        title_font_size=20,
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(range=[0, 100]),
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor='center'),
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Tambahkan grid
    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')
    
    return fig

def main():
    st.title("🌍 AQI Dashboard")
    st.markdown("---")
    
    # Generate data
    times, aqi_values, current_data = generate_aqi_data()
    
    # Layout dengan 2 kolom
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # AQI Trend Chart
        fig = create_aqi_trend_chart(times, aqi_values)
        st.plotly_chart(fig, use_container_width=True)
        
        # Tampilkan waktu dan nilai AQI saat ini
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
            <h3>📊 Data Saat Ini</h3>
            <p><strong>⏰ Waktu:</strong> {times[-1]}</p>
            <p><strong>🌬️ AQI:</strong> {current_data['AQI']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Current Readings
        st.markdown("### 📈 Current Readings")
        
        # PM2.5 dan PM10
        col_pm1, col_pm2 = st.columns(2)
        with col_pm1:
            st.metric("PM2.5", f"{current_data['PM2.5']} μg/m³")
        with col_pm2:
            st.metric("PM10", f"{current_data['PM10']} μg/m³")
        
        # Temperature dan Humidity
        col_temp, col_hum = st.columns(2)
        with col_temp:
            st.metric("Temperature", f"{current_data['Temperature']}°C")
        with col_hum:
            st.metric("Humidity", f"{current_data['Humidity']}%")
        
        # CO2 dan AQI
        col_co2, col_aqi = st.columns(2)
        with col_co2:
            st.metric("CO2", f"{current_data['CO2']} ppm")
        with col_aqi:
            aqi_status, aqi_color = get_aqi_status(current_data['AQI'])
            st.metric("AQI", current_data['AQI'])
            st.markdown(f"<p style='color: {aqi_color}; font-weight: bold;'>{aqi_status}</p>", 
                       unsafe_allow_html=True)
    
    # Tombol untuk kirim notifikasi Telegram
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("📱 Kirim Notifikasi Telegram", type="primary"):
            # Buat pesan notifikasi
            message = f"""
🌍 <b>AQI Dashboard Report</b>
📅 <b>Waktu:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 <b>Current Readings:</b>
• AQI: {current_data['AQI']} ({aqi_status})
• PM2.5: {current_data['PM2.5']} μg/m³
• PM10: {current_data['PM10']} μg/m³
• Temperature: {current_data['Temperature']}°C
• Humidity: {current_data['Humidity']}%
• CO2: {current_data['CO2']} ppm

🌬️ <b>Status Udara:</b> {aqi_status}
            """
            
            # Kirim pesan
            with st.spinner("Mengirim notifikasi..."):
                result = send_telegram_message(message)
                if result:
                    st.success("✅ Notifikasi berhasil dikirim ke Telegram!")
                else:
                    st.error("❌ Gagal mengirim notifikasi")
    
    # Auto refresh setiap 30 detik
    st.markdown("---")
    st.markdown("🔄 **Dashboard akan refresh otomatis setiap 30 detik**")
    
    # Auto refresh
    time.sleep(30)
    st.rerun()

if __name__ == "__main__":
    main()
