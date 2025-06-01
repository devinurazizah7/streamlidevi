import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(
    page_title="Air Quality Monitor IoT",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk generate data simulasi (nanti diganti dengan data real dari IoT)
@st.cache_data
def generate_sample_data(n_samples=1000):
    """Generate sample air quality data untuk simulasi"""
    np.random.seed(42)
    
    # Generate timestamp
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_samples)
    
    # Generate sensor data dengan pola realistis
    temperature = np.random.normal(28, 5, n_samples)  # Celsius
    humidity = np.random.normal(70, 15, n_samples)    # %
    gas_level = np.random.exponential(200, n_samples)  # PPM
    co_level = np.random.exponential(50, n_samples)    # PPM
    pm25 = np.random.exponential(35, n_samples)        # μg/m³
    pm10 = np.random.exponential(50, n_samples)        # μg/m³
    
    # Calculate Air Quality Index (AQI)
    aqi = calculate_aqi(pm25, pm10, co_level, gas_level)
    
    # Create categories with imbalanced distribution (real-world scenario)
    quality_labels = []
    for a in aqi:
        if a <= 50:
            quality_labels.append('Good')
        elif a <= 100:
            quality_labels.append('Moderate')
        elif a <= 150:
            quality_labels.append('Unhealthy for Sensitive')
        elif a <= 200:
            quality_labels.append('Unhealthy')
        else:
            quality_labels.append('Very Unhealthy')
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'gas_level': gas_level,
        'co_level': co_level,
        'pm2.5': pm25,
        'pm10': pm10,
        'aqi': aqi,
        'quality': quality_labels
    })
    
    return df

def calculate_aqi(pm25, pm10, co, gas):
    """Calculate Air Quality Index"""
    # Simplified AQI calculation
    aqi_pm25 = (pm25 / 35.4) * 50
    aqi_pm10 = (pm10 / 54) * 50
    aqi_co = (co / 9.4) * 50
    aqi_gas = (gas / 200) * 50
    
    # Take maximum AQI
    aqi = np.maximum.reduce([aqi_pm25, aqi_pm10, aqi_co, aqi_gas])
    return np.clip(aqi, 0, 500)

def augment_data(df, augment_factor=2):
    """Data Augmentation untuk meningkatkan dataset"""
    augmented_data = []
    
    for _, row in df.iterrows():
        # Original data
        augmented_data.append(row.to_dict())
        
        # Augmented versions
        for i in range(augment_factor):
            aug_row = row.copy()
            
            # Add noise to sensor readings
            aug_row['temperature'] += np.random.normal(0, 0.5)
            aug_row['humidity'] += np.random.normal(0, 2)
            aug_row['gas_level'] += np.random.normal(0, 10)
            aug_row['co_level'] += np.random.normal(0, 5)
            aug_row['pm2.5'] += np.random.normal(0, 2)
            aug_row['pm10'] += np.random.normal(0, 3)
            
            # Recalculate AQI
            aug_row['aqi'] = calculate_aqi(
                np.array([aug_row['pm2.5']]),
                np.array([aug_row['pm10']]),
                np.array([aug_row['co_level']]),
                np.array([aug_row['gas_level']])
            )[0]
            
            # Update quality label
            aqi_val = aug_row['aqi']
            if aqi_val <= 50:
                aug_row['quality'] = 'Good'
            elif aqi_val <= 100:
                aug_row['quality'] = 'Moderate'
            elif aqi_val <= 150:
                aug_row['quality'] = 'Unhealthy for Sensitive'
            elif aqi_val <= 200:
                aug_row['quality'] = 'Unhealthy'
            else:
                aug_row['quality'] = 'Very Unhealthy'
            
            augmented_data.append(aug_row)
    
    return pd.DataFrame(augmented_data)

def train_weighted_model(df):
    """Train model dengan class weights untuk handle imbalanced data"""
    
    # Prepare features
    features = ['temperature', 'humidity', 'gas_level', 'co_level', 'pm2.5', 'pm10', 'aqi']
    X = df[features]
    y = df['quality']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Train model with class weights
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight_dict,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, class_weight_dict

# Main App
def main():
    st.title("🌬️ Air Quality Monitoring System with IoT")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Generate or load data
    if st.sidebar.button("🔄 Generate New Data"):
        st.cache_data.clear()
    
    data_size = st.sidebar.slider("Data Size", 100, 2000, 1000)
    augment_factor = st.sidebar.slider("Data Augmentation Factor", 0, 5, 2)
    
    # Load data
    with st.spinner("Loading air quality data..."):
        df = generate_sample_data(data_size)
    
    # Data Augmentation
    if augment_factor > 0:
        with st.spinner("Applying data augmentation..."):
            df_augmented = augment_data(df, augment_factor)
        st.sidebar.success(f"Data augmented: {len(df)} → {len(df_augmented)} samples")
        df_to_use = df_augmented
    else:
        df_to_use = df
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Current readings (latest data)
    latest = df_to_use.iloc[-1]
    
    with col1:
        st.metric(
            label="🌡️ Temperature",
            value=f"{latest['temperature']:.1f}°C",
            delta=f"{latest['temperature'] - df_to_use.iloc[-10]['temperature']:.1f}"
        )
    
    with col2:
        st.metric(
            label="💧 Humidity", 
            value=f"{latest['humidity']:.1f}%",
            delta=f"{latest['humidity'] - df_to_use.iloc[-10]['humidity']:.1f}"
        )
    
    with col3:
        st.metric(
            label="🏭 Gas Level",
            value=f"{latest['gas_level']:.0f} PPM"
        )
    
    with col4:
        aqi_color = "normal"
        if latest['aqi'] > 100:
            aqi_color = "inverse"
        st.metric(
            label="📊 Air Quality Index",
            value=f"{latest['aqi']:.0f}",
            delta=latest['quality']
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Monitoring", "🤖 AI Model Analysis", "📊 Data Analytics", "⚙️ IoT Settings"])
    
    with tab1:
        st.subheader("Real-time Air Quality Monitoring")
        
        # Time series charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_temp = px.line(df_to_use.tail(100), x='timestamp', y='temperature', 
                              title='Temperature Trend')
            st.plotly_chart(fig_temp, use_container_width=True)
            
            fig_gas = px.line(df_to_use.tail(100), x='timestamp', y='gas_level',
                             title='Gas Level Trend')
            st.plotly_chart(fig_gas, use_container_width=True)
        
        with col2:
            fig_humidity = px.line(df_to_use.tail(100), x='timestamp', y='humidity',
                                  title='Humidity Trend')
            st.plotly_chart(fig_humidity, use_container_width=True)
            
            fig_aqi = px.line(df_to_use.tail(100), x='timestamp', y='aqi',
                             title='Air Quality Index Trend')
            st.plotly_chart(fig_aqi, use_container_width=True)
        
        # Alert system
        st.subheader("🚨 Alert System")
        if latest['aqi'] > 150:
            st.error(f"⚠️ UNHEALTHY AIR QUALITY DETECTED! AQI: {latest['aqi']:.0f}")
        elif latest['aqi'] > 100:
            st.warning(f"⚠️ Moderate air quality. AQI: {latest['aqi']:.0f}")
        else:
            st.success(f"✅ Good air quality. AQI: {latest['aqi']:.0f}")
    
    with tab2:
        st.subheader("🤖 AI Model with Weighted Classes")
        
        # Train model
        with st.spinner("Training AI model with class weights..."):
            model, X_test, y_test, y_pred, class_weights = train_weighted_model(df_to_use)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Class Weights Applied:**")
            weight_df = pd.DataFrame(list(class_weights.items()), 
                                   columns=['Air Quality Class', 'Weight'])
            st.dataframe(weight_df)
            
            # Feature importance
            feature_names = ['temperature', 'humidity', 'gas_level', 'co_level', 'pm2.5', 'pm10', 'aqi']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                  title='Feature Importance')
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Confusion Matrix
            st.write("**Model Performance:**")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                              title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
        
        # Prediction for current conditions
        st.subheader("🔮 Current Prediction")
        current_features = np.array([[
            latest['temperature'], latest['humidity'], latest['gas_level'],
            latest['co_level'], latest['pm2.5'], latest['pm10'], latest['aqi']
        ]])
        
        prediction = model.predict(current_features)[0]
        probabilities = model.predict_proba(current_features)[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Predicted Air Quality:** {prediction}")
            st.write(f"**Actual Air Quality:** {latest['quality']}")
        
        with col2:
            prob_df = pd.DataFrame({
                'Class': model.classes_,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            fig_prob = px.bar(prob_df, x='Probability', y='Class',
                             title='Prediction Probabilities')
            st.plotly_chart(fig_prob, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Data Analytics & Insights")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_dist = px.histogram(df_to_use, x='quality', 
                                   title='Air Quality Distribution')
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Correlation heatmap
            numeric_cols = ['temperature', 'humidity', 'gas_level', 'co_level', 'pm2.5', 'pm10', 'aqi']
            corr_matrix = df_to_use[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Sensor Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # AQI vs other parameters
            fig_scatter = px.scatter(df_to_use, x='aqi', y='gas_level', 
                                   color='quality', title='AQI vs Gas Level')
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Box plot by quality
            fig_box = px.box(df_to_use, x='quality', y='aqi',
                           title='AQI Distribution by Quality Class')
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        st.dataframe(df_to_use.describe())
    
    with tab4:
        st.subheader("⚙️ IoT Device Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Sensor Calibration**")
            temp_offset = st.number_input("Temperature Offset (°C)", -5.0, 5.0, 0.0)
            humidity_offset = st.number_input("Humidity Offset (%)", -10.0, 10.0, 0.0)
            
            st.write("**Sampling Settings**")
            sampling_rate = st.selectbox("Sampling Rate", ["30 seconds", "1 minute", "5 minutes", "15 minutes"])
            data_retention = st.selectbox("Data Retention", ["7 days", "30 days", "90 days", "1 year"])
        
        with col2:
            st.write("**Alert Thresholds**")
            aqi_threshold = st.slider("AQI Alert Threshold", 50, 300, 150)
            temp_threshold = st.slider("Temperature Alert (°C)", 20, 50, 35)
            
            st.write("**Device Status**")
            st.success("🟢 ESP32 Connected")
            st.success("🟢 WiFi Signal: Strong")
            st.success("🟢 All Sensors Active")
            st.info("📡 Last Update: 30 seconds ago")
        
        if st.button("💾 Save IoT Configuration"):
            st.success("Configuration saved successfully!")
    
    # Real-time data simulation
    placeholder = st.empty()
    
    if st.sidebar.checkbox("🔄 Enable Real-time Updates"):
        for i in range(10):
            time.sleep(2)
            # Simulate new data point
            new_data = {
                'timestamp': datetime.now(),
                'temperature': np.random.normal(28, 2),
                'humidity': np.random.normal(70, 5),
                'gas_level': np.random.exponential(150),
                'co_level': np.random.exponential(40),
                'aqi': np.random.normal(80, 20)
            }
            
            with placeholder.container():
                st.write(f"🔄 **Live Update #{i+1}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{new_data['temperature']:.1f}°C")
                with col2:
                    st.metric("AQI", f"{new_data['aqi']:.0f}")
                with col3:
                    st.metric("Gas Level", f"{new_data['gas_level']:.0f} PPM")

if __name__ == "__main__":
    main()
