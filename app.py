import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error

# -----------------------------------------------------------------------------
# Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI Weather Station", layout="wide", page_icon="🌤️")

# Custom CSS for a premium glassmorphism look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #A0AEC0;
        margin-bottom: 30px;
    }
    hr {
        border-color: #2D3748;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model & Data Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

@st.cache_data
def load_and_evaluate_data():
    # Load raw data
    df = pd.read_csv('data/weather_data_ist.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    
    # Replicate chronological split from notebook (last 30% for testing)
    split_index = int(len(df) * 0.7)
    test_df = df.iloc[split_index:].copy()
    
    X_test = test_df[['light', 'humidity', 'hour_sin', 'hour_cos']]
    y_test = test_df['temperature']
    
    model = load_model()
    predictions = model.predict(X_test)
    test_df['predicted_temp'] = predictions
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    return test_df, r2, mae

try:
    model = load_model()
except FileNotFoundError:
    st.error("🚨 Model not found! Please run the Jupyter Notebook to train and save 'best_model.pkl'.")
    st.stop()

# -----------------------------------------------------------------------------
# App Layout
# -----------------------------------------------------------------------------
st.markdown('<p class="main-header">AI Weather Station</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Real-Time Temperature Prediction utilizing XGBoost and Cyclical Time Analytics.</p>', unsafe_allow_html=True)

# Create two beautiful tabs
tab1, tab2 = st.tabs(["🎛️ Interactive Predictor", "📊 Model Evaluation & Proof"])

# =============================================================================
# TAB 1: INTERACTIVE PREDICTOR
# =============================================================================
with tab1:
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown("### 🎛️ Sensor Control Panel")
        st.write("Adjust the environmental factors below to see how the AI predicts the temperature.")
        
        st.markdown("---")
        # Time of Day Slider
        hour_input = st.slider(
            "🕒 Time of Day", 
            min_value=0, 
            max_value=23, 
            value=14, 
            format="%d:00",
            help="24-Hour format (0 = Midnight, 12 = Noon)"
        )
        
        # Light Intensity Select Slider
        st.markdown("<br>", unsafe_allow_html=True)
        light_options = {
            "🌙 Pitch Black": 0.0,
            "☁️ Dim/Cloudy": 300.0,
            "⛅ Normal Daylight": 600.0,
            "☀️ Bright Sunlight": 1000.0
        }
        light_choice = st.select_slider(
            "💡 Light Intensity", 
            options=list(light_options.keys()), 
            value="⛅ Normal Daylight",
            help="Instead of a raw number, select the visual light condition."
        )
        light_val = light_options[light_choice]
        
        # Humidity Slider
        st.markdown("<br>", unsafe_allow_html=True)
        humidity_val = st.slider(
            "💧 Relative Humidity (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=50.0,
            help="0% is bone dry, 100% is raining/fog."
        )

    with col2:
        st.markdown("### 🌡️ Prediction Results")
        
        # Math & Prep
        hour_sin = np.sin(2 * np.pi * hour_input / 24.0)
        hour_cos = np.cos(2 * np.pi * hour_input / 24.0)
        
        input_data = pd.DataFrame({
            'light': [light_val],
            'humidity': [humidity_val],
            'hour_sin': [hour_sin],
            'hour_cos': [hour_cos]
        })
        
        # Prediction
        prediction = model.predict(input_data)[0]
        
        # Render Plotly Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            title = {'text': "Predicted Temperature (°C)", 'font': {'size': 24}},
            number = {'font': {'size': 50}, 'suffix': "°"},
            gauge = {
                'axis': {'range': [0, 50], 'tickwidth': 1},
                'bar': {'color': "rgba(255,255,255,0.7)"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 15], 'color': '#2C5282'},     # Cold Blue
                    {'range': [15, 25], 'color': '#2F855A'},    # Mild Green
                    {'range': [25, 35], 'color': '#C05621'},    # Warm Orange
                    {'range': [35, 50], 'color': '#9B2C2C'}     # Hot Red
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights Expander
        with st.expander("🧠 Behind the Scenes (Model Inputs)"):
            st.write("Here is the exact data array passed to the XGBoost Model. Notice how the Time of Day was converted into mathematical Sine and Cosine waves to represent cyclical time!")
            st.dataframe(input_data.style.format("{:.3f}"))

# =============================================================================
# TAB 2: MODEL EVALUATION
# =============================================================================
with tab2:
    st.markdown("### 📈 Model Evaluation (Testing on Unseen Data)")
    st.write("To prove this AI actually works, we feed it historical sensor data it has **never seen before** (the test set) and compare its predictions to the real temperature recorded at that time.")
    
    with st.spinner("Evaluating model..."):
        test_df, r2, mae = load_and_evaluate_data()
        
    met1, met2, met3 = st.columns(3)
    met1.metric(label="Accuracy (R² Score)", value=f"{r2 * 100:.2f}%", help="Percentage of variance explained by the model.")
    met2.metric(label="Average Error (MAE)", value=f"± {mae:.2f} °C", help="On average, the model's prediction is off by this many degrees.")
    met3.metric(label="Test Samples Evaluated", value=f"{len(test_df):,}", help="Number of real-world records evaluated.")
    
    st.markdown("---")
    st.markdown("#### 🎯 Actual vs. Predicted (Last 100 Records)")
    
    # Plotly Line Chart for the last 100 samples
    plot_df = test_df.tail(100).reset_index()
    
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['temperature'], mode='lines+markers', name='Actual Temp (°C)', line=dict(color='#4ECDC4', width=2)))
    line_fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['predicted_temp'], mode='lines+markers', name='AI Predicted Temp (°C)', line=dict(color='#FF6B6B', width=2, dash='dash')))
    
    line_fig.update_layout(
        xaxis_title="Time (Last 100 Test Samples)",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(line_fig, use_container_width=True)
    
    with st.expander("🔍 View Raw Comparison Data"):
        st.dataframe(test_df[['timestamp', 'temperature', 'predicted_temp', 'light', 'humidity']].tail(100))
