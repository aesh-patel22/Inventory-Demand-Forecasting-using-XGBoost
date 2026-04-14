import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Demand AI | Inventory Forecasting", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="📦"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Styling for the overall theme */
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 2rem;
    }
    
    /* Metrics block styling to look like premium cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f2631 0%, #151a22 100%);
        border: 1px solid #2b3543;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    hr {
        border-top: 1px solid #2b3543;
    }
    
    h1 {
        font-family: 'Inter', sans-serif;
        color: #e5e7eb;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #9ca3af;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dashboard_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        return None

df = load_data()

@st.cache_data
def load_metadata():
    try:
        stores_df = pd.read_csv('stores.csv')
    except FileNotFoundError:
        stores_df = None
        
    try:
        encoders = joblib.load('encoders.pkl')
    except FileNotFoundError:
        encoders = None
        
    return stores_df, encoders

stores_df, encoders = load_metadata()

def format_store(store_id):
    if stores_df is not None:
        store_info = stores_df[stores_df['store_nbr'] == store_id]
        if not store_info.empty:
            city = store_info.iloc[0]['city']
            state = store_info.iloc[0]['state']
            return f"Store {store_id} - {city} ({state})"
    return f"Store {store_id}"

def format_family(family_id):
    if encoders is not None and 'family' in encoders:
        return str(encoders['family'].inverse_transform([family_id])[0])
    return f"Category {family_id}"
st.title("📦 Intelligent Inventory Demand Forecasting")
st.markdown("Automate restocking and minimize stockouts using AI models trained on historical sales patterns.")

if df is None:
    st.warning("Dashboard data not found. Please run `python model_training.py` first to train the model and generate forecast data.")
    st.stop()


# --- SIDEBAR FILTERS ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=60)
st.sidebar.title("Filters")
st.sidebar.markdown("Filter demand by location and category.")

# Find original categoricals (using encoders.pkl if needed, but since we have IDs here, we will filter by them)
# For the sake of the dashboard, let's just create dropdowns based on unique values
selected_store = st.sidebar.selectbox("Select Store", sorted(df['store_nbr'].unique()), format_func=format_store)

filtered_df = df[df['store_nbr'] == selected_store]
selected_family = st.sidebar.selectbox("Select Product Family", sorted(filtered_df['family'].unique()), format_func=format_family)

final_df = filtered_df[filtered_df['family'] == selected_family].sort_values('date')

# Current Inventory Simulator (Since Kaggle data only has sales)
current_inventory = st.sidebar.number_input("Current Inventory Level (Simulated)", min_value=0, value=10)

st.sidebar.markdown("---")
st.sidebar.info("Model: **XGBoost Regressor**\n\nFeatures: *Time Lags, Rolling Means, Calendar Elements*")

# --- MAIN CONTENT DYNAMICS ---

if final_df.empty:
    st.error("No data available for the selected filters.")
    st.stop()

# KPIs
st.markdown("### 📊 Live Operations Overview")
col1, col2, col3, col4 = st.columns(4)

total_predicted = final_df['predicted_sales'].sum()
total_actual = final_df['sales'].sum()

# Stockout Risk Predictor (Formula: If remaining inventory < predicted sales over 15 days -> Risk!)
next_15_days_demand = total_predicted
risk_percentage = min((next_15_days_demand / max(current_inventory, 1)) * 100, 100)

if current_inventory < next_15_days_demand:
    restock_order = int(next_15_days_demand - current_inventory + (next_15_days_demand * 0.2)) # 20% safety stock
    risk_color = "🔴 High Risk"
else:
    restock_order = 0
    risk_color = "🟢 Low Risk"

col1.metric("Predicted 15-Day Demand", f"{int(total_predicted)} units")
col2.metric("Current Inventory", f"{current_inventory} units")
col3.metric("Stockout Risk", risk_color, f"{risk_percentage:.1f}% depletion")
col4.metric("Smart Restocking Advisor", f"Order {restock_order}", "Units to order", delta_color="inverse" if restock_order > 0 else "normal")


st.markdown("---")

# --- CHARTS ---
st.markdown("### 📈 Time Series Forecast Analysis")

fig = go.Figure()

# Actual Sales
fig.add_trace(go.Scatter(
    x=final_df['date'], 
    y=final_df['sales'],
    mode='lines+markers',
    name='Actual Sales',
    line=dict(color='#00d4ff', width=3),
    marker=dict(size=6)
))

# Predicted Sales
fig.add_trace(go.Scatter(
    x=final_df['date'], 
    y=final_df['predicted_sales'],
    mode='lines',
    name='AI Forecast',
    line=dict(color='#ff2a5f', width=3, dash='dash')
))

# Add a fill to represent uncertainty/confidence area visually though not statistical here
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='#9ca3af',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(showgrid=False, linecolor='#2b3543'),
    yaxis=dict(showgrid=True, gridcolor='#2b3543', linecolor='#2b3543', title="Units Sold")
)

st.plotly_chart(fig, use_container_width=True)

# --- DATA TABLE ---
with st.expander("Show detailed data view"):
    display_df = final_df[['date', 'sales', 'predicted_sales']].copy()
    display_df.rename(columns={'date': 'Date', 'sales': 'Actual Demand', 'predicted_sales': 'Predicted Demand'}, inplace=True)
    display_df['Difference'] = display_df['Actual Demand'] - display_df['Predicted Demand']
    st.dataframe(display_df.style.background_gradient(subset=['Difference'], cmap='coolwarm'), use_container_width=True)
