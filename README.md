# Inventory-Demand-Forecasting-using-XGBoost
# 📦 Intelligent Inventory Demand Forecasting

An end-to-end Machine Learning project designed to automate inventory restocking and minimize stockouts using AI models trained on historical sales patterns. The project utilizes time-series feature engineering and an XGBoost Regressor to predict future product demand across various store locations.

## 🚀 Features

- **Store & Category Level Forecasting:** Filter and forecast demand specifically for individual stores and product families.
- **Stockout Risk Predictor:** Calculates risk percentage based on current inventory simulation and projected 15-day demand.
- **Smart Restocking Advisor:** Automatically recommends order quantities with a built-in safety stock buffer to prevent depletion.
- **Interactive Time-Series Visualization:** Understand trends quickly by comparing AI Forecasts against actual sales data.
- **Dynamic Dark-Themed UI:** Professional, interactive web dashboard built with Streamlit and Plotly.

## 📁 Project Structure

- `data_preparation.py`: Handles loading, cleaning, and merging historical data (sales, stores, oil prices, transactions).
- `feature_engineering.py`: Extracts critical time-series features like time lags, rolling means, and calendar elements.
- `model_training.py`: Trains the XGBoost Regressor and exports the model (`xgboost_model.pkl`) and necessary encoders.
- `app.py`: The main Streamlit web application providing the interactive dashboard.
- `requirements.txt`: Lists all Python module dependencies required to run the project.

## ⚙️ Installation & Setup

1. **Navigate to the project directory:**
   ```bash
   cd "Inventory Forecasting"
   ```

2. **Install dependencies:**
   Make sure you have Python installed. It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model (First Time Setup):**
   Before running the dashboard, you need to generate the `dashboard_data.csv` and model files. Run the following script:
   ```bash
   python model_training.py
   ```

4. **Launch the Dashboard:**
   Start the interactive Streamlit UI:
   ```bash
   streamlit run app.py
   ```

## 🛠️ Technology Stack
- **Python**
- **Machine Learning:** XGBoost, scikit-learn
- **Data Manipulation:** pandas, NumPy
- **Web App / UI:** Streamlit
- **Data Visualization:** Plotly

## 📊 Dataset Reference
This project leverages historical sales data, including store locations, daily transactions, oil prices, and holiday events, designed for classical time-series retail forecasting based on Kaggle datasets.

---
*Created to demonstrate AI-driven business intelligence and supply chain solutions.*
