import pandas as pd
import numpy as np
from data_preparation import load_and_merge_data
from feature_engineering import create_features
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def train_and_evaluate():
    print("--- 1. Pulling and Preparing Data ---")
    # To keep model training snappy locally, use 2017 data only
    df = load_and_merge_data(start_date='2017-01-01')
    
    print("--- 2. Feature Engineering ---")
    df_features, encoders = create_features(df)
    
    # Save encoders for Streamlit
    joblib.dump(encoders, 'encoders.pkl')
    
    # Define features and target
    # Exclude id, date, sales (target)
    features = [
        'store_nbr', 'family', 'onpromotion', 'cluster', 
        'day_of_week', 'month', 'year', 'is_weekend',
        'sales_lag_1', 'sales_lag_7', 'rolling_mean_7', 'rolling_mean_30',
        'city', 'state', 'type', 'dcoilwtico', 'is_holiday', 'is_national_holiday'
    ]
    target = 'sales'
    
    print("--- 3. Splitting into Train and Test (Time-Based) ---")
    # Let's say last 15 days is testing
    split_date = df_features['date'].max() - pd.Timedelta(days=15)
    
    train_df = df_features[df_features['date'] <= split_date]
    test_df = df_features[df_features['date'] > split_date]
    
    X_train = train_df[features]
    y_train = train_df[target]
    
    X_test = test_df[features]
    y_test = test_df[target]
    
    print("--- 4. Training XGBoost Model ---")
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("--- 5. Evaluating Model ---")
    predictions = model.predict(X_test)
    # Never predict negative sales
    predictions = np.clip(predictions, 0, None)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # Save model
    print("Saving model to xgboost_model.pkl...")
    joblib.dump(model, 'xgboost_model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    # Save a small sample of test data with predictions for the Dashboard
    test_df_dashboard = test_df.copy()
    test_df_dashboard['predicted_sales'] = predictions
    test_df_dashboard.to_csv('dashboard_data.csv', index=False)
    print("Saved sample dashboard data!")

if __name__ == "__main__":
    train_and_evaluate()
