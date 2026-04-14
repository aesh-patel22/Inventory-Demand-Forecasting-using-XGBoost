import pandas as pd
import numpy as np

def load_and_merge_data(train_path='train.csv', stores_path='stores.csv', oil_path='oil.csv', start_date='2017-01-01'):
    """
    Loads train and stores data, merges them, and cleans up dates.
    For time-series, random sampling breaks sequence, so we filter from a start date.
    We use 2017 data by default to keep dataset size manageable for local computing.
    """
    print("Loading data...")
    # Load dataset
    df_train = pd.read_csv(train_path)
    
    # Filtering by date directly after loading
    df_train['date'] = pd.to_datetime(df_train['date'])
    if start_date:
        df_train = df_train[df_train['date'] >= pd.to_datetime(start_date)]
        
    df_stores = pd.read_csv(stores_path)
    
    print("Loading external exogenous data...")
    # Load Oil Data
    df_oil = pd.read_csv(oil_path)
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    
    print("Merging datasets...")
    # Merge train with store info
    df = df_train.merge(df_stores, on='store_nbr', how='left')
    
    # Merge train with oil info
    df = df.merge(df_oil, on='date', how='left')
    # Oil data is missing on weekends, so we forward-fill the prices from Friday.
    df['dcoilwtico'] = df['dcoilwtico'].ffill().bfill()
    
    # Sort values by date to keep temporal order for lag features
    df = df.sort_values(by=['store_nbr', 'family', 'date'])
    df = df.reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    # Test load
    df = load_and_merge_data()
    print("Data loaded successfully! Shape:", df.shape)
    print("Min date:", df['date'].min(), "Max date:", df['date'].max())
