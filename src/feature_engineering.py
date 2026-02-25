import numpy as np
import pandas as pd

def create_time_features(df):
    df = df.copy()
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['Month'] = df['Time'].dt.month
    df['DayOfMonth'] = df['Time'].dt.day
    df['Quarter'] = df['Time'].dt.quarter
    df['WeekOfYear'] = df['Time'].dt.isocalendar().week
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Date'] = df['Time'].dt.date
    return df

def create_lag_features(df, target_col, lags):
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rolling_features(df, target_col, windows):
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
    return df

def engineer_all_features(df, target_col='Total_Consumption', lags=None, windows=None):
    if lags is None:
        lags = [1, 4, 8, 24, 96]
    if windows is None:
        windows = [4, 24, 96]
    df = create_time_features(df)
    df = create_lag_features(df, target_col, lags)
    df = create_rolling_features(df, target_col, windows)
    return df

def prepare_model_data(df, target='Total_Consumption', household_cols=None):
    df_model = df.dropna().copy()
    
    # Define columns to exclude
    exclude_cols = ['Time', 'Electricity.Timestep', 'Date', target]
    if household_cols:
        exclude_cols.extend(household_cols)
    
    # Get feature columns
    feature_cols = [col for col in df_model.columns if col not in exclude_cols]
    
    X = df_model[feature_cols]
    y = df_model[target]
    
    return X, y, feature_cols, df_model
