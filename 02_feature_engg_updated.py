import pandas as pd
import numpy as np
import os

# Load the dataset
file_path = 'data/electricity_consumption_dataset.csv'
df = pd.read_csv(file_path)

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')

# Create output directory for processed data
os.makedirs('processed_data/', exist_ok=True)

# Initialize a list to collect all the engineered features DataFrames
all_features = [df[['Time']]]  # Start with the Time column

# Iterate through each household column and generate features
household_cols = [col for col in df.columns if col.startswith('Sum [kWh]') and not any(sub in col for sub in ['_hour', '_day', '_month', '_quarter', '_lag', '_rolling'])]

print(f"Processing {len(household_cols)} households for feature engineering...")

for household in household_cols:
    print(f"Generating features for {household}...")
    features = pd.DataFrame()  # Temporary DataFrame to store features for the current household

    # Time-based features
    features[f'{household}_hour'] = df['Time'].dt.hour
    features[f'{household}_minute'] = df['Time'].dt.minute
    features[f'{household}_day_of_week'] = df['Time'].dt.dayofweek
    features[f'{household}_day_of_month'] = df['Time'].dt.day
    features[f'{household}_month'] = df['Time'].dt.month
    features[f'{household}_quarter'] = df['Time'].dt.quarter
    features[f'{household}_half_hour_of_day'] = df['Time'].dt.hour * 2 + df['Time'].dt.minute // 30

    # Lag features (15-minute intervals)
    lag_features = [1, 4, 12, 96, 192, 288, 384]  # Example lags: 15-min, 1-hour, 3-hours, 1-day, 2-days, etc.
    for lag in lag_features:
        features[f'{household}_lag_{lag}'] = df[household].shift(lag)

    # Rolling statistics (15-minute intervals)
    rolling_windows = [4, 12, 96, 192, 384]  # 1-hour, 3-hours, 1-day, 2-days, etc.
    for window in rolling_windows:
        features[f'{household}_rolling_mean_{window}'] = df[household].rolling(window=window).mean()
        features[f'{household}_rolling_std_{window}'] = df[household].rolling(window=window).std()

    # Weekly and monthly rolling statistics (96 * 7 = 672 for a week, 96 * 30 = 2880 for a month)
    weekly_window = 672  # 1 week
    monthly_window = 2880  # 1 month

    features[f'{household}_rolling_mean_weekly'] = df[household].rolling(window=weekly_window).mean()
    features[f'{household}_rolling_std_weekly'] = df[household].rolling(window=weekly_window).std()
    features[f'{household}_rolling_mean_monthly'] = df[household].rolling(window=monthly_window).mean()
    features[f'{household}_rolling_std_monthly'] = df[household].rolling(window=monthly_window).std()

    # Add the original target column
    features[household] = df[household]

    # Add the features for this household to the all_features list
    all_features.append(features)

# Concatenate all the features into a single DataFrame at once
print("Concatenating all features into a single DataFrame...")
final_features = pd.concat(all_features, axis=1)

# Drop rows with NaN values (due to shifting and rolling)
final_features = final_features.dropna()

# Additional check for NaN values
if final_features.isnull().any().sum() == 0:
    print("No NaN values after dropna. Data is consistent.")
else:
    nan_count = final_features.isnull().sum().sum()
    print(f"Warning: {nan_count} NaN values detected even after dropna. Recheck sequence consistency.")

# Save the processed dataset
processed_file_path = 'processed_data/engineered_features_all_households.csv'
final_features.to_csv(processed_file_path, index=False)

print(f"Feature engineering completed and saved to '{processed_file_path}'")
