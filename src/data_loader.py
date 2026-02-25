import pandas as pd

def load_consumption_data(filepath):
    df = pd.read_csv(filepath)
    df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=True)
    household_cols = [col for col in df.columns if col.startswith('Sum [kWh]')]
    df['Total_Consumption'] = df[household_cols].sum(axis=1)
    return df, household_cols

def get_data_summary(df, household_cols):
    print(f"  Shape: {df.shape}")
    print(f"  Date Range: {df['Time'].min()} to {df['Time'].max()}")
    print(f"  Households: {len(household_cols)}")
