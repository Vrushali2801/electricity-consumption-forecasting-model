# 01_EDA.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.fftpack import fft
import numpy as np

# Load the dataset
file_path = 'data/electricity_consumption_dataset.csv'
df = pd.read_csv(file_path)

# Convert 'Time' column to datetime
df['Time'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')

# Create directories for saving plots
os.makedirs('eda_results/saved_plots/', exist_ok=True)

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(df.head())

# Display data types and non-null counts
print("\nDataset information:")
print(df.info())

# Generate summary statistics
print("\nDescriptive statistics:")
print(df.describe())

# Analyzing the time frequency of the data
df['Time_Diff'] = df['Time'].diff().dropna()
print("\nTime frequency of data (seconds between measurements):")
print(df['Time_Diff'].value_counts().head())  # Most common time difference

# Analyzing the time period covered by the dataset
start_time = df['Time'].min()
end_time = df['Time'].max()
print(f"\nData covers the period from {start_time} to {end_time}")

# Number of households represented
num_households = df.filter(like='Sum [kWh]').shape[1]
print(f"\nNumber of households represented in the dataset: {num_households}")

# Select a household for detailed time series analysis
household = 'Sum [kWh].0'  # Example household

# Decompose the time series to analyze trend, seasonality, and residuals
decomposition = seasonal_decompose(df[household], model='additive', period=96)  # Assuming a daily cycle with 15-min intervals

# plt.figure(figsize=(14, 10))
# decomposition.plot()
# plt.suptitle('Seasonal Decomposition of Household Electricity Consumption')
# plt.savefig('eda_results/saved_plots/01_seasonal_decomposition.png')
# plt.close()

# Plot Autocorrelation and Partial Autocorrelation
plt.figure(figsize=(14, 8))
plot_acf(df[household].dropna(), lags=50)
plt.title('Autocorrelation of Household Electricity Consumption')
plt.savefig('eda_results/saved_plots/01_acf.png')
plt.close()

plt.figure(figsize=(14, 8))
plot_pacf(df[household].dropna(), lags=50)
plt.title('Partial Autocorrelation of Household Electricity Consumption')
plt.savefig('eda_results/saved_plots/01_pacf.png')
plt.close()

# Fourier Transform for Frequency Analysis
fft_vals = fft(df[household].dropna().values)
fft_freq = np.fft.fftfreq(len(fft_vals))

# plt.figure(figsize=(10, 6))
# plt.plot(fft_freq, np.abs(fft_vals))
# plt.title('Fourier Transform')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.savefig('eda_results/saved_plots/01_fourier_transform.png')
# plt.close()

# Rolling Mean and Standard Deviation
rolling_window = 96  # Assuming a daily cycle with 15-min intervals

df['Rolling_Mean'] = df[household].rolling(window=rolling_window).mean()
df['Rolling_Std'] = df[household].rolling(window=rolling_window).std()

plt.figure(figsize=(14, 8))
plt.plot(df['Time'], df[household], label='Original')
plt.plot(df['Time'], df['Rolling_Mean'], label='Rolling Mean', color='orange')
plt.plot(df['Time'], df['Rolling_Std'], label='Rolling Std Dev', color='green')
plt.title('Rolling Mean and Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Electricity Consumption (kWh)')
plt.legend()
plt.savefig('eda_results/saved_plots/01_rolling_stats.png')
plt.close()

# Differencing to achieve stationarity
df['Differenced'] = df[household].diff()

plt.figure(figsize=(14, 8))
plt.plot(df['Time'], df['Differenced'], label='Differenced Series', color='purple')
plt.title('Differenced Household Electricity Consumption')
plt.xlabel('Time')
plt.ylabel('Differenced Electricity Consumption (kWh)')
plt.legend()
plt.savefig('eda_results/saved_plots/01_differenced_series.png')
plt.close()

# Create an interactive plot for a specific household over a given time period using Plotly
start_date = '2022-01-01'
end_date = '2022-01-31'

# Filter data for the given time period
filtered_df = df[(df['Time'] >= start_date) & (df['Time'] <= end_date)]

# Create the Plotly figure
fig = px.line(filtered_df, x='Time', y=household, title=f'Electricity Consumption for {household} from {start_date} to {end_date}',
              labels={'Time': 'Time', household: 'Electricity Consumption (kWh)'})

# Save the Plotly figure as an HTML file
fig.write_html('eda_results/saved_plots/01_interactive_consumption_plot.html')

# Create boxplots for different time aggregations (hourly, daily, monthly, quarterly, yearly)
df['Hour'] = df['Time'].dt.hour
df['Day'] = df['Time'].dt.day
df['Month'] = df['Time'].dt.month
df['Quarter'] = df['Time'].dt.quarter
df['Year'] = df['Time'].dt.year

fig = make_subplots(rows=3, cols=2, subplot_titles=('Hourly', 'Daily', 'Monthly', 'Quarterly', 'Yearly'),
                    specs=[[{"type": "box"}, {"type": "box"}], [{"type": "box"}, {"type": "box"}], [{"type": "box"}, None]])

# Hourly boxplot
hourly_box = go.Box(y=df[household], x=df['Hour'], name='Hourly', boxmean='sd')
fig.add_trace(hourly_box, row=1, col=1)

# Daily boxplot
daily_box = go.Box(y=df[household], x=df['Day'], name='Daily', boxmean='sd')
fig.add_trace(daily_box, row=1, col=2)

# Monthly boxplot
monthly_box = go.Box(y=df[household], x=df['Month'], name='Monthly', boxmean='sd')
fig.add_trace(monthly_box, row=2, col=1)

# Quarterly boxplot
quarterly_box = go.Box(y=df[household], x=df['Quarter'], name='Quarterly', boxmean='sd')
fig.add_trace(quarterly_box, row=2, col=2)

# Yearly boxplot
yearly_box = go.Box(y=df[household], x=df['Year'], name='Yearly', boxmean='sd')
fig.add_trace(yearly_box, row=3, col=1)

# Update layout
fig.update_layout(height=900, width=1200, title_text="Boxplots of Household Consumption by Time Aggregations")

# Save the Plotly figure as an HTML file
fig.write_html('eda_results/saved_plots/01_boxplots_time_aggregations.html')

# Additional insights
print("\nPotential Additional Inputs for Improved Forecasting:")
print("- Time-based features: Hour, Day of the week, Month, and Season.")
print("- Lag features: Historical consumption values at previous time steps.")
print("- Weather data: Temperature, humidity, etc.")
print("- Holiday indicators: Binary indicator for holidays.")
