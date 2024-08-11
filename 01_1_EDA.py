import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load the dataset
file_path = 'data/electricity_consumption_dataset.csv'
data = pd.read_csv(file_path)

# Create directories for saving plots
os.makedirs('eda_results/saved_plots/', exist_ok=True)

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Display data types and non-null counts
print("\nDataset information:")
print(data.info())

# Generate summary statistics
print("\nDescriptive statistics:")
print(data.describe())

# Convert 'Time' column to datetime
data['Time'] = pd.to_datetime(data['Time'], format='%m/%d/%Y %H:%M')

# Select a household for detailed time series analysis
household = 'Sum [kWh].0'

# Seasonal decomposition of one household's time series
decomposition = seasonal_decompose(data[household], model='additive', period=96)
decomposition.plot()
plt.show()

# Calculate rolling mean and standard deviation
window_size = 96
data['Rolling_Mean'] = data[household].rolling(window=window_size).mean()
data['Rolling_Std'] = data[household].rolling(window=window_size).std()

plt.figure(figsize=(14, 8))
plt.plot(data['Time'], data[household], label='Original')
plt.plot(data['Time'], data['Rolling_Mean'], label='Rolling Mean', color='orange')
plt.plot(data['Time'], data['Rolling_Std'], label='Rolling Std Dev', color='green')
plt.title('Rolling Mean and Standard Deviation')
plt.xlabel('Time')
plt.ylabel('Electricity Consumption (kWh)')
plt.legend()
plt.savefig('eda_results/saved_plots/01_1_rolling_stats.png')
plt.close()


# Create boxplots for different time aggregations (hourly, daily, monthly, quarterly, yearly)
data['Hour'] = data['Time'].dt.hour
data['Day'] = data['Time'].dt.day
data['Month'] = data['Time'].dt.month
data['Quarter'] = data['Time'].dt.quarter
data['Year'] = data['Time'].dt.year

fig = make_subplots(rows=3, cols=2, subplot_titles=('Hourly', 'Daily', 'Monthly', 'Quarterly', 'Yearly'),
                    specs=[[{"type": "box"}, {"type": "box"}], [{"type": "box"}, {"type": "box"}], [{"type": "box"}, None]])

# Hourly boxplot
hourly_box = go.Box(y=data[household], x=data['Hour'], name='Hourly', boxmean='sd')
fig.add_trace(hourly_box, row=1, col=1)

# Daily boxplot
daily_box = go.Box(y=data[household], x=data['Day'], name='Daily', boxmean='sd')
fig.add_trace(daily_box, row=1, col=2)

# Monthly boxplot
monthly_box = go.Box(y=data[household], x=data['Month'], name='Monthly', boxmean='sd')
fig.add_trace(monthly_box, row=2, col=1)

# Quarterly boxplot
quarterly_box = go.Box(y=data[household], x=data['Quarter'], name='Quarterly', boxmean='sd')
fig.add_trace(quarterly_box, row=2, col=2)

# Yearly boxplot
yearly_box = go.Box(y=data[household], x=data['Year'], name='Yearly', boxmean='sd')
fig.add_trace(yearly_box, row=3, col=1)

# Update layout
fig.update_layout(height=900, width=1200, title_text="Boxplots of Household Consumption by Time Aggregations")

# Save the Plotly figure as an HTML file
fig.write_html('eda_results/saved_plots/01_1_boxplots_time_aggregations.html')
