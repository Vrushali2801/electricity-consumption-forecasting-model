"""
SIMPLE ELECTRICITY FORECASTING PIPELINE
========================================
Beginner-friendly, no advanced programming, just the essentials.

Usage: python simple_forecast.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent display issues
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

print("="*60)
print("SIMPLE ELECTRICITY FORECASTING")
print("="*60)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/5] Loading data...")

data_file = 'data/electricity _consumption.csv'
df = pd.read_csv(data_file)

# Parse dates
df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=True)

# Calculate total consumption from all households
household_cols = [col for col in df.columns if 'Sum [kWh]' in col]
df['consumption'] = df[household_cols].sum(axis=1)

print(f"   ✓ Loaded {len(df):,} rows")
print(f"   ✓ Date range: {df['Time'].min()} to {df['Time'].max()}")

# ============================================================================
# STEP 2: CREATE SIMPLE FEATURES
# ============================================================================
print("\n[2/5] Creating features...")

# Time features (simple!)
df['hour'] = df['Time'].dt.hour
df['day_of_week'] = df['Time'].dt.dayofweek
df['month'] = df['Time'].dt.month

# Past consumption (lag features - most important!)
df['consumption_15min_ago'] = df['consumption'].shift(1)
df['consumption_1hour_ago'] = df['consumption'].shift(4)
df['consumption_1day_ago'] = df['consumption'].shift(96)

# Remove rows with missing values (from lag features)
df = df.dropna()

print(f"   ✓ Created 6 features")
print(f"   ✓ Rows after cleanup: {len(df):,}")

# ============================================================================
# STEP 3: PREPARE DATA FOR TRAINING
# ============================================================================
print("\n[3/5] Preparing data...")

# Features to use
feature_cols = ['hour', 'day_of_week', 'month', 
                'consumption_15min_ago', 'consumption_1hour_ago', 'consumption_1day_ago']

X = df[feature_cols].values
y = df['consumption'].values

# Split: 80% train, 20% test (simple split)
split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"   ✓ Training samples: {len(X_train):,}")
print(f"   ✓ Testing samples: {len(X_test):,}")

# ============================================================================
# STEP 4: TRAIN MODEL (Just one good model!)
# ============================================================================
print("\n[4/5] Training model...")

# Use Gradient Boosting (best performer)
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)
print("   ✓ Model trained!")

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("\n   MODEL PERFORMANCE:")
print(f"   • RMSE: {rmse:.4f} kWh")
print(f"   • MAE:  {mae:.4f} kWh")
print(f"   • R²:   {r2:.4f}")
print(f"   • MAPE: {mape:.2f}%")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================
print("\n[5/5] Saving results...")

# Create output folder
os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Save model
model_path = 'models/simple_model.pkl'
joblib.dump(model, model_path)
print(f"   ✓ Model saved: {model_path}")

# Save predictions
test_dates = df['Time'].iloc[split_point:].reset_index(drop=True)
results_df = pd.DataFrame({
    'Time': test_dates,
    'Actual': y_test,
    'Predicted': y_pred,
    'Error': y_test - y_pred
})
results_df.to_csv('outputs/simple_forecasts.csv', index=False)
print(f"   ✓ Forecasts saved: outputs/simple_forecasts.csv ({len(results_df):,} predictions)")

# Create simple plot
plt.figure(figsize=(12, 5))
plt.plot(results_df['Time'][:500], results_df['Actual'][:500], 
         label='Actual', alpha=0.7, linewidth=1)
plt.plot(results_df['Time'][:500], results_df['Predicted'][:500], 
         label='Predicted', alpha=0.7, linewidth=1)
plt.xlabel('Time')
plt.ylabel('Consumption (kWh)')
plt.title('Actual vs Predicted (First 500 samples)')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/simple_predictions.png', dpi=150)
plt.close()
print(f"   ✓ Plot saved: outputs/simple_predictions.png")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   TOP FEATURES:")
for idx, row in feature_importance.iterrows():
    print(f"   • {row['Feature']}: {row['Importance']:.3f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("✅ FORECASTING COMPLETE!")
print("="*60)
print(f"\n📊 Model Accuracy: R² = {r2:.4f} (explains {r2*100:.1f}% of variance)")
print(f"📈 Average Error: {mae:.4f} kWh")
print(f"\n📁 Check your results:")
print(f"   • outputs/simple_forecasts.csv - Predictions")
print(f"   • outputs/simple_predictions.png - Visualization")
print(f"   • models/simple_model.pkl - Saved model")
print("\n" + "="*60 + "\n")
