import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from sklearn.preprocessing import StandardScaler
from src.config import *
from src.data_loader import load_consumption_data, get_data_summary
from src.feature_engineering import engineer_all_features, prepare_model_data
from src.models import *
from src.visualization import generate_all_plots
from src.utils import save_forecasts, print_summary

print("\n[1/6] Loading data...")
df, household_cols = load_consumption_data(DATA_FILE)
get_data_summary(df, household_cols)

print("\n[2/6] Creating features...")
df = engineer_all_features(df, 'Total_Consumption', LAG_PERIODS, ROLLING_WINDOWS)
print(f"  Created {df.shape[1]} features")

print("\n[3/6] Preparing data...")
X, y, feature_cols, df_model = prepare_model_data(df, 'Total_Consumption', household_cols)

# Split data
split_idx = int(len(df_model) * (1 - TEST_SIZE))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[4/6] Training models...")

# Train all 4 models
baseline = train_baseline_model(X_train_scaled, y_train)
baseline_pred = baseline.predict(X_test_scaled)
baseline_metrics = evaluate_model(y_test, baseline_pred, "Linear Regression")

ridge = train_ridge_model(X_train_scaled, y_train, **MODELS_CONFIG['ridge'])
ridge_pred = ridge.predict(X_test_scaled)
ridge_metrics = evaluate_model(y_test, ridge_pred, "Ridge Regression")

rf = train_random_forest(X_train, y_train, **MODELS_CONFIG['random_forest'])
rf_pred = rf.predict(X_test)
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")

gb = train_gradient_boosting(X_train, y_train, **MODELS_CONFIG['gradient_boosting'])
gb_pred = gb.predict(X_test)
gb_metrics = evaluate_model(y_test, gb_pred, "Gradient Boosting")

# Compare models
print("\n[5/6] Comparing models...")
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
    'MAE': [baseline_metrics['MAE'], ridge_metrics['MAE'], rf_metrics['MAE'], gb_metrics['MAE']],
    'RMSE': [baseline_metrics['RMSE'], ridge_metrics['RMSE'], rf_metrics['RMSE'], gb_metrics['RMSE']],
    'R²': [baseline_metrics['R2'], ridge_metrics['R2'], rf_metrics['R2'], gb_metrics['R2']]
})

print("\n", results_df.to_string(index=False))
best_idx = results_df['RMSE'].idxmin()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\n  Best Model: {best_model_name}")

# Feature importance
importance_df = get_feature_importance(rf, feature_cols)
print("\n  Top 5 Features:")
for idx, row in importance_df.head(5).iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.3f}")

# Generate visualizations
print("\n[6/6] Saving results...")
test_df = df_model.iloc[split_idx:].copy()
predictions_dict = {
    'Gradient Boosting': gb_pred,
    'Random Forest': rf_pred,
    'Linear Baseline': baseline_pred
}

generate_all_plots(df, df_model, household_cols, importance_df,
                   test_df, predictions_dict, y_test, gb_pred, OUTPUT_DIR)

# Save models and forecasts
models_dict = {'gradient_boosting': gb, 'random_forest': rf, 'baseline': baseline}
save_models(models_dict, scaler, feature_cols, MODEL_DIR)
forecast_df = save_forecasts(test_df, gb_pred, 'Gradient Boosting', f'{OUTPUT_DIR}/forecasts.csv')

# Summary
print_summary(results_df, best_model_name, gb_metrics, len(forecast_df), importance_df)
