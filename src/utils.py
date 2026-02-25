import pandas as pd
import os

def save_forecasts(test_df, predictions, model_name, output_path):
    forecast_df = test_df[['Time', 'Total_Consumption']].copy()
    forecast_df.rename(columns={'Total_Consumption': 'Actual_Consumption'}, inplace=True)
    forecast_df['Predicted_Consumption'] = predictions
    forecast_df['Model'] = model_name
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    return forecast_df

def print_summary(results_df, best_model_name, best_metrics, forecast_count, importance_df):
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print(f"  R² = {best_metrics['R2']:.4f}")
    print(f"  RMSE = {best_metrics['RMSE']:.4f} kWh")
    print(f"  MAE = {best_metrics['MAE']:.4f} kWh")
    print(f"\nGenerated:")
    print(f"  - 8 plots in outputs/")
    print(f"  - 4 models in models/")
    print(f"  - {forecast_count} forecasts in outputs/forecasts.csv")
    print("\n" + "=" * 60)
