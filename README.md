# Electricity Consumption Forecasting

Machine learning pipeline to forecast electricity consumption for multiple households with ~24-hour delay in smart meter data.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis (trains models, generates forecasts & visualizations)
python main.py
```

**That's it!** All outputs saved to `models/` and `outputs/`

## Results

| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|-----|------|
| **Gradient Boosting** ✅ | 0.218 | 0.136 | 0.707 | 27.3% |
| Random Forest | 0.219 | 0.137 | 0.703 | 27.5% |
| Linear Regression | 0.229 | 0.143 | 0.677 | 29.4% |

---

## Project Structure

```
electricity_consumption_forecasting/
│
├── main.py                              # Complete analysis pipeline
├── requirements.txt
│
├── src/                                 # Modular source code
│   ├── config.py                       # Configuration
│   ├── data_loader.py                  # Data loading
│   ├── feature_engineering.py          # Feature creation
│   ├── models.py                       # Model training/evaluation
│   ├── visualization.py                # Plotting functions
│   └── utils.py                        # Utilities
│
├── data/                               # Dataset
├── models/                             # Trained models (auto-generated)
└── outputs/                            # Visualizations + forecasts (auto-generated)
```

- **87,552 observations** across 10 households, 2 years (2022-2023)
- **15-minute intervals**, no missing values
- **Clear patterns:** Daily peaks at 6-9 PM, higher weekday consumption

## Features (30 total)

**Time-based:** Hour, day of week, month + cyclical encodings (sin/cos)  
**Lag features:** Consumption at 15min, 1hr, 2hr, 6hr, 24hr ago  
**Rolling stats:** Mean, std, min, max over 1hr, 6hr, 24hr windows  

**Top 3 Most Important:**
1. `Total_Consumption_lag_1` (80%) - Previous 15-min consumption
2. `Hour` (1.4%) - Time of day
3. `Total_Consumption_lag_96` (1.2%) - 24 hours ago

## Approach

1. **Data Exploration:** Analyzed patterns, verified data quality
2. **Feature Engineering:** Created 30 time-based, lag, and rolling features
3. **Model Training:** Trained 4 models with time-series split (80/20)
4. **Evaluation:** Gradient Boosting selected as best performing
5. **Forecasting:** Generate predictions for any future time period

## Usage

```bash
python main.py
```

**What it does:**
1. Loads data from `data/electricity _consumption.csv`
2. Engineers 30 features (time-based, lags, rolling stats)
3. Trains 4 models (Linear, Ridge, Random Forest, Gradient Boosting)
4. Evaluates performance on test set (time-series split)
5. Generates 8 visualization plots → `outputs/`
6. Saves trained models → `models/`
7. Creates forecasts CSV → `outputs/forecasts.csv`

**Outputs:**
- `outputs/forecasts.csv` - 17,492 predictions with actuals
- `outputs/*.png` - 8 visualization plots
- `models/*.pkl` - Trained model files

---

## Potential Improvements

**Data Sources (+5-10% accuracy each):**
- Weather data (temperature, humidity)
- Calendar info (holidays, events)
- Household metadata (size, occupants)

**Model Enhancements:**
- Hyperparameter tuning (GridSearchCV, Optuna)
- LSTM/GRU for sequence modeling
- Ensemble methods (stacking, blending)

---

## Key Findings

1. **Lag features dominate** - Recent consumption is strongest predictor (80% importance)
2. **Strong temporal patterns** - Clear daily and weekly cycles
3. **Tree models win** - Gradient Boosting outperforms linear baseline (RMSE: 0.218 vs 0.229)

