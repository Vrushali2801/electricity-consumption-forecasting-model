import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_baseline_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_model(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, **kwargs):
    model = RandomForestRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train, **kwargs):
    model = GradientBoostingRegressor(**kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"  {model_name}: R²={r2:.4f}, RMSE={rmse:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}


def save_models(models_dict, scaler, feature_names, output_dir='models'):
    """
    Save trained models and artifacts
    
    Args:
        models_dict: Dictionary of {name: model} pairs
        scaler: Fitted scaler object
        feature_names: List of feature names
        output_dir: Directory to save models
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    for name, model in models_dict.items():
        filepath = os.path.join(output_dir, f'{name}_model.pkl')
        joblib.dump(model, filepath)
        print(f"✓ Saved: {name}_model.pkl")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    print("✓ Saved: scaler.pkl")
    
    # Save feature names
    with open(os.path.join(output_dir, 'feature_names.txt'), 'w') as f:
        f.write('\n'.join(feature_names))
    print("✓ Saved: feature_names.txt")


def get_feature_importance(model, feature_names, top_n=15):
    """
    Get feature importance from tree-based model
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    import pandas as pd
    
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df.head(top_n)
