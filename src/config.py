# Data paths
DATA_FILE = 'data/electricity _consumption.csv'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature engineering
LAG_PERIODS = [1, 4, 8, 24, 96]
ROLLING_WINDOWS = [4, 24, 96]

# Model configurations
MODELS_CONFIG = {
    'ridge': {'alpha': 1.0},
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': RANDOM_STATE
    }
}


