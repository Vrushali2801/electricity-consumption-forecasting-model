"""
Visualization Functions
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


def setup_plot_style():
    """Setup matplotlib style"""
    plt.style.use('default')
    sns.set_palette("husl")


def plot_total_consumption(df, output_path):
    """Plot total consumption over time"""
    plt.figure(figsize=(15, 5))
    plt.plot(df['Time'], df['Total_Consumption'], linewidth=0.5, alpha=0.7)
    plt.title('Total Electricity Consumption Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Total Consumption (kWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_consumption_distribution(df, output_path):
    """Plot distribution of consumption"""
    plt.figure(figsize=(12, 6))
    df['Total_Consumption'].hist(bins=100, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Total Electricity Consumption', fontsize=14, fontweight='bold')
    plt.xlabel('Consumption (kWh)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_hourly_pattern(df, output_path):
    """Plot average consumption by hour"""
    hourly_avg = df.groupby('Hour')['Total_Consumption'].mean()
    plt.figure(figsize=(12, 6))
    hourly_avg.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title('Average Consumption by Hour of Day', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Consumption (kWh)')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_weekly_pattern(df, output_path):
    """Plot average consumption by day of week"""
    weekly_avg = df.groupby('DayOfWeek')['Total_Consumption'].mean()
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 6))
    plt.bar(days, weekly_avg.values, color='coral', edgecolor='black')
    plt.title('Average Consumption by Day of Week', fontsize=14, fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Consumption (kWh)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_heatmap(df, household_cols, output_path):
    """Plot correlation between households"""
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[household_cols].corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Between Household Consumptions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df, output_path, top_n=20):
    """Plot feature importance"""
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Top Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(test_df, predictions_dict, output_path, last_samples=1344):
    """Plot actual vs predicted values"""
    plot_df = test_df.tail(last_samples)
    
    plt.figure(figsize=(15, 6))
    plt.plot(plot_df['Time'], plot_df['Total_Consumption'], 
             label='Actual', linewidth=1.5, alpha=0.8, color='black')
    
    colors = ['red', 'green', 'blue', 'orange']
    for idx, (name, pred) in enumerate(predictions_dict.items()):
        plt.plot(plot_df['Time'], pred[-last_samples:], 
                label=name, linewidth=1, alpha=0.7, color=colors[idx % len(colors)])
    
    plt.title('Actual vs Predicted (Last 2 Weeks)', fontsize=14, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Consumption (kWh)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_residuals(y_true, y_pred, output_path):
    """Plot residual analysis"""
    residuals = y_true.values - y_pred
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, residuals, alpha=0.3, s=10)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_all_plots(df, df_model, household_cols, importance_df, 
                       test_df, predictions_dict, y_test, best_pred, output_dir='outputs'):
    """Generate all visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    setup_plot_style()
    
    plots = [
        (plot_total_consumption, (df, os.path.join(output_dir, '01_total_consumption_time.png'))),
        (plot_consumption_distribution, (df, os.path.join(output_dir, '02_consumption_distribution.png'))),
        (plot_hourly_pattern, (df_model, os.path.join(output_dir, '03_hourly_pattern.png'))),
        (plot_weekly_pattern, (df_model, os.path.join(output_dir, '04_weekly_pattern.png'))),
        (plot_correlation_heatmap, (df, household_cols, os.path.join(output_dir, '05_household_correlation.png'))),
        (plot_feature_importance, (importance_df, os.path.join(output_dir, '06_feature_importance.png'))),
        (plot_predictions, (test_df, predictions_dict, os.path.join(output_dir, '07_predictions_comparison.png'))),
        (plot_residuals, (y_test, best_pred, os.path.join(output_dir, '08_residuals.png')))
    ]
    
    for plot_func, args in plots:
        plot_func(*args)
        print(f"✓ Saved: {os.path.basename(args[-1])}")
