import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the engineered dataset
file_path = 'processed_data/engineered_features_all_households.csv'
df = pd.read_csv(file_path)

# Identify the household main columns (target variables) BEFORE renaming
household_main_cols = [col for col in df.columns if col.startswith('Sum [kWh].') and not any(
    sub in col for sub in ['_minute', '_hour', '_day', '_month', '_quarter', '_lag', '_rolling'])]

# Sanitize column names for compatibility
df.columns = df.columns.str.replace('[', '_', regex=False)
df.columns = df.columns.str.replace(']', '', regex=False)
df.columns = df.columns.str.replace(' ', '_', regex=False)

# Adjust household_main_cols based on sanitized columns
household_main_cols = [col.replace('[', '_').replace(']', '').replace(' ', '_') for col in household_main_cols]

# Create output directory for correlation results
output_dir = 'correlation_results/'
os.makedirs(output_dir, exist_ok=True)

# Perform correlation analysis for each household
for household in household_main_cols:
    print(f"\nCorrelation analysis for household: {household}")

    # Define features and target
    feature_cols = [col for col in df.columns if col.startswith(household) and col != household]

    # Pearson Correlation
    pearson_corr_matrix = df[feature_cols + [household]].corr(method='pearson')
    pearson_target_corr = pearson_corr_matrix[household].drop(household)

    # Spearman Correlation
    spearman_corr_matrix = df[feature_cols + [household]].corr(method='spearman')
    spearman_target_corr = spearman_corr_matrix[household].drop(household)

    # Display Pearson and Spearman correlation values
    print("\nPearson Correlation with Target:")
    print(pearson_target_corr.sort_values(ascending=False))
    print("\nSpearman Correlation with Target:")
    print(spearman_target_corr.sort_values(ascending=False))

    # Plot Pearson Correlation Heatmap
    plt.figure(figsize=(14, 10))  # Increased figure size
    sns.heatmap(pearson_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 10})
    plt.title(f'Pearson Correlation Matrix for {household}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{household}_pearson_correlation_heatmap.png')
    plt.close()

    # Plot Spearman Correlation Heatmap
    plt.figure(figsize=(14, 10))  # Increased figure size
    sns.heatmap(spearman_corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 10})
    plt.title(f'Spearman Correlation Matrix for {household}', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{household}_spearman_correlation_heatmap.png')
    plt.close()

    # Plot Pearson Correlation with Target
    plt.figure(figsize=(14, 6))
    pearson_target_corr.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title(f'Pearson Feature Correlation with {household}', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{household}_pearson_correlation_with_target.png')
    plt.close()

    # Plot Spearman Correlation with Target
    plt.figure(figsize=(14, 6))
    spearman_target_corr.sort_values(ascending=False).plot(kind='bar', color='salmon')
    plt.title(f'Spearman Feature Correlation with {household}', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Correlation', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}{household}_spearman_correlation_with_target.png')
    plt.close()

# Combine data across households using generic feature names
print("\nCombining data across all households with generic feature names")

# Create a new DataFrame to hold combined data
combined_df = pd.DataFrame()

for i, household in enumerate(household_main_cols):
    feature_cols = [col for col in df.columns if col.startswith(household) and col != household]

    # Rename features to generic names (e.g., "feature_1", "feature_2", ...)
    renamed_features = {feature: f'feature_{idx + 1}' for idx, feature in enumerate(feature_cols)}

    # Append to combined_df
    temp_df = df[feature_cols].rename(columns=renamed_features)
    temp_df['target'] = df[household]  # Add the target column as 'target'
    combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

# Pearson Correlation for combined data
combined_pearson_corr_matrix = combined_df.corr(method='pearson')

# Spearman Correlation for combined data
combined_spearman_corr_matrix = combined_df.corr(method='spearman')

# Plot Combined Pearson Correlation Heatmap
plt.figure(figsize=(22, 18))
sns.heatmap(combined_pearson_corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Combined Pearson Correlation Matrix for All Households (Generic Features)', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}combined_pearson_correlation_heatmap_generic.png')
plt.close()

# Plot Combined Spearman Correlation Heatmap
plt.figure(figsize=(22, 18))
sns.heatmap(combined_spearman_corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Combined Spearman Correlation Matrix for All Households (Generic Features)', fontsize=18)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}combined_spearman_correlation_heatmap_generic.png')
plt.close()

print("Combined correlation analysis completed. Plots are saved in the 'correlation_results/' directory.")
