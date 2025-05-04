import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Create render directory if it doesn't exist
os.makedirs('render', exist_ok=True)

# Load the data
df = pd.read_csv('resultat_2015.csv')

# List of countries to analyze
countries = [
    'Afghanistan', 
    'Albania', 
    'Algeria', 
    'Angola', 
    'Antigua and Barbuda', 
    'Argentina', 
    'Armenia',
    'Australia'
]

# Select only the rows for these countries
selected_countries = df[df['country'].isin(countries)]

# List of the 8 environmental indicators we want to analyze
indicators = [
    'annual_temperature_anomalies',  # Temperature anomalies
    '(GVI)',                        # Vulnerability index
    '(ton)',                        # CO2 emissions
    '(m)',                          # Precipitation
    '(ºC)',                         # Temperature
    'HDR23_24_Composite_indices_complete_time_series',  # Human development
    'number_of_deaths_from_disasters',  # Disaster deaths
    'share_global_forest'           # Forest coverage
]

# Select only the relevant columns and set 'country' as the index
data_matrix = selected_countries.set_index('country')[indicators]

# Print the original data matrix
print("Original Data Matrix:")
print(data_matrix)
print("\n")

# Standardize the data (center and reduce)
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_matrix)

# Convert back to a DataFrame with proper labeling
standardized_df = pd.DataFrame(
    standardized_data, 
    index=data_matrix.index, 
    columns=data_matrix.columns
)

# Normalize the data (scale to range 0-1)
normalizer = MinMaxScaler()
normalized_data = normalizer.fit_transform(data_matrix)

# Convert normalized data back to a DataFrame with proper labeling
normalized_df = pd.DataFrame(
    normalized_data, 
    index=data_matrix.index, 
    columns=data_matrix.columns
)

# Print the standardized (centered and reduced) matrix
print("Standardized Matrix (Centered and Reduced, Z-score):")
print(standardized_df)
print("\n")

# Print the normalized matrix
print("Normalized Matrix (Min-Max Scaling, range 0-1):")
print(normalized_df)
print("\n")

# Create a heatmap visualization of the standardized data
plt.figure(figsize=(12, 8))
sns.heatmap(standardized_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Standardized Environmental Indicators for Selected Countries')
plt.tight_layout()
plt.savefig('render/standardized_matrix_heatmap.png', dpi=300)
plt.close()

# Create a heatmap visualization of the normalized data
plt.figure(figsize=(12, 8))
sns.heatmap(normalized_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
plt.title('Normalized Environmental Indicators for Selected Countries')
plt.tight_layout()
plt.savefig('render/normalized_matrix_heatmap.png', dpi=300)
plt.close()

# Calculate and display correlations between indicators
correlation_matrix = data_matrix.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Between Environmental Indicators')
plt.tight_layout()
plt.savefig('render/correlation_matrix.png', dpi=300)
plt.close()

# Create a comparison plot to show the difference between standardization and normalization
fig, axes = plt.subplots(3, 1, figsize=(14, 18))

# Original data visualization
sns.heatmap(data_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=.5, ax=axes[0])
axes[0].set_title('Original Data Matrix')

# Standardized data visualization
sns.heatmap(standardized_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=axes[1])
axes[1].set_title('Standardized Data (Z-score)')

# Normalized data visualization
sns.heatmap(normalized_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5, ax=axes[2])
axes[2].set_title('Normalized Data (0-1 scale)')

plt.tight_layout()
plt.savefig('render/data_transformation_comparison.png', dpi=300)
plt.close()

# Create a radar chart to visualize the normalized data
categories = data_matrix.columns
N = len(categories)

# Create a function to plot the radar chart
def radar_chart(df, title):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Plot for each country
    for i, country in enumerate(df.index):
        values = df.loc[country].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=country)
        ax.fill(angles, values, alpha=0.1)
    
    # Set the labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title(title, size=15)
    return fig

# Create radar chart for normalized data
radar_fig = radar_chart(normalized_df, 'Environmental Indicators Comparison (Normalized)')
radar_fig.savefig('render/radar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some summary statistics
print("Summary Statistics for Original Data:")
print(data_matrix.describe())

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Print a brief interpretation
print("\nInterpretation:")
print("- Values in the standardized matrix (Z-score) represent how many standard deviations")
print("  each country's value is from the mean for each indicator.")
print("- Positive values are above the mean, negative values are below the mean.")
print("- Values in the normalized matrix range from 0 to 1, where 0 represents the minimum")
print("  value in the dataset and 1 represents the maximum value.")
print("- The correlation matrix shows relationships between different environmental indicators.")
print("- All visualizations have been saved to the 'render' directory.")