import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import sqrt

# Define pretty names for indicators
pretty_names = {
    'annual_temperature_anomalies': 'Annual Temperature Anomalies (ºC)',
    '(GVI)': 'Global Vegetation Index (GVI)',
    '(ton)': 'CO₂ Emissions (ton)',
    '(m)': 'Sea Level Rise (m)',
    '(ºC)': 'Average Temperature (ºC)',
    'HDR23_24_Composite_indices_complete_time_series': 'HDR Composite Index',
    'number_of_deaths_from_disasters': 'Disaster Deaths',
    'share_global_forest': 'Share of Global Forest (%)'
}

# Use pretty names for axis labels and heatmap ticks
def rename_for_plot(df):
    return df.rename(index=pretty_names, columns=pretty_names)

# Create render directory if it doesn't exist
os.makedirs('render', exist_ok=True)

# Load the data
print("Loading data from resultat_2015.csv...")
df = pd.read_csv('resultat_2015.csv')

# List of 8 countries to analyze
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

# Select all environmental indicators (excluding the country column)
indicators = [col for col in df.columns if col != 'country']

# Select only the relevant columns and set 'country' as the index
data_matrix = selected_countries.set_index('country')[indicators]

# --- Add visualization of the original data sample ---
print("\nCreating visualization of the original data sample...")

# Create a heatmap of the original data
plt.figure(figsize=(16, 10))
sns.heatmap(
    rename_for_plot(data_matrix), 
    annot=True, 
    cmap='YlGnBu', 
    fmt='.3f',
    linewidths=.5,
    annot_kws={"size": 8}
)
plt.title('Original Data Sample - 8 Countries with Environmental Indicators', fontsize=18)
plt.xlabel('Environmental Indicators', fontsize=14)
plt.ylabel('Countries', fontsize=14)
plt.tight_layout()
plt.savefig('render/original_data_sample.png', dpi=300)
print("Original data visualization saved to render/original_data_sample.png")

# Make sure all data is numeric
for col in data_matrix.columns:
    data_matrix[col] = pd.to_numeric(data_matrix[col], errors='coerce')

# Flag to enable/disable normalization
normalize_data = True  # Set to True to enable normalization

# Apply normalization if enabled
if normalize_data:
    # Normalize data: (x - min) / (max - min) for each column
    for col in data_matrix.columns:
        min_val = data_matrix[col].min()
        max_val = data_matrix[col].max()
        if max_val > min_val:  # Avoid division by zero
            data_matrix[col] = (data_matrix[col] - min_val) / (max_val - min_val)
    print("Data normalization applied: values scaled to [0,1] range")
else:
    print("Data normalization disabled: using original values")

# Create the centered data matrix (Xc)
# Subtract the mean of each column from each value in that column
column_means = data_matrix.mean()
centered_matrix = data_matrix.copy()
for col in centered_matrix.columns:
    centered_matrix[col] = (centered_matrix[col] - column_means[col]) / sqrt(len(data_matrix))

# Create the centered reduced data matrix (Xs)
# Standardize: subtract mean and divide by std deviation (z-score), then divide by sqrt(n)
column_stds = data_matrix.std(ddof=0)
centered_reduced_matrix = data_matrix.copy()
for col in centered_reduced_matrix.columns:
    if column_stds[col] != 0:
        centered_reduced_matrix[col] = (centered_reduced_matrix[col] - column_means[col]) / column_stds[col] / sqrt(len(data_matrix))
    else:
        centered_reduced_matrix[col] = 0  # Avoid division by zero

# --- Plot Xc and Xs side by side ---
fig, axes = plt.subplots(1, 2, figsize=(24, 10))

sns.heatmap(
    rename_for_plot(centered_matrix),
    annot=True,
    fmt='.4f',
    cmap='Blues',
    linewidths=.5,
    annot_kws={"size": 8},
    ax=axes[0]
)
axes[0].set_title('Centered Data Matrix (Xc)', fontsize=16)
axes[0].set_xlabel('Indicators', fontsize=12)
axes[0].set_ylabel('Country', fontsize=12)

sns.heatmap(
    rename_for_plot(centered_reduced_matrix),
    annot=True,
    fmt='.4f',
    cmap='Greens',
    linewidths=.5,
    annot_kws={"size": 8},
    ax=axes[1]
)
axes[1].set_title('Centered Reduced Data Matrix (Xs)', fontsize=16)
axes[1].set_xlabel('Indicators', fontsize=12)
axes[1].set_ylabel('Country', fontsize=12)

plt.tight_layout()
plt.savefig('render/centered_and_centered_reduced_matrices.png', dpi=300)
print("Plot of Xc and Xs saved to render/centered_and_centered_reduced_matrices.png")

# Calculate the variance-covariance matrix (Σ)
# Formula: Σ = (1/n) * Xc^T * Xc where n is the number of observations
n = len(data_matrix)
variance_covariance_matrix = centered_matrix.T.dot(centered_matrix) #/ n

# Print the variance-covariance matrix Σ in decimal format
pd.set_option('display.float_format', '{:.10f}'.format)
print("\nVariance-Covariance Matrix (Σ) in decimal format:")
print(variance_covariance_matrix)

# Create a heatmap of the variance-covariance matrix without scientific notation
plt.figure(figsize=(16, 14))

# Format the annotations to show decimal values without scientific notation
sns.heatmap(
    rename_for_plot(variance_covariance_matrix), 
    annot=True, 
    cmap='viridis', 
    fmt='.10f',  # Use fixed-point notation with 10 decimal places
    linewidths=.5,
    annot_kws={"size": 8}  # Make the text smaller to fit
)

plt.title('Variance-Covariance Matrix (Σ) of Environmental Indicators', fontsize=18)
plt.xlabel('Indicators', fontsize=14)
plt.ylabel('Indicators', fontsize=14)
plt.tight_layout()
plt.savefig('render/variance_covariance_decimal.png', dpi=300)
print("Decimal variance-covariance matrix visualization saved to render/variance_covariance_decimal.png")

# Make a more readable version by scaling the values to a better range
# Create a formatted version for display with better scaling
scaled_variance = variance_covariance_matrix.copy()
for col in scaled_variance.columns:
    max_val = abs(scaled_variance[col]).max()
    # Find an appropriate multiplier to get values in a readable range
    if max_val < 0.0001:
        multiplier = 1000000
        suffix = ' (×10⁻⁶)'
    elif max_val < 0.001:
        multiplier = 100000
        suffix = ' (×10⁻⁵)'
    elif max_val < 0.01:
        multiplier = 10000
        suffix = ' (×10⁻⁴)'
    elif max_val < 0.1:
        multiplier = 1000
        suffix = ' (×10⁻³)'
    elif max_val < 1:
        multiplier = 100
        suffix = ' (×10⁻²)'
    else:
        multiplier = 1
        suffix = ''
    
    scaled_variance[col] = scaled_variance[col] * multiplier
    # Rename columns to indicate scaling
    scaled_variance.rename(columns={col: col + suffix}, inplace=True)
    
# Create a better-formatted heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(
    rename_for_plot(scaled_variance), 
    annot=True, 
    cmap='viridis', 
    fmt='.4f',  # 4 decimal places is enough for the scaled values
    linewidths=.5,
    annot_kws={"size": 8}
)
plt.title('Scaled Variance-Covariance Matrix (Σ) for Environmental Indicators', fontsize=18)
plt.xlabel('Indicators', fontsize=14)
plt.ylabel('Indicators', fontsize=14)
plt.tight_layout()
plt.savefig('render/variance_covariance_scaled.png', dpi=300)
print("Scaled variance-covariance matrix visualization saved to render/variance_covariance_scaled.png")

# --- Add correlation matrix (R) calculation and visualization ---

# Calculate the correlation matrix (R)
correlation_matrix = data_matrix.corr()

# Print the correlation matrix R in decimal format
print("\nCorrelation Matrix (R) in decimal format:")
print(correlation_matrix)

# Create a heatmap of the correlation matrix
plt.figure(figsize=(16, 14))
sns.heatmap(
    rename_for_plot(correlation_matrix),
    annot=True,
    cmap='coolwarm',
    fmt='.4f',
    linewidths=.5,
    annot_kws={"size": 8}
)
plt.title('Correlation Matrix (R) of Environmental Indicators', fontsize=18)
plt.xlabel('Indicators', fontsize=14)
plt.ylabel('Indicators', fontsize=14)
plt.tight_layout()
plt.savefig('render/correlation_matrix.png', dpi=300)
print("Correlation matrix visualization saved to render/correlation_matrix.png")







# --- PCA: Calculate eigenvalues and eigenvectors of the correlation matrix ---
print("\nCalculating PCA components...")
eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

# Sort eigenvectors by eigenvalues in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(eigenvalues)
print(eigenvectors)

# Calculate the explained variance ratio
explained_variance_ratio = eigenvalues / sum(eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Calculate Principal Components
principal_components = centered_reduced_matrix.dot(eigenvectors)

# Calculate the total coordinates squared for each observation
total_squared_coords = pd.DataFrame(index=data_matrix.index)
for i in range(len(eigenvalues)):
    total_squared_coords[f'PC{i+1}'] = principal_components.iloc[:, i]**2
total_squared_coords['sum'] = total_squared_coords.sum(axis=1)

# Calculate QLT (Quality) for each observation with respect to each axis
# QLT(ind_i, α) = coord²(ind_i, α) / Σ coord²(ind_i)
qlts = pd.DataFrame(index=data_matrix.index)
for i in range(min(2, len(eigenvalues))):  # Calculate for first two components
    axis_name = f'PC{i+1}'
    qlts[f'QLT_{axis_name}'] = total_squared_coords[axis_name] / total_squared_coords['sum']

# Calculate total QLT for first two components
qlts['QLT_PC1_PC2'] = qlts.sum(axis=1)

# Calculate variable contributions to axes (OQE)
# OQE represents how much information a factor provides
oqes = pd.DataFrame(index=data_matrix.columns)
for i in range(min(2, len(eigenvalues))):
    axis_name = f'PC{i+1}'
    # For each variable, calculate its contribution to this component
    for j, var in enumerate(data_matrix.columns):
        # Squared correlation between variable and component
        oqes.loc[var, f'OQE_{axis_name}'] = eigenvectors[j, i]**2

# Calculate total OQE for first two components
oqes['OQE_PC1_PC2'] = oqes[[col for col in oqes.columns]].sum(axis=1)

# Print QLT and OQE values
print("\nQuality (QLT) values for observations:")
print(qlts)

print("\nObject Quality of Estimation (OQE) for variables:")
print(oqes)

# --- Visualize QLT and OQE ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Visualize QLT - Observation Quality for first two components
qlts_sorted = qlts.sort_values(by='QLT_PC1_PC2', ascending=False)
ax1.bar(qlts_sorted.index, qlts_sorted['QLT_PC1_PC2'], color='lightgreen')
ax1.bar(qlts_sorted.index, qlts_sorted['QLT_PC1'], color='darkgreen')
ax1.set_title('Quality (QLT) by Observation for PC1 and PC2', fontsize=16)
ax1.set_xlabel('Countries', fontsize=14)
ax1.set_ylabel('Quality (QLT)', fontsize=14)
ax1.set_xticklabels(qlts_sorted.index, rotation=45, ha='right', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.legend(['PC1+PC2', 'PC1 only'])

# Visualize OQE - Variable Quality
oqes_sorted = oqes.sort_values(by='OQE_PC1_PC2', ascending=False)
ax2.bar(
    [pretty_names.get(var, var) for var in oqes_sorted.index], 
    oqes_sorted['OQE_PC1_PC2'], 
    color='skyblue'
)
ax2.bar(
    [pretty_names.get(var, var) for var in oqes_sorted.index], 
    oqes_sorted['OQE_PC1'], 
    color='darkblue'
)
ax2.set_title('Object Quality of Estimation (OQE) by Variable', fontsize=16)
ax2.set_xlabel('Variables', fontsize=14)
ax2.set_ylabel('OQE Value', fontsize=14)
ax2.set_xticklabels([pretty_names.get(var, var) for var in oqes_sorted.index], rotation=45, ha='right', fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend(['PC1+PC2', 'PC1 only'])

plt.tight_layout()
plt.savefig('render/oqe_qlt_visualization.png', dpi=300)
print("OQE and QLT visualization saved to render/oqe_qlt_visualization.png")

# Create a biplot of the first two principal components with QLT as marker size
plt.figure(figsize=(14, 10))

# Plot observations (countries) with QLT as marker size
for i, country in enumerate(data_matrix.index):
    plt.scatter(
        principal_components.iloc[i, 0], 
        principal_components.iloc[i, 1],
        s=qlts.loc[country, 'QLT_PC1_PC2'] * 1000,  # Scale marker size by QLT
        alpha=0.7,
        label=country
    )

# Plot variable vectors
for i, var in enumerate(data_matrix.columns):
    plt.arrow(
        0, 0,  # Start at origin
        eigenvectors[i, 0] * np.sqrt(eigenvalues[0]) * 3,  # Scale by eigenvalue
        eigenvectors[i, 1] * np.sqrt(eigenvalues[1]) * 3, 
        head_width=0.05,
        head_length=0.1,
        fc='red',
        ec='red',
        alpha=0.5
    )
    plt.text(
        eigenvectors[i, 0] * np.sqrt(eigenvalues[0]) * 3.2,
        eigenvectors[i, 1] * np.sqrt(eigenvalues[1]) * 3.2,
        pretty_names.get(var, var),
        fontsize=10
    )

plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.title('PCA Biplot with QLT as Marker Size', fontsize=16)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', fontsize=14)
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', fontsize=14)
plt.legend(title='Countries', loc='best')
plt.tight_layout()
plt.savefig('render/pca_biplot_with_qlt.png', dpi=300)
print("PCA Biplot with QLT visualization saved to render/pca_biplot_with_qlt.png")

# Create a contribution biplot to show both QLT and OQE
plt.figure(figsize=(14, 10))

# Create a scatter plot with countries
max_contrib = 0
for i, country in enumerate(data_matrix.index):
    contrib = qlts.loc[country, 'QLT_PC1_PC2']
    max_contrib = max(max_contrib, contrib)
    plt.scatter(
        principal_components.iloc[i, 0], 
        principal_components.iloc[i, 1],
        s=contrib * 1000,  # Scale by contribution
        c='blue',
        alpha=0.6,
        label=country
    )
    plt.text(
        principal_components.iloc[i, 0] + 0.05,
        principal_components.iloc[i, 1] + 0.05,
        country,
        fontsize=9
    )

# Draw variable vectors with OQE as color intensity
for i, var in enumerate(data_matrix.columns):
    contrib = oqes.loc[var, 'OQE_PC1_PC2']
    color_intensity = 0.2 + 0.8 * (contrib / oqes['OQE_PC1_PC2'].max())
    plt.arrow(
        0, 0,
        eigenvectors[i, 0] * 3,
        eigenvectors[i, 1] * 3,
        head_width=0.1,
        head_length=0.2,
        fc=(1, 0, 0, color_intensity),  # Red with variable transparency
        ec=(1, 0, 0, color_intensity),
        linewidth=2
    )
    plt.text(
        eigenvectors[i, 0] * 3.2,
        eigenvectors[i, 1] * 3.2,
        pretty_names.get(var, var),
        fontsize=10,
        color='red'
    )

# Add grid and axis
plt.grid(True, linestyle='--', alpha=0.3)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)

# Add a circle of radius 1
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
plt.gca().add_patch(circle)

plt.title('PCA Contribution Biplot: QLT (point size) and OQE (arrow intensity)', fontsize=16)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)', fontsize=14)
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)', fontsize=14)
plt.axis('equal')  # Equal scaling
plt.tight_layout()
plt.savefig('render/pca_contribution_biplot.png', dpi=300)
print("PCA Contribution Biplot saved to render/pca_contribution_biplot.png")




