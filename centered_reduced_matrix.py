import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from math import sqrt

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

# Make sure all data is numeric
for col in data_matrix.columns:
    data_matrix[col] = pd.to_numeric(data_matrix[col], errors='coerce')

# Print the original data matrix X
print("\nOriginal Data Matrix (X):")
print(data_matrix)

# Calculate the mean of each column
column_means = data_matrix.mean()
print("\nColumn Means:")
print(column_means)

# Calculate the standard deviation of each column
column_std = data_matrix.std()
print("\nColumn Standard Deviations:")
print(column_std)

n = len(data_matrix)
print(f"\nNumber of observations (n): {n}")

# Create the centered data matrix (Xc)
# Subtract the mean of each column from each value in that column
centered_matrix = data_matrix.copy()
for col in centered_matrix.columns:
    centered_matrix[col] = (centered_matrix[col] - column_means[col]) / sqrt(n)

# Print the centered data matrix Xc
print("\nCentered Data Matrix (Xc):")
print(centered_matrix)

# Create the centered reduced data matrix (Xs)
# Divide each value in the centered matrix by the standard deviation of its column
centered_reduced_matrix = centered_matrix.copy()
for col in centered_reduced_matrix.columns:
    if column_std[col] > 0:  # Avoid division by zero
        centered_reduced_matrix[col] = centered_reduced_matrix[col] / column_std[col]

# Print the centered reduced data matrix Xs
print("\nCentered Reduced Data Matrix (Xs):")
print(centered_reduced_matrix)

# Calculate the variance-covariance matrix (Σ)
# Formula: Σ = (1/n) * Xc^T * Xc where n is the number of observations
variance_covariance_matrix = centered_matrix.T.dot(centered_matrix) # / n

# Print the variance-covariance matrix Σ
print("\nVariance-Covariance Matrix (Σ):")
print(variance_covariance_matrix)

# Calculate the correlation matrix (R)
# We can use pandas built-in corr() function
correlation_matrix = data_matrix.corr()

# Print the correlation matrix R
print("\nCorrelation Matrix (R):")
print(correlation_matrix)

# Verify with sklearn's StandardScaler
scaler = StandardScaler()
standardized_data = scaler.fit_transform(data_matrix)
standardized_df = pd.DataFrame(
    standardized_data, 
    index=data_matrix.index, 
    columns=data_matrix.columns
)

print("\nScikitlearn StandardScaler output (should match Xs):")
print(standardized_df)

# Create a side-by-side visualization of X, Xc, and Xs matrices
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Original data visualization (X)
sns.heatmap(data_matrix, annot=True, cmap='Blues', fmt='.2f', linewidths=.5, ax=axes[0])
axes[0].set_title('Original Data Matrix (X)', fontsize=14)

# Centered data visualization (Xc)
sns.heatmap(centered_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=.5, ax=axes[1])
axes[1].set_title('Centered Data Matrix (Xc)', fontsize=14)

# Centered reduced data visualization (Xs)
sns.heatmap(centered_reduced_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f', linewidths=.5, ax=axes[2])
axes[2].set_title('Centered Reduced Data Matrix (Xs)', fontsize=14)

plt.tight_layout()
plt.savefig('render/centered_reduced_matrices.png', dpi=300)
print("Visualization saved to render/centered_reduced_matrices.png")

# Create visualizations of variance-covariance matrix Σ and correlation matrix R
fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# Variance-covariance matrix visualization (Σ)
sns.heatmap(variance_covariance_matrix, annot=True, cmap='viridis', fmt='.2e', linewidths=.5, ax=axes[0])
axes[0].set_title('Variance-Covariance Matrix (Σ)', fontsize=14)

# Correlation matrix visualization (R)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5, ax=axes[1])
axes[1].set_title('Correlation Matrix (R)', fontsize=14)

plt.tight_layout()
plt.savefig('render/variance_correlation_matrices.png', dpi=300)
print("Variance-covariance and correlation matrices visualization saved to render/variance_correlation_matrices.png")

# Create a more detailed visualization of the centered reduced matrix (Xs)
plt.figure(figsize=(14, 10))
sns.heatmap(centered_reduced_matrix, annot=True, cmap='RdBu_r', center=0, 
           fmt='.2f', linewidths=.5, cbar_kws={'label': 'Standard Deviations'})
plt.title('Centered Reduced Data Matrix (Xs)', fontsize=16)
plt.tight_layout()
plt.savefig('render/centered_reduced_matrix_detailed.png', dpi=300)
print("Detailed visualization saved to render/centered_reduced_matrix_detailed.png")

# Create detailed visualization of the correlation matrix (R)
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
           fmt='.2f', linewidths=.5, mask=mask, 
           cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Correlation Matrix (R)', fontsize=16)
plt.tight_layout()
plt.savefig('render/correlation_matrix_detailed.png', dpi=300)
print("Detailed correlation matrix visualization saved to render/correlation_matrix_detailed.png")

# Create a bar chart showing the mean values of each indicator
plt.figure(figsize=(14, 6))
column_means.plot(kind='bar', color='skyblue')
plt.title('Mean Values of Environmental Indicators', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Value')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('render/indicator_means.png', dpi=300)
print("Mean values visualization saved to render/indicator_means.png")

# Create a bar chart showing the standard deviation of each indicator
plt.figure(figsize=(14, 6))
column_std.plot(kind='bar', color='lightgreen')
plt.title('Standard Deviations of Environmental Indicators', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Standard Deviation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('render/indicator_std_deviations.png', dpi=300)
print("Standard deviation visualization saved to render/indicator_std_deviations.png")

print("\nMatrices calculated:")
print("1. Original Data Matrix (X)") 
print("2. Centered Data Matrix (Xc): Xc = X - μ")
print("3. Centered Reduced Data Matrix (Xs): Xs = Xc / σ") 
print("4. Variance-Covariance Matrix (Σ): Σ = (1/n) * Xc^T * Xc")
print("5. Correlation Matrix (R): R = corr(X) or equivalently Xs^T * Xs / (n-1)")

print("\nFormulas used:")
print("1. Centered Data Matrix (Xc): Xc = X - μ")
print("   where μ is the mean vector of X")
print("2. Centered Reduced Data Matrix (Xs): Xs = Xc / σ")
print("   where σ is the standard deviation vector of X")
print("3. Variance-Covariance Matrix (Σ): Σ = (1/n) * Xc^T * Xc")
print("   where n is the number of observations and Xc^T is the transpose of Xc")
print("4. Correlation Matrix (R): R = corr(X)")
print("   can also be calculated as R = diag(Σ)^(-1/2) * Σ * diag(Σ)^(-1/2)")

print("\nInterpretation:")
print("- In the centered matrix (Xc), values represent the deviation from the mean")
print("- Positive values are above the mean, negative values are below the mean")
print("- In the centered reduced matrix (Xs), values represent how many standard")
print("  deviations each observation is from the mean (z-scores)")
print("- The diagonal elements of the variance-covariance matrix (Σ) are the variances")
print("  of each variable, and the off-diagonal elements are the covariances between pairs")
print("  of variables")
print("- The correlation matrix (R) contains correlation coefficients between all pairs")
print("  of variables, ranging from -1 (perfect negative correlation) to +1 (perfect")
print("  positive correlation), with 0 indicating no linear correlation")