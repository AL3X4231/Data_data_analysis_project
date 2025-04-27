import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Define paths to datasets
data_folder = "data"
datasets = {
    "forest": "share-global-forest.csv",
    "temperature": "GDL-Yearly-Average-Surface-Temperature-(ÂºC)-data.csv",
    "precipitation": "GDL-Total-Yearly-Precipitation-(m)-data.csv",
    "co2_emissions": "GDL-Total-Yearly-CO2-Emissions-(ton)-data.csv",
    "vulnerability": "GDL-GDL-Vulnerability-Index-(GVI)-data.csv",
    "temp_anomalies": "annual-temperature-anomalies.csv",
    "disaster_deaths": "number-of-deaths-from-disasters.csv",
    "human_dev": "HDR23-24_Composite_indices_complete_time_series.csv"
}

# Function to read and process the forest dataset which has no headers
def process_forest_data(file_path):
    print(f"Reading forest dataset from {file_path}")
    
    # The file doesn't have headers, so we'll add them
    df = pd.read_csv(file_path)
    
    if "xEntity" in df.columns:
        # Rename columns for clarity
        df = df.rename(columns={"xEntity": "Country", "Code": "CountryCode", "Year": "Year", "Share of global forest area": "ForestShare"})
        
        # Select the most recent year (2020) for each country
        most_recent_data = df[df["Year"] == 2020].copy()
        
        # If a country doesn't have 2020 data, use the most recent available
        countries_with_2020 = set(most_recent_data["Country"])
        
        # Create a dataframe with just the most recent data for each country
        result_df = most_recent_data.drop_duplicates(subset=["Country"])[["Country", "ForestShare"]].copy()
        return result_df.set_index("Country")
    else:
        print(f"Unexpected format in forest data file")
        return pd.DataFrame()

# Function to read datasets with specific formats
def read_dataset(name, file_path):
    print(f"Processing {name} dataset")
    
    # Handle forest data specially
    if name == "forest":
        return process_forest_data(file_path)
    
    try:
        # For other files, try to read with headers
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        
        # Different datasets may have different structures
        # Attempt to find country column (could be 'Country', 'Entity', etc.)
        country_col = None
        for col in df.columns:
            if isinstance(col, str) and col.lower() in ['country', 'entity', 'nation', 'area', 'xentity']:
                country_col = col
                break
        
        if country_col is None:
            print(f"Could not identify country column in {name} dataset")
            print(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # For time series data, select the most recent year with good coverage
        year_cols = []
        for col in df.columns:
            if isinstance(col, str) and col.isdigit() and int(col) >= 2000:
                year_cols.append(col)
            elif isinstance(col, int) and col >= 2000:
                year_cols.append(str(col))
        
        if year_cols:
            # Select the most recent year with good data coverage
            most_recent = max(year_cols)
            
            # We need just one value per country
            result_df = df[[country_col, most_recent]].copy()
            result_df.columns = [country_col, name]
            
            # Convert to numeric, coercing errors to NaN
            result_df[name] = pd.to_numeric(result_df[name], errors='coerce')
            
            return result_df.set_index(country_col)
        else:
            # If there are no year columns, try to identify a key indicator column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = numeric_cols[0]
                result_df = df[[country_col, selected_col]].copy()
                result_df.columns = [country_col, name]
                
                # Convert to numeric, coercing errors to NaN
                result_df[name] = pd.to_numeric(result_df[name], errors='coerce')
                
                return result_df.set_index(country_col)
            else:
                print(f"No suitable numeric data found in {name} dataset")
                return pd.DataFrame()
                
    except Exception as e:
        print(f"Error processing {name} dataset: {e}")
        return pd.DataFrame()

# Load all datasets
processed_data = {}
for name, filename in datasets.items():
    file_path = os.path.join(data_folder, filename)
    if os.path.exists(file_path):
        df = read_dataset(name, file_path)
        if not df.empty:
            processed_data[name] = df
            print(f"Successfully processed {name} dataset with {len(df)} countries")
        else:
            print(f"Empty dataframe returned for {name}")
    else:
        print(f"File not found: {file_path}")

# Merge all datasets on country names
# Start with the dataset that has the most countries
dataset_sizes = {name: len(df) for name, df in processed_data.items() if isinstance(df, pd.DataFrame)}
if not dataset_sizes:
    print("No valid datasets found.")
    exit(1)

# Find the dataset with the most countries
largest_dataset = max(dataset_sizes, key=dataset_sizes.get)
merged_data = processed_data[largest_dataset].copy()
print(f"Starting with {largest_dataset} dataset which has {len(merged_data)} countries")

# Merge other datasets
for name, df in processed_data.items():
    if name != largest_dataset and isinstance(df, pd.DataFrame) and not df.empty:
        merged_data = merged_data.join(df, how='left')
        print(f"After merging {name}, shape is {merged_data.shape}")

print(f"Initial merged dataset shape: {merged_data.shape}")

# Print number of missing values per column
missing_values = merged_data.isna().sum()
print("\nMissing values per column:")
for col, missing in missing_values.items():
    print(f"{col}: {missing} missing values ({missing/len(merged_data)*100:.2f}%)")

# Drop countries with too many missing values
missing_threshold = len(merged_data.columns) // 2  # At least half of the columns must have data
merged_data = merged_data.dropna(thresh=missing_threshold)
print(f"\nAfter dropping rows with many missing values: {merged_data.shape}")

# Make sure all columns are numeric
for col in merged_data.columns:
    merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')

# Fill remaining NAs with column means
for col in merged_data.columns:
    col_mean = merged_data[col].mean()
    merged_data[col] = merged_data[col].fillna(col_mean)
    print(f"Filled missing values in {col} with mean: {col_mean:.4f}")

# Drop any remaining rows with NaN values
merged_data = merged_data.dropna()
print(f"Final dataset shape: {merged_data.shape}")

# Print first few rows to verify data
print("\nSample of the merged data:")
print(merged_data.head())

# Standardize the data for PCA
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data)

# Perform PCA
pca = PCA()
principal_components = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

print(f"\nExplained variance by components: {explained_variance}")

# Create a biplot - this shows both countries and variables in the same plot
def biplot(score, coeff, labels=None, variable_names=None):
    plt.figure(figsize=(14, 10))
    
    # Plot countries as points
    xs = score[:,0]
    ys = score[:,1]
    plt.scatter(xs, ys, c='blue', alpha=0.5)
    
    # Add country labels
    if labels is not None:
        for i, label in enumerate(labels):
            plt.annotate(label, (xs[i], ys[i]), fontsize=8)
    
    # Plot variable vectors
    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0]*3, coeff[i,1]*3, 
                  color='red', alpha=0.5, head_width=0.05)
        if variable_names is not None:
            plt.text(coeff[i,0]*3.2, coeff[i,1]*3.2, variable_names[i], 
                    color='green', fontsize=10)
    
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid()
    plt.title('PCA Biplot: Countries and Environmental Variables')
    plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}%)')
    plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}%)')
    
    # Add a unit circle
    circle = plt.Circle((0,0), 1, color='green', fill=False)
    plt.gca().add_artist(circle)
    
    plt.tight_layout()
    plt.savefig('pca_biplot.png', dpi=300)
    plt.close()

# Visualize the loadings (how each variable contributes to each principal component)
def plot_loadings_heatmap(pca, feature_names):
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings.iloc[:, :min(4, loadings.shape[1])], annot=True, cmap='coolwarm', center=0)
    plt.title('Variable Contributions to Principal Components')
    plt.tight_layout()
    plt.savefig('pca_loadings_heatmap.png', dpi=300)
    plt.close()
    
    return loadings

# Correlation matrix heatmap
def plot_correlation_matrix(data):
    corr = data.corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                mask=mask, linewidths=.5, fmt='.2f')
    plt.title('Correlation Matrix Between Environmental Indicators')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()
    
    return corr

# Create visualizations
if merged_data.shape[0] > 0 and merged_data.shape[1] > 1:
    # Create the biplot
    biplot(principal_components[:, :2], 
           pca.components_.T[:, :2],
           labels=merged_data.index,
           variable_names=merged_data.columns)
    
    # Create the loadings heatmap
    loadings = plot_loadings_heatmap(pca, merged_data.columns)
    
    # Create the correlation matrix
    corr_matrix = plot_correlation_matrix(merged_data)
    
    # Create a bar plot for explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid')
    plt.axhline(y=0.8, color='r', linestyle='-')
    plt.title('Scree Plot: Explained Variance by Principal Components')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.tight_layout()
    plt.savefig('pca_scree_plot.png', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\nSummary of Principal Component Analysis:")
    print(f"Number of countries analyzed: {merged_data.shape[0]}")
    print(f"Number of environmental indicators: {merged_data.shape[1]}")
    
    print("\nExplained Variance by Principal Components:")
    for i, var in enumerate(explained_variance[:min(5, len(explained_variance))]):  # Show first 5 components or fewer
        print(f"PC{i+1}: {var*100:.2f}% ({np.sum(explained_variance[:i+1])*100:.2f}% cumulative)")
    
    print("\nTop variables contributing to PC1:")
    pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
    for var in pc1_loadings.index[:min(3, len(pc1_loadings))]:
        print(f"{var}: {loadings.loc[var, 'PC1']:.4f}")
    
    if loadings.shape[1] > 1:
        print("\nTop variables contributing to PC2:")
        pc2_loadings = loadings['PC2'].abs().sort_values(ascending=False)
        for var in pc2_loadings.index[:min(3, len(pc2_loadings))]:
            print(f"{var}: {loadings.loc[var, 'PC2']:.4f}")
    
    print("\nStrongest correlations between variables:")
    # Get strongest correlations (excluding self-correlations)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            correlations.append((col_i, col_j, corr_val))
    
    # Sort by absolute correlation value and print top 5
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    for var1, var2, corr in correlations[:min(5, len(correlations))]:
        print(f"{var1} and {var2}: {corr:.4f}")
else:
    print("Not enough data to perform PCA. Please check your datasets.")