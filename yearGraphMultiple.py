import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# Dictionary to map file patterns to prettier display names
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

def get_pretty_name(filename):
    """Get a pretty name for a dataset based on its filename."""
    for pattern, name in pretty_names.items():
        if pattern in filename:
            return name
    return os.path.splitext(os.path.basename(filename))[0]

def load_csv_files(data_folder='data'):
    """Load all CSV files from the data folder into a dictionary."""
    data_dict = {}
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    for file in csv_files:
        file_path = os.path.join(data_folder, file)
        try:
            df = pd.read_csv(file_path)
            # Ensure there are 'country' and 'year' columns
            if 'country' in df.columns and 'year' in df.columns:
                # Store the dataframe with a pretty name
                data_dict[get_pretty_name(file)] = df
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return data_dict

def filter_data(data_dict, countries, year_range):
    """Filter data for selected countries and year range."""
    filtered_data = {}
    for name, df in data_dict.items():
        # Filter by country and year
        mask = (df['country'].isin(countries)) & (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
        filtered_df = df[mask]
        
        # Only include datasets that have data for the selected criteria
        if not filtered_df.empty:
            filtered_data[name] = filtered_df
        else:
            print(f"Warning: No data available for '{name}' with the selected criteria. Skipping.")
    
    return filtered_data

def find_common_columns(filtered_data):
    """Find data columns (excluding 'country' and 'year')."""
    data_columns = {}
    
    for name, df in filtered_data.items():
        # Get columns that are not 'country' or 'year'
        cols = [col for col in df.columns if col not in ['country', 'year']]
        if cols:
            data_columns[name] = cols[0]  # For simplicity, use the first data column
    
    return data_columns

def create_3d_plot(filtered_data, data_columns, selected_datasets, countries, year_range):
    """Create 3D visualization for selected datasets."""
    if len(selected_datasets) < 3:
        print("Need at least 3 datasets for a 3D plot!")
        return
    
    # Create render directory if it doesn't exist
    render_dir = "render"
    os.makedirs(render_dir, exist_ok=True)
    
    # Use only the first 3 datasets for the 3D plot
    x_dataset, y_dataset, z_dataset = selected_datasets[:3]
    
    # Debug information
    print(f"Selected datasets: {x_dataset}, {y_dataset}, {z_dataset}")
    print(f"Data columns mapping: {data_columns}")
    for dataset in selected_datasets[:3]:
        print(f"\nDataset: {dataset}")
        print(f"Column names: {filtered_data[dataset].columns.tolist()}")
        print(f"Sample data:\n{filtered_data[dataset].head()}")
    
    # Prepare figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for countries
    colors = plt.cm.tab10.colors
    
    # Track which countries were successfully plotted
    plotted_countries = []
    
    for i, country in enumerate(countries):
        try:
            # Extract data for this country from each dataset
            x_data = filtered_data[x_dataset][filtered_data[x_dataset]['country'] == country]
            y_data = filtered_data[y_dataset][filtered_data[y_dataset]['country'] == country]
            z_data = filtered_data[z_dataset][filtered_data[z_dataset]['country'] == country]
            
            if x_data.empty or y_data.empty or z_data.empty:
                raise ValueError(f"Missing data for {country} in one or more datasets")
            
            # Get column names for each dataset
            x_col = data_columns[x_dataset]
            y_col = data_columns[y_dataset]
            z_col = data_columns[z_dataset]
            
            # Merge datasets on country and year
            merged_data = pd.merge(x_data[['country', 'year', x_col]], 
                                  y_data[['country', 'year', y_col]], 
                                  on=['country', 'year'], how='inner')
            
            merged_data = pd.merge(merged_data, 
                                  z_data[['country', 'year', z_col]], 
                                  on=['country', 'year'], how='inner')
            
            if merged_data.empty:
                raise ValueError(f"No overlapping years for {country} across all datasets")
            
            # Plot points for this country
            ax.scatter(
                merged_data[x_col], 
                merged_data[y_col], 
                merged_data[z_col],
                color=colors[i % len(colors)],
                label=country,
                s=50,
                alpha=0.7
            )
            
            # Connect points chronologically with lines
            sorted_data = merged_data.sort_values('year')
            ax.plot(
                sorted_data[x_col], 
                sorted_data[y_col], 
                sorted_data[z_col],
                color=colors[i % len(colors)],
                alpha=0.5
            )
            
            # Add year labels to points
            for idx, row in sorted_data.iterrows():
                ax.text(
                    row[x_col], row[y_col], row[z_col], 
                    str(int(row['year'])), 
                    size=8, 
                    color=colors[i % len(colors)]
                )
            
            plotted_countries.append(country)
        except Exception as e:
            print(f"Warning: Could not plot data for {country}: {e}")
    
    if not plotted_countries:
        print("No countries could be plotted with the selected criteria.")
        return
    
    # Set labels and title
    ax.set_xlabel(f"{x_dataset} ({x_col})")
    ax.set_ylabel(f"{y_dataset} ({y_col})")
    ax.set_zlabel(f"{z_dataset} ({z_col})")
    
    years_str = f"({year_range[0]}-{year_range[1]})"
    countries_str = ", ".join(plotted_countries) if len(plotted_countries) <= 3 else f"{plotted_countries[0]}, {plotted_countries[1]}, ... ({len(plotted_countries)} countries)"
    
    ax.set_title(f"3D Correlation: {countries_str}\n{years_str}")
    
    # Add legend and grid
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid(True)
    
    plt.tight_layout()
    # Save to the render directory with a timestamp
    filename = os.path.join(render_dir, f"3D_correlation_{x_dataset}_{y_dataset}_{z_dataset}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {filename}")
    plt.show()
        
    
def main():
    # Load and process data
    data_dict = load_csv_files()
    
    if not data_dict:
        print("No suitable CSV files found. Please check the data folder.")
        return
    
    print(f"Found {len(data_dict)} datasets:")
    for i, name in enumerate(data_dict.keys()):
        print(f"{i+1}. {name}")
    
    # A more comprehensive list of countries that might be in your datasets
    all_countries = [
        'Afghanistan', 'Albania', 'Algeria', 'Angola', 'Argentina', 'Australia',
        'Austria', 'Belgium', 'Brazil', 'Canada', 'China', 'Denmark', 'Egypt',
        'Finland', 'France', 'Germany', 'Greece', 'India', 'Indonesia', 'Italy',
        'Japan', 'Kenya', 'Mexico', 'Netherlands', 'New Zealand', 'Norway',
        'Russia', 'South Africa', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
        'United Kingdom', 'United States', 'World'
    ]
    
    # User-selectable parameters (these can be modified)
    selected_countries = [
        'Afghanistan', 
        'Albania', 
        'Algeria', 
        'Angola', 
        'Antigua and Barbuda', 
        'Argentina', 
        'Armenia',
        'Australia'
    ]
    
    # Extended year range to see climate change evolution
    year_range = (1990, 2020)
    
    # Filter data based on selected countries and years
    filtered_data = filter_data(data_dict, selected_countries, year_range)
    
    if not filtered_data:
        print("No data available for any of the selected datasets. Please check your criteria.")
        return
    
    if len(filtered_data) < 3:
        print(f"Warning: Only {len(filtered_data)} datasets have data for the selected criteria.")
        print("Need at least 3 datasets for a 3D plot. Please adjust your selection.")
        return
    
    # Print available countries in each dataset for debugging
    print("\nAvailable countries in each dataset:")
    for name, df in filtered_data.items():
        if 'country' in df.columns:
            countries = df['country'].unique()
            print(f"{name}: {', '.join(countries[:5])}{'...' if len(countries) > 5 else ''}")
        else:
            print(f"{name}: No 'country' column found!")
        
    # Find the data columns in each dataset
    data_columns = find_common_columns(filtered_data)
    
    # Select specific datasets that show climate change relationships
    # Looking for CO2 emissions, temperature changes, and another impact factor
    climate_datasets = []
    
    # Priority order for datasets (based on analysis interest)
    dataset_priorities = [
        'CO₂ Emissions (ton)',
        'Annual Temperature Anomalies (ºC)',
        'Average Temperature (ºC)',
        'Global Vegetation Index (GVI)',
        'Sea Level Rise (m)',
        'Share of Global Forest (%)',
        'Disaster Deaths',
        'HDR Composite Index'
    ]
    
    # Get available datasets in priority order
    for dataset in dataset_priorities:
        if dataset in filtered_data:
            climate_datasets.append(dataset)
            if len(climate_datasets) == 3:
                break
    
    # If we couldn't find the specific datasets, use the first 3 available
    if len(climate_datasets) < 3:
        climate_datasets = list(filtered_data.keys())[:3]
    
    print(f"\nSelected datasets for visualization: {climate_datasets}")
    
    # Create 3D visualization
    create_3d_plot(filtered_data, data_columns, climate_datasets, selected_countries, year_range)
    
    print("\nTo customize this visualization, modify:")
    print("- selected_countries list")
    print("- year_range tuple")
    print("- climate_datasets (currently using priority climate change indicators)")

if __name__ == "__main__":
    main()
