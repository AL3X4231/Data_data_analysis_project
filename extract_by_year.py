import pandas as pd
import os
import sys

def extract_data_for_year(year):
    """
    Extracts data for a specific year from all the CSV files and combines them into a single dataframe.
    
    Args:
        year (str or int): The year to extract data for
    
    Returns:
        pandas.DataFrame: A dataframe containing combined data for the specified year
    """
    # Convert year to string for column matching
    year_str = str(year)
    
    # Define the data folder path
    data_folder = "data"
    
    # Define the datasets to be processed (with their file paths)
    datasets = {
        "temperature": os.path.join(data_folder, "GDL-Yearly-Average-Surface-Temperature-(ºC)-data.csv"),
        "precipitation": os.path.join(data_folder, "GDL-Total-Yearly-Precipitation-(m)-data.csv"),
        "co2_emissions": os.path.join(data_folder, "GDL-Total-Yearly-CO2-Emissions-(ton)-data.csv")
    }
    
    # Also handle temperature anomalies which has a different format
    temp_anomalies_file = os.path.join(data_folder, "annual-temperature-anomalies.csv")
    
    # Dictionary to store the extracted data for each dataset
    extracted_data = {}
    
    # Process standard datasets (those with year in column headers)
    for dataset_name, file_path in datasets.items():
        try:
            print(f"Processing {dataset_name} dataset from {file_path}")
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            
            # Check if the year column exists
            if f'"{year_str}"' in df.columns:
                year_col = f'"{year_str}"'
            elif year_str in df.columns:
                year_col = year_str
            else:
                print(f"Year {year} not found in {dataset_name} dataset. Available years: {[col for col in df.columns if col.strip('\"').isdigit()]}")
                continue
                
            # Extract country and data for the specified year
            result = df[["Country", year_col]].copy()
            result.columns = ["Country", f"{dataset_name}_{year}"]
            
            # Add to the extracted data dictionary
            extracted_data[dataset_name] = result
            print(f"Successfully extracted {dataset_name} data for {year}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    # Process temperature anomalies which has a different structure (rows contain years)
    try:
        if os.path.exists(temp_anomalies_file):
            print(f"Processing temperature anomalies from {temp_anomalies_file}")
            df = pd.read_csv(temp_anomalies_file, encoding='utf-8', on_bad_lines='skip', header=None)
            
            # Assuming the format is: Country, CountryCode, Year, Value
            if len(df.columns) >= 4:
                # Rename columns for clarity
                df.columns = ["Country", "CountryCode", "Year", "Temperature_Anomaly"] if len(df.columns) == 4 else df.columns
                
                # Filter for the specified year
                year_data = df[df["Year"] == int(year)]
                
                if not year_data.empty:
                    # Extract country and temperature anomaly data
                    result = year_data[["Country", "Temperature_Anomaly"]].copy()
                    result.columns = ["Country", f"temp_anomaly_{year}"]
                    
                    # Add to the extracted data dictionary
                    extracted_data["temp_anomaly"] = result
                    print(f"Successfully extracted temperature anomaly data for {year}")
                else:
                    print(f"No temperature anomaly data found for year {year}")
            else:
                print("Temperature anomalies file does not have the expected columns")
    except Exception as e:
        print(f"Error processing temperature anomalies: {e}")
    
    # Ajouter le traitement pour share-global-forest.csv
    share_forest_file = os.path.join(data_folder, "share-global-forest.csv")
    try:
        if os.path.exists(share_forest_file):
            print(f"Processing share of global forest area from {share_forest_file}")
            df = pd.read_csv(share_forest_file, encoding='utf-8', on_bad_lines='skip')
            df_year = df[df["Year"] == int(year)][["xEntity", "Share of global forest area"]].copy()
            df_year.columns = ["Country", f"share_global_forest_{year}"]
            extracted_data["share_global_forest"] = df_year
            print(f"Successfully extracted share of global forest area for {year}")
    except Exception as e:
        print(f"Error processing share-global-forest: {e}")

    # Merge all datasets on Country
    if not extracted_data:
        print(f"No data found for year {year}")
        return None
    
    # Start with the first dataset
    first_dataset = next(iter(extracted_data.values()))
    merged_data = first_dataset.copy()
    
    # Merge with the rest of the datasets
    for name, df in extracted_data.items():
        if df is not first_dataset:
            merged_data = pd.merge(merged_data, df, on="Country", how="outer")
    
    return merged_data

def main():
    """
    Main function to run the script.
    """
    if len(sys.argv) > 1:
        try:
            year = int(sys.argv[1])
            print(f"Extracting data for year {year}...")
        except ValueError:
            print("Error: Year must be a valid integer.")
            return
    else:
        # If no year is provided, ask the user
        year = input("Enter the year to extract data for: ")
        try:
            year = int(year)
        except ValueError:
            print("Error: Year must be a valid integer.")
            return
    
    # Extract data for the specified year
    result = extract_data_for_year(year)
    
    if result is not None:
        # Save the result to a CSV file
        output_file = f"combined_data_{year}.csv"
        result.to_csv(output_file, index=False)
        print(f"Data successfully saved to {output_file}")
        
        # Print a sample of the data
        print("\nSample of the combined data:")
        print(result.head())
        
        # Print some statistics
        print("\nStatistics:")
        print(f"Number of countries with data: {len(result)}")
        print(f"Number of data columns: {len(result.columns) - 1}")  # Subtract 1 for the Country column
        
        # Check for missing values
        missing_values = result.isna().sum()
        print("\nMissing values per column:")
        for col, missing in missing_values.items():
            if missing > 0:
                print(f"{col}: {missing} missing values ({missing/len(result)*100:.2f}%)")
    else:
        print("No data extracted. Please check the available years in your datasets.")

if __name__ == "__main__":
    main()