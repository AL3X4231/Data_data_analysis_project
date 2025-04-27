import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('HDR23-24_Composite_indices_complete_time_series.csv')

# Trier par le nom du pays
df_sorted = df.sort_values(by=['country', 'year'])

# Sauvegarder le résultat
df_sorted.to_csv('HDR23-24_Composite_indices_complete_time_series.csv', index=False)