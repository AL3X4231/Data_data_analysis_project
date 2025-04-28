import pandas as pd

# Charger le fichier CSV
df = pd.read_csv('annual-temperature-anomalies.csv')

# Trier par le nom du pays
df_sorted = df.sort_values(by=['country', 'year'])

# Sauvegarder le résultat
df_sorted.to_csv('annual-temperature-anomalies.csv', index=False)