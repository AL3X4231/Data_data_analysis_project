import os
import pandas as pd

folder = '.'  # Le script est déjà dans le dossier 'data'
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

# Récupérer l'ensemble des années présentes dans chaque fichier
years_per_file = []
for file in csv_files:
    df = pd.read_csv(os.path.join(folder, file), usecols=['year'])
    years_per_file.append(set(df['year'].dropna().unique()))

# Intersection de toutes les années présentes dans tous les fichiers
all_years = set.union(*years_per_file)
common_years = set.intersection(*years_per_file)

# Les années manquantes dans au moins un fichier
missing_years = all_years - common_years

print("Années absentes dans au moins un fichier CSV :", sorted(missing_years))