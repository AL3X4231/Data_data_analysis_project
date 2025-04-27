import pandas as pd

# Charger le fichier
df = pd.read_csv("data/GDL-GDL-Vulnerability-Index-(GVI)-data.csv", encoding='utf-8')

# Supprimer les colonnes inutiles
colonnes_a_supprimer = ["Continent", "ISO_Code", "Level", "GDLCODE", "Region"]
df = df.drop(columns=colonnes_a_supprimer, errors='ignore')

# Sauvegarder le résultat dans un nouveau fichier CSV
df.to_csv("data/GDL-GVI-data_clean.csv", index=False)