import pandas as pd

# Charger le fichier
df = pd.read_csv('GDL-GDL-Vulnerability-Index-(GVI)-data.csv')

# Transformation
df_long = df.melt(id_vars=['country'], var_name='year', value_name='Precipiation')

# Sauvegarder le résultat
df_long.to_csv('fichier_long.csv', index=False)