import pandas as pd

df = pd.read_csv('GDL-Total-Yearly-CO2-Emissions-(ton)-data.csv')
colonnes_a_supprimer = ["Continent","ISO_Code","Level","GDLCODE","Region"]
#df.columns = [col.replace('hdi_', '') for col in df.columns]
df = df.drop(columns=colonnes_a_supprimer)
df.to_csv('temp.csv', index=False)