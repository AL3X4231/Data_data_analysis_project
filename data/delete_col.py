import pandas as pd

df = pd.read_csv('share-global-forest.csv')
colonnes_a_supprimer = ['Code']
#df.columns = [col.replace('hdi_', '') for col in df.columns]
df = df.drop(columns=colonnes_a_supprimer)
df.to_csv('temp.csv', index=False)