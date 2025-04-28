import os
import pandas as pd
import glob

def get_csv_files():
    """Récupère tous les fichiers CSV du dossier data"""
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    return [f for f in csv_files if os.path.basename(f) != 'resultat.csv']  # Exclure le fichier résultat s'il existe déjà

def extract_metric_name(file_path):
    """Extrait un nom de métrique à partir du nom du fichier CSV"""
    base_name = os.path.basename(file_path)
    name = os.path.splitext(base_name)[0]
    
    # Essayer d'obtenir la partie importante du nom
    if 'GDL-' in name:
        metric = name.split('-')[-2] if '-' in name else name
    else:
        metric = name.replace('-', '_')
    
    return metric

def process_csv_files(year):
    """Traite tous les fichiers CSV pour une année donnée et crée un nouveau DataFrame"""
    csv_files = get_csv_files()
    
    # Dictionnaire pour stocker tous les pays et leurs valeurs
    all_data = {}
    metrics = []
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Vérifier si les colonnes nécessaires existent
            if 'country' not in df.columns or 'year' not in df.columns:
                print(f"Format incorrect dans {file_path}, colonnes attendues non trouvées.")
                continue
                
            # Identifier la 3ème colonne (métrique)
            if len(df.columns) < 3:
                print(f"Le fichier {file_path} n'a pas assez de colonnes.")
                continue
                
            metric_column = df.columns[2]
            metric_name = extract_metric_name(file_path)
            metrics.append(metric_name)
            
            # Filtrer les données pour l'année demandée
            df_year = df[df['year'] == year]
            
            # Ajouter les données au dictionnaire
            for _, row in df_year.iterrows():
                country = row['country']
                if country not in all_data:
                    all_data[country] = {}
                    
                value = row[metric_column]
                all_data[country][metric_name] = value
                
        except Exception as e:
            print(f"Erreur lors du traitement de {file_path}: {str(e)}")
    
    # Créer un DataFrame à partir des données collectées
    result_df = pd.DataFrame.from_dict(all_data, orient='index')
    result_df.index.name = 'country'
    
    # Réorganiser les colonnes si nécessaire
    if len(metrics) == len(result_df.columns):
        result_df = result_df[metrics]
        
    return result_df

def main():
    # Demander l'année à l'utilisateur
    try:
        year = int(input("Entrez l'année pour laquelle vous voulez extraire les données: "))
    except ValueError:
        print("Année invalide. Utilisation de l'année 2020 par défaut.")
        year = 2020
    
    # Traiter les fichiers CSV et créer le résultat
    result_df = process_csv_files(year)
    
    # Enregistrer le résultat
    output_file = f"resultat_{year}.csv"
    result_df.to_csv(output_file)
    
    print(f"Les données ont été extraites pour {len(result_df)} pays et {len(result_df.columns)} métriques.")
    print(f"Résultat enregistré dans {output_file}")
    
    # Afficher les premières lignes pour vérification
    print("\nAperçu du résultat:")
    print(result_df.head())

if __name__ == "__main__":
    main()