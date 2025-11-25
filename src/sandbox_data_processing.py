import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather as feather

# CONFIGURACIÓ
FILE_PATH = r'AC-07\data\raw\dataset_full.feather' 
SAMPLE_SIZE = 100000000  # Número de files a carregar en forma de mostra

def carregar_mostra(path, n_rows):
    """Carrega una mostra de les dades en pandas."""
    print(f" Carregant les primeres {n_rows} files...")
    try:
        # Carreguem amb llibreria pyarrow
        arrow_table = feather.read_table(path, memory_map=True)
            
        # Tallem (slice) les files QUE VOLEM abans de convertir a Pandas
        # Això evita carregar els 322M de files
        slice_table = arrow_table.slice(0, length=n_rows)
        df = slice_table.to_pandas()
        return df
    except Exception as e:
        print(f"Error carregant: {e}")
        return None

def netejar_dades(df):
    """ Normalitza valors i neteja dades abans d'agregar."""
    print("Netejant i normalitzant dades...")
    
    #Normalitzar Sistema Operatiu (Apple iOS -> iOS)
    # Utilitzem replace per unificar
    df['cpe_model_os_type'] = df['cpe_model_os_type'].replace('Apple iOS', 'iOS')
    return df

def tractar_nuls_i_preus(df):
    """ Omple els nuls del preu basant-se en el model del dispositiu."""
    print("Tractant valors nuls en 'price'...")
    
    nuls_abans = df['price'].isnull().sum()
    
    # 1. Calculem el preu mitjà per cada model de dispositiu
    # Transform permet omplir els nuls mantenint la mida original del DF
    df['price'] = df['price'].fillna(
        df.groupby('cpe_model_name')['price'].transform('mean')
    )
    
    # 2. Si encara queden nuls (models que mai tenen preu), usem la mitjana global
    mitjana_global = df['price'].mean()
    df['price'] = df['price'].fillna(mitjana_global)
    
    nuls_ara = df['price'].isnull().sum()
    print(f"Nuls a 'price' corregits: {nuls_abans} -> {nuls_ara}")
    
    return df

def analisi_exploratori(df):
    """Mostra estadístiques bàsiques i distribucions."""
    print("\n--- INFO DEL DATASET ---")
    print(df.info())
    
    print("\n--- VALORS NULS ---")
    print(df.isnull().sum())
    """
    # 1. Distribució de Dispositius
    if 'cpe_type_cd' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, y='cpe_type_cd', order=df['cpe_type_cd'].value_counts().index)
        plt.title("Distribució de Tipus de Dispositius")
        plt.show()

    # 2. Preu del dispositiu
    if 'price' in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df['price'], bins=30, kde=True)
        plt.title("Distribució de Preus dels Dispositius")
        plt.show()
    """
    if 'cpe_model_os_type' in df.columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, y='cpe_model_os_type', order=df['cpe_model_os_type'].value_counts().index)
        plt.title("Distribució de Sistemes Operatius (Normalitzat)")
        plt.show()

def agregar_per_usuari(df):
    """ Transforma el dataset de LOGS (1 fila per visita) a un dataset d'USUARIS (1 fila per user_id)."""
    print("\n Agregant dades per usuari...")
    
    # Definim com agreguem cada columna
    agregacions = {
        # --- Mètriques d'activitat ---
        'request_cnt': 'sum',       # Total de requests
        'url_host': 'count',        # Nombre de llocs visitats (rows)
        'date': 'nunique',          # Dies diferents que s'ha connectat (Active Days)
        
        # --- Dades del dispositiu ---
        'price': 'mean',            # Preu mitjà del dispositiu
        'cpe_manufacturer_name': 'first',
        'cpe_model_name': 'first',
        'cpe_type_cd': 'first',
        'cpe_model_os_type': 'first', # Ja està netejat
        
        # --- Geolocalització ---
        'region_name': 'first',
        'city_name': 'first',
        
        # --- Comportament ---
        'part_of_day': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0] # La part del dia més freqüent
    }

    # Fem el GroupBy
    # Només utilitzem columnes que existeixin al df
    agg_rules = {k: v for k, v in agregacions.items() if k in df.columns}
    
    df_users = df.groupby('user_id').agg(agg_rules).reset_index()
    
    # Renombrem la columna 'date' a 'active_days' per claredat
    if 'date' in df_users.columns:
        df_users.rename(columns={'date': 'active_days_count'}, inplace=True)
        
    # Renombrem url_host a total_hits
    if 'url_host' in df_users.columns:
        df_users.rename(columns={'url_host': 'total_logs_count'}, inplace=True)

    print(f"Agregació completada. Tenim {len(df_users)} usuaris únics a la mostra.")
    return df_users

# --- EXECUTAR ---
def main():
    # 1. Carregar
    df = carregar_mostra(FILE_PATH, SAMPLE_SIZE)
    print(df.head())
    print(df['price'].unique())
    print(df['cpe_model_os_type'].unique())
    print(df['part_of_day'].unique())
    print(df['cpe_type_cd'].unique())
    print(len(df['url_host'].unique()))  
    if df is not None:
        # 1. Netejar dades
        df =netejar_dades(df)
        # 2. EDA sobre els logs
        analisi_exploratori(df)
        
        # 3. Crear dataset d'entrenament (agregat)
        df_final = agregar_per_usuari(df)
        
        # 4. Veure resultat
        print("\n--- HEAD DEL DATASET D'USUARIS ---")
        print(df_final.head())
        print(df_final.info())
        
        # Opcional: Guardar la mostra neta
        df_final.to_csv('sample_train_data.csv', index=False)

main()