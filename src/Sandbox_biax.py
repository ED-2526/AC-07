import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather as feather

# CONFIGURACIÓ
FILE_PATH = r'data\raw\target_train.feather' 
SAMPLE_SIZE = 25000000  # Número de files a carregar en forma de mostra

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

def analitzar_biaix(df):
    """
    Analitza i guarda a WandB la distribució de les etiquetes.
    """
    print("📊 Analitzant biaix de les dades...")
    stats = {}
    
    # 1. Biaix de Gènere
    if 'is_male' in df.columns:
        dist_genere = df['is_male'].value_counts(normalize=True).to_dict()
        print(f"   Distribució Gènere: {dist_genere}")
        stats['bias_gender_male_ratio'] = dist_genere.get(1.0, 0) # Ratio d'homes
        
        # Gràfic per a l'informe
        plt.figure(figsize=(6,4))
        sns.countplot(x='is_male', data=df)
        plt.title("Distribució de Gènere (desequilibrada?)")
        plt.savefig("gender_dist.png")
        
    # 2. Biaix d'Edat
    if 'age' in df.columns:
        print(f"   Edat mitjana: {df['age'].mean():.2f}")
        stats['avg_age'] = df['age'].mean()
        
        plt.figure(figsize=(10,5))
        sns.histplot(df['age'], bins=20, kde=True)
        plt.title("Distribució d'Edat")
        plt.savefig("age_dist.png")


if __name__ == "__main__":
    df = carregar_mostra(FILE_PATH, SAMPLE_SIZE)
    
    if df is not None:
        analitzar_biaix(df)