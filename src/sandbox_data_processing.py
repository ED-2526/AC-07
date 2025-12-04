import pandas as pd
import numpy as np
import pyarrow.feather as feather
import gc
import os

# --- CONFIGURACIÓ ---
# Assegura't que tens els dos fitxers a aquesta ruta!
LOGS_PATH = 'data/raw/dataset_full.feather'
TARGETS_PATH = 'data/raw/target_train.feather' 
CHUNK_SIZE = 10000000  # 10M per chunk

# --- 1. FUNCIONS DE NETEJA (Logs) ---
def netejar_dades(df):
    """Normalitza noms i categories."""
    if 'cpe_model_os_type' in df.columns:
        df['cpe_model_os_type'] = df['cpe_model_os_type'].replace('Apple iOS', 'iOS')
    return df

def tractar_nuls_i_preus(df):
    """Omple preus basant-se en el model del dispositiu."""
    if 'price' in df.columns:
        # Si tenim model, imputem per model
        if 'cpe_model_name' in df.columns:
            df['price'] = df['price'].fillna(
                df.groupby('cpe_model_name')['price'].transform('mean')
            )
        # Si encara queden nuls, mitjana del chunk
        df['price'] = df['price'].fillna(df['price'].mean())
    return df

def agregar_avancat(df):
    print(f"\n🔄 Agregació AVANÇADA de {len(df)} logs...")

    # --- 1. PIVOT PART OF DAY (La clau per no perdre info) ---
    # Volem comptar quants requests hi ha a cada franja per usuari
    # Això crea columnes: user_id | morning | day | evening | night
    print("   -> Desglossant per part del dia...")
    
    # Agrupem per usuari i part del dia i sumem requests
    day_counts = df.groupby(['user_id', 'part_of_day'])['request_cnt'].sum().unstack(fill_value=0)
    
    # Renombrem les columnes per que quedi net (ex: req_morning, req_night)
    day_counts.columns = [f'req_{col}' for col in day_counts.columns]
    
    # --- 2. INFORMACIÓ TEMPORAL (El que deia el profe) ---
    print("   -> Calculant patrons temporals...")
    time_stats = df.groupby('user_id')['date'].agg([
        ('active_days', 'nunique'), # El que teníem abans
        ('min_date', 'min'),        # Primera connexió
        ('max_date', 'max')         # Última connexió
    ])
    
    # Calculem el "Life Span" (dies entre primera i última connexió)
    time_stats['activity_span_days'] = (time_stats['max_date'] - time_stats['min_date']).dt.days
    
    # Feature nova: Intensitat (Requests per dia actiu)
    # Necessitem el total de requests per fer això, ho farem al final del merge
    
    # Eliminem les dates brutes per estalviar espai, ja tenim el span
    time_stats = time_stats.drop(columns=['min_date', 'max_date'])

    # --- 3. URLs (Optimitzat com abans) ---
    url_data = None
    if 'url_host' in df.columns:
        print("   -> Processant URLs...")
        url_counts = df.groupby(['user_id', 'url_host'], sort=False, observed=True).size().reset_index(name='count')
        url_counts['url_feature'] = url_counts['url_host'].astype(str) + ':' + url_counts['count'].astype(str)
        url_data = url_counts.groupby('user_id', sort=False)['url_feature'].apply(' '.join)
        del url_counts

    # --- 4. ESTÀTICS (Dispositiu i Preu) ---
    print("   -> Agregant perfil estàtic...")
    static_aggs = {
        'request_cnt': 'sum', # Total global
        'price': 'mean',
        'cpe_model_os_type': 'first',
        'cpe_manufacturer_name': 'first',
        'region_name': 'first',
        'city_name': 'first'
    }
    agg_rules = {k: v for k, v in static_aggs.items() if k in df.columns}
    df_static = df.groupby('user_id', sort=False).agg(agg_rules)

    # --- 5. UNIFICACIÓ FINAL ---
    print("   -> Unint totes les peces...")
    
    # Comencem amb l'estàtic
    df_final = df_static.join(day_counts, how='left') # Afegim columnes matí/tarda/nit
    df_final = df_final.join(time_stats, how='left')  # Afegim dies actius i span
    
    if url_data is not None:
        df_final = df_final.join(url_data)

    # Càlcul final de la intensitat (Requests totals / Dies actius)
    df_final['daily_intensity'] = df_final['request_cnt'] / df_final['active_days']

    df_final = df_final.reset_index()
    df_final.rename(columns={'url_feature': 'url_counts_list'}, inplace=True)
    
    return df_final

# --- 3. PROCESSAMENT MASSIU (LOGS) ---
def processar_logs_massius(path, chunk_size):
    print(f"🚀 Iniciant processament de logs: {path}")
    
    resultats_parcials = []
    reader = feather.read_table(path, memory_map=True)
    total_rows = reader.num_rows
    
    print(f"📊 Total files logs: {total_rows}")
    
    for i in range(0, total_rows, chunk_size):
        print(f"   📦 Processant chunk {i//chunk_size + 1}...")
        
        # Load
        df_chunk = reader.slice(i, length=min(chunk_size, total_rows - i)).to_pandas()
        
        # Clean
        df_chunk = netejar_dades(df_chunk)
        df_chunk = tractar_nuls_i_preus(df_chunk)
        
        # Aggregate
        df_agregat = agregar_avancat(df_chunk)
        resultats_parcials.append(df_agregat)
        
        # Cleanup RAM
        del df_chunk
        gc.collect()
        
    # Unir tots els chunks de logs
    print("🧩 Unint resultats parcials dels logs...")
    df_logs_total = pd.concat(resultats_parcials)
    
    # Re-agrupació final (per si un usuari estava partit en dos chunks)
    print("🏁 Re-agregant final per user_id...")
    
    regles_finals = {
        'request_cnt': 'sum',
        'date': 'sum', # Suma de dies actius (aprox)
        'price': 'mean',
        'url_feature': lambda x: ' '.join(x.dropna().astype(str)),
        # Per la resta agafem el primer
        'cpe_model_os_type': 'first',
        'region_name': 'first',
        'city_name': 'first',
        'part_of_day': 'first'
    }
    
    # Filtrem regles actives
    regles_actives = {k: v for k, v in regles_finals.items() if k in df_logs_total.columns}
    # Si falten columnes categòriques, afegim 'first'
    for col in df_logs_total.columns:
        if col not in regles_actives and col != 'user_id':
            regles_actives[col] = 'first'

    df_final_logs = df_logs_total.groupby('user_id').agg(regles_actives).reset_index()
    
    # Rename
    rename_map = {'date': 'active_days_count', 'url_feature': 'url_counts_list'}
    df_final_logs.rename(columns=rename_map, inplace=True)
    
    return df_final_logs

# --- 4. CARREGAR TARGETS I FER MERGE ---
def afegir_targets(df_logs, targets_path):
    print(f"🎯 Carregant targets des de: {targets_path}")
    
    if not os.path.exists(targets_path):
        print("❌ ALERTA: No s'ha trobat el fitxer de targets. Es guardarà sense etiquetes.")
        return df_logs

    # Carreguem targets (és petit, cap a la RAM)
    try:
        df_targets = pd.read_feather(targets_path)
    except:
        # Si no és feather, prova csv o parquet
        try:
            df_targets = pd.read_csv(targets_path)
        except:
            print("❌ Format de targets no reconegut.")
            return df_logs

    print(f"   -> Targets carregats: {len(df_targets)} usuaris.")
    
    # FEM EL MERGE
    print("🔗 Fusionant Logs + Targets (Inner Join)...")
    
    # 'inner': Només ens quedem usuaris que tinguin logs I TAMBÉ tinguin target.
    # Si vols predir usuaris de test (sense target), hauries de fer 'left' o carregar un altre fitxer.
    df_final = df_logs.merge(df_targets, on='user_id', how='inner')
    
    return df_final

# --- MAIN ---
if __name__ == "__main__":
    # 1. Processar Logs (La part pesada)
    df_usuaris = processar_logs_massius(LOGS_PATH, CHUNK_SIZE)
    print(f"✅ Logs processats. Usuaris únics trobats: {len(df_usuaris)}")
    
    # 2. Afegir Targets (Edat i Gènere)
    df_complet = afegir_targets(df_usuaris, TARGETS_PATH)
    
    # 3. Neteja final de seguretat (Nuls)
    print("🛡️ Neteja final de nuls post-merge...")
    if 'price' in df_complet.columns:
        df_complet['price'] = df_complet['price'].fillna(df_complet['price'].mean())
        
    # 4. Resultats i Guardat
    print("\n--- DATASET FINAL (TRAIN) ---")
    print(df_complet.info())
    print(df_complet.head())
    
    if 'age' in df_complet.columns:
        print("\nExemple Target:")
        print(df_complet[['user_id', 'age', 'is_male']].head())
    
    output_file = 'data/procesed/train_data_final.csv'
    print(f"💾 Guardant a {output_file}...")
    df_complet.to_csv(output_file, index=False)
    print("✅ Procés acabat amb èxit!")