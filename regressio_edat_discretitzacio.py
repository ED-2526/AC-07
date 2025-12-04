import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import wandb
import os

# CONFIGURACIÓ
DATA_PATH = 'data/procesed/train_data_final.csv'
MAX_URL_FEATURES = 2000 

# Rangs d'Edat (Target)
AGE_BINS = [0, 18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ['<18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']

def tokenizer_urls(text):
    if pd.isna(text) or text == "": return []
    return [t.split(':')[0] for t in text.split()]

def preparar_target(df):
    print("🎂 Discretitzant edat (Target)...")
    df = df.dropna(subset=['age'])
    df['age_group'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS)
    df = df.dropna(subset=['age_group'])
    return df

def entrenar_logistica(X_train, y_train, X_test, y_test, penalitzacio, nom_model, preprocessor):
    """
    Entrena una Regressió Logística amb la penalització especificada.
    """
    print(f"\n🚀 Entrenant {nom_model} (Penalització {penalitzacio})...")
    
    # Pipeline del Model
    # Solver 'saga' sol ser més ràpid per datasets grans i suporta l1/l2/elasticnet
    clf = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(penalty=penalitzacio, solver='liblinear', C=1.0, max_iter=1000))
    ])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   🎯 {nom_model} Accuracy: {acc:.4f}")
    
    # Report simplificat per no omplir la pantalla
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Log a WandB
    if wandb.run is not None:
        wandb.log({f"{nom_model}_accuracy": acc})
    
    return acc

def main():
    try:
        wandb.init(project="mts-cookies-age-classification", job_type="comparison", name="L1-vs-L2-Enhanced", mode="offline")
    except: pass
    
    print("⏳ Carregant dades...")
    try:
        df = pd.read_csv(DATA_PATH)
        df = preparar_target(df)
    except FileNotFoundError:
        print(f"❌ Error: No trobo {DATA_PATH}")
        return

    # --- DEFINICIÓ DE FEATURES ---
    col_text = 'url_counts_list' if 'url_counts_list' in df.columns else 'url_host'
    
    # 1. Numèriques (totes les noves)
    num_cols = [
        'request_cnt', 'price', 'active_days_count', 
        'activity_span_days', 'daily_intensity',
        'req_morning', 'req_day', 'req_evening', 'req_night'
    ]
    # Filtrem les que existeixen
    num_features = [f for f in num_cols if f in df.columns]
    
    # 2. Categòriques
    cat_features = [f for f in ['cpe_type_cd', 'cpe_manufacturer_name'] if f in df.columns]

    # Splits
    X = df.drop(columns=['age', 'age_group', 'user_id', 'is_male', 'part_of_day'], errors='ignore')
    y = df['age_group']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- PREPROCESSADOR COMÚ ---
    # Per a la logística, discretitzar les numèriques (binning) sol anar millor que escalar
    # perquè captura no-linealitat.
    
    print("🔧 Configurant Pipeline amb Discretització de Features...")
    preprocessor = ColumnTransformer(transformers=[
        # Text
        ('text', TfidfVectorizer(max_features=MAX_URL_FEATURES, tokenizer=tokenizer_urls, token_pattern=None), col_text),
        
        # Numèriques -> Bins (Important per Logística!)
        ('num_bins', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('bins', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'))
        ]), num_features),
        
        # Categòriques -> OneHot
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

    # --- EXPERIMENTS ---
    
    # 1. L2 (Ridge)
    acc_l2 = entrenar_logistica(X_train, y_train, X_test, y_test, 'l2', "LogReg_L2_Ridge", preprocessor)

    # 2. L1 (Lasso)
    acc_l1 = entrenar_logistica(X_train, y_train, X_test, y_test, 'l1', "LogReg_L1_Lasso", preprocessor)

    print("\n🏆 CONCLUSIÓ FINAL:")
    if acc_l1 > acc_l2:
        print(f"   L1 (Lasso) és millor ({acc_l1:.4f} vs {acc_l2:.4f}). Selecció de features funciona.")
    else:
        print(f"   L2 (Ridge) és millor ({acc_l2:.4f} vs {acc_l1:.4f}). La informació està distribuïda.")

    if wandb.run is not None: wandb.finish()

if __name__ == "__main__":
    main()