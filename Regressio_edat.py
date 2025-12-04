import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
# Models de Regressió
from sklearn.linear_model import Ridge 
# Mètriques de Regressió
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb
import joblib

# CONFIGURACIÓ
DATA_PATH = 'data/procesed/train_data_final.csv'
TARGET_COL = 'age'      
MAX_URL_FEATURES = 2000 

def tokenizer_urls(text):
    """Neteja el format 'url:count' per quedar-se només amb l'URL"""
    if pd.isna(text) or text == "": return []
    return [t.split(':')[0] for t in text.split()]

def entrenar_i_avaluar(model, X_train, y_train, X_test, y_test, model_name):
    print(f"🚀 Entrenant {model_name}...")
    model.fit(X_train, y_train)
    
    # Predicció
    y_pred = model.predict(X_test)
    
    # Mètriques (Regressió)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   🎯 {model_name} Resultats:")
    print(f"      Error Mitjà Absolut (MAE): {mae:.2f} anys") 
    print(f"      R2 Score (Explicabilitat): {r2:.4f}")
    
    # Log a WandB
    if wandb.run is not None:
        wandb.log({
            f"{model_name}_mae": mae,
            f"{model_name}_r2": r2
        })
    
    return model

def main():
    # Mode offline per defecte per evitar errors de xarxa si no tens VPN
    try:
        wandb.init(project="mts-cookies-age-gender", job_type="regression", name="Age-Regression-Ridge-Enhanced", mode="offline")
    except: pass
    
    print("⏳ Carregant dades...")
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=[TARGET_COL])
    except FileNotFoundError:
        print(f"❌ No trobo el fitxer {DATA_PATH}")
        return

    # 1. Definir X i y
    col_text = 'url_counts_list' if 'url_counts_list' in df.columns else 'url_host'
    
    # Eliminem columnes que no volem
    # Nota: Treiem 'part_of_day' perquè ara tenim 'req_morning', 'req_day', etc.
    cols_to_drop = [TARGET_COL, 'user_id', 'is_male', 'part_of_day']
    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Pipeline amb les NOVES Features
    print("🔧 Configurant Pipeline (TF-IDF + Noves Features)...")
    
    # A. TF-IDF per a les URLs
    text_transformer = TfidfVectorizer(
        max_features=MAX_URL_FEATURES, 
        tokenizer=tokenizer_urls, 
        token_pattern=None
    )

    # B. Numèriques: AFEGIM LES NOVES VARIABLES AQUÍ
    # Busquem totes les possibles, però només usem les que existeixin al CSV
    possibles_cols_numeriques = [
        'request_cnt', 'price', 'active_days_count', 
        'activity_span_days', 'daily_intensity',       # <--- Noves temporals
        'req_morning', 'req_day', 'req_evening', 'req_night' # <--- Noves horàries
    ]
    numeric_features = [f for f in possibles_cols_numeriques if f in X.columns]
    print(f"   -> Features numèriques detectades: {numeric_features}")

    # Pipeline numèric amb Imputer per seguretat (per si algun 'span' és null)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())
    ])

    # C. Categòriques
    categorical_features = [f for f in ['cpe_type_cd', 'cpe_manufacturer_name'] if f in X.columns]
    
    # Pipeline categòric
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, col_text),
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 3. MODEL: Ridge Regression (Lineal)
    pipe_ridge = Pipeline([
        ('prep', preprocessor),
        ('regressor', Ridge(alpha=1.0)) 
    ])
    
    entrenar_i_avaluar(pipe_ridge, X_train, y_train, X_test, y_test, "RidgeRegression")

    if wandb.run is not None: wandb.finish()

if __name__ == "__main__":
    main()