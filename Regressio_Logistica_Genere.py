import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# --- CONFIGURACIÓ ---
DATA_PATH = 'data/procesed/train_data_final.csv'
TARGET_COL = 'is_male'
TRAIN_SUBSAMPLE = 0.25  # 25% per anar ràpid en l'anàlisi d'importància

# Tokenizer per netejar el format "url:count"
def tokenizer_urls(text):
    if pd.isna(text) or text == "": return []
    return [t.split(':')[0] for t in text.split()]

def plot_feature_importance(noms_features, valors, titol):
    """Genera el gràfic de pesos"""
    df_imp = pd.DataFrame({'Feature': noms_features, 'Importance': valors})
    
    # Top 20 més importants (en valor absolut)
    df_imp['Abs_Imp'] = df_imp['Importance'].abs()
    df_imp = df_imp.sort_values('Abs_Imp', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    # Fem servir una paleta divergent (Vermell per negatiu/Dona, Blau per positiu/Home)
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='coolwarm', hue='Feature', legend=False)
    
    plt.title(titol)
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel("Pes del Coeficient ( < 0 Dona | > 0 Home )")
    plt.tight_layout()
    
    # Guardar
    os.makedirs('reports', exist_ok=True)
    filename = f"reports/{titol.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"📊 Gràfic guardat a: {filename}")
    plt.show()

def main():
    # WandB (Mode offline per evitar errors de xarxa)
    try:
        wandb.init(project="mts-cookies-analysis", name="Logistic-Gender-Only", mode="offline")
    except: pass
    
    print("⏳ Carregant dades...")
    try:
        df = pd.read_csv(DATA_PATH)
        df = df.dropna(subset=[TARGET_COL])
    except FileNotFoundError:
        print(f"❌ Error: No trobo {DATA_PATH}")
        return

    # --- 1. SELECCIÓ DE FEATURES ---
    print("🔧 Configurant transformacions...")

    # A. Text (URLs) -> TF-IDF
    col_text = 'url_counts_list' if 'url_counts_list' in df.columns else 'url_host'
    
    # B. Numèriques per Discretitzar (Rangs)
    # Això ajuda a la Regressió Logística a entendre patrons no lineals
    features_to_bin = ['price', 'active_days_count'] 
    features_to_bin = [f for f in features_to_bin if f in df.columns]

    # C. Numèriques per Escalar (Standard)
    features_to_scale = ['request_cnt', 'daily_intensity']
    features_to_scale = [f for f in features_to_scale if f in df.columns]
    
    # D. Categòriques (OneHot)
    cat_features = ['cpe_type_cd', 'part_of_day', 'cpe_manufacturer_name']
    cat_features = [f for f in cat_features if f in df.columns]

    # Splits
    X = df.drop(columns=[TARGET_COL, 'user_id', 'age'], errors='ignore')
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SUBSAMPLE, stratify=y, random_state=42)
    print(f"🔬 Entrenant amb {len(X_train)} mostres (Subsampling)...")

    # --- 2. PIPELINE ---
    transformers = [
        ('text', TfidfVectorizer(max_features=1000, tokenizer=tokenizer_urls, token_pattern=None), col_text),
        ('bins', KBinsDiscretizer(n_bins=5, encode='onehot-dense', strategy='quantile'), features_to_bin),
        ('num', StandardScaler(), features_to_scale),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]

    preprocessor = ColumnTransformer(transformers=transformers)

    # Utilitzem Lasso (L1) amb C=0.5 per ser una mica agressius seleccionant features
    pipe = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(solver='liblinear', penalty='l1', C=0.5, max_iter=1000))
    ])

    # --- 3. ENTRENAMENT I MÈTRIQUES ---
    print("🚀 Entrenant...")
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 RESULTATS:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print("\nInforme de Classificació:")
    print(classification_report(y_test, y_pred))

    # --- 4. VISUALITZACIÓ DE PESOS ---
    print("\n🔍 Analitzant quines variables decideixen el gènere...")
    
    # Recuperem noms de les columnes generades
    feature_names = []
    
    # Text
    feature_names.extend(pipe.named_steps['prep'].named_transformers_['text'].get_feature_names_out().tolist())
    # Bins
    if features_to_bin:
        feature_names.extend(pipe.named_steps['prep'].named_transformers_['bins'].get_feature_names_out(features_to_bin).tolist())
    # Num
    feature_names.extend(features_to_scale)
    # Cat
    if cat_features:
        feature_names.extend(pipe.named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_features).tolist())
    
    # Pesos
    coefs = pipe.named_steps['clf'].coef_[0]
    
    if len(feature_names) == len(coefs):
        plot_feature_importance(feature_names, coefs, "Factors_Determinants_Gènere_(Lasso)")
    else:
        print("⚠️ No s'ha pogut generar el gràfic per desajust de dimensions.")

if __name__ == "__main__":
    main()