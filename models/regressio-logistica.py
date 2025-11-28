import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIGURACIÓ
DATA_PATH = r"sample_train_data.csv"
TARGET_COL = 'is_male' 

def preparar_pipeline_dades():
    """
    Defineix com tractem les dades:
    - Numèriques: Estandarització (StandardScaler)
    - Categòriques: OneHotEncoding (crea columnes binàries per cada marca/model)
    """
    # Definim quines columnes són de quin tipus
    # No utilitzem algunes features categoriques ja que tenir 10.000+ categories és inviable per OneHotEncoder, aixo ho solucionarem amb embeddings en futurs models.
    numeric_features = ['request_cnt', 'price', 'active_days', 'total_logs']
    categorical_features = ['cpe_type_cd', 'part_of_day', 'cpe_model_os_type']
    

    # Creem transformadors
    # Afegim imputació per tractar valors faltants abans d'estandarditzar / codificar
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) 
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Unim-ho tot en un pre-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def analitzar_importancia_features(model, preprocessor, feature_names_cat):
    """
    Extreu els pesos del model per veure què és més rellevant.
    """
    # Obtenim els noms de les features numèriques
    num_names = ['request_cnt', 'price', 'active_days', 'total_logs']
    
    # Obtenim els noms de les features categòriques (que ha generat el OneHot)
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(feature_names_cat)
    
    all_names = np.concatenate([num_names, cat_names])
    coefs = model.named_steps['classifier'].coef_[0]
    
    # Creem un DataFrame per ordenar-ho
    feature_importance = pd.DataFrame({'Feature': all_names, 'Weight': coefs})
    feature_importance['Abs_Weight'] = feature_importance['Weight'].abs()
    feature_importance = feature_importance.sort_values(by='Abs_Weight', ascending=False).head(15) # Top 15
    
    # Gràfic
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Weight', y='Feature', data=feature_importance, palette='viridis')
    plt.title('Top 15 Features que decideixen el Gènere (Reg. Logística)')
    plt.xlabel('Pes (Positiu = Més prob. Home | Negatiu = Més prob. Dona)')
    plt.tight_layout()
    plt.savefig("logistic_feature_importance.png")
    
    return "logistic_feature_importance.png"

def entrenar_logistic():
    # 2. Carregar Dades
    try:
        print("⏳ Carregant CSV...")
        df = pd.read_csv(DATA_PATH)
        
        # Normalitzar noms de columnes (el CSV d'exemple té sufix _count)
        rename_map = {
            'total_logs_count': 'total_logs',
            'active_days_count': 'active_days',
            'total_logs_count': 'total_logs',
            'active_days_count': 'active_days',
            'total_logs_count': 'total_logs'
        }
        df = df.rename(columns=rename_map)

        # Neteja ràpida de soroll
        if 'request_cnt' in df.columns:
            df = df[df['request_cnt'] > 5]

        # Si no tenim la columna target, intentem carregar i unir el fitxer de target si existeix
        if TARGET_COL not in df.columns:
            possible_target = Path('data/raw/target_train.feather')
            if possible_target.exists():
                try:
                    target_df = pd.read_feather(possible_target)
                    if TARGET_COL in target_df.columns:
                        df = df.merge(target_df[['user_id', TARGET_COL]], on='user_id', how='left')
                    else:
                        print(f"❌ El fitxer de target existeix però no conté la columna '{TARGET_COL}'.")
                        return
                except Exception as e:
                    print(f"❌ Error llegint el fitxer de target: {e}")
                    return
            else:
                print(f"❌ La columna target '{TARGET_COL}' no està present al CSV i no s'ha trobat 'data/raw/target_train.feather'. Executa primer el preprocesat per crear el target.")
                return

        # Finalment, eliminem files sense target
        df = df.dropna(subset=[TARGET_COL])
        
    except FileNotFoundError:
        print("❌ No s'ha trobat el fitxer. Executa primer 'data_processing.py'")
        return

    # 3. Split X/y
    X = df.drop(columns=[TARGET_COL, 'user_id', 'age', 'region_name', 'city_name', 'cpe_model_name', 'cpe_manufacturer_name'], errors='ignore')
    # Nota: Treiem 'model_name' i 'city' perquè tenen massa valors únics per OneHotEncoder ara mateix.
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Dades llestes. Train shape: {X_train.shape}")

    # 4. Pipeline (Preprocessament + Model)
    # Utilitzar Pipelines és "Best Practice" total en Sklearn
    clf = Pipeline(steps=[
        ('preprocessor', preparar_pipeline_dades()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # 5. Entrenament
    print(f"🚀 Entrenant Regressió Logística...")
    clf.fit(X_train, y_train)
    
    # 6. Predicció
    print("🔮 Predint...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] # Probabilitat de ser '1' (Home)
    
    # 7. Mètriques
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n🎯 Resultats Regressió Logística:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   ROC AUC:  {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 8. Matriu de Confusió
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriu de Confusió (Acc: {acc:.2f})')
    plt.savefig("logistic_conf_matrix.png")
    
    # 9. Feature Importance (El punt fort d'aquest model)
    # Passem els noms de les columnes categòriques que hem usat
    feat_plot_path = analitzar_importancia_features(clf, clf.named_steps['preprocessor'], 
                                                    ['cpe_type_cd', 'part_of_day', 'cpe_model_os_type'])

if __name__ == "__main__":
    entrenar_logistic()