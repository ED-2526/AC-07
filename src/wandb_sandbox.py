import wandb

# A l'inici de la funció main o de configuració
def iniciar_wandb():
    # Inicia un run. Canvia 'grup-XX' pel vostre nom d'equip a WandB
    wandb.init(project="mts-cookies-age-gender", entity="grup-XX", job_type="data-processing")

def loguejar_estadistiques(df):
    """Envia estadístiques bàsiques a WandB."""
    print("📈 Enviant estadístiques a WandB...")
    
    stats = {
        "total_users": len(df),
        "avg_price": df['price'].mean(),
        "missing_values": df.isnull().sum().sum()
    }
    
    # Si tenim labels, loguegem la distribució
    if 'is_male' in df.columns:
        stats["gender_distribution"] = df['is_male'].value_counts().to_dict()
        
    wandb.log(stats)
    print("✅ Estadístiques pujades!")

# --- Dins del bloc __main__ ---
if __name__ == "__main__":
    iniciar_wandb()  # <--- AFEGIR
    
    # ... (tot el teu codi de càrrega i neteja) ...
    
    if df_final is not None:
        loguejar_estadistiques(df_final) # <--- AFEGIR
        
    wandb.finish() # <--- Tancar la sessió