import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

import pyarrow.feather as feather
import matplotlib.pyplot as plt


# ==========================================================
# 1. Cargar dataset agregado por usuario del script original
# ==========================================================

df_users = pd.read_csv(r'AC-07\data\processed\sample_train_data.csv')  
print("Usuarios agrupados:", df_users.shape)


# ==========================================================
# 2. Cargar target_train.feather (NO AGREGADO)
# ==========================================================

target = r'AC-07\data\raw\target_train.feather'
arrow_table = feather.read_table(target, memory_map=True)
slice_table = arrow_table.slice(0, length=100000000)
target = slice_table.to_pandas()
print("Target bruto:", target.shape)
print(target.head())


# ==========================================================
# 3. Agregar target por usuario
# ==========================================================

target_users = target.groupby("user_id").agg({
    "age": "first",
    "is_male": "first"
}).reset_index()

print("Target agregado:", target_users.shape)
print(target_users.head())


# ==========================================================
# 4. Unir features + target
# ==========================================================

df = df_users.merge(target_users, on="user_id", how="inner")

print("Dataset final para entrenamiento:", df.shape)
print(df.head())


# ==========================================================
# 5. Seleccionar features y variables objetivo
# ==========================================================

features = [
    "request_cnt", "total_logs_count", "active_days_count", "price",
    "cpe_manufacturer_name", "cpe_model_name", "cpe_type_cd",
    "cpe_model_os_type", "region_name", "city_name", "part_of_day"
]

categorical = [
    "cpe_manufacturer_name", "cpe_model_name", "cpe_type_cd",
    "cpe_model_os_type", "region_name", "city_name", "part_of_day"
]

numerical = [
    "request_cnt", "total_logs_count", "active_days_count", "price"
]

X = df[features]
y_gender = df["is_male"]
y_age = df["age"]

# Remove rows with missing targets
mask = y_gender.notnull() & y_age.notnull()
X = X[mask].reset_index(drop=True)
y_gender = y_gender[mask].reset_index(drop=True)
y_age = y_age[mask].reset_index(drop=True)

print("\nMissing values per feature before preprocess:\n", X.isnull().sum())

# ==========================================================
# 6. Preprocesamiento (ahora con imputación)
# ==========================================================

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

try:
    cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    cat_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
    ("onehot", cat_ohe)
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numerical),
        ("cat", cat_pipeline, categorical)
    ]
)


# ==========================================================
# 7A. KNN → Clasificación (sexo)
# ==========================================================

# Split the data once and reuse for both targets so comparisons are fair
X_train, X_test, y_train_gender, y_test_gender, y_train_age, y_test_age = train_test_split(
    X, y_gender, y_age, test_size=0.2, random_state=42
)

knn_gender = Pipeline(steps=[
    ("prep", preprocess),
    ("model", KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1))
])

knn_gender.fit(X_train, y_train_gender)

pred_gender = knn_gender.predict(X_test)
acc = accuracy_score(y_test_gender, pred_gender)

print("\n===== KNN GÉNERO =====")
print("Accuracy:", round(acc, 4))


# ==========================================================
# 7B. KNN → Regresión (edad)
# ==========================================================

knn_age = Pipeline(steps=[
    ("prep", preprocess),
    ("model", KNeighborsRegressor(n_neighbors=11, weights="distance", n_jobs=-1))
])

knn_age.fit(X_train, y_train_age)

pred_age = knn_age.predict(X_test)
mae = mean_absolute_error(y_test_age, pred_age)

print("\n===== KNN EDAD =====")
print("MAE:", round(mae, 2))


# ==========================================================
# 8. Evaluar diversos valores de K
# ==========================================================

k_values = range(1,500, 25)
accuracy_scores = []
mae_scores = []

# Optional: check transformed data has no NaNs (for debug)
Xt_train = preprocess.fit_transform(X_train)
if np.isnan(Xt_train).any():
    print("Warning: NaN detected in preprocessed X_train")

for k in k_values:
    # ---------- MODEL GÉNERE (Classificació) ----------
    knn_gender_k = Pipeline(steps=[
        ("prep", preprocess),
        ("model", KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1))
    ])
    
    knn_gender_k.fit(X_train, y_train_gender)
    pred_g = knn_gender_k.predict(X_test)
    acc_k = accuracy_score(y_test_gender, pred_g)
    accuracy_scores.append(acc_k)

    # ---------- MODEL EDAT (Regressió) ----------
    knn_age_k = Pipeline(steps=[
        ("prep", preprocess),
        ("model", KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1))
    ])
    
    knn_age_k.fit(X_train, y_train_age)
    pred_a = knn_age_k.predict(X_test)
    mae_k = mean_absolute_error(y_test_age, pred_a)
    mae_scores.append(mae_k)


# ==========================================================
# 9. Gráficas
# ==========================================================

plt.figure(figsize=(12, 5))

# ---------- Accuracy género ----------
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracy_scores, marker='o')
plt.title("Accuracy vs K (Género)")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.grid(True)

# ---------- MAE edad ----------
plt.subplot(1, 2, 2)
plt.plot(k_values, mae_scores, marker='o', color='orange')
plt.title("MAE vs K (Edad)")
plt.xlabel("K")
plt.ylabel("Mean Absolute Error")
plt.grid(True)

plt.tight_layout()
plt.show()

# Print best K values
best_k_gender = k_values[int(np.argmax(accuracy_scores))]
best_acc = max(accuracy_scores)
best_k_age = k_values[int(np.argmin(mae_scores))]
best_mae = min(mae_scores)

print(f"\nMejor K género: {best_k_gender} (Accuracy={best_acc:.4f})")
print(f"Mejor K edad: {best_k_age} (MAE={best_mae:.4f})")

