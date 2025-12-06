import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


RUTA_DATA = 'd://Data/'
RUTA_GRAFICOS = 'd://Data/Graf1/'


df = pd.read_csv(RUTA_DATA + 'dataset_sin_saldos.csv')
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Features numericas directas
features_numericas = [
    'DIA_PRODUCCION_NUM',
    'CANTIDAD_ACUMULADA',
    'DIAS_DESDE_INICIO',
    'DIAS_DESDE_ULTIMA',
    'MINUTOS_ESTILO',
    'CANTIDAD_PROMEDIO_HISTORICA'
]

# Features categoricas
features_categoricas = [
    'LINEA',
    'TIPO_PRENDA',
    'FASE_PRODUCCION',
    'DIA_SEMANA'
]

# Variable objetivo
target = 'CANTIDAD_NORMALIZADA'

# Verificar que todas las columnas existen
columnas_faltantes = []
for col in features_numericas + features_categoricas + [target]:
    if col not in df.columns:
        columnas_faltantes.append(col)


df_ml = df[features_numericas + features_categoricas + [target]].copy()


df_ml = df_ml.dropna()



label_encoders = {}
for col in features_categoricas:
    le = LabelEncoder()
    df_ml[col + '_encoded'] = le.fit_transform(df_ml[col])
    label_encoders[col] = le


features_finales = features_numericas + [col + '_encoded' for col in features_categoricas]


print(f"\n📊 ESTADÍSTICAS DE LA VARIABLE OBJETIVO ({target}):")
print(f"   Media: {df_ml[target].mean():.2f}")
print(f"   Mediana: {df_ml[target].median():.2f}")
print(f"   Desv. Estándar: {df_ml[target].std():.2f}")
print(f"   Mínimo: {df_ml[target].min():.2f}")
print(f"   Máximo: {df_ml[target].max():.2f}")


print("DATOS EN ENTRENAMIENTO Y PRUEBA 80 / 20 ")
print("-"*80)


X = df_ml[features_finales]
y = df_ml[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print(f"   • Entrenamiento: {len(X_train):,} registros ({len(X_train)/len(df_ml)*100:.1f}%)")
print(f"   • Prueba: {len(X_test):,} registros ({len(X_test)/len(df_ml)*100:.1f}%)")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



print("\n" + "-"*80)
print("ENTRENAMIENTO DE 7 MODELOS")
print("-"*80)
print("""
1. Regresión Lineal: Baseline simple, relación lineal
2. Ridge Regression: Regresión con regularización L2 (evita overfitting)
3. Lasso Regression: Regresión con regularización L1 (selección de features)
4. Random Forest: Ensemble de árboles, captura no-linealidades
5. Gradient Boosting: Boosting secuencial, muy preciso
6. XGBoost: Versión optimizada de Gradient Boosting
7. Red Neuronal (MLP): Captura patrones complejos no-lineales
""")

modelos = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    ),
    'Neural Network': MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}


modelos_necesitan_escalar = ['Linear Regression', 'Ridge Regression', 
                              'Lasso Regression', 'Neural Network']


print("\n" + "-"*80)
print("ENTRENAMIENTO Y EVALUACIÓN")
print("-"*80)

resultados = []

for nombre, modelo in modelos.items():
    print(f"\n{'='*60}")
    print(f"Entrenando: {nombre}")
    print(f"{'='*60}")
    
    # Seleccionar datos apropiados (escalados o no)
    if nombre in modelos_necesitan_escalar:
        X_train_use = X_train_scaled
        X_test_use = X_test_scaled
        print("   ✓ Usando datos escalados")
    else:
        X_train_use = X_train
        X_test_use = X_test
        print("   ✓ Usando datos sin escalar")
    
    # Entrenar
    print("   Entrenando modelo...")
    modelo.fit(X_train_use, y_train)
    
    # Predicciones
    y_pred_train = modelo.predict(X_train_use)
    y_pred_test = modelo.predict(X_test_use)
    
    # Métricas en conjunto de entrenamiento
    r2_train = r2_score(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    
    # Métricas en conjunto de prueba
    r2_test = r2_score(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    # Cross-validation (5-fold)
    print("   Ejecutando validación cruzada (5-fold)...")
    cv_scores = cross_val_score(
        modelo, X_train_use, y_train, 
        cv=5, scoring='r2', n_jobs=-1
    )
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"      TRAIN - R²: {r2_train:.4f} | MAE: {mae_train:.2f} | RMSE: {rmse_train:.2f} | MAPE: {mape_train:.2f}%")
    print(f"      TEST  - R²: {r2_test:.4f} | MAE: {mae_test:.2f} | RMSE: {rmse_test:.2f} | MAPE: {mape_test:.2f}%")
    print(f"      CV (5-fold) - R² medio: {cv_mean:.4f} ± {cv_std:.4f}")
    
    # Detectar overfitting
    diff_r2 = r2_train - r2_test
    if diff_r2 > 0.1:
        print(f"      ⚠️ POSIBLE OVERFITTING (diferencia R²: {diff_r2:.4f})")
    elif diff_r2 < -0.05:
        print(f"      ⚠️ POSIBLE UNDERFITTING (diferencia R²: {diff_r2:.4f})")
    else:
        print(f"      ✓ Buen balance train/test")
    
    # Guardar resultados
    resultados.append({
        'Modelo': nombre,
        'R2_Train': r2_train,
        'R2_Test': r2_test,
        'MAE_Train': mae_train,
        'MAE_Test': mae_test,
        'RMSE_Train': rmse_train,
        'RMSE_Test': rmse_test,
        'MAPE_Train': mape_train,
        'MAPE_Test': mape_test,
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std,
        'Diff_R2': diff_r2
    })

# Convertir a DataFrame
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('R2_Test', ascending=False)



print("\n📊 Ordenado por R² en Test (mejor desempeño en datos no vistos)")
print("\n" + df_resultados.to_string(index=False))


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gráfico 1: R² Train vs Test
x_pos = np.arange(len(df_resultados))
width = 0.35

axes[0, 0].bar(x_pos - width/2, df_resultados['R2_Train'], width, 
               label='Train', alpha=0.8, color='skyblue')
axes[0, 0].bar(x_pos + width/2, df_resultados['R2_Test'], width, 
               label='Test', alpha=0.8, color='coral')
axes[0, 0].set_xlabel('Modelo')
axes[0, 0].set_ylabel('R² Score')
axes[0, 0].set_title('Comparación R² - Train vs Test', fontweight='bold')
axes[0, 0].set_xticks(x_pos)
axes[0, 0].set_xticklabels(df_resultados['Modelo'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')

# Gráfico 2: MAE Test
axes[0, 1].barh(df_resultados['Modelo'], df_resultados['MAE_Test'], color='green', alpha=0.7)
axes[0, 1].set_xlabel('MAE (Mean Absolute Error)')
axes[0, 1].set_title('Error Absoluto Medio en Test\n(menor es mejor)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df_resultados['MAE_Test']):
    axes[0, 1].text(v, i, f' {v:.2f}', va='center')

# Gráfico 3: MAPE Test
axes[1, 0].barh(df_resultados['Modelo'], df_resultados['MAPE_Test'], color='purple', alpha=0.7)
axes[1, 0].set_xlabel('MAPE (%)')
axes[1, 0].set_title('Error Porcentual Absoluto Medio en Test\n(menor es mejor)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df_resultados['MAPE_Test']):
    axes[1, 0].text(v, i, f' {v:.1f}%', va='center')

# Gráfico 4: Cross-Validation R²
axes[1, 1].barh(df_resultados['Modelo'], df_resultados['CV_R2_Mean'], 
                xerr=df_resultados['CV_R2_Std'], color='orange', alpha=0.7)
axes[1, 1].set_xlabel('R² (Cross-Validation)')
axes[1, 1].set_title('R² con Validación Cruzada (5-fold)\ncon desviación estándar', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df_resultados['CV_R2_Mean']):
    axes[1, 1].text(v, i, f' {v:.3f}', va='center')

plt.tight_layout()
plt.savefig(RUTA_GRAFICOS + '06_comparacion_modelos.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {RUTA_GRAFICOS}06_comparacion_modelos.png")
plt.show()



mejor_modelo_nombre = df_resultados.iloc[0]['Modelo']
mejor_modelo = modelos[mejor_modelo_nombre]

print(f"\n🏆 MEJOR MODELO: {mejor_modelo_nombre}")
print(f"   R² Test: {df_resultados.iloc[0]['R2_Test']:.4f}")
print(f"   MAE Test: {df_resultados.iloc[0]['MAE_Test']:.2f}")
print(f"   MAPE Test: {df_resultados.iloc[0]['MAPE_Test']:.2f}%")


if mejor_modelo_nombre in modelos_necesitan_escalar:
    y_pred_mejor = mejor_modelo.predict(X_test_scaled)
else:
    y_pred_mejor = mejor_modelo.predict(X_test)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))



axes[0].scatter(y_test, y_pred_mejor, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel('Valores Reales')
axes[0].set_ylabel('Valores Predichos')
axes[0].set_title(f'Predicciones vs Reales - {mejor_modelo_nombre}\nR² = {df_resultados.iloc[0]["R2_Test"]:.4f}', 
                 fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


errores = y_test - y_pred_mejor
axes[1].hist(errores, bins=50, edgecolor='black', alpha=0.7, color='coral')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
axes[1].axvline(errores.mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Media: {errores.mean():.2f}')
axes[1].set_xlabel('Error (Real - Predicho)')
axes[1].set_ylabel('Frecuencia')
axes[1].set_title(f'Distribución de Errores - {mejor_modelo_nombre}', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RUTA_GRAFICOS + '07_mejor_modelo_predicciones.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {RUTA_GRAFICOS}07_mejor_modelo_predicciones.png")
plt.show()
