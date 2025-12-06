import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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


print("MODELOS")

def modelo_potencial(x, a, b):
    """Modelo de Wright: Y = a * X^b"""
    return a * np.power(x, b)

def modelo_logaritmico(x, a, b):
    """Modelo logarítmico: Y = a + b * log(X)"""
    return a + b * np.log(x)

def modelo_lineal(x, a, b):
    """Modelo lineal: Y = a + b * X"""
    return a + b * x

def ajustar_modelos(x, y):
    resultados = {}
    
    # Filtrar valores válidos
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 5:
        return None
    
    try:
        # Modelo Potencial
        params_pot, _ = curve_fit(modelo_potencial, x_clean, y_clean, 
                                   p0=[y_clean.mean(), 0.1], maxfev=5000)
        y_pred_pot = modelo_potencial(x_clean, *params_pot)
        resultados['potencial'] = {
            'params': params_pot,
            'r2': r2_score(y_clean, y_pred_pot),
            'mae': mean_absolute_error(y_clean, y_pred_pot),
            'tasa_aprendizaje': params_pot[1]
        }
    except:
        resultados['potencial'] = None
    
    try:
        # Modelo Logarítmico
        params_log, _ = curve_fit(modelo_logaritmico, x_clean, y_clean,
                                   p0=[y_clean.mean(), 10], maxfev=5000)
        y_pred_log = modelo_logaritmico(x_clean, *params_log)
        resultados['logaritmico'] = {
            'params': params_log,
            'r2': r2_score(y_clean, y_pred_log),
            'mae': mean_absolute_error(y_clean, y_pred_log),
            'pendiente': params_log[1]
        }
    except:
        resultados['logaritmico'] = None
    
    try:
        # Modelo Lineal
        model_lin = LinearRegression()
        model_lin.fit(x_clean.reshape(-1, 1), y_clean)
        y_pred_lin = model_lin.predict(x_clean.reshape(-1, 1))
        resultados['lineal'] = {
            'params': [model_lin.intercept_, model_lin.coef_[0]],
            'r2': r2_score(y_clean, y_pred_lin),
            'mae': mean_absolute_error(y_clean, y_pred_lin),
            'pendiente': model_lin.coef_[0]
        }
    except:
        resultados['lineal'] = None
    
    return resultados


print("\n" + "="*80)
print(" CURVAS DE APRENDIZAJE POR TIPO DE PRENDA")
print("="*80)

curvas_tipo = df.groupby(['TIPO_PRENDA', 'DIA_PRODUCCION_NUM']).agg({
    'CANTIDAD': 'mean',
    'CANTIDAD_NORMALIZADA': 'mean',
    'COMBO_ID': 'count'
}).reset_index()


MIN_OBS = 5
curvas_tipo = curvas_tipo[curvas_tipo['COMBO_ID'] >= MIN_OBS]

print(f"\n   Puntos de datos por tipo (días con ≥{MIN_OBS} observaciones):")
for tipo in df['TIPO_PRENDA'].unique():
    n_puntos = len(curvas_tipo[curvas_tipo['TIPO_PRENDA'] == tipo])
    print(f"   • {tipo}: {n_puntos} puntos")


resultados_por_tipo = {}

for tipo in df['TIPO_PRENDA'].unique():
    datos_tipo = curvas_tipo[curvas_tipo['TIPO_PRENDA'] == tipo]
    
    if len(datos_tipo) < 10:
        print(f"   ⚠️ {tipo}: Insuficientes puntos ({len(datos_tipo)}), se omite")
        continue
    
    x = datos_tipo['DIA_PRODUCCION_NUM'].values
    y = datos_tipo['CANTIDAD_NORMALIZADA'].values
    
    modelos = ajustar_modelos(x, y)
    
    if modelos:
        resultados_por_tipo[tipo] = modelos

print(f"\n✅ Modelos ajustados exitosamente para {len(resultados_por_tipo)} tipos de prenda")

# Crear tabla resumen
resumen_tipos = []
for tipo, modelos in resultados_por_tipo.items():
    if modelos['potencial']:
        resumen_tipos.append({
            'TIPO_PRENDA': tipo,
            'R2_Potencial': modelos['potencial']['r2'],
            'Tasa_Aprendizaje': modelos['potencial']['tasa_aprendizaje'],
            'R2_Logaritmico': modelos['logaritmico']['r2'] if modelos['logaritmico'] else np.nan,
            'R2_Lineal': modelos['lineal']['r2'] if modelos['lineal'] else np.nan,
            'Mejor_Modelo': max(
                [('Potencial', modelos['potencial']['r2']),
                 ('Logarítmico', modelos['logaritmico']['r2'] if modelos['logaritmico'] else 0),
                 ('Lineal', modelos['lineal']['r2'] if modelos['lineal'] else 0)],
                key=lambda x: x[1]
            )[0]
        })

df_resumen_tipos = pd.DataFrame(resumen_tipos).sort_values('R2_Potencial', ascending=False)

print("\n📊 RESULTADOS ")
print("="*80)
print(df_resumen_tipos.to_string(index=False))



n_tipos = len(resultados_por_tipo)
n_rows = (n_tipos + 2) // 3
fig, axes = plt.subplots(n_rows, 3, figsize=(18, n_rows*4.5))
axes = axes.flatten() if n_tipos > 1 else [axes]

for idx, tipo in enumerate(sorted(resultados_por_tipo.keys())):
    datos_tipo = curvas_tipo[curvas_tipo['TIPO_PRENDA'] == tipo]
    x = datos_tipo['DIA_PRODUCCION_NUM'].values
    y = datos_tipo['CANTIDAD_NORMALIZADA'].values
    
    # Scatter plot
    axes[idx].scatter(x, y, alpha=0.5, s=30, label='Datos observados')
    
    # Línea de tendencia (modelo potencial)
    if resultados_por_tipo[tipo]['potencial']:
        params = resultados_por_tipo[tipo]['potencial']['params']
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_pred = modelo_potencial(x_smooth, *params)
        
        r2 = resultados_por_tipo[tipo]['potencial']['r2']
        tasa = resultados_por_tipo[tipo]['potencial']['tasa_aprendizaje']
        
        axes[idx].plot(x_smooth, y_pred, 'r-', linewidth=2, 
                      label=f'Y = {params[0]:.1f} * X^{params[1]:.3f}')
        
        # Color del título según tasa
        color_titulo = 'green' if tasa > 0.05 else 'orange' if tasa > 0 else 'red'
        axes[idx].set_title(f'{tipo}\nR² = {r2:.3f}, b = {tasa:.3f}', 
                           fontsize=10, fontweight='bold', color=color_titulo)
    else:
        axes[idx].set_title(tipo, fontsize=10, fontweight='bold')
    
    axes[idx].set_xlabel('Día de Producción')
    axes[idx].set_ylabel('Cantidad Normalizada')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

# Ocultar subplots vacíos
for idx in range(len(resultados_por_tipo), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Curvas de Aprendizaje por Tipo de Prenda\n(Modelo Potencial: Y = a * X^b)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(RUTA_GRAFICOS + '03_curvas_por_tipo.png', dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: {RUTA_GRAFICOS}03_curvas_por_tipo.png")
plt.show()


print("\n" + "="*80)
print("COMPARACION")
print("="*80)

curvas_linea = df.groupby(['LINEA', 'DIA_PRODUCCION_NUM']).agg({
    'CANTIDAD_NORMALIZADA': 'mean',
    'COMBO_ID': 'count'
}).reset_index()

curvas_linea = curvas_linea[curvas_linea['COMBO_ID'] >= 3]

print(f"\n   Puntos de datos por línea (días con ≥3 observaciones):")
for linea in df['LINEA'].unique():
    n_puntos = len(curvas_linea[curvas_linea['LINEA'] == linea])
    print(f"   • {linea}: {n_puntos} puntos")

resultados_por_linea = {}

for linea in df['LINEA'].unique():
    datos_linea = curvas_linea[curvas_linea['LINEA'] == linea]
    
    if len(datos_linea) < 10:
        print(f"   ⚠️ {linea}: Insuficientes puntos ({len(datos_linea)}), se omite")
        continue
    
    x = datos_linea['DIA_PRODUCCION_NUM'].values
    y = datos_linea['CANTIDAD_NORMALIZADA'].values
    
    modelos = ajustar_modelos(x, y)
    
    if modelos:
        resultados_por_linea[linea] = modelos

print(f"\n✅ Modelos ajustados para {len(resultados_por_linea)} líneas")

comparacion_lineas = []
for linea, modelos in resultados_por_linea.items():
    if modelos['potencial']:
        comparacion_lineas.append({
            'LINEA': linea,
            'R2': modelos['potencial']['r2'],
            'Tasa_Aprendizaje': modelos['potencial']['tasa_aprendizaje'],
            'a (nivel_inicial)': modelos['potencial']['params'][0],
            'b (tasa)': modelos['potencial']['params'][1]
        })

df_comp_lineas = pd.DataFrame(comparacion_lineas).sort_values('Tasa_Aprendizaje', ascending=False)

print("\n📊 COMPARACION:")
print("="*80)
print(df_comp_lineas.to_string(index=False))

# Interpretación
print("\n💡 INTERPRETACION:")
for _, row in df_comp_lineas.iterrows():
    tasa = row['Tasa_Aprendizaje']
    r2 = row['R2']
    
    if tasa > 0.05:
        nivel = "FUERTE"
    elif tasa > 0:
        nivel = "moderado"
    else:
        nivel = "nulo o NEGATIVO"
    
    print(f"   • {row['LINEA']}: Aprendizaje {nivel} (b = {tasa:.3f}, R² = {r2:.3f})")



fig, axes = plt.subplots(1, 2, figsize=(16, 6))


for linea in resultados_por_linea.keys():
    datos_linea = curvas_linea[curvas_linea['LINEA'] == linea]
    x = datos_linea['DIA_PRODUCCION_NUM'].values
    y = datos_linea['CANTIDAD_NORMALIZADA'].values
    
    axes[0].scatter(x, y, alpha=0.4, s=30, label=f'{linea} (datos)')
    
    if resultados_por_linea[linea]['potencial']:
        params = resultados_por_linea[linea]['potencial']['params']
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_pred = modelo_potencial(x_smooth, *params)
        axes[0].plot(x_smooth, y_pred, linewidth=2.5, 
                    label=f'{linea} (Y = {params[0]:.1f} * X^{params[1]:.3f})')

axes[0].set_xlabel('Día de Producción', fontsize=11)
axes[0].set_ylabel('Cantidad Normalizada', fontsize=11)
axes[0].set_title('Curvas de Aprendizaje por Línea', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


if len(df_comp_lineas) > 0:
    lineas = df_comp_lineas['LINEA'].values
    tasas = df_comp_lineas['Tasa_Aprendizaje'].values
    colores = ['green' if t > 0 else 'red' for t in tasas]

    axes[1].barh(lineas, tasas, color=colores, alpha=0.7)
    axes[1].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Tasa de Aprendizaje (b)', fontsize=11)
    axes[1].set_title('Comparación de Tasas de Aprendizaje\n(valores positivos = mejora)', 
                     fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    for i, (linea, tasa) in enumerate(zip(lineas, tasas)):
        axes[1].text(tasa, i, f'  {tasa:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(RUTA_GRAFICOS + '04_comparacion_lineas.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("IMPACTO DE GAPS")
print("="*80)

df_con_gap = df[df['TIENE_GAP_LARGO'] == True]
df_sin_gap = df[df['TIENE_GAP_LARGO'] == False]

print(f"\n   Registros con gap largo (>20 días): {len(df_con_gap):,} ({len(df_con_gap)/len(df)*100:.1f}%)")
print(f"   Registros sin gap largo: {len(df_sin_gap):,} ({len(df_sin_gap)/len(df)*100:.1f}%)")

print("\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
print(f"   Cantidad normalizada promedio SIN gap: {df_sin_gap['CANTIDAD_NORMALIZADA'].mean():.2f}")
print(f"   Cantidad normalizada promedio CON gap: {df_con_gap['CANTIDAD_NORMALIZADA'].mean():.2f}")
print(f"   Diferencia: {df_con_gap['CANTIDAD_NORMALIZADA'].mean() - df_sin_gap['CANTIDAD_NORMALIZADA'].mean():.2f}")

# Test estadístico
if len(df_con_gap) > 30 and len(df_sin_gap) > 30:
    stat, pvalue = stats.ttest_ind(
        df_sin_gap['CANTIDAD_NORMALIZADA'].dropna(),
        df_con_gap['CANTIDAD_NORMALIZADA'].dropna()
    )
    print(f"\n📈 TEST ESTADÍSTICO (T-test):")
    print(f"   t-statistic: {stat:.3f}")
    print(f"   p-value: {pvalue:.4f}")
    
    if pvalue < 0.05:
        print("   ✅ Diferencia estadísticamente significativa (p < 0.05)")
        if df_con_gap['CANTIDAD_NORMALIZADA'].mean() < df_sin_gap['CANTIDAD_NORMALIZADA'].mean():
            print("   → Los gaps largos SÍ afectan negativamente la productividad")
        else:
            print("   → Los gaps largos mejoran la productividad (posible descanso beneficioso)")
    else:
        print("   ⚠️ No hay diferencia estadísticamente significativa (p ≥ 0.05)")
        print("   → Los gaps largos NO tienen impacto significativo")
else:
    print("\n⚠️ Insuficientes datos para test estadístico")



analisis_gaps = []
for combo_id, group in df.groupby('COMBO_ID'):
    if len(group) < 10:
        continue
    
    tiene_gaps = group['TIENE_GAP_LARGO'].sum() > 0
    
    # Cantidad promedio antes y después del primer gap
    if tiene_gaps:
        primer_gap_idx = group[group['TIENE_GAP_LARGO']].index.min()
        grupo_idx = group.index.tolist()
        pos_gap = grupo_idx.index(primer_gap_idx)
        
        if pos_gap > 0 and pos_gap < len(group) - 1:
            antes_gap = group.iloc[:pos_gap]['CANTIDAD_NORMALIZADA'].mean()
            despues_gap = group.iloc[pos_gap:]['CANTIDAD_NORMALIZADA'].mean()
            cambio_pct = ((despues_gap - antes_gap) / antes_gap * 100) if antes_gap > 0 else 0
        else:
            antes_gap = np.nan
            despues_gap = np.nan
            cambio_pct = np.nan
    else:
        antes_gap = np.nan
        despues_gap = np.nan
        cambio_pct = np.nan
    
    analisis_gaps.append({
        'COMBO_ID': combo_id,
        'LINEA': group['LINEA'].iloc[0],
        'TIPO_PRENDA': group['TIPO_PRENDA'].iloc[0],
        'tiene_gaps': tiene_gaps,
        'n_gaps': group['TIENE_GAP_LARGO'].sum(),
        'cantidad_antes_gap': antes_gap,
        'cantidad_despues_gap': despues_gap,
        'cambio_porcentual': cambio_pct
    })

df_gaps = pd.DataFrame(analisis_gaps)
df_gaps_con = df_gaps[df_gaps['tiene_gaps'] & df_gaps['cambio_porcentual'].notna()]

print(f"   Combinaciones analizadas: {len(df_gaps)}")
print(f"   Combinaciones con al menos un gap largo: {df_gaps['tiene_gaps'].sum()}")
print(f"   Combinaciones con cambio medible: {len(df_gaps_con)}")

if len(df_gaps_con) > 0:
    cambio_promedio = df_gaps_con['cambio_porcentual'].mean()
    print(f"\n   Cambio promedio después del gap: {cambio_promedio:.1f}%")
    
    mejoran = (df_gaps_con['cambio_porcentual'] > 0).sum()
    empeoran = (df_gaps_con['cambio_porcentual'] < 0).sum()
    total = len(df_gaps_con)
    
    print(f"   • Mejoran después del gap: {mejoran} ({mejoran/total*100:.1f}%)")
    print(f"   • Empeoran después del gap: {empeoran} ({empeoran/total*100:.1f}%)")
    print(f"   • Sin cambio: {total - mejoran - empeoran}")
    
    print("\n   💡 CONCLUSIÓN:")
    if empeoran > mejoran * 1.2:
        print("      → Los gaps largos tienden a PERJUDICAR el aprendizaje")
    elif mejoran > empeoran * 1.2:
        print("      → Los gaps largos pueden ser BENEFICIOSOS (descanso/rotación)")
    else:
        print("      → El impacto de los gaps es MIXTO o neutral")


fig, axes = plt.subplots(1, 2, figsize=(16, 6))


axes[0].hist(df_sin_gap['CANTIDAD_NORMALIZADA'].dropna(), bins=50, 
            alpha=0.5, label='Sin gap largo', density=True, color='blue')
if len(df_con_gap) > 0:
    axes[0].hist(df_con_gap['CANTIDAD_NORMALIZADA'].dropna(), bins=50, 
                alpha=0.5, label='Con gap largo', density=True, color='red')
axes[0].set_xlabel('Cantidad Normalizada')
axes[0].set_ylabel('Densidad')
axes[0].set_title('Distribución de Productividad\nSegún presencia de gaps largos (>20 días)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)


if len(df_gaps_con) > 0:
    cambios = df_gaps_con['cambio_porcentual'].dropna()
    axes[1].hist(cambios, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Sin cambio')
    axes[1].axvline(cambios.median(), color='blue', linestyle='--', linewidth=2, 
                   label=f'Mediana: {cambios.median():.1f}%')
    axes[1].set_xlabel('Cambio Porcentual en Productividad')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Cambio en Productividad Después del Gap\n(negativo = empeora, positivo = mejora)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'Insuficientes datos\npara análisis', 
                ha='center', va='center', fontsize=14)
    axes[1].axis('off')

plt.tight_layout()

