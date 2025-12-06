import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10




df = pd.read_excel('d://Data/DATA_SALIDA_RESUMEN.xlsx', sheet_name='Hoja1')

print(f"   Registros: {len(df):,}")
print(f"   Período: {df['FECHA'].min()} a {df['FECHA'].max()}")


print("\n1. Estructura:")
print(df.info())

print("\n2. Primeras filas:")
print(df.head(10))

print("\n3. Estadísticas descriptivas:")
print(df.describe())

print("\n4. Valores únicos por columna:")
for col in ['LINEA', 'TIPO_PRENDA', 'ESTILO']:
    print(f"   {col}: {df[col].nunique()} valores únicos")
    
print("\n5. Valores faltantes:")
print(df.isnull().sum())

print("\n6. Distribución por dimensiones clave:")
print("\n   Registros por LÍNEA:")
print(df['LINEA'].value_counts())

print("\n   Registros por TIPO_PRENDA:")
print(df['TIPO_PRENDA'].value_counts().sort_values(ascending=False))

print("\n   Top 10 ESTILOS más producidos:")
print(df['ESTILO'].value_counts().head(10))


print("\n" + "-"*80)
print("EXPLORATORIO")


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Distribución de cantidad producida
axes[0, 0].hist(df['CANTIDAD'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Cantidad Producida')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Cantidad Producida por Día')
axes[0, 0].axvline(df['CANTIDAD'].mean(), color='red', 
                   linestyle='--', label=f'Media: {df["CANTIDAD"].mean():.0f}')
axes[0, 0].axvline(df['CANTIDAD'].median(), color='green', 
                   linestyle='--', label=f'Mediana: {df["CANTIDAD"].median():.0f}')
axes[0, 0].legend()

# 2. Cantidad por tipo de prenda
tipo_cantidad = df.groupby('TIPO_PRENDA')['CANTIDAD'].agg(['mean', 'count'])
tipo_cantidad = tipo_cantidad.sort_values('mean', ascending=False)
axes[0, 1].barh(tipo_cantidad.index, tipo_cantidad['mean'])
axes[0, 1].set_xlabel('Cantidad Promedio')
axes[0, 1].set_title('Cantidad Promedio por Tipo de Prenda')
for i, v in enumerate(tipo_cantidad['mean']):
    axes[0, 1].text(v, i, f' {v:.0f}', va='center')

# 3. Cantidad por línea
linea_stats = df.groupby('LINEA')['CANTIDAD'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(linea_stats))
axes[1, 0].bar(x_pos, linea_stats['mean'], yerr=linea_stats['std'], 
               capsize=5, alpha=0.7)
axes[1, 0].set_xlabel('Línea')
axes[1, 0].set_ylabel('Cantidad Promedio')
axes[1, 0].set_title('Cantidad Promedio por Línea (con desv. estándar)')
axes[1, 0].set_xticks(x_pos)
axes[1, 0].set_xticklabels(linea_stats.index, rotation=45)

# 4. Series temporal de producción
produccion_diaria = df.groupby('FECHA')['CANTIDAD'].sum().sort_index()
axes[1, 1].plot(produccion_diaria.index, produccion_diaria.values, 
                linewidth=0.8, alpha=0.7)
axes[1, 1].set_xlabel('Fecha')
axes[1, 1].set_ylabel('Cantidad Total')
axes[1, 1].set_title('Serie Temporal de Producción Total')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('01_exploracion_inicial.png', dpi=300, bbox_inches='tight')
plt.show()


# ANALISIS


print("\n1. Detección de valores atípicos en CANTIDAD:")
Q1 = df['CANTIDAD'].quantile(0.25)
Q3 = df['CANTIDAD'].quantile(0.75)
IQR = Q3 - Q1
outliers_lower = df['CANTIDAD'] < (Q1 - 1.5 * IQR)
outliers_upper = df['CANTIDAD'] > (Q3 + 1.5 * IQR)

print(f"   Q1 (25%): {Q1:.0f}")
print(f"   Q3 (75%): {Q3:.0f}")
print(f"   IQR: {IQR:.0f}")
print(f"   Límite inferior: {Q1 - 1.5 * IQR:.0f}")
print(f"   Límite superior: {Q3 + 1.5 * IQR:.0f}")
print(f"   Outliers inferiores: {outliers_lower.sum()} ({outliers_lower.sum()/len(df)*100:.1f}%)")
print(f"   Outliers superiores: {outliers_upper.sum()} ({outliers_upper.sum()/len(df)*100:.1f}%)")

print("\n2. Análisis de continuidad temporal por ESTILO-LINEA:")
# Agrupar por combinación única
df_sorted = df.sort_values(['LINEA', 'ESTILO', 'FECHA'])
df_sorted['FECHA'] = pd.to_datetime(df_sorted['FECHA'])

# Calcular gaps entre producciones
combos_produccion = []
for (linea, estilo), group in df_sorted.groupby(['LINEA', 'ESTILO']):
    if len(group) >= 5:  # Solo combinaciones con al menos 5 días
        fechas = group['FECHA'].sort_values()
        gaps = fechas.diff().dt.days.dropna()
        
        combos_produccion.append({
            'LINEA': linea,
            'ESTILO': estilo,
            'TIPO_PRENDA': group['TIPO_PRENDA'].iloc[0],
            'dias_produccion': len(group),
            'gap_promedio': gaps.mean(),
            'gap_maximo': gaps.max(),
            'cantidad_promedio': group['CANTIDAD'].mean()
        })

df_combos = pd.DataFrame(combos_produccion)
print(f"\n   Combinaciones LINEA-ESTILO con ≥5 días: {len(df_combos)}")
print(f"   Gap promedio entre producciones: {df_combos['gap_promedio'].mean():.1f} días")
print(f"   Gap máximo encontrado: {df_combos['gap_maximo'].max():.0f} días")

print("\n3. Top 10 combinaciones LINEA-ESTILO más producidas:")
top_combos = df_combos.nlargest(10, 'dias_produccion')[
    ['LINEA', 'ESTILO', 'TIPO_PRENDA', 'dias_produccion', 'cantidad_promedio']
]
print(top_combos.to_string(index=False))

print("\n" + "="*80)
print("RESUMEN FASE 1 Y 2:")
print("="*80)
print(f"""
   - Total de registros: {len(df):,}
   - Período de análisis: {df['FECHA'].min()} a {df['FECHA'].max()}
   - Líneas de producción: {df['LINEA'].nunique()}
   - Tipos de prenda: {df['TIPO_PRENDA'].nunique()}
   - Estilos únicos: {df['ESTILO'].nunique()}
   
✅ Calidad
   - Outliers identificados: {(outliers_lower.sum() + outliers_upper.sum())/len(df)*100:.1f}%
""")
