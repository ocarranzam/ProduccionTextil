
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

df_original = pd.read_excel('d://Data/DATA_SALIDA_RESUMEN.xlsx', sheet_name='Hoja1')


print(f"✅ Datos: {len(df_original):,} registros")


print("CONVERSION DE FECHAS")

df = df_original.copy()
df['FECHA'] = pd.to_datetime(df['FECHA'])

print(f"   Rango temporal: {df['FECHA'].min().date()} a {df['FECHA'].max().date()}")
print(f"   Días totales en el período: {(df['FECHA'].max() - df['FECHA'].min()).days}")

df['AÑO'] = df['FECHA'].dt.year
df['MES'] = df['FECHA'].dt.month
df['TRIMESTRE'] = df['FECHA'].dt.quarter
df['DIA_SEMANA'] = df['FECHA'].dt.day_name()

print("FILTRADO")
# combinaciones LINEA-ESTILO con suficientes observaciones
MIN_DIAS_PRODUCCION = 10  # Mínimo de días para considerar válida la combinación


combo_counts = df.groupby(['LINEA', 'ESTILO']).size().reset_index(name='n_dias')
combos_validos = combo_counts[combo_counts['n_dias'] >= MIN_DIAS_PRODUCCION]

print(f"\n   Combinaciones totales: {len(combo_counts)}")
print(f"   Combinaciones válidas: {len(combos_validos)} ({len(combos_validos)/len(combo_counts)*100:.1f}%)")

# Filtrar dataset
df_filtrado = df.merge(
    combos_validos[['LINEA', 'ESTILO']], 
    on=['LINEA', 'ESTILO'], 
    how='inner'
)

print(f"\n   Registros antes del filtro: {len(df):,}")
print(f"   Registros después del filtro: {len(df_filtrado):,}")
print(f"   Registros eliminados: {len(df) - len(df_filtrado):,} ({(len(df) - len(df_filtrado))/len(df)*100:.1f}%)")

df = df_filtrado.copy()


print(" VARIABLES DE EXPERIENCIA")

# Ordenar por combinacion y fecha
df = df.sort_values(['LINEA', 'ESTILO', 'FECHA']).reset_index(drop=True)

# Crear ID único de combinacion
df['COMBO_ID'] = df['LINEA'] + '_' + df['ESTILO']


experiencia_vars = []

for combo_id, group in df.groupby('COMBO_ID'):
    group = group.sort_values('FECHA').reset_index(drop=True)
    
    # 1. Número de día de producción (secuencia)
    group['DIA_PRODUCCION_NUM'] = range(1, len(group) + 1)
    
    # 2. Cantidad acumulada
    group['CANTIDAD_ACUMULADA'] = group['CANTIDAD'].cumsum()
    
    # 3. Días desde primera producción
    fecha_inicio = group['FECHA'].min()
    group['DIAS_DESDE_INICIO'] = (group['FECHA'] - fecha_inicio).dt.days
    
    # 4. Días desde última producción (gap)
    group['DIAS_DESDE_ULTIMA'] = group['FECHA'].diff().dt.days
    group.loc[0, 'DIAS_DESDE_ULTIMA'] = 0  # Primer día no tiene gap
    
    # 5. Cantidad promedio histórica (hasta ese día)
    group['CANTIDAD_PROMEDIO_HISTORICA'] = group['CANTIDAD'].expanding().mean()
    
    # 6. Cantidad normalizada por minutos estándar
    group['CANTIDAD_NORMALIZADA'] = group['CANTIDAD'] / group['MINUTOS_ESTILO']
    
    # 7. Cantidad relativa vs promedio histórico
    group['CANTIDAD_RELATIVA'] = group['CANTIDAD'] / group['CANTIDAD_PROMEDIO_HISTORICA']
    
    experiencia_vars.append(group)

df_experiencia = pd.concat(experiencia_vars, ignore_index=True)

variables_creadas = {
    'DIA_PRODUCCION_NUM': 'Número secuencial de día de producción (1, 2, 3...)',
    'CANTIDAD_ACUMULADA': 'Suma acumulada de unidades producidas',
    'DIAS_DESDE_INICIO': 'Días calendario desde primera producción',
    'DIAS_DESDE_ULTIMA': 'Días desde la última vez que se produjo (gap)',
    'CANTIDAD_PROMEDIO_HISTORICA': 'Promedio móvil de cantidad producida',
    'CANTIDAD_NORMALIZADA': 'Cantidad / Minutos estándar (productividad)',
    'CANTIDAD_RELATIVA': 'Cantidad actual / Promedio histórico'
}

for var, desc in variables_creadas.items():
    print(f"   • {var}: {desc}")
print("PROBABLES SALDOS")
print("-"*80)

# cantidad menor al 30% del promedio histórico de esa combinación
UMBRAL_SALDO = 0.3

df_experiencia['ES_PROBABLE_SALDO'] = (
    df_experiencia['CANTIDAD_RELATIVA'] < UMBRAL_SALDO
)

n_saldos = df_experiencia['ES_PROBABLE_SALDO'].sum()
print(f"\n   Criterio: Cantidad < {UMBRAL_SALDO*100:.0f}% del promedio histórico")
print(f"   Probables saldos detectados: {n_saldos:,} ({n_saldos/len(df_experiencia)*100:.1f}%)")


# CLASIFICACIÓN POR FASE DE PRODUCCIÓN
print("   Dividir: INICIO (primeros 25%), INTERMEDIO, FINAL (últimos 25%)")

def clasificar_fase(group):
    n = len(group)
    group = group.copy()
    
    # percentiles
    p25 = int(n * 0.25)
    p75 = int(n * 0.75)
    
    # Clasificar
    group['FASE_PRODUCCION'] = 'INTERMEDIO'
    group.loc[group['DIA_PRODUCCION_NUM'] <= p25, 'FASE_PRODUCCION'] = 'INICIO'
    group.loc[group['DIA_PRODUCCION_NUM'] > p75, 'FASE_PRODUCCION'] = 'FINAL'
    
    return group

df_experiencia = df_experiencia.groupby('COMBO_ID', group_keys=False).apply(clasificar_fase)


print(df_experiencia['FASE_PRODUCCION'].value_counts())


print("GAPS LARGOS")

UMBRAL_GAP_LARGO = 20  # días

df_experiencia['TIENE_GAP_LARGO'] = df_experiencia['DIAS_DESDE_ULTIMA'] > UMBRAL_GAP_LARGO

n_gaps = df_experiencia['TIENE_GAP_LARGO'].sum()
print(f"\n   Criterio: Gap > {UMBRAL_GAP_LARGO} días")
print(f"   Registros con gap largo: {n_gaps:,} ({n_gaps/len(df_experiencia)*100:.1f}%)")

print("\n   Estadísticas de gaps:")
print(f"   Promedio: {df_experiencia['DIAS_DESDE_ULTIMA'].mean():.1f} días")
print(f"   Mediana: {df_experiencia['DIAS_DESDE_ULTIMA'].median():.1f} días")
print(f"   Máximo: {df_experiencia['DIAS_DESDE_ULTIMA'].max():.0f} días")


print("VISUALIZACIÓN DE VARIABLES CREADAS")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Distribución de días de producción por combo
dias_por_combo = df_experiencia.groupby('COMBO_ID')['DIA_PRODUCCION_NUM'].max()
axes[0, 0].hist(dias_por_combo, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Días de Producción')
axes[0, 0].set_ylabel('Número de Combinaciones')
axes[0, 0].set_title('Distribución de Días de Producción\npor Combinación LINEA-ESTILO')
axes[0, 0].axvline(dias_por_combo.median(), color='red', 
                   linestyle='--', label=f'Mediana: {dias_por_combo.median():.0f}')
axes[0, 0].legend()

# 2. Cantidad acumulada promedio por día de producción
cantidad_por_dia = df_experiencia.groupby('DIA_PRODUCCION_NUM').agg({
    'CANTIDAD_ACUMULADA': 'mean',
    'COMBO_ID': 'count'
}).reset_index()
cantidad_por_dia = cantidad_por_dia[cantidad_por_dia['COMBO_ID'] >= 10]  # Min 10 combos

axes[0, 1].plot(cantidad_por_dia['DIA_PRODUCCION_NUM'], 
                cantidad_por_dia['CANTIDAD_ACUMULADA'], 
                linewidth=2, marker='o', markersize=3)
axes[0, 1].set_xlabel('Día de Producción')
axes[0, 1].set_ylabel('Cantidad Acumulada Promedio')
axes[0, 1].set_title('Evolución de Cantidad Acumulada\n(Promedio entre combinaciones)')
axes[0, 1].grid(True, alpha=0.3)

# 3. Distribución de gaps
gaps_no_cero = df_experiencia[df_experiencia['DIAS_DESDE_ULTIMA'] > 0]['DIAS_DESDE_ULTIMA']
axes[0, 2].hist(gaps_no_cero, bins=50, edgecolor='black', alpha=0.7)
axes[0, 2].set_xlabel('Días desde Última Producción')
axes[0, 2].set_ylabel('Frecuencia')
axes[0, 2].set_title('Distribución de Gaps entre Producciones')
axes[0, 2].axvline(UMBRAL_GAP_LARGO, color='red', 
                   linestyle='--', label=f'Umbral gap largo: {UMBRAL_GAP_LARGO}d')
axes[0, 2].legend()
axes[0, 2].set_xlim(0, 100)

# 4. Cantidad relativa por fase
df_no_saldo = df_experiencia[~df_experiencia['ES_PROBABLE_SALDO']]
axes[1, 0].boxplot([
    df_no_saldo[df_no_saldo['FASE_PRODUCCION']=='INICIO']['CANTIDAD_RELATIVA'],
    df_no_saldo[df_no_saldo['FASE_PRODUCCION']=='INTERMEDIO']['CANTIDAD_RELATIVA'],
    df_no_saldo[df_no_saldo['FASE_PRODUCCION']=='FINAL']['CANTIDAD_RELATIVA']
], labels=['INICIO', 'INTERMEDIO', 'FINAL'])
axes[1, 0].set_ylabel('Cantidad Relativa')
axes[1, 0].set_title('Cantidad Relativa por Fase\n(excluyendo saldos)')
axes[1, 0].axhline(1, color='red', linestyle='--', alpha=0.5, label='Promedio histórico')
axes[1, 0].legend()

# 5. Cantidad normalizada por tipo de prenda
tipo_norm = df_experiencia.groupby('TIPO_PRENDA')['CANTIDAD_NORMALIZADA'].mean().sort_values()
axes[1, 1].barh(range(len(tipo_norm)), tipo_norm.values)
axes[1, 1].set_yticks(range(len(tipo_norm)))
axes[1, 1].set_yticklabels(tipo_norm.index)
axes[1, 1].set_xlabel('Cantidad Normalizada Promedio')
axes[1, 1].set_title('Productividad por Tipo de Prenda\n(Cantidad / Minutos Estándar)')

# 6. Proporción de saldos por fase
fase_saldo = pd.crosstab(
    df_experiencia['FASE_PRODUCCION'], 
    df_experiencia['ES_PROBABLE_SALDO'], 
    normalize='index'
) * 100
fase_saldo.plot(kind='bar', ax=axes[1, 2], stacked=False)
axes[1, 2].set_xlabel('Fase de Producción')
axes[1, 2].set_ylabel('Porcentaje')
axes[1, 2].set_title('Proporción de Probables Saldos\npor Fase')
axes[1, 2].legend(['No es saldo', 'Es saldo'], loc='upper right')
axes[1, 2].set_xticklabels(axes[1, 2].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('d://Data/02_variables_experiencia.png', dpi=300, bbox_inches='tight')
print("\n✅ Gráfico guardado: 02_variables_experiencia.png")
plt.show()







