"""
Procesador de Exports GEE → Dataset Listo para Pipeline
=========================================================
Toma los CSVs que GEE exportó a Google Drive y los convierte en datasets
por departamento listos para el pipeline de Fase 4.

Input (descargar de Google Drive a FASE-4/gee/):
  - CHIRPS_COLOMBIA_DIARIO_2018_2026.csv   (~50MB)
  - SAR_COLOMBIA_MENSUAL_2018_2026.csv      (~15MB)

Output (por departamento):
  - FASE-4/datos_vectoriales/{cod_depto}/dataset_{nombre}.parquet

Uso:
  python FASE-4/gee/procesar_exports.py          # Todos los deptos
  python FASE-4/gee/procesar_exports.py 05       # Solo Antioquia
  python FASE-4/gee/procesar_exports.py 05 76 27 # Antioquia + Valle + Chocó
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, unicodedata, json

# ─── CONFIG ────────────────────────────────────────────────────
GEE_DIR = Path('FASE-4/gee')
OUT_BASE = Path('FASE-4/datos_vectoriales')

CHIRPS_FILE = GEE_DIR / 'CHIRPS_COLOMBIA_DIARIO_2018_2026.csv'
SAR_FILE    = GEE_DIR / 'SAR_COLOMBIA_MENSUAL_2018_2026.csv'
NOAA_FILE   = Path('ENTREGA_FASE2/reproducir/indices_climaticos_mensuales.parquet')

# Mapeo ADM1_NAME (GAUL) → código DIVIPOLA
# GAUL usa nombres en inglés/español mezclados. Mapeamos a código.
DEPTO_GAUL_TO_CODE = {
    'Amazonas': '91', 'Antioquia': '05', 'Arauca': '81',
    'Atlántico': '08', 'Bolívar': '13', 'Boyacá': '15',
    'Caldas': '17', 'Caquetá': '18', 'Casanare': '85',
    'Cauca': '19', 'Cesar': '20', 'Chocó': '27',
    'Córdoba': '23', 'Cundinamarca': '25', 'Guainía': '94',
    'Guaviare': '95', 'Huila': '41', 'La Guajira': '44',
    'Magdalena': '47', 'Meta': '50', 'Nariño': '52',
    'Norte de Santander': '54', 'Putumayo': '86', 'Quindío': '63',
    'Risaralda': '66', 'Santander': '68', 'Sucre': '70',
    'Tolima': '73', 'Valle del Cauca': '76', 'Vaupés': '97',
    'Vichada': '99',
    # Posibles variaciones GAUL
    'San Andrés y Providencia': '88',
    'Bogotá': '11', 'Bogota': '11', 'Bogotá D.C.': '11',
    'Distrito Capital': '11',
    # Con tilde o sin tilde, GAUL es inconsistente
    'Bolivar': '13', 'Cordoba': '23', 'Guajira': '44',
    'Norte De Santander': '54', 'Santander': '68',
    'Cundinamarca': '25', 'Quindio': '63',
}


def strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn')


def procesar_chirps(csv_path: Path) -> pd.DataFrame:
    """
    Procesa CHIRPS diario de GEE → DataFrame (municipio, fecha, precip_mm).
    GEE exporta con columnas: ADM1_NAME, ADM2_NAME, fecha, mean
    """
    print(f'Cargando CHIRPS: {csv_path.name}...')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    # Renombrar columnas de GEE
    col_map = {}
    for c in df.columns:
        if 'adm1' in c: col_map[c] = 'departamento'
        elif 'adm2' in c: col_map[c] = 'municipio'
        elif 'fecha' in c: col_map[c] = 'fecha'
        elif 'mean' in c or 'precip' in c: col_map[c] = 'precip_mm'

    if col_map:
        df = df.rename(columns={k: v for k, v in col_map.items() if v not in df.columns})

    df['municipio'] = df['municipio'].str.upper().apply(strip_acc)
    df['departamento'] = df['departamento'].apply(strip_acc)
    df['fecha'] = pd.to_datetime(df['fecha'])

    print(f'   {len(df):,} filas, {df["municipio"].nunique()} municipios')
    return df


def procesar_sar(csv_path: Path) -> pd.DataFrame:
    """
    Procesa SAR mensual de GEE → DataFrame mensual.
    GEE exporta con: ADM1_NAME, ADM2_NAME, fecha_mes, VV_mean, VH_mean, ...
    """
    print(f'Cargando SAR: {csv_path.name}...')
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    col_map = {}
    for c in df.columns:
        if 'adm1' in c: col_map[c] = 'departamento'
        elif 'adm2' in c: col_map[c] = 'municipio'
        elif 'fecha' in c: col_map[c] = 'fecha_mes'

    if col_map:
        df = df.rename(columns={k: v for k, v in col_map.items() if v not in df.columns})

    df['municipio'] = df['municipio'].str.upper().apply(strip_acc)
    df['departamento'] = df['departamento'].apply(strip_acc)

    # SAR viene mensual. Crear fecha como primer día del mes.
    df['fecha'] = pd.to_datetime(df['fecha_mes'] + '-01')

    # Asegurar columnas SAR
    for col in ['vv_mean', 'vh_mean', 'vv_stddev', 'vh_stddev', 'vvvh_ratio', 'n_scenes']:
        if col not in df.columns:
            df[col] = np.nan

    print(f'   {len(df):,} filas, {df["municipio"].nunique()} municipios')
    return df


def merge_noaa(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega índices climáticos NOAA (globales, mensuales)."""
    if not NOAA_FILE.exists():
        print('   ⚠️  NOAA no encontrado. Continuando sin índices climáticos.')
        return df

    noaa = pd.read_parquet(NOAA_FILE)
    noaa['anio_mes'] = noaa['YR'].astype(str) + '-' + noaa['MON'].astype(str).str.zfill(2)
    df['anio_mes'] = df['fecha'].dt.year.astype(str) + '-' + df['fecha'].dt.month.astype(str).str.zfill(2)

    cols = ['anio_mes', 'SOI', 'QBO30', 'QBO50', 'ZWND200',
            'N12_ANOM', 'N3_ANOM', 'N4_ANOM', 'oni_anom', 'oni_total']
    available = [c for c in cols if c in noaa.columns]
    df = df.merge(noaa[available], on='anio_mes', how='left')
    return df


def construir_dataset_departamento(chirps: pd.DataFrame, sar: pd.DataFrame,
                                    cod_depto: str, nombre_depto: str) -> pd.DataFrame:
    """Une CHIRPS + SAR + NOAA para un departamento específico."""

    # Mapear nombre GAUL → filtrar
    # El CSV de GEE tiene nombre de departamento. Filtramos por mapeo.
    # Si no hay match exacto, intentamos por código en el GeoJSON local.
    depto_chirps = chirps[chirps['departamento'].apply(
        lambda x: _match_depto(x, cod_depto, nombre_depto)
    )].copy()

    depto_sar = sar[sar['departamento'].apply(
        lambda x: _match_depto(x, cod_depto, nombre_depto)
    )].copy()

    if len(depto_chirps) == 0:
        print(f'   ⚠️  Sin datos CHIRPS para {nombre_depto}')
        return pd.DataFrame()

    # CHIRPS: calcular acumulados
    depto_chirps = depto_chirps.sort_values(['municipio', 'fecha'])
    for w in [1, 3, 7, 15, 30]:
        depto_chirps[f'precip_acum_{w}d'] = depto_chirps.groupby('municipio')['precip_mm'].transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        )

    # Renombrar para consistencia
    depto_chirps = depto_chirps.rename(columns={'precip_mm': 'chirps_precip_mm_dia'})

    # SAR: merge al dataset diario (forward fill mensual)
    if len(depto_sar) > 0:
        sar_cols = ['municipio', 'fecha', 'vv_mean', 'vh_mean', 'vv_stddev',
                     'vh_stddev', 'vvvh_ratio', 'n_scenes']
        sar_merge = depto_sar[sar_cols].rename(columns={
            'vv_mean': 'VV_mean', 'vh_mean': 'VH_mean',
            'vv_stddev': 'VV_stdDev', 'vh_stddev': 'VH_stdDev',
            'vvvh_ratio': 'VV_minus_VH', 'n_scenes': 'n_scenes'
        })
        # SAR es mensual. Para el merge diario, forward-fill al mes siguiente.
        # Simple: creamos un rango de fechas mensual y merge por mes-año.
        depto_chirps['mes_key'] = depto_chirps['fecha'].dt.to_period('M')
        sar_merge['mes_key'] = sar_merge['fecha'].dt.to_period('M')
        sar_merge = sar_merge.drop(columns=['fecha'])

        depto_chirps = depto_chirps.merge(sar_merge, on=['municipio', 'mes_key'], how='left')
        depto_chirps = depto_chirps.drop(columns=['mes_key'])
    else:
        for c in ['VV_mean', 'VH_mean', 'VV_stdDev', 'VH_stdDev', 'VV_minus_VH', 'n_scenes']:
            depto_chirps[c] = np.nan

    # Calcular z_VV_mean (anomalía estandarizada de VV)
    if 'VV_mean' in depto_chirps.columns:
        vv_mean = depto_chirps.groupby('municipio')['VV_mean'].transform('mean')
        vv_std = depto_chirps.groupby('municipio')['VV_mean'].transform('std')
        depto_chirps['z_VV_mean'] = (depto_chirps['VV_mean'] - vv_mean) / vv_std.replace(0, 1)

    # P90 de precip 3d
    depto_chirps['p90_precip_3d'] = depto_chirps.groupby('municipio')['chirps_precip_mm_dia'].transform(
        lambda x: x.rolling(3, min_periods=1).quantile(0.9)
    )

    # NOAA
    depto_chirps = merge_noaa(depto_chirps)

    # Agregar código depto
    depto_chirps['cod_depto'] = cod_depto

    return depto_chirps


def _match_depto(nombre_gaul: str, cod: str, nombre: str) -> bool:
    """Determina si un nombre de departamento de GAUL coincide con nuestro código."""
    n = strip_acc(str(nombre_gaul)).upper()
    expected = DEPTO_GAUL_TO_CODE.get(n, None)
    if expected is not None:
        return expected == cod
    # Fallback: comparar nombre
    expected_name = strip_acc(str(nombre)).upper()
    return n == expected_name


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if not CHIRPS_FILE.exists():
        print(f'❌ CHIRPS no encontrado: {CHIRPS_FILE}')
        print('   Descargá el CSV de Google Drive a FASE-4/gee/')
        sys.exit(1)

    # Cargar ambos
    chirps = procesar_chirps(CHIRPS_FILE)
    sar = procesar_sar(SAR_FILE) if SAR_FILE.exists() else pd.DataFrame()

    # Determinar qué departamentos procesar
    if len(sys.argv) > 1:
        codigos = sys.argv[1:]
    else:
        # Todos los códigos mapeados
        codigos = sorted(set(DEPTO_GAUL_TO_CODE.values()))

    print(f'\nProcesando {len(codigos)} departamento(s)...\n')

    for cod in codigos:
        # Buscar nombre
        nombre = next((k for k, v in DEPTO_GAUL_TO_CODE.items() if v == cod), f'Depto_{cod}')

        dataset = construir_dataset_departamento(chirps, sar, cod, nombre)
        if len(dataset) == 0:
            continue

        # Guardar
        out_dir = OUT_BASE / cod
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'dataset_{nombre.lower().replace(" ", "_")}.parquet'
        dataset.to_parquet(out_path, index=False)

        n_flood = dataset.get('flood_target', pd.Series([0])).sum()
        print(f'   {cod} {nombre}: {len(dataset):,} filas, '
              f'{dataset["municipio"].nunique()} municipios → {out_path.name}')

    print(f'\n✅ Datasets guardados en: {OUT_BASE}/')
    print('   Listos para: python FASE-4/{depto}/run.py')
