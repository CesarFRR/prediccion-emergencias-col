#!/usr/bin/env python3
"""
Prepara datos departamentales para subir a HuggingFace.
========================================================
Crea un ZIP por departamento con sus CHIRPS, SAR y DEM.
Opcional: crea los ZIPs de datos nacionales completos.

Uso:
    python core/preparar_datos_huggingface.py          # ZIPs por depto
    python core/preparar_datos_huggingface.py 05       # Solo Antioquia
    python core/preparar_datos_huggingface.py --nacional  # ZIP nacional
"""

import sys, shutil, zipfile
from pathlib import Path
import pandas as pd
import unicodedata

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'datos_globales'
HF_DIR = BASE_DIR / 'huggingface_upload'
DEPTOS_DIR = BASE_DIR / 'departamentos'

CHIRPS_PATH = DATA_DIR / 'CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet'
SAR_PATH    = DATA_DIR / 'SAR_COLOMBIA_MENSUAL_2018_2026.parquet'
DEM_PATH    = DATA_DIR / 'dem_srtm30_colombia.tif'
NOAA_PATH   = DATA_DIR / 'indices_climaticos_mensuales.parquet'
GPKG_PATH   = DATA_DIR / 'Carto100000_Colombia_DI_2022.gpkg'


def strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn')


DEPARTAMENTOS = {
    '05':'Antioquia','08':'Atlantico','11':'Bogota','13':'Bolivar','15':'Boyaca',
    '17':'Caldas','18':'Caqueta','19':'Cauca','20':'Cesar','23':'Cordoba',
    '25':'Cundinamarca','27':'Choco','41':'Huila','44':'La Guajira','47':'Magdalena',
    '50':'Meta','52':'Narino','54':'Norte de Santander','63':'Quindio','66':'Risaralda',
    '68':'Santander','70':'Sucre','73':'Tolima','76':'Valle del Cauca','81':'Arauca',
    '85':'Casanare','86':'Putumayo','88':'San Andres','91':'Amazonas','94':'Guainia',
    '95':'Guaviare','97':'Vaupes','99':'Vichada',
}

GAUL_TO_CODE = {strip_acc(k).lower(): v for k, v in {
    'Amazonas': '91', 'Antioquia': '05', 'Arauca': '81',
    'Atlantico': '08', 'Atlántico': '08', 'Bolivar': '13', 'Bolívar': '13',
    'Boyaca': '15', 'Boyacá': '15', 'Caldas': '17', 'Caqueta': '18', 'Caquetá': '18',
    'Casanare': '85', 'Cauca': '19', 'Cesar': '20', 'Choco': '27', 'Chocó': '27',
    'Cordoba': '23', 'Córdoba': '23', 'Cundinamarca': '25',
    'Guainia': '94', 'Guainía': '94', 'Guaviare': '95',
    'Huila': '41', 'Guajira': '44', 'La Guajira': '44',
    'Magdalena': '47', 'Meta': '50', 'Narino': '52', 'Nariño': '52',
    'Norte de Santander': '54', 'Putumayo': '86', 'Quindio': '63', 'Quindío': '63',
    'Risaralda': '66', 'Santander': '68', 'Sucre': '70',
    'Tolima': '73', 'Valle del Cauca': '76', 'Vaupes': '97', 'Vaupés': '97',
    'Vichada': '99', 'San Andres y Providencia': '88',
    'Bogota': '11', 'Bogotá': '11', 'Buenaventura': '76',
}.items()}


def preparar_departamento(codigo: str):
    """Crea un ZIP con CHIRPS + SAR + DEM para un departamento."""
    nombre = DEPARTAMENTOS.get(codigo, f'Depto_{codigo}')
    out_dir = HF_DIR / 'departamentos'
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f'{nombre.lower().replace(" ","_")}_datos.zip'

    if zip_path.exists():
        print(f'   ✅ {codigo} {nombre} (ya existe)')
        return zip_path

    # Crear directorio temporal
    tmp = BASE_DIR / f'_tmp_{codigo}'
    tmp.mkdir(exist_ok=True)
    files_added = []

    # 1. CHIRPS filtrado
    if CHIRPS_PATH.exists():
        chirps = pd.read_parquet(CHIRPS_PATH)
        chirps['municipio'] = chirps['ADM2_NAME'].str.upper().apply(strip_acc)
        mask = chirps['ADM1_NAME'].apply(
            lambda x: GAUL_TO_CODE.get(strip_acc(str(x)).lower()) == codigo
        )
        chirps_depto = chirps[mask]
        if len(chirps_depto) > 0:
            chirps_out = tmp / f'CHIRPS_{nombre.lower().replace(" ","_")}.parquet'
            chirps_depto.to_parquet(chirps_out, index=False)
            files_added.append(chirps_out.name)

    # 2. SAR filtrado
    if SAR_PATH.exists():
        sar = pd.read_parquet(SAR_PATH)
        mask = sar['ADM1_NAME'].apply(
            lambda x: GAUL_TO_CODE.get(strip_acc(str(x)).lower()) == codigo
        )
        sar_depto = sar[mask]
        if len(sar_depto) > 0:
            sar_out = tmp / f'SAR_{nombre.lower().replace(" ","_")}.parquet'
            sar_depto.to_parquet(sar_out, index=False)
            files_added.append(sar_out.name)

    # 3. DEM (ya recortado por departamento)
    dem_depto = DEPTOS_DIR / codigo / 'processed' / 'dem_srtm30.tif'
    if dem_depto.exists():
        dem_out = tmp / f'DEM30_{nombre.lower().replace(" ","_")}.tif'
        shutil.copy2(dem_depto, dem_out)
        files_added.append(dem_out.name)

    # 4. NOAA (mismo para todos, pero pequeno, incluirlo)
    if NOAA_PATH.exists():
        noaa_out = tmp / 'indices_climaticos_mensuales.parquet'
        shutil.copy2(NOAA_PATH, noaa_out)
        files_added.append(noaa_out.name)

    if not files_added:
        shutil.rmtree(tmp)
        print(f'   ❌ {codigo} {nombre} (sin datos)')
        return None

    # Crear ZIP
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_added:
            zf.write(tmp / f, f)

    shutil.rmtree(tmp)
    size_mb = zip_path.stat().st_size / 1e6
    print(f'   ✅ {codigo} {nombre}: {len(files_added)} archivos, {size_mb:.1f} MB')
    return zip_path


def preparar_nacional():
    """Crea ZIPs con los datos nacionales completos."""
    out_dir = HF_DIR / 'nacional'
    out_dir.mkdir(parents=True, exist_ok=True)

    total_mb = 0

    # CHIRPS
    if CHIRPS_PATH.exists():
        dest = out_dir / CHIRPS_PATH.name
        shutil.copy2(CHIRPS_PATH, dest)
        mb = dest.stat().st_size / 1e6
        total_mb += mb
        print(f'   ✅ CHIRPS nacional: {mb:.0f} MB')

    # SAR
    if SAR_PATH.exists():
        dest = out_dir / SAR_PATH.name
        shutil.copy2(SAR_PATH, dest)
        mb = dest.stat().st_size / 1e6
        total_mb += mb
        print(f'   ✅ SAR nacional: {mb:.0f} MB')

    # DEM nacional
    if DEM_PATH.exists():
        dest = out_dir / DEM_PATH.name
        shutil.copy2(DEM_PATH, dest)
        mb = dest.stat().st_size / 1e6
        total_mb += mb
        print(f'   ✅ DEM nacional: {mb:.0f} MB')

    # NOAA
    if NOAA_PATH.exists():
        dest = out_dir / NOAA_PATH.name
        shutil.copy2(NOAA_PATH, dest)
        mb = dest.stat().st_size / 1e6
        total_mb += mb
        print(f'   ✅ NOAA: {mb:.0f} MB')

    print(f'\n   Total nacional: {total_mb:.0f} MB')


if __name__ == '__main__':
    if '--nacional' in sys.argv:
        preparar_nacional()
    elif len(sys.argv) > 1:
        for cod in sys.argv[1:]:
            if cod in DEPARTAMENTOS:
                preparar_departamento(cod)
    else:
        # Todos los departamentos
        print(f'Preparando {len(DEPARTAMENTOS)} departamentos para HuggingFace...\n')
        for cod in sorted(DEPARTAMENTOS):
            preparar_departamento(cod)
        print(f'\n✅ Listo. Subir el contenido de: {HF_DIR}/')
