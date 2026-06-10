#!/usr/bin/env python3
"""
FloodModel Colombia — Utilidad de descarga de datos globales
==============================================================
Verifica los 4 archivos de datos nacionales. Si faltan, ofrece
descargarlos desde HuggingFace, Zenodo, GitHub Releases o MediaFire.

Los archivos .zip se descomprimen automaticamente.

Ejecutar una vez al clonar el repositorio:
    python app.py
"""

import sys, os, zipfile
from pathlib import Path
from urllib.request import urlretrieve

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'datos_globales'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Archivos requeridos que deben existir al final
ARCHIVOS_REQUERIDOS = [
    'CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet',
    'SAR_COLOMBIA_MENSUAL_2018_2026.parquet',
    'indices_climaticos_mensuales.parquet',
    'Carto100000_Colombia_DI_2022.gpkg',
]

# Fuentes de descarga (en orden de intento)
FUENTES_ZIP = [
    ('GitHub Releases',  'https://github.com/Antioquia-Flood-AI/antioquia-flood-lstm/releases/download/datasets/datos_globales.zip'),
    ('Zenodo',           'https://zenodo.org/api/records/20619169/files-archive'),
    ('MediaFire',        'https://www.mediafire.com/file/atsb5vklle6zecs/datos_globales.zip/file'),
]

FUENTES_INDIVIDUALES = {
    'CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet': {
        'tamano': '18 MB',
        'descripcion': 'Precipitacion diaria CHIRPS (2018-2026)',
        'urls': [
            'https://huggingface.co/buckets/rokudev/floodmodel-colombia/resolve/datos_globales/CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet',
        ],
    },
    'SAR_COLOMBIA_MENSUAL_2018_2026.parquet': {
        'tamano': '5 MB',
        'descripcion': 'Radar Sentinel-1 mensual (2018-2026)',
        'urls': [
            'https://huggingface.co/buckets/rokudev/floodmodel-colombia/resolve/datos_globales/SAR_COLOMBIA_MENSUAL_2018_2026.parquet',
        ],
    },
    'indices_climaticos_mensuales.parquet': {
        'tamano': '24 KB',
        'descripcion': 'Indices NOAA (SOI, QBO, ONI, Nino)',
        'urls': [
            'https://huggingface.co/buckets/rokudev/floodmodel-colombia/resolve/datos_globales/indices_climaticos_mensuales.parquet',
        ],
    },
    'Carto100000_Colombia_DI_2022.gpkg': {
        'tamano': '1.7 GB',
        'descripcion': 'Cartografia IGAC 1:100,000 (mapa nacional)',
        'urls': [
            'https://huggingface.co/buckets/rokudev/floodmodel-colombia/resolve/datos_globales/Carto100000_Colombia_DI_2022.gpkg',
        ],
    },
}


def verificar_todos() -> dict:
    """Retorna {nombre: True/False} para cada archivo requerido."""
    return {n: (DATA_DIR / n).exists() and (DATA_DIR / n).stat().st_size > 0
            for n in ARCHIVOS_REQUERIDOS}


def _progreso(bloque_num, bloque_tam, tamano_total):
    if tamano_total > 0:
        pct = min(100, bloque_num * bloque_tam * 100 // tamano_total)
        mb = bloque_num * bloque_tam / 1_048_576
        total_mb = tamano_total / 1_048_576
        print(f'\r   {pct:3d}%  {mb:.0f}/{total_mb:.0f} MB', end='', flush=True)


def descargar_zip(fuente_nombre: str, url: str) -> bool:
    """Descarga un ZIP y extrae su contenido a datos_globales/."""
    tmp_zip = DATA_DIR / '_descarga_temporal.zip'
    print(f'   Intentando {fuente_nombre}...')
    try:
        urlretrieve(url, tmp_zip, _progreso)
        print()

        if not tmp_zip.exists() or tmp_zip.stat().st_size < 1000:
            print(f'   ❌ Archivo descargado muy pequeno o vacio.')
            return False

        # Extraer
        print('   Extrayendo...')
        with zipfile.ZipFile(tmp_zip, 'r') as zf:
            for member in zf.namelist():
                fname = Path(member).name
                if fname and not fname.startswith('.'):
                    dest = DATA_DIR / fname
                    with zf.open(member) as src, open(dest, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
            # Tambien extraer archivos dentro de subdirectorios
            for member in zf.namelist():
                if '/' in member:
                    fname = Path(member).name
                    if fname and not member.startswith('.'):
                        dest = DATA_DIR / fname
                        if not dest.exists():
                            with zf.open(member) as src, open(dest, 'wb') as dst:
                                shutil.copyfileobj(src, dst)

        tmp_zip.unlink()

        # Verificar que tenemos los archivos
        estado = verificar_todos()
        nuevos = sum(1 for v in estado.values() if v)
        print(f'   {nuevos}/{len(estado)} archivos presentes.')
        return nuevos >= 3  # Al menos los criticos (sin GPKG)

    except Exception as e:
        print(f'\n   ❌ Error: {e}')
        if tmp_zip.exists():
            tmp_zip.unlink()
        return False


def descargar_individual(nombre: str, info: dict) -> bool:
    """Descarga un archivo individual desde sus URLs."""
    path = DATA_DIR / nombre
    for url in info['urls']:
        dominio = url.split('/')[2]
        print(f'   Fuente: {dominio}  ({info["tamano"]})')
        tmp = path.with_suffix(path.suffix + '.tmp')
        try:
            urlretrieve(url, tmp, _progreso)
            print()
            tmp.rename(path)
            print(f'   ✅ {nombre}')
            return True
        except Exception as e:
            print(f'\n   Fallo ({dominio}): {e}')
            if tmp.exists():
                tmp.unlink()
    return False


def intentar_descarga():
    """Orquesta la descarga: primero ZIP, luego individuales."""
    estado = verificar_todos()
    completos = sum(1 for v in estado.values() if v)

    if completos == len(ARCHIVOS_REQUERIDOS):
        print('   ✅ Todos los archivos estan listos.\n')
        return True

    print(f'   {completos}/{len(ARCHIVOS_REQUERIDOS)} archivos encontrados.\n')

    # Mostrar que falta
    for nombre, ok in estado.items():
        icono = '✅' if ok else '❌'
        info = FUENTES_INDIVIDUALES.get(nombre, {})
        tam = info.get('tamano', '?')
        desc = info.get('descripcion', nombre)
        print(f'   {icono} {nombre}')
        if not ok:
            print(f'      {desc} ({tam})')
    print()

    # Si falta el GPKG, preguntar (es 1.7 GB)
    falta_gpkg = not estado.get('Carto100000_Colombia_DI_2022.gpkg', False)
    if falta_gpkg:
        print('   ⚠️  El mapa nacional (GPKG) pesa 1.7 GB. Es opcional:')
        print('      sin el, el modelo funciona en Nivel 1-2 (sin Physics-AI).')
        print('      Con el, se extraen municipios, rios y grafos de drenaje.')
        bajar = input('\n   ¿Descargar tambien el GPKG? (s/N): ').strip().lower()
        if bajar != 's':
            ARCHIVOS_REQUERIDOS.remove('Carto100000_Colombia_DI_2022.gpkg')
            del estado['Carto100000_Colombia_DI_2022.gpkg']
            print('   Omitido. Se puede descargar despues.\n')

    print('   Intentando descarga...')

    # Estrategia 1: ZIP completo
    for fuente, url in FUENTES_ZIP:
        print(f'\n   ── {fuente} ──')
        if descargar_zip(fuente, url):
            estado = verificar_todos()
            completos = sum(1 for v in estado.values() if v)
            if completos >= 3:
                print(f'\n   ✅ {completos} archivos descargados via {fuente}.\n')
                return True
            print('   Faltan archivos. Intentando siguiente fuente...\n')

    # Estrategia 2: Individuales (HuggingFace)
    print('\n   ── HuggingFace (archivos individuales) ──')
    faltantes = [n for n, ok in estado.items() if not ok]
    for nombre in faltantes:
        if nombre in FUENTES_INDIVIDUALES:
            print()
            if not descargar_individual(nombre, FUENTES_INDIVIDUALES[nombre]):
                print(f'   ❌ No se pudo descargar {nombre}')

    # Verificar estado final
    estado = verificar_todos()
    completos_final = sum(1 for v in estado.values() if v)

    criticos = ['CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet',
                'SAR_COLOMBIA_MENSUAL_2018_2026.parquet',
                'indices_climaticos_mensuales.parquet']
    criticos_ok = all(estado.get(c, False) for c in criticos)

    if criticos_ok:
        print(f'\n   ✅ Datos minimos listos ({completos_final} archivos).')
        print('   Ejecuta: python main.py')
        return True
    else:
        print(f'\n   ❌ No se pudieron descargar los datos.')
        print(f'   Descargalos manualmente y descomprimi en: {DATA_DIR}/')
        print(f'   Enlace: https://www.mediafire.com/file/atsb5vklle6zecs/datos_globales.zip/file')
        return False


import shutil  # para copyfileobj

if __name__ == '__main__':
    print('📦 FloodModel Colombia — Datos Globales')
    print()
    intentar_descarga()
