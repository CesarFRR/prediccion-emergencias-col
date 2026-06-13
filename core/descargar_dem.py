"""
Descarga DEM SRTM 30m por departamento
========================================
Usa OpenTopography API (gratis, sin token para SRTM global).
Descarga tiles SRTM GL1 (30m) que cubren el bounding box del
departamento y los fusiona en un solo GeoTIFF.

Uso:
    python core/descargar_dem.py 05    # Antioquia
    python core/descargar_dem.py 76    # Valle del Cauca
    python core/descargar_dem.py 27    # Choco

Fuente: SRTM GL1 via OpenTopography (https://opentopography.org/)
"""

import sys, os
from pathlib import Path
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.io import MemoryFile
import requests

# ─── CONFIG ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DEPTOS_DIR = BASE_DIR / 'departamentos'
GPKG_PATH = BASE_DIR / 'datos_globales' / 'Carto100000_Colombia_DI_2022.gpkg'
OT_API = 'https://portal.opentopography.org/API/globaldem'


def descargar_dem_depto(codigo: str, output_dir: Path = None):
    """
    Descarga DEM SRTM 30m para un departamento usando su GeoJSON de municipios
    o extrayendo el limite del GPKG.
    """
    depto_dir = DEPTOS_DIR / codigo
    processed_dir = output_dir or depto_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    dem_path = processed_dir / 'dem_srtm30.tif'

    if dem_path.exists():
        print(f'✅ DEM ya existe: {dem_path}')
        return dem_path

    # 1. Obtener bounding box del departamento
    bbox = _obtener_bbox(codigo)
    if bbox is None:
        print('❌ No se pudo obtener el bounding box.')
        return None

    west, south, east, north = bbox
    print(f'   BBox: [{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]')
    print()

    # 2. Descargar via OpenTopography
    params = {
        'demtype': 'SRTMGL1',
        'west': west,
        'south': south,
        'east': east,
        'north': north,
        'outputFormat': 'GTiff',
    }

    print('Descargando de OpenTopography...')
    try:
        r = requests.get(OT_API, params=params, stream=True, timeout=600)
        r.raise_for_status()

        total = int(r.headers.get('content-length', 0))
        descargado = 0
        chunks = []
        for chunk in r.iter_content(chunk_size=8192):
            chunks.append(chunk)
            descargado += len(chunk)
            if total > 0:
                pct = min(100, descargado * 100 // total)
                mb = descargado / 1_048_576
                total_mb = total / 1_048_576
                print(f'\r   {pct:3d}%  {mb:.0f}/{total_mb:.0f} MB', end='', flush=True)

        print()
        data = b''.join(chunks)

        if len(data) < 1000:
            print(f'   ❌ Respuesta muy pequena ({len(data)} bytes).')
            print(f'   Posiblemente OpenTopography no tiene datos para esta zona.')
            return None

        with open(dem_path, 'wb') as f:
            f.write(data)

        # Verificar
        with rasterio.open(dem_path) as src:
            print(f'   ✅ DEM: {src.width}x{src.height} px, CRS={src.crs}')
            print(f'   Tamano: {dem_path.stat().st_size/1e6:.1f} MB')

        return dem_path

    except Exception as e:
        print(f'\n   ❌ Error: {e}')
        if dem_path.exists():
            dem_path.unlink()
        return None


def _obtener_bbox(codigo: str):
    """Obtiene el bounding box del departamento desde el GPKG o GeoJSON local."""
    # 1. Intentar desde GPKG nacional
    if GPKG_PATH.exists():
        try:
            deptos = gpd.read_file(GPKG_PATH, layer='Limite_Departamental')
            depto = deptos[deptos['DeCodigo'] == codigo]
            if len(depto) > 0:
                return tuple(depto.total_bounds)
        except Exception:
            pass

    # 2. Intentar desde GeoJSON de municipios del departamento
    geo_dir = DEPTOS_DIR / codigo / 'processed'
    for geojson in geo_dir.glob('municipios_*.geojson'):
        try:
            gdf = gpd.read_file(geojson)
            return tuple(gdf.total_bounds)
        except Exception:
            pass
    for geojson in (DEPTOS_DIR / codigo / 'datos').glob('*.geojson'):
        try:
            gdf = gpd.read_file(geojson)
            return tuple(gdf.total_bounds)
        except Exception:
            pass

    # 3. Bounding boxes predefinidos para los 32 departamentos (fallback)
    BBOXES = {
        '05': (-77.15,  5.26, -73.69,  8.93),   # Antioquia
        '08': (-75.10, 10.20, -74.70, 11.10),   # Atlantico
        '11': (-74.40,  4.30, -73.90,  4.90),   # Bogota
        '76': (-77.50,  3.00, -75.60,  5.00),   # Valle del Cauca
        '27': (-77.90,  4.00, -76.20,  8.60),   # Choco
        '25': (-75.00,  3.70, -73.20,  5.80),   # Cundinamarca
        '17': (-75.80,  4.80, -74.60,  5.80),   # Caldas
        '68': (-74.50,  5.60, -72.50,  8.20),   # Santander
        '52': (-78.80,  0.30, -76.80,  2.60),   # Narino
    }
    if codigo in BBOXES:
        return BBOXES[codigo]

    return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python core/descargar_dem.py CODIGO')
        print('Ej:   python core/descargar_dem.py 05   # Antioquia')
        sys.exit(1)

    codigo = sys.argv[1]
    path = descargar_dem_depto(codigo)
    if path:
        print(f'\n✅ DEM guardado en: {path}')
    else:
        print(f'\n❌ No se pudo descargar.')
        print(f'   Alternativa manual: https://earthexplorer.usgs.gov/')
        print(f'   Descargar tiles SRTM 1-Arc-Second y fusionar con:')
        print(f'   gdal_merge.py -o dem_{codigo}.tif tile1.tif tile2.tif ...')
