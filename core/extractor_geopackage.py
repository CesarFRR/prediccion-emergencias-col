"""
Extractor de Datos Vectoriales — Desde Carto100000 IGAC (GPKG)
================================================================
Toma el GeoPackage nacional y extrae por departamento:
  1. GeoJSON de municipios (simplificado, <500KB)
  2. Red de drenaje (ríos y quebradas)
  3. Límite departamental

Uso:
    python FASE-4/core/extractor_geopackage.py 05    # Antioquia
    python FASE-4/core/extractor_geopackage.py 76    # Valle del Cauca
    python FASE-4/core/extractor_geopackage.py --all # Los 32 departamentos
"""

import geopandas as gpd
import pandas as pd
from pathlib import Path
import sys, json

# ─── CONFIG ────────────────────────────────────────────────────
GPKG_PATH = 'PROYECTO ANTIOQUIA/raw/mapa vectorial colombia geopackage/Carto100000_Colombia_DI_2022.gpkg'
OUT_DIR = Path('FASE-4/datos_vectoriales')

# Mapeo: código DIVIPOLA depto → nombre
DEPTO_NAMES = {}

# ─── FUNCIONES ─────────────────────────────────────────────────
def cargar_nombres_departamentos(gpkg: str) -> dict:
    """Carga mapeo de códigos a nombres desde Limite_Departamental."""
    deptos = gpd.read_file(gpkg, layer='Limite_Departamental')
    mapping = {}
    for _, row in deptos.iterrows():
        mapping[row['DeCodigo']] = row['DeNombre']
    return mapping


def extraer_departamento(gpkg: str, cod_depto: str, nombre_depto: str = None,
                         simplify_municipios: float = 0.02,
                         simplify_drenaje: float = 0.005,
                         output_dir: Path = None):
    """
    Extrae todos los datos vectoriales para un departamento.

    Args:
        gpkg: ruta al GeoPackage nacional
        cod_depto: código DIVIPOLA del depto (2 dígitos, ej. '05')
        nombre_depto: nombre (si no se da, se busca en Limite_Departamental)
        simplify_tolerance: tolerancia Douglas-Peucker (grados, 0.005 ≈ 500m)
        output_dir: directorio de salida (si None, OUT_DIR / cod_depto)
    """
    if output_dir is None:
        output_dir = OUT_DIR / cod_depto
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f' Extrayendo depto {cod_depto}...')

    # 1. Municipios
    municipios = gpd.read_file(gpkg, layer='Limite_Municipal_Poligono')
    mun_depto = municipios[municipios['MpCodigo'].str[:2] == cod_depto].copy()

    if len(mun_depto) == 0:
        print(f'   WARNING: Cero municipios para código {cod_depto}')
        return

    # Simplificar geometría (agresivo para GEE: 0.02° ≈ 2km, suficiente para promedios espaciales)
    mun_depto['geometry'] = mun_depto['geometry'].simplify(
        tolerance=simplify_municipios, preserve_topology=True
    )

    # Guardar GeoJSON ligero
    geojson_path = output_dir / f'municipios_{cod_depto}.geojson'
    mun_depto.to_file(geojson_path, driver='GeoJSON')
    size_kb = geojson_path.stat().st_size / 1024
    print(f'   {len(mun_depto)} municipios → {geojson_path.name} ({size_kb:.0f} KB)')

    # 2. Límite departamental (para GEE, recortes)
    deptos = gpd.read_file(gpkg, layer='Limite_Departamental')
    limite = deptos[deptos['DeCodigo'] == cod_depto].copy()
    if len(limite) > 0:
        limite['geometry'] = limite['geometry'].simplify(tolerance=simplify_municipios, preserve_topology=True)
        limite_path = output_dir / f'limite_{cod_depto}.geojson'
        limite.to_file(limite_path, driver='GeoJSON')
        depto_nombre = limite.iloc[0]['DeNombre'] if nombre_depto is None else nombre_depto
        print(f'   Límite departamental → {limite_path.name}')

        # Unir geometrías de municipios para polígono del depto (para GEE)
        depto_poly = mun_depto.unary_union
        depto_gdf = gpd.GeoDataFrame(geometry=[depto_poly], crs=mun_depto.crs)
        depto_gdf = gpd.GeoDataFrame({
            'codigo': [cod_depto],
            'nombre': [depto_nombre],
            'geometry': depto_gdf.geometry.simplify(tolerance=simplify_municipios/2, preserve_topology=True)
        }, crs=mun_depto.crs)
        depto_poly_path = output_dir / f'departamento_{cod_depto}.geojson'
        depto_gdf.to_file(depto_poly_path, driver='GeoJSON')
        print(f'   Polígono depto unificado → {depto_poly_path.name}')
    else:
        depto_nombre = nombre_depto or f'Depto_{cod_depto}'

    # 3. Drenaje (recortado al depto)
    drenaje = gpd.read_file(gpkg, layer='Drenaje_Sencillo')
    # Recortar al bounding box del depto para reducir antes del clip espacial
    bbox = mun_depto.total_bounds  # [minx, miny, maxx, maxy]
    dren_bbox = drenaje.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].copy()
    if len(dren_bbox) > 0:
        dren_bbox['geometry'] = dren_bbox['geometry'].simplify(
            tolerance=simplify_drenaje, preserve_topology=True
        )
        dren_path = output_dir / f'drenaje_{cod_depto}.geojson'
        dren_bbox.to_file(dren_path, driver='GeoJSON')
        size_kb = dren_path.stat().st_size / 1024
        print(f'   Drenaje: {len(dren_bbox)} segmentos → {dren_path.name} ({size_kb:.0f} KB)')
    else:
        print(f'   WARNING: Sin segmentos de drenaje en el bbox')

    return {
        'codigo': cod_depto,
        'nombre': depto_nombre,
        'n_municipios': len(mun_depto),
        'output_dir': str(output_dir),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    DEPTO_NAMES = cargar_nombres_departamentos(GPKG_PATH)

    if '--all' in sys.argv:
        print(f' Extrayendo TODOS los departamentos...\n')
        for cod, nombre in sorted(DEPTO_NAMES.items()):
            if cod == '00':  # Skip litigio
                continue
            print(f'\n{'='*50}')
            extraer_departamento(GPKG_PATH, cod, nombre)
        print(f'\n Todos extraídos en: {OUT_DIR}/')
    elif len(sys.argv) > 1:
        cod = sys.argv[1]
        nombre = DEPTO_NAMES.get(cod, f'Depto_{cod}')
        extraer_departamento(GPKG_PATH, cod, nombre)
    else:
        print('Uso:')
        print('  python FASE-4/core/extractor_geopackage.py 05       # Antioquia')
        print('  python FASE-4/core/extractor_geopackage.py 76       # Valle del Cauca')
        print('  python FASE-4/core/extractor_geopackage.py --all    # Los 32 deptos')
        print()
        print('Departamentos disponibles:')
        for cod, nombre in sorted(DEPTO_NAMES.items()):
            if cod != '00':
                print(f'  {cod} = {nombre}')
