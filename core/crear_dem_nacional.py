#!/usr/bin/env python3
"""
Descarga y fusiona el DEM30 nacional de Colombia.
==================================================
Ejecutar UNA SOLA VEZ. Descarga los 32 DEM departamentales,
los fusiona en un solo GeoTIFF nacional comprimido.

Tiempo estimado: 30-60 minutos (depende de internet)
Tamano final:    ~300-500 MB (comprimido LZW)

Uso:
    python core/crear_dem_nacional.py
"""

import sys
from pathlib import Path
import rasterio
from rasterio.merge import merge

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'datos_globales'
DEM_NACIONAL = DATA_DIR / 'dem_srtm30_colombia.tif'

DEPARTAMENTOS = {
    '05':'Antioquia','08':'Atlantico','11':'Bogota D.C.','13':'Bolivar','15':'Boyaca',
    '17':'Caldas','18':'Caqueta','19':'Cauca','20':'Cesar','23':'Cordoba',
    '25':'Cundinamarca','27':'Choco','41':'Huila','44':'La Guajira','47':'Magdalena',
    '50':'Meta','52':'Narino','54':'Norte de Santander','63':'Quindio','66':'Risaralda',
    '68':'Santander','70':'Sucre','73':'Tolima','76':'Valle del Cauca','81':'Arauca',
    '85':'Casanare','86':'Putumayo','88':'San Andres','91':'Amazonas','94':'Guainia',
    '95':'Guaviare','97':'Vaupes','99':'Vichada',
}

sys.path.insert(0, str(BASE_DIR))
from core.descargar_dem import descargar_dem_depto


def crear_dem_nacional():
    print('🇨🇴 Creando DEM30 nacional de Colombia')
    print(f'   {len(DEPARTAMENTOS)} departamentos\n')

    # 1. Descargar (o verificar) cada departamento
    dems = {}
    ok = 0
    for cod, nombre in sorted(DEPARTAMENTOS.items()):
        try:
            path = descargar_dem_depto(cod)
            if path:
                dems[cod] = path
                ok += 1
                print(f'   ✅ {cod} {nombre}')
            else:
                print(f'   ❌ {cod} {nombre}')
        except Exception as e:
            print(f'   ❌ {cod} {nombre}: {e}')

    print(f'\n   {ok}/{len(DEPARTAMENTOS)} departamentos listos.')

    if ok < 30:
        print(f'   ⚠️  Faltan departamentos. Revisa los errores arriba.')
        return None

    # 2. Fusionar
    print(f'\n   Fusionando {ok} archivos...')
    datasets = []
    for cod, path in dems.items():
        if path and path.exists():
            datasets.append(rasterio.open(path))

    mosaic, transform = merge(datasets, method='first')
    out_meta = datasets[0].meta.copy()
    out_meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': transform,
        'compress': 'lzw',
        'predictor': 2,
        'BIGTIFF': 'YES',
    })

    with rasterio.open(DEM_NACIONAL, 'w', **out_meta) as dst:
        dst.write(mosaic)

    for ds in datasets:
        ds.close()

    mb = DEM_NACIONAL.stat().st_size / 1e6
    print(f'   ✅ DEM nacional: {mb:.0f} MB')
    print(f'   {DEM_NACIONAL}')
    return DEM_NACIONAL


if __name__ == '__main__':
    try:
        crear_dem_nacional()
    except KeyboardInterrupt:
        print('\n   Cancelado.')
