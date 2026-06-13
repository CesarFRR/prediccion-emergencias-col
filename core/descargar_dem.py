"""
Descarga DEM SRTM 30m por departamento
========================================
Usa elevation (gratis, sin API key) como metodo principal.
OpenTopography como alternativa.

Requiere: pip install elevation rasterio geopandas

Uso:
    python core/descargar_dem.py 05    # Antioquia
    python core/descargar_dem.py 88    # San Andres (prueba rapida)
    python core/descargar_dem.py 76    # Valle del Cauca
"""

import sys, os
from pathlib import Path
import tempfile
import geopandas as gpd
import rasterio

BASE_DIR = Path(__file__).resolve().parent.parent
DEPTOS_DIR = BASE_DIR / 'departamentos'
GPKG_PATH = BASE_DIR / 'datos_globales' / 'Carto100000_Colombia_DI_2022.gpkg'


def _descargar_por_cuadrantes(west, south, east, north, output_path, product='SRTM1'):
    """Divide un bbox grande en cuadrantes de ~2° y descarga cada uno."""
    import elevation as _elv
    chunks = []
    step = 2.0
    w = west
    while w < east:
        s = south
        while s < north:
            w2 = min(w + step, east)
            s2 = min(s + step, north)
            fname = output_path.parent / f'_chunk_{w:.1f}_{s:.1f}.tif'
            if not fname.exists():
                try:
                    _elv.clip(bounds=(w, s, w2, s2), output=str(fname), product=product)
                    chunks.append(fname)
                    print(f'      Chunk [{w:.1f},{s:.1f}] OK')
                except Exception as e:
                    print(f'      Chunk [{w:.1f},{s:.1f}] fallo: {e}')
            else:
                chunks.append(fname)
            s = s2
        w = w2

    # Fusionar chunks
    if len(chunks) > 1:
        print(f'   Fusionando {len(chunks)} chunks...')
        datasets = [rasterio.open(c) for c in chunks if c.exists()]
        if datasets:
            from rasterio.merge import merge as _merge
            mosaic, transform = _merge(datasets)
            meta = datasets[0].meta.copy()
            meta.update(driver='GTiff', height=mosaic.shape[1], width=mosaic.shape[2],
                       transform=transform, compress='lzw', predictor=2)
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(mosaic)
            for ds in datasets:
                ds.close()
            # Limpiar chunks
            for c in chunks:
                c.unlink(missing_ok=True)
    elif len(chunks) == 1 and chunks[0].exists():
        chunks[0].rename(output_path)

    return chunks


def _merge_chunks(chunks, output_path):
    """Fusiona una lista de archivos GeoTIFF en uno solo."""
    if not chunks:
        return
    if len(chunks) == 1:
        chunks[0].rename(output_path)
        return
    from rasterio.merge import merge as _merge
    datasets = [rasterio.open(c) for c in chunks if c.exists()]
    if datasets:
        mosaic, transform = _merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update(driver='GTiff', height=mosaic.shape[1], width=mosaic.shape[2],
                   transform=transform, compress='lzw', predictor=2)
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mosaic)
        for ds in datasets:
            ds.close()
        for c in chunks:
            c.unlink(missing_ok=True)


def descargar_dem_depto(codigo: str, output_dir: Path = None):
    """Descarga DEM SRTM 30m para un departamento."""
    depto_dir = DEPTOS_DIR / codigo
    processed_dir = output_dir or depto_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    dem_path = processed_dir / 'dem_srtm30.tif'

    if dem_path.exists():
        print(f'✅ DEM ya existe: {dem_path}')
        return dem_path

    bbox = _obtener_bbox(codigo)
    if bbox is None:
        print('❌ No se pudo obtener el bounding box.')
        return None

    west, south, east, north = bbox
    print(f'   BBox: [{west:.2f}, {south:.2f}, {east:.2f}, {north:.2f}]')
    print('   Descargando SRTM 30m (elevation)...')

    try:
        import elevation as _elv
        from rasterio.merge import merge as _merge

        # elevation limita a ~12 tiles. Si el bbox es muy grande, partir en sub-bboxes.
        width = east - west
        height = north - south
        if width > 3.0 or height > 3.0:
            print(f'   Departamento grande ({width:.1f}°x{height:.1f}°), partiendo en sub-bloques...')
            chunks = _descargar_por_cuadrantes(west, south, east, north, dem_path, product='SRTM1')
        else:
            _elv.clip(bounds=(west, south, east, north), output=str(dem_path), product='SRTM1')
            chunks = [dem_path] if dem_path.exists() else []
        
        if not dem_path.exists() and chunks:
            _merge_chunks(chunks, dem_path)
    except Exception as e:
        print(f'   Elevation fallo: {e}')
        print('   Probando OpenTopography...')
        api_key = os.environ.get('OT_API_KEY', '')
        if not api_key:
            print('   ❌ Se necesita OT_API_KEY. Registrate en https://portal.opentopography.org/')
            print('      export OT_API_KEY=tu_clave')
            return None
        return _descargar_opentopo(west, south, east, north, dem_path, api_key)

    if not dem_path.exists():
        print('   ❌ No se genero el archivo.')
        return None

    with rasterio.open(dem_path) as src:
        mb = dem_path.stat().st_size / 1e6
        print(f'   ✅ {src.width}x{src.height} px, {mb:.1f} MB, CRS={src.crs}')

    return dem_path


def _descargar_opentopo(west, south, east, north, dem_path, api_key):
    """Descarga via OpenTopography con API key."""
    import requests
    params = {
        'demtype': 'SRTMGL1',
        'west': west, 'south': south, 'east': east, 'north': north,
        'outputFormat': 'GTiff', 'API_Key': api_key,
    }
    print('   Descargando de OpenTopography...')
    try:
        r = requests.get('https://portal.opentopography.org/API/globaldem',
                         params=params, stream=True, timeout=600)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        chunks = []
        for chunk in r.iter_content(chunk_size=8192):
            chunks.append(chunk)
            if total > 0:
                mb = len(b''.join(chunks)) / 1_048_576
                total_mb = total / 1_048_576
                print(f'\r   {int(mb/total_mb*100):3d}%  {mb:.0f}/{total_mb:.0f} MB', end='', flush=True)
        print()
        with open(dem_path, 'wb') as f:
            f.write(b''.join(chunks))
        return dem_path
    except Exception as e:
        print(f'\n   ❌ Error: {e}')
        return None


def _obtener_bbox(codigo: str):
    """Obtiene el bounding box en EPSG:4326 del departamento."""
    if GPKG_PATH.exists():
        try:
            deptos = gpd.read_file(GPKG_PATH, layer='Limite_Departamental')
            depto = deptos[deptos['DeCodigo'] == codigo]
            if len(depto) > 0:
                if depto.crs and str(depto.crs) != 'EPSG:4326':
                    depto = depto.to_crs('EPSG:4326')
                return tuple(depto.total_bounds)
        except Exception:
            pass

    # Fallback: bounding boxes predefinidos
    bboxes = {
        '05': (-77.15,  5.26, -73.69,  8.93),
        '88': (-81.75, 12.45, -81.65, 12.60),
        '76': (-77.50,  3.00, -75.60,  5.00),
        '27': (-77.90,  4.00, -76.20,  8.60),
    }
    if codigo in bboxes:
        return bboxes[codigo]

    return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Uso: python core/descargar_dem.py CODIGO')
        print('Ej:   python core/descargar_dem.py 05   # Antioquia')
        print('      python core/descargar_dem.py 88   # San Andres')
        sys.exit(1)

    codigo = sys.argv[1]
    path = descargar_dem_depto(codigo)
    if path:
        print(f'\n✅ DEM: {path}')
    else:
        print(f'\n❌ No se pudo descargar.')
        print(f'   Alternativa: export OT_API_KEY=tu_clave && python core/descargar_dem.py {codigo}')
