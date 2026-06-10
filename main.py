#!/usr/bin/env python3
"""
FloodModel Colombia — Entrenamiento automatico de modelos de inundacion
=========================================================================
Interfaz principal. Ejecutar desde la raiz del proyecto.

Uso:
    python app.py                   # Primero: descargar datos globales
    python main.py                  # Wizard interactivo
    python main.py --nuevo          # Agregar un departamento
    python main.py --depto 05       # Entrenar Antioquia
    python main.py --listar         # Ver departamentos configurados

Requisitos:
    pip install pandas numpy lightgbm scikit-learn pyarrow
"""

import sys, os, json, argparse, shutil, re
from pathlib import Path
import pandas as pd
import unicodedata

BASE_DIR = Path(__file__).resolve().parent
GLOBAL_DIR = BASE_DIR / 'datos_globales'
DEPTOS_DIR = BASE_DIR / 'departamentos'

DATOS_CRITICOS = {
    'CHIRPS': GLOBAL_DIR / 'CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet',
    'SAR':    GLOBAL_DIR / 'SAR_COLOMBIA_MENSUAL_2018_2026.parquet',
    'NOAA':   GLOBAL_DIR / 'indices_climaticos_mensuales.parquet',
}

DEPARTAMENTOS = {
    '05': 'Antioquia', '08': 'Atlantico', '11': 'Bogota D.C.',
    '13': 'Bolivar', '15': 'Boyaca', '17': 'Caldas',
    '18': 'Caqueta', '19': 'Cauca', '20': 'Cesar',
    '23': 'Cordoba', '25': 'Cundinamarca', '27': 'Choco',
    '41': 'Huila', '44': 'La Guajira', '47': 'Magdalena',
    '50': 'Meta', '52': 'Narino', '54': 'Norte de Santander',
    '63': 'Quindio', '66': 'Risaralda', '68': 'Santander',
    '70': 'Sucre', '73': 'Tolima', '76': 'Valle del Cauca',
    '81': 'Arauca', '85': 'Casanare', '86': 'Putumayo',
    '88': 'San Andres', '91': 'Amazonas', '94': 'Guainia',
    '95': 'Guaviare', '97': 'Vaupes', '99': 'Vichada',
}

FORMATO_TARGET = """
El archivo de emergencias debe ser CSV o Parquet con estas columnas
minimas (no importa si estan en mayusculas o minusculas):

   ┌──────────────┬────────────┬──────────────┐
   │ MUNICIPIO    │ FECHA      │ EVENTO       │
   ├──────────────┼────────────┼──────────────┤
   │ MEDELLIN     │ 2022-03-15 │ INUNDACION   │
   │ ZARAGOZA     │ 2025-01-10 │ INUNDACION   │
   │ CAUCASIA     │ 2024-11-02 │ INUNDACION   │
   └──────────────┴────────────┴──────────────┘

   - MUNICIPIO: nombre del municipio (se normaliza solo)
   - FECHA:     YYYY-MM-DD
   - EVENTO:    tipo de evento (se filtra por inundacion)

   Opcional: si el archivo tiene columna DEPARTAMENTO,
   se filtra automaticamente para el departamento actual."""


def strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn')


def titulo(texto):
    print(f'\n{"="*55}')
    print(f'  {texto}')
    print(f'{"="*55}')


# ═══════════════════════════════════════════════════════════════
# WIZARD: AGREGAR DEPARTAMENTO
# ═══════════════════════════════════════════════════════════════

def wizard_nuevo_departamento():
    """Guia paso a paso para configurar un departamento nuevo."""
    titulo('AGREGAR DEPARTAMENTO NUEVO')

    # Verificar datos criticos
    faltan = [n for n, p in DATOS_CRITICOS.items() if not p.exists()]
    if faltan:
        print(f'❌ Faltan datos: {", ".join(faltan)}.')
        print('   Ejecuta primero: python app.py')
        return

    # Paso 1: Elegir departamento
    print('\nDepartamentos de Colombia:')
    regiones = {
        'Andina':    ['05','15','17','19','25','41','52','54','63','66','68','73'],
        'Caribe':    ['08','13','20','23','44','47','70'],
        'Pacifico':  ['27','76'],
        'Orinoquia': ['50','81','85'],
        'Amazonia':  ['18','86','91','94','95','97','99'],
        'Insular':   ['88'],
        'Capital':   ['11'],
    }
    for region, codigos in regiones.items():
        print(f'\n  {region}:')
        for cod in codigos:
            nombre = DEPARTAMENTOS[cod]
            existe = (DEPTOS_DIR / cod / 'config.json').exists()
            tag = '✅' if existe else '  '
            print(f'     {tag} {cod} - {nombre}')

    print()
    codigo = input('Codigo del departamento (ej. 05, 76, 27): ').strip()
    if codigo not in DEPARTAMENTOS:
        print(f'❌ "{codigo}" no es un codigo valido.')
        return
    nombre = DEPARTAMENTOS[codigo]

    # Paso 2: Archivo de emergencias
    titulo(f'PASO 1/3: Archivo de emergencias para {nombre}')
    print(FORMATO_TARGET)

    target_path = input('Ruta al archivo CSV/Parquet (arrastra el archivo a la terminal): ').strip().strip("'\"")
    if not target_path or not Path(target_path).exists():
        print(f'❌ Archivo no encontrado: {target_path}')
        return

    fpath = Path(target_path)
    try:
        if fpath.suffix == '.parquet':
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath, nrows=10)
    except Exception as e:
        print(f'❌ No se pudo leer: {e}')
        return

    # Detectar columnas con regex (no importa mayusculas/minusculas)
    cols_original = list(df.columns)
    col_map = {}  # nombre_original -> nombre_normalizado

    for c in cols_original:
        cu = str(c).upper().strip()
        if re.search(r'MUNICIPIO|MUN\b|NOMBRE.?MUN', cu):
            col_map[c] = 'MUNICIPIO'
        elif re.search(r'FECHA|DATE|FECHA_', cu) and 'FECHA_' not in cu:
            col_map[c] = 'FECHA'
        elif re.search(r'EVENTO|TIPO|SUCESO|AMENAZA|FENOMENO|DESASTRE', cu):
            col_map[c] = 'EVENTO'
        elif re.search(r'DEPARTAMENTO|DPTO|DEP\b|ADM1', cu):
            col_map[c] = 'DEPARTAMENTO'

    tiene_mun = 'MUNICIPIO' in col_map.values()
    tiene_fecha = 'FECHA' in col_map.values()
    tiene_depto = 'DEPARTAMENTO' in col_map.values()

    if not tiene_mun or not tiene_fecha:
        print(f'❌ Columnas encontradas: {cols_original}')
        print('   No se detectaron MUNICIPIO y/o FECHA.')
        print('   Nombres validos: municipio, MUNICIPIO, Municipio, fecha, FECHA, Fecha, date')
        return

    # Renombrar columnas detectadas
    df = df.rename(columns={v: k for k, v in col_map.items()})
    print(f'   ✅ Columnas detectadas: {list(col_map.values())}')
    if tiene_depto:
        print(f'   ✅ Columna DEPARTAMENTO detectada. Se filtrara por "{nombre}".')

    # Paso 3: Crear estructura
    titulo(f'PASO 2/3: Creando estructura para {nombre}')

    depto_dir = DEPTOS_DIR / codigo
    datos_dir = depto_dir / 'datos'
    output_dir = depto_dir / 'output'
    processed_dir = depto_dir / 'processed'
    for d in [datos_dir, output_dir, processed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    dest = datos_dir / f'emergencias_{nombre.lower().replace(" ", "_")}{fpath.suffix}'
    shutil.copy2(fpath, dest)
    print(f'   ✅ Target copiado: {dest.name}')

    # Opcional: extraer vectoriales del GPKG
    gpkg_path = GLOBAL_DIR / 'Carto100000_Colombia_DI_2022.gpkg'
    if gpkg_path.exists():
        print()
        extraer = input('   ¿Extraer municipios y drenaje del mapa nacional? (s/N): ').strip().lower()
        if extraer == 's':
            sys.path.insert(0, str(BASE_DIR))
            from core.extractor_geopackage import extraer_departamento
            extraer_departamento(str(gpkg_path), codigo, nombre, output_dir=processed_dir)
            print('   ✅ Datos vectoriales extraidos.')
        else:
            print('   Omitido. Se puede extraer despues con:')
            print(f'   python core/extractor_geopackage.py {codigo}')

    # Paso 4: Municipios ciegos (opcional)
    titulo(f'PASO 3/3: Configuracion final')
    print(f'   ✅ {nombre} configurado en: {depto_dir}')
    print()
    print('   Si hay municipios donde el radar Sentinel-1 no funciona')
    print('   (zonas con mucha vegetacion o bosque denso), anotalos aqui.')
    print('   El pipeline tambien puede detectarlos automaticamente.')
    ciegos_str = input('   Nombres separados por coma (o Enter para omitir): ').strip()
    ciegos = [m.strip().upper() for m in ciegos_str.split(',') if m.strip()] if ciegos_str else []

    # Guardar config
    config = {
        'nombre': nombre,
        'codigo_departamento': codigo,
        'municipios_ciegos': ciegos,
        'chirps_path': str(DATOS_CRITICOS['CHIRPS']),
        'sar_path': str(DATOS_CRITICOS['SAR']),
        'noaa_path': str(DATOS_CRITICOS['NOAA']),
        'hand_path': str(processed_dir / 'hand_municipio.csv'),
        'flow_path': str(processed_dir / 'flow_accum_buffer_30m.csv'),
        'grafo_path': str(processed_dir / 'grafo_drenaje.json'),
        'target_path': str(dest),
        'target_column': 'flood_target',
        'target_from_dataset': True,
        'target_tiene_depto': tiene_depto,
        'output_dir': str(output_dir),
        'tiene_chirps': True,
        'tiene_sar': True,
        'tiene_dem_30m': (processed_dir / 'hand_municipio.csv').exists(),
        'tiene_hand': (processed_dir / 'hand_municipio.csv').exists(),
        'tiene_grafo_drenaje': (processed_dir / 'grafo_drenaje.json').exists(),
        'tiene_alos2': False,
        'tiene_gauges_tiempo_real': False,
        'tiene_target': True,
        'n_municipios': 0,
    }

    with open(depto_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print()
    print(f'{"="*55}')
    print(f'✅ {nombre} listo para entrenar.')
    print(f'   python main.py --depto {codigo}')
    print(f'{"="*55}')


# ═══════════════════════════════════════════════════════════════
# ENTRENAR
# ═══════════════════════════════════════════════════════════════

def entrenar_departamento(codigo: str):
    """Entrena el modelo para un departamento ya configurado."""
    depto_dir = DEPTOS_DIR / codigo
    config_path = depto_dir / 'config.json'

    if not config_path.exists():
        print(f'❌ {DEPARTAMENTOS.get(codigo, "?")} no esta configurado.')
        print(f'   Ejecuta: python main.py --nuevo')
        return

    with open(config_path) as f:
        config = json.load(f)

    # Actualizar rutas de datos globales (por si se movio el proyecto)
    for key, nombre in [('chirps_path', 'CHIRPS'), ('sar_path', 'SAR'), ('noaa_path', 'NOAA')]:
        if key in config:
            p = Path(config[key])
            if not p.exists() and DATOS_CRITICOS[nombre].exists():
                config[key] = str(DATOS_CRITICOS[nombre])

    # Verificar datos criticos
    for key in ['chirps_path', 'sar_path', 'noaa_path']:
        if key in config and not Path(config[key]).exists():
            print(f'❌ Falta {key}. Ejecuta: python app.py')
            return

    sys.path.insert(0, str(BASE_DIR))
    from core.pipeline import entrenar

    resultados = entrenar(config)
    if resultados.get('status') == 'success':
        t = resultados.get('test', {})
        print(f'\n✅ {config["nombre"]} entrenado.')
        print(f'   AUROC: {t.get("auroc", 0):.4f}')
        print(f'   F1:    {t.get("f1", 0):.4f}')
        print(f'   Modelo: {resultados.get("model_path")}')
        if resultados.get('physics_ai'):
            p = resultados['physics_ai']
            print(f'   Physics-AI: {resultados.get("physics_model_path")}')
            if 'ciegos' in p:
                print(f'   Municipios ciegos: {p["ciegos"].get("tp",0)}/{p["ciegos"].get("total",0)} detectados')
    else:
        print(f'❌ {resultados.get("reason", "fallo")}')


def listar_departamentos():
    """Muestra departamentos configurados."""
    configurados = []
    for cod in sorted(DEPARTAMENTOS):
        if (DEPTOS_DIR / cod / 'config.json').exists():
            configurados.append(cod)

    print(f'\nDepartamentos configurados: {len(configurados)}/32')
    if configurados:
        for c in configurados:
            print(f'   ✅ {c} - {DEPARTAMENTOS[c]}')
    else:
        print('   (ninguno)')
        print(f'\nEjecuta: python main.py --nuevo')


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FloodModel Colombia — Prediccion de Inundaciones'
    )
    parser.add_argument('--depto', type=str, help='Codigo departamento (ej. 05, 76)')
    parser.add_argument('--nuevo', action='store_true', help='Agregar un departamento (wizard)')
    parser.add_argument('--listar', action='store_true', help='Listar departamentos configurados')
    args = parser.parse_args()

    if args.listar:
        listar_departamentos()
    elif args.nuevo:
        wizard_nuevo_departamento()
    elif args.depto:
        entrenar_departamento(args.depto)
    else:
        # Modo interactivo
        print('🌊' * 20)
        print('   FloodModel Colombia')
        print('🌊' * 20)

        # Verificar datos globales
        faltan = [n for n, p in DATOS_CRITICOS.items() if not p.exists()]
        if faltan:
            print(f'\n❌ Faltan datos globales: {", ".join(faltan)}')
            print('   Ejecuta primero: python app.py')
            sys.exit(1)

        # Verificar departamentos configurados
        configurados = []
        for cod in DEPARTAMENTOS:
            if (DEPTOS_DIR / cod / 'config.json').exists():
                configurados.append(cod)

        if not configurados:
            print('\n🎯 No hay ningun departamento configurado.')
            print()
            print('   Opciones:')
            print('   1. Agregar un departamento (wizard guiado)')
            print('   2. Salir')
            choice = input('\n   Opcion [1]: ').strip() or '1'
            if choice == '1':
                wizard_nuevo_departamento()
        else:
            print(f'\n   {len(configurados)} departamento(s) configurado(s):')
            for c in configurados:
                print(f'   ✅ {c} - {DEPARTAMENTOS[c]}')
            print()
            print('   Opciones:')
            print('   1. Entrenar modelo')
            print('   2. Agregar otro departamento')
            print('   3. Salir')
            choice = input('\n   Opcion [1]: ').strip() or '1'
            if choice == '1':
                cod = input(f'   Codigo [{", ".join(configurados)}]: ').strip()
                if cod in configurados:
                    entrenar_departamento(cod)
                else:
                    print(f'   ❌ "{cod}" no esta configurado.')
            elif choice == '2':
                wizard_nuevo_departamento()
