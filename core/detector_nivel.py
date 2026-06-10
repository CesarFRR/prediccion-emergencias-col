"""
FASE-4: Fábrica de Modelos de Inundación por Departamento
==========================================================
Núcleo compartido. Cada departamento solo necesita su config.py + GeoJSON.

El detector determina automáticamente qué nivel de datos tiene
el departamento y selecciona features + modelo + estrategia.

Decision tree:
    START
    ├── ¿target de inundaciones?       → NO: ❌ Imposible
    ├── ¿CHIRPS diario?                → NO: ❌ Sin precip
    ├── ¿SAR Sentinel-1 tabular?       → NO: Nivel 1
    ├── ¿DEM 30m + HAND + grafo?       → NO: Nivel 2
    ├── ¿ALOS-2 + gauges tiempo real?  → NO: Nivel 3
    └── SÍ a todo                      → Nivel 4
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional


@dataclass
class NivelDatos:
    """Resultado del detector: qué nivel y qué capacidades tiene el depto."""
    nivel: int               # 1-4 (0 = imposible)
    label: str               # "Nivel 2: Datos moderados"
    tiene_chirps: bool = False
    tiene_sar: bool = False
    tiene_dem: bool = False
    tiene_hand: bool = False
    tiene_grafo: bool = False
    tiene_alos2: bool = False
    tiene_gauges: bool = False
    n_municipios: int = 0
    puede_physics_ai: bool = False
    puede_optuna: bool = False
    puede_gnn: bool = False
    features_activas: List[str] = field(default_factory=list)
    features_prohibidas: Set[str] = field(default_factory=set)
    warnings: List[str] = field(default_factory=list)


def detectar_nivel(config: dict) -> NivelDatos:
    """
    Árbol de decisión: determina el nivel a partir de la config del depto.

    config debe tener:
        tiene_chirps: bool
        tiene_sar: bool
        tiene_dem_30m: bool
        tiene_hand: bool
        tiene_grafo_drenaje: bool
        tiene_alos2: bool
        tiene_gauges_tiempo_real: bool
        tiene_target: bool
        n_municipios: int
    """
    w = []

    # Gate 0: sin target no se puede entrenar
    if not config.get('tiene_target', False):
        return NivelDatos(nivel=0, label='IMPOSIBLE: Sin target de inundaciones',
                          warnings=['Necesitás reportes de emergencia (UNGRD/DAGRAN)'])

    # Gate 1: sin CHIRPS no hay predicción de lluvia
    if not config.get('tiene_chirps', False):
        return NivelDatos(nivel=0, label='IMPOSIBLE: Sin datos de precipitación',
                          warnings=['CHIRPS es gratuito y global. Descargalo vía GEE o ClimateSERV.'])

    # ─── NIVEL 1: Solo CHIRPS + topo básica ───
    nivel = 1
    features = [
        'chirps_precip_mm_dia', 'precip_acum_3d', 'precip_acum_7d',
        'precip_acum_15d', 'precip_acum_30d', 'p90_precip_3d',
    ]
    prohibidas = set()
    puede_physics = False
    puede_optuna = False
    puede_gnn = False

    tiene_sar = config.get('tiene_sar', False)
    tiene_dem = config.get('tiene_dem_30m', False)
    tiene_hand = config.get('tiene_hand', False)
    tiene_grafo = config.get('tiene_grafo_drenaje', False)
    tiene_alos2 = config.get('tiene_alos2', False)
    tiene_gauges = config.get('tiene_gauges_tiempo_real', False)
    n_mun = config.get('n_municipios', 0)

    # ─── NIVEL 2: + SAR + NOAA ───
    if tiene_sar:
        nivel = 2
        features += [
            'VV_mean', 'VH_mean', 'VV_minus_VH', 'z_VV_mean', 'n_scenes',
            'SOI', 'QBO30', 'QBO50', 'ZWND200',
            'N12_ANOM', 'N3_ANOM', 'N4_ANOM', 'oni_anom', 'oni_total',
            'humedad_media', 'viento_medio', 'temperatura_media', 'temperatura_max',
            'pct_inundable', 'river_length_km', 'river_density_km_per_km2', 'river_area_pct',
        ]
        puede_optuna = True
        puede_gnn = (n_mun >= 500)  # Solo si hay suficientes nodos
    else:
        w.append('Sin SAR Sentinel-1. ¿El depto tiene cobertura de GEE? Si sí, extraé features tabulares.')

    # ─── NIVEL 3: + DEM 30m + HAND + grafo ───
    if tiene_dem and tiene_hand and tiene_grafo:
        nivel = 3
        features += [
            'slope_mean', 'slope_max', 'slope_p90',
            'twi_mean', 'twi_max', 'twi_p90', 'twi_p99',
            'acc_mean', 'acc_max', 'acc_p90', 'acc_p99',
            'hand_m', 'max_flow_acc_buffer', 'p95_flow_acc_buffer', 'p99_flow_acc_buffer',
            'precip_aguas_arriba_3d', 'riesgo_fisico',
        ]
        puede_physics = True
    else:
        faltantes = []
        if not tiene_dem: faltantes.append('DEM 30m')
        if not tiene_hand: faltantes.append('HAND')
        if not tiene_grafo: faltantes.append('grafo drenaje')
        w.append(f'Sin {", ".join(faltantes)}. Physics-AI requiere los 3. Sin esto, los municipios con dosel forestal tendrán recall=0.')

    # ─── NIVEL 4: + ALOS-2 + gauges tiempo real ───
    if tiene_alos2 and tiene_gauges:
        nivel = 4
        # Nivel 4 añade features de Banda L + gauge fusion
        features += ['alos2_HH_mean', 'alos2_HV_mean', 'gauge_precip_fusion']
        puede_gnn = (n_mun >= 500)  # Re-evaluar con más nodos
    else:
        if not tiene_alos2:
            w.append('Sin ALOS-2 Banda L. Los municipios bajo dosel dependen 100% de Physics-AI.')
        if not tiene_gauges:
            w.append('Sin gauges en tiempo real. El forecast depende solo de CHIRPS (~5km).')

    # ─── Ajustes por número de municipios ───
    if n_mun < 30:
        w.append(f'Solo {n_mun} municipios. Dataset pequeño — monitorear sobreajuste.')
        puede_optuna = False  # Muy pocos datos para 50 trials
    if n_mun < 500:
        puede_gnn = False

    # ─── Features estáticas (siempre disponibles si se configuran) ───
    static_features = [
        'elevacion_msnm', 'pendiente_val', 'distancia_rio_km_real',
    ]
    for sf in static_features:
        if sf not in features:
            features.insert(6, sf)  # Insertar después de las de precip

    return NivelDatos(
        nivel=nivel,
        label=f'Nivel {nivel}: {"Premium" if nivel==4 else "Datos ricos" if nivel==3 else "Datos moderados" if nivel==2 else "Datos básicos"}',
        tiene_chirps=True,
        tiene_sar=tiene_sar,
        tiene_dem=tiene_dem,
        tiene_hand=tiene_hand,
        tiene_grafo=tiene_grafo,
        tiene_alos2=tiene_alos2,
        tiene_gauges=tiene_gauges,
        n_municipios=n_mun,
        puede_physics_ai=puede_physics,
        puede_optuna=puede_optuna,
        puede_gnn=puede_gnn,
        features_activas=features,
        features_prohibidas=prohibidas,
        warnings=w,
    )
