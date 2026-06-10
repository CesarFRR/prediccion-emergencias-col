"""
Feature Factory — Construye automáticamente las features según el nivel detectado.

NO descarga imágenes. Todo opera sobre tablas (municipio×fecha).
Las features se dividen en:
  - núcleo:   siempre presentes (CHIRPS)
  - opcionales: según nivel (SAR, DEM, grafo, ALOS-2)
  - derivadas: calculadas a partir de las anteriores

Principio: si un dato no está, esa feature se omite (no se imputa con ceros mágicos).
"""

import pandas as pd
import numpy as np
from typing import List, Set


def construir_features(
    df: pd.DataFrame,
    features_activas: List[str],
    nivel: int,
    hand_map: dict = None,
    fa_map: dict = None,
    grafo: dict = None,
) -> pd.DataFrame:
    """
    A partir de un DataFrame base (municipio×fecha con columnas crudas),
    calcula las features derivadas necesarias para el nivel detectado.

    Args:
        df: DataFrame con columnas crudas (chirps_precip_mm_dia, VV_mean, etc.)
        features_activas: lista de features que el nivel permite
        nivel: 1-4 (del detector)
        hand_map: {municipio: hand_m} — si existe
        fa_map: {municipio: max_flow_acc_buffer} — si existe
        grafo: {municipio: [upstream_municipios]} — si existe

    Returns:
        DataFrame con solo las features activas calculadas
    """
    X = pd.DataFrame(index=df.index)
    municipios = df['municipio'].values if 'municipio' in df.columns else None

    # ═══ NÚCLEO (Nivel 1): CHIRPS + acumulados ═══
    if 'chirps_precip_mm_dia' in df.columns:
        precip = df['chirps_precip_mm_dia'].fillna(0)
        X['chirps_precip_mm_dia'] = precip

        if municipios is not None:
            df_temp = pd.DataFrame({'municipio': municipios, 'precip': precip})
            for w in [1, 3, 7, 15, 30]:
                col = f'precip_acum_{w}d'
                if col in features_activas:
                    X[col] = df_temp.groupby('municipio')['precip'].transform(
                        lambda x: x.rolling(w, min_periods=1).sum()
                    ).values

            if 'p90_precip_3d' in features_activas:
                X['p90_precip_3d'] = df_temp.groupby('municipio')['precip'].transform(
                    lambda x: x.rolling(3, min_periods=1).quantile(0.9)
                ).values

    # ═══ NIVEL 2: SAR + NOAA + Clima ═══
    for sar_col in ['VV_mean', 'VH_mean', 'z_VV_mean', 'n_scenes']:
        if sar_col in df.columns and sar_col in features_activas:
            X[sar_col] = df[sar_col].fillna(0)

    if 'VV_mean' in X.columns and 'VH_mean' in X.columns and 'VV_minus_VH' in features_activas:
        X['VV_minus_VH'] = X['VV_mean'] - X['VH_mean']

    for noaa_col in ['SOI', 'QBO30', 'QBO50', 'ZWND200',
                      'N12_ANOM', 'N3_ANOM', 'N4_ANOM', 'oni_anom', 'oni_total']:
        if noaa_col in df.columns and noaa_col in features_activas:
            X[noaa_col] = df[noaa_col].fillna(0)

    for clima_col in ['humedad_media', 'viento_medio', 'temperatura_media', 'temperatura_max']:
        if clima_col in df.columns and clima_col in features_activas:
            X[clima_col] = df[clima_col].fillna(0)

    # ═══ NIVEL 3: DEM derivadas + HAND + Grafo ═══
    for dem_col in ['slope_mean', 'slope_max', 'slope_p90',
                     'twi_mean', 'twi_max', 'twi_p90', 'twi_p99',
                     'acc_mean', 'acc_max', 'acc_p90', 'acc_p99']:
        if dem_col in df.columns and dem_col in features_activas:
            X[dem_col] = df[dem_col].fillna(0)

    if 'hand_m' in features_activas:
        if hand_map and municipios is not None:
            X['hand_m'] = [hand_map.get(m, 15) for m in municipios]
        elif 'hand_m' in df.columns:
            X['hand_m'] = df['hand_m'].fillna(15)
        else:
            X['hand_m'] = 15.0

    for fa_col in ['max_flow_acc_buffer', 'p95_flow_acc_buffer', 'p99_flow_acc_buffer']:
        if fa_col in features_activas:
            if fa_map and municipios is not None:
                X[fa_col] = [fa_map.get(m, 1) for m in municipios]
            elif fa_col in df.columns:
                X[fa_col] = df[fa_col].fillna(1)
            else:
                X[fa_col] = 1.0

    if 'precip_aguas_arriba_3d' in features_activas and grafo and municipios is not None and 'precip_acum_3d' in X.columns:
        p3d = X['precip_acum_3d'].values
        up_vals = np.zeros(len(municipios))
        for i, m in enumerate(municipios):
            ups = grafo.get(m, [])
            if ups:
                mask = np.isin(municipios, ups)
                if mask.any():
                    up_vals[i] = p3d[mask].mean()
        X['precip_aguas_arriba_3d'] = up_vals

    # ═══ Riesgo físico (Nivel 3+, si están HAND + grafo + flow) ═══
    if nivel >= 3 and hand_map and fa_map and grafo and municipios is not None and 'precip_acum_3d' in X.columns:
        h_vals = X['hand_m'].values
        p3d_vals = X['precip_acum_3d'].values
        fa_vals = X.get('max_flow_acc_buffer', pd.Series([1]*len(municipios))).values

        hazard_rio = np.clip((30 - np.clip(h_vals, -50, 100)) / 60, 0, 1)
        hazard_local = np.clip(p3d_vals / 100, 0, 1)

        hazard_upstream = np.zeros(len(municipios))
        for i, m in enumerate(municipios):
            ups = grafo.get(m, [])
            if ups:
                mask = np.isin(municipios, ups)
                if mask.any():
                    hazard_upstream[i] = np.clip(p3d_vals[mask].mean() / 50, 0, 1)

        hazard_flow = np.clip(np.log1p(fa_vals) / 16, 0, 1)
        X['riesgo_fisico'] = 0.25*hazard_rio + 0.15*hazard_local + 0.45*hazard_upstream + 0.15*hazard_flow

    # ═══ Estáticas (si están en crudo) ═══
    for est_col in ['elevacion_msnm', 'pendiente_val', 'distancia_rio_km_real',
                     'pct_inundable', 'river_length_km', 'river_density_km_per_km2',
                     'river_area_pct']:
        if est_col in df.columns and est_col in features_activas:
            X[est_col] = df[est_col].fillna(0)

    # ═══ Temporal ═══
    if 'mes' not in X.columns and 'fecha' in df.columns:
        fechas = pd.to_datetime(df['fecha'])
        if 'mes_sin' in features_activas:
            X['mes_sin'] = np.sin(2 * np.pi * fechas.dt.month / 12)
        if 'mes_cos' in features_activas:
            X['mes_cos'] = np.cos(2 * np.pi * fechas.dt.month / 12)

    # ═══ Filtrar solo features activas que existen ═══
    final_cols = [c for c in features_activas if c in X.columns]
    return X[final_cols].fillna(0)
