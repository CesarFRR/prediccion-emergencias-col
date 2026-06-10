"""
Pipeline de Entrenamiento — Fase 4 (Fábrica de Modelos)
========================================================
Una función `entrenar()` que recibe la config de un depto y entrena
el modelo apropiado según el nivel de datos detectado.

Flujo:
  1. Cargar datos crudos (según config)
  2. Detectar nivel de datos (detector_nivel.py)
  3. Construir features (features.py, según nivel)
  4. Elegir y entrenar modelo (según nivel)
  5. Evaluar y guardar
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json, unicodedata, warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    precision_recall_curve, confusion_matrix
)

from core.detector_nivel import detectar_nivel, NivelDatos
from core.features import construir_features

warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# Mapeo GAUL → código DIVIPOLA para filtrar datasets nacionales
GAUL_TO_CODE = {
    'amazonas': '91', 'antioquia': '05', 'arauca': '81',
    'atlantico': '08', 'atlántico': '08', 'bolivar': '13', 'bolívar': '13',
    'boyaca': '15', 'boyacá': '15', 'caldas': '17', 'caqueta': '18', 'caquetá': '18',
    'casanare': '85', 'cauca': '19', 'cesar': '20', 'choco': '27', 'chocó': '27',
    'cordoba': '23', 'córdoba': '23', 'cundinamarca': '25',
    'guainia': '94', 'guainía': '94', 'guaviare': '95',
    'huila': '41', 'guajira': '44', 'la guajira': '44',
    'magdalena': '47', 'meta': '50', 'narino': '52', 'nariño': '52',
    'norte de santander': '54', 'putumayo': '86', 'quindio': '63', 'quindío': '63',
    'risaralda': '66', 'santander': '68', 'sucre': '70',
    'tolima': '73', 'valle del cauca': '76', 'vaupes': '97', 'vaupés': '97',
    'vichada': '99', 'san andres y providencia': '88',
    'bogota': '11', 'bogotá': '11', 'buenaventura': '76',
}


def _match_depto_general(nombre_gaul: str, config: dict) -> bool:
    """Determina si un nombre GAUL pertenece a este departamento."""
    n = strip_acc(str(nombre_gaul)).lower()
    expected = GAUL_TO_CODE.get(n)
    if expected:
        return expected == config['codigo_departamento']
    nombre_depto = strip_acc(str(config['nombre'])).lower()
    return n == nombre_depto


def strip_acc(s):
    return ''.join(c for c in unicodedata.normalize('NFD', str(s))
                   if unicodedata.category(c) != 'Mn')


def log(msg: str):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')


def entrenar(config: dict) -> dict:
    """
    Entrena un modelo de inundación para un departamento.

    Args:
        config: dict con las claves definidas en antioquia/config.py

    Returns:
        dict con métricas, paths de artefactos, y diagnóstico
    """
    depto = config['nombre']
    log(f'{"═"*55}')
    log(f'Pipeline Fase 4 — {depto}')
    log(f'{"═"*55}')

    # ─── 1. DETECTAR NIVEL ───
    nivel_data = detectar_nivel(config)
    log(f'[{nivel_data.label}] {len(nivel_data.features_activas)} features activas')
    for w in nivel_data.warnings:
        log(f'   ⚠️  {w}')
    if nivel_data.nivel == 0:
        log(f'❌ Imposible entrenar modelo para {depto}.')
        return {'status': 'failed', 'reason': nivel_data.label}

    # ─── 2. CARGAR DATOS ───
    log(f'Cargando datos desde fuentes nacionales...')

    # 2a. CHIRPS (diario nacional → filtrar depto)
    chirps_path = config.get('chirps_path')
    if chirps_path and Path(chirps_path).exists():
        chirps = pd.read_parquet(chirps_path)
        chirps['fecha'] = pd.to_datetime(chirps['fecha'])
        chirps['municipio'] = chirps['ADM2_NAME'].str.upper().apply(strip_acc)
        # Filtrar por departamento
        depto_mask = chirps['ADM1_NAME'].apply(
            lambda x: _match_depto_general(x, config)
        )
        chirps = chirps[depto_mask].copy()
        log(f'   CHIRPS: {len(chirps):,} filas, {chirps["municipio"].nunique()} municipios')

        # Calcular acumulados
        chirps = chirps.sort_values(['municipio', 'fecha'])
        for w in [1, 3, 7, 15, 30]:
            chirps[f'precip_acum_{w}d'] = chirps.groupby('municipio')['mean'].transform(
                lambda x: x.rolling(w, min_periods=1).sum())
        chirps = chirps.rename(columns={'mean': 'chirps_precip_mm_dia'})
        chirps['p90_precip_3d'] = chirps.groupby('municipio')['chirps_precip_mm_dia'].transform(
            lambda x: x.rolling(3, min_periods=1).quantile(0.9))
        df = chirps
    elif config.get('dataset_path'):
        df = pd.read_parquet(config['dataset_path'])
        log(f'   Dataset pre-armado: {len(df):,} filas')
    else:
        log(f'❌ Sin CHIRPS ni dataset. Imposible continuar.')
        return {'status': 'failed', 'reason': 'Sin fuente de datos'}

    df['fecha'] = pd.to_datetime(df['fecha'])
    if 'municipio' not in df.columns and 'ADM2_NAME' in df.columns:
        df['municipio'] = df['ADM2_NAME'].str.upper().apply(strip_acc)

    # 2b. SAR (mensual nacional → merge al dataset diario)
    sar_path = config.get('sar_path')
    if sar_path and Path(sar_path).exists():
        sar = pd.read_parquet(sar_path)
        sar['municipio'] = sar['ADM2_NAME'].str.upper().apply(strip_acc)
        sar['fecha'] = pd.to_datetime(sar['fecha_mes'] + '-01')
        depto_mask = sar['ADM1_NAME'].apply(lambda x: _match_depto_general(x, config))
        sar = sar[depto_mask].copy()

        # Renombrar columnas SAR
        sar = sar.rename(columns={
            'VV_mean': 'VV_mean', 'VH_mean': 'VH_mean',
            'VV_stdDev': 'VV_stdDev', 'VH_stdDev': 'VH_stdDev',
            'VVVH_ratio': 'VV_minus_VH', 'n_scenes': 'n_scenes'
        })
        sar_cols = ['municipio', 'fecha', 'VV_mean', 'VH_mean', 'VV_stdDev',
                     'VH_stdDev', 'VV_minus_VH', 'n_scenes']
        sar_cols = [c for c in sar_cols if c in sar.columns]
        sar_merge = sar[['municipio', 'fecha'] + sar_cols[2:]].copy()

        # Merge mensual → diario (forward fill por mes)
        df['mes_key'] = df['fecha'].dt.to_period('M')
        sar_merge['mes_key'] = sar_merge['fecha'].dt.to_period('M')
        sar_merge = sar_merge.drop(columns=['fecha'])
        df = df.merge(sar_merge, on=['municipio', 'mes_key'], how='left')
        df = df.drop(columns=['mes_key'])

        # Calcular z_VV_mean
        if 'VV_mean' in df.columns:
            vv_mean = df.groupby('municipio')['VV_mean'].transform('mean')
            vv_std = df.groupby('municipio')['VV_mean'].transform('std')
            df['z_VV_mean'] = (df['VV_mean'] - vv_mean) / vv_std.replace(0, 1)
        log(f'   SAR: {len(sar):,} filas mensuales mergueadas')
    else:
        log(f'   ⚠️  Sin SAR. Entrenando sin features de radar.')

    # 2c. Target
    target_col = config.get('target_column', 'flood_target')
    if config.get('target_from_dataset', True) and target_col in df.columns:
        pass  # ya está en el dataset
    elif config.get('target_path'):
        target_df = pd.read_parquet(config['target_path'])
        if 'municipio' in target_df.columns and 'fecha' in target_df.columns:
            target_df['municipio'] = target_df['municipio'].str.upper().apply(strip_acc)
            target_df['fecha'] = pd.to_datetime(target_df['fecha'])
            df = df.merge(target_df[['municipio', 'fecha', target_col]], on=['municipio', 'fecha'], how='left')
            df[target_col] = df[target_col].fillna(0).astype(int)
        else:
            log(f'   ⚠️  target_path no tiene columnas municipio/fecha. Esperando columna en dataset.')

    if target_col not in df.columns:
        log(f'❌ Columna target \"{target_col}\" no encontrada.')
        return {'status': 'failed', 'reason': f'Target \"{target_col}\" no encontrado'}

    # 2d. NOAA (merge)
    if config.get('noaa_path') and Path(config['noaa_path']).exists():
        noaa = pd.read_parquet(config['noaa_path'])
        noaa['anio_mes'] = noaa['YR'].astype(str) + '-' + noaa['MON'].astype(str).str.zfill(2)
        df['anio_mes'] = df['fecha'].dt.year.astype(str) + '-' + df['fecha'].dt.month.astype(str).str.zfill(2)
        noaa_cols = ['anio_mes', 'SOI', 'QBO30', 'QBO50', 'ZWND200',
                     'N12_ANOM', 'N3_ANOM', 'N4_ANOM']
        available = [c for c in noaa_cols if c in noaa.columns]
        df = df.merge(noaa[available], on='anio_mes', how='left')

    # Cargar infraestructura física (Nivel 3+)
    hand_map, fa_map, grafo = None, None, None
    if nivel_data.nivel >= 3:
        if config.get('hand_path') and Path(config['hand_path']).exists():
            hand = pd.read_csv(config['hand_path'])
            hand['municipio'] = hand['municipio'].str.upper().apply(strip_acc)
            hand_map = hand.set_index('municipio')['hand_m'].to_dict()
        if config.get('flow_path') and Path(config['flow_path']).exists():
            flow = pd.read_csv(config['flow_path'])
            flow['municipio'] = flow['municipio'].str.upper().apply(strip_acc)
            fa_map = flow.set_index('municipio')['max_flow_acc_buffer'].to_dict()
        if config.get('grafo_path') and Path(config['grafo_path']).exists():
            with open(config['grafo_path']) as f:
                grafo = json.load(f)

    # ─── 3. SPLIT TEMPORAL ───
    train = df[df['fecha'].dt.year <= 2023].copy()
    val   = df[df['fecha'].dt.year == 2024].copy()
    test  = df[df['fecha'].dt.year >= 2025].copy()
    target_col = config.get('target', 'flood_target')

    log(f'   Split: train≤2023 ({len(train):,}), val=2024 ({len(val):,}), test≥2025 ({len(test):,})')
    log(f'   Positivos: train={train[target_col].sum():,}, val={val[target_col].sum():,}, test={test[target_col].sum():,}')

    # ─── 4. CONSTRUIR FEATURES ───
    log(f'Construyendo features (nivel {nivel_data.nivel})...')
    X_train = construir_features(train, nivel_data.features_activas, nivel_data.nivel,
                                  hand_map, fa_map, grafo)
    X_val   = construir_features(val, nivel_data.features_activas, nivel_data.nivel,
                                  hand_map, fa_map, grafo)
    X_test  = construir_features(test, nivel_data.features_activas, nivel_data.nivel,
                                  hand_map, fa_map, grafo)

    y_train = train[target_col].values
    y_val   = val[target_col].values
    y_test  = test[target_col].values

    log(f'   Features finales: {len(X_train.columns)}')
    if len(X_train.columns) < 5:
        log(f'❌ Muy pocas features. Verificá los datos crudos.')
        return {'status': 'failed', 'reason': 'Pocas features'}

    # ─── 5. ENTRENAR MODELO PRINCIPAL ───
    log(f'Entrenando modelo principal...')
    pos_w = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'class_weight': 'balanced',
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }

    # Ajustar complejidad según nivel
    if nivel_data.nivel <= 1:
        params.update({'num_leaves': 63, 'max_depth': 5, 'min_child_samples': 100,
                       'learning_rate': 0.01, 'subsample': 0.4, 'colsample_bytree': 0.4,
                       'reg_alpha': 0.1, 'reg_lambda': 0.1, 'min_split_gain': 0.1})
    elif nivel_data.nivel == 2:
        params.update({'num_leaves': 251, 'max_depth': 7, 'min_child_samples': 46,
                       'learning_rate': 0.006, 'subsample': 0.33, 'colsample_bytree': 0.39,
                       'reg_alpha': 0.03, 'reg_lambda': 0.012, 'min_split_gain': 0.118})
    else:  # Nivel 3-4: usar best params de Optuna v3
        params.update({'num_leaves': 151, 'max_depth': 9, 'min_child_samples': 55,
                       'learning_rate': 0.0039, 'subsample': 0.579, 'colsample_bytree': 0.574,
                       'reg_alpha': 0.014, 'reg_lambda': 0.063, 'min_split_gain': 0.226,
                       'path_smooth': 1.96})

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )
    log(f'   Mejor iteración: {model.best_iteration}')

    # ─── 6. EVALUAR ───
    resultados = _evaluar_modelo(model, X_train, y_train, X_val, y_val, X_test, y_test,
                                  test, nivel_data, config)

    # ─── 6b. PHYSICS-AI (si Nivel 3+ con HAND + grafo + flow) ───
    model_physics = None
    if nivel_data.puede_physics_ai and 'riesgo_fisico' in X_train.columns:
        log(f'Entrenando Physics-AI corrector (regresión residual)...')
        model_physics, resultados_physics = _entrenar_physics_ai(
            X_train, y_train, X_val, y_val, X_test, y_test,
            test, hand_map, fa_map, grafo, config, nivel_data
        )
        resultados['physics_ai'] = resultados_physics

    # ─── 7. GUARDAR ───
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f'modelo_{depto.lower().replace(" ", "_")}.txt'
    model.save_model(str(model_path))
    log(f'✅ Modelo principal guardado: {model_path}')

    resultados['model_path'] = str(model_path)
    resultados['features'] = list(X_train.columns)
    resultados['nivel'] = nivel_data.nivel
    resultados['nivel_label'] = nivel_data.label
    resultados['warnings'] = nivel_data.warnings
    resultados['status'] = 'success'

    if model_physics:
        phys_path = output_dir / f'physics_ai_{depto.lower().replace(" ", "_")}.txt'
        model_physics.save_model(str(phys_path))
        log(f'✅ Physics-AI guardado: {phys_path}')
        resultados['physics_model_path'] = str(phys_path)

    metrics_path = output_dir / f'metricas_{depto.lower().replace(" ", "_")}.json'
    with open(metrics_path, 'w') as f:
        json.dump({k: v for k, v in resultados.items()
                   if isinstance(v, (str, int, float, bool, list))}, f, indent=2, default=str)
    log(f'✅ Métricas guardadas: {metrics_path}')

    return resultados


def _entrenar_physics_ai(X_train, y_train, X_val, y_val, X_test, y_test,
                        test_df, hand_map, fa_map, grafo, config, nivel_data):
    """
    Physics-AI: modelo físico + corrector LightGBM de residuales.
    Fórmula: riesgo = 0.25*HAND + 0.15*precip_local + 0.45*upstream + 0.15*flow
    Corrector: LightGBM regressor que aprende (flood - riesgo).
    """
    # ─── Ya tenemos riesgo_fisico en X_* gracias a la feature factory ───
    riesgo_train = X_train['riesgo_fisico'].values
    riesgo_val   = X_val['riesgo_fisico'].values
    riesgo_test  = X_test['riesgo_fisico'].values

    residual_train = y_train - riesgo_train
    residual_val   = y_val   - riesgo_val

    # Corrector: regresión del residual
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 127,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }

    dtrain = lgb.Dataset(X_train, label=residual_train)
    dval   = lgb.Dataset(X_val, label=residual_val, reference=dtrain)

    corr = lgb.train(
        params, dtrain,
        num_boost_round=3000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100)],
    )

    # ─── Predicciones Physics-AI ───
    prob_train = np.clip(riesgo_train + corr.predict(X_train), 0, 1)
    prob_val   = np.clip(riesgo_val   + corr.predict(X_val),   0, 1)
    prob_test  = np.clip(riesgo_test  + corr.predict(X_test),  0, 1)

    # ─── Identificar municipios ciegos ───
    ciegos = config.get('municipios_ciegos', [])
    if not ciegos and 'n_scenes' in X_train.columns:
        # Auto-detectar: municipios con <20% de cobertura SAR
        cobertura = test_df.groupby('municipio')['n_scenes'].mean() if 'n_scenes' in test_df.columns else None
        if cobertura is not None:
            ciegos_auto = cobertura[cobertura < cobertura.median() * 0.3].index.tolist()
            if len(ciegos_auto) > 0:
                ciegos = ciegos_auto
                log(f'   Auto-detectados {len(ciegos)} municipios ciegos (baja cobertura SAR)')

    # ─── Métricas globales ───
    def _m(y_true, y_prob):
        prec, rec, thrs = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
        idx = np.argmax(f1s)
        preds = (y_prob >= thrs[idx]).astype(int)
        return {
            'auroc': float(roc_auc_score(y_true, y_prob)),
            'f1': float(f1s[idx]),
            'recall': float(rec[idx]),
            'precision': float(prec[idx]),
            'best_threshold': float(thrs[idx]),
            'tp': int(np.sum((preds == 1) & (y_true == 1))),
            'fp': int(np.sum((preds == 1) & (y_true == 0))),
            'total': int(y_true.sum()),
        }

    resultados = {
        'test': _m(y_test, prob_test),
        'val': _m(y_val, prob_val),
    }

    print(f'\n  🔬 PHYSICS-AI:')
    for split in ['val', 'test']:
        m = resultados[split]
        print(f'     {split.upper():5s} (thr={m["best_threshold"]:.3f}): '
              f'AUROC={m["auroc"]:.4f} F1={m["f1"]:.4f} '
              f'TP={m["tp"]}/{m["total"]} FP={m["fp"]}')

    # ─── Métricas SOLO en ciegos ───
    if ciegos:
        mask_ciegos = test_df['municipio'].isin(
            [strip_acc(m) for m in ciegos]
        ).values
        if mask_ciegos.any() and y_test[mask_ciegos].sum() > 0:
            mc = _m(y_test[mask_ciegos], prob_test[mask_ciegos])
            resultados['ciegos'] = mc
            print(f'     CIEGOS  (thr={mc["best_threshold"]:.3f}): '
                  f'AUROC={mc["auroc"]:.4f} F1={mc["f1"]:.4f} '
                  f'TP={mc["tp"]}/{mc["total"]} FP={mc["fp"]}')
            resultados['ciegos_list'] = ciegos
            resultados['umbral_recomendado_ciegos'] = mc['best_threshold']

    return corr, resultados


def _evaluar_modelo(model, X_train, y_train, X_val, y_val, X_test, y_test,
                    test_df, nivel_data: NivelDatos, config: dict) -> dict:
    """Evalúa en train/val/test y retorna métricas completas."""

    def _metricas(y_true, y_prob):
        prec, rec, thrs = precision_recall_curve(y_true, y_prob)
        f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
        best_idx = np.argmax(f1s)
        best_thr = float(thrs[best_idx])
        preds = (y_prob >= best_thr).astype(int)
        return {
            'auroc': float(roc_auc_score(y_true, y_prob)),
            'f1': float(f1_score(y_true, preds, zero_division=0)),
            'recall': float(recall_score(y_true, preds, zero_division=0)),
            'precision': float(precision_score(y_true, preds, zero_division=0)),
            'best_f1': float(f1s[best_idx]),
            'best_threshold': best_thr,
            'tp': int(np.sum((preds == 1) & (y_true == 1))),
            'fp': int(np.sum((preds == 1) & (y_true == 0))),
            'total_positivos': int(y_true.sum()),
        }

    resultados = {}
    for name, X, y in [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)]:
        prob = model.predict(X)
        resultados[name] = _metricas(y, prob)

    # ─── Imprimir ───
    print(f'\n{"="*55}')
    print(f'📊 RESULTADOS — {config["nombre"]} [{nivel_data.label}]')
    print(f'{"="*55}')
    for name in ['train', 'val', 'test']:
        m = resultados[name]
        print(f'  {name.upper():6s} (thr={m["best_threshold"]:.3f}): '
              f'AUROC={m["auroc"]:.4f} F1={m["f1"]:.4f} '
              f'Recall={m["recall"]:.4f} Prec={m["precision"]:.4f} '
              f'TP={m["tp"]} FP={m["fp"]} (de {m["total_positivos"]} positivos)')

    # ─── Feature importance ───
    imp = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance(importance_type='gain'),
    })
    imp['pct'] = imp['importance'] / imp['importance'].sum() * 100
    imp = imp.sort_values('importance', ascending=False)
    print(f'\n  TOP 10 FEATURES:')
    for _, row in imp.head(10).iterrows():
        print(f'     {row["feature"]:35s} {row["pct"]:5.1f}%')

    return resultados
