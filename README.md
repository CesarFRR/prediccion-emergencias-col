# 🌊 FloodModel Colombia

**Entrenamiento automático de modelos de predicción de inundaciones para cualquier departamento de Colombia.**

Basado en la investigación de Fase 2 y Fase 3 del proyecto Antioquia (AUROC 0.946, Optuna v3, Physics-AI Residual).  
Diseñado para que cualquier profesor, universidad u organización de gestión del riesgo pueda replicar los resultados sin ser experto en ML.

---

## 🚀 Arranque rápido (5 minutos)

```bash
# 1. Clonar
git clone https://github.com/usuario/floodmodel-colombia.git
cd floodmodel-colombia

# 2. Instalar dependencias
pip install pandas numpy lightgbm scikit-learn geopandas pyarrow

# 3. Descargar datos globales (~24 MB, +1.7 GB opcional)
python app.py

# 4. Agregar un departamento (wizard guiado)
python main.py --nuevo

# 5. Entrenar
python main.py --depto 05
```

Los datos se descargan de HuggingFace. El script app.py verifica que esten todos y ofrece bajar los que faltan.

---

## 📁 Estructura del repositorio

```
floodmodel-colombia/
│
├── main.py                     # Interfaz principal (train, validate, predict)
│
├── datos_globales/             # Datos nacionales (NO se tocan por depto)
│   ├── CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet    # Precipitación satelital (19MB)
│   ├── SAR_COLOMBIA_MENSUAL_2018_2026.parquet      # Radar Sentinel-1 (5.4MB)
│   ├── indices_climaticos_mensuales.parquet         # Índices NOAA (21KB)
│   └── Carto100000_Colombia_DI_2022.gpkg            # Cartografía IGAC (1.7GB)
│
├── departamentos/              # Uno por departamento
│   └── 05/                     # Antioquia (ejemplo)
│       ├── datos/              # Target de emergencias (CSV/Parquet)
│       │   └── emergencias_antioquia.csv
│       ├── processed/          # Datos procesados (HAND, grafo, etc.)
│       └── output/             # Modelos entrenados + métricas
│
├── core/                       # Motor (NO se toca por depto)
│   ├── detector_nivel.py       # Árbol de decisión: Nivel 1-4
│   ├── features.py             # Fábrica de features según nivel
│   ├── pipeline.py             # Entrenamiento automático (ML + Physics-AI)
│   └── extractor_geopackage.py # Extrae municipios + drenaje del GPKG
│
├── gee/                        # Scripts para Google Earth Engine
│   ├── extraccion_colombia.js  # Extrae CHIRPS + SAR para toda Colombia
│   └── procesar_exports.py     # Convierte CSVs de GEE a parquet
│
└── utils/                      # Utilidades
    └── enlaces.json            # URLs de fuentes de datos
```

---

## 📊 ¿Qué datos necesito para mi departamento?

### Obligatorios (sin esto no se puede entrenar)

| Dato | Fuente | Formato |
|------|--------|---------|
| **Target de emergencias** | UNGRD, DAGRAN, DesInventar | CSV con columnas: `municipio, fecha, evento` |
| **CHIRPS** | Google Earth Engine (script incluido) | Ya descargado para toda Colombia |
| **SAR Sentinel-1** | Google Earth Engine (script incluido) | Ya descargado para toda Colombia |
| **Índices NOAA** | NOAA CPC | Ya descargado (global) |

### Opcionales (mejoran el modelo)

| Dato | Para qué | Nivel que desbloquea |
|------|----------|:---------------------:|
| **DEM AW3D30** | HAND, pendiente, flow accumulation | Nivel 3 (Physics-AI) |
| **Grafo de drenaje** | Precipitación aguas arriba | Nivel 3 (Physics-AI) |
| **ALOS-2 Banda L** | Penetrar dosel forestal | Nivel 4 |

---

## 🧠 ¿Cómo funciona?

```
main.py
  │
  ├─ 1. Valida datos globales (CHIRPS, SAR, NOAA, GPKG)
  ├─ 2. Detecta qué departamentos tienen target
  ├─ 3. Muestra niveles posibles (1-4)
  └─ 4. Invoca core/pipeline.py
       │
       ├─ detector_nivel: evalúa capacidades → Nivel 1/2/3/4
       ├─ features: construye features según nivel
       ├─ Entrena modelo principal (LightGBM configurado al nivel)
       └─ Entrena Physics-AI (si Nivel 3+)
```

### Niveles de datos

| Nivel | Datos disponibles | Modelo | AUROC esperado |
|:-----:|-------------------|--------|:--------------:|
| **1** | CHIRPS + topo básica | LightGBM simple (15 feat) | 0.75-0.82 |
| **2** | + SAR + NOAA | LightGBM F2 (52 feat) | 0.85-0.93 |
| **3** | + DEM + HAND + grafo | Optuna v3 + Physics-AI | 0.93-0.95 |
| **4** | + ALOS-2 + gauges | ConvLSTM + fusión | 0.95-0.97 |

---

## ➕ Agregar un departamento nuevo

```bash
# 1. Crear carpeta
mkdir -p departamentos/76/datos departamentos/76/output

# 2. Poner el target de emergencias (CSV o Parquet)
#    Columnas requeridas: municipio, fecha, evento
cp mis_emergencias_valle.csv departamentos/76/datos/

# 3. (Opcional) Extraer vectoriales del GPKG nacional
python core/extractor_geopackage.py 76

# 4. Ejecutar
python main.py --depto 76
```

---

## 📚 Documentación

| Documento | Contenido |
|-----------|-----------|
| `FASE-3/AVANCE_FASE3.md` | Resumen técnico completo (modelos, fracasos, lecciones) |
| `FASE-3/GUIA_NIVELES_MODELOS.md` | Qué modelo usar según datos disponibles |
| `FASE-3/GUIA_OBTENCION_DATOS.md` | Cómo conseguir cada fuente de datos |
| `FASE-3/CONTEXTO_COMPLETO.md` | Contexto completo para consumo por IA |

---

## 🔬 Resultados en Antioquia (referencia)

| Modelo | AUROC | F1 | Recall | Precision |
|--------|:-----:|:---:|:------:|:---------:|
| F2 (entregado) | 0.946 | 0.593 | 0.561 | 0.628 |
| **Optuna v3** | 0.944 | **0.673** | **0.628** | **0.725** |
| Physics-AI (ciegos) | 0.947 | 0.621 | 0.604 | 0.639 |

---

## ⚠️ Requisitos

- Python 3.10+
- 4GB RAM mínimo (8GB recomendado)
- Los datos globales (CHIRPS, SAR, NOAA) ya están incluidos en `datos_globales/`
- Google Earth Engine solo si necesitás re-generar CHIRPS/SAR

---

## 📝 Licencia y citación

Si usás este trabajo en tu investigación, por favor citá:

> C. Rincón et al. (2026). "FloodModel Colombia: A Replicable Framework for Flood Prediction in Data-Scarce Regions." Universidad Nacional de Colombia.
