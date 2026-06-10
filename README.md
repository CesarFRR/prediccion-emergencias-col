# 🌊 Prediccion de Emergencias — Colombia

**Framework automatizado de prediccion de emergencias por departamento.**

Actualmente enfocado en **inundaciones**, disenado para extenderse a deslizamientos, sequias y otros eventos. Basado en la investigacion Fase 2 y 3 del proyecto Antioquia (AUROC 0.946, Optuna v3, Physics-AI Residual).

Disenado para que cualquier universidad, entidad de gestion del riesgo u organizacion pueda replicar resultados sin ser experto en machine learning.

---

## 🚀 Arranque rapido

```bash
# 1. Clonar
git clone https://github.com/CesarFRR/prediccion-emergencias-col.git
cd prediccion-emergencias-col

# 2. Instalar dependencias
pip install pandas numpy lightgbm scikit-learn geopandas pyarrow

# 3. Descargar datos nacionales (~24 MB, +1.7 GB opcional)
python app.py

# 4. Agregar un departamento (wizard guiado)
python main.py --nuevo

# 5. Entrenar
python main.py --depto 05
```

---

## 📊 Datos necesarios

### Proporcionados (descarga automatica con `python app.py`)

| Dato | Tamano | Fuente |
|------|:------:|--------|
| CHIRPS (precipitacion diaria) | 18 MB | Google Earth Engine → HuggingFace |
| SAR Sentinel-1 (radar mensual) | 5 MB | Google Earth Engine → HuggingFace |
| Indices NOAA (SOI, QBO, ONI) | 24 KB | NOAA CPC |
| Cartografia IGAC 1:100,000 | 1.7 GB | IGAC (opcional, para Physics-AI) |

### Debe proporcionar el usuario

| Dato | Formato | Ejemplo |
|------|---------|---------|
| **Historico de emergencias** | CSV o Parquet | `MUNICIPIO,FECHA,EVENTO` |

El unico archivo que necesitas aportar es el registro historico de emergencias de tu departamento. El wizard (`python main.py --nuevo`) te guia con el formato exacto.

### Opcionales (mejoran el modelo)

| Dato | Para que sirve | Como obtenerlo |
|------|----------------|----------------|
| **DEM 30m** (modelo de elevacion) | Pendiente, elevacion, HAND, flow accumulation | Descargar tiles AW3D30 de [JAXA](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/) (~200 MB por depto) |
| **HAND** (altura sobre rio) | Riesgo de inundacion fluvial | Se calcula con pysheds a partir del DEM |
| **Grafo de drenaje** | Precipitacion aguas arriba, Physics-AI | Se extrae del GPKG con `core/extractor_geopackage.py` |
| **ALOS-2 Banda L** | Penetrar dosel forestal (municipios ciegos al SAR) | [JAXA G-Portal](https://auig2.jaxa.jp/) (requiere cuenta) |

**Sin estos datos el modelo funciona en Nivel 1-2 (AUROC 0.75-0.93).** Con DEM + HAND + grafo se activa Physics-AI (Nivel 3, AUROC 0.93-0.95) que detecta inundaciones incluso en municipios donde el radar Sentinel-1 no funciona por la vegetacion densa.

---

## 📁 Estructura

```
prediccion-emergencias-col/
├── main.py                     # Interfaz principal (wizard + entrenamiento)
├── app.py                      # Descarga de datos nacionales
│
├── datos_globales/             # Datos nacionales (descarga automatica)
│   ├── CHIRPS_COLOMBIA_DIARIO_2018_2026.parquet
│   ├── SAR_COLOMBIA_MENSUAL_2018_2026.parquet
│   ├── indices_climaticos_mensuales.parquet
│   └── Carto100000_Colombia_DI_2022.gpkg
│
├── departamentos/              # Un directorio por departamento
│   └── 05/                     # Antioquia (ejemplo)
│       ├── datos/              # Target de emergencias del usuario
│       ├── processed/          # HAND, grafo de drenaje, etc.
│       └── output/             # Modelos entrenados + metricas
│
├── core/                       # Motor (no se modifica por departamento)
│   ├── detector_nivel.py       # Arbol de decision: Nivel 1-4
│   ├── features.py             # Fabrica de features segun nivel
│   ├── pipeline.py             # Entrenamiento (ML principal + Physics-AI)
│   └── extractor_geopackage.py # Extrae municipios + drenaje del GPKG
│
├── gee/                        # Scripts Google Earth Engine
│   ├── extraccion_colombia.js  # Extrae CHIRPS + SAR para toda Colombia
│   └── procesar_exports.py     # Convierte CSVs de GEE a parquet
│
└── utils/
    └── enlaces.json            # URLs de fuentes de datos oficiales
```

---

## 🧠 Como funciona

```
main.py
  │
  ├─ 1. Verifica datos nacionales (app.py si faltan)
  ├─ 2. Wizard: elige departamento, carga target de emergencias
  ├─ 3. Detecta nivel de datos (1-4)
  └─ 4. Entrena automaticamente:
       ├─ Modelo principal (LightGBM configurado al nivel)
       └─ Physics-AI (si hay DEM + drenaje, para municipios con poca cobertura SAR)
```

### Niveles de datos

| Nivel | Datos disponibles | Modelo | AUROC esperado |
|:-----:|-------------------|--------|:--------------:|
| **1** | CHIRPS + topo basica | LightGBM simple (15 features) | 0.75-0.82 |
| **2** | + SAR + NOAA | LightGBM 52 features | 0.85-0.93 |
| **3** | + DEM + HAND + grafo | Optuna v3 + Physics-AI | 0.93-0.95 |
| **4** | + ALOS-2 + estaciones | ConvLSTM + fusion | 0.95-0.97 |

---

## 🔬 Resultados en Antioquia (referencia)

| Modelo | AUROC | F1 | Recall | Precision |
|--------|:-----:|:---:|:------:|:---------:|
| F2 (entregado) | 0.946 | 0.593 | 0.561 | 0.628 |
| **Optuna v3** | 0.944 | **0.673** | **0.628** | **0.725** |
| Physics-AI (municipios sin SAR) | 0.947 | 0.621 | 0.604 | 0.639 |

---

## 🛣️ Hoja de ruta

- [x] Inundaciones (actual)
- [ ] Deslizamientos
- [ ] Sequias
- [ ] Incendios forestales
- [ ] Agrupacion por zonas hidrograficas (no por departamento)
- [ ] Interfaz web

---

## ⚠️ Requisitos

- Python 3.10+
- 4 GB RAM (8 GB recomendado)
- Los datos nacionales se descargan con `python app.py`
- Google Earth Engine solo si se necesita re-generar CHIRPS/SAR desde cero

---

## 📝 Licencia y citacion

Si usas este trabajo en investigacion, cita:

> Rincon, C. et al. (2026). "Prediccion de Emergencias Colombia: Framework Replicable de Prediccion de Inundaciones en Regiones con Datos Limitados." Universidad Nacional de Colombia.
