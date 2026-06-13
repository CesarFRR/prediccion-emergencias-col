# Graph Report - .  (2026-06-11)

## Corpus Check
- Corpus is ~8,455 words - fits in a single context window. You may not need a graph.

## Summary
- 90 nodes · 129 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Feature Engineering & Pipeline|Feature Engineering & Pipeline]]
- [[_COMMUNITY_GeoPackage Extraction & CLI|GeoPackage Extraction & CLI]]
- [[_COMMUNITY_GEE Export Processing|GEE Export Processing]]
- [[_COMMUNITY_Web App Download Interface|Web App Download Interface]]
- [[_COMMUNITY_Data Level Detection & Docs|Data Level Detection & Docs]]
- [[_COMMUNITY_Department Mapping (GAULDIVIPOLA)|Department Mapping (GAUL/DIVIPOLA)]]
- [[_COMMUNITY_GEE SARCHIRPS Extraction|GEE SAR/CHIRPS Extraction]]
- [[_COMMUNITY_Static Assets & Links|Static Assets & Links]]
- [[_COMMUNITY_OpenCode Config|OpenCode Config]]
- [[_COMMUNITY_Plugin Package Config|Plugin Package Config]]

## God Nodes (most connected - your core abstractions)
1. `entrenar()` - 16 edges
2. `strip_acc()` - 8 edges
3. `detectar_nivel()` - 7 edges
4. `extraer_departamento()` - 7 edges
5. `strip_acc()` - 7 edges
6. `construir_dataset_departamento()` - 7 edges
7. `construir_features()` - 6 edges
8. `procesar_chirps()` - 6 edges
9. `procesar_sar()` - 6 edges
10. `intentar_descarga()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `detectar_nivel()` --conceptually_related_to--> `Niveles de Datos (1-4): modelo progresivo segun datos disponibles con AUROC esperado por nivel`  [INFERRED]
  core/detector_nivel.py → README.md
- `strip_acc()` --semantically_similar_to--> `strip_acc()`  [INFERRED] [semantically similar]
  core/pipeline.py → gee/procesar_exports.py
- `strip_acc()` --semantically_similar_to--> `strip_acc()`  [INFERRED] [semantically similar]
  core/pipeline.py → main.py
- `strip_acc()` --semantically_similar_to--> `strip_acc()`  [INFERRED] [semantically similar]
  main.py → gee/procesar_exports.py
- `Physics-AI Corrector (modelo fisico + LightGBM residual)` --conceptually_related_to--> `Physics-AI Residual: modelo fisico HAND+flow + corrector LightGBM para municipios sin cobertura SAR`  [INFERRED]
  core/pipeline.py → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Niveles de Datos Pipeline: Model Factory Pattern por nivel 1-4 (detectar, construir features, entrenar, physics-ai)** — core_detector_nivel_detectar_nivel, core_features_construir_features, core_pipeline_entrenar, core_pipeline__entrenar_physics_ai, core_detector_nivel_niveldatos [INFERRED 0.85]
- **GAUL to DIVIPOLA Department Name Mapping (parallel implementations in pipeline and procesar_exports)** — core_pipeline_gaul_to_code, gee_procesar_exports_depto_gaul_to_code, core_pipeline__match_depto_general, gee_procesar_exports__match_depto [INFERRED 0.85]
- **GEE Data Extraction to Training Pipeline Flow (CHIRPS + SAR CSV exports -> parquet datasets -> LightGBM training)** — gee_extraccion_colombia_extraerchirps, gee_extraccion_colombia_extraersarperiodo, gee_procesar_exports_procesar_chirps, gee_procesar_exports_procesar_sar, gee_procesar_exports_construir_dataset_departamento, core_pipeline_entrenar [INFERRED 0.85]

## Communities (11 total, 2 thin omitted)

### Community 0 - "Feature Engineering & Pipeline"
Cohesion: 0.16
Nodes (18): construir_features(), DataFrame, Feature Factory — Construye automáticamente las features según el nivel detectad, A partir de un DataFrame base (municipio×fecha con columnas crudas),     calcula, Physics-AI Corrector (modelo fisico + LightGBM residual), Evaluar Modelo (metricas AUROC/F1/Recall/Precision/PR-curve), entrenar(), _entrenar_physics_ai() (+10 more)

### Community 1 - "GeoPackage Extraction & CLI"
Cohesion: 0.15
Nodes (14): cargar_nombres_departamentos(), extraer_departamento(), Path, Extractor de Datos Vectoriales — Desde Carto100000 IGAC (GPKG) =================, Carga mapeo de códigos a nombres desde Limite_Departamental., Extrae todos los datos vectoriales para un departamento.      Args:         gpkg, entrenar_departamento(), listar_departamentos() (+6 more)

### Community 2 - "GEE Export Processing"
Cohesion: 0.22
Nodes (14): construir_dataset_departamento(), _match_depto(), merge_noaa(), procesar_chirps(), procesar_sar(), DataFrame, Path, Procesador de Exports GEE → Dataset Listo para Pipeline ======================== (+6 more)

### Community 3 - "Web App Download Interface"
Cohesion: 0.29
Nodes (8): descargar_individual(), descargar_zip(), intentar_descarga(), Descarga un archivo individual desde sus URLs., Orquesta la descarga: primero ZIP, luego individuales., Retorna {nombre: True/False} para cada archivo requerido., Descarga un ZIP y extrae su contenido a datos_globales/., verificar_todos()

### Community 4 - "Data Level Detection & Docs"
Cohesion: 0.22
Nodes (9): detectar_nivel(), NivelDatos, FASE-4: Fábrica de Modelos de Inundación por Departamento ======================, Resultado del detector: qué nivel y qué capacidades tiene el depto., Árbol de decisión: determina el nivel a partir de la config del depto.      conf, FloodModel Colombia: Framework de Prediccion de Emergencias por Departamento, Niveles de Datos (1-4): modelo progresivo segun datos disponibles con AUROC esperado por nivel, Physics-AI Residual: modelo fisico HAND+flow + corrector LightGBM para municipios sin cobertura SAR (+1 more)

### Community 5 - "Department Mapping (GAUL/DIVIPOLA)"
Cohesion: 0.67
Nodes (4): Match Departamento General (GAUL name to DIVIPOLA code), GAUL to DIVIPOLA Code Mapping (35 entries), Match Depto (GAUL name to DIVIPOLA code), Depto GAUL to DIVIPOLA Code Mapping (32 entries)

### Community 6 - "GEE SAR/CHIRPS Extraction"
Cohesion: 0.67
Nodes (3): extraerChirps(), extraerSARPeriodo(), Municipios Colombia GAUL level2 FeatureCollection (1122 features)

### Community 7 - "Static Assets & Links"
Cohesion: 0.67
Nodes (3): Archivos de Datos Globales Requeridos (CHIRPS, SAR, NOAA, Carto100000 GPKG), CHIRPS Precipitacion Diaria (UCSB-CHG/CHIRPS/DAILY, 0.05deg), IGAC Cartografia 1:100,000 de Colombia (GPKG Carto100000)

## Knowledge Gaps
- **8 isolated node(s):** `$schema`, `plugin`, `@opencode-ai/plugin`, `Path`, `DataFrame` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `entrenar()` connect `Feature Engineering & Pipeline` to `GeoPackage Extraction & CLI`, `GEE Export Processing`, `Data Level Detection & Docs`, `Department Mapping (GAUL/DIVIPOLA)`?**
  _High betweenness centrality (0.315) - this node is a cross-community bridge._
- **Why does `construir_dataset_departamento()` connect `GEE Export Processing` to `Feature Engineering & Pipeline`, `Department Mapping (GAUL/DIVIPOLA)`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `strip_acc()` connect `GEE Export Processing` to `Feature Engineering & Pipeline`, `GeoPackage Extraction & CLI`, `Department Mapping (GAUL/DIVIPOLA)`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `entrenar()` (e.g. with `extraer_departamento()` and `construir_dataset_departamento()`) actually correct?**
  _`entrenar()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `strip_acc()` (e.g. with `strip_acc()` and `strip_acc()`) actually correct?**
  _`strip_acc()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `detectar_nivel()` (e.g. with `construir_features()` and `Niveles de Datos (1-4): modelo progresivo segun datos disponibles con AUROC esperado por nivel`) actually correct?**
  _`detectar_nivel()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `strip_acc()` (e.g. with `strip_acc()` and `strip_acc()`) actually correct?**
  _`strip_acc()` has 2 INFERRED edges - model-reasoned connections that need verification._