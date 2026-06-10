/**
 * EXTRACCIÓN COMPLETA — Colombia (1,122 municipios)
 * ==================================================
 * UN SOLO SCRIPT para extraer CHIRPS + SAR Sentinel-1 de toda Colombia.
 * Ejecutar en GEE Code Editor una vez. Exporta 2 archivos CSV.
 * 
 * REQUISITOS:
 *   - Cuenta Google Earth Engine activada
 *   - Proyecto GCP con API habilitada
 *   - Carpeta Drive para exports (se crea sola)
 * 
 * ⚠️  SAR se parte en 3 períodos para evitar timeout (>6h en GEE):
 *     2018-2020, 2021-2023, 2024-2026
 *     CHIRPS es rápido (~30 min, 1 solo export).
 * 
 * ADAPTACIÓN PARA OTRO PAÍS:
 *   Cambiar 'ADM0_NAME' de 'Colombia' al país deseado.
 *   GAUL cubre todo el mundo. Si tu país tiene <1000 municipios,
 *   debería funcionar igual.
 */

// ═══════════════════════════════════════════════════════════════
// 0. CONFIGURACIÓN
// ═══════════════════════════════════════════════════════════════
var municipios = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM0_NAME', 'Colombia'));

// ═══════════════════════════════════════════════════════════════
// PARTE A: CHIRPS — Precipitación diaria (1 solo export, ~30 min)
// ═══════════════════════════════════════════════════════════════

function extraerChirps() {
  var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate('2018-01-01', '2026-05-31')
    .select('precipitation');
  
  var diario = chirps.map(function(img) {
    var date = img.date().format('YYYY-MM-dd');
    var stats = img.reduceRegions({
      collection: municipios,
      reducer: ee.Reducer.mean(),
      scale: 5566
    });
    return stats.map(function(f) {
      return f.set('fecha', date);
    });
  }).flatten();
  
  Export.table.toDrive({
    collection: diario,
    description: 'CHIRPS_COLOMBIA_DIARIO_2018_2026',
    folder: 'COLOMBIA_FLOOD_PIPELINE',
    fileNamePrefix: 'CHIRPS_COLOMBIA_DIARIO_2018_2026',
    fileFormat: 'CSV',
    selectors: ['ADM1_NAME', 'ADM2_NAME', 'fecha', 'mean']
  });
  
  print('CHIRPS: Export iniciado. ~4M filas. Tiempo estimado: 30-60 min.');
}


// ═══════════════════════════════════════════════════════════════
// PARTE B: SAR Sentinel-1 — Backscatter mensual (3 exports)
// ═══════════════════════════════════════════════════════════════
// Partido en 3 períodos porque 1,122 municipios × 102 meses = timeout.

function extraerSARPeriodo(startDate, endDate, label) {
  function maskEdge(img) {
    var edge = img.lt(-30.0);
    var masked = img.mask().and(edge.not());
    return img.updateMask(masked);
  }
  
  var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(municipios.geometry())
    .filterDate(startDate, endDate)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .map(maskEdge);
  
  var totalMonths = ee.Date(endDate).difference(ee.Date(startDate), 'month').round();
  var months = ee.List.sequence(0, totalMonths.subtract(1));
  
  var monthlyFeatures = ee.FeatureCollection(months.map(function(m) {
    var s = ee.Date(startDate).advance(m, 'month');
    var e = s.advance(1, 'month');
    
    var monthlyS1 = s1.filterDate(s, e);
    var dateLabel = s.format('YYYY-MM');
    
    var mean = monthlyS1.select(['VV', 'VH']).mean()
      .rename(['VV_mean', 'VH_mean']);
    
    var stdDev = monthlyS1.select(['VV', 'VH'])
      .reduce(ee.Reducer.stdDev())
      .rename(['VV_stdDev', 'VH_stdDev']);
    
    var ratio = mean.select('VV_mean').subtract(mean.select('VH_mean'))
      .rename('VVVH_ratio');
    
    var count = monthlyS1.select('VV').map(function(img) {
      return ee.Image.constant(1).rename('n_scenes');
    }).sum();
    
    var composite = mean.addBands(stdDev).addBands(ratio).addBands(count);
    
    var reduced = composite.reduceRegions({
      collection: municipios,
      reducer: ee.Reducer.mean(),
      scale: 20
    });
    
    return reduced.map(function(feature) {
      return feature.set({'fecha_mes': dateLabel});
    });
  })).flatten();
  
  var finalFeatures = monthlyFeatures.filter(
    ee.Filter.notNull(['VV_mean', 'VH_mean'])
  );
  
  Export.table.toDrive({
    collection: finalFeatures,
    description: 'SAR_COLOMBIA_MENSUAL_' + label,
    folder: 'COLOMBIA_FLOOD_PIPELINE',
    fileNamePrefix: 'SAR_COLOMBIA_MENSUAL_' + label,
    fileFormat: 'CSV',
    selectors: [
      'ADM1_NAME', 'ADM2_NAME', 'fecha_mes',
      'VV_mean', 'VH_mean', 'VV_stdDev', 'VH_stdDev',
      'VVVH_ratio', 'n_scenes'
    ]
  });
  
  print('SAR ' + label + ': Export iniciado.');
}


// ═══════════════════════════════════════════════════════════════
// EJECUTAR
// ═══════════════════════════════════════════════════════════════
extraerChirps();    // 1 export,  ~30-60 min

// SAR partido en 9 años para evitar timeout (1,122 municipios es mucha carga):
extraerSARPeriodo('2018-01-01', '2018-12-31', '2018');
extraerSARPeriodo('2019-01-01', '2019-12-31', '2019');
extraerSARPeriodo('2020-01-01', '2020-12-31', '2020');
extraerSARPeriodo('2021-01-01', '2021-12-31', '2021');
extraerSARPeriodo('2022-01-01', '2022-12-31', '2022');
extraerSARPeriodo('2023-01-01', '2023-12-31', '2023');
extraerSARPeriodo('2024-01-01', '2024-12-31', '2024');
extraerSARPeriodo('2025-01-01', '2025-12-31', '2025');
extraerSARPeriodo('2026-01-01', '2026-05-31', '2026');

// ═══════════════════════════════════════════════════════════════
// VERIFICACIÓN
// ═══════════════════════════════════════════════════════════════
Map.centerObject(municipios, 5);
Map.addLayer(municipios, {color: 'blue'}, 'Municipios Colombia');
print('Total municipios:', municipios.size());
print('10 exports iniciados (1 CHIRPS + 9 SAR). Revisá la pestaña Tasks.');
print('Cuando terminen, descargá los CSVs de Drive a FASE-4/gee/');
print('Después: python FASE-4/gee/procesar_exports.py');
