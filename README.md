# AeroDetect - Sistema de Clasificación de Sonidos Aeronáuticos

## Estructura Final

### Archivos Python Principales

```
analysis_metrics.py  - Análisis de métricas y generación de reportes
api.py              - API REST para predicciones
gui.py              - Interfaz gráfica básica
gui_improved.py     - Interfaz gráfica mejorada con soporte YouTube
```

### Modelos

```
models/
  ├── audio_model_robusto_v3.keras          (Modelo en formato Keras)
  ├── audio_model_robusto_v3.h5             (Modelo en formato HDF5)
  └── normalization_robusto_v3.pkl          (Normalizador RobustScaler)
```

---

## Cómo Usar

### 1. Análisis de Métricas (150 audios: 50 por clase)

```bash
python analysis_metrics.py
```

**Genera:**
- `metrics/plots/analysis_metrics.png` - Gráficos de matriz de confusión y accuracy
- `metrics/json/analysis_results.json` - Resultados en JSON
- `metrics/reports/resumen_YYYYMMDD_HHMMSS.txt` - Reporte detallado

**Espera:** ~2-3 minutos (150 audios × modelo)

---

### 2. Interfaz Gráfica

#### Opción 1: GUI Básica
```bash
python gui.py
```
- Carga archivos locales
- Muestra predicción y confianza

#### Opción 2: GUI Mejorada (Recomendado)
```bash
python gui_improved.py
```
- Carga archivos locales **O** URL de YouTube
- Descarga automático de audio
- Visualización de espectrograma
- Confianza por cada clase

---

### 3. API REST

```bash
python api.py
```

Accede en: `http://localhost:5000`

**Endpoints:**
- `GET /` - Estado del servidor
- `POST /predict` - Predicción de audio

**Ejemplo:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "audio=@audio.wav"
```

---

## Características Clave

✅ **Modelo V3 Balanceado**
- 300 épocas de entrenamiento
- Class weights para balancear clases
- Accuracy 98.35%

✅ **3 Clases Soportadas**
- Avión
- Dron
- Helicóptero

✅ **Procesamiento Uniforme**
- 40 MFCCs
- 170 frames
- RobustScaler normalización

✅ **Herramientas de Análisis**
- Matriz de confusión
- Accuracy por clase
- Gráficos automatizados

---

## Entrenamiento del Modelo

Si necesitas reentrenar el modelo:

1. Prepara datos en: `dataset_selected/avion/`, `dataset_selected/dron/`, `dataset_selected/helicoptero/`
2. Crea un script de entrenamiento con:
   - 300 épocas
   - Class weights balanceados
   - RobustScaler para normalización
   - Architecture: Conv1D (64-128-256-512) + GlobalAveragePooling + Dense

---

## Solución de Problemas

### "No se encontró normalizador"
```bash
# Regenerar normalizador
python gen_scaler.py
```

### "Accuracy bajo en analysis_metrics"
- Verifica que `normalization_robusto_v3.pkl` existe
- Usa `transform()` del scaler, no `fit_transform()`

### Gráficos no se generan
- Verifica carpeta `metrics/plots/` existe
- Asegúrate de tener permisos de escritura

---

## Información Técnica

**Modelo:**
- Tipo: CNN Conv1D
- Parámetros: 594,627
- Input shape: (170, 40)
- Output: 3 clases con softmax

**Audio Processing:**
- Sample rate: 22050 Hz
- Duration: 4 segundos
- MFCC: n_mfcc=40, n_fft=2048
- Padding: 170 frames

**Normalizador:**
- Tipo: RobustScaler (sklearn)
- Entrenado con 300 audios (100 por clase)
- Percentilos: 25, 75

---

**Última actualización:** 2025-12-10
**Versión:** V3.1 (Balanceada)
