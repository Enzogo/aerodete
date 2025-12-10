# README.md - AeroDetect

## 🎵 AeroDetect - Clasificación de Sonidos de Aeronaves

Sistema de Deep Learning para clasificar sonidos de aviones, drones y helicópteros en tiempo real usando **TensorFlow**, **Librosa** y **Flask**.

---

## ✨ Características

✅ **Modelo CNN 1D** entrenado con TensorFlow  
✅ **Procesamiento de audio** con Librosa (MFCC features)  
✅ **GUI interactiva** con Tkinter  
✅ **API REST** con Flask  
✅ **Matriz de confusión** y análisis ROC automático  
✅ **4 clases**: Avión, Dron, Helicóptero, Ruido

---

## 🏗️ Arquitectura del Modelo

```
Input (170, 40) MFCC Features
    ↓
Conv1D(64, 7) → BatchNorm → Dropout(0.3)
    ↓
Conv1D(128, 5) → BatchNorm → Dropout(0.3)
    ↓
Conv1D(256, 3) → BatchNorm → Dropout(0.3)
    ↓
GlobalAveragePooling1D
    ↓
Dense(256) → Dropout(0.4) → BatchNorm
    ↓
Dense(128) → Dropout(0.3)
    ↓
Dense(4, softmax)
Output: [avion, dron, helicoptero, ruido]
```

**Parámetros totales**: 259,652

---

## 📦 Instalación

```bash
# Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales:
- **TensorFlow 2.13.0** - Deep Learning
- **Librosa 0.10.0** - Audio processing (MFCC)
- **Flask 2.3.3** - API REST
- **Scikit-learn 1.3.0** - Métricas y validación
- **Matplotlib + Seaborn** - Visualización

---

## 🚀 Uso Rápido

### 1. Entrenar modelo (OPCIONAL)
```bash
python train.py
```
- Lee 2,656 audios del dataset
- Extrae 40 MFCC features con Librosa
- Entrena 50 épocas con TensorFlow
- Guarda modelo en `models/audio_model_working.h5`

**Duración**: ~45 minutos

### 2. Generar gráficos y métricas
```bash
python evaluate.py
```

**Genera**:
- ✓ Matriz de confusión (PNG)
- ✓ Curva ROC (PNG)
- ✓ Precisión por clase (PNG)
- ✓ Distribución de confianza (PNG)
- ✓ Reporte de clasificación (TXT)
- ✓ Métricas (JSON)

**Duración**: ~2-3 minutos

### 3. GUI Interactiva
```bash
python app.py
```

**Características**:
- Tab 1: Clasificación individual
  - Cargar archivo .wav
  - Predicción en tiempo real
  - Visualizar MFCC
  - Análisis de frecuencias

- Tab 2: Evaluación dataset
  - Evaluar todos los audios
  - Ver matriz de confusión
  - Ver curva ROC
  - Métricas por clase

### 4. API REST (Flask)
```bash
python api.py
```

**Endpoints**:
```bash
# Predecir audio
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict

# Estado del sistema
curl http://localhost:5000/status

# Información del modelo
curl http://localhost:5000/model-info
```

---

## 📊 Dataset

```
dataset/
├── avion/           674 archivos (25.4%)
├── dron/          1,001 archivos (37.7%)
├── helicoptero/     353 archivos (13.3%)
└── ruido/           628 archivos (23.6%)

Total: 2,656 archivos de audio
```

---

## 📁 Estructura del Proyecto

```
AeroDetect/
├── 📜 ARCHIVOS PRINCIPALES
│   ├── app.py                 GUI interactiva (Tkinter)
│   ├── train.py              Entrenar modelo (TensorFlow)
│   ├── evaluate.py           Evaluar y generar gráficos
│   ├── api.py                API REST (Flask)
│   ├── README.md             Esta documentación
│   ├── GUIA_RAPIDA.md       Guía en español
│   └── requirements.txt      Dependencias
│
├── 📊 DATASET
│   └── dataset/
│       ├── avion/
│       ├── dron/
│       ├── helicoptero/
│       └── ruido/
│
├── 🤖 MODELO
│   └── models/
│       └── audio_model_working.h5
│
└── 📈 RESULTADOS
    └── metrics/
        ├── plots/            5 gráficos PNG
        ├── reports/          Reportes TXT
        └── results.json      Métricas JSON
```

---

## 🧠 Tecnologías Utilizadas

| Tecnología | Versión | Uso |
|------------|---------|-----|
| **TensorFlow** | 2.13.0 | Deep Learning - Modelo CNN 1D |
| **Librosa** | 0.10.0 | Audio Processing - Extracción MFCC |
| **Flask** | 2.3.3 | API REST - Servidor web |
| **Scikit-learn** | 1.3.0 | Métricas - Matriz confusión, ROC |
| **NumPy** | 1.24.3 | Computación numérica |
| **Matplotlib** | 3.7.2 | Visualización - Gráficos |
| **Seaborn** | 0.12.2 | Gráficos estadísticos |
| **Tkinter** | Built-in | GUI - Interfaz gráfica |

---

## 📊 Métricas Esperadas

```
Accuracy General:  ~98.7%

Por clase:
- Avión:      95%+ precisión
- Dron:       85%+ precisión
- Helicóptero: 90%+ precisión
- Ruido:      80%+ precisión
```

---

## 💡 Ejemplos de Uso

### Python: Cargar y predecir
```python
import tensorflow as tf
import librosa
import numpy as np

# Cargar modelo
model = tf.keras.models.load_model("models/audio_model_working.h5")

# Cargar audio con Librosa
y, sr = librosa.load("audio.wav", sr=22050)

# Extraer MFCC
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc = mfcc.T

# Predecir
prediction = model.predict(np.expand_dims(mfcc, axis=0))
print(prediction)
```

### API REST: Predecir
```bash
curl -X POST -F "file=@audio.wav" http://localhost:5000/predict

# Respuesta:
{
  "predicted_class": "dron",
  "confidence": 95.3,
  "probabilities": {
    "avion": 2.1,
    "dron": 95.3,
    "helicoptero": 1.5,
    "ruido": 1.1
  }
}
```

---

## 🔧 Troubleshooting

### Error: Modelo no encontrado
```bash
python train.py  # Entrenar nuevo modelo
```

### Error: Librosa no encontrado
```bash
pip install librosa
```

### Error: TensorFlow no compatible
```bash
pip install --upgrade tensorflow
```

### La GUI no se abre
```bash
# Asegurate de estar en el directorio correcto
cd c:\Users\enzog\OneDrive\Escritorio\Programacion\AeroDetect
python app.py
```

---

## 📝 Licencia

Proyecto académico de clasificación de audio.

---

## 👨‍💻 Autor

Creado con TensorFlow, Librosa y Flask.

---

**Documentación completa**: Ver `GUIA_RAPIDA.md` para Spanish guide  
**Stack tecnológico**: Ver `STACK_TECNOLOGICO.md`
