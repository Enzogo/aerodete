# AERODETECT - Clasificacion de Sonidos Aereos

Sistema de clasificacion de audio usando Deep Learning con TensorFlow.

## Caracteristicas

- 3 Clases: Avion, Dron, Helicoptero
- Precision: 99.22% en validacion
- Dataset: 2550 audios de entrenamiento
- Framework: TensorFlow 2.13 + Librosa

## Modelo Entrenado

- Arquitectura: CNN 1D (3 capas convolucionales)
- Parametros: 174819
- Entrada: MFCC 40x170
- Accuracy: 99.22% (validation)
- Confianza: 0.9999 promedio

## Validacion

Resultados con 450 audios (150 por clase):
- Avion: 100% (150/150)
- Dron: 100% (150/150)
- Helicoptero: 99.33% (149/150)
- Accuracy General: 99.67%

## Dataset

Total: 2550 audios
- Avion: 750 audios (22050 Hz, 3 segundos)
- Dron: 1050 audios (22050 Hz, 3 segundos)
- Helicoptero: 750 audios (22050 Hz, 3 segundos)

## Instalacion

pip install -r requirements.txt

## Uso

Clasificar un audio:

import tensorflow as tf
import librosa
import pickle
import numpy as np

model = tf.keras.models.load_model('models/audio_model_working.h5')

with open('models/normalization.pkl', 'rb') as f:
    norm = pickle.load(f)

y, sr = librosa.load('audio.wav', sr=22050)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc = mfcc.T

mfcc_norm = (mfcc - norm['mean']) / norm['std']
mfcc_expanded = np.expand_dims(mfcc_norm, axis=0)

predictions = model.predict(mfcc_expanded)
class_names = ['avion', 'dron', 'helicoptero']
predicted_class = class_names[np.argmax(predictions)]
confidence = np.max(predictions)

print(f"Prediccion: {predicted_class} ({confidence:.2%})")

## Archivos del Proyecto

models/
  audio_model_working.h5    - Modelo CNN 1D entrenado
  normalization.pkl         - Normalizador MFCC
dataset/
  avion/                    - 750 audios
  dron/                     - 1050 audios
  helicoptero/              - 750 audios
metrics/
  plots/
    01_confusion_matrix.png
    02_learning_curves.png
    03_summary.png
    04_confidence_analysis.png
requirements.txt            - Dependencias
README.md                   - Este archivo
training_history.json       - Historico

## Metricas

Accuracy por Clase
- Avion: 100%
- Dron: 100%
- Helicoptero: 99.33%
- Promedio: 99.67%

Confianza
- Promedio: 0.9999
- Minima: 0.9906
- Maxima: 1.0000

## Entrenamiento

- Epocas: 200
- Batch Size: 16
- Learning Rate: 0.0005
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Callback: ReduceLROnPlateau
- Accuracy Final: 1.0000
- Validation Accuracy: 0.9922 (99.22%)

## Autor

Proyecto AERODETECT - Clasificacion de sonidos aereos

## Licencia

MIT
