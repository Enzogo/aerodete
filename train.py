#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRAIN.PY - Entrenar modelo CNN 1D con TensorFlow + Librosa
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime

# TensorFlow
import tensorflow as tf

# Librosa - Audio Processing
import librosa
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("ENTRENAR MODELO - AUDIO CLASSIFICATION CON TENSORFLOW + LIBROSA")
print("="*80 + "\n")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LABELS = ['avion', 'dron', 'helicoptero']
SR = 22050
DURATION = 4.0
MFCC_N = 40
FFT_SIZE = 512

# ============================================================================
# CARGAR DATASET CON LIBROSA
# ============================================================================

def load_all_audios():
    """Cargar todos los audios del dataset usando Librosa"""
    X_data = []
    y_data = []
    
    print("Cargando dataset...")
    for label_idx, label in enumerate(LABELS):
        label_path = Path(f"dataset/{label}")
        audio_files = list(label_path.glob("*.wav"))
        
        print(f"  {label:15} : ", end="", flush=True)
        
        for audio_file in audio_files:
            try:
                # Cargar con Librosa
                y, sr = librosa.load(str(audio_file), sr=SR, duration=DURATION)
                
                # Extraer MFCC con Librosa
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N, 
                                           n_fft=FFT_SIZE, hop_length=FFT_SIZE//2)
                mfcc = mfcc.T  # (time_steps, features)
                
                # Redimensionar a (170, 40)
                if mfcc.shape[0] > 170:
                    mfcc = mfcc[:170, :]
                else:
                    mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
                
                X_data.append(mfcc)
                y_data.append(label_idx)
            except Exception as e:
                pass
        
        print(f"{len(audio_files):4} archivos")
    
    return np.array(X_data), np.array(y_data)

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def augment_data(X, y, factor=3):
    """Data augmentation - crear múltiples variaciones"""
    X_aug = [X]
    y_aug = [y]
    
    for _ in range(factor - 1):
        # Ruido gaussiano
        noise = np.random.normal(0, 0.01, X.shape)
        X_aug.append(X + noise)
        y_aug.append(y)
    
    return np.vstack(X_aug), np.hstack(y_aug)

# ============================================================================
# CARGAR DATOS
# ============================================================================

X_data, y_data = load_all_audios()
print(f"\nTotal de muestras cargadas: {len(X_data)}")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

print("\nAplicando data augmentation (3x)...")
X_combined, y_combined = augment_data(X_data, y_data, factor=3)
print(f"Total después de augmentation: {len(X_combined)}")

# ============================================================================
# NORMALIZACIÓN
# ============================================================================

print("\nNormalizando features...")
X_mean = np.mean(X_combined, axis=0)
X_std = np.std(X_combined, axis=0)
X_combined = (X_combined - X_mean) / (X_std + 1e-7)

# ============================================================================
# SPLIT TRAIN/TEST
# ============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42
)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ============================================================================
# CLASS WEIGHTS
# ============================================================================

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print(f"Class weights: {class_weight_dict}")

# ============================================================================
# CONSTRUIR MODELO CON TENSORFLOW
# ============================================================================

print("\nCreando modelo con TensorFlow...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(170, 40)),
    
    # Block 1
    tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Block 2
    tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Block 3
    tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Global pooling
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # Dense layers
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    
    # Output
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Modelo: {model.count_params():,} parámetros")

# ============================================================================
# ENTRENAR
# ============================================================================

print("\nEntrenando...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    verbose=1
)

# ============================================================================
# GUARDAR MODELO
# ============================================================================

Path("models").mkdir(exist_ok=True)
model.save("models/audio_model_working.h5")
print(f"\n✓ Modelo guardado: models/audio_model_working.h5")

# Guardar history
with open("models/training_history.json", 'w') as f:
    json.dump({
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
    }, f)

print("✓ History guardado: models/training_history.json")
print("\n" + "="*80 + "\n")
