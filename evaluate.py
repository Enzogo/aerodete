#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUATE.PY - Evaluar modelo y generar gráficos con TensorFlow + Librosa
Genera: Matriz de confusión + Curva ROC + Métricas
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

# Scikit-learn - Métricas
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Matplotlib - Visualización
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("EVALUAR MODELO - GENERAR GRÁFICOS")
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
# CARGAR MODELO
# ============================================================================

print("[1/5] Cargando modelo...")
try:
    model = tf.keras.models.load_model("models/audio_model_working.h5")
    print(f"✓ Modelo cargado: {model.count_params():,} parámetros")
except Exception as e:
    print(f"✗ Error cargando modelo: {e}")
    print("Intenta ejecutar: python train.py")
    exit(1)

# ============================================================================
# CARGAR DATASET CON LIBROSA
# ============================================================================

print("[2/5] Cargando dataset...")
X_data = []
y_data = []

for label_idx, label in enumerate(LABELS):
    label_path = Path(f"dataset/{label}")
    audio_files = list(label_path.glob("*.wav"))
    print(f"  {label:15} : {len(audio_files):4} archivos")
    
    for audio_file in audio_files:
        try:
            # Cargar con Librosa
            y, sr = librosa.load(str(audio_file), sr=SR, duration=DURATION)
            
            # Extraer MFCC con Librosa
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N, 
                                       n_fft=FFT_SIZE, hop_length=FFT_SIZE//2)
            mfcc = mfcc.T
            
            # Redimensionar
            if mfcc.shape[0] > 170:
                mfcc = mfcc[:170, :]
            else:
                mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
            
            X_data.append(mfcc)
            y_data.append(label_idx)
        except:
            pass

X_data = np.array(X_data)
y_data = np.array(y_data)
print(f"\n✓ Total de muestras: {len(X_data)}")

# ============================================================================
# NORMALIZACIÓN
# ============================================================================

print("\n[3/5] Normalizando features...")
X_mean = np.mean(X_data, axis=0)
X_std = np.std(X_data, axis=0)
X_data = (X_data - X_mean) / (X_std + 1e-7)

# ============================================================================
# PREDICCIONES
# ============================================================================

print("[4/5] Generando predicciones...")
y_pred_probs = model.predict(X_data, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# ============================================================================
# MÉTRICAS
# ============================================================================

print("[5/5] Generando gráficos...\n")

# Crear carpeta de resultados
Path("metrics/plots").mkdir(parents=True, exist_ok=True)
Path("metrics/reports").mkdir(parents=True, exist_ok=True)

# 1. MATRIZ DE CONFUSIÓN
print("  ✓ Generando matriz de confusión...")
cm = confusion_matrix(y_data, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar_kws={'label': 'Cantidad'})
ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
ax.set_ylabel('Real', fontsize=12)
ax.set_xlabel('Predicho', fontsize=12)
plt.tight_layout()
plt.savefig('metrics/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. MATRIZ DE CONFUSIÓN NORMALIZADA
print("  ✓ Matriz de confusión normalizada...")
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=LABELS, yticklabels=LABELS, ax=ax, cbar_kws={'label': 'Porcentaje'})
ax.set_title('Matriz de Confusión Normalizada', fontsize=14, fontweight='bold')
ax.set_ylabel('Real', fontsize=12)
ax.set_xlabel('Predicho', fontsize=12)
plt.tight_layout()
plt.savefig('metrics/plots/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. CURVA ROC (One-vs-Rest)
print("  ✓ Generando curva ROC...")
y_data_bin = label_binarize(y_data, classes=range(len(LABELS)))
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for i in range(len(LABELS)):
    fpr, tpr, _ = roc_curve(y_data_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{LABELS[i]} (AUC = {roc_auc:.4f})', 
            linewidth=2.5, color=colors[i])

ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.50)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Curva ROC - Análisis por Clase', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('metrics/plots/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. DISTRIBUCIÓN DE CONFIANZA
print("  ✓ Distribución de confianza...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, label in enumerate(LABELS):
    ax = axes[i // 2, i % 2]
    confidences = y_pred_probs[:, i] * 100
    ax.hist(confidences, bins=30, color=colors[i], alpha=0.7, edgecolor='black')
    ax.set_title(f'Confianza para {label}', fontweight='bold')
    ax.set_xlabel('Confianza (%)')
    ax.set_ylabel('Frecuencia')
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('metrics/plots/confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. PRECISIÓN POR CLASE
print("  ✓ Precisión por clase...")
class_accuracies = []
for i in range(len(LABELS)):
    mask = y_data == i
    acc = np.mean(y_pred[mask] == y_data[mask])
    class_accuracies.append(acc * 100)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(LABELS, class_accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylim([0, 105])
ax.set_ylabel('Precisión (%)', fontsize=12)
ax.set_title('Precisión por Clase', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Añadir valores en barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('metrics/plots/accuracy_per_class.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# REPORTE DE CLASIFICACIÓN
# ============================================================================

print("  ✓ Reporte de clasificación...")
# Usar solo las clases presentes en los datos
unique_labels = sorted(np.unique(np.concatenate([y_data, y_pred])))
label_names = [LABELS[i] for i in unique_labels]
report = classification_report(y_data, y_pred, labels=unique_labels, target_names=label_names)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
report_text = f"""AUDIO CLASSIFICATION REPORT
================================================================================
Fecha: {timestamp}
Total muestras: {len(y_data)}
Accuracy general: {np.mean(y_pred == y_data)*100:.2f}%

{report}

================================================================================
MATRIZ DE CONFUSION
================================================================================
{cm}

Clases: {LABELS}
"""

with open('metrics/reports/classification_report.txt', 'w') as f:
    f.write(report_text)

# ============================================================================
# RESULTADOS JSON
# ============================================================================

results = {
    'timestamp': timestamp,
    'total_samples': int(len(y_data)),
    'accuracy': float(np.mean(y_pred == y_data)),
    'confusion_matrix': cm.tolist(),
    'class_accuracies': {label: float(acc) for label, acc in zip(LABELS, class_accuracies)},
    'labels': LABELS
}

with open('metrics/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ============================================================================
# RESUMEN
# ============================================================================

print("\n" + "="*80)
print("✓ EVALUACIÓN COMPLETADA")
print("="*80)
print("\nGráficos generados en metrics/plots/:")
print("  • confusion_matrix.png")
print("  • confusion_matrix_normalized.png")
print("  • roc_curve.png")
print("  • confidence_distribution.png")
print("  • accuracy_per_class.png")
print("\nReportes en metrics/reports/:")
print("  • classification_report.txt")
print("\nMétricas en metrics/:")
print("  • results.json")
print("\nAccuracy por clase:")
for label, acc in zip(LABELS, class_accuracies):
    print(f"  {label:15}: {acc:6.2f}%")
print("\n" + "="*80 + "\n")
