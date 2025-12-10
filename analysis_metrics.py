"""
AERODETECT - ANALISIS Y GENERACION DE METRICAS
Versión con gráficos separados y flexible para cualquier cantidad de audios
"""

import os
import sys
import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Configurar encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("\n" + "="*80)
print("AERODETECT - ANALISIS Y GENERACION DE METRICAS")
print("="*80)

# [1] Cargar modelo
print("\n[1] Cargando modelo...")

model_v3_keras = Path('models/audio_model_robusto_v3.keras')
model_v3 = Path('models/audio_model_robusto_v3.h5')
model_v2 = Path('models/audio_model_final.h5')

if model_v3_keras.exists():
    model_path = str(model_v3_keras)
    model_version = "v3"
elif model_v3.exists():
    model_path = str(model_v3)
    model_version = "v3"
elif model_v2.exists():
    model_path = str(model_v2)
    model_version = "v2"
else:
    print("[ERROR] No hay modelo disponible")
    exit(1)

try:
    model = load_model(model_path, compile=False)
    print(f"[OK] Modelo {model_version} cargado: {model.count_params():,} parámetros")
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

# [2] Cargar normalizador
print("\n[2] Cargando normalizador...")

scaler_path = Path('models/normalization_robusto_v3.pkl')
if not scaler_path.exists():
    print("[ERROR] No se encontro normalizador!")
    exit(1)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"[OK] Normalizador cargado: {scaler_path.name}")

CLASSES = ['avion', 'dron', 'helicoptero']

# [3] Cargar audios de prueba (100+ POR CLASE)
print("\n[3] Cargando audios de prueba (100+ por clase)...")

dataset_base = Path('dataset_selected') if Path('dataset_selected').exists() else Path('test_audios')
print(f"    Usando directorio: {dataset_base}")

test_results = {cls: [] for cls in CLASSES}
y_true = []
y_pred = []
all_predictions = []

MIN_AUDIOS_PER_CLASS = 100

for class_idx, cls in enumerate(CLASSES):
    test_dir = dataset_base / cls
    
    if not test_dir.exists():
        print(f"[!] Directorio no encontrado: {test_dir}")
        continue
    
    # Buscar archivos de audio
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']:
        audio_files.extend(test_dir.glob(ext))
    
    audio_files = sorted(audio_files)
    
    # Seleccionar primeros 100+ audios (distribuidos uniformemente si hay más)
    if len(audio_files) > MIN_AUDIOS_PER_CLASS:
        # Tomar cada N-ésimo audio para distribuir a lo largo del conjunto
        step = len(audio_files) // MIN_AUDIOS_PER_CLASS
        selected_files = audio_files[::step][:MIN_AUDIOS_PER_CLASS]
    else:
        selected_files = audio_files
    
    audio_files = selected_files
    print(f"\n  {cls.upper()}: {len(audio_files)} audios seleccionados de {len(sorted(test_dir.glob('*.wav')))} disponibles")
    
    for i, filepath in enumerate(audio_files):
        try:
            # Cargar y procesar audio
            y, sr = librosa.load(str(filepath), sr=22050, duration=4.0)
            
            # Extraer MFCC (40 coeficientes, 170 frames)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
            
            # Padding a 170 frames
            if mfcc.shape[0] < 170:
                mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
            else:
                mfcc = mfcc[:170]
            
            # NORMALIZAR CON SCALER GUARDADO
            mfcc_flat = mfcc.flatten()
            mfcc_norm_flat = scaler.transform(mfcc_flat.reshape(1, -1))
            mfcc_norm = mfcc_norm_flat.reshape(170, 40)
            
            # Predecir
            pred = model.predict(np.expand_dims(mfcc_norm, 0), verbose=0)
            pred_class_idx = np.argmax(pred[0])
            
            pred_class = CLASSES[pred_class_idx]
            is_correct = pred_class == cls
            
            test_results[cls].append({
                'filename': filepath.name,
                'prediccion': pred_class,
                'confianzas': {CLASSES[i]: float(pred[0][i]) for i in range(len(CLASSES))},
                'confianza': float(pred[0][pred_class_idx]),
                'correcto': is_correct
            })
            
            all_predictions.append({
                'real': cls,
                'prediccion': pred_class,
                'confianzas': {CLASSES[i]: float(pred[0][i]) for i in range(len(CLASSES))}
            })
            
            y_true.append(class_idx)
            y_pred.append(pred_class_idx)
            
            if (i + 1) % 50 == 0:
                print(f"    Procesados: {i+1}/{len(audio_files)}")
            
        except Exception as e:
            print(f"    [!] Error en {filepath.name}: {e}")

# [4] Calcular métricas
print(f"\n[4] Calculando métricas...")

if len(y_true) > 0:
    overall_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n[OK] ACCURACY GENERAL: {overall_accuracy:.2%}")
    print(f"[OK] Total de audios procesados: {len(y_true)}")
    print(f"\n[OK] ACCURACY POR CLASE:")
    accuracies = {}
    for i, cls in enumerate(CLASSES):
        results = test_results[cls]
        if results:
            correct = sum(1 for r in results if r['correcto'])
            total = len(results)
            acc = correct / total
            accuracies[cls] = acc
            print(f"    {cls:12}: {acc:.2%} ({correct}/{total})")
    
    # [5] Generar visualizaciones SEPARADAS
    print("\n[5] Generando gráficos separados...")
    
    output_dir = Path('metrics/plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # GRAFICO 1: Matriz de Confusión
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax, cbar_kws={'label': 'Cantidad'})
        ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        ax.set_ylabel('Clase Real', fontsize=12)
        ax.set_xlabel('Predicción del Modelo', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=100, bbox_inches='tight')
        print("[OK] confusion_matrix.png")
        plt.close()
    except Exception as e:
        print(f"[!] Error: {e}")
    
    # GRAFICO 2: Accuracy por Clase
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        accs = [accuracies.get(c, 0) for c in CLASSES]
        colors = ['green' if a >= 0.95 else 'orange' if a >= 0.80 else 'red' for a in accs]
        
        bars = ax.bar(CLASSES, accs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy por Clase', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(0.95, color='green', linestyle='--', linewidth=2, alpha=0.5, label='95% (excelente)')
        ax.axhline(0.80, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='80% (bueno)')
        ax.legend()
        
        # Agregar valores en las barras
        for i, (bar, acc) in enumerate(zip(bars, accs)):
            results = test_results[CLASSES[i]]
            correct = sum(1 for r in results if r['correcto'])
            total = len(results)
            ax.text(bar.get_x() + bar.get_width()/2, acc + 0.02, 
                   f'{acc:.1%}\n({correct}/{total})', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_by_class.png', dpi=100, bbox_inches='tight')
        print("[OK] accuracy_by_class.png")
        plt.close()
    except Exception as e:
        print(f"[!] Error: {e}")
    
    # GRAFICO 3: Confianza Promedio por Clase
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        confs = []
        conf_stds = []
        
        for cls in CLASSES:
            results = test_results[cls]
            if results:
                confs_list = [r['confianza'] for r in results]
                confs.append(np.mean(confs_list))
                conf_stds.append(np.std(confs_list))
            else:
                confs.append(0)
                conf_stds.append(0)
        
        bars = ax.bar(CLASSES, confs, yerr=conf_stds, capsize=5, 
                     color='skyblue', alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Confianza (Confidence)', fontsize=12, fontweight='bold')
        ax.set_title('Confianza Promedio por Clase', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(0.95, color='green', linestyle='--', linewidth=2, alpha=0.5)
        
        # Agregar valores en las barras
        for bar, conf, std in zip(bars, confs, conf_stds):
            ax.text(bar.get_x() + bar.get_width()/2, conf + std + 0.02, 
                   f'{conf:.3f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'confidence_by_class.png', dpi=100, bbox_inches='tight')
        print("[OK] confidence_by_class.png")
        plt.close()
    except Exception as e:
        print(f"[!] Error: {e}")
    
    # [6] Generar reporte JSON
    print("\n[6] Generando reporte JSON...")
    
    report = {
        'fecha': datetime.now().isoformat(),
        'modelo_version': model_version,
        'normalizador': scaler_path.name,
        'total_audios': len(y_true),
        'accuracy_general': float(overall_accuracy),
        'accuracy_por_clase': {cls: float(accuracies.get(cls, 0)) for cls in CLASSES},
        'confianza_promedio': {cls: float(np.mean([r['confianza'] for r in test_results[cls]])) if test_results[cls] else 0 for cls in CLASSES},
        'matriz_confusion': cm.tolist(),
        'audios_por_clase': {cls: len(test_results[cls]) for cls in CLASSES},
        'detalles': all_predictions
    }
    
    json_file = Path('metrics/json/analysis_results.json')
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[OK] analysis_results.json")
    
    # [7] Generar reporte TXT
    print("\n[7] Generando reporte TXT...")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    txt_file = Path(f'metrics/reports/resumen_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("                    AERODETECT - REPORTE DE ANALISIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Fecha: {timestamp}\n")
        f.write(f"Total de Audios Analizados: {len(y_true)}\n")
        f.write(f"Normalizador: {scaler_path.name}\n")
        f.write(f"Modelo: {model_version}\n\n")
        
        f.write("RESULTADOS GENERALES\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy General: {overall_accuracy:.2%}\n\n")
        
        f.write("ACCURACY POR CLASE\n")
        f.write("="*80 + "\n")
        for cls in CLASSES:
            acc = accuracies.get(cls, 0)
            results = test_results[cls]
            correct = sum(1 for r in results if r['correcto'])
            total = len(results)
            status = "OK" if acc >= 0.95 else "WARN" if acc >= 0.80 else "FAIL"
            f.write(f"{cls:15} {acc:.2%} ({correct}/{total}) [{status}]\n")
        
        f.write("\n\nMATRIZ DE CONFUSION\n")
        f.write("="*80 + "\n")
        f.write(f"                Predicción\n")
        f.write(f"                {CLASSES[0]:>10} {CLASSES[1]:>10} {CLASSES[2]:>10}\n")
        for i, cls in enumerate(CLASSES):
            f.write(f"Real    {cls:>8}")
            for j in range(len(CLASSES)):
                f.write(f" {cm[i,j]:>10}")
            f.write("\n")
        
        f.write("\n\nCONFIANZA PROMEDIO\n")
        f.write("="*80 + "\n")
        for cls in CLASSES:
            results = test_results[cls]
            if results:
                avg_conf = np.mean([r['confianza'] for r in results])
                std_conf = np.std([r['confianza'] for r in results])
                f.write(f"{cls:15} {avg_conf:.4f} (±{std_conf:.4f})\n")
        
        f.write("\n\nARCHIVOS GENERADOS\n")
        f.write("="*80 + "\n")
        f.write("OK metrics/plots/confusion_matrix.png\n")
        f.write("OK metrics/plots/accuracy_by_class.png\n")
        f.write("OK metrics/plots/confidence_by_class.png\n")
        f.write("OK metrics/json/analysis_results.json\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"[OK] Reporte TXT guardado")
    
    print("\n" + "="*80)
    print("ANALISIS COMPLETADO")
    print("="*80)
    print(f"\nGráficos guardados en: metrics/plots/")
    print(f"  - confusion_matrix.png")
    print(f"  - accuracy_by_class.png")
    print(f"  - confidence_by_class.png")
    print("\n" + "="*80 + "\n")

else:
    print("[ERROR] No se procesaron audios")
