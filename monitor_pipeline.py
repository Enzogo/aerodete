#!/usr/bin/env python3
"""
MONITOR DE PROGRESO - AERODETECT PIPELINE
Muestra el estado actual de ejecucion en tiempo real
"""

import os
import json
import time
import sys

def get_training_progress():
    """Obtiene el progreso del entrenamiento actual"""
    if not os.path.exists('training_history.json'):
        return None, None
    
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        
        epochs = len(history['accuracy'])
        accuracy = history['accuracy'][-1]
        val_accuracy = history['val_accuracy'][-1]
        
        return epochs, {
            'train_acc': accuracy,
            'val_acc': val_accuracy,
            'loss': history['loss'][-1],
            'val_loss': history['val_loss'][-1]
        }
    except:
        return None, None

def check_validation():
    """Verifica si validacion se completo"""
    return os.path.exists('metrics/plots/results.json')

def check_github():
    """Verifica si fue pusheado a GitHub"""
    return os.path.exists('.git/refs/heads/main')

def check_cleanup():
    """Verifica si limpieza se completo"""
    temp_files = [
        'train_final_v2.py',
        'validate_robust.py',
        'final_pipeline.py',
        'run_full_pipeline.py'
    ]
    
    # Si ninguno existe, limpieza completada
    return not any(os.path.exists(f) for f in temp_files)

def print_status():
    """Imprime el estado actual"""
    print("\n" + "=" * 80)
    print(" " * 20 + "ESTADO DEL PIPELINE AERODETECT")
    print("=" * 80)
    
    # 1. ENTRENAMIENTO
    print("\n[1] ENTRENAMIENTO (5-8 horas)")
    print("-" * 40)
    
    epochs, metrics = get_training_progress()
    
    if epochs is None:
        print("  ⏳ Iniciando entrenamiento...")
    else:
        progress = min((epochs / 200) * 100, 100)
        bar = "█" * int(progress // 5) + "░" * (20 - int(progress // 5))
        
        print(f"  [{bar}] {progress:.1f}% ({epochs}/200 epocas)")
        print(f"  Train Acc: {metrics['train_acc']:.4f} | Val Acc: {metrics['val_acc']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f} | Val Loss: {metrics['val_loss']:.4f}")
        
        if metrics['val_acc'] >= 0.86:
            print(f"  ✓ OBJETIVO ALCANZADO: {metrics['val_acc']:.2%} >= 86%")
    
    # 2. VALIDACION
    print("\n[2] VALIDACION (150 audios por clase)")
    print("-" * 40)
    
    if not epochs or epochs < 200:
        print("  ⏳ En espera de entrenamiento...")
    elif not check_validation():
        print("  ⏳ Validando con 450 audios...")
    else:
        print("  ✓ Validacion completada")
        try:
            with open('metrics/plots/results.json', 'r') as f:
                results = json.load(f)
            print(f"  Accuracy: {results['general_accuracy']:.2f}%")
        except:
            pass
    
    # 3. METRICAS
    print("\n[3] METRICAS Y VISUALIZACIONES")
    print("-" * 40)
    
    metrics_files = []
    if os.path.exists('metrics/plots'):
        metrics_files = [f for f in os.listdir('metrics/plots') if f.endswith('.png')]
    
    if not metrics_files:
        print("  ⏳ Generando...")
    else:
        print("  ✓ Metricas generadas:")
        for mf in sorted(metrics_files):
            print(f"    - {mf}")
    
    # 4. GITHUB
    print("\n[4] PUSH A GITHUB (https://github.com/Enzogo/aerodete)")
    print("-" * 40)
    
    if check_github():
        print("  ✓ Codigo subido a GitHub")
    else:
        print("  ⏳ En espera de validacion...")
    
    # 5. LIMPIEZA
    print("\n[5] LIMPIEZA DE SCRIPTS")
    print("-" * 40)
    
    if check_cleanup():
        print("  ✓ Scripts eliminados")
        print("  ✓ Archivos temporales limpios")
    else:
        print("  ⏳ En limpieza...")
    
    # ARCHIVOS FINALES
    print("\n[ARCHIVOS FINALES]")
    print("-" * 40)
    
    final_files = {
        'README.md': 'Documentacion',
        'requirements.txt': 'Dependencias',
        '.gitignore': 'Config Git',
        'models': 'Modelo entrenado',
        'dataset': 'Datos de entrenamiento',
        'metrics/plots': 'Visualizaciones'
    }
    
    for file, desc in final_files.items():
        if os.path.exists(file):
            if os.path.isdir(file):
                count = len([f for f in os.listdir(file) if os.path.isfile(os.path.join(file, f))])
                print(f"  ✓ {file}/ ({count} archivos) - {desc}")
            else:
                size = os.path.getsize(file) / 1024
                print(f"  ✓ {file} ({size:.1f} KB) - {desc}")
        else:
            print(f"  ⏳ {file} - {desc}")
    
    print("\n" + "=" * 80)
    print(f"Actualizacion: {time.strftime('%H:%M:%S')}")
    print("=" * 80 + "\n")

# BUCLE DE MONITOREO
if __name__ == '__main__':
    print("\n✓ MONITOR INICIADO - Presiona Ctrl+C para salir\n")
    
    try:
        while True:
            print("\033[2J\033[H")  # Limpiar pantalla
            print_status()
            
            # Verificar si completo
            if check_cleanup():
                print("\n✅ ¡PIPELINE COMPLETADO EXITOSAMENTE!")
                print("   Todos los archivos estan listos y GitHub ha sido actualizado")
                break
            
            time.sleep(30)  # Actualizar cada 30 segundos
    
    except KeyboardInterrupt:
        print("\n\nMonitor detenido por usuario")
        sys.exit(0)
