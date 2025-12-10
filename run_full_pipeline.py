#!/usr/bin/env python3
"""
AERODETECT - INICIO AUTOMATICO COMPLETO
Ejecuta: Entrenamiento → Validacion → GitHub → Limpieza
Solo quedan: README.md, requirements.txt, .gitignore, models/, dataset/, metrics/plots/

INSTRUCCIONES:
    python run_full_pipeline.py

Este script ejecuta final_pipeline.py que hace todo automaticamente.
"""

import subprocess
import sys

print("\n" + "=" * 80)
print(" " * 15 + "AERODETECT - INICIANDO PIPELINE COMPLETO")
print("=" * 80)
print(f"""
FASES:
1. ENTRENAMIENTO (5-8 horas) - 200 epocas, 2,550 audios
2. VALIDACION (10 min) - 450 audios (150 por clase)
3. METRICAS - Confusion matrix, learning curves, confidence analysis
4. GITHUB PUSH - Codigo subido a https://github.com/Enzogo/aerodete
5. LIMPIEZA - Eliminacion de todos los scripts auxiliares

Solo quedan al final:
  ✓ README.md
  ✓ requirements.txt
  ✓ .gitignore
  ✓ models/ (modelo entrenado)
  ✓ dataset/ (datos)
  ✓ metrics/plots/ (visualizaciones)

INICIANDO EN 3 SEGUNDOS...
""")

import time
for i in range(3, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

print("\n" + "=" * 80 + "\n")

try:
    subprocess.run([sys.executable, 'final_pipeline.py'], check=True)
except KeyboardInterrupt:
    print("\n\n❌ INTERRUMPIDO POR USUARIO")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
