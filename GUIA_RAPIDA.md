# GUÍA RÁPIDA - AeroDetect

## 🚀 Inicio en 3 pasos

### Paso 1: Instalar
```bash
pip install -r requirements.txt
```

### Paso 2: Evaluar (Generar Gráficos)
```bash
python evaluate.py
```
Esto genera:
- ✓ Matriz de confusión
- ✓ Curva ROC
- ✓ Métricas completas

### Paso 3: Usar GUI
```bash
python app.py
```

¡La interfaz se abre automáticamente! 🎉

---

## 📊 Opciones Disponibles

### Entrenar Modelo (OPCIONAL)
```bash
python train.py
```
Si quieres re-entrenar con tu propio dataset.

### API REST
```bash
python api.py
```
Servidor en `http://localhost:5000`

---

## 🎵 Usar la GUI

### Tab 1: Clasificación
1. Click **"Cargar Audio"**
2. Selecciona `.wav`
3. Click **"Clasificar"**
4. Ver predicción ✓

### Tab 2: Evaluación
1. Click **"Evaluar Dataset"**
2. Espera análisis
3. Ver resultados ✓

---

## 📁 Archivos Generados

```
metrics/
├── plots/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── roc_curve.png
│   ├── accuracy_per_class.png
│   └── confidence_distribution.png
├── reports/
│   └── classification_report.txt
└── results.json
```

---

## ❓ FAQ

**P: ¿Ya está entrenado?**
R: Sí, listo para usar.

**P: ¿Debo entrenar?**
R: No, a menos que quieras mejorar.

**P: ¿Qué formatos de audio?**
R: `.wav` principalmente (Librosa soporta .mp3, .ogg, .flac)

**P: ¿Funciona sin internet?**
R: Sí, todo local.

---

## 📊 Stack Tecnológico

- **TensorFlow** - Deep Learning
- **Librosa** - Audio Processing
- **Flask** - API REST
- **Tkinter** - GUI

Ver `STACK_TECNOLOGICO.md` para detalles.

---

## 🆘 Problemas

| Problema | Solución |
|----------|----------|
| Módulo no encontrado | `pip install -r requirements.txt` |
| GUI no abre | Cambiar a carpeta correcta |
| Modelo no existe | `python train.py` |

---

¡Listo! Disfruta AeroDetect 🎉
