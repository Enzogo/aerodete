#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API.PY - API REST con Flask
Proporciona endpoints para predicción usando TensorFlow + Librosa
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np

# Flask - API REST
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

# Deep Learning - TensorFlow
import tensorflow as tf

# Audio Processing - Librosa
import librosa

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN FLASK
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = Path('temp_uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

MODEL = None
LABELS = ['avion', 'dron', 'helicoptero']
SR = 22050
DURATION = 4.0
MFCC_N = 40
FFT_SIZE = 512

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_model():
    """Cargar modelo TensorFlow"""
    global MODEL
    model_path = "models/audio_model_working.h5"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    MODEL = tf.keras.models.load_model(model_path)
    print(f"✓ Modelo cargado: {MODEL.count_params():,} parámetros")

def extract_features(audio_path):
    """Extraer features MFCC con Librosa"""
    y, sr = librosa.load(str(audio_path), sr=SR, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2)
    mfcc = mfcc.T
    
    if mfcc.shape[0] > 170:
        mfcc = mfcc[:170, :]
    else:
        mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
    
    return mfcc

def predict_audio(audio_path):
    """Predecir clase con TensorFlow"""
    if MODEL is None:
        raise RuntimeError("Modelo no cargado")
    
    mfcc = extract_features(audio_path)
    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)
    mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-7)
    
    mfcc_batch = np.expand_dims(mfcc, axis=0)
    predictions = MODEL.predict(mfcc_batch, verbose=0)[0]
    
    pred_class = LABELS[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100
    
    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': {label: float(prob*100) for label, prob in zip(LABELS, predictions)}
    }

# ============================================================================
# RUTAS - ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Documentación de API"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AeroDetect API</title>
        <style>
            body { font-family: Arial; margin: 40px; background: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }
            h1 { color: #333; }
            .endpoint { background: #f9f9f9; padding: 15px; margin: 15px 0; border-left: 4px solid #007bff; }
            code { background: #eee; padding: 2px 6px; border-radius: 3px; }
            pre { background: #f4f4f4; padding: 10px; overflow-x: auto; }
            .method { display: inline-block; padding: 4px 8px; border-radius: 3px; font-weight: bold; color: white; }
            .post { background: #28a745; }
            .get { background: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎵 AeroDetect API - REST</h1>
            <p>Clasificación de sonidos de aeronaves con Deep Learning</p>
            
            <h2>Stack Tecnológico:</h2>
            <ul>
                <li><strong>Flask</strong> - Servidor REST</li>
                <li><strong>TensorFlow</strong> - Deep Learning CNN 1D</li>
                <li><strong>Librosa</strong> - Procesamiento de audio MFCC</li>
            </ul>
            
            <h2>Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span> <code>/predict</code>
                <p>Predecir clase de audio</p>
                <pre>curl -X POST -F "file=@audio.wav" http://localhost:5000/predict</pre>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/status</code>
                <p>Estado del sistema</p>
                <pre>curl http://localhost:5000/status</pre>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <code>/model-info</code>
                <p>Información del modelo</p>
                <pre>curl http://localhost:5000/model-info</pre>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predicción"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(filepath))
        
        result = predict_audio(str(filepath))
        result['timestamp'] = datetime.now().isoformat()
        result['file'] = filename
        
        filepath.unlink()
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Estado del sistema"""
    return jsonify({
        'status': 'online',
        'model_loaded': MODEL is not None,
        'labels': LABELS,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    """Información del modelo"""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'CNN 1D (TensorFlow)',
        'parameters': MODEL.count_params(),
        'input_shape': list(MODEL.input_shape),
        'output_shape': list(MODEL.output_shape),
        'classes': LABELS
    }), 200

@app.before_request
def setup():
    """Setup antes del primer request"""
    global MODEL
    if MODEL is None:
        try:
            load_model()
        except Exception as e:
            print(f"Error cargando modelo: {e}")

if __name__ == '__main__':
    print("\n" + "="*80)
    print("AeroDetect API - Flask + TensorFlow + Librosa")
    print("="*80)
    
    try:
        load_model()
        print("\n✓ API iniciada en http://localhost:5000")
        print("✓ Documentación en http://localhost:5000")
        print("\nEndpoints:")
        print("  POST  /predict      → Predecir audio")
        print("  GET   /status       → Estado del sistema")
        print("  GET   /model-info   → Información del modelo")
        print("\n" + "="*80 + "\n")
        
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
