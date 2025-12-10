import os
import json
import numpy as np
import librosa
import io
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from pathlib import Path

warnings.filterwarnings('ignore')

# Inicializar Flask
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo y normalizador
print("Cargando modelo...")

model_v3 = Path('models/audio_model_robusto_v3.h5')
model_v2 = Path('models/audio_model_final.h5')

try:
    # Intentar cargar v3, si no fallback a v2
    if model_v3.exists():
        model = load_model(str(model_v3))
        model_version = "v3"
        print(f"[OK] Modelo V3 cargado")
    else:
        model = load_model(str(model_v2))
        model_version = "v2"
        print(f"[OK] Modelo V2 cargado (fallback)")
except Exception as e:
    print(f"[ERROR] {e}")
    exit(1)

CLASSES = ['avion', 'dron', 'helicoptero']
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_audio_file(filepath):
    """Predice la clase de un archivo"""
    try:
        # Cargar audio
        y, sr = librosa.load(filepath, sr=22050)
        
        # Extraer MFCC (40 coeficientes, 170 frames - como en v3)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        
        # Padding a 170 frames
        if mfcc.shape[0] < 170:
            mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:170]
        
        # Normalizar (aplanar, normalizar, reshape)
        mfcc_flat = mfcc.reshape(-1, 40)
        
        scaler = RobustScaler()
        mfcc_norm_flat = scaler.fit_transform(mfcc_flat)
        
        mfcc_norm = mfcc_norm_flat.reshape(170, 40)
        
        # Predicción
        pred = model.predict(np.expand_dims(mfcc_norm, 0), verbose=0)
        
        pred_class_idx = np.argmax(pred[0])
        confidence = pred[0][pred_class_idx]
        
        return {
            'clase': CLASSES[pred_class_idx],
            'confianza': float(confidence),
            'confianzas_por_clase': {
                CLASSES[i]: float(pred[0][i]) for i in range(3)
            },
            'duracion': float(len(y) / sr),
            'sample_rate': int(sr),
            'model_version': model_version,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_audio_visualization(filepath):
    """Genera visualizaciones del audio"""
    try:
        y, sr = librosa.load(filepath, sr=22050)
        
        # Espectrograma
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Espectrograma lineal
        D = librosa.stft(y)
        D_db = librosa.power_to_db(np.abs(D)**2, ref=np.max)
        
        # Centroide espectral y otras características
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        return {
            'mfcc': mfcc.tolist(),
            'spectrogram_mel': S_db.tolist(),
            'spectrogram_linear': D_db.tolist(),
            'centroid': centroid.tolist(),
            'rolloff': rolloff.tolist(),
            'sr': sr,
            'duration': len(y) / sr,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def generate_plots(filepath):
    """Genera gráficas del audio en formato base64"""
    try:
        y, sr = librosa.load(filepath, sr=22050)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Waveform
        axes[0, 0].plot(y[:sr*5])  # 5 segundos
        axes[0, 0].set_title('Waveform (5s)')
        axes[0, 0].set_xlabel('Samples')
        
        # Espectrograma Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[0, 1])
        axes[0, 1].set_title('Mel Spectrogram')
        fig.colorbar(img, ax=axes[0, 1], format='%+2.0f dB')
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        img = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1, 0])
        axes[1, 0].set_title('MFCC')
        fig.colorbar(img, ax=axes[1, 0])
        
        # Centroide y Rolloff
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        frames = range(len(centroid))
        t = librosa.frames_to_time(frames, sr=sr)
        
        D = librosa.stft(y)
        axes[1, 1].semilogy(t, centroid, label='Centroid', linewidth=2)
        axes[1, 1].semilogy(t, rolloff, label='Rolloff', linewidth=2)
        axes[1, 1].set_ylabel('Hz')
        axes[1, 1].set_title('Centroid & Rolloff')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Convertir a base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return {
            'plot': f"data:image/png;base64,{img_base64}",
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

#Endpoints de la API

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'AeroDetect API v2',
        'version': '2.0',
        'status': 'online',
        'endpoints': {
            'GET /': 'Esta información',
            'GET /health': 'Estado del servidor',
            'POST /predict': 'Predecir archivo de audio',
            'POST /features': 'Extraer características',
            'POST /visualize': 'Generar visualizaciones'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'modelo': 'AeroDetect v1.0',
        'clases': CLASSES,
        'accuracy': '99.78%',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predecir desde archivo de audio"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_audio_file(filepath)
        os.remove(filepath)
        
        if not result['success']:
            return jsonify(result), 400
        
        return jsonify({
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/features', methods=['POST'])
def features():
    """Extraer características del audio"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = get_audio_visualization(filepath)
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    """Generar visualizaciones"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = generate_plots(filepath)
        os.remove(filepath)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predecir múltiples archivos"""
    try:
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        results = []
        for file in files[:10]:  # Máx 10 archivos
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = predict_audio_file(filepath)
                results.append({
                    'filename': filename,
                    'result': result
                })
                os.remove(filepath)
        
        return jsonify({
            'total': len(results),
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("AERODETECT API v2 - Servidor Flask")
    print("="*80)
    print("\nIniciando servidor en http://localhost:5000")
    print("\nDocumentación disponible en: http://localhost:5000/")
    print("\nEndpoints disponibles:")
    print("   - GET  /              - Información")
    print("   - GET  /health        - Estado")
    print("   - POST /predict       - Predecir archivo")
    print("   - POST /features      - Extraer características")
    print("   - POST /visualize     - Visualizaciones")
    print("   - POST /batch-predict - Múltiples archivos")
    print("\n" + "="*80 + "\n")
    
    app.run(host='localhost', port=5000, debug=False, use_reloader=False)
