#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AeroDetect GUI MEJORADA - Con soporte para archivos locales y URLs de YouTube
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
import threading
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import pickle


# CARGAR MODELO Y NORMALIZADOR

print("Cargando modelo...")

# Intentar cargar modelo v3 (keras primero, luego h5)
model_v3_keras = Path('models/audio_model_robusto_v3.keras')
model_v3 = Path('models/audio_model_robusto_v3.h5')
model_v2 = Path('models/audio_model_final.h5')

try:
    if model_v3_keras.exists():
        model = load_model(str(model_v3_keras), compile=False)
        model_version = "v3"
        print(f"[OK] Modelo V3 (keras) cargado")
    elif model_v3.exists():
        model = load_model(str(model_v3), compile=False)
        model_version = "v3"
        print(f"[OK] Modelo V3 (h5) cargado")
    else:
        model = load_model(str(model_v2), compile=False)
        model_version = "v2"
        print(f"[OK] Modelo V2 cargado (fallback)")
except Exception as e:
    print(f"[ERROR] Cargando modelo: {e}")
    exit(1)

# CARGAR NORMALIZADOR
print("Cargando normalizador...")
scaler_path = Path('models/normalization_robusto_v3.pkl')
if scaler_path.exists():
    with open(scaler_path, 'rb') as f:
        global_scaler = pickle.load(f)
    print(f"[OK] Normalizador cargado: {scaler_path.name}")
else:
    print(f"[ERROR] No se encontró normalizador en {scaler_path}")
    exit(1)

CLASSES = ['avion', 'dron', 'helicoptero']
CLASS_COLORS = {
    'avion': '#FF6B6B',
    'dron': '#4ECDC4',
    'helicoptero': '#45B7D1'
}


# FUNCIONES DE PREDICCIÓN

def predict_audio(filepath):
    """Predice la clase del audio usando normalizador guardado"""
    try:
        # Cargar y procesar audio
        y, sr = librosa.load(filepath, sr=22050, duration=4.0)
        
        # Extraer MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        
        # Padding a 170 frames
        if mfcc.shape[0] < 170:
            mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:170]
        
        # NORMALIZAR CON SCALER GUARDADO (CRÍTICO)
        mfcc_flat = mfcc.flatten()  # Aplanar a (6800,)
        mfcc_norm_flat = global_scaler.transform(mfcc_flat.reshape(1, -1))  # transform, NO fit_transform
        mfcc_norm = mfcc_norm_flat.reshape(170, 40)
        
        # Predecir
        pred = model.predict(np.expand_dims(mfcc_norm, 0), verbose=0)
        pred_class_idx = np.argmax(pred[0])
        confidence = pred[0][pred_class_idx]
        
        return {
            'clase': CLASSES[pred_class_idx],
            'confianza': float(confidence),
            'confianzas': {CLASSES[i]: float(pred[0][i]) for i in range(3)},
            'y': y,
            'sr': sr,
            'mfcc': mfcc,
            'duracion': len(y) / sr
        }
    except Exception as e:
        return {'error': str(e)}

def download_from_youtube(url):
    """Descarga audio de YouTube usando yt-dlp"""
    try:
        import yt_dlp
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp', '-q'], check=True)
        import yt_dlp
    
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_path = temp_file.name
        temp_file.close()
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': temp_path[:-4],
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
        
        return temp_path
    except Exception as e:
        raise Exception(f"Error descargando: {str(e)}")


# GUI

class AeroDetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AeroDetect - Clasificador de Audios Aéreos")
        self.root.geometry("1000x750")
        self.root.configure(bg='#f0f0f0')
        
        self.current_result = None
        self.current_filepath = None
        
        self.create_ui()
    
    def create_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#2c3e50')
        header.pack(fill=tk.X, padx=0, pady=0)
        
        title = tk.Label(header, text="AeroDetect - Clasificador de Audios", 
                        font=("Arial", 18, "bold"), bg='#2c3e50', fg='white')
        title.pack(pady=10)
        
        subtitle = tk.Label(header, text=f"Modelo: {model_version} v3 | Soporte: Archivo + YouTube", 
                           font=("Arial", 10), bg='#2c3e50', fg='#ecf0f1')
        subtitle.pack(pady=5)
        
        # Frame de entrada
        input_frame = tk.Frame(self.root, bg='white', relief=tk.RAISED, bd=1)
        input_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Botón archivo
        tk.Label(input_frame, text="Archivo Local:", font=("Arial", 10, "bold"), bg='white').pack(anchor=tk.W, pady=(10, 5), padx=10)
        
        file_btn_frame = tk.Frame(input_frame, bg='white')
        file_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_file = tk.Button(file_btn_frame, text="📁 Seleccionar Audio", 
                                  command=self.select_file, bg='#3498db', fg='white',
                                  font=("Arial", 10, "bold"), padx=15, pady=8)
        self.btn_file.pack(side=tk.LEFT, padx=5)
        
        self.file_label = tk.Label(file_btn_frame, text="Ninguno seleccionado", 
                                   font=("Arial", 9), fg='#7f8c8d', bg='white')
        self.file_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        # URL YouTube
        tk.Label(input_frame, text="URL de YouTube:", font=("Arial", 10, "bold"), bg='white').pack(anchor=tk.W, pady=(15, 5), padx=10)
        
        url_frame = tk.Frame(input_frame, bg='white')
        url_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.url_entry = tk.Entry(url_frame, font=("Arial", 10), width=50)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.btn_youtube = tk.Button(url_frame, text="🎥 Descargar", 
                                     command=self.download_youtube, bg='#e74c3c', fg='white',
                                     font=("Arial", 10, "bold"), padx=15, pady=8)
        self.btn_youtube.pack(side=tk.LEFT, padx=5)
        
        # Botones de acción
        action_frame = tk.Frame(input_frame, bg='white')
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(action_frame, text="🔍 Analizar", command=self.predict, 
                 bg='#27ae60', fg='white', font=("Arial", 10, "bold"), padx=15, pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(action_frame, text="🗑️ Limpiar", command=self.clear, 
                 bg='#95a5a6', fg='white', font=("Arial", 10, "bold"), padx=15, pady=8).pack(side=tk.LEFT, padx=5)
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Panel izquierdo - Resultado
        left_panel = tk.Frame(main_frame, bg='white', relief=tk.RIDGE, bd=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        result_header = tk.Label(left_panel, text="RESULTADO", bg='#2c3e50', fg='white',
                                font=("Arial", 12, "bold"), pady=10)
        result_header.pack(fill=tk.X)
        
        result_content = tk.Frame(left_panel, bg='white')
        result_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        self.result_label = tk.Label(result_content, text="Selecciona un audio", 
                                     font=("Arial", 28, "bold"), fg='#2c3e50')
        self.result_label.pack(pady=20)
        
        self.confidence_label = tk.Label(result_content, text="", 
                                         font=("Arial", 14), fg='#27ae60')
        self.confidence_label.pack(pady=10)
        
        # Probabilidades
        tk.Label(result_content, text="Probabilidades:", font=("Arial", 11, "bold"),
                bg='white').pack(anchor=tk.W, pady=(20, 10))
        
        self.confidence_bars = {}
        for cls in CLASSES:
            frame = tk.Frame(result_content, bg='white')
            frame.pack(fill=tk.X, pady=8)
            
            lbl = tk.Label(frame, text=cls.upper(), font=("Arial", 10),
                          bg='white', width=12, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            
            canvas = tk.Canvas(frame, height=20, bg='#ecf0f1', highlightthickness=0)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
            val_lbl = tk.Label(frame, text="0%", font=("Arial", 9),
                              bg='white', width=6)
            val_lbl.pack(side=tk.LEFT)
            
            self.confidence_bars[cls] = {'canvas': canvas, 'label': val_lbl}
        
        # Panel derecho - Gráficos
        right_panel = tk.Frame(main_frame, bg='white', relief=tk.RIDGE, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas_frame = tk.Frame(right_panel, bg='white')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status
        self.status_label = tk.Label(self.root, text="Listo", font=("Arial", 9), 
                                     fg='#7f8c8d', bg='#ecf0f1')
        self.status_label.pack(fill=tk.X, padx=15, pady=5)
    
    def select_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg"), ("All", "*.*")]
        )
        
        if filepath:
            self.current_filepath = filepath
            self.file_label.config(text=os.path.basename(filepath))
            self.status_label.config(text="Archivo cargado. Presiona 'Analizar'")
    
    def download_youtube(self):
        url = self.url_entry.get().strip()
        
        if not url:
            messagebox.showwarning("Advertencia", "Ingresa una URL de YouTube")
            return
        
        self.status_label.config(text="Descargando...")
        self.btn_youtube.config(state=tk.DISABLED)
        
        def process():
            try:
                filepath = download_from_youtube(url)
                self.current_filepath = filepath
                self.root.after(0, lambda: self.file_label.config(
                    text=f"YouTube - {os.path.basename(filepath)}"
                ))
                self.root.after(0, lambda: self.status_label.config(
                    text="Audio descargado. Presiona 'Analizar'"
                ))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.root.after(0, lambda: self.status_label.config(text="Error descargando"))
            finally:
                self.root.after(0, lambda: self.btn_youtube.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def predict(self):
        if not self.current_filepath:
            messagebox.showwarning("Advertencia", "Selecciona un archivo primero")
            return
        
        self.status_label.config(text="Analizando...")
        
        def process():
            result = predict_audio(self.current_filepath)
            self.root.after(0, lambda: self.update_ui(result))
        
        thread = threading.Thread(target=process)
        thread.daemon = True
        thread.start()
    
    def update_ui(self, result):
        if 'error' in result:
            messagebox.showerror("Error", result['error'])
            return
        
        self.current_result = result
        
        # Resultado
        clase = result['clase'].upper()
        color = CLASS_COLORS[result['clase']]
        
        self.result_label.config(text=clase, fg=color)
        self.confidence_label.config(
            text=f"Confianza: {result['confianza']*100:.1f}%"
        )
        
        # Barras
        for cls in CLASSES:
            conf = result['confianzas'][cls]
            bar = self.confidence_bars[cls]['canvas']
            val_lbl = self.confidence_bars[cls]['label']
            
            bar.delete("all")
            bar_width = 200
            fill_width = bar_width * conf
            bar_color = CLASS_COLORS[cls]
            
            bar.create_rectangle(0, 0, fill_width, 20, fill=bar_color, outline=bar_color)
            bar.create_rectangle(0, 0, bar_width, 20, outline='#bdc3c7')
            
            val_lbl.config(text=f"{conf*100:.0f}%")
        
        # Gráfico
        self.show_spectrogram()
        
        self.status_label.config(text=f"[OK] {result['clase']} - {result['confianza']*100:.1f}%")
    
    def show_spectrogram(self):
        if not self.current_result:
            return
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        result = self.current_result
        
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        D = librosa.power_to_db(np.abs(librosa.stft(result['y']))**2, ref=np.max)
        img = librosa.display.specshow(D, sr=result['sr'], x_axis='time', y_axis='log', ax=ax, cmap='viridis')
        ax.set_title('Espectrograma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def clear(self):
        self.result_label.config(text="Selecciona un audio")
        self.confidence_label.config(text="")
        self.url_entry.delete(0, tk.END)
        self.file_label.config(text="Ninguno seleccionado")
        
        for cls in CLASSES:
            bar = self.confidence_bars[cls]['canvas']
            val_lbl = self.confidence_bars[cls]['label']
            bar.delete("all")
            val_lbl.config(text="0%")
        
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        self.current_result = None
        self.current_filepath = None
        self.status_label.config(text="Listo")


if __name__ == "__main__":
    root = tk.Tk()
    app = AeroDetectGUI(root)
    root.mainloop()
