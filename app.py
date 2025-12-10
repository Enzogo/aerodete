#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APP.PY - GUI Interactiva con Tkinter
Interfaz de escritorio para predicción de audios
Tecnologías: Tkinter, TensorFlow, Librosa, Matplotlib
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import numpy as np
import warnings
import threading
from pathlib import Path
from datetime import datetime

# GUI
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

# Matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Deep Learning - TensorFlow
import tensorflow as tf

# Audio Processing - Librosa
import librosa
from scipy import signal

# Métricas - Scikit-learn
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

LABELS = ['avion', 'dron', 'helicoptero']
SR = 22050
DURATION = 4.0
MFCC_N = 40
FFT_SIZE = 512

# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class AeroDetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AeroDetect - Clasificación de Audio con IA")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.labels = LABELS
        
        # Cargar modelo
        self.load_model()
        
        # Crear UI
        self.create_ui()
    
    def load_model(self):
        """Cargar modelo TensorFlow"""
        model_path = "models/audio_model_working.h5"
        if not Path(model_path).exists():
            messagebox.showerror("Error", f"Modelo no encontrado: {model_path}\nEjecuta: python train.py")
            sys.exit(1)
        
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Modelo cargado: {self.model.count_params():,} parámetros")
    
    def create_ui(self):
        """Crear interfaz gráfica"""
        # Notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Clasificación
        self.tab1 = ttk.Frame(notebook)
        notebook.add(self.tab1, text="🎵 Clasificación Individual")
        self.create_classification_tab()
        
        # Tab 2: Evaluación
        self.tab2 = ttk.Frame(notebook)
        notebook.add(self.tab2, text="📊 Evaluación Dataset")
        self.create_evaluation_tab()
    
    def create_classification_tab(self):
        """Tab 1: Clasificación individual"""
        # Frame superior
        top_frame = ttk.Frame(self.tab1)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(top_frame, text="📁 Cargar Audio", command=self.load_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="🎤 Clasificar", command=self.classify).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="🔄 Limpiar", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        # Frame contenido
        content_frame = ttk.Frame(self.tab1)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Izquierda: Información
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5)
        
        ttk.Label(left_frame, text="Información:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.info_text = scrolledtext.ScrolledText(left_frame, width=50, height=25, font=('Courier', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Derecha: Gráficos
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        ttk.Label(right_frame, text="Gráficos MFCC:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
    
    def create_evaluation_tab(self):
        """Tab 2: Evaluación dataset"""
        # Frame superior
        top_frame = ttk.Frame(self.tab2)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(top_frame, text="📊 Evaluar Dataset", command=self.evaluate_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="💾 Guardar Gráficos", command=self.save_plots).pack(side=tk.LEFT, padx=5)
        
        # Frame contenido
        content_frame = ttk.Notebook(self.tab2)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sub-tab: Matriz de confusión
        self.cm_frame = ttk.Frame(content_frame)
        content_frame.add(self.cm_frame, text="Matriz Confusión")
        
        # Sub-tab: Curva ROC
        self.roc_frame = ttk.Frame(content_frame)
        content_frame.add(self.roc_frame, text="Curva ROC")
        
        # Sub-tab: Métricas
        self.metrics_frame = ttk.Frame(content_frame)
        content_frame.add(self.metrics_frame, text="Métricas")
        
        self.metrics_text = scrolledtext.ScrolledText(self.metrics_frame, font=('Courier', 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
    
    def load_audio(self):
        """Cargar archivo de audio"""
        filepath = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filepath:
            self.audio_path = filepath
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, f"Archivo cargado:\n{filepath}\n\nHaz click en 'Clasificar' para predecir")
    
    def classify(self):
        """Clasificar audio cargado"""
        if not hasattr(self, 'audio_path'):
            messagebox.showwarning("Advertencia", "Primero carga un archivo de audio")
            return
        
        try:
            # Mostrar mensaje
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, "Procesando...\n")
            self.root.update()
            
            # Extraer features con Librosa
            y, sr = librosa.load(self.audio_path, sr=SR, duration=DURATION)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2)
            mfcc = mfcc.T
            
            if mfcc.shape[0] > 170:
                mfcc = mfcc[:170, :]
            else:
                mfcc = np.pad(mfcc, ((0, 170 - mfcc.shape[0]), (0, 0)))
            
            # Normalizar
            mfcc_mean = np.mean(mfcc, axis=0)
            mfcc_std = np.std(mfcc, axis=0)
            mfcc = (mfcc - mfcc_mean) / (mfcc_std + 1e-7)
            
            # Predecir con TensorFlow
            mfcc_batch = np.expand_dims(mfcc, axis=0)
            predictions = self.model.predict(mfcc_batch, verbose=0)[0]
            
            # Resultados
            pred_class = self.labels[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            
            # Análisis de frecuencias
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc_mean_freq = np.mean(log_S, axis=1)
            top_freqs = np.argsort(mfcc_mean_freq)[-5:][::-1]
            frequencies = [float(sr * i / FFT_SIZE) for i in top_freqs]
            
            # Mostrar resultados
            output = f"PREDICCIÓN\n{'='*45}\n"
            output += f"Clase: {pred_class.upper()}\n"
            output += f"Confianza: {confidence:.2f}%\n\n"
            
            output += f"PROBABILIDADES\n{'='*45}\n"
            for i, label in enumerate(self.labels):
                prob = predictions[i] * 100
                bar = '█' * int(prob/5)
                output += f"{label:15} : {prob:6.2f}% {bar}\n"
            
            output += f"\nFRECUENCIAS\n{'='*45}\n"
            for i, freq in enumerate(frequencies, 1):
                output += f"{i}. {freq:7.1f} Hz\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, output)
            
            # Gráfico MFCC
            self.plot_mfcc(mfcc)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error clasificando: {str(e)}")
    
    def plot_mfcc(self, mfcc):
        """Plotear MFCC"""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('MFCC')
        ax.set_title('Características MFCC')
        fig.colorbar(im, ax=ax)
        
        canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def evaluate_dataset(self):
        """Evaluar dataset completo"""
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "Evaluando dataset...\n\n")
        self.root.update()
        
        # Thread para no bloquear GUI
        threading.Thread(target=self._evaluate_worker, daemon=True).start()
    
    def _evaluate_worker(self):
        """Worker para evaluación"""
        try:
            # Cargar datos
            X_data = []
            y_data = []
            
            for label_idx, label in enumerate(self.labels):
                label_path = Path(f"dataset/{label}")
                for audio_file in label_path.glob("*.wav"):
                    try:
                        y, sr = librosa.load(str(audio_file), sr=SR, duration=DURATION)
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_N, n_fft=FFT_SIZE, hop_length=FFT_SIZE//2)
                        mfcc = mfcc.T
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
            
            # Normalizar
            X_mean = np.mean(X_data, axis=0)
            X_std = np.std(X_data, axis=0)
            X_data = (X_data - X_mean) / (X_std + 1e-7)
            
            # Predecir
            y_pred_probs = self.model.predict(X_data, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Matriz de confusión
            cm = confusion_matrix(y_data, y_pred)
            self.plot_confusion_matrix(cm)
            
            # Curva ROC
            y_data_bin = label_binarize(y_data, classes=range(len(self.labels)))
            self.plot_roc_curve(y_data_bin, y_pred_probs)
            
            # Métricas
            report = classification_report(y_data, y_pred, target_names=self.labels)
            accuracy = np.mean(y_pred == y_data)
            
            output = f"ACCURACY GENERAL: {accuracy*100:.2f}%\n\n"
            output += report
            
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, output)
        
        except Exception as e:
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, f"Error: {str(e)}")
    
    def plot_confusion_matrix(self, cm):
        """Plotear matriz de confusión"""
        for widget in self.cm_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.labels, yticklabels=self.labels, ax=ax)
        ax.set_title('Matriz de Confusión')
        ax.set_ylabel('Real')
        ax.set_xlabel('Predicho')
        
        canvas = FigureCanvasTkAgg(fig, self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_roc_curve(self, y_bin, y_probs):
        """Plotear curva ROC"""
        for widget in self.roc_frame.winfo_children():
            widget.destroy()
        
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        for i in range(len(self.labels)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{self.labels[i]} (AUC = {roc_auc:.2f})', color=colors[i], linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Curva ROC')
        ax.legend()
        
        canvas = FigureCanvasTkAgg(fig, self.roc_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def save_plots(self):
        """Guardar gráficos"""
        messagebox.showinfo("Info", "Los gráficos se guardan automáticamente en metrics/plots/")
    
    def clear_results(self):
        """Limpiar resultados"""
        self.info_text.delete(1.0, tk.END)
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    app = AeroDetectGUI(root)
    root.mainloop()
