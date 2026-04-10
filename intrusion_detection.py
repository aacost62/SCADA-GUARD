# Industrial IoT Intrusion Detection System - Enterprise UI Upgrade (Thread-Safe)
# Authors: Juan Soto Valdez, Arturo Acosta Andrade, Axcel Hurtado
# Purpose: SCADA/ICS Anomaly Detection via Machine Learning

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import random
import joblib
import os
import csv
import logging
import queue
from collections import deque
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PIPELINE_FILE = "iot_intrusion_pipeline.pkl"
ALERT_LOG_CSV = "iot_attack_logs.csv"

# ==========================================
# 1. DATA PIPELINE & AI ENGINE
# ==========================================
class DataPipeline:
    @staticmethod
    def load_and_prep_data(filepath):
        df = pd.read_csv(filepath)
        df.columns = df.columns.astype(str).str.strip()
        target_col = "Normal/Attack" if "Normal/Attack" in df.columns else df.columns[-1]
        if df[target_col].dtype == object:
            normalized = df[target_col].astype(str).str.strip().str.lower()
            df[target_col] = normalized.map({"normal": 0, "attack": 1}).fillna(0).astype(int)
        return df, target_col

    @staticmethod
    def log_alert_to_csv(alert_data):
        file_exists = os.path.isfile(ALERT_LOG_CSV)
        try:
            with open(ALERT_LOG_CSV, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=alert_data.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(alert_data)
        except Exception as e:
            logging.error(f"Failed to log alert: {e}")

class IoTAnomalyDetector:
    def __init__(self, window_size=10):
        self.attack_counts = {"Sensor Spoofing": 0, "Command Injection": 0, "Network DDoS": 0, "Replay Attack": 0}
        self.recent_attacks = deque(maxlen=30)
        self.lock = threading.Lock()
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def classify_attack(self, features, anomaly_prob):
        detected, desc, severity = None, "", "SUSPICIOUS"
        num_df = features.select_dtypes(include=[np.number])
        num_features = num_df.values[0] if (not num_df.empty and len(num_df.values[0]) > 0) else np.array([0])
        
        self.history.append({'prob': anomaly_prob, 'features': num_features})

        if np.max(num_features) > 99999: 
            detected, desc, severity = "Sensor Spoofing", "Physical limits exceeded.", "CRITICAL"
        elif len(self.history) == self.window_size:
            recent_features = np.array([x['features'] for x in self.history])
            rolling_std = np.std(recent_features, axis=0)
            avg_prob = np.mean([x['prob'] for x in self.history])
            max_vol = np.max(rolling_std)
            min_vol = np.min(rolling_std)

            if anomaly_prob > 0.90: severity = "CRITICAL"
            elif anomaly_prob > 0.75: severity = "HIGH"

            if anomaly_prob > 0.85 and max_vol > 100: detected, desc = "Sensor Spoofing", "Erratic sensor volatility."
            elif anomaly_prob > 0.80 and random.random() > 0.7: detected, desc = "Command Injection", "Anomalous state change."
            elif anomaly_prob > 0.75 and min_vol < 0.01: detected, desc = "Replay Attack", "Stagnant data packets (playback)."
            elif avg_prob > 0.80: detected, desc = "Network DDoS", "Sustained high-frequency anomalies."

        if detected:
            with self.lock:
                self.attack_counts[detected] += 1
                log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] [{severity}] {detected.upper()} - {desc}"
                self.recent_attacks.appendleft((log_entry, severity))
            return True, detected
        return False, "Clear"

class IntrusionDetectionEngine:
    def __init__(self, alert_threshold=0.70): 
        self.pipeline = None
        self.alert_threshold = alert_threshold

    def is_anomalous(self, prob): return prob >= self.alert_threshold

# ==========================================
# 2. UI: ENTERPRISE SCADA DASHBOARD
# ==========================================
class IndustrialDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("SCADA-GUARD | Advanced Threat Detection")
        self.root.geometry("1600x900")

        # Create GUI queue immediately so it's always available
        self.gui_queue = queue.Queue()
        
        # --- ENTERPRISE CYBER THEME ---
        self.colors = {
            "bg": "#070B14",          # Deep space background
            "sidebar": "#0D1322",     # Slightly lighter sidebar
            "panel": "#131A2D",       # Main widget panels
            "border": "#1E293B",      # Subtle borders
            "accent": "#00D2FF",      # Cyber cyan
            "text": "#E2E8F0",        # Crisp white text
            "muted": "#64748B",       # Gray text
            "danger": "#FF3366",      # Neon Red
            "warn": "#FFB800",        # Warning Orange
            "success": "#00E676"      # Secure Green
        }
        
        self.root.configure(bg=self.colors["bg"])
        plt.style.use('dark_background') # Base style for charts

        self.anomaly_detector = IoTAnomalyDetector()
        self.ml_engine = IntrusionDetectionEngine()
        self.raw_df = None
        self.target_col = None
        self.is_monitoring = False
        self.timeline_data = []

        self.apply_styles()
        self.build_ui()
        self._load_pipeline_silently()

        # Start processing the queue
        self.process_gui_queue()

    def process_gui_queue(self):
        """Safely processes UI updates in the main thread."""
        try:
            while True:
                task = self.gui_queue.get_nowait()
                task()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_gui_queue)

    def apply_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Sidebar.TFrame', background=self.colors["sidebar"])
        style.configure('Main.TFrame', background=self.colors["bg"])
        style.configure('Panel.TFrame', background=self.colors["panel"])
        
        # Standard buttons
        style.configure('Action.TButton', background=self.colors["border"], foreground=self.colors["text"], 
                        font=("Segoe UI", 10, "bold"), borderwidth=0, padding=10)
        style.map('Action.TButton', background=[('active', self.colors["muted"])])
        
        # Primary Action Button
        style.configure('Start.TButton', background=self.colors["accent"], foreground="#000", 
                        font=("Segoe UI", 11, "bold"), borderwidth=0, padding=12)
        style.map('Start.TButton', background=[('active', "#0099CC")])

        # Danger Button
        style.configure('Stop.TButton', background=self.colors["danger"], foreground="#FFF", 
                        font=("Segoe UI", 11, "bold"), borderwidth=0, padding=12)
        style.map('Stop.TButton', background=[('active', "#CC0033")])

    def create_panel(self, parent, title):
        """Helper to create a unified SCADA card/panel"""
        frame = tk.Frame(parent, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
        if title:
            lbl = tk.Label(frame, text=title.upper(), font=("Segoe UI", 10, "bold"), fg=self.colors["muted"], bg=self.colors["panel"], anchor="w")
            lbl.pack(fill="x", padx=15, pady=(15, 5))
        return frame

    def build_ui(self):
        # --- LEFT SIDEBAR ---
        sidebar = ttk.Frame(self.root, style='Sidebar.TFrame', width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Branding
        tk.Label(sidebar, text="SCADA-GUARD", font=("Segoe UI", 18, "bold", "italic"), fg=self.colors["accent"], bg=self.colors["sidebar"]).pack(pady=(30, 5))
        tk.Label(sidebar, text="AI INTRUSION SYSTEM", font=("Segoe UI", 9, "bold"), fg=self.colors["muted"], bg=self.colors["sidebar"]).pack(pady=(0, 30))

        # Controls
        ttk.Button(sidebar, text="📂 LOAD DATASET", style='Action.TButton', command=self.load_data).pack(fill="x", padx=20, pady=10)
        ttk.Button(sidebar, text="🧠 TRAIN AI MODEL", style='Action.TButton', command=self.train_ai).pack(fill="x", padx=20, pady=10)
        
        self.btn_monitor = ttk.Button(sidebar, text="▶ START MONITORING", style='Start.TButton', command=self.toggle_monitor)
        self.btn_monitor.pack(fill="x", padx=20, pady=30)

        # Status
        tk.Label(sidebar, text="SYSTEM STATUS", font=("Segoe UI", 9, "bold"), fg=self.colors["muted"], bg=self.colors["sidebar"]).pack(pady=(20, 0))
        self.lbl_status = tk.Label(sidebar, text="STANDBY", font=("Consolas", 14, "bold"), fg=self.colors["text"], bg=self.colors["sidebar"])
        self.lbl_status.pack(pady=5)

        # --- MAIN CONTENT AREA ---
        main_area = ttk.Frame(self.root, style='Main.TFrame')
        main_area.pack(side="left", fill="both", expand=True, padx=20, pady=20)

        # Grid Configuration
        main_area.grid_rowconfigure(0, weight=0) # Counters
        main_area.grid_rowconfigure(1, weight=3) # Charts Top
        main_area.grid_rowconfigure(2, weight=2) # Logs/Heatmap
        main_area.grid_columnconfigure(0, weight=3) # Left wide col
        main_area.grid_columnconfigure(1, weight=2) # Right narrow col

        # 1. Threat Counters (Row 0, Col 0-1)
        counter_container = tk.Frame(main_area, bg=self.colors["bg"])
        # FIX: Changed fill="x" to sticky="ew" for the grid manager
        counter_container.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        
        self.counter_vars = {}
        threats = [("Sensor Spoofing", self.colors["warn"]), ("Command Injection", self.colors["danger"]), 
                   ("Network DDoS", self.colors["danger"]), ("Replay Attack", self.colors["accent"])]
        
        for i, (threat, color) in enumerate(threats):
            card = tk.Frame(counter_container, bg=self.colors["panel"], highlightbackground=self.colors["border"], highlightthickness=1)
            card.pack(side="left", fill="both", expand=True, padx=(0 if i==0 else 10, 0))
            tk.Label(card, text=threat.upper(), font=("Segoe UI", 9, "bold"), fg=self.colors["muted"], bg=self.colors["panel"]).pack(pady=(15, 0))
            var = tk.StringVar(value="0")
            self.counter_vars[threat] = var
            tk.Label(card, textvariable=var, font=("Consolas", 24, "bold"), fg=color, bg=self.colors["panel"]).pack(pady=(0, 15))

        # 2. Timeline Chart (Row 1, Col 0)
        timeline_panel = self.create_panel(main_area, "Live Network Anomaly Trend")
        timeline_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 20), pady=(0, 20))
        self.init_timeline(timeline_panel)

        # 3. AI Gauge & Radar (Row 1, Col 1) - Stacked
        right_panel = self.create_panel(main_area, "AI Anomaly Confidence")
        right_panel.grid(row=1, column=1, sticky="nsew", pady=(0, 20))
        self.init_ai_meter(right_panel)

        # 4. Heatmap/Sensors (Row 2, Col 0)
        heatmap_panel = self.create_panel(main_area, "Sensor Data Signatures (Scaled)")
        heatmap_panel.grid(row=2, column=0, sticky="nsew", padx=(0, 20))
        self.init_heatmap(heatmap_panel)

        # 5. Terminal Logs (Row 2, Col 1)
        log_panel = self.create_panel(main_area, "Security Event Terminal")
        log_panel.grid(row=2, column=1, sticky="nsew")
        
        self.defense_log = tk.Text(log_panel, bg="#05080F", fg=self.colors["text"], font=("Consolas", 10), 
                                   bd=0, highlightthickness=0, state="disabled")
        self.defense_log.pack(fill="both", expand=True, padx=15, pady=15)
        # Setup text tags for color coding logs
        self.defense_log.tag_config("CRITICAL", foreground=self.colors["danger"])
        self.defense_log.tag_config("HIGH", foreground=self.colors["warn"])
        self.defense_log.tag_config("SUSPICIOUS", foreground=self.colors["accent"])

    # --- MATPLOTLIB & CANVAS WIDGETS ---
    def init_timeline(self, parent):
        fig, ax = plt.subplots(figsize=(8, 3), facecolor=self.colors["panel"])
        self.timeline_ax = ax
        self.timeline_ax.set_facecolor(self.colors["panel"])
        # Clean chart look
        for spine in self.timeline_ax.spines.values(): spine.set_visible(False)
        self.timeline_ax.tick_params(colors=self.colors["muted"], bottom=False, left=False)
        self.timeline_ax.set_ylim(-0.05, 1.05)
        self.timeline_ax.grid(color=self.colors["border"], linestyle='--', alpha=0.5, axis='y')
        
        self.timeline_line, = self.timeline_ax.plot([], [], color=self.colors["success"], linewidth=2)
        self.timeline_fill = None

        self.timeline_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.timeline_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def update_timeline(self, prob):
        self.timeline_data.append(prob)
        if len(self.timeline_data) > 60: self.timeline_data.pop(0)
        
        color = self.colors["danger"] if prob >= self.ml_engine.alert_threshold else self.colors["accent"]
        x_data = range(len(self.timeline_data))
        
        self.timeline_line.set_data(x_data, self.timeline_data)
        self.timeline_line.set_color(color)
        
        # dynamic fill for aesthetics
        if self.timeline_fill: self.timeline_fill.remove()
        self.timeline_fill = self.timeline_ax.fill_between(x_data, self.timeline_data, 0, color=color, alpha=0.1)

        self.timeline_ax.set_xlim(0, max(60, len(self.timeline_data)))
        self.timeline_canvas.draw()

    def init_heatmap(self, parent):
        fig, ax = plt.subplots(figsize=(8, 2), facecolor=self.colors["panel"])
        self.heat_ax = ax
        self.heat_ax.set_facecolor(self.colors["panel"])
        self.heat_ax.axis('off')
        self.feature_bars = None
        self.heat_canvas = FigureCanvasTkAgg(fig, master=parent)
        self.heat_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def update_heatmap(self, row):
        num_data = row.to_frame().T.select_dtypes(include=[np.number])
        if self.target_col in num_data.columns: num_data = num_data.drop(columns=[self.target_col])
        
        if not num_data.empty:
            values = num_data.values[0]
            max_val = np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else 1
            scaled = np.abs(values) / max_val
            
            if self.feature_bars is None or len(self.feature_bars) != len(values):
                self.heat_ax.clear()
                self.heat_ax.axis('off')
                self.feature_bars = self.heat_ax.bar(range(len(scaled)), scaled, color=self.colors["accent"], width=0.8)
            else:
                for bar, h in zip(self.feature_bars, scaled):
                    bar.set_height(h)
                    bar.set_color(self.colors["danger"] if h > 0.85 else self.colors["accent"])
        self.heat_canvas.draw()

    def init_ai_meter(self, parent):
        # Custom drawn sleek gauge
        self.meter = tk.Canvas(parent, bg=self.colors["panel"], highlightthickness=0)
        self.meter.pack(fill="both", expand=True, pady=20)
        self.meter.bind("<Configure>", self._draw_meter_bg) 

    def _draw_meter_bg(self, event=None):
        self.update_ai_meter(0) 

    def update_ai_meter(self, prob):
        self.meter.delete("all")
        w, h = self.meter.winfo_width(), self.meter.winfo_height()
        if w < 10 or h < 10: w, h = 300, 200 
        
        cx, cy = w / 2, h * 0.75
        r = min(w, h) * 0.45
        
        color = self.colors["danger"] if prob >= self.ml_engine.alert_threshold else self.colors["success"]
        angle = prob * 180
        
        # Background Track
        self.meter.create_arc(cx-r, cy-r, cx+r, cy+r, start=0, extent=180, outline=self.colors["bg"], width=15, style=tk.ARC)
        # Fill Track
        self.meter.create_arc(cx-r, cy-r, cx+r, cy+r, start=180, extent=-angle, outline=color, width=15, style=tk.ARC)
        
        status = "CRITICAL" if prob > 0.90 else "ANOMALY" if prob >= self.ml_engine.alert_threshold else "SECURE"
        
        self.meter.create_text(cx, cy - 30, text=f"{prob:.1%}", fill=self.colors["text"], font=("Consolas", 28, "bold"))
        self.meter.create_text(cx, cy + 10, text=status, fill=color, font=("Segoe UI", 12, "bold"))

    def update_defense_ui(self):
        # Update Counter Widgets
        for threat, count in self.anomaly_detector.attack_counts.items():
            if threat in self.counter_vars:
                self.counter_vars[threat].set(str(count))
            
        # Update Terminal
        self.defense_log.config(state="normal")
        self.defense_log.delete(1.0, tk.END)
        with self.anomaly_detector.lock:
            for log_entry, severity in self.anomaly_detector.recent_attacks:
                self.defense_log.insert(tk.END, log_entry + "\n", severity)
        self.defense_log.config(state="disabled")

    # --- LOGIC & BINDINGS ---
    def _set_status(self, text, fg=None):
        def _apply(): 
            self.lbl_status.config(text=text, fg=fg if fg else self.colors["text"])
        self.gui_queue.put(_apply) # PUSH TO QUEUE

    def load_data(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path: return
        self.raw_df, self.target_col = DataPipeline.load_and_prep_data(path)
        self._set_status(f"DATA: {len(self.raw_df)} ROWS", self.colors["accent"])

    def train_ai(self):
        if self.raw_df is None: return messagebox.showwarning("Warning", "Load dataset first.")
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        try:
            self._set_status("TRAINING AI...", self.colors["warn"])
            X = self.raw_df.drop(columns=[self.target_col, "Timestamp"], errors="ignore")
            y = self.raw_df[self.target_col]

            num_cols = X.select_dtypes(include=[np.number]).columns
            cat_cols = X.select_dtypes(include=["object"]).columns

            transformers = []
            if len(num_cols) > 0: transformers.append(("num", StandardScaler(), num_cols))
            if len(cat_cols) > 0: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

            self.ml_engine.pipeline = Pipeline([
                ("prep", ColumnTransformer(transformers, remainder="drop")),
                ("clf", RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.ml_engine.pipeline.fit(X_train, y_train)
            joblib.dump(self.ml_engine.pipeline, PIPELINE_FILE)
            self._set_status("AI OPTIMIZED", self.colors["success"])
        except Exception as e:
            self._set_status("TRAIN FAILED", self.colors["danger"])
            self.gui_queue.put(lambda: messagebox.showerror("Error", str(e))) 

    def _load_pipeline_silently(self):
        try:
            self.ml_engine.pipeline = joblib.load(PIPELINE_FILE)
            self._set_status("AI LOADED", self.colors["success"])
        except: pass

    def toggle_monitor(self):
        if getattr(self.ml_engine, 'pipeline', None) is None or self.raw_df is None:
            return messagebox.showerror("Error", "Pipeline or Data missing. Load Data and Train.")

        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.btn_monitor.config(text="⏹ STOP MONITORING", style='Stop.TButton')
            self._set_status("🛡️ MONITORING", self.colors["success"])
            threading.Thread(target=self.stream_sensor_data, daemon=True).start()
        else:
            self.btn_monitor.config(text="▶ START MONITORING", style='Start.TButton')
            self._set_status("STANDBY", self.colors["muted"])

    def stream_sensor_data(self):
        sim_data = self.raw_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        for _, row in sim_data.iterrows():
            if not self.is_monitoring: break

            features = row.to_frame().T.drop(columns=[self.target_col, "Timestamp"], errors="ignore")
            features.columns = features.columns.astype(str).str.strip()
            
            try: prob = self.ml_engine.pipeline.predict_proba(features)[0][1]
            except: prob = 0.0


            self.gui_queue.put(lambda p=prob: self.update_ai_meter(p))
            self.gui_queue.put(lambda p=prob: self.update_timeline(p))
            self.gui_queue.put(lambda r=row: self.update_heatmap(r))

            num_df = features.select_dtypes(include=[np.number])
            hard_rule_triggered = False if num_df.empty else np.max(num_df.values[0]) > 99999

            if self.ml_engine.is_anomalous(prob) or hard_rule_triggered:
                is_attack, attack_type = self.anomaly_detector.classify_attack(features, prob)
                if is_attack:
                    self.gui_queue.put(self.update_defense_ui) 
                    DataPipeline.log_alert_to_csv({"timestamp": datetime.now().isoformat(), "score": round(prob,3), "type": attack_type})
            else:
                self.anomaly_detector.classify_attack(features, prob) 

            time.sleep(0.4) 

if __name__=="__main__":
    root = tk.Tk()
    app = IndustrialDashboard(root)
    root.mainloop()
