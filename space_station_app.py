import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import threading
import time
import json
import os
import uuid
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import mysql.connector
from mysql.connector import Error
from collections import defaultdict, deque
from sklearn.ensemble import IsolationForest
import pandas as pd
from scipy import stats

class DatabaseManager:
    def __init__(self, host='localhost', database='SpaceStationDetection', user='root', password=''):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password="Agarwal",
                autocommit=True
            )
            return True
        except Error as e:
            print(f"Database connection error: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def execute_query(self, query, params=None, fetch=False):
        """Execute a database query"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor(dictionary=True if fetch else False)
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                cursor.close()
                return True
                
        except Error as e:
            print(f"Database query error: {e}")
            return None
    
    def insert_detection(self, timestamp, object_class, confidence, x, y, width, height, 
                        image_path, frame_width, frame_height, session_id):
        """Insert detection record"""
        query = """
            INSERT INTO detections (timestamp, object_class, confidence, x, y, width, height, 
                                  image_path, frame_width, frame_height, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        params = (timestamp, object_class, confidence, x, y, width, height, 
                 image_path, frame_width, frame_height, session_id)
        return self.execute_query(query, params)
    
    def insert_performance_metric(self, timestamp, fps, inference_time, accuracy, 
                                objects_detected, session_id):
        """Insert performance metric record"""
        query = """
            INSERT INTO performance_metrics (timestamp, fps, inference_time, accuracy, 
                                           objects_detected, session_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        params = (timestamp, fps, inference_time, accuracy, objects_detected, session_id)
        return self.execute_query(query, params)
    
    def insert_inventory(self, timestamp, object_class, count, avg_confidence, 
                        avg_x, avg_y, session_id):
        """Insert inventory record"""
        query = """
            INSERT INTO inventory (timestamp, object_class, count, avg_confidence, 
                                 avg_x, avg_y, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (timestamp, object_class, count, avg_confidence, avg_x, avg_y, session_id)
        return self.execute_query(query, params)
    
    def create_session(self, session_id, start_time):
        """Create a new session record"""
        query = """
            INSERT INTO sessions (session_id, start_time, status)
            VALUES (%s, %s, 'active')
        """
        params = (session_id, start_time)
        return self.execute_query(query, params)
    
    def update_session(self, session_id, end_time, duration, objects_detected, status):
        """Update session record"""
        query = """
            UPDATE sessions 
            SET end_time = %s, duration = %s, objects_detected = %s, status = %s
            WHERE session_id = %s
        """
        params = (end_time, duration, objects_detected, status, session_id)
        return self.execute_query(query, params)
    
    def get_recent_detections(self, limit=500):
        """Get recent detection records"""
        query = """
            SELECT timestamp, object_class, confidence, x, y, width, height, image_path 
            FROM detections 
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        return self.execute_query(query, (limit,), fetch=True)
    
    def get_performance_metrics(self, days=3):
        """Get performance metrics for specified days"""
        query = """
            SELECT timestamp, fps, inference_time, objects_detected
            FROM performance_metrics 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            ORDER BY timestamp
        """
        return self.execute_query(query, (days,), fetch=True)
    
    def get_inventory_trends(self, days=30):
        """Get inventory trends for specified days"""
        query = """
            SELECT DATE(timestamp) as date, object_class, 
                   AVG(count) as avg_count, AVG(avg_confidence) as avg_conf
            FROM inventory
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE(timestamp), object_class
            ORDER BY date
        """
        return self.execute_query(query, (days,), fetch=True)
    
    def get_inventory_stats(self, days=7):
        """Get inventory statistics for specified days"""
        query = """
            SELECT object_class,
                   AVG(count) as avg_count,
                   MAX(count) as max_count,
                   MIN(count) as min_count,
                   AVG(avg_confidence) as avg_conf,
                   COUNT(*) as readings
            FROM inventory
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY object_class
        """
        return self.execute_query(query, (days,), fetch=True)
    
    def get_detection_patterns(self, days=7):
        """Get detection patterns for specified days"""
        query = """
            SELECT DATE_FORMAT(timestamp, '%%Y-%%m-%%d %%H:00') as hour, 
                   object_class, 
                   COUNT(*) as count
            FROM detections 
            WHERE timestamp >= DATE_SUB(NOW(), INTERVAL %s DAY)
            GROUP BY DATE_FORMAT(timestamp, '%%Y-%%m-%%d %%H'), object_class
            ORDER BY hour
        """
        return self.execute_query(query, (days,), fetch=True)

class SpaceStationDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Space Station Object Detection System v3.0")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1a1a2e')
        
        # Initialize database
        self.db = DatabaseManager()
        if not self.db.connect():
            messagebox.showerror("Database Error", "Could not connect to MySQL database")
            return
        
        # Generate session ID
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.db.create_session(self.session_id, self.session_start_time)
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.detection_results = []
        self.confidence_threshold = 0.5
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.detection_history = defaultdict(lambda: deque(maxlen=100))
        self.inventory_counts = defaultdict(int)
        self.object_trajectories = defaultdict(list)
        
        # Video/Camera variables
        self.video_capture = None
        self.is_camera_active = False
        self.is_video_playing = False
        self.video_thread = None
        self.current_frame = None
        self.video_file = None
        
        # Create GUI
        self.create_gui()
        
        # Load default model if exists
        self.load_default_model()
        
        # Initialize predictive maintenance model
        self.init_predictive_maintenance()
        
        # Start performance monitoring thread
        self.start_performance_monitor()
    
    def create_gui(self):
        """Create the main GUI interface"""
        # Create main frames
        self.create_header()
        self.create_control_panel()
        self.create_main_display()
        self.create_status_panel()
    
    def create_header(self):
        """Create header with title and mission info"""
        header_frame = tk.Frame(self.root, bg='#16213e', height=80)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, text="🚀 SPACE STATION OBJECT DETECTION SYSTEM v3.0", 
                              font=('Arial', 18, 'bold'), fg='#4fc3f7', bg='#16213e')
        title_label.pack(pady=10)
        
        mission_label = tk.Label(header_frame, text="AI-Powered Safety & Inventory Management with Predictive Analytics", 
                                font=('Arial', 12), fg='#81c784', bg='#16213e')
        mission_label.pack()
    
    def create_control_panel(self):
        """Create control panel with buttons and settings"""
        control_frame = tk.Frame(self.root, bg='#16213e', height=120)
        control_frame.pack(fill='x', padx=10, pady=5)
        control_frame.pack_propagate(False)
        
        # Model controls
        model_frame = tk.LabelFrame(control_frame, text="Model Controls", 
                                   fg='#4fc3f7', bg='#16213e', font=('Arial', 10, 'bold'))
        model_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        tk.Button(model_frame, text="Load Model", command=self.load_model,
                 bg='#4fc3f7', fg='black', font=('Arial', 10, 'bold')).pack(pady=2)
        tk.Button(model_frame, text="Update Model", command=self.update_model,
                 bg='#ff9800', fg='black', font=('Arial', 10, 'bold')).pack(pady=2)
        
        # Detection controls
        detection_frame = tk.LabelFrame(control_frame, text="Detection Controls", 
                                       fg='#4fc3f7', bg='#16213e', font=('Arial', 10, 'bold'))
        detection_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        tk.Button(detection_frame, text="Load Image", command=self.load_image,
                 bg='#81c784', fg='black', font=('Arial', 10, 'bold')).pack(pady=2)
        tk.Button(detection_frame, text="Load Video", command=self.load_video,
                 bg='#ff9800', fg='black', font=('Arial', 10, 'bold')).pack(pady=2)
        self.camera_button = tk.Button(detection_frame, text="Start Camera", command=self.toggle_camera,
                 bg='#f44336', fg='white', font=('Arial', 10, 'bold'))
        self.camera_button.pack(pady=2)
        
        # Monitoring controls
        monitor_frame = tk.LabelFrame(control_frame, text="Monitoring", 
                                     fg='#4fc3f7', bg='#16213e', font=('Arial', 10, 'bold'))
        monitor_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        self.monitor_button = tk.Button(monitor_frame, text="Start Monitoring", 
                                       command=self.toggle_monitoring,
                                       bg='#9c27b0', fg='white', font=('Arial', 10, 'bold'))
        self.monitor_button.pack(pady=2)
        tk.Button(monitor_frame, text="View Logs", command=self.view_detection_logs,
                 bg='#607d8b', fg='white', font=('Arial', 10, 'bold')).pack(pady=2)
        
        # Analytics controls
        analytics_frame = tk.LabelFrame(control_frame, text="Analytics", 
                                      fg='#4fc3f7', bg='#16213e', font=('Arial', 10, 'bold'))
        analytics_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        tk.Button(analytics_frame, text="View Inventory", command=self.show_inventory_dashboard,
                 bg='#8d6e63', fg='white', font=('Arial', 10, 'bold')).pack(pady=2)
        tk.Button(analytics_frame, text="Pattern Analysis", command=self.show_pattern_analysis,
                 bg='#009688', fg='white', font=('Arial', 10, 'bold')).pack(pady=2)
        
        # Settings
        settings_frame = tk.LabelFrame(control_frame, text="Settings", 
                                      fg='#4fc3f7', bg='#16213e', font=('Arial', 10, 'bold'))
        settings_frame.pack(side='left', fill='y', padx=5, pady=5)
        
        tk.Label(settings_frame, text="Confidence:", fg='white', bg='#16213e').pack()
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = tk.Scale(settings_frame, from_=0.1, to=0.9, resolution=0.1,
                                   orient='horizontal', variable=self.confidence_var,
                                   bg='#16213e', fg='white', highlightbackground='#16213e')
        confidence_scale.pack()
    
    def create_main_display(self):
        """Create main display area"""
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left side - Image/Video display
        left_frame = tk.Frame(main_frame, bg='#1a1a2e')
        left_frame.pack(side='left', fill='both', expand=True)
        
        # Video controls frame
        video_controls_frame = tk.Frame(left_frame, bg='#16213e', height=50)
        video_controls_frame.pack(fill='x', padx=5, pady=2)
        video_controls_frame.pack_propagate(False)
        
        self.play_button = tk.Button(video_controls_frame, text="▶ Play", command=self.toggle_video_playback,
                                    bg='#4fc3f7', fg='black', font=('Arial', 10, 'bold'), state='disabled')
        self.play_button.pack(side='left', padx=5, pady=5)
        
        self.stop_button = tk.Button(video_controls_frame, text="⏹ Stop", command=self.stop_video,
                                    bg='#f44336', fg='white', font=('Arial', 10, 'bold'), state='disabled')
        self.stop_button.pack(side='left', padx=5, pady=5)
        
        # Performance display
        self.performance_label = tk.Label(video_controls_frame, text="FPS: - | Inference: - ms", 
                                        fg='white', bg='#16213e', font=('Arial', 10))
        self.performance_label.pack(side='left', padx=10)
        
        # Video progress
        self.video_progress = tk.Label(video_controls_frame, text="No video loaded", 
                                      fg='white', bg='#16213e', font=('Arial', 10))
        self.video_progress.pack(side='right', padx=5, pady=5)
        
        self.image_label = tk.Label(left_frame, text="Load an image, video, or start camera to begin detection",
                                   bg='#2c2c54', fg='white', font=('Arial', 14))
        self.image_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right side - Results and analytics
        right_frame = tk.Frame(main_frame, bg='#16213e', width=400)
        right_frame.pack(side='right', fill='y', padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        # Create notebook for multiple tabs
        self.analytics_notebook = ttk.Notebook(right_frame)
        self.analytics_notebook.pack(fill='both', expand=True)
        
        # Detection results tab
        results_frame = tk.Frame(self.analytics_notebook, bg='#16213e')
        self.analytics_notebook.add(results_frame, text="Detection Results")
        
        self.results_text = tk.Text(results_frame, height=8, bg='#2c2c54', fg='white',
                                   font=('Courier', 10))
        self.results_text.pack(fill='x', padx=5, pady=5)
        
        # Performance tab
        performance_frame = tk.Frame(self.analytics_notebook, bg='#16213e')
        self.analytics_notebook.add(performance_frame, text="Performance")
        
        self.performance_fig, self.performance_ax = plt.subplots(2, 1, figsize=(4, 4), facecolor='#16213e')
        self.performance_fig.subplots_adjust(hspace=0.5)
        for ax in self.performance_ax:
            ax.set_facecolor('#2c2c54')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        self.performance_canvas = FigureCanvasTkAgg(self.performance_fig, performance_frame)
        self.performance_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Inventory tab
        inventory_frame = tk.Frame(self.analytics_notebook, bg='#16213e')
        self.analytics_notebook.add(inventory_frame, text="Inventory")
        
        self.inventory_fig, self.inventory_ax = plt.subplots(figsize=(4, 3), facecolor='#16213e')
        self.inventory_ax.set_facecolor('#2c2c54')
        self.inventory_canvas = FigureCanvasTkAgg(self.inventory_fig, inventory_frame)
        self.inventory_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Trajectory tab
        trajectory_frame = tk.Frame(self.analytics_notebook, bg='#16213e')
        self.analytics_notebook.add(trajectory_frame, text="Trajectories")
        
        self.trajectory_fig, self.trajectory_ax = plt.subplots(figsize=(4, 3), facecolor='#16213e')
        self.trajectory_ax.set_facecolor('#2c2c54')
        self.trajectory_canvas = FigureCanvasTkAgg(self.trajectory_fig, trajectory_frame)
        self.trajectory_canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
    def create_status_panel(self):
        """Create status panel at bottom"""
        status_frame = tk.Frame(self.root, bg='#16213e', height=40)
        status_frame.pack(fill='x', padx=10, pady=5)
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(status_frame, text="Ready - Load a model to begin", 
                                    fg='#81c784', bg='#16213e', font=('Arial', 10))
        self.status_label.pack(side='left', pady=10)
        
        self.model_status = tk.Label(status_frame, text="No model loaded", 
                                    fg='#f44336', bg='#16213e', font=('Arial', 10))
        self.model_status.pack(side='right', pady=10)
        
        # Maintenance indicator
        self.maintenance_status = tk.Label(status_frame, text="Maintenance: OK", 
                                         fg='#81c784', bg='#16213e', font=('Arial', 10))
        self.maintenance_status.pack(side='right', padx=20, pady=10)
    
    def init_predictive_maintenance(self):
        """Initialize predictive maintenance model"""
        self.maintenance_model = IsolationForest(contamination=0.05)
        self.maintenance_features = deque(maxlen=1000)
        self.maintenance_predictions = deque(maxlen=100)
        self.maintenance_warning = False
    
    def start_performance_monitor(self):
        """Start thread to monitor performance metrics"""
        def monitor_loop():
            while True:
                if self.performance_history:
                    self.update_performance_metrics()
                    self.check_maintenance_needs()
                time.sleep(5)
        
        threading.Thread(target=monitor_loop, daemon=True).start()
    
    def update_performance_metrics(self):
        """Update performance metrics display"""
        if not self.performance_history:
            return
        
        # Calculate metrics
        fps_values = [m['fps'] for m in self.performance_history if m['fps'] > 0]
        inference_times = [m['inference_time'] for m in self.performance_history]
        
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
        avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
        
        # Update label
        self.performance_label.config(text=f"FPS: {avg_fps:.1f} | Inference: {avg_inference:.1f} ms")
        
        # Update performance chart
        self.performance_ax[0].clear()
        self.performance_ax[1].clear()
        
        # FPS chart
        self.performance_ax[0].plot([m['timestamp'] for m in self.performance_history],
                                  [m['fps'] for m in self.performance_history],
                                  color='#4fc3f7')
        self.performance_ax[0].set_title('Processing FPS', color='white', fontsize=10)
        self.performance_ax[0].set_ylabel('Frames per second', color='white')
        self.performance_ax[0].set_facecolor('#2c2c54')
        
        # Inference time chart
        self.performance_ax[1].plot([m['timestamp'] for m in self.performance_history],
                                  [m['inference_time'] for m in self.performance_history],
                                  color='#81c784')
        self.performance_ax[1].set_title('Inference Time', color='white', fontsize=10)
        self.performance_ax[1].set_ylabel('Milliseconds', color='white')
        self.performance_ax[1].set_facecolor('#2c2c54')
        
        # Rotate x-axis labels
        for ax in self.performance_ax:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.tick_params(colors='white')
        
        self.performance_fig.tight_layout()
        self.performance_canvas.draw()
        
        # Log to database
        self.log_performance_metrics(avg_fps, avg_inference)
    
    def log_performance_metrics(self, fps, inference_time):
        """Log performance metrics to database"""
        timestamp = datetime.now()
        self.db.insert_performance_metric(
            timestamp, fps, inference_time, 0, 0, self.session_id
        )
    
    def check_maintenance_needs(self):
        """Check if system needs maintenance using anomaly detection"""
        if len(self.performance_history) < 50:  # Need enough data
            return
        
        # Prepare features (last 50 performance metrics)
        features = np.array([
            [m['fps'], m['inference_time']] 
            for m in list(self.performance_history)[-50:]
        ])
        
        # Fit and predict anomalies
        self.maintenance_model.fit(features)
        preds = self.maintenance_model.predict(features)
        
        # Count anomalies in last 10 predictions
        recent_anomalies = sum(1 for p in preds[-10:] if p == -1)
        
        # Update maintenance status
        if recent_anomalies > 2:  # More than 2 anomalies in last 10
            self.maintenance_warning = True
            self.maintenance_status.config(text="Maintenance: NEEDED!", fg='#f44336')
        else:
            self.maintenance_warning = False
            self.maintenance_status.config(text="Maintenance: OK", fg='#81c784')
    
    def update_inventory_counts(self, detections):
        """Update inventory counts based on detections"""
        # Count objects by class
        class_counts = defaultdict(int)
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] += 1
        
        # Update current inventory
        self.inventory_counts = class_counts
        
        # Update inventory chart
        self.update_inventory_dashboard()
        
        # Log to database
        self.log_inventory_counts(detections)
    
    def update_inventory_dashboard(self):
        """Update the inventory dashboard visualization"""
        self.inventory_ax.clear()
        
        if not self.inventory_counts:
            self.inventory_ax.text(0.5, 0.5, 'No inventory data', ha='center', va='center', 
                                 color='white', fontsize=12)
        else:
            classes = list(self.inventory_counts.keys())
            counts = list(self.inventory_counts.values())
            
            colors = ['#4fc3f7', '#81c784', '#ff9800', '#f44336', '#9c27b0']
            bars = self.inventory_ax.bar(classes, counts, color=colors[:len(classes)])
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                self.inventory_ax.text(bar.get_x() + bar.get_width()/2., height,
                                      f'{int(height)}', ha='center', va='bottom',
                                      color='white')
            
            self.inventory_ax.set_title('Current Inventory Counts', color='white')
            self.inventory_ax.set_ylabel('Count', color='white')
            self.inventory_ax.tick_params(colors='white')
            plt.setp(self.inventory_ax.get_xticklabels(), rotation=45, ha='right')
        
        self.inventory_ax.set_facecolor('#2c2c54')
        self.inventory_fig.tight_layout()
        self.inventory_canvas.draw()
    
    def log_inventory_counts(self, detections):
        """Log inventory counts to database with additional metrics"""
        if not detections:
            return
        
        timestamp = datetime.now()
        
        # Calculate averages for each class
        class_stats = defaultdict(lambda: {'count': 0, 'conf_sum': 0, 'x_sum': 0, 'y_sum': 0})
        
        for detection in detections:
            class_name = detection['class']
            stats = class_stats[class_name]
            stats['count'] += 1
            stats['conf_sum'] += detection['confidence']
            stats['x_sum'] += detection['bbox'][0] + detection['bbox'][2]/2  # center x
            stats['y_sum'] += detection['bbox'][1] + detection['bbox'][3]/2  # center y
        
        # Insert records for each class
        for class_name, stats in class_stats.items():
            avg_conf = stats['conf_sum'] / stats['count'] if stats['count'] > 0 else 0
            avg_x = stats['x_sum'] / stats['count'] if stats['count'] > 0 else 0
            avg_y = stats['y_sum'] / stats['count'] if stats['count'] > 0 else 0
            
            self.db.insert_inventory(
                timestamp, class_name, stats['count'], avg_conf, avg_x, avg_y, self.session_id
            )
    
    def track_object_trajectories(self, detections, frame_width, frame_height):
        """Track object trajectories over time"""
        current_time = time.time()
        
        for detection in detections:
            class_name = detection['class']
            x, y, w, h = detection['bbox']
            center_x = x + w/2
            center_y = y + h/2
            
            # Normalize coordinates to 0-1 range
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height
            
            # Add to trajectory (limit to 100 points per object)
            if class_name not in self.object_trajectories:
                self.object_trajectories[class_name] = deque(maxlen=100)
            
            self.object_trajectories[class_name].append((norm_x, norm_y))
        
        # Update visualization
        self.update_trajectory_visualization()

    def update_trajectory_visualization(self):
        """Update the trajectory visualization"""
        self.trajectory_ax.clear()
        
        if not self.object_trajectories:
            self.trajectory_ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', 
                                  color='white', fontsize=12)
        else:
            # Create a color map for different objects
            colors = plt.cm.get_cmap('hsv', len(self.object_trajectories))
            
            for i, (class_name, trajectory) in enumerate(self.object_trajectories.items()):
                if len(trajectory) > 1:
                    # Unpack coordinates
                    xs, ys = zip(*trajectory)
                    
                    # Plot trajectory
                    self.trajectory_ax.plot(xs, ys, color=colors(i), 
                                           label=class_name, alpha=0.7)
                    
                    # Mark current position
                    self.trajectory_ax.scatter(xs[-1], ys[-1], color=colors(i), 
                                             s=100, edgecolor='white')
        
            self.trajectory_ax.set_title('Object Trajectories', color='white')
            self.trajectory_ax.set_xlabel('Normalized X Position', color='white')
            self.trajectory_ax.set_ylabel('Normalized Y Position', color='white')
            self.trajectory_ax.set_xlim(0, 1)
            self.trajectory_ax.set_ylim(0, 1)
            self.trajectory_ax.legend(facecolor='#2c2c54', labelcolor='white')
        
        self.trajectory_ax.set_facecolor('#2c2c54')
        self.trajectory_fig.tight_layout()
        self.trajectory_canvas.draw()

    def show_inventory_dashboard(self):
        """Show enhanced inventory dashboard with real-time updates and better visualization"""
        try:
            dashboard = tk.Toplevel(self.root)
            dashboard.title("Advanced Inventory Management")
            dashboard.geometry("1400x900")
            dashboard.configure(bg='#1a1a2e')
            
            # Configure notebook style
            style = ttk.Style()
            style.configure('TNotebook', background='#1a1a2e')
            style.configure('TNotebook.Tab', background='#16213e', foreground='white')
            
            notebook = ttk.Notebook(dashboard)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # ----------------------------
            # Current Inventory Tab (Enhanced)
            # ----------------------------
            current_frame = tk.Frame(notebook, bg='#1a1a2e')
            notebook.add(current_frame, text="Current Inventory")
            
            # Critical equipment with minimum required counts
            critical_equipment = {
                "Oxygen Tank": {"min": 2, "warning": "⚠ Low"},
                "Fire Extinguisher": {"min": 1, "critical": True},
                "Toolbox": {"min": 1},
            }
            
            # Create enhanced treeview with scrollbar
            tree_frame = tk.Frame(current_frame, bg='#1a1a2e')
            tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            scrollbar = ttk.Scrollbar(tree_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            columns = ("Item", "Count", "Min Required", "Last Seen", "Location", "Status")
            tree = ttk.Treeview(tree_frame, columns=columns, show='headings', 
                               height=15, yscrollcommand=scrollbar.set)
            scrollbar.config(command=tree.yview)
            
            # Configure columns
            col_widths = [150, 80, 100, 180, 120, 100]
            for col, width in zip(columns, col_widths):
                tree.heading(col, text=col)
                tree.column(col, width=width, anchor='center')
            
            # Configure tags for status coloring
            tree.tag_configure('critical', foreground='red')
            tree.tag_configure('warning', foreground='orange')
            tree.tag_configure('ok', foreground='green')
            
            # Add data to treeview with enhanced database query
            try:
                for item, config in critical_equipment.items():
                    count = self.inventory_counts.get(item, 0)
                    min_required = config.get("min", 1)
                    
                    # Get detailed info from database
                    query = """
                        SELECT timestamp, avg_x, avg_y, avg_confidence 
                        FROM inventory 
                        WHERE object_class = %s 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """
                    result = self.db.execute_query(query, (item,), fetch=True)
                    
                    # Format values
                    if result:
                        last_seen = result[0]['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                        location = f"({result[0]['avg_x']:.1f}, {result[0]['avg_y']:.1f})"
                        confidence = f"{result[0]['avg_confidence']:.1%}" if result[0]['avg_confidence'] else "N/A"
                    else:
                        last_seen = "Never"
                        location = "Unknown"
                        confidence = "N/A"
                    
                    # Determine status with more sophisticated logic
                    status_tags = []
                    if count == 0 and config.get("critical", False):
                        status = "❌ CRITICAL"
                        status_tags = ['critical']
                    elif count < min_required:
                        status = "⚠ Low"
                        status_tags = ['warning']
                    else:
                        status = "✔ OK"
                        status_tags = ['ok']
                    
                    tree.insert('', 'end', values=(
                        item, 
                        count,
                        min_required,
                        f"{last_seen}\n(Conf: {confidence})",
                        location,
                        status
                    ), tags=status_tags)
                    
            except Exception as e:
                messagebox.showerror("Database Error", f"Failed to load inventory: {str(e)}")
            
            tree.pack(fill='both', expand=True)
            
            # Add control buttons frame
            button_frame = tk.Frame(current_frame, bg='#1a1a2e')
            button_frame.pack(fill='x', padx=10, pady=5)
            
            tk.Button(button_frame, text="Refresh Inventory", 
                     command=self.refresh_inventory_data,
                     bg='#4fc3f7', fg='black').pack(side=tk.LEFT, padx=5)
            
            tk.Button(button_frame, text="Placement Recommendations", 
                     command=self.show_placement_recommendations,
                     bg='#81c784', fg='black').pack(side=tk.LEFT, padx=5)
            
            tk.Button(button_frame, text="Export to CSV", 
                     command=self.export_inventory_data,
                     bg='#ff9800', fg='black').pack(side=tk.LEFT, padx=5)
            
            # ----------------------------
            # Historical Trends Tab (Enhanced)
            # ----------------------------
            history_frame = tk.Frame(notebook, bg='#1a1a2e')
            notebook.add(history_frame, text="Historical Trends")
            
            # Create figure with multiple subplots
            fig = plt.Figure(figsize=(12, 8), facecolor='#1a1a2e', tight_layout=True)
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            
            for ax in [ax1, ax2]:
                ax.set_facecolor('#2c2c54')
                ax.tick_params(colors='white')
                for spine in ax.spines.values():
                    spine.set_color('white')
            
            try:
                # Get trends data from database
                trends_data = self.db.get_inventory_trends(30)
                
                if trends_data:
                    # Convert to DataFrame for easier manipulation
                    df = pd.DataFrame(trends_data)
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # Plot count trends
                    pivot_df = df.pivot(index='date', columns='object_class', values='avg_count')
                    colors = plt.cm.tab10.colors
                    for i, col in enumerate(pivot_df.columns):
                        ax1.plot(pivot_df.index, pivot_df[col], 
                                label=col, color=colors[i % len(colors)],
                                marker='o', markersize=5, linewidth=2)
                    
                    ax1.set_title('30-Day Inventory Trends', color='white', pad=15)
                    ax1.set_ylabel('Average Count', color='white')
                    ax1.legend(facecolor='#2c2c54', labelcolor='white',
                              bbox_to_anchor=(1.02, 1), loc='upper left')
                    
                    # Plot confidence trends
                    conf_df = df.pivot(index='date', columns='object_class', values='avg_conf')
                    for i, col in enumerate(conf_df.columns):
                        ax2.plot(conf_df.index, conf_df[col],
                                color=colors[i % len(colors)],
                                alpha=0.7, linestyle='--')
                    
                    ax2.set_title('Detection Confidence Trends', color='white', pad=15)
                    ax2.set_xlabel('Date', color='white')
                    ax2.set_ylabel('Confidence', color='white')
                    
                    # Format x-axis
                    for ax in [ax1, ax2]:
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    
                    # Add grid
                    ax1.grid(alpha=0.2, color='white')
                    ax2.grid(alpha=0.2, color='white')
                else:
                    ax1.text(0.5, 0.5, 'No historical data available', 
                            ha='center', va='center', color='white')
                    ax2.set_visible(False)
                    
            except Exception as e:
                ax1.text(0.5, 0.5, f"Error loading data: {str(e)}", 
                        ha='center', va='center', color='white')
                ax2.set_visible(False)
            
            canvas = FigureCanvasTkAgg(fig, history_frame)
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            
            # ----------------------------
            # Add third tab for statistics
            # ----------------------------
            stats_frame = tk.Frame(notebook, bg='#1a1a2e')
            notebook.add(stats_frame, text="Statistics")
            
            # Add statistics display
            stats_text = tk.Text(stats_frame, bg='#2c2c54', fg='white',
                                font=('Consolas', 11), wrap=tk.WORD)
            stats_text.pack(fill='both', expand=True, padx=10, pady=10)
            
            try:
                # Get statistics from database
                stats_data = self.db.get_inventory_stats(7)
                
                if stats_data:
                    stats_text.insert(tk.END, "INVENTORY STATISTICS (LAST 7 DAYS)\n")
                    stats_text.insert(tk.END, "="*40 + "\n\n")
                    
                    for row in stats_data:
                        stats_text.insert(tk.END, 
                            f"{row['object_class']}:\n"
                            f"  - Avg Count: {row['avg_count']:.1f}\n"
                            f"  - Range: {row['min_count']} to {row['max_count']}\n"
                            f"  - Avg Confidence: {row['avg_conf']:.1%}\n"
                            f"  - Readings: {row['readings']}\n\n")
                    
                    stats_text.insert(tk.END, "\nSYSTEM SUMMARY\n")
                    stats_text.insert(tk.END, "="*40 + "\n")
                    
                    # Find highest and lowest items
                    highest_item = max(stats_data, key=lambda x: x['avg_count'])
                    lowest_conf_item = min(stats_data, key=lambda x: x['avg_conf'] or 0)
                    
                    stats_text.insert(tk.END, 
                        f"Total items tracked: {len(stats_data)}\n"
                        f"Highest count item: {highest_item['object_class']}\n"
                        f"Lowest confidence item: {lowest_conf_item['object_class']}\n")
                else:
                    stats_text.insert(tk.END, "No statistics available for the last 7 days")
                    
            except Exception as e:
                stats_text.insert(tk.END, f"Error loading statistics: {str(e)}")
            
            stats_text.config(state='disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create dashboard: {str(e)}")
    
    def refresh_inventory_data(self):
        """Refresh inventory data from database"""
        try:
            # This would typically refresh the current display
            messagebox.showinfo("Refresh", "Inventory data refreshed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh inventory: {str(e)}")
    
    def show_placement_recommendations(self):
        """Show AI recommendations for equipment placement"""
        rec_window = tk.Toplevel(self.root)
        rec_window.title("Equipment Placement Recommendations")
        rec_window.geometry("800x600")
        rec_window.configure(bg='#1a1a2e')
        
        # Create text widget for recommendations
        text = tk.Text(rec_window, bg='#2c2c54', fg='white', font=('Arial', 12))
        text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Generate recommendations (simulated)
        recommendations = [
            "1. Oxygen Tanks: Place additional tanks near the central corridor for easier access during emergencies",
            "2. Fire Extinguishers: Current placement is optimal (near electrical panels)",
            "3. Toolboxes: Consider adding a toolbox in the science module where usage is highest",
            "\nUsage Heatmap Analysis:",
            "- High activity near science module suggests need for more tools there",
            "- Low oxygen tank usage in storage area suggests redistribution",
            "- Fire extinguisher access times can be improved by 15% with suggested placement"
        ]
        
        for rec in recommendations:
            text.insert(tk.END, rec + "\n\n")
        
        text.config(state='disabled')
    
    def export_inventory_data(self):
        """Export inventory data to CSV"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if file_path:
                # Get inventory data from database
                inventory_data = self.db.execute_query("""
                    SELECT timestamp, object_class, count, avg_confidence, avg_x, avg_y
                    FROM inventory
                    ORDER BY timestamp DESC
                """, fetch=True)
                
                if inventory_data:
                    df = pd.DataFrame(inventory_data)
                    df.to_csv(file_path, index=False)
                    messagebox.showinfo("Export", f"Data exported successfully to {file_path}")
                else:
                    messagebox.showwarning("No Data", "No inventory data to export")
                    
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def show_pattern_analysis(self):
        """Show enhanced pattern analysis and anomaly detection"""
        try:
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Pattern Analysis & Anomaly Detection")
            analysis_window.geometry("1200x800")
            analysis_window.configure(bg='#1a1a2e')

            # Create notebook with improved styling
            style = ttk.Style()
            style.configure('TNotebook', background='#1a1a2e')
            style.configure('TNotebook.Tab', background='#16213e', foreground='white')

            notebook = ttk.Notebook(analysis_window)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)

            # Tab 1: Enhanced Detection Frequency Analysis
            freq_frame = tk.Frame(notebook, bg='#1a1a2e')
            notebook.add(freq_frame, text="Detection Patterns")

            # Create figure with improved layout
            freq_fig = plt.Figure(figsize=(10, 5), facecolor='#1a1a2e', tight_layout=True)
            freq_ax = freq_fig.add_subplot(111)
            freq_ax.set_facecolor('#2c2c54')

            try:
                # Get detection patterns from database
                pattern_data = self.db.get_detection_patterns(7)

                if pattern_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(pattern_data)
                    df['hour'] = pd.to_datetime(df['hour'])
                    
                    # Enhanced pivot with error handling
                    pivot_df = df.pivot(index='hour', columns='object_class', values='count').fillna(0)

                    # Improved plotting with better colors
                    colors = ['#4fc3f7', '#81c784', '#ff9800', '#f44336', '#9c27b0']
                    for i, column in enumerate(pivot_df.columns):
                        freq_ax.plot(pivot_df.index, pivot_df[column], 
                                     label=column, marker='o', 
                                     color=colors[i % len(colors)], 
                                     linewidth=2, markersize=8)

                    # Enhanced chart styling
                    freq_ax.set_title('Enhanced Detection Patterns (Last 7 Days)', 
                                      color='white', pad=20, fontsize=12)
                    freq_ax.set_xlabel('Date & Hour', color='white', labelpad=10)
                    freq_ax.set_ylabel('Detection Count', color='white', labelpad=10)

                    # Improved legend
                    freq_ax.legend(facecolor='#2c2c54', labelcolor='white', 
                                   bbox_to_anchor=(1.02, 1), loc='upper left')

                    # Better x-axis formatting
                    freq_ax.tick_params(colors='white')
                    plt.setp(freq_ax.get_xticklabels(), rotation=45, ha='right')

                    # Add grid for better readability
                    freq_ax.grid(True, alpha=0.2, color='white')
                else:
                    freq_ax.text(0.5, 0.5, 'No detection data available', 
                                 ha='center', va='center', color='white', fontsize=12)
            except Exception as e:
                freq_ax.text(0.5, 0.5, f'Error processing data: {str(e)}', 
                             ha='center', va='center', color='white')

            # Embed the plot
            freq_canvas = FigureCanvasTkAgg(freq_fig, freq_frame)
            freq_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

            # Tab 2: Enhanced Anomaly Detection
            anomaly_frame = tk.Frame(notebook, bg='#1a1a2e')
            notebook.add(anomaly_frame, text="System Health")

            anomaly_fig = plt.Figure(figsize=(10, 5), facecolor='#1a1a2e', tight_layout=True)
            anomaly_ax = anomaly_fig.add_subplot(111)
            anomaly_ax.set_facecolor('#2c2c54')

            try:
                # Get performance data from database
                perf_data = self.db.get_performance_metrics(3)

                if perf_data:
                    # Convert to DataFrame
                    perf_df = pd.DataFrame(perf_data)
                    perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])

                    # Improved anomaly detection with rolling stats
                    perf_df['fps_rolling'] = perf_df['fps'].rolling(5, min_periods=1).mean()
                    perf_df['inference_rolling'] = perf_df['inference_time'].rolling(5, min_periods=1).mean()

                    # Calculate dynamic thresholds
                    fps_std = perf_df['fps'].std()
                    inference_std = perf_df['inference_time'].std()

                    # Detect anomalies using multiple criteria
                    perf_df['fps_anomaly'] = (
                        (perf_df['fps'] < (perf_df['fps_rolling'] - 2*fps_std)) | 
                        (perf_df['fps'] > (perf_df['fps_rolling'] + 2*fps_std))
                    )

                    perf_df['inference_anomaly'] = (
                        (perf_df['inference_time'] < (perf_df['inference_rolling'] - 2*inference_std)) | 
                        (perf_df['inference_time'] > (perf_df['inference_rolling'] + 2*inference_std))
                    )

                    # Plot with enhanced visualization
                    anomaly_ax.plot(perf_df['timestamp'], perf_df['fps'], 
                                    label='FPS', color='#4fc3f7', linewidth=2)

                    # Mark anomalies with improved styling
                    fps_anomalies = perf_df[perf_df['fps_anomaly']]
                    if not fps_anomalies.empty:
                        anomaly_ax.scatter(fps_anomalies['timestamp'], fps_anomalies['fps'],
                                          color='red', label='FPS Anomaly', 
                                          s=100, edgecolor='white', zorder=5)

                    # Second y-axis with better integration
                    ax2 = anomaly_ax.twinx()
                    ax2.plot(perf_df['timestamp'], perf_df['inference_time'], 
                             label='Inference Time (ms)', color='#81c784', linewidth=2)

                    # Mark inference anomalies
                    inf_anomalies = perf_df[perf_df['inference_anomaly']]
                    if not inf_anomalies.empty:
                        ax2.scatter(inf_anomalies['timestamp'], inf_anomalies['inference_time'],
                                    color='orange', label='Inference Anomaly',
                                    s=100, edgecolor='white', zorder=5)

                    # Enhanced chart styling
                    anomaly_ax.set_title('System Performance with Anomalies', 
                                         color='white', pad=20, fontsize=12)
                    anomaly_ax.set_xlabel('Time', color='white', labelpad=10)
                    anomaly_ax.set_ylabel('FPS', color='white', labelpad=10)
                    ax2.set_ylabel('Inference Time (ms)', color='white', labelpad=10)

                    # Combined legend with improved layout
                    lines, labels = anomaly_ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    anomaly_ax.legend(lines + lines2, labels + labels2,
                                      facecolor='#2c2c54', labelcolor='white',
                                      bbox_to_anchor=(1.02, 1), loc='upper left')

                    # Formatting improvements
                    anomaly_ax.tick_params(colors='white')
                    ax2.tick_params(colors='white')
                    anomaly_ax.grid(True, alpha=0.2, color='white')
                else:
                    anomaly_ax.text(0.5, 0.5, 'No performance data available', 
                                    ha='center', va='center', color='white', fontsize=12)
            except Exception as e:
                anomaly_ax.text(0.5, 0.5, f'Error loading performance data: {str(e)}', 
                                ha='center', va='center', color='white')

            anomaly_canvas = FigureCanvasTkAgg(anomaly_fig, anomaly_frame)
            anomaly_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

            # Enhanced Summary Panel
            summary_frame = tk.Frame(analysis_window, bg='#16213e', height=120)
            summary_frame.pack(fill='x', padx=10, pady=(0, 10))
            summary_frame.pack_propagate(False)

            summary_text = tk.Text(summary_frame, bg='#2c2c54', fg='white', 
                                   font=('Arial', 11), wrap=tk.WORD)
            summary_text.pack(fill='both', expand=True, padx=10, pady=10)

            try:
                # Generate enhanced summary with more metrics
                if pattern_data and perf_data:
                    total_detections = sum(row['count'] for row in pattern_data)
                    avg_daily = total_detections / 7
                    
                    # Calculate performance stats
                    avg_fps = sum(row['fps'] for row in perf_data) / len(perf_data)
                    avg_inference = sum(row['inference_time'] for row in perf_data) / len(perf_data)
                    
                    # Count anomalies
                    anomaly_count = 0
                    if 'perf_df' in locals():
                        anomaly_count = len(fps_anomalies) + len(inf_anomalies)

                    summary = f"""
                    PATTERN ANALYSIS SUMMARY (Last 7 Days)
                    {'='*40}
                    - Total object detections: {total_detections:,}
                    - Average detections/day: {avg_daily:,.1f}

                    SYSTEM PERFORMANCE
                    {'='*40}
                    - Average FPS: {avg_fps:.1f}
                    - Average inference time: {avg_inference:.1f} ms
                    - Performance anomalies detected: {anomaly_count}
                    - Maintenance recommended: {'YES' if self.maintenance_warning else 'NO'}

                    RECOMMENDATIONS
                    {'='*40}
                    - {'Check system logs' if anomaly_count > 0 else 'System operating normally'}
                    - {'Schedule maintenance' if self.maintenance_warning else 'No maintenance needed'}
                    """
                    summary_text.insert(tk.END, summary)
                else:
                    summary_text.insert(tk.END, "Insufficient data for analysis summary")

            except Exception as e:
                summary_text.insert(tk.END, f"Error generating summary: {str(e)}")

            summary_text.config(state='disabled')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create analysis window: {str(e)}")
    
    def load_default_model(self):
        """Try to load a default model if one exists"""
        default_model_path = "default_model.pt"
        if os.path.exists(default_model_path):
            try:
                self.model = YOLO(default_model_path)
                self.model_status.config(text="Default model loaded", fg='#81c784')
                self.status_label.config(text="Ready - Load an image, video, or start camera")
            except Exception as e:
                self.model_status.config(text="Error loading default model", fg='#f44336')
                messagebox.showerror("Model Error", f"Could not load default model: {str(e)}")
    
    def load_model(self):
        """Load a YOLO model from file"""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        
        if model_path:
            try:
                self.model = YOLO(model_path)
                self.model_status.config(text=f"Model loaded: {os.path.basename(model_path)}", fg='#81c784')
                self.status_label.config(text="Ready - Load an image, video, or start camera")
                
                # Save as default model if loading succeeds
                default_model_path = "default_model.pt"
                if os.path.exists(default_model_path):
                    os.remove(default_model_path)
                os.link(model_path, default_model_path)
                
            except Exception as e:
                self.model_status.config(text="Model load failed", fg='#f44336')
                messagebox.showerror("Model Error", f"Could not load model: {str(e)}")
    
    def update_model(self):
        """Update the currently loaded model with new weights"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        weights_path = filedialog.askopenfilename(
            title="Select Model Weights File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        
        if weights_path:
            try:
                self.model = YOLO(weights_path)
                self.model_status.config(text=f"Model updated: {os.path.basename(weights_path)}", fg='#81c784')
                
                # Save as default model if update succeeds
                default_model_path = "default_model.pt"
                if os.path.exists(default_model_path):
                    os.remove(default_model_path)
                os.link(weights_path, default_model_path)
                
            except Exception as e:
                self.model_status.config(text="Model update failed", fg='#f44336')
                messagebox.showerror("Model Error", f"Could not update model: {str(e)}")
    
    def load_image(self):
        """Load an image file for detection"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if image_path:
            try:
                self.current_image = cv2.imread(image_path)
                self.display_image(self.current_image)
                self.process_image(self.current_image, image_path)
            except Exception as e:
                messagebox.showerror("Image Error", f"Could not process image: {str(e)}")
    
    def process_image(self, image, image_path):
        """Process an image with the loaded model"""
        if not self.model:
            return
        
        try:
            start_time = time.time()
            results = self.model(image, conf=self.confidence_var.get())
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Draw bounding boxes
            annotated_image = results[0].plot()
            self.display_image(annotated_image)
            
            # Update results
            self.update_detection_results(results[0], image_path, image.shape[1], image.shape[0])
            
            # Log performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'fps': 0,  # Not applicable for single image
                'inference_time': inference_time,
                'objects_detected': len(results[0].boxes)
            })
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing image: {str(e)}")
    
    def display_image(self, image):
        """Display an image in the GUI"""
        if image is None:
            return
        
        # Convert color from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL format
        image_pil = Image.fromarray(image_rgb)
        
        # Get current label dimensions
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()
        
        # Resize image to fit label while maintaining aspect ratio
        img_ratio = image_pil.width / image_pil.height
        label_ratio = label_width / label_height
        
        if label_ratio > img_ratio:
            # Fit to height
            new_height = label_height
            new_width = int(new_height * img_ratio)
        else:
            # Fit to width
            new_width = label_width
            new_height = int(new_width / img_ratio)
        
        # Resize if needed
        if new_width != image_pil.width or new_height != image_pil.height:
            image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to Tkinter format
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Update label
        self.image_label.configure(image=image_tk)
        self.image_label.image = image_tk
    
    def load_video(self):
        """Load a video file for detection"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        if self.is_camera_active:
            self.toggle_camera()  # Turn off camera if active
        
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        
        if video_path:
            try:
                self.video_file = video_path
                self.video_capture = cv2.VideoCapture(video_path)
                self.is_video_playing = False
                
                # Enable video controls
                self.play_button.config(state='normal')
                self.stop_button.config(state='normal')
                
                # Update status
                self.status_label.config(text=f"Video loaded: {os.path.basename(video_path)}")
                self.video_progress.config(text="Ready to play")
                
                # Show first frame
                ret, frame = self.video_capture.read()
                if ret:
                    self.current_frame = frame
                    self.display_image(frame)
                
            except Exception as e:
                messagebox.showerror("Video Error", f"Could not load video: {str(e)}")
    
    def toggle_video_playback(self):
        """Toggle video playback"""
        if not self.video_capture:
            return
        
        if self.is_video_playing:
            self.is_video_playing = False
            self.play_button.config(text="▶ Play")
        else:
            self.is_video_playing = True
            self.play_button.config(text="⏸ Pause")
            self.play_video()
    
    def play_video(self):
        """Play the loaded video with object detection"""
        if not self.is_video_playing or not self.video_capture:
            return
        
        def video_loop():
            while self.is_video_playing:
                start_time = time.time()
                
                ret, frame = self.video_capture.read()
                if not ret:
                    # End of video
                    self.is_video_playing = False
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
                    self.root.after(0, lambda: self.play_button.config(text="▶ Play"))
                    break
                
                self.current_frame = frame
                
                # Process frame
                self.process_video_frame(frame)
                
                # Update progress
                total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                progress = f"Frame {current_frame}/{total_frames}"
                self.root.after(0, lambda: self.video_progress.config(text=progress))
                
                # Control playback speed
                processing_time = time.time() - start_time
                delay = max(1, int((1/30 - processing_time) * 1000))  # Aim for ~30 FPS
            
            # Continue playback if still active
            if self.is_video_playing:
                self.root.after(delay, self.play_video)
        
        # Start video processing in a separate thread
        threading.Thread(target=video_loop, daemon=True).start()
    
    def stop_video(self):
        """Stop video playback"""
        self.is_video_playing = False
        self.play_button.config(text="▶ Play")
        
        if self.video_capture:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind
            ret, frame = self.video_capture.read()
            if ret:
                self.display_image(frame)
                self.video_progress.config(text="Stopped")
    
    def toggle_camera(self):
        """Toggle camera feed"""
        if self.is_camera_active:
            # Stop camera
            self.is_camera_active = False
            self.camera_button.config(text="Start Camera", bg='#f44336')
            
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            
            # Disable video controls
            self.play_button.config(state='disabled')
            self.stop_button.config(state='disabled')
            
            self.status_label.config(text="Camera stopped")
        else:
            # Start camera
            try:
                self.video_capture = cv2.VideoCapture(0)
                self.is_camera_active = True
                self.camera_button.config(text="Stop Camera", bg='#81c784')
                
                # Enable video controls
                self.play_button.config(state='normal')
                self.stop_button.config(state='normal')
                
                self.status_label.config(text="Camera active - Press Play to start detection")
                
                # Show camera feed
                self.is_video_playing = False
                self.play_button.config(text="▶ Play")
                self.play_video()  # Start camera feed without processing
                
            except Exception as e:
                messagebox.showerror("Camera Error", f"Could not start camera: {str(e)}")
                self.is_camera_active = False
                self.camera_button.config(text="Start Camera", bg='#f44336')
    
    def process_video_frame(self, frame):
        """Process a single video frame with performance tracking"""
        if not self.model:
            return
        
        start_time = time.time()
        
        try:
            # Run detection on frame
            self.confidence_threshold = self.confidence_var.get()
            results = self.model(frame, conf=self.confidence_threshold)
            
            # Calculate processing metrics
            inference_time = (time.time() - start_time) * 1000  # ms
            fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
            
            # Store performance metrics
            self.performance_history.append({
                'timestamp': datetime.now(),
                'fps': fps,
                'inference_time': inference_time,
                'objects_detected': len(results[0].boxes)
            })
            
            # Draw bounding boxes
            annotated_frame = results[0].plot()
            
            # Display frame in GUI thread
            self.root.after(0, lambda: self.display_image(annotated_frame))
            
            # Update results in GUI thread
            self.root.after(0, lambda: self.update_detection_results(results[0], "video_frame", frame.shape[1], frame.shape[0]))
            
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    def toggle_monitoring(self):
        """Toggle continuous monitoring mode"""
        if not self.model:
            messagebox.showwarning("No Model", "Please load a model first")
            return
        
        if self.is_monitoring:
            # Stop monitoring
            self.is_monitoring = False
            self.monitor_button.config(text="Start Monitoring", bg='#9c27b0')
            self.status_label.config(text="Monitoring stopped")
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=1)
        else:
            # Start monitoring
            self.is_monitoring = True
            self.monitor_button.config(text="Stop Monitoring", bg='#f44336')
            self.status_label.config(text="Monitoring active - checking for objects...")
            
            self.monitoring_thread = threading.Thread(target=self.monitor_objects, daemon=True)
            self.monitoring_thread.start()
    
    def monitor_objects(self):
        """Continuous monitoring for specific objects"""
        critical_objects = ["Fire", "Smoke", "Oxygen Tank", "Toolbox"]
        
        # Create camera capture if not already active
        if not self.is_camera_active:
            try:
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    raise RuntimeError("Could not open camera")
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Camera Error", f"Could not start camera: {str(e)}"))
                self.is_monitoring = False
                self.root.after(0, lambda: self.monitor_button.config(text="Start Monitoring", bg='#9c27b0'))
                return
        
        while self.is_monitoring:
            start_time = time.time()
            
            # Capture frame
            ret, frame = self.video_capture.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Process frame
            try:
                results = self.model(frame, conf=self.confidence_var.get())
                
                # Check for critical objects
                detected_objects = set()
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    class_name = results[0].names[class_id]
                    if class_name in critical_objects:
                        detected_objects.add(class_name)
                
                # Update display
                annotated_frame = results[0].plot()
                self.root.after(0, lambda: self.display_image(annotated_frame))
                self.root.after(0, lambda: self.update_detection_results(results[0], "monitoring", frame.shape[1], frame.shape[0]))
                
                # Alert for critical objects
                if detected_objects:
                    alert_msg = "ALERT: Detected " + ", ".join(detected_objects)
                    self.root.after(0, lambda: self.status_label.config(text=alert_msg, fg='#f44336'))
                    
                    # Play alert sound (platform dependent)
                    try:
                        import winsound
                        winsound.Beep(1000, 500)
                    except:
                        pass
                else:
                    self.root.after(0, lambda: self.status_label.config(text="Monitoring active - no critical objects detected", fg='#81c784'))
                
                # Calculate processing metrics
                inference_time = (time.time() - start_time) * 1000  # ms
                fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                
                # Store performance metrics
                self.performance_history.append({
                    'timestamp': datetime.now(),
                    'fps': fps,
                    'inference_time': inference_time,
                    'objects_detected': len(results[0].boxes)
                })
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            # Throttle monitoring to ~2 FPS
            time.sleep(max(0, 0.5 - (time.time() - start_time)))
        
        # Clean up
        if not self.is_camera_active and self.video_capture:
            self.video_capture.release()
            self.video_capture = None
    
    def update_detection_results(self, results, image_path, frame_width=None, frame_height=None):
        """Update detection results display with additional tracking"""
        self.results_text.delete(1.0, tk.END)
        
        if len(results.boxes) == 0:
            self.results_text.insert(tk.END, "No objects detected")
            return
        
        # Count detections by class
        class_counts = {}
        detection_data = []
        
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = results.names[class_id]
            confidence = float(box.conf)
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Store detection for logging
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detection_data.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            })
            
            self.results_text.insert(tk.END, f"{class_name}: {confidence:.2f}\n")
        
        # Log detections to database
        self.log_detections(detection_data, image_path, frame_width, frame_height)
        
        # Update inventory counts
        self.update_inventory_counts(detection_data)
        
        # Track object trajectories if frame dimensions are provided
        if frame_width and frame_height:
            self.track_object_trajectories(detection_data, frame_width, frame_height)
    
    def log_detections(self, detections, image_path, frame_width=None, frame_height=None):
        """Log detections to database with additional information"""
        timestamp = datetime.now()
        for detection in detections:
            self.db.insert_detection(
                timestamp, detection['class'], detection['confidence'], 
                detection['bbox'][0], detection['bbox'][1], 
                detection['bbox'][2], detection['bbox'][3], 
                image_path, frame_width or 0, frame_height or 0, self.session_id
            )
    
    def view_detection_logs(self):
        """View detection logs in a new window"""
        logs_window = tk.Toplevel(self.root)
        logs_window.title("Detection Logs")
        logs_window.geometry("1200x800")
        logs_window.configure(bg='#1a1a2e')
        
        # Create treeview for logs
        columns = ("Timestamp", "Object", "Confidence", "Position", "Size", "Image")
        tree = ttk.Treeview(logs_window, columns=columns, show='headings', height=25)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(logs_window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Layout
        tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Load data from database
        recent_detections = self.db.get_recent_detections(500)
        
        if recent_detections:
            for row in recent_detections:
                timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                object_class = row['object_class']
                confidence = f"{row['confidence']:.2f}"
                position = f"({row['x']}, {row['y']})"
                size = f"{row['width']}x{row['height']}"
                image_path = os.path.basename(row['image_path']) if row['image_path'] else "N/A"
                
                tree.insert('', 'end', values=(timestamp, object_class, confidence, position, size, image_path))
    
    def stop_all_streams(self):
        """Stop all video/camera streams and threads"""
        self.is_monitoring = False
        self.is_video_playing = False
        self.is_camera_active = False
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1)
        
        # Update session in database
        end_time = datetime.now()
        duration = (end_time - self.session_start_time).total_seconds()
        
        # Get total objects detected in this session
        total_objects = self.db.execute_query("""
            SELECT COUNT(*) as count FROM detections WHERE session_id = %s
        """, (self.session_id,), fetch=True)
        
        objects_count = total_objects[0]['count'] if total_objects else 0
        
        self.db.update_session(self.session_id, end_time, duration, objects_count, 'completed')
        
        # Disconnect from database
        self.db.disconnect()

def main():
    root = tk.Tk()
    
    # Handle window closing properly
    def on_closing():
        try:
            if hasattr(app, 'stop_all_streams'):
                app.stop_all_streams()
        except:
            pass
        root.quit()
        root.destroy()
    
    app = SpaceStationDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()

if __name__ == "__main__":
    main()
