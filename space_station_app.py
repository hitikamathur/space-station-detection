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
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sqlite3

class SpaceStationDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Space Station Object Detection System v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # Initialize variables
        self.model = None
        self.current_image = None
        self.detection_results = []
        self.confidence_threshold = 0.5
        self.is_monitoring = False
        self.monitoring_thread = None
        self.database_file = "detection_logs.db"
        
        # Video/Camera variables
        self.video_capture = None
        self.is_camera_active = False
        self.is_video_playing = False
        self.video_thread = None
        self.current_frame = None
        
        # Setup database
        self.setup_database()
        
        # Create GUI
        self.create_gui()
        
        # Load default model if exists
        self.load_default_model()
    
    def setup_database(self):
        """Initialize SQLite database for logging detections"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                object_class TEXT,
                confidence REAL,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                image_path TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
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
        
        title_label = tk.Label(header_frame, text="🚀 SPACE STATION OBJECT DETECTION SYSTEM", 
                              font=('Arial', 18, 'bold'), fg='#4fc3f7', bg='#16213e')
        title_label.pack(pady=10)
        
        mission_label = tk.Label(header_frame, text="AI-Powered Safety & Inventory Management", 
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
        
        # Video progress
        self.video_progress = tk.Label(video_controls_frame, text="No video loaded", 
                                      fg='white', bg='#16213e', font=('Arial', 10))
        self.video_progress.pack(side='right', padx=5, pady=5)
        
        self.image_label = tk.Label(left_frame, text="Load an image, video, or start camera to begin detection",
                                   bg='#2c2c54', fg='white', font=('Arial', 14))
        self.image_label.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Right side - Results and analytics
        right_frame = tk.Frame(main_frame, bg='#16213e', width=350)
        right_frame.pack(side='right', fill='y', padx=5, pady=5)
        right_frame.pack_propagate(False)
        
        # Detection results
        results_frame = tk.LabelFrame(right_frame, text="Detection Results", 
                                     fg='#4fc3f7', bg='#16213e', font=('Arial', 12, 'bold'))
        results_frame.pack(fill='x', pady=5)
        
        self.results_text = tk.Text(results_frame, height=8, bg='#2c2c54', fg='white',
                                   font=('Courier', 10))
        self.results_text.pack(fill='x', padx=5, pady=5)
        
        # Analytics frame
        analytics_frame = tk.LabelFrame(right_frame, text="Analytics", 
                                       fg='#4fc3f7', bg='#16213e', font=('Arial', 12, 'bold'))
        analytics_frame.pack(fill='both', expand=True, pady=5)
        
        # Create matplotlib figure for analytics
        self.fig, self.ax = plt.subplots(figsize=(4, 3), facecolor='#16213e')
        self.ax.set_facecolor('#2c2c54')
        self.canvas = FigureCanvasTkAgg(self.fig, analytics_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
    
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
    
    def load_default_model(self):
        """Try to load model.pt if it exists"""
        if os.path.exists("model.pt"):
            try:
                self.model = YOLO("model.pt")
                self.model_status.config(text="✓ Model loaded: model.pt", fg='#81c784')
                self.status_label.config(text="Model ready - Load image or start camera")
            except Exception as e:
                self.model_status.config(text=f"Error loading model: {str(e)}", fg='#f44336')
    
    def load_model(self):
        """Load YOLO model from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("PyTorch models", "*.pt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_status.config(text=f"✓ Model loaded: {os.path.basename(file_path)}", fg='#81c784')
                self.status_label.config(text="Model ready - Load image or start camera")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.model_status.config(text="Model load failed", fg='#f44336')
    
    def update_model(self):
        """Simulate model update using Falcon (placeholder for real implementation)"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        # Simulate update process
        update_window = tk.Toplevel(self.root)
        update_window.title("Model Update - Falcon Integration")
        update_window.geometry("400x300")
        update_window.configure(bg='#1a1a2e')
        
        tk.Label(update_window, text="🔄 Updating Model with Falcon", 
                font=('Arial', 14, 'bold'), fg='#4fc3f7', bg='#1a1a2e').pack(pady=20)
        
        progress = ttk.Progressbar(update_window, length=300, mode='indeterminate')
        progress.pack(pady=10)
        progress.start()
        
        status_text = tk.Text(update_window, height=10, width=45, bg='#2c2c54', fg='white')
        status_text.pack(pady=10)
        
        # Simulate update steps
        steps = [
            "Connecting to Falcon platform...",
            "Downloading synthetic data updates...",
            "Analyzing performance gaps...",
            "Generating additional training samples...",
            "Fine-tuning model weights...",
            "Validating updated model...",
            "Model update completed successfully!"
        ]
        
        def simulate_update():
            for i, step in enumerate(steps):
                status_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {step}\n")
                status_text.see(tk.END)
                update_window.update()
                time.sleep(1)
            
            progress.stop()
            tk.Button(update_window, text="Close", command=update_window.destroy,
                     bg='#81c784', fg='black').pack(pady=10)
        
        threading.Thread(target=simulate_update, daemon=True).start()
    
    def load_image(self):
        """Load and process image"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.process_image(file_path)
    
    def process_image(self, image_path):
        """Process image and display results"""
        try:
            # Load and resize image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run detection
            self.confidence_threshold = self.confidence_var.get()
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # Draw bounding boxes
            annotated_image = results[0].plot()
            
            # Display image
            self.display_image(annotated_image)
            
            # Update results
            self.update_detection_results(results[0], image_path)
            
            self.status_label.config(text=f"Detection completed - Found {len(results[0].boxes)} objects")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {str(e)}")
    
    def display_image(self, image):
        """Display image in the GUI"""
        # Resize image to fit display
        height, width = image.shape[:2]
        max_width, max_height = 800, 600
        
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to PIL and display
        image_pil = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image_pil)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def update_detection_results(self, results, image_path):
        """Update detection results display"""
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
        self.log_detections(detection_data, image_path)
        
        # Update analytics
        self.update_analytics(class_counts)
    
    def log_detections(self, detections, image_path):
        """Log detections to database"""
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        for detection in detections:
            cursor.execute('''
                INSERT INTO detections (timestamp, object_class, confidence, x, y, width, height, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, detection['class'], detection['confidence'], 
                  detection['bbox'][0], detection['bbox'][1], detection['bbox'][2], detection['bbox'][3], image_path))
        
        conn.commit()
        conn.close()
    
    def update_analytics(self, class_counts):
        """Update analytics visualization"""
        self.ax.clear()
        
        if class_counts:
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            colors = ['#4fc3f7', '#81c784', '#ff9800', '#f44336', '#9c27b0']
            self.ax.bar(classes, counts, color=colors[:len(classes)])
            self.ax.set_title('Object Detection Counts', color='white', fontsize=12)
            self.ax.set_xlabel('Object Class', color='white')
            self.ax.set_ylabel('Count', color='white')
            self.ax.tick_params(colors='white')
            
            # Rotate x-axis labels for better readability
            plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        else:
            self.ax.text(0.5, 0.5, 'No detections', ha='center', va='center', 
                        transform=self.ax.transAxes, color='white', fontsize=14)
        
        self.ax.set_facecolor('#2c2c54')
        self.fig.patch.set_facecolor('#16213e')
        self.canvas.draw()
    
    def load_video(self):
        """Load video file for processing"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if file_path:
            self.stop_all_streams()
            try:
                self.video_capture = cv2.VideoCapture(file_path)
                if not self.video_capture.isOpened():
                    messagebox.showerror("Error", "Failed to open video file")
                    return
                
                # Get video info
                fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                self.video_progress.config(text=f"Video loaded: {duration:.1f}s, {frame_count} frames")
                self.play_button.config(state='normal')
                self.stop_button.config(state='normal')
                self.status_label.config(text="Video loaded - Click Play to start detection")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if not self.is_camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera feed"""
        self.stop_all_streams()
        try:
            self.video_capture = cv2.VideoCapture(0)  # Default camera
            if not self.video_capture.isOpened():
                # Try other camera indices
                for i in range(1, 4):
                    self.video_capture = cv2.VideoCapture(i)
                    if self.video_capture.isOpened():
                        break
                else:
                    messagebox.showerror("Error", "No camera found. Please connect a camera.")
                    return
            
            self.is_camera_active = True
            self.camera_button.config(text="Stop Camera", bg='#4caf50')
            self.video_progress.config(text="Camera active - Live feed")
            self.status_label.config(text="Camera active - Real-time detection")
            
            # Start camera thread
            self.video_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop camera feed"""
        self.is_camera_active = False
        self.camera_button.config(text="Start Camera", bg='#f44336')
        self.video_progress.config(text="Camera stopped")
        self.status_label.config(text="Camera stopped")
        self.stop_all_streams()
    
    def toggle_video_playback(self):
        """Toggle video play/pause"""
        if not self.video_capture:
            return
        
        if not self.is_video_playing:
            self.is_video_playing = True
            self.play_button.config(text="⏸ Pause")
            self.status_label.config(text="Video playing - Processing frames")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
        else:
            self.is_video_playing = False
            self.play_button.config(text="▶ Play")
            self.status_label.config(text="Video paused")
    
    def stop_video(self):
        """Stop video playback and reset"""
        self.is_video_playing = False
        self.play_button.config(text="▶ Play", state='disabled')
        self.stop_button.config(state='disabled')
        self.video_progress.config(text="No video loaded")
        self.status_label.config(text="Video stopped")
        self.stop_all_streams()
    
    def stop_all_streams(self):
        """Stop all video streams and cleanup"""
        self.is_camera_active = False
        self.is_video_playing = False
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # Wait for threads to finish
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=1.0)
    
    def camera_loop(self):
        """Main loop for camera processing"""
        while self.is_camera_active and self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                break
            
            try:
                # Process frame
                self.process_video_frame(frame)
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                print(f"Camera processing error: {e}")
                break
        
        self.cleanup_camera()
    
    def video_loop(self):
        """Main loop for video file processing"""
        while self.is_video_playing and self.video_capture:
            ret, frame = self.video_capture.read()
            if not ret:
                # End of video
                self.root.after(0, self.stop_video)
                break
            
            try:
                # Process frame
                self.process_video_frame(frame)
                
                # Update progress
                current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
                progress = (current_frame / total_frames) * 100 if total_frames > 0 else 0
                self.root.after(0, lambda: self.video_progress.config(
                    text=f"Progress: {progress:.1f}% ({int(current_frame)}/{int(total_frames)})"))
                
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                print(f"Video processing error: {e}")
                break
    
    def process_video_frame(self, frame):
        """Process a single video frame"""
        if not self.model:
            return
        
        try:
            # Run detection on frame
            self.confidence_threshold = self.confidence_var.get()
            results = self.model(frame, conf=self.confidence_threshold)
            
            # Draw bounding boxes
            annotated_frame = results[0].plot()
            
            # Display frame in GUI thread
            self.root.after(0, lambda: self.display_image(annotated_frame))
            
            # Update results in GUI thread
            self.root.after(0, lambda: self.update_detection_results(results[0], "video_frame"))
            
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    def cleanup_camera(self):
        """Cleanup camera resources"""
        self.root.after(0, lambda: self.camera_button.config(text="Start Camera", bg='#f44336'))
        self.root.after(0, lambda: self.video_progress.config(text="Camera disconnected"))
        self.root.after(0, lambda: self.status_label.config(text="Camera stopped"))
    
    def toggle_monitoring(self):
        """Toggle continuous monitoring mode"""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_button.config(text="Stop Monitoring", bg='#f44336')
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.status_label.config(text="Monitoring active - Scanning for objects...")
        else:
            self.is_monitoring = False
            self.monitor_button.config(text="Start Monitoring", bg='#9c27b0')
            self.status_label.config(text="Monitoring stopped")
    
    def monitoring_loop(self):
        """Continuous monitoring loop (placeholder for real implementation)"""
        while self.is_monitoring:
            # In real implementation, this would capture from camera or monitor folder
            time.sleep(5)  # Simulate monitoring interval
            if self.is_monitoring:  # Check again in case it was stopped
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Monitoring... Last scan: {datetime.now().strftime('%H:%M:%S')}"))
    
    def view_detection_logs(self):
        """View detection logs in a new window"""
        logs_window = tk.Toplevel(self.root)
        logs_window.title("Detection Logs")
        logs_window.geometry("800x600")
        logs_window.configure(bg='#1a1a2e')
        
        # Create treeview for logs
        columns = ('Timestamp', 'Class', 'Confidence', 'Position')
        tree = ttk.Treeview(logs_window, columns=columns, show='headings', height=20)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Load data from database
        conn = sqlite3.connect(self.database_file)
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, object_class, confidence, x, y FROM detections ORDER BY timestamp DESC LIMIT 100')
        
        for row in cursor.fetchall():
            timestamp = datetime.fromisoformat(row[0]).strftime('%Y-%m-%d %H:%M:%S')
            tree.insert('', 'end', values=(timestamp, row[1], f"{row[2]:.2f}", f"({row[3]}, {row[4]})"))
        
        conn.close()
        
        tree.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add export button
        tk.Button(logs_window, text="Export to JSON", 
                 command=lambda: self.export_logs(),
                 bg='#4fc3f7', fg='black').pack(pady=10)
    
    def export_logs(self):
        """Export detection logs to JSON"""
        try:
            conn = sqlite3.connect(self.database_file)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM detections ORDER BY timestamp DESC')
            
            logs = []
            for row in cursor.fetchall():
                logs.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'object_class': row[2],
                    'confidence': row[3],
                    'x': row[4],
                    'y': row[5],
                    'width': row[6],
                    'height': row[7],
                    'image_path': row[8]
                })
            
            conn.close()
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(logs, f, indent=2)
                messagebox.showinfo("Export", f"Logs exported to {file_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {str(e)}")

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