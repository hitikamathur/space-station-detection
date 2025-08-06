# Space Station Object Detection System

## Project Overview

A comprehensive AI-powered object detection system designed specifically for space station safety and inventory management. This application leverages YOLOv8's state-of-the-art computer vision capabilities to provide real-time monitoring, automated inventory tracking, and predictive maintenance analytics for space station operations.

## Demo Video

**Watch the system in action**: [Space Station Detection Demo](https://youtu.be/y9YVjFVRF4g)

## Tech Stack

**Core Technologies:**
- **Python 3.8+** - Main programming language
- **YOLOv8 (Ultralytics)** - Object detection model
- **OpenCV** - Computer vision processing
- **Tkinter** - GUI framework
- **MySQL** - Database management
- **Matplotlib** - Data visualization

**Key Libraries:**
- opencv-python - Image/video processing
- ultralytics - YOLO model implementation
- mysql-connector-python - Database connectivity
- scikit-learn - Machine learning utilities
- pandas - Data manipulation
- numpy - Numerical computing
- scipy - Scientific computing

## Features

### Core Detection Capabilities
- **Multi-Source Input Support**: Process static images, video files, and live camera feeds
- **Real-Time Object Detection**: YOLOv8-powered detection with sub-second response times
- **Configurable Confidence Thresholds**: Adjustable sensitivity from 0.1 to 0.9
- **Dynamic Model Management**: Load, update, and switch between YOLO models

### Advanced Analytics & Monitoring
- **Performance Tracking**: Real-time FPS monitoring and inference time analysis
- **Inventory Management**: Automated object counting and tracking with historical trends
- **Trajectory Analysis**: Monitor object movement patterns and spatial distribution
- **Predictive Maintenance**: AI-powered anomaly detection using Isolation Forest algorithms
- **Pattern Recognition**: Historical analysis with automated anomaly alerts

### Database Integration
- **MySQL Backend**: Comprehensive logging of all detection events and metrics
- **Session Management**: Track detection sessions with detailed metadata
- **Historical Analytics**: Query and analyze detection patterns over time
- **Data Export**: Export inventory and analytics data to CSV format
- **Performance Monitoring**: Database-backed performance metric tracking

### User Interface
- **Modern Dark Theme**: Professional space-station inspired interface
- **Multi-Tab Dashboard**: Organized views for results, performance, inventory, and trajectories
- **Real-Time Visualization**: Live charts and performance graphs
- **Video Controls**: Full playback controls for video analysis
- **Interactive Analytics**: Clickable charts and detailed data exploration

## Model Performance

Our YOLOv8 model has been optimized for space station environments with exceptional accuracy:

| Metric           | Score     | Description                                                              |
|------------------|-----------|--------------------------------------------------------------------------|
| **Precision**    | **93.9%** | Extremely low false positive rate - 94 out of 100 detections are correct |
| **Recall**       | **81.3%** | Strong detection rate - catches over 8 out of 10 actual objects          |
| **mAP@0.5**      | **87.7%** | Excellent overall detection accuracy at standard IoU threshold           |
| **mAP@0.5:0.95** | **77.7%** | High-quality localization under strict evaluation criteria               |

These metrics demonstrate production-ready performance suitable for critical space station monitoring applications.

## Installation & Setup

### 1. Install Dependencies
```bash
pip install opencv-python ultralytics matplotlib mysql-connector-python scikit-learn pandas numpy scipy
```

### 2. Database Setup

#### Install MySQL Server
```bash
# Windows: Download from https://dev.mysql.com/downloads/installer/
# Ubuntu/Debian:
sudo apt update && sudo apt install mysql-server

# macOS:
brew install mysql && brew services start mysql
```

#### Initialize Database Schema
1. Connect to MySQL:
   ```bash
   mysql -u root -p
   ```

2. Use the provided SQL file to create the database and tables:
   ```bash
   mysql -u root -p < database.sql
   ```

   This will automatically create the `SpaceStationDetection` database and all required tables with proper indexes.

#### Configure Database Connection
Update the database credentials in your Python code:
```python
# In DatabaseManager class
def __init__(self):
    self.host = 'localhost'
    self.database = 'SpaceStationDetection'
    self.user = 'root'
    self.password = 'YOUR_MYSQL_PASSWORD'  # Update this
```

#### Verify Setup
```bash
mysql -u root -p -e "USE SpaceStationDetection; SHOW TABLES;"
```

### 3. Run Application
```bash
python space_station_detection.py
```

### 4. Quick Start Guide
1. **Load Model** → **Choose Input Source** → **Start Detection**

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Sources │───▶│   YOLOv8 Model   │───▶│   Processing    │
│ • Camera Feed   │    │   • Detection    │    │ • Analytics     │
│ • Video Files   │    │   • Classification│    │ • Tracking      │
│ • Static Images │    │   • Localization │    │ • Visualization │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                  │
┌─────────────────┐    ┌──────────▼──────────┐    ┌─────────────────┐
│   MySQL DB      │◀───│   Data Management   │───▶│  GUI Interface │
│ • Detections    │    │ • Session Tracking  │    │ • Multi-tab View │
│ • Performance   │    │ • Inventory Mgmt    │    │ • Real-time Charts│
│ • Analytics     │    │ • Export Functions  │    │ • Controls      │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
```

## Use Cases

- **Safety Monitoring**: Real-time detection of unauthorized objects or safety hazards
- **Inventory Management**: Automated tracking and counting of tools and equipment
- **Maintenance Scheduling**: Predictive analytics for equipment maintenance needs
- **Operational Efficiency**: Pattern analysis for optimizing space station workflows
- **Anomaly Detection**: Early warning system for unusual activities or objects

## System Requirements

- **OS**: Windows 10+, macOS 10.14+, or Linux Ubuntu 18+
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 2GB free space
- **GPU**: Optional but recommended for optimal performance
- **Database**: MySQL 5.7+ or MariaDB 10.3+

## Key Differentiators

- **Emergency Response Ready**: Real-time object tracking and trajectory analysis enables rapid location of critical safety equipment (fire extinguishers, oxygen masks, emergency tools) during emergencies like fires or system failures
- **Space-Optimized**: Designed specifically for space station environments
- **Production Ready**: High accuracy metrics (93.9% precision) suitable for critical operations
- **Comprehensive Analytics**: Beyond detection - full inventory and performance management
- **Scalable Architecture**: MySQL backend supports enterprise-scale deployments
- **Real-Time Processing**: Sub-second detection with live performance monitoring

---
Built for space station operations with enterprise-grade reliability and accuracy.
