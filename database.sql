-- Create the database
CREATE DATABASE IF NOT EXISTS SpaceStationDetection;
USE SpaceStationDetection;

-- Create schema_version table
CREATE TABLE IF NOT EXISTS schema_version (
    version INT PRIMARY KEY,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Create detections table
CREATE TABLE IF NOT EXISTS detections (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    object_class VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    x INT NOT NULL,
    y INT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    image_path VARCHAR(255),
    frame_width INT,
    frame_height INT,
    session_id VARCHAR(100)
);

-- Create performance_metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    fps DECIMAL(10,2),
    inference_time DECIMAL(10,2),
    accuracy DECIMAL(5,2),
    objects_detected INT,
    session_id VARCHAR(100)
);

-- Create inventory table
CREATE TABLE IF NOT EXISTS inventory (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    object_class VARCHAR(100) NOT NULL,
    count INT NOT NULL,
    avg_confidence DECIMAL(5,2),
    avg_x DECIMAL(10,2),
    avg_y DECIMAL(10,2),
    session_id VARCHAR(100)
);

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(100) PRIMARY KEY,
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    duration DECIMAL(10,2),
    objects_detected INT,
    status VARCHAR(50)
);

-- Create indexes
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_detections_class ON detections(object_class);
CREATE INDEX idx_performance_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_inventory_timestamp ON inventory(timestamp);

-- Set initial schema version
INSERT INTO schema_version (version) VALUES (1);