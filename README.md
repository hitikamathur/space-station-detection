# 🚀 Space Station Object Detection System

A smart, AI-powered object detection system designed for space station environments. This desktop app uses real-time computer vision and YOLOv8 to detect and identify objects, assisting astronauts or mission operators in maintaining safety and automation.

---

## 🧠 Features

- 🔍 **YOLOv8-Based Object Detection**
- 🎥 Real-Time Camera and Image Input
- 📊 Auto-Logging and Detection Visualization
- 🚀 Space-Themed Desktop Interface (Windows batch launcher)
- 🔁 Seamless Startup with `run_app.bat`
- 🔧 Built-in Installer for Requirements (`install_requirements.bat`)
- 💾 Torch-based model loading (`model.pt` file)

---

## 🖥️ How to Run (Windows)

1. ✅ Ensure **Python 3.8+** is installed and added to PATH
2. 📥 Clone or download this repository
3. 🧠 Place your trained `model.pt` file in the project root
4. 💽 Run `install_requirements.bat` to install dependencies
5. 🚀 Launch using `run_app.bat`

---

## 📦 Requirements

Installed automatically via `install_requirements.bat`:

```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=10.0.0
matplotlib>=3.7.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
psutil>=5.9.0
