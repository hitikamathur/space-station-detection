#!/usr/bin/env python3
"""
Space Station Object Detection System Launcher
==============================================

This script handles the setup and launch of the Space Station Detection Application.
It checks dependencies, validates the model file, and provides helpful error messages.

Usage: python launch_app.py
"""

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    print(f"✅ Python version: {sys.version.split()[0]} - Compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy',
        'torch': 'torch',
        'tkinter': 'tkinter (usually built-in)'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                import PIL
            else:
                importlib.import_module(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Not found")
            missing_packages.append(install_name)
    
    if missing_packages:
        print("\n📦 Missing packages detected. Installing...")
        for package in missing_packages:
            if package == 'tkinter (usually built-in)':
                print("⚠️  tkinter not found. Please install python3-tk on Linux or ensure tkinter is available.")
                continue
            
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}. Please install manually:")
                print(f"   pip install {package}")
                return False
    
    return True

def check_model_file():
    """Check if model.pt exists"""
    model_files = ['model.pt', 'best.pt', 'yolov8n.pt']
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Model file found: {model_file}")
            return True
    
    print("⚠️  No model file found (model.pt, best.pt, or yolov8n.pt)")
    print("   You can:")
    print("   1. Place your trained model.pt in this directory")
    print("   2. The app will prompt you to load a model when started")
    print("   3. Download a pre-trained model from Ultralytics")
    return True  # Not critical, app can handle this

def check_system_resources():
    """Check basic system resources"""
    try:
        import psutil
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 4:
            print(f"⚠️  Low RAM detected: {ram_gb:.1f}GB (8GB+ recommended)")
        else:
            print(f"✅ RAM: {ram_gb:.1f}GB - Sufficient")
        
        # Check if GPU is available
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✅ GPU detected: {gpu_name}")
            else:
                print("ℹ️  No GPU detected - CPU inference will be used")
        except:
            print("ℹ️  PyTorch not available for GPU check")
            
    except ImportError:
        print("ℹ️  psutil not available for system resource check")

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)"""
    if sys.platform == "win32":
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            path = os.path.join(desktop, "Space Station Detection.lnk")
            target = sys.executable
            wDir = os.getcwd()
            arguments = os.path.join(wDir, "launch_app.py")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.Targetpath = target
            shortcut.Arguments = arguments
            shortcut.WorkingDirectory = wDir
            shortcut.IconLocation = target
            shortcut.save()
            
            print("✅ Desktop shortcut created")
        except:
            pass  # Skip if can't create shortcut

def main():
    """Main launcher function"""
    print("🚀 Space Station Object Detection System Launcher")
    print("=" * 55)
    
    # System checks
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    print("\n📋 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please resolve the issues above.")
        input("Press Enter to exit...")
        return
    
    print("\n🔍 Checking model files...")
    check_model_file()
    
    print("\n💻 Checking system resources...")
    check_system_resources()
    
    # Try to create desktop shortcut
    create_desktop_shortcut()
    
    print("\n" + "=" * 55)
    print("🎯 All checks completed! Launching application...")
    print("=" * 55)
    
    # Launch the main application
    try:
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import and run the main application
        from space_station_app import main as app_main
        app_main()
        
    except ImportError as e:
        print(f"❌ Error importing application: {e}")
        print("   Make sure space_station_app.py is in the current directory")
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        print("\n\n👋 Application closed by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   Please check the error and try again")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
