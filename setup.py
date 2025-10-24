#!/usr/bin/env python3
"""
Setup script for Organoid Analyzer & TIL Analyzer
This script helps set up the environment and verify installation.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available. CPU-only mode will be used.")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. CUDA check will be done after installation.")
        return None

def create_directories():
    """Create necessary directories."""
    directories = [
        "Data",
        "Generated",
        "Generated/models", 
        "Results",
        "TIL-Analyzer-main/images",
        "TIL-Analyzer-main/chip annotations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_fiji():
    """Check if Fiji is installed."""
    fiji_paths = [
        "/Applications/Fiji.app",  # macOS
        "C:\\Program Files\\Fiji.app",  # Windows
        "C:\\Fiji.app",  # Windows alternative
        "/usr/local/Fiji.app",  # Linux
        "/opt/Fiji.app"  # Linux alternative
    ]
    
    for path in fiji_paths:
        if os.path.exists(path):
            print(f"✅ Fiji found at: {path}")
            return path
    
    print("⚠️  Fiji not found. Please install Fiji from https://fiji.sc/")
    print("   Update FIJI_PATH in Organoid Analyzer/Config.py after installation.")
    return None

def install_requirements():
    """Install Python requirements."""
    print("📦 Installing Python requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed."""
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-learn"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("lifelines", "Lifelines"),
        ("umap", "UMAP"),
        ("shap", "SHAP")
    ]
    
    print("\n🔍 Verifying package installation...")
    all_good = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good

def create_sample_config():
    """Create sample configuration files."""
    # Sample Config.py for Organoid Analyzer
    config_content = '''# Sample configuration - update paths as needed
import os

# Data directories
DATA_DIR = "./Data"
GENERATED_DIR = "./Generated"
MODEL_DIR = os.path.join(GENERATED_DIR, "models")
RESULTS_DIR = "./Results"

# Fiji path - UPDATE THIS PATH
FIJI_PATH = r"/path/to/Fiji.app"  # Update this path
JAVA_ARGUMENTS = '-Xmx12g'

# Dataset configurations
DATASET_CONFIGS = {
    "CART": {"annotation_path" : f"{DATA_DIR}/CART annotations.xlsx", 
             "data_folder" : f"{DATA_DIR}/CART"},
    "2nd": {"annotation_path" : f"{DATA_DIR}/2nd batch annotations.xlsx",
            "data_folder" : f"{DATA_DIR}/2ND"},
    "PDO": {"annotation_path" : f"{DATA_DIR}/PDO_annotation.xlsx",
            "data_folder" : f"{DATA_DIR}/PDO"},
}

# Training settings
SEQ_LEN = 100
MAX_EPOCHS = 400
BATCH_SIZE = 256
DROPOUT = 0.3

# Features
features = ['AREA', 'PERIMETER', 'CIRCULARITY', 'ELLIPSE_ASPECTRATIO', 
           'SOLIDITY', 'SPEED', "MEAN_SQUARE_DISPLACEMENT", "RADIUS"]
track_features = ["TRACK_DISPLACEMENT", "TRACK_STD_SPEED", 
                 "MEAN_DIRECTIONAL_CHANGE_RATE"]
'''
    
    config_path = Path("Organoid Analyzer/Config_sample.py")
    config_path.write_text(config_content)
    print(f"✅ Created sample config: {config_path}")

def main():
    """Main setup function."""
    print("🚀 Setting up Organoid Analyzer & TIL Analyzer")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Check Fiji
    print("\n🔬 Checking Fiji installation...")
    fiji_path = check_fiji()
    
    # Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        print("❌ Installation failed. Please check error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Some packages failed to install. Please check error messages above.")
        sys.exit(1)
    
    # Check CUDA
    print("\n🎮 Checking CUDA...")
    cuda_available = check_cuda()
    
    # Create sample config
    print("\n⚙️  Creating sample configuration...")
    create_sample_config()
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update paths in 'Organoid Analyzer/Config_sample.py'")
    print("2. Place your data in the appropriate directories:")
    print("   - Organoid data: ./Data/")
    print("   - TIL images: ./TIL-Analyzer-main/images/")
    print("3. Create annotation files as described in README.md")
    print("4. Run the analysis scripts:")
    print("   - Organoid: python 'Organoid Analyzer/train_and_test_models.py'")
    print("   - TIL: python TIL-Analyzer-main/main_test.py")
    
    if not cuda_available:
        print("\n⚠️  Note: CUDA not available. Training will be slower on CPU.")
    
    if not fiji_path:
        print("\n⚠️  Note: Fiji not found. Install Fiji for cell tracking functionality.")

if __name__ == "__main__":
    main()
