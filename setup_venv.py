#!/usr/bin/env python3
"""
Simple setup script for Organoid Analyzer & TIL Analyzer using venv
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

def create_virtual_environment():
    """Create virtual environment if it doesn't exist."""
    venv_path = Path("organoid-til-env")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("📦 Creating virtual environment...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", "organoid-til-env"])
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_pip_command():
    """Get the pip command for the virtual environment."""
    if platform.system() == "Windows":
        return str(Path("organoid-til-env/Scripts/pip"))
    else:
        return str(Path("organoid-til-env/bin/pip"))

def get_python_command():
    """Get the python command for the virtual environment."""
    if platform.system() == "Windows":
        return str(Path("organoid-til-env/Scripts/python"))
    else:
        return str(Path("organoid-til-env/bin/python"))

def install_pytorch():
    """Install PyTorch with appropriate backend for the platform."""
    pip_cmd = get_pip_command()
    system = platform.system()
    
    print(f"🔥 Installing PyTorch for {system}...")
    
    try:
        if system == "Darwin":  # macOS
            # Check if it's Apple Silicon
            import platform as pl
            if pl.machine() == "arm64":
                print("🍎 Detected Apple Silicon Mac - installing PyTorch with MPS support")
                subprocess.check_call([
                    pip_cmd, "install", "torch", "torchvision", "torchaudio"
                ])
                print("✅ PyTorch with MPS support installed")
            else:
                print("🍎 Detected Intel Mac - installing CPU-only PyTorch")
                subprocess.check_call([
                    pip_cmd, "install", "torch", "torchvision", "torchaudio"
                ])
                print("✅ PyTorch (CPU-only) installed")
        elif system == "Windows":
            print("🪟 Detected Windows - trying CUDA first...")
            try:
                # Try CUDA 11.8 first
                subprocess.check_call([
                    pip_cmd, "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ])
                print("✅ PyTorch with CUDA 11.8 installed")
            except subprocess.CalledProcessError:
                print("⚠️  CUDA 11.8 failed, trying CPU-only version...")
                subprocess.check_call([
                    pip_cmd, "install", "torch", "torchvision", "torchaudio"
                ])
                print("✅ PyTorch (CPU-only) installed")
        else:  # Linux
            print("🐧 Detected Linux - trying CUDA first...")
            try:
                # Try CUDA 11.8 first
                subprocess.check_call([
                    pip_cmd, "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu118"
                ])
                print("✅ PyTorch with CUDA 11.8 installed")
            except subprocess.CalledProcessError:
                print("⚠️  CUDA 11.8 failed, trying CPU-only version...")
                subprocess.check_call([
                    pip_cmd, "install", "torch", "torchvision", "torchaudio"
                ])
                print("✅ PyTorch (CPU-only) installed")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_requirements():
    """Install Python requirements."""
    pip_cmd = get_pip_command()
    
    print("📦 Installing other requirements...")
    try:
        subprocess.check_call([pip_cmd, "install", "--upgrade", "pip"])
        subprocess.check_call([pip_cmd, "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

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

def check_acceleration():
    """Check what acceleration is available."""
    python_cmd = get_python_command()
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            result = subprocess.run([
                python_cmd, "-c", 
                "import torch; "
                "print('MPS available:', torch.backends.mps.is_available()); "
                "print('MPS built:', torch.backends.mps.is_built()); "
                "print('CUDA available:', torch.cuda.is_available())"
            ], capture_output=True, text=True)
            
            print(result.stdout.strip())
            mps_available = "MPS available: True" in result.stdout
            cuda_available = "CUDA available: True" in result.stdout
            
            if mps_available:
                print("🍎 Apple Silicon GPU acceleration available!")
                return True
            elif cuda_available:
                print("🎮 CUDA acceleration available!")
                return True
            else:
                print("💻 CPU-only mode")
                return False
        else:  # Windows/Linux
            result = subprocess.run([
                python_cmd, "-c", 
                "import torch; print('CUDA available:', torch.cuda.is_available()); "
                "print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
            ], capture_output=True, text=True)
            
            print(result.stdout.strip())
            cuda_available = "CUDA available: True" in result.stdout
            
            if cuda_available:
                print("🎮 CUDA acceleration available!")
                return True
            else:
                print("💻 CPU-only mode")
                return False
                
    except Exception as e:
        print(f"⚠️  Could not check acceleration: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed."""
    python_cmd = get_python_command()
    
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
            result = subprocess.run([
                python_cmd, "-c", f"import {package}; print('✅ {name}')"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"❌ {name} - not installed")
                all_good = False
        except Exception:
            print(f"❌ {name} - not installed")
            all_good = False
    
    return all_good

def create_sample_config():
    """Create sample configuration files."""
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
    print("🚀 Setting up Organoid Analyzer & TIL Analyzer with venv")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install PyTorch
    if not install_pytorch():
        print("❌ PyTorch installation failed. Please check error messages above.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Requirements installation failed. Please check error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("❌ Some packages failed to install. Please check error messages above.")
        sys.exit(1)
    
    # Check acceleration
    print("\n🎮 Checking acceleration...")
    acceleration_available = check_acceleration()
    
    # Create sample config
    print("\n⚙️  Creating sample configuration...")
    create_sample_config()
    
    # Final instructions
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate your virtual environment:")
    if platform.system() == "Windows":
        print("   organoid-til-env\\Scripts\\activate")
    else:
        print("   source organoid-til-env/bin/activate")
    print("2. Update paths in 'Organoid Analyzer/Config_sample.py'")
    print("3. Place your data in the appropriate directories:")
    print("   - Organoid data: ./Data/")
    print("   - TIL images: ./TIL-Analyzer-main/images/")
    print("4. Create annotation files as described in README.md")
    print("5. Run the analysis scripts:")
    print("   - Organoid: python 'Organoid Analyzer/train_and_test_models.py'")
    print("   - TIL: python TIL-Analyzer-main/main_test.py")
    
    if not acceleration_available:
        print("\n⚠️  Note: No GPU acceleration available. Training will be slower on CPU.")
    
    print(f"\n💡 To activate the environment later, run:")
    if platform.system() == "Windows":
        print("   organoid-til-env\\Scripts\\activate")
    else:
        print("   source organoid-til-env/bin/activate")

if __name__ == "__main__":
    main()
