#!/bin/bash
# Student Auto-Setup Script for ML4ME Textbook
# This script sets up everything needed to run the machine learning examples

set -e  # Exit on any error

echo "🚀 Setting up ML4ME Textbook Environment..."
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniforge first:"
    echo "   https://github.com/conda-forge/miniforge"
    echo "   Then restart your terminal and run this script again."
    exit 1
fi

echo "✅ Conda found"

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux-musl"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    OS="windows"
else
    # Fallback: try to detect from uname
    case "$(uname -s)" in
        Linux*)     OS="linux";;
        Darwin*)    OS="macos";;
        CYGWIN*)    OS="windows";;
        MINGW*)     OS="windows";;
        *)          OS="unknown";;
    esac
fi

echo "🖥️  Detected OS: $OS"

# Check for NVIDIA GPU on Linux/Windows
HAS_NVIDIA_GPU=false
if [[ "$OS" == "linux" ]] || [[ "$OS" == "windows" ]]; then
    # Try multiple ways to detect NVIDIA GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected via nvidia-smi"
        HAS_NVIDIA_GPU=true
    elif command -v lspci &> /dev/null && lspci | grep -i nvidia &> /dev/null; then
        echo "🎮 NVIDIA GPU detected via lspci"
        HAS_NVIDIA_GPU=true
    else
        echo "💻 No NVIDIA GPU detected, using CPU-only PyTorch"
    fi
fi

# Choose environment file based on GPU availability
if [[ "$HAS_NVIDIA_GPU" == true ]]; then
    echo "🫙 Creating conda environment with CUDA support..."
    conda env create -f environment-gpu.yml
else
    echo "🫙 Creating conda environment..."
    conda env create -f environment.yml
fi

echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ml4me-student

echo "📦 Installing PyTorch..."
if [[ "$OS" == "macos" ]]; then
    echo "   Installing PyTorch for macOS..."
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
elif [[ "$HAS_NVIDIA_GPU" == true ]]; then
    echo "   Installing PyTorch with CUDA support..."
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
else
    echo "   Installing PyTorch CPU-only version..."
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi

echo "📚 Installing ML and Data Science libraries..."
pip install .

echo "🧪 Testing the setup..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} installed successfully')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__} installed successfully')"
python -c "import matplotlib; print(f'✅ Matplotlib {matplotlib.__version__} installed successfully')"
python -c "import sklearn; print(f'✅ Scikit-learn {sklearn.__version__} installed successfully')"

echo ""
echo "🎉 Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate ml4me-student"
echo "2. Start Jupyter: jupyter notebook"
echo "3. Open any notebook from the 'notebooks' folder"
echo ""
echo "Useful commands:"
echo "  conda activate ml4me-student    # Activate environment"
echo "  jupyter notebook                # Start Jupyter"
echo "  python notebooks/example.py     # Run Python scripts"
echo ""
echo "Happy learning! 🎓"
