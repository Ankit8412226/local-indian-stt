#!/usr/bin/env bash
# setup_aws_gpu.sh — Automated Environment Setup Script for AWS GPU EC2 Instances
# Supports Ubuntu 20.04/22.04 and Amazon Linux Deep Learning AMIs with NVIDIA GPUs

set -e

echo "========================================================================"
echo "          AWS GPU EC2 Environment Setup — Local Indian STT"
echo "========================================================================"

# 1. System Hardware Audit
echo -e "\n[1/6] Auditing System Hardware..."
uname -a
if command -v lsb_release &> /dev/null; then
    lsb_release -a
elif [ -f /etc/os-release ]; then
    cat /etc/os-release | grep PRETTY_NAME
fi

echo -e "\n--- CPU & Memory ---"
lscpu | grep "Model name\|CPU(s):" || true
free -h

echo -e "\n--- Disk Space ---"
df -h /

echo -e "\n--- NVIDIA GPU Check ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "⚠️ WARNING: nvidia-smi not found. Ensure NVIDIA GPU drivers are installed."
fi

# 2. Install System Dependencies
echo -e "\n[2/6] Installing System Dependencies (ffmpeg, libsndfile, git, python3-venv)..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-pip python3-venv ffmpeg libsndfile1 git build-essential wget curl
elif command -v yum &> /dev/null; then
    sudo yum install -y python3-pip ffmpeg libsndfile git wget curl
fi

# 3. Create Virtual Environment
echo -e "\n[3/6] Setting up Python Virtual Environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# 4. Install PyTorch & Project Dependencies
echo -e "\n[4/6] Installing PyTorch with CUDA Support & Project Dependencies..."
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu121 || pip install -q torch torchaudio
pip install -q -r requirements.txt

# 5. Verify CUDA & PyTorch GPU Acceleration
echo -e "\n[5/6] Verifying PyTorch CUDA Acceleration..."
python3 -c "
import torch
print('PyTorch Version: ', torch.__version__)
print('CUDA Available:  ', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Count:    ', torch.cuda.device_count())
    print('Device Name:     ', torch.cuda.get_device_name(0))
    print('VRAM Total:      ', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), 'GB')
else:
    print('⚠️ Running on CPU mode (No CUDA GPU detected)')
"

# 6. Verify MongoDB Connection
echo -e "\n[6/6] Verifying MongoDB Atlas Connection..."
python3 -m src.database || echo "⚠️ MongoDB check skipped (set MONGODB_URI in .env)"

echo -e "\n========================================================================"
echo "✓ AWS GPU Setup Complete. Run './scripts/aws_train_workflow.sh' to proceed."
echo "========================================================================"
