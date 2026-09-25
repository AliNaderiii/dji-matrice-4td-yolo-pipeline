#!/bin/bash
# Setup environment for DJI Matrice 4TD Pipeline
# Author: Ali Naderi

set -e

echo "=== DJI Matrice 4TD Pipeline - Environment Setup ==="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "Conda not found, please install Miniconda first"
    exit 1
fi

# Create env
echo "Creating conda env: dji"
conda create -n dji python=3.10 -y

echo "Activating env and installing dependencies..."
eval "$(conda shell.bash hook)"
conda activate dji

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Environment setup complete"
echo ""
echo "To activate:"
echo "  conda activate dji"
echo ""
echo "To test:"
echo "  python scripts/download_dataset.py --source synthetic --limit 100"
echo "  python scripts/train.py --data datasets/dji-hardhat/dji_matrice.yaml --epochs 5"
echo ""
echo "For Docker MQTT broker:"
echo "  docker-compose -f docker/docker-compose.yml up -d"
