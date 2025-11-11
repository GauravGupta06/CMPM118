#!/bin/bash
# setup_venv.sh — creates venv and installs dependencies

# Go to project directory
cd /root/CMPM118 || exit

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install numpy tonic matplotlib snntorch torch Lempel-Ziv-Complexity

echo "Virtual environment ready. Run 'source /root/CMPM118/.venv/bin/activate' next time to reactivate it."


# TO RUN: bash setup_venv.sh




