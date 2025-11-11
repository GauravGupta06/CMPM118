#!/bin/bash
cd /root/CMPM118
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt


# TO RUN: bash setup_venv.sh

