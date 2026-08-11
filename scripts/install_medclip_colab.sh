#!/usr/bin/env bash
# Install MedCLIP in a fresh Python 3.12 Colab evaluation runtime.
# Do not run this in the RoentGen/BioGPT training runtime.
set -euo pipefail

python -m pip install --no-deps \
  git+https://github.com/RyanWangZf/MedCLIP.git@9c3396f20d5d54e4fae241b8cb06ca45848e98c9
python -m pip install -r requirements-medclip.txt
