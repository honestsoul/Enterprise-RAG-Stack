#!/usr/bin/env bash
# Set up a Python virtual environment and install project dependencies.
# This script is idempotent and can be re-run to ensure the environment
# is up to date.

set -euo pipefail

# Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate the environment
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete. Activate with 'source .venv/bin/activate'"
