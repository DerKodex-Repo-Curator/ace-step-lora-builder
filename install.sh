#!/usr/bin/env bash
set -euo pipefail

# URL Rotation Script Installation Script
# Creates a virtual environment and installs dependencies

echo "Setting up URL Rotation Script environment..."

# Check if pyenv is available
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed or not in PATH"
    echo "Please install pyenv first: https://github.com/pyenv/pyenv#installation"
    exit 1
fi

# Set Python version from .python-version file
echo "Setting Python version from .python-version..."
pyenv install --skip-existing
pyenv local

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo "Using Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Initialize and checkout submodules
echo "Initializing and checking out submodules..."
if [ -f ".gitmodules" ]; then
    git submodule init
    git submodule update --recursive
    echo "Submodules initialized and checked out successfully"
else
    echo "No .gitmodules file found, skipping submodule setup"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Set up virtual environments for submodules with requirements.txt
echo "Setting up virtual environments for submodules..."
for submodule_dir in dependencies/*/; do
    if [ -d "$submodule_dir" ] && [ -f "${submodule_dir}requirements.txt" ]; then
        submodule_name=$(basename "$submodule_dir")
        echo "Found requirements.txt in $submodule_name, setting up virtual environment..."
        
        # Create virtual environment in submodule directory
        if [ ! -d "${submodule_dir}venv" ]; then
            echo "Creating virtual environment for $submodule_name..."
            python -m venv "${submodule_dir}venv"
        else
            echo "Virtual environment already exists for $submodule_name"
        fi
        
        # Activate submodule virtual environment and install dependencies
        echo "Installing dependencies for $submodule_name..."
        source "${submodule_dir}venv/bin/activate"
        pip install --upgrade pip
        pip install -r "${submodule_dir}requirements.txt"
        deactivate
        
        echo "Successfully set up environment for $submodule_name"
    fi
done

echo "All submodule environments configured successfully"
