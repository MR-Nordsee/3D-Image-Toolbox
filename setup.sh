#!/bin/bash

# Get the absolute path to this script's directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/tools"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Creating virtual environment at '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
else
    echo "📦 Virtual environment already exists at '$VENV_DIR'."
fi

# Activate the environment
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated."

# Install required packages
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "📥 Installing packages from requirements.txt..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "⚠️ requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

# Download Depth Anything AI Model
echo "🔧 Download Depth-Anything-V2 Git Repo..."
rm -R -f $SCRIPT_DIR/Depth-Anything-V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
if [ ! -d "depth_anything_v2" ]; then
    echo "⚠️ Delete existing Depth Anything Folder $SCRIPT_DIR/depth_anything_v2"
    rm -R -f depth_anything_v2
fi
echo "📦 Copy Module from Repo to Main Folder and remove Repo."
cp -R -f $SCRIPT_DIR/Depth-Anything-V2/depth_anything_v2 $SCRIPT_DIR
rm -R -f $SCRIPT_DIR/Depth-Anything-V2

echo "🔧 Download Depth-Anything-V2 AI Model..."
curl -L "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true" --output "$SCRIPT_DIR/depth_anything_v2_vitl.pth"
echo "✅ Finished. ✅"