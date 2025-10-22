#!/bin/bash
# MotionBlend API Setup Script
# Run this on your server after downloading

set -e

echo "🚀 MotionBlend API Setup"
echo "======================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo "Installing dependencies..."
pip install -q flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch gunicorn

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your Google Cloud credentials to: credentials/"
echo "2. Test the server: python3 api_server.py"
echo "3. Configure systemd service (see MANUAL_DEPLOYMENT.md)"
echo ""
