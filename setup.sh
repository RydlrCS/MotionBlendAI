#!/bin/bash
# MotionBlend AI Server Setup Script
# Run this once on your server to set up the complete environment

set -e

echo "🚀 MotionBlend AI Server Setup"
echo "================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as rydlr user
if [[ "$USER" != "rydlr" ]]; then
    print_error "This script must be run as user 'rydlr'"
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p /home/rydlr/domains/moverse.rydlr.com/motionblend-api
mkdir -p /home/rydlr/domains/moverse.rydlr.com/public_html/assets
print_status "Directories created"

# Setup Python virtual environment
echo "🐍 Setting up Python environment..."
cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api

if [[ ! -d ~/.venv/motionblend ]]; then
    python3 -m venv ~/.venv/motionblend
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

source ~/.venv/motionblend/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
pip install -r requirements-api.txt
print_status "Python dependencies installed"

# Download API server
echo "📥 Downloading API server..."
wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
chmod +x api_server.py
print_status "API server downloaded"

# Setup systemd service
echo "⚙️ Setting up systemd service..."
wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/moverse-api.service
sudo cp moverse-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable moverse-api
print_status "Systemd service configured"

# Setup UI files
echo "🎨 Setting up UI files..."
cd /home/rydlr/domains/moverse.rydlr.com/public_html

wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html
wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-5dea55ab.css -O assets/
wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-ac34afc1.js -O assets/
print_status "UI files downloaded"

# Setup Apache configuration
echo "🌐 Setting up Apache configuration..."
wget -q https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain -O .htaccess
sudo systemctl reload apache2
print_status "Apache configuration updated"

# Start the service
echo "▶️ Starting API service..."
sudo systemctl start moverse-api

# Wait a moment for service to start
sleep 3

# Check service status
if sudo systemctl is-active --quiet moverse-api; then
    print_status "API service started successfully"
else
    print_error "API service failed to start"
    echo "Check logs with: sudo journalctl -u moverse-api -f"
    exit 1
fi

# Health checks
echo "🏥 Running health checks..."

# Test API health
if curl -f -s http://localhost:8080/health > /dev/null; then
    print_status "API health check passed"
else
    print_warning "API health check failed - service may still be starting"
fi

# Test UI accessibility
if curl -f -s -I https://moverse.rydlr.com/ > /dev/null; then
    print_status "UI accessibility check passed"
else
    print_warning "UI accessibility check failed - Apache may need reload"
fi

echo
print_status "Setup completed successfully!"
echo
echo "🌐 Your application is available at:"
echo "   UI:  https://moverse.rydlr.com/"
echo "   API: https://moverse.rydlr.com/health"
echo
echo "🔄 To check service status: sudo systemctl status moverse-api"
echo "📜 To view logs: sudo journalctl -u moverse-api -f"
echo
echo "🎉 Next: Set up GitHub integration for automatic deployments!"