#!/bin/bash
# Manual deployment script for MotionBlend
# Run this on your server to update from GitHub

set -e

echo "🚀 Starting manual deployment from GitHub..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root or with sudo
if [[ $EUID -eq 0 ]]; then
   print_warning "Running as root - this is not recommended for production"
fi

# API Deployment
echo "📦 Deploying API server..."
cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api

# Backup current files
cp api_server.py api_server.py.backup 2>/dev/null || true

# Download latest files
wget -q -O api_server.py https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
wget -q -O requirements-api.txt https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt

# Check if files changed
if cmp -s api_server.py api_server.py.backup; then
    print_status "API server unchanged"
else
    print_status "API server updated"
fi

print_status "API deployment completed"

# UI Deployment
echo "🎨 Deploying UI files..."
cd /home/rydlr/domains/moverse.rydlr.com/public_html

# Create assets directory if it doesn't exist
mkdir -p assets

# Download UI files
wget -q -O index.html https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html
wget -q -O assets/index-5dea55ab.css https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-5dea55ab.css
wget -q -O assets/index-ac34afc1.js https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-ac34afc1.js

print_status "UI deployment completed"

# Apache Configuration
echo "⚙️ Updating Apache configuration..."
wget -q -O .htaccess https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain

print_status "Apache configuration updated"

# Service Management
echo "🔄 Managing services..."

# Check if API service exists and restart it
if systemctl is-active --quiet moverse-api; then
    sudo systemctl restart moverse-api
    print_status "API service restarted"
else
    print_warning "API service not running - you may need to start it manually"
    echo "Run: sudo systemctl start moverse-api"
fi

# Reload Apache
sudo systemctl reload apache2
print_status "Apache reloaded"

# Health Checks
echo "🏥 Running health checks..."

# Wait a moment for services to start
sleep 5

# Test API
if curl -f -s http://localhost:8080/health > /dev/null; then
    print_status "API health check passed"
else
    print_error "API health check failed"
fi

# Test UI
if curl -f -s -I https://moverse.rydlr.com/ > /dev/null; then
    print_status "UI accessibility check passed"
else
    print_error "UI accessibility check failed"
fi

echo
print_status "Deployment completed successfully!"
echo
echo "🌐 Your application is available at:"
echo "   UI:  https://moverse.rydlr.com/"
echo "   API: https://moverse.rydlr.com/health"
echo
echo "📊 Check status: https://moverse.rydlr.com/status"