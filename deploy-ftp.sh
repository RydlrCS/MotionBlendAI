#!/bin/bash
# FTP Deployment Script for MotionBlend AI
# Alternative to SSH deployment using FTP access

set -e

echo "📤 MotionBlend AI FTP Deployment"
echo "================================="

# FTP Configuration
FTP_HOST="rydlr-ftp.moverse"
FTP_USER="rydlr-ftp"
FTP_PASS="W0kd2yTiJ5y2pg0"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Create temporary directory for FTP operations
TEMP_DIR=$(mktemp -d)
echo "📁 Using temp directory: $TEMP_DIR"

# Download latest files
echo "📥 Downloading latest files..."

# API files
curl -s -o "$TEMP_DIR/api_server.py" https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
curl -s -o "$TEMP_DIR/requirements-api.txt" https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt

# UI files
curl -s -o "$TEMP_DIR/index.html" https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html
curl -s -o "$TEMP_DIR/index-08f65cc5.css" https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-08f65cc5.css
curl -s -o "$TEMP_DIR/index-0d9f8f93.js" https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-0d9f8f93.js

# Config files
curl -s -o "$TEMP_DIR/.htaccess" https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain

print_status "Files downloaded to temp directory"

# Create FTP upload script
cat > "$TEMP_DIR/ftp_upload.txt" << EOF
open $FTP_HOST
user $FTP_USER $FTP_PASS
binary

# Upload API files to motionblend-api directory
mkdir motionblend-api
cd motionblend-api
put api_server.py
put requirements-api.txt
cd ..

# Upload UI files to public_html directory
cd public_html
put index.html
mkdir assets
cd assets
put index-08f65cc5.css
put index-0d9f8f93.js
cd ..
put .htaccess

# Close connection
bye
EOF

echo "🔄 Uploading files via FTP..."
# Upload API files
curl -T "$TEMP_DIR/api_server.py" ftp://$FTP_USER:$FTP_PASS@$FTP_HOST/motionblend-api/
curl -T "$TEMP_DIR/requirements-api.txt" ftp://$FTP_USER:$FTP_PASS@$FTP_HOST/motionblend-api/

# Upload UI files
curl -T "$TEMP_DIR/index.html" ftp://$FTP_USER:$FTP_PASS@$FTP_HOST/public_html/
curl -T "$TEMP_DIR/index-08f65cc5.css" ftp://$FTP_USER:$FTP_PASS@$FTP_HOST/public_html/assets/
curl -T "$TEMP_DIR/index-0d9f8f93.js" ftp://$FTP_USER:$FTP_PASS@$FTP_HOST/public_html/assets/
curl -T "$TEMP_DIR/.htaccess" ftp://$FTP_USER:$FTP_PASS@$FTP_HOST/public_html/

if [ $? -eq 0 ]; then
    print_status "FTP upload completed successfully"
else
    print_error "FTP upload failed"
    exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"
print_status "Cleanup completed"

echo
print_status "FTP Deployment Summary:"
echo "  🌐 UI: https://moverse.rydlr.com/"
echo "  🔗 API: https://moverse.rydlr.com/health"
echo "  📊 Status: https://moverse.rydlr.com/status"
echo
echo "Note: You may need to restart services manually:"
echo "  sudo systemctl restart moverse-api"
echo "  sudo systemctl reload apache2"
echo
print_warning "Remember to keep FTP credentials secure and never commit them to version control!"