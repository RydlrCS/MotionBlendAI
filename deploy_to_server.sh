#!/bin/bash
# Quick deployment script for moverse.rydlr.com
# Run this from your Mac to deploy to the server

set -e  # Exit on error

echo "🚀 MotionBlend API Deployment to moverse.rydlr.com"
echo "=================================================="

# Configuration
SERVER="moverse.rydlr.com"
USER="rydlr"
API_DIR="/home/rydlr/domains/moverse.rydlr.com/api"
REPO_DIR="/Users/ted/blenderkit_data/MotionBlendAI-1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo ""
echo -e "${YELLOW}Step 1: Creating API directory on server...${NC}"
ssh $USER@$SERVER "mkdir -p $API_DIR/credentials"

echo ""
echo -e "${YELLOW}Step 2: Uploading API server...${NC}"
scp $REPO_DIR/api_server.py $USER@$SERVER:$API_DIR/
scp $REPO_DIR/requirements-api.txt $USER@$SERVER:$API_DIR/

echo ""
echo -e "${YELLOW}Step 3: Setting up Python environment...${NC}"
ssh $USER@$SERVER << 'ENDSSH'
cd /home/rydlr/domains/moverse.rydlr.com/api

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate
pip install -q flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch gunicorn

echo "✓ Python environment ready"
ENDSSH

echo ""
echo -e "${YELLOW}Step 4: Uploading Google Cloud credentials...${NC}"
if [ -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
    scp $HOME/.config/gcloud/application_default_credentials.json $USER@$SERVER:$API_DIR/credentials/
    echo -e "${GREEN}✓ Credentials uploaded${NC}"
else
    echo -e "${RED}⚠ Credentials not found at $HOME/.config/gcloud/application_default_credentials.json${NC}"
    echo "Please upload manually or run: gcloud auth application-default login"
fi

echo ""
echo -e "${YELLOW}Step 5: Creating systemd service...${NC}"
cat > /tmp/moverse-api.service << 'EOF'
[Unit]
Description=Moverse MotionBlend API Server
After=network.target

[Service]
Type=simple
User=rydlr
Group=rydlr
WorkingDirectory=/home/rydlr/domains/moverse.rydlr.com/api

Environment="GCS_BUCKET=motionblend-mocap"
Environment="BQ_PROJECT=motionblend-ai"
Environment="BQ_DATASET=RAW_DEV"
Environment="ELASTICSEARCH_URL=http://localhost:9200"
Environment="ES_INDEX=mb_blends_v1"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/rydlr/domains/moverse.rydlr.com/api/credentials/application_default_credentials.json"

ExecStart=/home/rydlr/domains/moverse.rydlr.com/api/venv/bin/gunicorn \
    --bind 127.0.0.1:8080 \
    --workers 4 \
    --timeout 120 \
    --access-logfile /home/rydlr/domains/moverse.rydlr.com/logs/api-access.log \
    --error-logfile /home/rydlr/domains/moverse.rydlr.com/logs/api-error.log \
    api_server:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

scp /tmp/moverse-api.service $USER@$SERVER:/tmp/
ssh $USER@$SERVER "sudo mv /tmp/moverse-api.service /etc/systemd/system/moverse-api.service"
rm /tmp/moverse-api.service

echo ""
echo -e "${YELLOW}Step 6: Starting service...${NC}"
ssh $USER@$SERVER << 'ENDSSH'
sudo systemctl daemon-reload
sudo systemctl enable moverse-api
sudo systemctl restart moverse-api
sleep 2
sudo systemctl status moverse-api --no-pager
ENDSSH

echo ""
echo -e "${YELLOW}Step 7: Testing API...${NC}"
sleep 3

# Test health endpoint
if curl -s -f https://$SERVER/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is responding at https://$SERVER/health${NC}"
else
    echo -e "${YELLOW}⚠ API not yet accessible via HTTPS (Apache configuration needed)${NC}"
    echo "Testing direct connection..."
    
    # Test direct connection
    if ssh $USER@$SERVER "curl -s http://127.0.0.1:8080/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is running locally on port 8080${NC}"
        echo -e "${YELLOW}→ You need to configure Apache reverse proxy${NC}"
        echo "   See VIRTUALMIN_DEPLOYMENT.md for Apache configuration"
    else
        echo -e "${RED}✗ API is not responding${NC}"
        echo "Check logs: ssh $USER@$SERVER 'sudo journalctl -u moverse-api -n 50'"
    fi
fi

echo ""
echo "=================================================="
echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Configure Apache reverse proxy (see VIRTUALMIN_DEPLOYMENT.md)"
echo "2. Enable SSL certificate for moverse.rydlr.com"
echo "3. Test endpoints: curl https://moverse.rydlr.com/health"
echo "4. Deploy UI: cd ui && npm run build"
echo ""
echo "Useful commands:"
echo "  - View logs: ssh $USER@$SERVER 'sudo journalctl -u moverse-api -f'"
echo "  - Restart: ssh $USER@$SERVER 'sudo systemctl restart moverse-api'"
echo "  - Status: ssh $USER@$SERVER 'sudo systemctl status moverse-api'"
echo ""
