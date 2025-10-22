# MotionBlend Deployment to Subdomain: moverse.rydlr.com

## Overview

Deploy the MotionBlend API server to the subdomain `moverse.rydlr.com` instead of path-based routing on the main domain.

## Directory Structure

```
/home/rydlr/domains/moverse.rydlr.com/
├── public_html/           # Document root (contains .htaccess)
│   └── .htaccess         # Apache proxy configuration
└── motionblend-api/       # API server files
    ├── api_server.py
    ├── requirements-api.txt
    ├── setup.sh
    ├── .env
    └── venv/
```

## Step 1: Create API Directory

```bash
mkdir -p /home/rydlr/domains/moverse.rydlr.com/motionblend-api
```

## Step 2: Download API Files

Use Virtualmin File Manager or wget to download files:

```bash
cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api

# Download files
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh

# Make setup script executable
chmod +x setup.sh
```

## Step 3: Setup Python Environment

```bash
cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api
./setup.sh
```

## Step 4: Configure Environment Variables

Create `/home/rydlr/domains/moverse.rydlr.com/motionblend-api/.env`:

```bash
# Google Cloud Storage
GCS_BUCKET=motionblend-mocap

# BigQuery
BQ_PROJECT=your-gcp-project-id
BQ_DATASET=RAW_DEV

# Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL=https://elasticsearch-motionblend-ba986d.es.us-central1.gcp.elastic.cloud
ES_API_KEY=V2VfQ0RKb0JMZW14WHRBTENhYWI6MW93UjJrZ2s1ZEVWcXdUdW1CVENEUQ==
ES_INDEX=mb_blends_v1

# Flask
FLASK_ENV=production
PORT=8080
```

## Step 5: Create Systemd Service

Create `/etc/systemd/system/motionblend-subdomain.service`:

```ini
[Unit]
Description=MotionBlend API Server (Subdomain)
After=network.target

[Service]
Type=simple
User=rydlr
Group=rydlr
WorkingDirectory=/home/rydlr/domains/moverse.rydlr.com/motionblend-api
Environment="PATH=/home/rydlr/domains/moverse.rydlr.com/motionblend-api/venv/bin"
EnvironmentFile=/home/rydlr/domains/moverse.rydlr.com/motionblend-api/.env
ExecStart=/home/rydlr/domains/moverse.rydlr.com/motionblend-api/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 120 api_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable motionblend-subdomain
sudo systemctl start motionblend-subdomain
sudo systemctl status motionblend-subdomain
```

## Step 6: Configure Apache for Subdomain

### Option A: Edit Apache Configuration Directly

In Virtualmin: **moverse.rydlr.com → Server Configuration → Edit Directives**

Add before `</VirtualHost>`:

```apache
# Proxy all requests to Python API
ProxyPass / http://localhost:8080/
ProxyPassReverse / http://localhost:8080/

# CORS Headers
Header always set Access-Control-Allow-Origin "*"
Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"

# Handle OPTIONS preflight
RewriteEngine On
RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^(.*)$ $1 [R=200,L]

<Location />
    Require all granted
</Location>
```

### Option B: Use .htaccess (Simpler)

Create `/home/rydlr/domains/moverse.rydlr.com/public_html/.htaccess`:

```apache
RewriteEngine On
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

<IfModule mod_headers.c>
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
</IfModule>

RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^(.*)$ - [R=200,L]

Options -Indexes +FollowSymLinks
Require all granted
```

## Step 7: Enable Apache Modules

```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod headers
sudo a2enmod rewrite
sudo systemctl restart apache2
```

## Step 8: Test Deployment

```bash
# Test API directly
curl http://localhost:8080/health

# Test through subdomain
curl https://moverse.rydlr.com/health

# Test all endpoints
curl https://moverse.rydlr.com/status
curl https://moverse.rydlr.com/motions
```

## Architecture

```
Browser Request
    ↓
https://moverse.rydlr.com/health
    ↓
Apache (subdomain:443)
    ↓
ProxyPass / → http://localhost:8080/
    ↓
Gunicorn (4 workers)
    ↓
Flask API (api_server.py)
    ↓
Google Cloud (GCS, BigQuery, Elasticsearch)
```

## DNS Configuration

Ensure `moverse.rydlr.com` points to your server IP (102.219.23.35).

## SSL Certificate

Virtualmin should automatically provision SSL for the subdomain if Let's Encrypt is enabled.

## Files Required

- `api_server.py` - Flask application
- `requirements-api.txt` - Python dependencies
- `setup.sh` - Environment setup
- `.env` - Environment configuration
- `.htaccess` - Apache proxy configuration (Option B)

## Endpoints Available

All API endpoints are available directly on the subdomain:

- `GET /health` - Health check
- `GET /status` - Detailed status
- `GET /motions` - List motions
- `POST /api/blend` - Create blend
- `GET /api/artifacts` - List artifacts
- `GET /api/artifact/{id}/describe` - Artifact details
- `GET /api/artifact/{id}/analysis` - Quality metrics
- `POST /search/vector` - Vector search
- `GET /` - API documentation

## Troubleshooting

### 403 Forbidden
```bash
# Fix permissions
sudo chown -R rydlr:www-data /home/rydlr/domains/moverse.rydlr.com/
sudo chmod -R 755 /home/rydlr/domains/moverse.rydlr.com/public_html/
```

### API Not Responding
```bash
# Check service status
sudo systemctl status motionblend-subdomain

# Check logs
sudo journalctl -u motionblend-subdomain -f

# Test local API
curl http://localhost:8080/health
```

### CORS Errors
Ensure the Apache configuration includes CORS headers as shown above.

## Alternative: Cloud Run Deployment

If you prefer Cloud Run, the `cloudbuild.yaml` is configured for subdomain deployment.

## UI Connection

The UI is now configured to connect to `https://moverse.rydlr.com` instead of path-based routing.