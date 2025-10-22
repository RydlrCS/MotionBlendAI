# MotionBlend Deployment to /public_html/motionblend

## Quick Setup for www.rydlr.com/motionblend

### 1. Directory Structure

Create the following structure in your Virtualmin File Manager:

```
/home/rydlr/public_html/motionblend/
├── api/                    # API server files
│   ├── api_server.py
│   ├── requirements-api.txt
│   ├── setup.sh
│   ├── .env              # Environment variables
│   └── venv/             # Python virtual environment
└── ui/                     # React UI build (optional, for static hosting)
    ├── index.html
    ├── assets/
    └── ...
```

### 2. Download API Files

Use Virtualmin File Manager > "Download from remote URL":

**API Server:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
→ Save to: /home/rydlr/public_html/motionblend/api/api_server.py
```

**Requirements:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
→ Save to: /home/rydlr/public_html/motionblend/api/requirements-api.txt
```

**Setup Script:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh
→ Save to: /home/rydlr/public_html/motionblend/api/setup.sh
```

### 3. Setup Python Environment

In Virtualmin Command Shell:

```bash
cd /home/rydlr/public_html/motionblend/api
chmod +x setup.sh
./setup.sh
```

### 4. Configure Environment Variables

Create `/home/rydlr/public_html/motionblend/api/.env`:

```bash
# Google Cloud Storage
GCS_BUCKET=motionblend-mocap

# BigQuery
BQ_PROJECT=your-gcp-project-id
BQ_DATASET=RAW_DEV

# Elasticsearch (Elastic Cloud)
ELASTICSEARCH_URL=https://my-elasticsearch-project-ba986d.es.us-central1.gcp.elastic.cloud:443
ES_API_KEY=V2VfQ0RKb0JMZW14WHRBTENhYWI6MW93UjJrZ2s1ZEVWcXdUdW1CVENEUQ==
ES_INDEX=mb_blends_v1

# Flask
FLASK_ENV=production
PORT=8080
```

### 5. Create Systemd Service

Create `/etc/systemd/system/motionblend-api.service`:

```ini
[Unit]
Description=MotionBlend API Server
After=network.target

[Service]
Type=simple
User=rydlr
Group=rydlr
WorkingDirectory=/home/rydlr/public_html/motionblend/api
Environment="PATH=/home/rydlr/public_html/motionblend/api/venv/bin"
EnvironmentFile=/home/rydlr/public_html/motionblend/api/.env
ExecStart=/home/rydlr/public_html/motionblend/api/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 120 api_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable motionblend-api
sudo systemctl start motionblend-api
sudo systemctl status motionblend-api
```

### 6. Configure Apache Reverse Proxy

Add to your Apache configuration (or Virtualmin > Server Configuration > Edit Directives):

```apache
<Location /motionblend>
    ProxyPass http://localhost:8080
    ProxyPassReverse http://localhost:8080
    
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization"
    
    RewriteEngine On
    RewriteCond %{REQUEST_METHOD} OPTIONS
    RewriteRule ^(.*)$ $1 [R=200,L]
</Location>
```

Enable required modules and restart:

```bash
sudo a2enmod proxy proxy_http headers rewrite
sudo systemctl restart apache2
```

### 7. Test Deployment

```bash
# Health check
curl https://www.rydlr.com/motionblend/health

# Status
curl https://www.rydlr.com/motionblend/status

# List motions
curl https://www.rydlr.com/motionblend/motions
```

### 8. Deploy UI (Optional)

If hosting the UI statically:

```bash
# Build the UI locally
cd ui
npm run build

# Upload dist/ contents to /home/rydlr/public_html/motionblend/ui/
# via SFTP or Virtualmin File Manager
```

Configure UI routing in Apache:

```apache
<Directory /home/rydlr/public_html/motionblend/ui>
    Options -Indexes +FollowSymLinks
    AllowOverride All
    Require all granted
    
    RewriteEngine On
    RewriteBase /motionblend/ui/
    RewriteRule ^index\.html$ - [L]
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteRule . /motionblend/ui/index.html [L]
</Directory>
```

Access UI at: `https://www.rydlr.com/motionblend/ui/`

### 9. Monitoring

Check API logs:

```bash
# Service logs
sudo journalctl -u motionblend-api -f

# Apache logs
sudo tail -f /var/log/apache2/error.log
sudo tail -f /var/log/apache2/access.log
```

### 10. Troubleshooting

**API not responding:**
```bash
# Check service status
sudo systemctl status motionblend-api

# Check if port is listening
sudo netstat -tlnp | grep 8080

# Test local connection
curl http://localhost:8080/health
```

**CORS errors:**
```bash
# Verify Apache headers module
sudo apache2ctl -M | grep headers

# Check Apache error log for CORS issues
sudo tail -f /var/log/apache2/error.log
```

**Python environment issues:**
```bash
cd /home/rydlr/public_html/motionblend/api
source venv/bin/activate
python -c "import flask; print(flask.__version__)"
pip list
```

## Architecture

```
Browser Request
    ↓
https://www.rydlr.com/motionblend/health
    ↓
Apache (port 443)
    ↓
ProxyPass /motionblend → http://localhost:8080
    ↓
Gunicorn (4 workers)
    ↓
Flask API (api_server.py)
    ↓
Google Cloud (GCS, BigQuery, Elasticsearch)
```

## Files Deployed

- `api_server.py` - Flask application (700 lines, 10 endpoints)
- `requirements-api.txt` - Python dependencies
- `setup.sh` - Environment setup script
- `.env` - Environment configuration
- `apache-motionblend.conf` - Apache configuration
- `cloudbuild.yaml` - Cloud Build configuration (optional)

## Endpoints Available

- `GET /health` - Health check
- `GET /status` - Detailed status
- `GET /motions` - List available motions
- `POST /api/blend` - Create blend operation
- `GET /api/artifacts` - List artifacts
- `GET /api/artifacts/manifest` - Artifacts metadata
- `GET /api/artifact/{id}/describe` - Artifact details
- `GET /api/artifact/{id}/analysis` - Quality metrics
- `POST /search/vector` - Elasticsearch vector search
- `GET /` - API documentation

All accessible at `https://www.rydlr.com/motionblend/[endpoint]`
