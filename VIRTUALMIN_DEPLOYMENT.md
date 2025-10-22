# MotionBlend API - Virtualmin Deployment Guide

## Your Setup

- **Domain:** `moverse.rydlr.com`
- **Home Directory:** `/home/rydlr/domains/moverse.rydlr.com`
- **IP Address:** `102.219.23.35`
- **Admin User:** `rydlr`
- **Admin Group:** `rydlr`
- **Parent Domain:** `rydlr.com`

## Features Enabled
- ✅ DNS domain
- ✅ Apache website
- ✅ Apache SSL website
- ✅ MariaDB database
- ✅ PostgreSQL database
- ✅ Log file rotation
- ✅ Status monitoring
- ✅ Webalizer reporting

---

## Deployment Steps

### 1. SSH to Your Server

```bash
ssh rydlr@moverse.rydlr.com
# Or if using parent domain:
# ssh rydlr@rydlr.com
```

### 2. Set Up API Directory

```bash
# Navigate to domain directory
cd /home/rydlr/domains/moverse.rydlr.com

# Create API directory
mkdir -p api
cd api

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch gunicorn
```

### 3. Upload API Server

**Option A: Using Git (Recommended)**

```bash
cd /home/rydlr/domains/moverse.rydlr.com/api

# Clone your repository
git clone https://github.com/RydlrCS/MotionBlendAI.git temp
mv temp/api_server.py .
mv temp/requirements-api.txt .
rm -rf temp

# Or just download the file directly
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
```

**Option B: Using SCP from your Mac**

```bash
# From your Mac
scp /Users/ted/blenderkit_data/MotionBlendAI-1/api_server.py rydlr@moverse.rydlr.com:/home/rydlr/domains/moverse.rydlr.com/api/
scp /Users/ted/blenderkit_data/MotionBlendAI-1/requirements-api.txt rydlr@moverse.rydlr.com:/home/rydlr/domains/moverse.rydlr.com/api/
```

### 4. Set Up Google Cloud Credentials

```bash
cd /home/rydlr/domains/moverse.rydlr.com/api

# Create credentials directory
mkdir -p credentials

# Upload your service account key
# From your Mac:
scp ~/.config/gcloud/application_default_credentials.json rydlr@moverse.rydlr.com:/home/rydlr/domains/moverse.rydlr.com/api/credentials/

# Or create new service account key:
# 1. Go to https://console.cloud.google.com/iam-admin/serviceaccounts
# 2. Select project: motionblend-ai
# 3. Create key for service account
# 4. Download JSON and upload to server
```

### 5. Create Systemd Service

```bash
sudo nano /etc/systemd/system/moverse-api.service
```

Add the following content:

```ini
[Unit]
Description=Moverse MotionBlend API Server
After=network.target

[Service]
Type=simple
User=rydlr
Group=rydlr
WorkingDirectory=/home/rydlr/domains/moverse.rydlr.com/api

# Environment variables
Environment="GCS_BUCKET=motionblend-mocap"
Environment="BQ_PROJECT=motionblend-ai"
Environment="BQ_DATASET=RAW_DEV"
Environment="ELASTICSEARCH_URL=http://localhost:9200"
Environment="ES_INDEX=mb_blends_v1"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/rydlr/domains/moverse.rydlr.com/api/credentials/application_default_credentials.json"

# Run with gunicorn for production
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
```

### 6. Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable moverse-api

# Start the service
sudo systemctl start moverse-api

# Check status
sudo systemctl status moverse-api

# View logs
sudo journalctl -u moverse-api -f
```

### 7. Configure Apache Reverse Proxy

**Option A: Using Virtualmin UI**

1. Log in to Virtualmin: `https://rydlr.com:10000`
2. Select domain: `moverse.rydlr.com`
3. Go to **Server Configuration → Website Options**
4. Scroll to **Configure website redirects and aliases**
5. Add custom directives (see below)

**Option B: Edit Apache Config Directly**

```bash
sudo nano /etc/apache2/sites-available/moverse.rydlr.com.conf
```

Add inside the `<VirtualHost *:443>` block:

```apache
<VirtualHost *:443>
    ServerName moverse.rydlr.com
    
    # Existing SSL configuration...
    SSLEngine on
    SSLCertificateFile /home/rydlr/domains/moverse.rydlr.com/ssl.cert
    SSLCertificateKeyFile /home/rydlr/domains/moverse.rydlr.com/ssl.key
    
    # Enable required modules
    # Run: sudo a2enmod proxy proxy_http headers ssl rewrite
    
    # Root API reverse proxy
    ProxyPreserveHost On
    ProxyPass / http://127.0.0.1:8080/
    ProxyPassReverse / http://127.0.0.1:8080/
    
    # CORS headers
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization"
    Header always set Access-Control-Max-Age "86400"
    
    # Handle OPTIONS preflight requests
    RewriteEngine On
    RewriteCond %{REQUEST_METHOD} OPTIONS
    RewriteRule ^(.*)$ $1 [R=204,L]
    
    # Logging
    ErrorLog /home/rydlr/domains/moverse.rydlr.com/logs/error_log
    CustomLog /home/rydlr/domains/moverse.rydlr.com/logs/access_log combined
</VirtualHost>

# Redirect HTTP to HTTPS
<VirtualHost *:80>
    ServerName moverse.rydlr.com
    
    RewriteEngine On
    RewriteCond %{HTTPS} off
    RewriteRule ^(.*)$ https://%{HTTP_HOST}$1 [R=301,L]
</VirtualHost>
```

### 8. Enable Apache Modules and Restart

```bash
# Enable required modules
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod headers
sudo a2enmod ssl
sudo a2enmod rewrite

# Test Apache configuration
sudo apache2ctl configtest

# Restart Apache
sudo systemctl restart apache2
```

### 9. Configure Firewall (if applicable)

```bash
# Allow port 8080 for local connections only (gunicorn)
sudo ufw allow from 127.0.0.1 to any port 8080

# Ensure HTTPS is allowed
sudo ufw allow 443/tcp
sudo ufw allow 80/tcp

# Check status
sudo ufw status
```

---

## Testing the Deployment

### 1. Test API Server Directly

```bash
# From the server
curl http://127.0.0.1:8080/health

# Expected output:
# {
#   "status": "healthy",
#   "service": "MotionBlend AI API",
#   "version": "1.0.0",
#   "timestamp": "2025-10-22T17:00:00Z"
# }
```

### 2. Test Through Apache (HTTPS)

```bash
# From your Mac or any computer
curl https://moverse.rydlr.com/health

# Test CORS headers
curl -I -X OPTIONS https://moverse.rydlr.com/health \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET"
```

### 3. Test All Endpoints

```bash
# Health check
curl https://moverse.rydlr.com/health

# List motions
curl https://moverse.rydlr.com/motions

# Create blend
curl -X POST https://moverse.rydlr.com/api/blend \
  -H "Content-Type: application/json" \
  -d '{
    "motion1": "walking_forward",
    "motion2": "jump_landing",
    "weight": 0.5
  }'

# Get artifacts
curl https://moverse.rydlr.com/api/artifacts/manifest
```

---

## Deploy React UI

### Option 1: Static Files on Same Server

```bash
# On your Mac, build the UI
cd /Users/ted/blenderkit_data/MotionBlendAI-1/ui
npm run build

# Upload to server
scp -r dist/* rydlr@moverse.rydlr.com:/home/rydlr/domains/moverse.rydlr.com/public_html/

# Update Apache to serve UI and proxy API
```

Update Apache config to serve UI on subdomain `app.moverse.rydlr.com` or separate path:

```apache
<VirtualHost *:443>
    ServerName moverse.rydlr.com
    DocumentRoot /home/rydlr/domains/moverse.rydlr.com/public_html
    
    # Serve static UI files
    <Directory /home/rydlr/domains/moverse.rydlr.com/public_html>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
        
        # SPA fallback - redirect all to index.html
        RewriteEngine On
        RewriteBase /
        RewriteRule ^index\.html$ - [L]
        RewriteCond %{REQUEST_FILENAME} !-f
        RewriteCond %{REQUEST_FILENAME} !-d
        RewriteRule . /index.html [L]
    </Directory>
    
    # API endpoints - proxy to backend
    ProxyPreserveHost On
    ProxyPass /api/ http://127.0.0.1:8080/api/
    ProxyPassReverse /api/ http://127.0.0.1:8080/api/
    
    ProxyPass /health http://127.0.0.1:8080/health
    ProxyPassReverse /health http://127.0.0.1:8080/health
    
    ProxyPass /motions http://127.0.0.1:8080/motions
    ProxyPassReverse /motions http://127.0.0.1:8080/motions
    
    ProxyPass /search/ http://127.0.0.1:8080/search/
    ProxyPassReverse /search/ http://127.0.0.1:8080/search/
    
    # CORS headers
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization"
</VirtualHost>
```

### Option 2: Deploy UI to Vercel/Netlify (Recommended)

```bash
cd /Users/ted/blenderkit_data/MotionBlendAI-1/ui

# Ensure .env uses production API
echo "VITE_API_URL=https://moverse.rydlr.com" > .env

# Deploy to Vercel
npx vercel --prod

# Or deploy to Netlify
npx netlify deploy --prod
```

---

## Monitoring & Maintenance

### View Logs

```bash
# API server logs
sudo journalctl -u moverse-api -f

# Gunicorn access logs
tail -f /home/rydlr/domains/moverse.rydlr.com/logs/api-access.log

# Gunicorn error logs
tail -f /home/rydlr/domains/moverse.rydlr.com/logs/api-error.log

# Apache logs
tail -f /home/rydlr/domains/moverse.rydlr.com/logs/error_log
tail -f /home/rydlr/domains/moverse.rydlr.com/logs/access_log
```

### Service Management

```bash
# Check status
sudo systemctl status moverse-api

# Restart service
sudo systemctl restart moverse-api

# Stop service
sudo systemctl stop moverse-api

# View service configuration
sudo systemctl cat moverse-api
```

### Update API Server

```bash
cd /home/rydlr/domains/moverse.rydlr.com/api

# Pull latest changes
git pull

# Or upload new version
# scp api_server.py rydlr@moverse.rydlr.com:/home/rydlr/domains/moverse.rydlr.com/api/

# Restart service
sudo systemctl restart moverse-api
```

---

## Troubleshooting

### Issue: Service won't start

```bash
# Check logs
sudo journalctl -u moverse-api -n 50

# Common causes:
# 1. Port 8080 already in use
sudo lsof -i :8080

# 2. Python dependencies missing
cd /home/rydlr/domains/moverse.rydlr.com/api
source venv/bin/activate
pip install -r requirements-api.txt

# 3. Permissions wrong
sudo chown -R rydlr:rydlr /home/rydlr/domains/moverse.rydlr.com/api
chmod +x api_server.py
```

### Issue: 502 Bad Gateway

```bash
# Check if API is running
sudo systemctl status moverse-api
curl http://127.0.0.1:8080/health

# Check Apache proxy configuration
sudo apache2ctl configtest

# Check Apache error logs
tail -f /home/rydlr/domains/moverse.rydlr.com/logs/error_log
```

### Issue: CORS errors

```bash
# Verify CORS headers are being sent
curl -I https://moverse.rydlr.com/health

# Should include:
# Access-Control-Allow-Origin: *
# Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS

# If missing, check Apache modules
sudo a2enmod headers
sudo systemctl restart apache2
```

### Issue: SSL certificate errors

```bash
# Let's Encrypt certificate (if using)
sudo certbot --apache -d moverse.rydlr.com

# Or use Virtualmin to manage SSL certificates
# Virtualmin → Server Configuration → SSL Certificate
```

---

## Security Hardening

### 1. Restrict CORS Origins

Edit `api_server.py`:

```python
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://moverse.rydlr.com",
            "https://app.moverse.rydlr.com",  # If UI on subdomain
            "http://localhost:5173"  # For development
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### 2. Add Rate Limiting

```bash
# Install mod_evasive for Apache
sudo apt-get install libapache2-mod-evasive
sudo a2enmod evasive
```

Add to Apache config:

```apache
<IfModule mod_evasive20.c>
    DOSHashTableSize 3097
    DOSPageCount 10
    DOSSiteCount 100
    DOSPageInterval 1
    DOSSiteInterval 1
    DOSBlockingPeriod 10
</IfModule>
```

### 3. Update Firewall

```bash
# Block port 8080 from external access
sudo ufw deny 8080/tcp
sudo ufw allow from 127.0.0.1 to any port 8080

# Only allow HTTPS
sudo ufw allow 443/tcp
sudo ufw deny 80/tcp  # Or redirect to 443
```

---

## Quick Reference

### Service Commands
- Start: `sudo systemctl start moverse-api`
- Stop: `sudo systemctl stop moverse-api`
- Restart: `sudo systemctl restart moverse-api`
- Status: `sudo systemctl status moverse-api`
- Logs: `sudo journalctl -u moverse-api -f`

### Apache Commands
- Test config: `sudo apache2ctl configtest`
- Restart: `sudo systemctl restart apache2`
- Reload: `sudo systemctl reload apache2`

### Paths
- API: `/home/rydlr/domains/moverse.rydlr.com/api/`
- Public HTML: `/home/rydlr/domains/moverse.rydlr.com/public_html/`
- Logs: `/home/rydlr/domains/moverse.rydlr.com/logs/`
- Config: `/etc/apache2/sites-available/moverse.rydlr.com.conf`

---

## Next Steps

1. ✅ SSH to server
2. ✅ Set up API directory structure
3. ✅ Upload `api_server.py`
4. ✅ Install Python dependencies
5. ✅ Upload Google Cloud credentials
6. ✅ Create systemd service
7. ✅ Configure Apache reverse proxy
8. ✅ Enable Apache modules
9. ✅ Test API endpoints
10. ✅ Deploy React UI (optional)

**Your API will be live at:** `https://moverse.rydlr.com` 🚀
