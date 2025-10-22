# MotionBlend AI - Production Deployment Guide

## Quick Start (Local Testing)

### 1. Test the API Locally

```bash
# Install dependencies
pip install flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch

# Set environment variables
export GCS_BUCKET="motionblend-mocap"
export BQ_PROJECT="motionblend-ai"
export BQ_DATASET="RAW_DEV"
export ELASTICSEARCH_URL="http://localhost:9200"

# Run the server
python api_server.py
```

Server will start at `http://localhost:8080`

### 2. Test the UI Connection

```bash
cd ui

# Update .env for local testing
echo "VITE_API_URL=http://localhost:8080" > .env

# Install and run
npm install
npm run dev
```

UI will start at `http://localhost:5173`

---

## Production Deployment to rydlr.com/motionblend

### Option 1: Deploy to Google Cloud Run (Recommended)

**Benefits:**
- Automatic HTTPS
- Auto-scaling
- Pay-per-use pricing
- Easy deployment

**Steps:**

1. **Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api_server.py .

# Expose port
EXPOSE 8080

# Run the server
CMD ["python", "api_server.py"]
```

2. **Create requirements.txt**

```txt
flask==3.0.0
flask-cors==4.0.0
google-cloud-storage==2.14.0
google-cloud-bigquery==3.14.0
elasticsearch==8.11.0
gunicorn==21.2.0
```

3. **Build and Deploy**

```bash
# Set project
gcloud config set project motionblend-ai

# Build container
gcloud builds submit --tag gcr.io/motionblend-ai/api-server

# Deploy to Cloud Run
gcloud run deploy motionblend-api \
  --image gcr.io/motionblend-ai/api-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GCS_BUCKET=motionblend-mocap,BQ_PROJECT=motionblend-ai,BQ_DATASET=RAW_DEV \
  --memory 1Gi \
  --cpu 1 \
  --max-instances 10
```

4. **Configure Custom Domain**

```bash
# Map to your domain
gcloud run domain-mappings create \
  --service motionblend-api \
  --domain www.rydlr.com \
  --region us-central1

# Follow the DNS instructions to add records
```

5. **Update DNS**

Add the following DNS records at your domain registrar:

```
Type: CNAME
Name: www.rydlr.com
Value: ghs.googlehosted.com
```

6. **Set up Path-based Routing**

If you want `/motionblend/*` path, use Cloud Load Balancer:

```bash
# Create backend service
gcloud compute backend-services create motionblend-backend \
  --global \
  --enable-cdn

# Create URL map
gcloud compute url-maps create rydlr-urlmap \
  --default-service=motionblend-backend

# Add path rule
gcloud compute url-maps add-path-matcher rydlr-urlmap \
  --path-matcher-name=motionblend-matcher \
  --default-service=motionblend-backend \
  --path-rules="/motionblend/*=motionblend-backend"
```

---

### Option 2: Deploy to Existing Apache Server

If you already have Apache running at `rydlr.com`, add a reverse proxy:

1. **Run API as a Service**

Create `/etc/systemd/system/motionblend-api.service`:

```ini
[Unit]
Description=MotionBlend AI API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/motionblend
Environment="GCS_BUCKET=motionblend-mocap"
Environment="BQ_PROJECT=motionblend-ai"
Environment="BQ_DATASET=RAW_DEV"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/var/www/motionblend/credentials.json"
ExecStart=/usr/bin/python3 /var/www/motionblend/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

2. **Enable and Start Service**

```bash
sudo systemctl daemon-reload
sudo systemctl enable motionblend-api
sudo systemctl start motionblend-api
sudo systemctl status motionblend-api
```

3. **Configure Apache Reverse Proxy**

Add to your Apache config (e.g., `/etc/apache2/sites-available/rydlr.conf`):

```apache
<VirtualHost *:443>
    ServerName www.rydlr.com
    
    # SSL configuration
    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/rydlr.crt
    SSLCertificateKeyFile /etc/ssl/private/rydlr.key
    
    # Reverse proxy to API server
    ProxyPreserveHost On
    
    <Location /motionblend/>
        ProxyPass http://127.0.0.1:8080/
        ProxyPassReverse http://127.0.0.1:8080/
        
        # CORS headers
        Header always set Access-Control-Allow-Origin "*"
        Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        Header always set Access-Control-Allow-Headers "Content-Type, Authorization"
        Header always set Access-Control-Max-Age "86400"
    </Location>
    
    # Handle OPTIONS preflight
    <Location /motionblend/>
        <LimitExcept GET POST PUT DELETE>
            Require all granted
        </LimitExcept>
    </Location>
</VirtualHost>
```

4. **Enable Required Apache Modules**

```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod headers
sudo a2enmod ssl
sudo systemctl restart apache2
```

---

### Option 3: Deploy with Nginx

If using Nginx instead of Apache:

1. **Configure Nginx**

Create `/etc/nginx/sites-available/rydlr`:

```nginx
server {
    listen 443 ssl http2;
    server_name www.rydlr.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/rydlr.crt;
    ssl_certificate_key /etc/ssl/private/rydlr.key;
    
    # API reverse proxy
    location /motionblend/ {
        proxy_pass http://127.0.0.1:8080/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization' always;
        add_header 'Access-Control-Max-Age' '86400' always;
        
        # Handle OPTIONS preflight
        if ($request_method = 'OPTIONS') {
            return 204;
        }
    }
}
```

2. **Enable and Reload**

```bash
sudo ln -s /etc/nginx/sites-available/rydlr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Environment Variables

Set these on your production server:

```bash
# Required
export GCS_BUCKET="motionblend-mocap"
export BQ_PROJECT="motionblend-ai"
export BQ_DATASET="RAW_DEV"

# Optional
export ELASTICSEARCH_URL="http://localhost:9200"
export ES_INDEX="mb_blends_v1"

# Google Cloud Authentication
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

---

## Testing the Deployment

### 1. Test Health Endpoint

```bash
curl https://www.rydlr.com/motionblend/health
```

Expected:
```json
{
  "status": "healthy",
  "service": "MotionBlend AI API",
  "version": "1.0.0",
  "timestamp": "2025-10-22T14:00:00Z"
}
```

### 2. Test CORS

```bash
curl -I -X OPTIONS https://www.rydlr.com/motionblend/health \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET"
```

Expected headers:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
```

### 3. Test Motions Endpoint

```bash
curl https://www.rydlr.com/motionblend/motions
```

### 4. Test Blend Creation

```bash
curl -X POST https://www.rydlr.com/motionblend/api/blend \
  -H "Content-Type: application/json" \
  -d '{
    "motion1": "walking_forward",
    "motion2": "jump_landing",
    "weight": 0.5
  }'
```

---

## Update UI to Use Production Server

Once deployed, update your UI:

```bash
cd ui

# Update .env
echo "VITE_API_URL=https://www.rydlr.com/motionblend" > .env

# Rebuild
npm run build

# Deploy (choose one):
# - Vercel: vercel --prod
# - Netlify: netlify deploy --prod
# - GitHub Pages: npm run deploy
# - Cloud Storage: gsutil -m rsync -r dist/ gs://your-ui-bucket
```

---

## Monitoring & Logs

### Cloud Run Logs

```bash
gcloud run services logs read motionblend-api \
  --region us-central1 \
  --limit 50
```

### Systemd Logs

```bash
sudo journalctl -u motionblend-api -f
```

### Apache Logs

```bash
tail -f /var/log/apache2/error.log
tail -f /var/log/apache2/access.log
```

### Nginx Logs

```bash
tail -f /var/log/nginx/error.log
tail -f /var/log/nginx/access.log
```

---

## Security Best Practices

### 1. Restrict CORS Origins

In `api_server.py`, change:

```python
CORS(app, resources={
    r"/*": {
        "origins": ["https://your-ui-domain.com"],  # Specific domains only
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### 2. Add API Authentication

```python
from functools import wraps

API_KEY = os.environ.get('API_KEY', 'your-secret-key')

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('Authorization')
        if not key or key != f"Bearer {API_KEY}":
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/blend', methods=['POST'])
@require_api_key
def create_blend():
    # ...
```

### 3. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/blend', methods=['POST'])
@limiter.limit("10 per minute")
def create_blend():
    # ...
```

### 4. HTTPS Only

Force HTTPS redirects in your web server config.

---

## Troubleshooting

### Issue: 403 Forbidden

**Cause:** Server not configured properly or file permissions wrong

**Fix:**
```bash
sudo chown -R www-data:www-data /var/www/motionblend
sudo chmod -R 755 /var/www/motionblend
```

### Issue: CORS Errors

**Cause:** Missing CORS headers

**Fix:** Ensure proxy/web server adds CORS headers (see configs above)

### Issue: 502 Bad Gateway

**Cause:** API server not running

**Fix:**
```bash
sudo systemctl status motionblend-api
sudo systemctl restart motionblend-api
```

### Issue: Connection Timeout

**Cause:** Firewall blocking port 8080

**Fix:**
```bash
sudo ufw allow 8080/tcp
# Or use nginx/apache as reverse proxy (recommended)
```

---

## Cost Optimization

### Cloud Run

- Set min instances to 0 (scale to zero when idle)
- Set max instances based on expected traffic
- Use Cloud Scheduler to keep warm if needed

```bash
gcloud run services update motionblend-api \
  --min-instances 0 \
  --max-instances 3 \
  --region us-central1
```

### BigQuery

- Use clustering on frequently queried columns
- Set appropriate expiration on staging tables
- Use BI Engine for caching

### Cloud Storage

- Use Nearline/Coldline for archival data
- Set lifecycle policies to auto-delete old data

---

## Next Steps

1. ✅ Deploy API server to production
2. ✅ Configure DNS and HTTPS
3. ✅ Test all endpoints
4. ✅ Update UI with production URL
5. ✅ Deploy UI to hosting platform
6. ✅ Set up monitoring and alerts
7. ✅ Configure backups
8. ✅ Add authentication (optional)
9. ✅ Set up CI/CD pipeline

---

## Support

- **GitHub Issues:** https://github.com/RydlrCS/MotionBlendAI/issues
- **Documentation:** See `ui/SERVER_INTEGRATION.md`
- **API Docs:** https://www.rydlr.com/motionblend/
