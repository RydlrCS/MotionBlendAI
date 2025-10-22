# Manual Deployment Guide for moverse.rydlr.com

## Your Current Setup

Your server uses **rssh (restricted shell)** which limits SSH commands. This means we need to deploy through Virtualmin's web interface or FTP/SFTP.

### DNS Configuration ✅
- **Domain:** moverse.rydlr.com
- **IP:** 102.219.23.35
- **SPF Record:** Enabled with moverse.rydlr.com allowed
- **DMARC:** Enabled (Quarantine policy)

---

## Deployment Method 1: Using Virtualmin File Manager (Easiest)

### Step 1: Access Virtualmin
1. Go to https://rydlr.com:10000
2. Login with your credentials
3. Select "moverse.rydlr.com" from the dropdown

### Step 2: Upload API Server Files

1. **Go to Files & Folders:**
   - Click **Tools → File Manager**
   - Navigate to `/home/rydlr/domains/moverse.rydlr.com/`
   - Create folder: `api`

2. **Upload Files:**
   - Click **Upload** button
   - Upload these files from your Mac:
     - `/Users/ted/blenderkit_data/MotionBlendAI-1/api_server.py`
     - `/Users/ted/blenderkit_data/MotionBlendAI-1/requirements-api.txt`

3. **Create credentials folder:**
   - Inside `api/` folder, create subfolder: `credentials`
   - Upload your Google Cloud credentials:
     - `~/.config/gcloud/application_default_credentials.json`
   - Rename it to: `application_default_credentials.json`

### Step 3: Install Python Dependencies

1. **Open Terminal in Virtualmin:**
   - Click **System Settings → Login to Webmin**
   - Go to **Others → Command Shell**

2. **Run these commands:**
   ```bash
   cd /home/rydlr/domains/moverse.rydlr.com/api
   python3 -m venv venv
   source venv/bin/activate
   pip install flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch gunicorn
   ```

### Step 4: Configure Systemd Service

1. **Create service file:**
   - In Virtualmin, go to **Webmin → System → Bootup and Shutdown**
   - Click **Create a new bootup and shutdown action**

2. **Or use SSH (if you have sudo access):**

   Create file `/etc/systemd/system/moverse-api.service`:
   ```ini
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
   ```

3. **Enable and start:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable moverse-api
   sudo systemctl start moverse-api
   ```

### Step 5: Configure Apache Reverse Proxy

1. **In Virtualmin:**
   - Select **moverse.rydlr.com**
   - Go to **Services → Configure Website**
   - Click **Edit Directives**

2. **Add this configuration inside `<VirtualHost *:443>` block:**

   ```apache
   # Enable required modules first (via command line or Webmin):
   # a2enmod proxy proxy_http headers ssl rewrite
   
   # Reverse proxy to API server
   ProxyPreserveHost On
   ProxyPass / http://127.0.0.1:8080/
   ProxyPassReverse / http://127.0.0.1:8080/
   
   # CORS headers
   Header always set Access-Control-Allow-Origin "*"
   Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
   Header always set Access-Control-Allow-Headers "Content-Type, Authorization"
   Header always set Access-Control-Max-Age "86400"
   
   # Handle OPTIONS preflight
   RewriteEngine On
   RewriteCond %{REQUEST_METHOD} OPTIONS
   RewriteRule ^(.*)$ $1 [R=204,L]
   ```

3. **Enable Apache modules (Terminal):**
   ```bash
   sudo a2enmod proxy proxy_http headers ssl rewrite
   sudo systemctl restart apache2
   ```

4. **Or in Webmin:**
   - Go to **Servers → Apache Webserver**
   - Click **Global Configuration → Configure Apache Modules**
   - Enable: `proxy`, `proxy_http`, `headers`, `ssl`, `rewrite`
   - Click **Enable Selected Modules**
   - Restart Apache

### Step 6: Configure SSL Certificate

1. **In Virtualmin:**
   - Select **moverse.rydlr.com**
   - Go to **Server Configuration → SSL Certificate**
   - Choose one:
     - **Let's Encrypt:** Click "Request Certificate"
     - **Existing:** Upload your certificate files

2. **Ensure HTTPS is enabled:**
   - Go to **Services → Configure Website**
   - Check "SSL website enabled"

---

## Deployment Method 2: Using SFTP/FTP

If you prefer FTP/SFTP:

1. **Connect with FTP client (FileZilla, Cyberduck):**
   - Host: `moverse.rydlr.com`
   - Username: `rydlr`
   - Password: (your password)
   - Port: 21 (FTP) or 22 (SFTP)

2. **Upload files to:**
   - `/home/rydlr/domains/moverse.rydlr.com/api/api_server.py`
   - `/home/rydlr/domains/moverse.rydlr.com/api/requirements-api.txt`
   - `/home/rydlr/domains/moverse.rydlr.com/api/credentials/application_default_credentials.json`

3. **Then continue with Steps 3-6 above**

---

## Deployment Method 3: Using Virtualmin Python Script

If your server has Python support through Virtualmin:

1. **Enable Python support:**
   - In Virtualmin, select **moverse.rydlr.com**
   - Go to **Services → Python Versions**
   - Enable Python 3.x

2. **Configure as Python app:**
   - Upload `api_server.py` to public_html or custom directory
   - Configure WSGI/Python execution
   - Set environment variables

---

## Testing the Deployment

Once deployed, test these URLs:

```bash
# From your Mac
curl https://moverse.rydlr.com/health

# Expected response:
{
  "status": "healthy",
  "service": "MotionBlend AI API",
  "version": "1.0.0",
  "timestamp": "2025-10-22T17:00:00Z"
}
```

### Test CORS:
```bash
curl -I -X OPTIONS https://moverse.rydlr.com/health \
  -H "Origin: http://localhost:5173" \
  -H "Access-Control-Request-Method: GET"

# Should see these headers:
# Access-Control-Allow-Origin: *
# Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
```

### Test All Endpoints:
```bash
# List motions
curl https://moverse.rydlr.com/motions

# Create blend
curl -X POST https://moverse.rydlr.com/api/blend \
  -H "Content-Type: application/json" \
  -d '{"motion1":"walk","motion2":"run","weight":0.5}'

# Get artifacts
curl https://moverse.rydlr.com/api/artifacts/manifest
```

---

## Troubleshooting

### Can't Access Virtualmin Terminal
- Use Webmin's Command Shell: **System Settings → Login to Webmin → Others → Command Shell**
- Or contact your hosting provider to enable SSH access

### Python Dependencies Won't Install
```bash
# Try with user install
pip install --user flask flask-cors google-cloud-storage google-cloud-bigquery elasticsearch gunicorn
```

### Service Won't Start
Check logs:
```bash
sudo journalctl -u moverse-api -n 50
# Or
tail -f /home/rydlr/domains/moverse.rydlr.com/logs/api-error.log
```

### 502 Bad Gateway
1. Check if API is running: `sudo systemctl status moverse-api`
2. Check if listening on 8080: `netstat -tlnp | grep 8080`
3. Test direct: `curl http://127.0.0.1:8080/health`

### CORS Errors
- Verify Apache modules are enabled: `apache2ctl -M | grep proxy`
- Check Apache config: `sudo apache2ctl configtest`
- Restart Apache: `sudo systemctl restart apache2`

---

## Alternative: Docker Deployment

If your server supports Docker:

1. **Build Docker image locally:**
   ```bash
   cd /Users/ted/blenderkit_data/MotionBlendAI-1
   docker build -f Dockerfile.api -t moverse-api .
   ```

2. **Save and transfer:**
   ```bash
   docker save moverse-api | gzip > moverse-api.tar.gz
   # Upload to server via FTP
   ```

3. **On server:**
   ```bash
   docker load < moverse-api.tar.gz
   docker run -d -p 8080:8080 \
     -e GCS_BUCKET=motionblend-mocap \
     -e BQ_PROJECT=motionblend-ai \
     -e BQ_DATASET=RAW_DEV \
     --name moverse-api \
     moverse-api
   ```

---

## Quick Reference

### Files to Upload
1. `api_server.py` → `/home/rydlr/domains/moverse.rydlr.com/api/`
2. `requirements-api.txt` → `/home/rydlr/domains/moverse.rydlr.com/api/`
3. `application_default_credentials.json` → `/home/rydlr/domains/moverse.rydlr.com/api/credentials/`

### Commands After Upload
```bash
cd /home/rydlr/domains/moverse.rydlr.com/api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-api.txt
```

### Test Locally on Server
```bash
cd /home/rydlr/domains/moverse.rydlr.com/api
source venv/bin/activate
python3 api_server.py
# Open another terminal and test:
curl http://127.0.0.1:8080/health
```

---

## Contact Hosting Support

If you encounter permission issues, contact your hosting provider and request:

1. **SSH access** with full shell (not rssh)
2. **Sudo privileges** for systemd service management
3. **Python 3.x** with pip and venv support
4. **Apache modules** enabled: proxy, proxy_http, headers, ssl, rewrite
5. **Port 8080** accessible locally (127.0.0.1)

---

## Next Steps After Deployment

1. ✅ Test API at https://moverse.rydlr.com/health
2. ✅ Deploy React UI:
   ```bash
   cd /Users/ted/blenderkit_data/MotionBlendAI-1/ui
   npm run build
   # Upload dist/ folder to server via FTP
   ```
3. ✅ Configure monitoring and backups
4. ✅ Set up SSL certificate auto-renewal

---

## Summary

Since your server uses **restricted SSH (rssh)**, you'll need to:

1. **Use Virtualmin's File Manager** to upload files
2. **Use Virtualmin's Command Shell** to run commands
3. **Configure Apache** through Virtualmin's web interface

The API server is ready to deploy - just follow the **Deployment Method 1** steps above using Virtualmin's web interface! 🚀
