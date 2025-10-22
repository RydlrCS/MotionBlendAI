# 🚀 Deploy to moverse.rydlr.com - Quick Guide

## Using Virtualmin "Download from remote URL"

### Step 1: Download Files

1. **Login to Virtualmin:** https://rydlr.com:10000
2. **Select:** `moverse.rydlr.com`
3. **Go to:** Tools → File Manager
4. **Navigate to:** `/home/rydlr/domains/moverse.rydlr.com/`
5. **Click:** New → Download from URL

**Download these 3 files:**

**File 1 - API Server:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
```

**File 2 - Requirements:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
```

**File 3 - Setup Script:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh
```

---

### Step 2: Run Setup

1. **Open Command Shell:** System Settings → Login to Webmin → Others → Command Shell

2. **Copy and paste these commands:**

```bash
cd /home/rydlr/domains/moverse.rydlr.com
mkdir -p api/credentials
cd api
chmod +x setup.sh
./setup.sh
```

**Wait for setup to complete** (installs Flask, gunicorn, etc.)

---

### Step 3: Upload Credentials

**Upload your Google Cloud credentials file:**

1. In File Manager, navigate to: `/home/rydlr/domains/moverse.rydlr.com/api/credentials/`
2. Click **Upload to current directory**
3. Upload from Mac: `~/.config/gcloud/application_default_credentials.json`
4. Make sure it's named: `application_default_credentials.json`

---

### Step 4: Test Locally

**In Command Shell:**

```bash
cd /home/rydlr/domains/moverse.rydlr.com/api
source venv/bin/activate
python3 api_server.py &
```

**Test it works:**
```bash
curl http://127.0.0.1:8080/health
```

You should see:
```json
{"status": "healthy", "service": "MotionBlend AI API", ...}
```

Press Ctrl+C to stop the test server.

---

### Step 5: Configure Apache

1. **In Virtualmin:** Select `moverse.rydlr.com`
2. **Go to:** Services → Configure Website
3. **Click:** Edit Directives
4. **Find the `<VirtualHost *:443>` section**
5. **Add these lines inside it:**

```apache
# Reverse proxy to API
ProxyPreserveHost On
ProxyPass / http://127.0.0.1:8080/
ProxyPassReverse / http://127.0.0.1:8080/

# CORS headers
Header always set Access-Control-Allow-Origin "*"
Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
Header always set Access-Control-Allow-Headers "Content-Type, Authorization"
```

6. **Click:** Save
7. **Restart Apache**

---

### Step 6: Enable Apache Modules

**In Command Shell:**

```bash
sudo a2enmod proxy proxy_http headers ssl
sudo systemctl restart apache2
```

---

### Step 7: Create Production Service

**Create the service file:**

```bash
sudo nano /etc/systemd/system/moverse-api.service
```

**Paste this content:**

```ini
[Unit]
Description=Moverse MotionBlend API
After=network.target

[Service]
Type=simple
User=rydlr
Group=rydlr
WorkingDirectory=/home/rydlr/domains/moverse.rydlr.com/api
Environment="GCS_BUCKET=motionblend-mocap"
Environment="BQ_PROJECT=motionblend-ai"
Environment="BQ_DATASET=RAW_DEV"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/rydlr/domains/moverse.rydlr.com/api/credentials/application_default_credentials.json"
ExecStart=/home/rydlr/domains/moverse.rydlr.com/api/venv/bin/gunicorn --bind 127.0.0.1:8080 --workers 4 api_server:app
Restart=always

[Install]
WantedBy=multi-user.target
```

**Save:** Ctrl+X, Y, Enter

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable moverse-api
sudo systemctl start moverse-api
sudo systemctl status moverse-api
```

---

### Step 8: Test Production

**From anywhere (your Mac):**

```bash
curl https://moverse.rydlr.com/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "service": "MotionBlend AI API",
  "version": "1.0.0"
}
```

---

## ✅ Done!

Your API is now live at: **https://moverse.rydlr.com**

### Test All Endpoints:

```bash
# Health check
curl https://moverse.rydlr.com/health

# List motions
curl https://moverse.rydlr.com/motions

# Create blend
curl -X POST https://moverse.rydlr.com/api/blend \
  -H "Content-Type: application/json" \
  -d '{"motion1":"walk","motion2":"run","weight":0.5}'
```

---

## Troubleshooting

### Can't download files?
Use curl in Command Shell:
```bash
cd /home/rydlr/domains/moverse.rydlr.com/api
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh
```

### Service won't start?
Check logs:
```bash
sudo journalctl -u moverse-api -n 50
```

### 502 Bad Gateway?
Check if API is running:
```bash
sudo systemctl status moverse-api
curl http://127.0.0.1:8080/health
```

---

## Support

Need more details? See:
- **MANUAL_DEPLOYMENT.md** - Full manual guide
- **VIRTUALMIN_DEPLOYMENT.md** - Complete Virtualmin instructions

🎉 Your MotionBlend API is live!
