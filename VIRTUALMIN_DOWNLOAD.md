# Quick Deployment via Virtualmin Download

## Step 1: Download Deployment Package to Server

In Virtualmin, use the **Download from remote URL** feature:

### Option A: Download from GitHub (Direct)

```
URL: https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
```

Download each file individually:
1. `api_server.py`
2. `requirements-api.txt`
3. `setup.sh`

### Option B: Use wget/curl in Virtualmin Command Shell

1. **In Virtualmin:** System Settings → Login to Webmin → Others → Command Shell

2. **Run these commands:**

```bash
# Navigate to your domain directory
cd /home/rydlr/domains/moverse.rydlr.com

# Create api directory
mkdir -p api
cd api

# Download files from GitHub
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh

# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh
```

## Step 2: Upload Google Cloud Credentials

1. **Create credentials directory:**
   ```bash
   mkdir -p /home/rydlr/domains/moverse.rydlr.com/api/credentials
   ```

2. **Upload via Virtualmin File Manager:**
   - Go to Tools → File Manager
   - Navigate to `/home/rydlr/domains/moverse.rydlr.com/api/credentials/`
   - Click Upload
   - Upload your file: `application_default_credentials.json`

3. **Or download from your Mac:**

   On your Mac, start a simple HTTP server:
   ```bash
   cd ~/.config/gcloud
   python3 -m http.server 8000
   ```

   Then on server:
   ```bash
   cd /home/rydlr/domains/moverse.rydlr.com/api/credentials/
   wget http://YOUR_MAC_IP:8000/application_default_credentials.json
   ```

   Stop the server on your Mac after download.

## Step 3: Test the API

```bash
cd /home/rydlr/domains/moverse.rydlr.com/api
source venv/bin/activate
python3 api_server.py
```

In another terminal/tab:
```bash
curl http://127.0.0.1:8080/health
```

## Step 4: Configure Production Service

Follow the systemd service setup in `MANUAL_DEPLOYMENT.md`

---

## Alternative: Direct File URLs

You can use these direct URLs in Virtualmin's download feature:

### Main API Server
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
```

### Requirements File
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
```

### Setup Script
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh
```

### Manual Deployment Guide
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/MANUAL_DEPLOYMENT.md
```

---

## Using Virtualmin's "Download from remote URL"

1. **Login to Virtualmin**
2. **Select moverse.rydlr.com**
3. **Go to: Tools → File Manager**
4. **Navigate to: `/home/rydlr/domains/moverse.rydlr.com/`**
5. **Click: New → Download from URL**
6. **Enter URL** (from list above)
7. **Click Download**
8. **Repeat** for each file

Then run the setup script via Command Shell:
```bash
cd /home/rydlr/domains/moverse.rydlr.com/api
chmod +x setup.sh
./setup.sh
```

---

## Quick Start Commands (Copy-Paste Ready)

```bash
# Create directory structure
mkdir -p /home/rydlr/domains/moverse.rydlr.com/api/credentials
cd /home/rydlr/domains/moverse.rydlr.com/api

# Download files
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
curl -O https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh

# Run setup
chmod +x setup.sh
./setup.sh

# Test
source venv/bin/activate
python3 api_server.py
```

🚀 Done! Your API server is ready to configure for production.
