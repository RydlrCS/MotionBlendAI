# Virtualmin Subdomain Configuration Guide

## Option 1: Manual Configuration (Recommended for API)

Since you're deploying a custom Python API, manual configuration is better than using Virtualmin's web app installer.

### Step 1: Access Virtualmin for Subdomain

1. **Login to Virtualmin**
2. **Select:** `moverse.rydlr.com` from the dropdown
3. **Go to:** Server Configuration → Website Options

### Step 2: Configure Document Root

**Current setup should be:**
- **Document Root:** `/home/rydlr/domains/moverse.rydlr.com/public_html`
- **This is correct** - keep it as is

### Step 3: Edit Apache Directives

**Navigate:** Server Configuration → Edit Directives

**Add this configuration** (before `</VirtualHost>`):

```apache
# Proxy all requests to Python API server
ProxyPass / http://localhost:8080/
ProxyPassReverse / http://localhost:8080/

# CORS Headers for API
Header always set Access-Control-Allow-Origin "*"
Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
Header always set Access-Control-Max-Age "3600"

# Handle OPTIONS preflight requests
RewriteEngine On
RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^(.*)$ $1 [R=200,L]

# Allow access to root
<Location />
    Require all granted
</Location>
```

**Click:** Save and Apply Configuration

### Step 4: Alternative .htaccess Method

Instead of editing Apache config, you can use `.htaccess`:

1. **Create:** `/home/rydlr/domains/moverse.rydlr.com/public_html/.htaccess`
2. **Content:**
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

## Option 2: Using Virtualmin Web Apps (Not Recommended)

If you wanted to install a standard web app instead of your custom API:

1. **Go to:** `moverse.rydlr.com` → Install Scripts
2. **Choose:** WordPress, Django, or Node.js
3. **Install Path:** `/home/rydlr/domains/moverse.rydlr.com/public_html`
4. **Database:** Create new MySQL/PostgreSQL database

But **this won't work for your Python Flask API** - you'd need to choose "Django" and then replace the Django app with your Flask app.

## Option 3: Virtualmin File Manager Deployment

**Recommended approach for your API:**

1. **Create directory structure:**
   ```
   /home/rydlr/domains/moverse.rydlr.com/
   ├── public_html/
   │   └── .htaccess          # Proxy configuration
   └── motionblend-api/       # Your API files
       ├── api_server.py
       ├── requirements-api.txt
       ├── setup.sh
       ├── .env
       └── venv/
   ```

2. **Upload files via Virtualmin File Manager:**
   - Go to `moverse.rydlr.com` → File Manager
   - Create `motionblend-api` folder
   - Upload your API files

3. **Run setup via Virtualmin Command Shell:**
   - Go to `moverse.rydlr.com` → Command Shell
   - Run: `cd motionblend-api && chmod +x setup.sh && ./setup.sh`

## Virtualmin Paths Summary

**For `moverse.rydlr.com` subdomain:**

| Path | Purpose | Permissions |
|------|---------|-------------|
| `/home/rydlr/domains/moverse.rydlr.com/public_html/` | Web root | `rydlr:www-data` |
| `/home/rydlr/domains/moverse.rydlr.com/cgi-bin/` | CGI scripts | `rydlr:www-data` |
| `/home/rydlr/domains/moverse.rydlr.com/logs/` | Apache logs | `rydlr:rydlr` |
| `/home/rydlr/domains/moverse.rydlr.com/motionblend-api/` | Your API | `rydlr:rydlr` |

## Quick Setup Commands

**Via Virtualmin Command Shell:**

```bash
# Create API directory
mkdir -p /home/rydlr/domains/moverse.rydlr.com/motionblend-api

# Download files
cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh

# Setup environment
chmod +x setup.sh
./setup.sh

# Create .htaccess in public_html
cat > /home/rydlr/domains/moverse.rydlr.com/public_html/.htaccess << 'EOF'
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
EOF
```

## Systemd Service Setup

**Create service file:**

```bash
# Via command shell
sudo tee /etc/systemd/system/moverse-api.service > /dev/null <<EOF
[Unit]
Description=MotionBlend API (moverse.rydlr.com)
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
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable moverse-api
sudo systemctl start moverse-api
```

## Testing

```bash
# Test API directly
curl http://localhost:8080/health

# Test through subdomain
curl https://moverse.rydlr.com/health

# Check service status
sudo systemctl status moverse-api
```

## Virtualmin Web Apps vs Manual

**Use Virtualmin Web Apps when:**
- Installing standard applications (WordPress, Joomla, etc.)
- Want automatic updates and management
- Need database setup and configuration

**Use Manual Configuration when:**
- Deploying custom applications (like your Python API)
- Need specific server configurations
- Want full control over the deployment

**For your MotionBlend API:** Use manual configuration with `.htaccess` proxy.