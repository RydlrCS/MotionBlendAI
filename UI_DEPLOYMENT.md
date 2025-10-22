# UI Deployment Guide

## Overview

The MotionBlend UI has been built for production and is ready to be served. You have two options:

1. **Serve UI alongside API** on the subdomain (recommended)
2. **Serve UI separately** on a different path or domain

## Option 1: Serve UI on Subdomain (Recommended)

### Directory Structure

```
/home/rydlr/domains/moverse.rydlr.com/
├── public_html/
│   ├── .htaccess          # API proxy configuration
│   ├── index.html         # UI entry point
│   ├── assets/            # UI assets (CSS, JS)
│   │   ├── index-5dea55ab.css
│   │   └── index-ac34afc1.js
│   └── ui/                # (Optional) UI in subfolder
└── motionblend-api/       # API server files
    ├── api_server.py
    └── ...
```

### Step 1: Upload Built UI Files

Upload the contents of `/Users/ted/blenderkit_data/MotionBlendAI-1/ui/dist/` to:
```
/home/rydlr/domains/moverse.rydlr.com/public_html/
```

**Files to upload:**
- `index.html`
- `assets/index-5dea55ab.css`
- `assets/index-ac34afc1.js`

### Step 2: Update .htaccess for UI + API

Your current `.htaccess` proxies API requests but serves UI files directly:

```apache
RewriteEngine On

# Proxy API requests to Python backend
RewriteCond %{REQUEST_URI} ^/api(/.*)?$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

RewriteCond %{REQUEST_URI} ^/search(/.*)?$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

RewriteCond %{REQUEST_URI} ^/motions$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

RewriteCond %{REQUEST_URI} ^/health$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

RewriteCond %{REQUEST_URI} ^/status$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

# CORS Headers for API responses
<IfModule mod_headers.c>
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
    Header always set Access-Control-Max-Age "3600"
</IfModule>

# Handle OPTIONS preflight requests
RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^(.*)$ - [R=200,L]

# Allow access to this directory
Options -Indexes +FollowSymLinks
Require all granted
```

### Step 3: Access Your Application

**UI:** `https://moverse.rydlr.com/`
**API:** `https://moverse.rydlr.com/api/*`

## Option 2: Serve UI in Subfolder

If you want to keep API at root and UI in `/ui/`:

### Directory Structure

```
/home/rydlr/domains/moverse.rydlr.com/public_html/
├── .htaccess              # API proxy for root
├── ui/
│   ├── index.html         # UI entry point
│   └── assets/            # UI assets
└── ...
```

### Update .htaccess

```apache
RewriteEngine On

# Proxy ALL requests to API except /ui/
RewriteCond %{REQUEST_URI} !^/ui/
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

# CORS Headers
<IfModule mod_headers.c>
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
</IfModule>

# Handle OPTIONS preflight
RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^(.*)$ - [R=200,L]

Options -Indexes +FollowSymLinks
Require all granted
```

### Access URLs

**UI:** `https://moverse.rydlr.com/ui/`
**API:** `https://moverse.rydlr.com/api/*`

## Testing UI Deployment

### Test UI Loading

```bash
# Test UI loads
curl -I https://moverse.rydlr.com/

# Should return 200 OK with HTML content-type
```

### Test API Still Works

```bash
# Test API endpoints
curl https://moverse.rydlr.com/health
curl https://moverse.rydlr.com/motions
curl https://moverse.rydlr.com/status
```

### Test CORS

```bash
# Test CORS headers
curl -H "Origin: https://example.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS \
     https://moverse.rydlr.com/api/blend
```

## UI Configuration

The UI is already configured to connect to:
- **API Base:** `https://moverse.rydlr.com`
- **Environment:** Production mode
- **Features:** All API endpoints enabled

## Built Files Summary

**Production Build Output:**
- `index.html` (494 bytes) - Main UI entry point
- `assets/index-5dea55ab.css` (54.52 kB) - Styles
- `assets/index-ac34afc1.js` (219.22 kB) - Application code

**Total Size:** ~274 kB (compressed: ~80 kB)

## Deployment Checklist

- [x] UI built for production
- [ ] UI files uploaded to server
- [ ] .htaccess configured for API proxy
- [ ] Apache modules enabled (proxy, headers, rewrite)
- [ ] API server running on port 8080
- [ ] SSL certificate configured
- [ ] DNS pointing to server
- [ ] UI loads at https://moverse.rydlr.com/
- [ ] API responds at https://moverse.rydlr.com/health

## Troubleshooting

### UI Not Loading

```bash
# Check file permissions
ls -la /home/rydlr/domains/moverse.rydlr.com/public_html/

# Check Apache error logs
sudo tail -20 /var/log/apache2/error.log
```

### API Not Working

```bash
# Check API server status
sudo systemctl status moverse-api

# Test local API
curl http://localhost:8080/health
```

### CORS Issues

```bash
# Verify CORS headers in .htaccess
grep -i "access-control" /home/rydlr/domains/moverse.rydlr.com/public_html/.htaccess
```

## Next Steps

1. Upload the built UI files to your server
2. Configure `.htaccess` for your chosen setup
3. Test both UI and API functionality
4. Update DNS if needed
5. Monitor logs for any issues

**Your MotionBlend UI is now production-ready!** 🎉