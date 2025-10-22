# QUICK FIX: Use .htaccess Instead

## Step 1: Upload .htaccess to Main Domain

**Upload this file to:** `/home/rydlr/public_html/.htaccess`

**Download from:**
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess
```

**Or create it manually with this content:**

```apache
RewriteEngine On

# Proxy /motionblend requests to Python API
RewriteCond %{REQUEST_URI} ^/motionblend(/.*)?$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

# CORS Headers for /motionblend requests
<IfModule mod_headers.c>
    Header always set Access-Control-Allow-Origin "*" env=REQUEST_URI
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" env=REQUEST_URI
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With" env=REQUEST_URI
</IfModule>

# Handle OPTIONS preflight
RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^motionblend(/.*)?$ - [R=200,L]

# Allow directory access
Options -Indexes +FollowSymLinks
Require all granted
```

## Step 2: Enable Apache Modules

```bash
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod headers
sudo a2enmod rewrite
sudo systemctl restart apache2
```

## Step 3: Ensure API is Running

```bash
# Check if API is running
sudo systemctl status motionblend-api

# If not running, start it
sudo systemctl start motionblend-api

# Test local access
curl http://localhost:8080/health
```

## Step 4: Test

```bash
# Test the API through Apache
curl https://www.rydlr.com/motionblend/health

# Should return: {"status": "healthy", "timestamp": "..."}
```

## Why This Works

- Uses `.htaccess` in the main domain's document root
- No need to edit Apache virtual host configuration
- Virtualmin will respect the `.htaccess` file
- All `/motionblend/*` requests get proxied to your Python API

## If Still Getting Errors

1. **Check permissions:**
   ```bash
   sudo chown rydlr:www-data /home/rydlr/public_html/.htaccess
   sudo chmod 644 /home/rydlr/public_html/.htaccess
   ```

2. **Verify AllowOverride is enabled** (should be by default in Virtualmin)

3. **Check Apache error log:**
   ```bash
   sudo tail -20 /var/log/apache2/error.log
   ```

4. **Test without .htaccess** temporarily by renaming it:
   ```bash
   mv /home/rydlr/public_html/.htaccess /home/rydlr/public_html/.htaccess.backup
   ```

This approach is simpler and doesn't require editing Apache configuration files directly.