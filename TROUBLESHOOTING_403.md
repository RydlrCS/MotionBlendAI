# TROUBLESHOOTING 403 FORBIDDEN ERROR

## Quick Fix Steps

### 1. Fix Directory Permissions

```bash
# Set correct ownership
sudo chown -R rydlr:rydlr /home/rydlr/public_html/motionblend

# Set directory permissions (755)
find /home/rydlr/public_html/motionblend -type d -exec chmod 755 {} \;

# Set file permissions (644)
find /home/rydlr/public_html/motionblend -type f -exec chmod 644 {} \;

# Make setup.sh executable
chmod +x /home/rydlr/public_html/motionblend/api/setup.sh
```

### 2. Upload .htaccess File

Place this `.htaccess` in `/home/rydlr/public_html/motionblend/`:

```apache
# Enable RewriteEngine
RewriteEngine On

# Proxy API requests to the Python backend
RewriteCond %{REQUEST_URI} ^/motionblend(/.*)?$
RewriteRule ^(.*)$ http://localhost:8080/$1 [P,L]

# CORS Headers
<IfModule mod_headers.c>
    Header always set Access-Control-Allow-Origin "*"
    Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
    Header always set Access-Control-Max-Age "3600"
</IfModule>

# Handle OPTIONS preflight
RewriteCond %{REQUEST_METHOD} OPTIONS
RewriteRule ^(.*)$ $1 [R=200,L]

# Allow directory access
Options -Indexes +FollowSymLinks
Require all granted
```

### 3. Update Apache Virtual Host Configuration

In Virtualmin: **Server Configuration → Website Options** or edit Apache config directly:

```apache
<VirtualHost *:443>
    ServerName www.rydlr.com
    DocumentRoot /home/rydlr/public_html
    
    # Enable proxy modules
    ProxyRequests Off
    ProxyPreserveHost On
    
    # Proxy /motionblend to Python API
    <Location /motionblend>
        ProxyPass http://localhost:8080
        ProxyPassReverse http://localhost:8080
        
        # CORS headers
        Header always set Access-Control-Allow-Origin "*"
        Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
        
        # Allow access
        Require all granted
    </Location>
    
    # SSL configuration
    SSLEngine on
    SSLCertificateFile /path/to/cert.pem
    SSLCertificateKeyFile /path/to/key.pem
</VirtualHost>
```

### 4. Enable Required Apache Modules

```bash
# Enable modules
sudo a2enmod proxy
sudo a2enmod proxy_http
sudo a2enmod headers
sudo a2enmod rewrite
sudo a2enmod ssl

# Restart Apache
sudo systemctl restart apache2
```

### 5. Check SELinux (if enabled)

```bash
# Check if SELinux is enforcing
getenforce

# If enabled, allow Apache to make network connections
sudo setsebool -P httpd_can_network_connect 1

# Allow Apache to access the directory
sudo chcon -R -t httpd_sys_content_t /home/rydlr/public_html/motionblend
```

### 6. Verify API Server is Running

```bash
# Check if API is running on port 8080
sudo netstat -tlnp | grep 8080

# Or with ss
sudo ss -tlnp | grep 8080

# Check service status
sudo systemctl status motionblend-api

# If not running, start it
sudo systemctl start motionblend-api

# Test local connection
curl http://localhost:8080/health
```

### 7. Test Direct Access

```bash
# Test without Apache proxy
curl http://localhost:8080/health

# Should return:
# {"status": "healthy", "timestamp": "..."}
```

### 8. Check Apache Error Logs

```bash
# View recent errors
sudo tail -f /var/log/apache2/error.log

# Check access log
sudo tail -f /var/log/apache2/access.log

# Look for specific errors
sudo grep -i "motionblend" /var/log/apache2/error.log
```

## Common Issues and Solutions

### Issue: "You don't have permission to access this resource"

**Cause:** Apache can't read the directory or .htaccess file

**Solution:**
```bash
# Fix ownership
sudo chown -R rydlr:www-data /home/rydlr/public_html/motionblend

# Fix permissions
sudo chmod 755 /home/rydlr/public_html/motionblend
sudo chmod 644 /home/rydlr/public_html/motionblend/.htaccess

# Allow Apache to override settings
# In Apache config, ensure AllowOverride is set:
<Directory /home/rydlr/public_html>
    AllowOverride All
</Directory>
```

### Issue: Proxy not working

**Cause:** mod_proxy not enabled or API not running

**Solution:**
```bash
# Enable proxy modules
sudo a2enmod proxy proxy_http

# Restart Apache
sudo systemctl restart apache2

# Ensure API is running
sudo systemctl start motionblend-api
```

### Issue: CORS errors in browser

**Cause:** CORS headers not being set

**Solution:**
```bash
# Enable headers module
sudo a2enmod headers

# Restart Apache
sudo systemctl restart apache2

# Verify headers are sent
curl -I https://www.rydlr.com/motionblend/health
```

## Alternative: Direct Apache Configuration (No .htaccess)

If `.htaccess` isn't working, add this directly to your Apache virtual host config:

**Edit:** `/etc/apache2/sites-available/www.rydlr.com.conf` (or via Virtualmin)

```apache
<VirtualHost *:443>
    ServerName www.rydlr.com
    DocumentRoot /home/rydlr/public_html
    
    # Proxy configuration for /motionblend
    ProxyRequests Off
    ProxyPreserveHost On
    ProxyPass /motionblend http://localhost:8080
    ProxyPassReverse /motionblend http://localhost:8080
    
    <Location /motionblend>
        # Allow access
        Require all granted
        
        # CORS headers
        Header always set Access-Control-Allow-Origin "*"
        Header always set Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        Header always set Access-Control-Allow-Headers "Content-Type, Authorization, X-Requested-With"
        Header always set Access-Control-Max-Age "3600"
    </Location>
    
    # SSL configuration
    SSLEngine on
    SSLCertificateFile /etc/ssl/certs/ssl-cert-snakeoil.pem
    SSLCertificateKeyFile /etc/ssl/private/ssl-cert-snakeoil.key
</VirtualHost>
```

Then restart:
```bash
sudo apache2ctl configtest
sudo systemctl restart apache2
```

## Virtualmin-Specific Steps

### Via Virtualmin Web Interface:

1. **Go to:** Virtualmin → www.rydlr.com → Server Configuration → Edit Directives
2. **Add before `</VirtualHost>`:**
   ```apache
   ProxyPass /motionblend http://localhost:8080
   ProxyPassReverse /motionblend http://localhost:8080
   
   <Location /motionblend>
       Require all granted
       Header always set Access-Control-Allow-Origin "*"
   </Location>
   ```
3. **Click:** Save
4. **Click:** Apply Configuration

### Via File Manager:

1. **Navigate to:** `/home/rydlr/public_html/motionblend/`
2. **Upload:** `.htaccess` file (download from GitHub)
3. **Set permissions:** 644
4. **Set owner:** rydlr

## Testing Checklist

After applying fixes:

```bash
# 1. Test API directly
curl http://localhost:8080/health

# 2. Test through Apache
curl https://www.rydlr.com/motionblend/health

# 3. Test with verbose output
curl -v https://www.rydlr.com/motionblend/health

# 4. Check Apache can reach backend
curl -v http://localhost:8080/health

# 5. Test CORS headers
curl -H "Origin: https://example.com" \
     -H "Access-Control-Request-Method: POST" \
     -H "Access-Control-Request-Headers: Content-Type" \
     -X OPTIONS \
     https://www.rydlr.com/motionblend/health
```

## Quick Diagnostic Script

Save and run this script to diagnose issues:

```bash
#!/bin/bash
echo "=== MotionBlend Diagnostics ==="

echo -e "\n1. Checking API service..."
systemctl status motionblend-api --no-pager

echo -e "\n2. Checking port 8080..."
netstat -tlnp | grep 8080 || echo "Port 8080 not listening!"

echo -e "\n3. Testing direct API access..."
curl -s http://localhost:8080/health || echo "API not responding!"

echo -e "\n4. Checking directory permissions..."
ls -la /home/rydlr/public_html/motionblend/

echo -e "\n5. Checking Apache modules..."
apache2ctl -M | grep -E "(proxy|headers|rewrite)"

echo -e "\n6. Testing through Apache..."
curl -s https://www.rydlr.com/motionblend/health || echo "Apache proxy not working!"

echo -e "\n7. Recent Apache errors..."
tail -20 /var/log/apache2/error.log | grep -i motionblend
```

## If All Else Fails

**Simplest working configuration:**

1. **Stop using .htaccess**, use direct Apache config
2. **Add to Apache virtualhost:**
   ```apache
   ProxyPass /motionblend http://localhost:8080
   ProxyPassReverse /motionblend http://localhost:8080
   ```
3. **Restart Apache:** `sudo systemctl restart apache2`
4. **Ensure API is running:** `sudo systemctl start motionblend-api`
5. **Test:** `curl https://www.rydlr.com/motionblend/health`

## Need Help?

Check logs in this order:
1. `/var/log/apache2/error.log` - Apache errors
2. `sudo journalctl -u motionblend-api` - API service logs
3. `sudo journalctl -u apache2` - Apache service logs
