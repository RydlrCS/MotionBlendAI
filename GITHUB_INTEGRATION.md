# GitHub Integration & Auto-Deployment

## Direct File URLs for Repository

Use these URLs to download files directly into your repository or server:

### API Server Files
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/deploy-manual.sh
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/deploy-ftp.sh
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/moverse-api.service
```

### Apache Configuration Files
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/apache-subdomain.conf
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain
```

### UI Files (Built for Production)
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-08f65cc5.css
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-0d9f8f93.js
```

### Deployment Guides
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/SUBDOMAIN_DEPLOYMENT.md
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/UI_DEPLOYMENT.md
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/VIRTUALMIN_SUBDOMAIN_SETUP.md
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/TROUBLESHOOTING_403.md
```

### Cloud Build Configuration
```
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/cloudbuild.yaml
https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/Dockerfile.api
```

## Automated Deployment Setup

### Option 1: GitHub Actions (Recommended)

Create `.github/workflows/deploy.yml` in your repository:

```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Setup SSH
      uses: webfactory/ssh-agent@v0.8.0
      with:
          ssh-private-key: ${{ secrets.SSH_PRIVATE_KEY }}
          
    - name: Deploy API Server
      run: |
        # Create API directory
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "mkdir -p /home/rydlr/domains/moverse.rydlr.com/motionblend-api"
        
        # Download and deploy API files
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api && wget -O api_server.py https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py"
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api && wget -O requirements-api.txt https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt"
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api && wget -O setup.sh https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/setup.sh && chmod +x setup.sh"
        
        # Create environment file
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "cat > /home/rydlr/domains/moverse.rydlr.com/motionblend-api/.env << 'EOF'
GCS_BUCKET=motionblend-mocap
BQ_PROJECT=${{ secrets.BQ_PROJECT }}
BQ_DATASET=RAW_DEV
ELASTICSEARCH_URL=https://elasticsearch-motionblend-ba986d.es.us-central1.gcp.elastic.cloud
ES_API_KEY=${{ secrets.ES_API_KEY }}
ES_INDEX=mb_blends_v1
FLASK_ENV=production
PORT=8080
EOF"
        
        # Setup Python environment
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "cd /home/rydlr/domains/moverse.rydlr.com/motionblend-api && ./setup.sh"
        
    - name: Deploy UI Files
      run: |
        # Download UI files
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "mkdir -p /home/rydlr/domains/moverse.rydlr.com/public_html/assets"
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "wget -O /home/rydlr/domains/moverse.rydlr.com/public_html/index.html https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html"
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "wget -O /home/rydlr/domains/moverse.rydlr.com/public_html/assets/index-5dea55ab.css https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-5dea55ab.css"
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "wget -O /home/rydlr/domains/moverse.rydlr.com/public_html/assets/index-ac34afc1.js https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-ac34afc1.js"
        
    - name: Deploy Apache Configuration
      run: |
        # Download .htaccess
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "wget -O /home/rydlr/domains/moverse.rydlr.com/public_html/.htaccess https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain"
        
    - name: Restart Services
      run: |
        # Restart API service
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "sudo systemctl restart moverse-api"
        
        # Reload Apache
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "sudo systemctl reload apache2"
        
    - name: Health Check
      run: |
        # Wait for services to start
        sleep 10
        
        # Test API health
        ssh -o StrictHostKeyChecking=no ${{ secrets.SERVER_USER }}@${{ secrets.SERVER_HOST }} "curl -f http://localhost:8080/health"
        
        # Test UI accessibility
        curl -f https://moverse.rydlr.com/
```

### Option 2: Webhook-Based Deployment

Create a webhook endpoint on your server that listens for GitHub push events.

#### 1. Create Webhook Handler

Create `/home/rydlr/domains/moverse.rydlr.com/webhook-deploy.php`:

```php
<?php
// GitHub Webhook Secret (set this in GitHub webhook settings)
$secret = 'your-webhook-secret-here';

// Get payload
$payload = file_get_contents('php://input');
$signature = $_SERVER['HTTP_X_HUB_SIGNATURE'] ?? '';

// Verify signature
if (!verifySignature($payload, $signature, $secret)) {
    http_response_code(401);
    die('Invalid signature');
}

// Only process push events to main branch
$data = json_decode($payload, true);
if ($data['ref'] !== 'refs/heads/main') {
    die('Not main branch');
}

// Deploy
exec('/home/rydlr/domains/moverse.rydlr.com/deploy.sh > /dev/null 2>&1 &');
echo 'Deployment started';

function verifySignature($payload, $signature, $secret) {
    $expected = 'sha1=' . hash_hmac('sha1', $payload, $secret);
    return hash_equals($expected, $signature);
}
?>
```

#### 2. Create Deployment Script

Create `/home/rydlr/domains/moverse.rydlr.com/deploy.sh`:

```bash
#!/bin/bash
cd /home/rydlr/domains/moverse.rydlr.com

echo "Starting deployment..."

# Update API files
cd motionblend-api
wget -q -O api_server.py https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
wget -q -O requirements-api.txt https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt

# Update UI files
cd ../public_html
wget -q -O index.html https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html
mkdir -p assets
wget -q -O assets/index-5dea55ab.css https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-5dea55ab.css
wget -q -O assets/index-ac34afc1.js https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-ac34afc1.js

# Update .htaccess
wget -q -O .htaccess https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain

# Restart services
sudo systemctl restart moverse-api
sudo systemctl reload apache2

echo "Deployment completed"
```

#### 3. Configure GitHub Webhook

1. Go to your GitHub repository → Settings → Webhooks
2. Add webhook:
   - **Payload URL:** `https://moverse.rydlr.com/webhook-deploy.php`
   - **Content type:** `application/json`
   - **Secret:** `your-webhook-secret-here`
   - **Events:** Just the `push` event
   - **Active:** ✅

### Option 3: FTP Deployment

For FTP-based deployment (useful when SSH access is not available):

#### FTP Deployment Script

```bash
#!/bin/bash
# deploy-ftp.sh - FTP-based deployment

# FTP Configuration (configure these)
FTP_HOST="rydlr-ftp.moverse"
FTP_USER="rydlr-ftp"
FTP_PASS="your-ftp-password"

# Download latest files
wget -q -O api_server.py https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/api_server.py
wget -q -O requirements-api.txt https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/requirements-api.txt
wget -q -O index.html https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/index.html
wget -q -O index-08f65cc5.css https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-08f65cc5.css
wget -q -O index-0d9f8f93.js https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/ui/dist/assets/index-0d9f8f93.js
wget -q -O .htaccess https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/.htaccess-subdomain

# Create FTP batch file
cat > ftp_upload.txt << EOF
open $FTP_HOST
user $FTP_USER $FTP_PASS
binary

# Upload API files
mkdir motionblend-api
cd motionblend-api
put api_server.py
put requirements-api.txt
cd ..

# Upload UI files
cd public_html
put index.html
mkdir assets
cd assets
put index-08f65cc5.css
put index-0d9f8f93.js
cd ..
put .htaccess

bye
EOF

# Execute FTP upload
ftp -n < ftp_upload.txt

# Cleanup
rm ftp_upload.txt

echo "FTP deployment completed!"
```

#### Usage
```bash
# Download and run the FTP deployment script
wget https://raw.githubusercontent.com/RydlrCS/MotionBlendAI/main/deploy-ftp.sh
chmod +x deploy-ftp.sh

# Edit FTP credentials in the script
nano deploy-ftp.sh

# Run deployment
./deploy-ftp.sh
```

**Note:** FTP deployment requires manual credential configuration and doesn't restart services automatically. Use this as a fallback when SSH is not available.

## GitHub Secrets Required

For GitHub Actions, add these secrets to your repository:

- `SSH_PRIVATE_KEY`: Your server's SSH private key (generate with `ssh-keygen`)
- `SERVER_HOST`: `rydlr.moverse` (your SSH hostname)
- `SERVER_USER`: `rydlr` (your SSH username)
- `BQ_PROJECT`: Your BigQuery project ID (`motionblend-ai`)
- `ES_API_KEY`: Your Elasticsearch API key

### Setting up SSH Access

1. **Generate SSH Key Pair** (on your local machine):
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions@rydlr.moverse"
# Save as ~/.ssh/github_actions_rsa (don't use default name)
```

2. **Add Public Key to Server**:
```bash
# Copy public key to server
ssh-copy-id -i ~/.ssh/github_actions_rsa.pub rydlr@rydlr.moverse

# Or manually append to ~/.ssh/authorized_keys on server
cat ~/.ssh/github_actions_rsa.pub >> ~/.ssh/authorized_keys
```

3. **Add Private Key to GitHub Secrets**:
   - Go to your GitHub repository → Settings → Secrets and variables → Actions
   - Add new secret named `SSH_PRIVATE_KEY`
   - Paste the entire private key content (including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`)

4. **Test SSH Connection**:
```bash
ssh -T rydlr@rydlr.moverse
# Should connect without password prompt
```

## Testing Auto-Deployment

### Test GitHub Actions
1. Push a change to the `main` branch
2. Check Actions tab for deployment status
3. Monitor server logs during deployment

### Test Webhook
1. Make a commit and push to main
2. Check webhook delivery in GitHub
3. Verify files were updated on server

### Test Manual Deployment
```bash
chmod +x deploy-manual.sh
./deploy-manual.sh
```

## File Integrity Verification

To verify files were downloaded correctly:

```bash
# Check file sizes
ls -lh /home/rydlr/domains/moverse.rydlr.com/public_html/
ls -lh /home/rydlr/domains/moverse.rydlr.com/motionblend-api/

# Test API
curl http://localhost:8080/health

# Test UI
curl -I https://moverse.rydlr.com/
```

## Rollback Strategy

If deployment fails:

```bash
# Quick rollback (if you have backups)
cp /home/rydlr/domains/moverse.rydlr.com/backup/api_server.py /home/rydlr/domains/moverse.rydlr.com/motionblend-api/
sudo systemctl restart moverse-api
```

## Monitoring

Set up monitoring for your deployment:

```bash
# Check deployment status
curl https://moverse.rydlr.com/health
curl https://moverse.rydlr.com/status

# Monitor logs
sudo journalctl -u moverse-api -f
sudo tail -f /var/log/apache2/error.log
```

## Security Considerations

- Use strong webhook secrets
- Limit SSH key permissions
- Regularly rotate API keys
- Monitor deployment logs
- Use HTTPS for all webhook communications

This setup ensures your website automatically stays in sync with your GitHub repository! 🚀