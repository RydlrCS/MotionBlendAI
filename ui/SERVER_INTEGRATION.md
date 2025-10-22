# MotionBlend UI - Server Integration Guide

## Overview
The MotionBlend UI is now configured to connect to your production server at `https://www.rydlr.com/motionblend/`

## Configuration

### Environment Variables
The UI uses Vite environment variables for API configuration:

```bash
# ui/.env
VITE_API_URL=https://www.rydlr.com/motionblend
```

### Client Configuration
The API client (`src/client.ts`) is configured with:
- **Production API**: `https://www.rydlr.com/motionblend`
- **Fallback behavior**: Mock data for offline development

## Required Server Endpoints

Your server at `https://www.rydlr.com/motionblend/` must implement these endpoints:

### 1. Motion Library
```http
GET /motions
```
Returns list of available motion capture files:
```json
{
  "motions": [
    {
      "id": "motion_001",
      "name": "Walking Forward",
      "metadata": {
        "category": "locomotion",
        "duration": 2.5,
        "frames": 75
      }
    }
  ]
}
```

### 2. Blend Operation
```http
POST /api/blend
Content-Type: application/json

{
  "motion1": "walking_forward",
  "motion2": "jump_landing",
  "weight": 0.5
}
```
Returns blend result:
```json
{
  "id": "blend_12345",
  "name": "walking_forward_jump_landing_blend",
  "status": "completed",
  "created_at": "2025-10-22T10:30:00Z",
  "metadata": {
    "source_motions": ["walking_forward", "jump_landing"],
    "blend_weight": 0.5,
    "frames": 120,
    "duration": 4.0
  }
}
```

### 3. Artifacts Management
```http
GET /api/artifacts/manifest
```
Returns manifest of generated artifacts:
```json
{
  "artifacts": [
    {
      "id": "blend_12345",
      "name": "walking_forward_jump_landing_blend",
      "created_at": "2025-10-22T10:30:00Z",
      "metadata": {...}
    }
  ],
  "total": 42,
  "last_updated": "2025-10-22T10:30:00Z"
}
```

### 4. Artifact Details
```http
GET /api/artifact/{artifact_id}/describe
GET /api/artifact/{artifact_id}/analysis
```

### 5. Elasticsearch Search
```http
POST /search/vector
Content-Type: application/json

{
  "vector": [0.123, 0.456, ...],  // 384-dim embedding
  "k": 5
}
```

## CORS Configuration

Your server **MUST** enable CORS to allow the UI to make requests. Add these headers:

### Flask Example
```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for your UI domain
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",  # Development
            "https://yourdomain.com"   # Production UI
        ],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### Express Example
```javascript
const express = require('express');
const cors = require('cors');

const app = express();

app.use(cors({
  origin: [
    'http://localhost:5173',
    'https://yourdomain.com'
  ],
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
```

### Nginx Example
```nginx
location /motionblend/ {
    proxy_pass http://backend:8080/;
    
    # CORS headers
    add_header 'Access-Control-Allow-Origin' '$http_origin' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization' always;
    add_header 'Access-Control-Max-Age' '86400' always;
    
    # Handle preflight requests
    if ($request_method = 'OPTIONS') {
        return 204;
    }
}
```

## Development

### Running Locally
```bash
cd ui
npm install
npm run dev
```

The UI will start at `http://localhost:5173` and connect to your production server.

### Testing with Local Server
To test with a local server instead:
```bash
# Terminal 1: Start local API server
cd project/search_api
python search_service.py

# Terminal 2: Update .env and start UI
echo "VITE_API_URL=http://localhost:8080" > .env
npm run dev
```

### Building for Production
```bash
npm run build
```

Output will be in `dist/` folder. Deploy to:
- Static hosting (Vercel, Netlify, GitHub Pages)
- Cloud Run (containerized)
- S3 + CloudFront

## Architecture

```
┌──────────────────┐
│   React UI       │
│  (localhost:5173)│
│                  │
│  - Motion Mixer  │
│  - Artifacts     │
│  - Search        │
└────────┬─────────┘
         │
         │ HTTPS
         │
         ▼
┌──────────────────────────────┐
│  rydlr.com/motionblend/      │
│                              │
│  - /motions                  │
│  - /api/blend                │
│  - /api/artifacts/manifest   │
│  - /search/vector            │
└──────────┬───────────────────┘
           │
           │
           ▼
┌────────────────────────────┐
│  Backend Services          │
│                            │
│  - BigQuery (Marts)        │
│  - Elasticsearch           │
│  - GCS (Motion Files)      │
│  - Ganimator (Blending)    │
└────────────────────────────┘
```

## API Client Features

### Automatic Fallbacks
The client provides mock data when the server is unavailable:
- Motion library with sample data
- Simulated blend operations
- Empty artifact manifests

### Error Handling
All API calls include try/catch with console logging:
```typescript
try {
  const res = await axios.get(`${API_BASE}/motions`)
  return { motions: res.data }
} catch (error) {
  console.error('Failed to fetch motions:', error)
  return { motions: [...] }  // Mock data
}
```

## Troubleshooting

### CORS Errors
```
Access to fetch at 'https://www.rydlr.com/motionblend/motions' has been blocked by CORS policy
```
**Solution**: Add CORS headers to your server (see CORS Configuration above)

### 404 Not Found
```
GET https://www.rydlr.com/motionblend/motions 404
```
**Solution**: Ensure your server implements all required endpoints

### Network Timeout
```
Request failed with status code 504
```
**Solution**: Check server health and firewall rules. Increase timeout:
```typescript
axios.get(url, { timeout: 30000 })  // 30 seconds
```

### Preflight Failures
```
OPTIONS https://www.rydlr.com/motionblend/api/blend 405
```
**Solution**: Handle OPTIONS requests on server:
```python
@app.route('/api/blend', methods=['OPTIONS'])
def blend_options():
    return '', 204
```

## Security Considerations

### API Authentication
Add authentication to your server:
```typescript
// In client.ts
const API_KEY = import.meta.env?.VITE_API_KEY

axios.defaults.headers.common['Authorization'] = `Bearer ${API_KEY}`
```

### Rate Limiting
Implement rate limiting on your server:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/blend', methods=['POST'])
@limiter.limit("10 per minute")
def blend():
    # ...
```

### HTTPS Only
Ensure production server uses HTTPS:
- Valid SSL certificate
- Redirect HTTP → HTTPS
- HSTS headers

## Deployment Checklist

- [ ] Server implements all required endpoints
- [ ] CORS headers configured correctly
- [ ] SSL certificate valid
- [ ] Environment variables set (`VITE_API_URL`)
- [ ] UI built for production (`npm run build`)
- [ ] Test all features (Motion Mixer, Artifacts, Search)
- [ ] Monitor error logs
- [ ] Set up analytics/monitoring

## Support

For issues with:
- **UI**: Check browser console for errors
- **Server**: Check server logs for failed requests
- **CORS**: Use browser DevTools Network tab to inspect headers
- **Authentication**: Verify API keys/tokens are valid

## Related Documentation

- [ELASTICSEARCH.md](../connectors/motionblend/ELASTICSEARCH.md) - Search service setup
- [METRICS_DOCUMENTATION.md](../analysis/METRICS_DOCUMENTATION.md) - Quality metrics API
- [README.md](./README.md) - UI development guide
