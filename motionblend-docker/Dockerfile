# Copy of Dockerfile.api to satisfy Cloud Build trigger expecting 'motionblend-docker' path
# This is a temporary, safe copy. Consider updating the Cloud Build trigger to use the canonical Dockerfile.api

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-api.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY api_server.py .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "api_server:app"]
