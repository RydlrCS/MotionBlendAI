# syntax=docker/dockerfile:1
FROM python:3.10-slim

# System deps for scientific Python and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first for better cache
COPY project/elastic_search/requirements.txt ./requirements_elastic.txt
COPY project/ganimator/requirements.txt ./requirements_ganimator.txt

# Install Python dependencies (heavy ML deps first for cache efficiency)
RUN pip install --upgrade pip && \
    pip install -r requirements_elastic.txt && \
    if [ -s requirements_ganimator.txt ]; then pip install -r requirements_ganimator.txt; fi

# Copy the rest of the code
COPY project/ ./project/
COPY scripts/ ./scripts/

# Expose Flask port
EXPOSE 5000

# Entrypoint (adjust if needed)
CMD ["python", "project/search_api/search_service.py"]
