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
COPY project/vertex_pipeline/requirements.txt ./requirements_vertex.txt

# Install Python dependencies (heavy ML deps first for cache efficiency)
RUN pip install --upgrade pip && \
    # Install a CPU-only PyTorch wheel first to avoid pulling CUDA/Cu* binaries
    # (this prevents architecture / ELF header mismatches when building on macOS
    # hosts or on CI without GPU support).
    pip install --no-cache-dir "torch==2.2.2+cpu" -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install -r requirements_elastic.txt && \
    pip install -r requirements_vertex.txt && \
    pip install pytest && \
    if [ -s requirements_ganimator.txt ]; then pip install -r requirements_ganimator.txt; fi && \
    # Pin huggingface_hub to a version compatible with sentence-transformers 2.2.2
    # (newer huggingface_hub releases removed `cached_download` which causes
    # ImportError in sentence-transformers 2.2.2). Adjust if you upgrade
    # sentence-transformers in the future.
    pip install --upgrade sentence-transformers==2.2.2 "huggingface_hub==0.13.4"

# Copy the rest of the code
COPY project/ ./project/
COPY scripts/ ./scripts/

# Expose Flask port
EXPOSE 5000

# Entrypoint (adjust if needed)
CMD ["python", "project/elastic_search/app.py"]
