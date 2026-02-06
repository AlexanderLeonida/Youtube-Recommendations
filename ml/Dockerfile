# TwinTube Vector ML Service
# Recommendation model serving (CPU mode for macOS, GPU mode available)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch (CPU version for compatibility)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install FAISS (CPU version)
RUN pip install faiss-cpu

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/checkpoints /app/index /app/data /app/cache

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run inference server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
