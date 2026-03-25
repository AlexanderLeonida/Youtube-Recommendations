## ── Stage 1: Build the C++ OpenCV preprocessor ──────────────────────
FROM python:3.11-slim AS cpp-builder

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY cpp/ .
RUN mkdir -p out && cd out && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j"$(nproc)"

## ── Stage 2: Python runtime ─────────────────────────────────────────
FROM python:3.11-slim

# Install system dependencies for Tesseract, OpenCV, Xvfb, and llama-cpp build
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    xvfb \
    ffmpeg \
    fonts-dejavu-core \
    libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev \
    libopencv-features2d-dev \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled C++ binary from builder stage
COPY --from=cpp-builder /build/out/ocr_preprocess /usr/local/bin/ocr_preprocess

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory for GGUF model downloads
RUN mkdir -p /app/models

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose port
EXPOSE 5000

# Set environment variable for Tesseract (best-effort)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Start Xvfb then run the Flask app
CMD ["./entrypoint.sh"]

