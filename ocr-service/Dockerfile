FROM python:3.11-slim

# Install system dependencies for Tesseract, OpenCV and Xvfb for virtual display
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Expose port
EXPOSE 5000

# Set environment variable for Tesseract (best-effort)
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Start Xvfb then run the Flask app
CMD ["./entrypoint.sh"]

