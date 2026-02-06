# TwinTube OCR Pipeline
# C++ OCR with OpenCV (CPU mode for macOS, GPU available on Linux with NVIDIA)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopencv-dev \
    libopencv-contrib-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY . .

# Build (CPU mode)
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF .. && \
    make -j$(nproc)

# Create directories
RUN mkdir -p /usr/src/app/input /usr/src/app/output

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

# Run OCR pipeline
CMD ["./build/twintube_ocr", "/usr/src/app/input/video.mp4", "/usr/src/app/output/results.json", "30"]
