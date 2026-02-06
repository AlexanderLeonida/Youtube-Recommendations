#!/usr/bin/env bash
set -euo pipefail

# Build and start core services (exclude optional profiles like opencv-ocr)
docker-compose up --build -d

# Show running status
docker-compose ps

# Stream logs to terminal
docker-compose logs -f
