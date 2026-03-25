#!/usr/bin/env bash
set -euo pipefail

# Build and start core services (exclude optional profiles like opencv-ocr)
docker-compose up --build -d

# Show running status
docker-compose ps

echo ""
echo "==> Llama model will be downloaded automatically on first OCR request."
echo "    (GGUF model from HuggingFace — no Ollama required)"
echo ""

# Stream logs to terminal
docker-compose logs -f
