#!/bin/sh
# Start a virtual X display and then run the OCR service
set -e

: ${DISPLAY:=:99}

echo "Starting Xvfb on display ${DISPLAY}..."
# Start X virtual framebuffer in background
Xvfb ${DISPLAY} -screen 0 1920x1080x24 &
XVFB_PID=$!

export DISPLAY=${DISPLAY}

echo "Running OCR service..."
python app.py

# If python exits, kill Xvfb
kill ${XVFB_PID} || true
