#!/bin/sh
# Start a virtual framebuffer and then launch the Flask OCR service.

export DISPLAY=:99

# Start Xvfb in the background
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp &
sleep 1  # brief pause for Xvfb to initialise

echo "Xvfb running on $DISPLAY"
exec python app.py
