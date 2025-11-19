# OCR Screen Recording Service

This service provides screen recording functionality with OCR (Optical Character Recognition) to extract video data from YouTube pages using PyTesseract and OpenCV.

## Features

- **Screen Recording**: Captures screen frames using MSS (Multi-Screen Shot)
- **OCR Pipeline**: Uses PyTesseract for text extraction
- **Video Data Extraction**: Automatically extracts:
  - Video titles
  - Channel names
  - View counts
  - Video durations
- **Database Integration**: Stores extracted data in MySQL database
- **REST API**: Flask-based API for controlling recording and retrieving data

## Requirements

- Python 3.11+
- Tesseract OCR installed on the system
- Access to display (for screen recording)

## Installation

### Local Development

1. Install system dependencies:

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**Windows:**
Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```env
DB_HOST=localhost
DB_USER=ytrecs
DB_PASSWORD=password
DB_NAME=ytrecs
DB_PORT=3306
PORT=5000
ALLOW_HEADLESS_MODE=true
```

4. Run the service:
```bash
python app.py
```

## Docker Usage

**Important Note**: Screen recording requires access to the host's display. Docker containers typically don't have direct access to the X11 display server. For production use, you have a few options:

1. **Run the OCR service directly on the host** (recommended for screen recording)
2. **Use X11 forwarding** (Linux only)
3. **Use a VNC server** inside the container
4. **Headless demo mode** (default): When `ALLOW_HEADLESS_MODE=true`, the service generates placeholder video data when display access is unavailable. This is helpful for local testing without screen recording.

### Running with Docker (for testing without screen access)

```bash
docker build -t ocr-service .
docker run -p 5000:5000 ocr-service
```

## API Endpoints

### Health Check
```
GET /health
```

### Start Recording
```
POST /api/start-recording
Body: {
  "duration": 30,        // Recording duration in seconds (optional, default: 30)
  "frame_interval": 2.0  // Interval between frame processing in seconds (optional, default: 2.0)
}
```

### Stop Recording
```
POST /api/stop-recording
```

### Capture Single Frame
```
POST /api/capture-frame
```

### Get Recording Status
```
GET /api/recording-status
```

### Get Extracted Videos
```
GET /api/videos
```

## Usage Example

1. Start the service
2. Navigate to YouTube in your browser
3. Call the `/api/start-recording` endpoint
4. The service will capture screen frames and extract video data
5. View extracted videos via `/api/videos` endpoint

## How It Works

1. **Screen Capture**: Uses MSS library to capture screen frames
2. **Image Preprocessing**: 
   - Converts to grayscale
   - Applies thresholding
   - Denoises the image
   - Enhances contrast
3. **OCR Processing**: Uses PyTesseract to extract text with bounding boxes
4. **Data Extraction**: 
   - Identifies video titles (top-center region)
   - Extracts channel names (below title)
   - Finds view counts (pattern matching)
   - Detects video durations (time format patterns)
5. **Database Storage**: Saves extracted data to MySQL

## Limitations

- Screen recording requires display access (doesn't work well in Docker without X11)
- OCR accuracy depends on screen resolution and text clarity
- YouTube's dynamic content may require adjustments to extraction logic
- Cross-origin restrictions may limit browser-based detection

## Troubleshooting

### Tesseract not found
- Ensure Tesseract is installed and in your PATH
- On macOS, you may need to set: `pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'`

### Display access issues
- Ensure you have proper display permissions
- On Linux, you may need to set `DISPLAY` environment variable
- Consider running directly on host instead of Docker

### Low OCR accuracy
- Adjust preprocessing parameters in `screen_recorder.py`
- Increase image resolution if possible
- Fine-tune confidence thresholds

