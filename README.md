# Youtube-Recommendations
Deep Neural Network / Computer Vision project which tracks where you are on Youtube, what videos you're watching, and predicts what videos you might want to watch in the future.

## Features

- **Screen Recording**: Captures screen frames when you're on YouTube
- **OCR Pipeline**: Uses PyTesseract and OpenCV to extract video data from screen recordings
- **Video Data Extraction**: Automatically extracts:
  - Video titles
  - Channel names
  - View counts
  - Video durations
- **Web Interface**: React frontend to control recording and view extracted data
- **REST API**: Backend API for managing recordings and video data

## Architecture

- **Frontend**: React/TypeScript application
- **Backend**: Node.js/Express API server
- **OCR Service**: Python/Flask service with PyTesseract and OpenCV
- **Database**: MySQL for storing extracted video data

## Setup

### Prerequisites

- Docker and Docker Compose
- Node.js 20+ (for local development)
- Python 3.11+ (for OCR service)
- Tesseract OCR (for OCR service)

### Installation

1. Clone the repository
2. Install Tesseract OCR:
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki

3. Set up environment variables:
   - Create `backend/.env` with database credentials
   - Create `ocr-service/.env` (see `ocr-service/.env.example`)

4. Start services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

### Important Note on Screen Recording

**The OCR service requires access to your display for screen recording.** Docker containers typically don't have direct access to the host's display server. 

**For screen recording to work properly, you have two options:**

1. **Run OCR service locally** (Recommended):
   ```bash
   cd ocr-service
   pip install -r requirements.txt
   python app.py
   ```

2. **Use Docker with X11 forwarding** (Linux only):
   - Configure X11 forwarding in docker-compose.yml
   - Requires additional setup for display access
3. **Headless demo mode** (default in Docker):
   - OCR service generates placeholder video data when it cannot access the display
   - Controlled via `ALLOW_HEADLESS_MODE=true` in `ocr-service` environment variables
   - Useful for testing the workflow without enabling screen recording

The other services (backend, frontend, MySQL) can run in Docker without issues.

## Usage

1. Start all services (or run OCR service locally)
2. Open the frontend at `http://localhost:3000`
3. Click "Start Recording" or navigate to YouTube (recording will auto-start)
4. Browse YouTube - the service will capture frames and extract video data
5. View extracted videos in the web interface

## API Endpoints

### Backend (Port 4000)
- `GET /api/health` - Health check
- `POST /api/recording/start` - Start screen recording
- `POST /api/recording/stop` - Stop screen recording
- `POST /api/recording/capture` - Capture single frame
- `GET /api/recording/status` - Get recording status
- `GET /api/videos` - Get extracted videos

### OCR Service (Port 5000)
- `GET /health` - Health check
- `POST /api/start-recording` - Start recording
- `POST /api/stop-recording` - Stop recording
- `POST /api/capture-frame` - Capture frame
- `GET /api/recording-status` - Recording status
- `GET /api/videos` - Get videos from database

## Project Structure

```
├── backend/          # Node.js/Express backend
├── frontend/         # React frontend
├── ocr-service/      # Python OCR service (screen recording + OCR)
├── opencv/           # OpenCV C++ service
├── mysql-init/       # Database initialization
└── docker-compose.yml
```

## Development

See individual service READMEs for development setup:
- `ocr-service/README.md` - OCR service documentation 
