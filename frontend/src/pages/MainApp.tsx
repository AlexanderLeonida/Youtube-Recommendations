import React, { useEffect, useState, useRef } from "react";
import { api } from "../services/api";

interface VideoData {
  id: number;
  title: string;
  channel_name: string;
  view_count: string;
  duration: string;
  extracted_at: string;
}

export default function MainApp() {
  const [status, setStatus] = useState("Checking backend connection...");
  const [isRecording, setIsRecording] = useState(false);
  const [videos, setVideos] = useState<VideoData[]>([]);
  const [isOnYouTube, setIsOnYouTube] = useState(false);
  const [captureInterval, setCaptureInterval] = useState(2000); // ms between captures
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const checkIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const screenVideoRef = useRef<HTMLVideoElement | null>(null);
  const screenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const captureLoopRef = useRef<NodeJS.Timeout | null>(null);

  // Check backend connection with retry logic
  useEffect(() => {
    const checkBackend = async (retries = 5) => {
      for (let i = 0; i < retries; i++) {
        try {
          console.log(`Attempting to connect to backend (attempt ${i + 1}/${retries})...`);
          const res = await api.health();
          console.log('Backend response:', res.data);
          setStatus(res.data.message);
          return;
        } catch (error: any) {
          console.error(`Backend connection attempt ${i + 1} failed:`, error.message);
          if (i === retries - 1) {
            const errorMsg = error.response 
              ? `Backend error: ${error.response.status} ${error.response.statusText}`
              : error.code === 'ECONNREFUSED'
              ? 'Backend connection refused. Is it running on port 4000?'
              : `Could not connect to backend: ${error.message}`;
            setStatus(errorMsg);
            console.error('Final connection error:', error);
          } else {
            // Wait before retrying (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
          }
        }
      }
    };
    checkBackend();
  }, []);

  // Check if user is on YouTube (kept for reference, but screen recording makes this less critical)
  useEffect(() => {
    const checkYouTube = () => {
      try {
        const currentUrl = window.location.href;
        const isYouTube = currentUrl.includes('youtube.com') || 
                         currentUrl.includes('youtu.be') ||
                         (window.parent && window.parent.location.href.includes('youtube.com'));
        setIsOnYouTube(isYouTube);
      } catch (e) {
        // Cross-origin restrictions
      }
    };

    checkYouTube();
    checkIntervalRef.current = setInterval(checkYouTube, 2000);

    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
      }
    };
  }, []);

  // Load videos on mount and periodically
  const loadVideos = () => {
    api.getVideos()
      .then(res => {
        if (res.data.videos) {
          setVideos(res.data.videos);
        }
      })
      .catch(err => console.error('Error loading videos:', err));
  };

  useEffect(() => {
    loadVideos();
    const interval = setInterval(loadVideos, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const startRecording = async () => {
    try {
      // Request screen capture permission
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      } as DisplayMediaStreamOptions);

      mediaStreamRef.current = stream;
      setIsRecording(true);
      setStatus("Screen recording started! Capturing frames and sending to OCR...");

      // Create hidden video element to receive stream
      const video = document.createElement('video');
      video.srcObject = stream;
      video.play();
      screenVideoRef.current = video;

      // Create hidden canvas for frame capture
      const canvas = document.createElement('canvas');
      canvas.width = 1920;
      canvas.height = 1080;
      screenCanvasRef.current = canvas;

      // Start capturing frames at regular interval
      captureLoopRef.current = setInterval(async () => {
        try {
          if (screenVideoRef.current && screenCanvasRef.current) {
            const ctx = screenCanvasRef.current.getContext('2d');
            if (ctx) {
              // Ensure canvas matches the actual video dimensions for accurate capture
              const v = screenVideoRef.current as HTMLVideoElement;
              if (v.videoWidth && v.videoHeight) {
                screenCanvasRef.current.width = v.videoWidth;
                screenCanvasRef.current.height = v.videoHeight;
              }
              ctx.drawImage(v, 0, 0, screenCanvasRef.current.width, screenCanvasRef.current.height);
              const imageBase64 = screenCanvasRef.current.toDataURL('image/png');
              
              // Send frame to OCR service
              const response = await api.uploadFrameToOCR(imageBase64);
              
              if (response.data.status === 'success' && response.data.video_data) {
                console.log('OCR extracted:', response.data.video_data);
                setStatus(`Extracted: ${response.data.video_data.title || 'Processing...'}`);
                // Refresh videos list
                loadVideos();
              }
            }
          }
        } catch (err) {
          console.error('Error uploading frame to OCR:', err);
        }
      }, captureInterval);

      // Handle stream ended (user clicked Stop in the share dialog)
      stream.getTracks().forEach((track) => {
        track.onended = () => {
          stopRecording();
        };
      });
    } catch (err: any) {
      if (err.name === 'NotAllowedError') {
        setStatus("Screen capture was cancelled. Please try again.");
      } else {
        console.error('Error starting screen recording:', err);
        setStatus(`Error: ${err.message || 'Could not start screen recording'}`);
      }
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    // Stop all tracks in the media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    // Clear capture loop
    if (captureLoopRef.current) {
      clearInterval(captureLoopRef.current);
      captureLoopRef.current = null;
    }

    // Clean up video and canvas
    if (screenVideoRef.current) {
      screenVideoRef.current.srcObject = null;
      screenVideoRef.current = null;
    }
    screenCanvasRef.current = null;

    setIsRecording(false);
    setStatus("Screen recording stopped.");
    loadVideos();
  };

  const handleGoToYouTube = () => {
    window.open("https://youtube.com", "_blank");
    // Note: Due to browser security, we can't directly detect when user navigates
    // to YouTube in a new tab. The user will need to manually start recording
    // or we can use a browser extension for better integration.
  };

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center" }}>YouTube Recommendations Tracker</h1>
      
      <div style={{ 
        padding: "20px", 
        backgroundColor: "#f5f5f5", 
        borderRadius: "8px", 
        marginBottom: "20px" 
      }}>
        <p><strong>Status:</strong> {status}</p>
        <p><strong>Recording:</strong> {isRecording ? "🟢 Active" : "🔴 Inactive"}</p>
        <p><strong>On YouTube:</strong> {isOnYouTube ? "✅ Yes" : "❌ No"}</p>
        <p>
          <strong>Capture Interval:</strong> {captureInterval}ms
          <br />
          <input 
            type="range" 
            min="500" 
            max="5000" 
            step="500" 
            value={captureInterval}
            onChange={(e) => setCaptureInterval(Number(e.target.value))}
            disabled={isRecording}
            style={{ width: "200px" }}
          />
        </p>
        
        <div style={{ marginTop: "15px", display: "flex", gap: "10px" }}>
          <button 
            onClick={startRecording} 
            disabled={isRecording}
            style={{
              padding: "10px 20px",
              backgroundColor: isRecording ? "#ccc" : "#4CAF50",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: isRecording ? "not-allowed" : "pointer"
            }}
          >
            Start Screen Recording
          </button>
          
          <button 
            onClick={stopRecording} 
            disabled={!isRecording}
            style={{
              padding: "10px 20px",
              backgroundColor: !isRecording ? "#ccc" : "#f44336",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: !isRecording ? "not-allowed" : "pointer"
            }}
          >
            Stop Recording
          </button>
          
          <button 
            onClick={handleGoToYouTube}
            style={{
              padding: "10px 20px",
              backgroundColor: "#FF0000",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Open YouTube
          </button>
          
          <button 
            onClick={loadVideos}
            style={{
              padding: "10px 20px",
              backgroundColor: "#2196F3",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Refresh Videos
          </button>
        </div>
      </div>

      <div>
        <h2>Extracted Videos ({videos.length})</h2>
        {videos.length === 0 ? (
          <p>No videos extracted yet. Start recording and navigate to YouTube!</p>
        ) : (
          <div style={{ display: "grid", gap: "15px" }}>
            {videos.map((video) => (
              <div 
                key={video.id}
                style={{
                  padding: "15px",
                  backgroundColor: "white",
                  border: "1px solid #ddd",
                  borderRadius: "8px",
                  boxShadow: "0 2px 4px rgba(0,0,0,0.1)"
                }}
              >
                <h3 style={{ margin: "0 0 10px 0", color: "#333" }}>
                  {video.title || "Unknown Title"}
                </h3>
                <p style={{ margin: "5px 0", color: "#666" }}>
                  <strong>Channel:</strong> {video.channel_name || "Unknown"}
                </p>
                <p style={{ margin: "5px 0", color: "#666" }}>
                  <strong>Views:</strong> {video.view_count || "N/A"}
                </p>
                <p style={{ margin: "5px 0", color: "#666" }}>
                  <strong>Duration:</strong> {video.duration || "N/A"}
                </p>
                <p style={{ margin: "5px 0", fontSize: "12px", color: "#999" }}>
                  Extracted: {new Date(video.extracted_at).toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
