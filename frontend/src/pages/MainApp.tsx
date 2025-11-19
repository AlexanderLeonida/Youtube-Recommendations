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
  const recordingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const checkIntervalRef = useRef<NodeJS.Timeout | null>(null);

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

  // Check if user is on YouTube
  useEffect(() => {
    const checkYouTube = () => {
      // Check if we're in an iframe or if the parent window is YouTube
      try {
        const currentUrl = window.location.href;
        const isYouTube = currentUrl.includes('youtube.com') || 
                         currentUrl.includes('youtu.be') ||
                         (window.parent && window.parent.location.href.includes('youtube.com'));
        
        setIsOnYouTube(prev => {
          if (prev !== isYouTube) {
            if (isYouTube && !isRecording) {
              // Start recording when detected on YouTube
              api.startRecording(300, 2.0)
                .then(res => {
                  setIsRecording(true);
                  setStatus("Recording started! Extracting video data from YouTube...");
                })
                .catch(err => {
                  console.error('Error starting recording:', err);
                  setStatus("Failed to start recording. Make sure OCR service is running.");
                });
            } else if (!isYouTube && isRecording) {
              // Stop recording when leaving YouTube
              api.stopRecording()
                .then(res => {
                  setIsRecording(false);
                  setStatus("Recording stopped.");
                  loadVideos();
                })
                .catch(err => {
                  console.error('Error stopping recording:', err);
                  setIsRecording(false);
                });
            }
            return isYouTube;
          }
          return prev;
        });
      } catch (e) {
        // Cross-origin restrictions might prevent checking parent
        // In that case, we'll rely on manual triggers
      }
    };

    checkIntervalRef.current = setInterval(checkYouTube, 2000);
    checkYouTube(); // Initial check

    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
      }
    };
  }, [isRecording]);

  // Periodic frame capture when recording
  useEffect(() => {
    if (isRecording && isOnYouTube) {
      // Capture a frame every 5 seconds
      recordingIntervalRef.current = setInterval(() => {
        api.captureFrame()
          .then(res => {
            if (res.data.status === 'success' && res.data.video_data) {
              console.log('Captured video data:', res.data.video_data);
              // Refresh videos list
              api.getVideos()
                .then(res => {
                  if (res.data.videos) {
                    setVideos(res.data.videos);
                  }
                })
                .catch(err => console.error('Error loading videos:', err));
            }
          })
          .catch(err => console.error('Error capturing frame:', err));
      }, 5000);
    } else {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
        recordingIntervalRef.current = null;
      }
    }

    return () => {
      if (recordingIntervalRef.current) {
        clearInterval(recordingIntervalRef.current);
      }
    };
  }, [isRecording, isOnYouTube]);

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

  const startRecording = () => {
    api.startRecording(300, 2.0) // Record for 5 minutes, process every 2 seconds
      .then(res => {
        setIsRecording(true);
        const serverMessage = res.data?.message || "Recording started! Extracting video data from YouTube...";
        setStatus(serverMessage);
        console.log('Recording started:', res.data);
      })
      .catch(err => {
        console.error('Error starting recording:', err);
        const errorMessage = err.response?.data?.message || 
                            err.response?.data?.error || 
                            err.message || 
                            "Failed to start recording. Make sure OCR service is running.";
        setStatus(`Recording error: ${errorMessage}`);
      });
  };

  const stopRecording = () => {
    api.stopRecording()
      .then(res => {
        setIsRecording(false);
        setStatus("Recording stopped.");
        console.log('Recording stopped:', res.data);
        loadVideos(); // Refresh videos after stopping
      })
      .catch(err => {
        console.error('Error stopping recording:', err);
        setIsRecording(false);
      });
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
            Start Recording
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
