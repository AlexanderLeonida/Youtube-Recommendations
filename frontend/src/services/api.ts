import axios from 'axios';

const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://backend:4000";

export const api = {
  // Health check
  health: () => axios.get(`${API_BASE_URL}/api/health`),

  // Recording endpoints
  startRecording: (duration?: number, frameInterval?: number) =>
    axios.post(`${API_BASE_URL}/api/recording/start`, {
      duration,
      frame_interval: frameInterval,
    }),

  stopRecording: () =>
    axios.post(`${API_BASE_URL}/api/recording/stop`),

  captureFrame: () =>
    axios.post(`${API_BASE_URL}/api/recording/capture`),

  getRecordingStatus: () =>
    axios.get(`${API_BASE_URL}/api/recording/status`),

  // Video endpoints
  getVideos: () =>
    axios.get(`${API_BASE_URL}/api/videos`),
};
