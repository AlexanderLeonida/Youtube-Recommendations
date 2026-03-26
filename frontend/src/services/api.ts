import axios from 'axios';

const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:4000";

export const api = {
  // Health check
  health: () => axios.get(`${API_BASE_URL}/api/health`),

  // Video endpoints
  getVideos: () =>
    axios.get(`${API_BASE_URL}/api/videos`),

  deleteVideo: (id: number) =>
    axios.delete(`${API_BASE_URL}/api/videos/${id}`),

  // Browse events from Chrome extension
  getEvents: (type?: string, limit?: number) =>
    axios.get(`${API_BASE_URL}/api/events`, {
      params: { type, limit: limit || 500 },
    }),

  // ML training data: sessions with impressions, clicks, watch times
  getTrainingData: (limit?: number) =>
    axios.get(`${API_BASE_URL}/api/training-data`, {
      params: { limit: limit || 500 },
    }),

  // ML model endpoints
  trainModel: (epochs?: number, batchSize?: number) =>
    axios.post(`${API_BASE_URL}/api/ml/train`, {
      epochs: epochs || 10,
      batch_size: batchSize || 32,
    }),

  getTrainStatus: () =>
    axios.get(`${API_BASE_URL}/api/ml/train/status`),

  getRecommendations: (topK?: number) =>
    axios.post(`${API_BASE_URL}/api/ml/recommend`, {
      top_k: topK || 20,
    }),

  getMLHealth: () =>
    axios.get(`${API_BASE_URL}/api/ml/health`),

  // Send browse events (used to record recommendation clicks)
  postEvents: (events: Record<string, unknown>[]) =>
    axios.post(`${API_BASE_URL}/api/events`, { events }),
};
