import express from 'express';
import mysql from 'mysql2';
import cors from 'cors';
import dotenv from 'dotenv';
import axios from 'axios';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// MySQL connection pool
const db = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  port: process.env.DB_PORT || 3306,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

// Test initial connection
db.getConnection((err, connection) => {
  if (err) {
    console.error('Initial database connection failed:', err.message);
  } else {
    console.log('Connected to MySQL successfully via pool.');
    connection.release();
  }
});

// OCR service URL
const OCR_SERVICE_URL = process.env.OCR_SERVICE_URL || 'http://ocr-service:5000';

// Health endpoint
app.get('/api/health', (req, res) => {
  db.query('SELECT 1', (err) => {
    if (err) {
      res.json({
        status: 'ok',
        message: 'Backend is running but database connection is pending...',
        database: 'disconnected',
      });
    } else {
      res.json({
        status: 'ok',
        message: 'Backend is running and connected to MySQL!',
        database: 'connected',
      });
    }
  });
});

// OCR proxy endpoints
app.post('/api/recording/start', async (req, res) => {
  try {
    const { duration, frame_interval } = req.body;
    const response = await axios.post(`${OCR_SERVICE_URL}/api/start-recording`, {
      duration: duration || 30,
      frame_interval: frame_interval || 2.0,
    });
    res.json(response.data);
  } catch (error) {
    console.error('Error starting recording:', error.message);
    res.status(error.response?.status || 500).json(error.response?.data || {
      error: 'Failed to start recording',
      details: error.message,
    });
  }
});

app.post('/api/recording/stop', async (req, res) => {
  try {
    const response = await axios.post(`${OCR_SERVICE_URL}/api/stop-recording`);
    res.json(response.data);
  } catch (error) {
    console.error('Error stopping recording:', error.message);
    res.status(500).json({ error: 'Failed to stop recording', details: error.message });
  }
});

app.post('/api/recording/capture', async (req, res) => {
  try {
    const response = await axios.post(`${OCR_SERVICE_URL}/api/capture-frame`);
    res.json(response.data);
  } catch (error) {
    console.error('Error capturing frame:', error.message);
    res.status(error.response?.status || 500).json(error.response?.data || {
      error: 'Failed to capture frame',
      details: error.message,
    });
  }
});

app.get('/api/recording/status', async (req, res) => {
  try {
    const response = await axios.get(`${OCR_SERVICE_URL}/api/recording-status`);
    res.json(response.data);
  } catch (error) {
    console.error('Error getting recording status:', error.message);
    res.status(500).json({ error: 'Failed to get recording status', details: error.message });
  }
});

// Get extracted videos
app.get('/api/videos', async (req, res) => {
  try {
    const response = await axios.get(`${OCR_SERVICE_URL}/api/videos`);
    res.json(response.data);
  } catch (error) {
    console.error('Error getting videos from OCR service:', error.message);
    // fallback to database query
    db.query('SELECT * FROM videos ORDER BY extracted_at DESC LIMIT 100', (err, results) => {
      if (err) {
        res.status(500).json({ error: 'Failed to get videos from DB', details: err.message });
      } else {
        res.json({ videos: results, count: results.length });
      }
    });
  }
});

app.listen(process.env.PORT || 4000, '0.0.0.0', () => {
  console.log(`Backend server listening on port ${process.env.PORT || 4000}`);
});
