import express from 'express';
import mysql from 'mysql2';
import cors from 'cors';
import dotenv from 'dotenv';
import axios from 'axios';

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
// Also parse text/plain as JSON — sendBeacon fallback sends JSON with text/plain content type
app.use(express.text({ type: 'text/plain' }));
app.use((req, res, next) => {
  if (typeof req.body === 'string' && req.headers['content-type'] === 'text/plain') {
    try { req.body = JSON.parse(req.body); } catch {}
  }
  next();
});

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

app.delete('/api/videos/:id', async (req, res) => {
  try {
    const { id } = req.params;
    const response = await axios.delete(`${OCR_SERVICE_URL}/api/videos/${id}`);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting video:', error.message);
    res.status(error.response?.status || 500).json(error.response?.data || {
      error: 'Failed to delete video',
      details: error.message,
    });
  }
});

// ── Debug log from Chrome extension ──────────────────────────────────────────
let debugLogs = [];
app.post('/api/debug', (req, res) => {
  const entry = { timestamp: new Date().toISOString(), ...req.body };
  debugLogs.push(entry);
  if (debugLogs.length > 100) debugLogs = debugLogs.slice(-50);
  res.json({ status: 'ok' });
});
app.get('/api/debug', (req, res) => {
  res.json({ logs: debugLogs });
});
app.delete('/api/debug', (req, res) => {
  debugLogs = [];
  res.json({ status: 'cleared' });
});

// ── Browse events from Chrome extension ─────────────────────────────────────

// Create browse_events table if it doesn't exist (in case DB predates schema update)
db.query(`
  CREATE TABLE IF NOT EXISTS browse_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_type ENUM('impression', 'click', 'watch_end') NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    video_id VARCHAR(20) NOT NULL,
    title VARCHAR(500),
    channel_name VARCHAR(255),
    views VARCHAR(100),
    duration VARCHAR(50),
    posted_ago VARCHAR(100),
    watch_duration_sec INT DEFAULT NULL,
    page_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_video (video_id),
    INDEX idx_type_time (event_type, created_at)
  ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
`, (err) => {
  if (err) console.error('browse_events table check:', err.message);
});

app.post('/api/events', (req, res) => {
  const { events } = req.body;
  if (!Array.isArray(events) || events.length === 0) {
    return res.status(400).json({ error: 'No events provided' });
  }

  const values = events.map((e) => [
    e.type,
    e.sessionId,
    e.videoId,
    e.title || null,
    e.channel || null,
    e.views || null,
    e.duration || null,
    e.postedAgo || null,
    e.watchDurationSec || null,
    e.url || null,
  ]);

  const sql = `
    INSERT INTO browse_events
      (event_type, session_id, video_id, title, channel_name, views, duration, posted_ago, watch_duration_sec, page_url)
    VALUES ?
  `;

  db.query(sql, [values], (err, result) => {
    if (err) {
      console.error('Error inserting events:', err.message);
      return res.status(500).json({ error: err.message });
    }
    // Also upsert videos from click/impression events so the videos table stays populated
    events
      .filter((e) => e.title && (e.type === 'click' || e.type === 'impression'))
      .forEach((e) => {
        db.query(
          `INSERT INTO videos (title, channel_name, view_count, duration, extracted_at, raw_data)
           VALUES (?, ?, ?, ?, NOW(), ?)
           ON DUPLICATE KEY UPDATE view_count = VALUES(view_count), duration = VALUES(duration)`,
          [e.title, e.channel, e.views, e.duration, JSON.stringify(e)],
          () => {} // fire and forget
        );
      });

    res.json({ status: 'ok', inserted: result.affectedRows });
  });
});

// Get all distinct clicked video IDs (for recommendation filtering)
app.get('/api/clicked-video-ids', (req, res) => {
  const sql = 'SELECT DISTINCT video_id FROM browse_events WHERE event_type = ?';
  db.query(sql, ['click'], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ video_ids: results.map((r) => r.video_id) });
  });
});

// Get browsing events (for dashboard / ML training data export)
app.get('/api/events', (req, res) => {
  const type = req.query.type; // optional filter: impression, click, watch_end
  const limit = Math.min(parseInt(req.query.limit) || 500, 5000);

  let sql = 'SELECT * FROM browse_events';
  const params = [];

  if (type) {
    sql += ' WHERE event_type = ?';
    params.push(type);
  }
  sql += ' ORDER BY created_at DESC LIMIT ?';
  params.push(limit);

  db.query(sql, params, (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ events: results, count: results.length });
  });
});

// Get training data: pairs of (impressions in a session, what was clicked)
app.get('/api/training-data', (req, res) => {
  const sql = `
    SELECT
      session_id,
      GROUP_CONCAT(CASE WHEN event_type = 'impression' THEN video_id END) AS impressions,
      GROUP_CONCAT(CASE WHEN event_type = 'click' THEN video_id END) AS clicks,
      GROUP_CONCAT(
        CASE WHEN event_type = 'watch_end'
          THEN CONCAT(video_id, ':', watch_duration_sec)
        END
      ) AS watch_times
    FROM browse_events
    GROUP BY session_id
    HAVING clicks IS NOT NULL
    ORDER BY MIN(created_at) DESC
    LIMIT ?
  `;
  const limit = Math.min(parseInt(req.query.limit) || 500, 5000);

  db.query(sql, [limit], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ sessions: results, count: results.length });
  });
});

// ── ML service proxy ────────────────────────────────────────────────────────

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://ml-service:8000';

app.post('/api/ml/train', async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/train`, req.body || {});
    res.json(response.data);
  } catch (error) {
    res.status(error.response?.status || 500).json(
      error.response?.data || { error: 'ML service unavailable', details: error.message }
    );
  }
});

app.get('/api/ml/train/status', async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/train/status`);
    res.json(response.data);
  } catch (error) {
    // If ML service is down, check local file state as fallback
    res.json({
      model_exists: false,
      id_mapper_exists: false,
      index_exists: false,
      error: 'ML service unavailable',
    });
  }
});

app.post('/api/ml/evaluate', async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/evaluate`, req.body || {}, { timeout: 120000 });
    res.json(response.data);
  } catch (error) {
    res.status(error.response?.status || 500).json(
      error.response?.data || { error: 'ML service unavailable', details: error.message }
    );
  }
});

app.post('/api/ml/recommend', async (req, res) => {
  try {
    const response = await axios.post(`${ML_SERVICE_URL}/recommend_from_history`, req.body || {});
    const data = response.data;

    // Enrich recommendations with titles/channels from browse_events
    const recs = data.recommendations || [];
    const videoIds = recs.map((r) => r.video_id).filter((v) => v && v !== 'unknown');

    if (videoIds.length > 0) {
      const placeholders = videoIds.map(() => '?').join(',');
      const sql = `
        SELECT video_id,
               MAX(title) AS title,
               MAX(channel_name) AS channel_name,
               MAX(views) AS views,
               MAX(duration) AS duration
        FROM browse_events
        WHERE video_id IN (${placeholders}) AND title IS NOT NULL
        GROUP BY video_id
        ORDER BY MAX(created_at) DESC
      `;
      const [rows] = await db.promise().query(sql, videoIds);
      const metaMap = {};
      for (const row of rows) {
        metaMap[row.video_id] = row;
      }
      for (const rec of recs) {
        const meta = metaMap[rec.video_id];
        if (meta) {
          rec.title = meta.title;
          rec.channel = meta.channel_name;
          rec.views = meta.views;
          rec.duration = meta.duration;
        }
      }
    }

    res.json(data);
  } catch (error) {
    res.status(error.response?.status || 500).json(
      error.response?.data || { error: 'ML service unavailable', details: error.message }
    );
  }
});

app.get('/api/ml/health', async (req, res) => {
  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`);
    res.json(response.data);
  } catch (error) {
    res.json({ status: 'unavailable', error: error.message });
  }
});

app.listen(process.env.PORT || 4000, '0.0.0.0', () => {
  console.log(`Backend server listening on port ${process.env.PORT || 4000}`);
});
