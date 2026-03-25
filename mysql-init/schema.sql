-- YouTube Recommendations Database Schema

CREATE DATABASE IF NOT EXISTS ytrecs;
USE ytrecs;

-- Videos table to store extracted video data from OCR
CREATE TABLE IF NOT EXISTS videos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(500),
    channel_name VARCHAR(255),
    view_count VARCHAR(100),
    duration VARCHAR(50),
    extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_title_channel (title(255), channel_name(100))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Sessions table to track user sessions
CREATE TABLE IF NOT EXISTS sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP NULL,
    youtube_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Video views table to track which videos were viewed in each session
CREATE TABLE IF NOT EXISTS video_views (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id INT,
    video_id INT,
    viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Browsing events: impressions, clicks, watch_end from the Chrome extension
CREATE TABLE IF NOT EXISTS browse_events (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    event_type ENUM('impression', 'click', 'watch_end') NOT NULL,
    session_id VARCHAR(100) NOT NULL,
    video_id VARCHAR(20) NOT NULL,
    title VARCHAR(500),
    channel VARCHAR(255),
    views VARCHAR(100),
    duration VARCHAR(50),
    posted_ago VARCHAR(100),
    watch_duration_sec INT DEFAULT NULL,
    page_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_video (video_id),
    INDEX idx_type_time (event_type, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
