-- Social Media Posts Table
CREATE TABLE IF NOT EXISTS social_media_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id TEXT UNIQUE,
    user_id TEXT,
    platform TEXT CHECK(platform IN ('Twitter', 'Instagram', 'Facebook')),
    sentiment TEXT CHECK(sentiment IN ('Positive', 'Negative', 'Neutral')),
    content TEXT,
    hashtags TEXT,
    retweets INTEGER DEFAULT 0,
    likes INTEGER DEFAULT 0,
    country TEXT,
    timestamp DATETIME,
    created_year INTEGER,
    created_month INTEGER,
    created_day INTEGER,
    created_hour INTEGER,
    engagement_score FLOAT,
    is_peak_hour BOOLEAN DEFAULT 0,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily Aggregated Metrics
CREATE TABLE IF NOT EXISTS daily_metrics (
    date DATE PRIMARY KEY,
    total_posts INTEGER DEFAULT 0,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    avg_retweets FLOAT DEFAULT 0,
    avg_likes FLOAT DEFAULT 0,
    total_engagement INTEGER DEFAULT 0,
    sentiment_ratio FLOAT DEFAULT 0,
    peak_hour_posts INTEGER DEFAULT 0
);

-- Platform Analysis
CREATE TABLE IF NOT EXISTS platform_analysis (
    platform TEXT,
    date DATE,
    sentiment TEXT,
    post_count INTEGER,
    avg_engagement FLOAT,
    PRIMARY KEY (platform, date, sentiment)
);

-- Country Analysis
CREATE TABLE IF NOT EXISTS country_analysis (
    country TEXT,
    date DATE,
    sentiment TEXT,
    post_count INTEGER,
    avg_engagement FLOAT,
    PRIMARY KEY (country, date, sentiment)
);

-- User Engagement Summary
CREATE TABLE IF NOT EXISTS user_summary (
    user_id TEXT PRIMARY KEY,
    total_posts INTEGER DEFAULT 0,
    avg_sentiment_score FLOAT DEFAULT 0,
    total_engagement INTEGER DEFAULT 0,
    favorite_platform TEXT,
    last_post_date DATE
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_timestamp ON social_media_posts(timestamp);
CREATE INDEX IF NOT EXISTS idx_sentiment ON social_media_posts(sentiment);
CREATE INDEX IF NOT EXISTS idx_platform ON social_media_posts(platform);
CREATE INDEX IF NOT EXISTS idx_country ON social_media_posts(country);
CREATE INDEX IF NOT EXISTS idx_engagement ON social_media_posts(engagement_score);