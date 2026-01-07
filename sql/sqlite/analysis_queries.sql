-- QUERY 1: Basic Data Overview
SELECT 
    COUNT(*) as total_posts,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(DISTINCT country) as countries_covered,
    MIN(timestamp) as first_post,
    MAX(timestamp) as last_post
FROM social_media_posts;

-- QUERY 2: Sentiment Distribution
SELECT 
    sentiment,
    COUNT(*) as post_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM social_media_posts), 2) as percentage,
    AVG(retweets) as avg_retweets,
    AVG(likes) as avg_likes,
    AVG(retweets + likes) as avg_total_engagement
FROM social_media_posts
GROUP BY sentiment
ORDER BY post_count DESC;

-- QUERY 3: Platform Comparison
SELECT 
    platform,
    COUNT(*) as post_count,
    AVG(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive_ratio,
    AVG(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as negative_ratio,
    AVG(retweets) as avg_retweets,
    AVG(likes) as avg_likes,
    AVG(engagement_score) as avg_engagement_score
FROM social_media_posts
GROUP BY platform
ORDER BY post_count DESC;

-- QUERY 4: Hourly Activity Patterns
SELECT 
    created_hour,
    COUNT(*) as post_count,
    AVG(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive_ratio,
    AVG(retweets + likes) as avg_engagement
FROM social_media_posts
GROUP BY created_hour
ORDER BY created_hour;

-- QUERY 5: Country-wise Analysis
SELECT 
    country,
    COUNT(*) as total_posts,
    ROUND(100.0 * SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) / COUNT(*), 2) as positive_percentage,
    AVG(retweets + likes) as avg_engagement,
    COUNT(DISTINCT user_id) as unique_users
FROM social_media_posts
WHERE country IS NOT NULL
GROUP BY country
ORDER BY total_posts DESC;

-- QUERY 6: Top Performing Posts
SELECT 
    post_id,
    user_id,
    platform,
    sentiment,
    content,
    retweets,
    likes,
    (retweets + likes) as total_engagement,
    country,
    timestamp
FROM social_media_posts
ORDER BY total_engagement DESC
LIMIT 10;

-- QUERY 7: Daily Trends
SELECT 
    DATE(timestamp) as post_date,
    COUNT(*) as daily_posts,
    AVG(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as daily_positive_ratio,
    SUM(retweets + likes) as daily_engagement
FROM social_media_posts
GROUP BY DATE(timestamp)
ORDER BY post_date;

-- QUERY 8: Hashtag Analysis
WITH hashtag_split AS (
    SELECT 
        post_id,
        TRIM(hashtag) as hashtag
    FROM social_media_posts,
    json_each('["' || REPLACE(REPLACE(hashtags, '#', ''), ' ', '","') || '"]') as hashtag
    WHERE hashtags != '' AND hashtag.value != ''
)
SELECT 
    hashtag,
    COUNT(*) as usage_count,
    AVG(sentiment_score) as avg_sentiment
FROM hashtag_split hs
JOIN (
    SELECT 
        post_id,
        CASE sentiment 
            WHEN 'Positive' THEN 1
            WHEN 'Negative' THEN -1
            ELSE 0 
        END as sentiment_score
    FROM social_media_posts
) s ON hs.post_id = s.post_id
GROUP BY hashtag
HAVING usage_count >= 2
ORDER BY usage_count DESC
LIMIT 20;