#!/usr/bin/env python3
"""
Social Media Sentiment Analysis Pipeline - Fixed Version
Complete end-to-end solution with proper database schema
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== DATABASE HANDLER ====================
class DatabaseHandler:
    """SQLite database handler with proper schema"""
    
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
            
    def create_schema(self):
        """Create database schema with CORRECT column names"""
        schema = """
        -- Drop existing tables to avoid conflicts
        DROP TABLE IF EXISTS social_media_posts;
        DROP TABLE IF EXISTS daily_metrics;
        DROP TABLE IF EXISTS platform_analysis;
        DROP TABLE IF EXISTS country_analysis;
        DROP TABLE IF EXISTS user_summary;
        DROP TABLE IF EXISTS hashtag_analysis;
        
        -- Main posts table
        CREATE TABLE social_media_posts (
            post_id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            platform TEXT,
            sentiment TEXT,
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
            engagement_score REAL DEFAULT 0.0,
            sentiment_score INTEGER DEFAULT 0,
            is_peak_hour INTEGER DEFAULT 0
        );
        
        -- Daily aggregated metrics - USING CORRECT COLUMN NAMES
        CREATE TABLE daily_metrics (
            date DATE PRIMARY KEY,
            total_posts INTEGER DEFAULT 0,
            positive_count INTEGER DEFAULT 0,
            negative_count INTEGER DEFAULT 0,
            neutral_count INTEGER DEFAULT 0,
            avg_retweets REAL DEFAULT 0.0,
            avg_likes REAL DEFAULT 0.0,
            total_engagement INTEGER DEFAULT 0,
            sentiment_ratio REAL DEFAULT 0.0,
            peak_hour_posts INTEGER DEFAULT 0,
            unique_users INTEGER DEFAULT 0
        );
        
        -- Platform analysis - USING CORRECT COLUMN NAMES
        CREATE TABLE platform_analysis (
            platform TEXT,
            date DATE,
            sentiment TEXT,
            post_count INTEGER DEFAULT 0,
            avg_engagement REAL DEFAULT 0.0
        );
        
        -- Country analysis - USING CORRECT COLUMN NAMES
        CREATE TABLE country_analysis (
            country TEXT,
            date DATE,
            sentiment TEXT,
            post_count INTEGER DEFAULT 0,
            avg_engagement REAL DEFAULT 0.0
        );
        
        -- User summary - USING CORRECT COLUMN NAMES
        CREATE TABLE user_summary (
            user_id TEXT PRIMARY KEY,
            total_posts INTEGER DEFAULT 0,
            avg_sentiment_score REAL DEFAULT 0.0,
            total_engagement INTEGER DEFAULT 0,
            favorite_platform TEXT,
            last_post_date DATE
        );
        
        -- Create indexes for performance
        CREATE INDEX idx_posts_timestamp ON social_media_posts(timestamp);
        CREATE INDEX idx_posts_sentiment ON social_media_posts(sentiment);
        CREATE INDEX idx_posts_platform ON social_media_posts(platform);
        CREATE INDEX idx_daily_date ON daily_metrics(date);
        """
        
        try:
            cursor = self.connection.cursor()
            cursor.executescript(schema)
            self.connection.commit()
            return True
        except Exception as e:
            print(f"❌ Schema creation failed: {e}")
            return False
            
    def insert_dataframe(self, df, table_name):
        """Insert DataFrame into table"""
        try:
            df.to_sql(table_name, self.connection, if_exists='replace', index=False)
            return True
        except Exception as e:
            print(f"❌ Data insertion failed for {table_name}: {e}")
            return False
            
    def execute_query(self, query, return_df=True):
        """Execute SQL query"""
        try:
            if return_df:
                df = pd.read_sql_query(query, self.connection)
                return df
            else:
                cursor = self.connection.cursor()
                cursor.execute(query)
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return None
            
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

# ==================== DATA PROCESSOR ====================
class DataProcessor:
    """Data processing and cleaning utilities"""
    
    @staticmethod
    def clean_data(df):
        """Clean and preprocess raw data"""
        df_clean = df.copy()
        
        # Handle missing values
        numeric_cols = ['retweets', 'likes']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(0)
                
        # Fill categorical missing values
        if 'country' in df_clean.columns:
            df_clean['country'] = df_clean['country'].fillna('Unknown')
        if 'hashtags' in df_clean.columns:
            df_clean['hashtags'] = df_clean['hashtags'].fillna('')
            
        # Convert timestamp
        if 'timestamp' in df_clean.columns:
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
            
        # Remove duplicates
        if 'post_id' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['post_id'], keep='first')
            
        return df_clean
        
    @staticmethod
    def calculate_features(df):
        """Calculate derived features"""
        df_features = df.copy()
        
        # Calculate engagement metrics
        if all(col in df_features.columns for col in ['retweets', 'likes']):
            df_features['total_engagement'] = df_features['retweets'] + df_features['likes']
            
            # Normalized engagement score (0-1)
            max_engagement = df_features['total_engagement'].max()
            if max_engagement > 0:
                df_features['engagement_score'] = df_features['total_engagement'] / max_engagement
            else:
                df_features['engagement_score'] = 0.0
                
        # Sentiment score mapping
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        if 'sentiment' in df_features.columns:
            df_features['sentiment_score'] = df_features['sentiment'].map(sentiment_map).fillna(0)
            
        # Extract date components
        if 'timestamp' in df_features.columns:
            df_features['created_year'] = pd.to_datetime(df_features['timestamp']).dt.year
            df_features['created_month'] = pd.to_datetime(df_features['timestamp']).dt.month
            df_features['created_day'] = pd.to_datetime(df_features['timestamp']).dt.day
            df_features['created_hour'] = pd.to_datetime(df_features['timestamp']).dt.hour
            
            # Peak hour identification (9 AM - 5 PM)
            df_features['is_peak_hour'] = df_features['created_hour'].between(9, 17).astype(int)
            
        return df_features

# ==================== ANALYSIS ENGINE ====================
class AnalysisEngine:
    """Perform comprehensive data analysis with FIXED queries"""
    
    def __init__(self, db_handler):
        self.db = db_handler
        
    def populate_aggregated_tables(self):
        """Populate all aggregated analysis tables with CORRECT queries"""
        try:
            print("   📊 Populating analysis tables...")
            
            # Clear existing aggregated data
            self.db.execute_query("DELETE FROM daily_metrics", return_df=False)
            self.db.execute_query("DELETE FROM platform_analysis", return_df=False)
            self.db.execute_query("DELETE FROM country_analysis", return_df=False)
            self.db.execute_query("DELETE FROM user_summary", return_df=False)
            
            # 1. Populate daily_metrics with CORRECT column names
            daily_query = """
            INSERT INTO daily_metrics 
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_posts,
                SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as negative_count,
                SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) as neutral_count,
                AVG(retweets) as avg_retweets,
                AVG(likes) as avg_likes,
                SUM(retweets + likes) as total_engagement,
                CASE 
                    WHEN SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) > 0 
                    THEN ROUND(1.0 * SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) / 
                              SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END), 2)
                    ELSE 0 
                END as sentiment_ratio,
                SUM(is_peak_hour) as peak_hour_posts,
                COUNT(DISTINCT user_id) as unique_users
            FROM social_media_posts
            WHERE timestamp IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date
            """
            self.db.execute_query(daily_query, return_df=False)
            print("     ✓ daily_metrics populated")
            
            # 2. Populate platform_analysis with CORRECT column names
            platform_query = """
            INSERT INTO platform_analysis 
            SELECT 
                platform,
                DATE(timestamp) as date,
                sentiment,
                COUNT(*) as post_count,
                AVG(retweets + likes) as avg_engagement
            FROM social_media_posts
            WHERE timestamp IS NOT NULL
            GROUP BY platform, DATE(timestamp), sentiment
            ORDER BY date, platform
            """
            self.db.execute_query(platform_query, return_df=False)
            print("     ✓ platform_analysis populated")
            
            # 3. Populate country_analysis with CORRECT column names
            country_query = """
            INSERT INTO country_analysis 
            SELECT 
                country,
                DATE(timestamp) as date,
                sentiment,
                COUNT(*) as post_count,
                AVG(retweets + likes) as avg_engagement
            FROM social_media_posts
            WHERE timestamp IS NOT NULL AND country IS NOT NULL AND country != 'Unknown'
            GROUP BY country, DATE(timestamp), sentiment
            ORDER BY country, date
            """
            self.db.execute_query(country_query, return_df=False)
            print("     ✓ country_analysis populated")
            
            # 4. Populate user_summary with CORRECT column names
            user_query = """
            INSERT INTO user_summary 
            WITH user_stats AS (
                SELECT 
                    user_id,
                    COUNT(*) as total_posts,
                    AVG(sentiment_score) as avg_sentiment_score,
                    SUM(retweets + likes) as total_engagement,
                    MAX(DATE(timestamp)) as last_post_date
                FROM social_media_posts
                GROUP BY user_id
            ),
            user_platforms AS (
                SELECT 
                    user_id,
                    platform,
                    COUNT(*) as platform_count,
                    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY COUNT(*) DESC) as rn
                FROM social_media_posts
                GROUP BY user_id, platform
            )
            SELECT 
                us.user_id,
                us.total_posts,
                us.avg_sentiment_score,
                us.total_engagement,
                up.platform as favorite_platform,
                us.last_post_date
            FROM user_stats us
            LEFT JOIN user_platforms up ON us.user_id = up.user_id AND up.rn = 1
            ORDER BY us.total_engagement DESC
            """
            self.db.execute_query(user_query, return_df=False)
            print("     ✓ user_summary populated")
            
            print("   ✅ All analysis tables populated successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Error populating tables: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def run_analysis_queries(self):
        """Execute comprehensive analysis queries with CORRECT column names"""
        try:
            print("\n   📈 Running analysis queries...")
            
            queries = {
                "1. Database Statistics": """
                SELECT 
                    COUNT(*) as total_posts,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT country) as countries_covered,
                    COUNT(DISTINCT platform) as platforms,
                    DATE(MIN(timestamp)) as analysis_start,
                    DATE(MAX(timestamp)) as analysis_end
                FROM social_media_posts
                """,
                
                "2. Sentiment Overview": """
                SELECT 
                    sentiment,
                    COUNT(*) as post_count,
                    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM social_media_posts), 2) as percentage,
                    AVG(retweets) as avg_retweets,
                    AVG(likes) as avg_likes,
                    AVG(engagement_score) as avg_engagement_score
                FROM social_media_posts
                GROUP BY sentiment
                ORDER BY post_count DESC
                """,
                
                "3. Platform Performance": """
                SELECT 
                    platform,
                    COUNT(*) as total_posts,
                    ROUND(100.0 * SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) / COUNT(*), 2) as positive_rate,
                    ROUND(100.0 * SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) / COUNT(*), 2) as negative_rate,
                    AVG(retweets + likes) as avg_engagement,
                    SUM(retweets + likes) as total_engagement
                FROM social_media_posts
                GROUP BY platform
                ORDER BY total_engagement DESC
                """,
                
                "4. Top Engaging Users": """
                SELECT 
                    user_id,
                    total_posts,
                    ROUND(avg_sentiment_score, 2) as avg_sentiment,
                    total_engagement,
                    favorite_platform,
                    last_post_date
                FROM user_summary
                ORDER BY total_engagement DESC
                LIMIT 10
                """,
                
                "5. Daily Trends": """
                SELECT 
                    date,
                    total_posts,
                    positive_count,
                    negative_count,
                    total_engagement,
                    sentiment_ratio
                FROM daily_metrics
                ORDER BY date
                LIMIT 10
                """,
                
                "6. Country Analysis": """
                SELECT 
                    country,
                    SUM(post_count) as total_posts,
                    ROUND(100.0 * SUM(CASE WHEN sentiment = 'Positive' THEN post_count ELSE 0 END) / SUM(post_count), 2) as positive_percentage,
                    ROUND(AVG(avg_engagement), 2) as avg_engagement
                FROM country_analysis
                GROUP BY country
                HAVING total_posts >= 2
                ORDER BY total_posts DESC
                LIMIT 10
                """
            }
            
            results = {}
            for title, query in queries.items():
                print(f"\n     {title}:")
                result = self.db.execute_query(query)
                if result is not None and not result.empty:
                    print(f"       Records: {len(result)}")
                    print(result.to_string(index=False))
                    results[title] = result
                else:
                    print(f"       No data returned")
                    
            return results
            
        except Exception as e:
            print(f"   ❌ Analysis queries failed: {e}")
            return {}

# ==================== VISUALIZATION ENGINE ====================
class VisualizationEngine:
    """Create professional visualizations with FIXED queries"""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_all_visualizations(self, db_handler):
        """Generate all visualization charts with CORRECT queries"""
        try:
            print("   🎨 Creating visualizations...")
            
            # Set style
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("husl")
            
            # 1. Sentiment Distribution
            sentiment_query = """
            SELECT sentiment, COUNT(*) as count 
            FROM social_media_posts 
            GROUP BY sentiment
            """
            sentiment_data = db_handler.execute_query(sentiment_query)
            
            if sentiment_data is not None and not sentiment_data.empty:
                plt.figure(figsize=(10, 6))
                colors = ['#4CAF50', '#FF9800', '#F44336']
                plt.pie(sentiment_data['count'], labels=sentiment_data['sentiment'], 
                       autopct='%1.1f%%', colors=colors, startangle=90, explode=[0.05, 0, 0])
                plt.title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
                plt.savefig(self.output_dir / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     ✓ sentiment_distribution.png")
            
            # 2. Platform Performance
            platform_query = """
            SELECT platform, COUNT(*) as posts, AVG(retweets + likes) as avg_engagement
            FROM social_media_posts
            GROUP BY platform
            ORDER BY posts DESC
            """
            platform_data = db_handler.execute_query(platform_query)
            
            if platform_data is not None and not platform_data.empty:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Posts by platform
                ax1.bar(platform_data['platform'], platform_data['posts'], color='#2196F3')
                ax1.set_title('Posts by Platform', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Platform')
                ax1.set_ylabel('Number of Posts')
                ax1.tick_params(axis='x', rotation=45)
                
                # Engagement by platform
                ax2.bar(platform_data['platform'], platform_data['avg_engagement'], color='#9C27B0')
                ax2.set_title('Average Engagement by Platform', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Platform')
                ax2.set_ylabel('Average Engagement')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'platform_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     ✓ platform_analysis.png")
            
            # 3. Daily Trends - USING CORRECT column names
            daily_query = """
            SELECT date, total_posts, positive_count, total_engagement
            FROM daily_metrics
            ORDER BY date
            """
            daily_data = db_handler.execute_query(daily_query)
            
            if daily_data is not None and not daily_data.empty:
                plt.figure(figsize=(14, 6))
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                
                # Plot posts
                ax = daily_data.plot(x='date', y='total_posts', kind='line', 
                                    marker='o', linewidth=2, color='#2196F3', 
                                    label='Total Posts', figsize=(14, 6))
                
                # Plot engagement (secondary axis)
                ax2 = ax.twinx()
                daily_data.plot(x='date', y='total_engagement', kind='line', 
                               marker='s', linewidth=2, color='#FF5722', 
                               label='Total Engagement', ax=ax2)
                
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Number of Posts', fontsize=12, color='#2196F3')
                ax2.set_ylabel('Total Engagement', fontsize=12, color='#FF5722')
                ax.set_title('Daily Posts and Engagement Trends', fontsize=16, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'daily_trends.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     ✓ daily_trends.png")
            
            # 4. Top Users
            users_query = """
            SELECT user_id, total_engagement, total_posts
            FROM user_summary
            ORDER BY total_engagement DESC
            LIMIT 10
            """
            users_data = db_handler.execute_query(users_query)
            
            if users_data is not None and not users_data.empty:
                plt.figure(figsize=(12, 6))
                bars = plt.barh(users_data['user_id'], users_data['total_engagement'], color='#607D8B')
                plt.xlabel('Total Engagement', fontsize=12)
                plt.title('Top 10 Users by Engagement', fontsize=16, fontweight='bold', pad=20)
                plt.gca().invert_yaxis()
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width, bar.get_y() + bar.get_height()/2, 
                            f'{int(width):,}', ha='left', va='center')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'top_users.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("     ✓ top_users.png")
            
            print("   ✅ All visualizations created successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Visualization creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# ==================== FEATURE ENGINEER ====================
class FeatureEngineer:
    """Create ML-ready feature sets with FIXED queries"""
    
    def __init__(self, output_dir, db_handler):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_handler
        
    def create_feature_sets(self):
        """Generate feature sets for machine learning with CORRECT column names"""
        try:
            print("   🤖 Creating feature sets...")
            
            # 1. Daily features - USING CORRECT column names
            daily_query = """
            SELECT 
                date,
                total_posts,
                positive_count,
                negative_count,
                neutral_count,
                avg_retweets,
                avg_likes,
                total_engagement,
                sentiment_ratio,
                peak_hour_posts,
                unique_users,
                ROUND(1.0 * positive_count / NULLIF(total_posts, 0), 3) as positive_ratio,
                ROUND(1.0 * negative_count / NULLIF(total_posts, 0), 3) as negative_ratio,
                ROUND(1.0 * peak_hour_posts / NULLIF(total_posts, 0), 3) as peak_hour_ratio
            FROM daily_metrics
            ORDER BY date
            """
            daily_features = self.db.execute_query(daily_query)
            if daily_features is not None:
                daily_features.to_csv(self.output_dir / "daily_features.csv", index=False)
                print(f"     ✓ daily_features.csv ({len(daily_features)} rows)")
            
            # 2. Platform features
            platform_query = """
            SELECT 
                platform,
                date,
                SUM(post_count) as daily_posts,
                AVG(avg_engagement) as avg_engagement,
                ROUND(1.0 * SUM(CASE WHEN sentiment = 'Positive' THEN post_count ELSE 0 END) / SUM(post_count), 3) as platform_positive_ratio
            FROM platform_analysis
            GROUP BY platform, date
            ORDER BY date, platform
            """
            platform_features = self.db.execute_query(platform_query)
            if platform_features is not None:
                platform_features.to_csv(self.output_dir / "platform_features.csv", index=False)
                print(f"     ✓ platform_features.csv ({len(platform_features)} rows)")
            
            # 3. User features
            user_query = """
            SELECT 
                user_id,
                total_posts,
                avg_sentiment_score,
                total_engagement,
                favorite_platform,
                ROUND(1.0 * total_engagement / NULLIF(total_posts, 0), 2) as engagement_per_post,
                CASE 
                    WHEN total_engagement > (SELECT AVG(total_engagement) * 2 FROM user_summary) THEN 'Top Influencer'
                    WHEN total_engagement > (SELECT AVG(total_engagement) FROM user_summary) THEN 'Active Contributor'
                    ELSE 'Regular User'
                END as user_segment
            FROM user_summary
            ORDER BY total_engagement DESC
            """
            user_features = self.db.execute_query(user_query)
            if user_features is not None:
                user_features.to_csv(self.output_dir / "user_features.csv", index=False)
                print(f"     ✓ user_features.csv ({len(user_features)} rows)")
            
            print("   ✅ Feature sets created successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Feature creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# ==================== REPORT GENERATOR ====================
class ReportGenerator:
    """Generate comprehensive HTML reports with FIXED queries"""
    
    def __init__(self, output_dir, db_handler):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_handler
        
    def generate_dashboard(self):
        """Generate interactive HTML dashboard with CORRECT queries"""
        try:
            print("   📋 Generating dashboard report...")
            
            # Get summary statistics
            stats_query = """
            SELECT 
                COUNT(*) as total_posts,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT country) as countries,
                COUNT(DISTINCT platform) as platforms,
                ROUND(100.0 * SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) / COUNT(*), 2) as positive_rate,
                AVG(retweets + likes) as avg_engagement
            FROM social_media_posts
            """
            stats_result = self.db.execute_query(stats_query)
            
            if stats_result is None or stats_result.empty:
                print("   ❌ No statistics data available")
                return False
                
            stats = stats_result.iloc[0]
            
            # Get top posts
            top_posts_query = """
            SELECT post_id, user_id, platform, sentiment, content, 
                   retweets, likes, (retweets + likes) as total_engagement
            FROM social_media_posts
            ORDER BY total_engagement DESC
            LIMIT 5
            """
            top_posts = self.db.execute_query(top_posts_query)
            
            # Get daily metrics - USING CORRECT column names
            daily_query = """
            SELECT date, total_posts, positive_count, negative_count, total_engagement
            FROM daily_metrics
            ORDER BY date DESC
            LIMIT 7
            """
            daily_data = self.db.execute_query(daily_query)
            
            # Generate HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Social Media Analytics Dashboard</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    }}
                    
                    body {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }}
                    
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                    }}
                    
                    .header {{
                        text-align: center;
                        margin-bottom: 30px;
                        color: white;
                    }}
                    
                    .header h1 {{
                        font-size: 2.5rem;
                        margin-bottom: 10px;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    }}
                    
                    .header p {{
                        font-size: 1.1rem;
                        opacity: 0.9;
                    }}
                    
                    .dashboard {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    
                    .card {{
                        background: white;
                        border-radius: 15px;
                        padding: 25px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                        transition: transform 0.3s ease;
                    }}
                    
                    .card:hover {{
                        transform: translateY(-5px);
                    }}
                    
                    .stat-card {{
                        text-align: center;
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    }}
                    
                    .stat-icon {{
                        font-size: 2.5rem;
                        margin-bottom: 15px;
                        color: #667eea;
                    }}
                    
                    .stat-value {{
                        font-size: 2.2rem;
                        font-weight: bold;
                        color: #2c3e50;
                        margin: 10px 0;
                    }}
                    
                    .stat-label {{
                        font-size: 1rem;
                        color: #7f8c8d;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                    }}
                    
                    .section-title {{
                        font-size: 1.5rem;
                        color: #2c3e50;
                        margin-bottom: 20px;
                        padding-bottom: 10px;
                        border-bottom: 2px solid #667eea;
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }}
                    
                    .section-title i {{
                        color: #667eea;
                    }}
                    
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 10px;
                    }}
                    
                    table th {{
                        background: #667eea;
                        color: white;
                        padding: 12px;
                        text-align: left;
                    }}
                    
                    table td {{
                        padding: 12px;
                        border-bottom: 1px solid #e0e0e0;
                    }}
                    
                    table tr:hover {{
                        background: #f5f7fa;
                    }}
                    
                    .sentiment-badge {{
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: bold;
                    }}
                    
                    .positive {{
                        background: #d4edda;
                        color: #155724;
                    }}
                    
                    .negative {{
                        background: #f8d7da;
                        color: #721c24;
                    }}
                    
                    .neutral {{
                        background: #fff3cd;
                        color: #856404;
                    }}
                    
                    .visualization-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                        gap: 20px;
                        margin: 30px 0;
                    }}
                    
                    .viz-card {{
                        background: white;
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    }}
                    
                    .viz-card img {{
                        width: 100%;
                        height: auto;
                        border-radius: 10px;
                    }}
                    
                    .footer {{
                        text-align: center;
                        margin-top: 40px;
                        padding: 20px;
                        color: white;
                        opacity: 0.8;
                    }}
                    
                    @media (max-width: 768px) {{
                        .dashboard {{
                            grid-template-columns: 1fr;
                        }}
                        
                        .visualization-grid {{
                            grid-template-columns: 1fr;
                        }}
                        
                        .header h1 {{
                            font-size: 2rem;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1><i class="fas fa-chart-line"></i> Social Media Analytics Dashboard</h1>
                        <p>Comprehensive Analysis of Social Media Sentiment and Engagement</p>
                        <p style="margin-top: 10px; font-size: 0.9rem;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="dashboard">
                        <div class="card stat-card">
                            <i class="fas fa-comment-dots stat-icon"></i>
                            <div class="stat-value">{int(stats['total_posts'])}</div>
                            <div class="stat-label">Total Posts</div>
                        </div>
                        
                        <div class="card stat-card">
                            <i class="fas fa-users stat-icon"></i>
                            <div class="stat-value">{int(stats['unique_users'])}</div>
                            <div class="stat-label">Unique Users</div>
                        </div>
                        
                        <div class="card stat-card">
                            <i class="fas fa-globe-americas stat-icon"></i>
                            <div class="stat-value">{int(stats['countries'])}</div>
                            <div class="stat-label">Countries</div>
                        </div>
                        
                        <div class="card stat-card">
                            <i class="fas fa-thumbs-up stat-icon"></i>
                            <div class="stat-value">{stats['positive_rate']:.1f}%</div>
                            <div class="stat-label">Positive Sentiment</div>
                        </div>
                    </div>
                    
                    <div class="visualization-grid">
                        <div class="viz-card">
                            <h3 class="section-title"><i class="fas fa-chart-pie"></i> Sentiment Distribution</h3>
                            <img src="visualizations/sentiment_distribution.png" alt="Sentiment Distribution">
                        </div>
                        
                        <div class="viz-card">
                            <h3 class="section-title"><i class="fas fa-chart-bar"></i> Platform Analysis</h3>
                            <img src="visualizations/platform_analysis.png" alt="Platform Analysis">
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3 class="section-title"><i class="fas fa-trophy"></i> Top Performing Posts</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Post ID</th>
                                    <th>User</th>
                                    <th>Platform</th>
                                    <th>Sentiment</th>
                                    <th>Retweets</th>
                                    <th>Likes</th>
                                    <th>Total Engagement</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add top posts rows
            if top_posts is not None and not top_posts.empty:
                for _, row in top_posts.iterrows():
                    sentiment_class = row['sentiment'].lower() if row['sentiment'] else 'neutral'
                    html_content += f"""
                                <tr>
                                    <td>{row['post_id']}</td>
                                    <td>{row['user_id']}</td>
                                    <td>{row['platform']}</td>
                                    <td><span class="sentiment-badge {sentiment_class}">{row['sentiment']}</span></td>
                                    <td>{int(row['retweets'])}</td>
                                    <td>{int(row['likes'])}</td>
                                    <td><strong>{int(row['total_engagement'])}</strong></td>
                                </tr>
                    """
            
            html_content += f"""
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card" style="margin-top: 20px;">
                        <h3 class="section-title"><i class="fas fa-calendar-alt"></i> Recent Daily Metrics</h3>
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Total Posts</th>
                                    <th>Positive Posts</th>
                                    <th>Negative Posts</th>
                                    <th>Total Engagement</th>
                                </tr>
                            </thead>
                            <tbody>
            """
            
            # Add daily metrics rows
            if daily_data is not None and not daily_data.empty:
                for _, row in daily_data.iterrows():
                    html_content += f"""
                                <tr>
                                    <td>{row['date']}</td>
                                    <td>{int(row['total_posts'])}</td>
                                    <td>{int(row['positive_count'])}</td>
                                    <td>{int(row['negative_count'])}</td>
                                    <td><strong>{int(row['total_engagement'])}</strong></td>
                                </tr>
                    """
            
            html_content += f"""
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="card" style="margin-top: 20px; background: linear-gradient(135deg, #e0f7fa 0%, #80deea 100%);">
                        <h3 class="section-title"><i class="fas fa-lightbulb"></i> Insights & Recommendations</h3>
                        <div style="padding: 15px; line-height: 1.6;">
                            <p><strong>📈 Key Findings:</strong></p>
                            <ul style="margin-left: 20px; margin-bottom: 15px;">
                                <li>Positive sentiment rate: <strong>{stats['positive_rate']:.1f}%</strong></li>
                                <li>Average engagement per post: <strong>{stats['avg_engagement']:.0f}</strong></li>
                                <li>Analysis covers: <strong>{int(stats['total_posts'])} posts</strong> from <strong>{int(stats['unique_users'])} users</strong></li>
                            </ul>
                            
                            <p><strong>🎯 Recommendations:</strong></p>
                            <ul style="margin-left: 20px;">
                                <li>Focus on platforms with highest engagement rates</li>
                                <li>Schedule posts during peak hours for maximum visibility</li>
                                <li>Monitor negative sentiment trends for brand protection</li>
                                <li>Engage with top users to build brand advocacy</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="footer">
                        <p>© 2024 Social Media Analytics Dashboard | Generated with ❤️ using Python</p>
                        <p style="font-size: 0.9rem; margin-top: 5px;">
                            <i class="fas fa-database"></i> Database: {int(stats['total_posts'])} posts | 
                            <i class="fas fa-user-friends"></i> {int(stats['unique_users'])} users | 
                            <i class="fas fa-chart-bar"></i> {int(stats['platforms'])} platforms
                        </p>
                    </div>
                </div>
                
                <script>
                    // Simple animation for stat cards
                    document.addEventListener('DOMContentLoaded', function() {{
                        const statCards = document.querySelectorAll('.stat-card');
                        statCards.forEach((card, index) => {{
                            setTimeout(() => {{
                                card.style.opacity = '1';
                                card.style.transform = 'translateY(0)';
                            }}, index * 100);
                        }});
                    }});
                </script>
            </body>
            </html>
            """
            
            # Save HTML file
            report_path = self.output_dir / "dashboard.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            print("     ✓ dashboard.html generated")
            
            # Generate simple text summary
            summary_content = f"""
            ===============================
            SOCIAL MEDIA ANALYSIS SUMMARY
            ===============================
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            📊 OVERVIEW STATISTICS:
            • Total Posts: {int(stats['total_posts'])}
            • Unique Users: {int(stats['unique_users'])}
            • Countries: {int(stats['countries'])}
            • Platforms: {int(stats['platforms'])}
            • Positive Rate: {stats['positive_rate']:.1f}%
            • Avg Engagement: {stats['avg_engagement']:.0f}
            
            📈 ANALYSIS COMPLETE
            Files Generated:
            1. Database: data/processed/social_media.db
            2. Dashboard: reports/dashboard.html  
            3. Visualizations: reports/visualizations/*.png
            4. Feature Sets: data/feature_sets/*.csv
            """
            
            summary_path = self.output_dir / "analysis_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)
                
            print("     ✓ analysis_summary.txt generated")
            print("   ✅ Dashboard generation completed")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Dashboard generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

# ==================== MAIN ANALYZER ====================
class SocialMediaAnalyzer:
    """Main orchestrator for social media analysis"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.raw_data_path = self.data_dir / "raw" / "social_media_data.csv"
        self.processed_data_path = self.data_dir / "processed" / "cleaned_data.csv"
        
        # Create required directories
        for path in [self.data_dir, self.data_dir / "raw", 
                     self.data_dir / "processed", self.data_dir / "feature_sets"]:
            path.mkdir(parents=True, exist_ok=True)
            
        # Initialize components
        self.db = DatabaseHandler(self.data_dir / "processed" / "social_media.db")
        
        print("=" * 70)
        print("🤖 SOCIAL MEDIA SENTIMENT ANALYZER - FIXED VERSION")
        print("=" * 70)
        
    def run_pipeline(self):
        """Execute complete analysis pipeline"""
        print("\n🚀 Starting analysis pipeline...")
        
        steps = [
            ("Environment Check", self.check_environment),
            ("Data Loading", self.load_data),
            ("Data Processing", self.process_data),
            ("Database Setup", self.setup_database),
            ("Analysis Engine", self.run_analysis),
            ("Visualization", self.create_visualizations),
            ("Feature Engineering", self.create_features),
            ("Reporting", self.generate_reports)
        ]
        
        results = []
        for name, func in steps:
            print(f"\n{'='*50}")
            print(f"📋 STEP: {name}")
            print(f"{'='*50}")
            
            try:
                success = func()
                results.append((name, success, "✅" if success else "❌"))
            except Exception as e:
                print(f"   ❌ Error in {name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False, "❌"))
        
        # Display summary
        print(f"\n{'='*70}")
        print("📊 ANALYSIS PIPELINE SUMMARY")
        print(f"{'='*70}")
        
        success_count = 0
        for name, success, symbol in results:
            status = "PASS" if success else "FAIL"
            print(f"{symbol} {name:<20} {status}")
            if success:
                success_count += 1
        
        print(f"\n📈 Success Rate: {success_count}/{len(steps)} steps completed")
        
        if success_count == len(steps):
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"📊 Dashboard: reports/dashboard.html")
            print(f"📈 Visualizations: reports/visualizations/")
            print(f"🤖 Features: data/feature_sets/")
            print(f"💾 Database: data/processed/social_media.db")
        else:
            print("\n⚠️ Analysis completed with some errors")
            
        self.db.close()
        return success_count == len(steps)
        
    def check_environment(self):
        """Check system environment and dependencies"""
        print("   🔍 Checking environment...")
        
        # Python version
        python_version = sys.version_info
        print(f"     Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Required packages
        required = ['pandas', 'numpy', 'matplotlib', 'seaborn']
        
        for package in required:
            try:
                __import__(package)
                print(f"     ✓ {package}")
            except ImportError:
                print(f"     ✗ {package}")
                return False
        
        # Check data file
        if not self.raw_data_path.exists():
            print(f"     ❌ Data file not found: {self.raw_data_path}")
            return False
        else:
            print(f"     ✓ Data file found")
            
        print("   ✅ Environment check completed")
        return True
        
    def load_data(self):
        """Load raw data from CSV"""
        print("   📥 Loading data...")
        
        try:
            self.raw_df = pd.read_csv(self.raw_data_path)
            print(f"     ✓ Loaded {len(self.raw_df)} records")
            return True
        except Exception as e:
            print(f"     ❌ Error loading data: {e}")
            return False
            
    def process_data(self):
        """Clean and process data"""
        print("   🧹 Processing data...")
        
        try:
            initial_rows = len(self.raw_df)
            
            # Clean data
            processor = DataProcessor()
            self.df = processor.clean_data(self.raw_df)
            
            # Calculate features
            self.df = processor.calculate_features(self.df)
            
            # Save processed data
            self.df.to_csv(self.processed_data_path, index=False)
            
            print(f"     ✓ Processed {len(self.df)} records")
            print(f"     ✓ Saved: {self.processed_data_path}")
            return True
        except Exception as e:
            print(f"     ❌ Error processing data: {e}")
            return False
            
    def setup_database(self):
        """Setup database and load data"""
        print("   💾 Setting up database...")
        
        try:
            # Connect to database
            if not self.db.connect():
                return False
                
            # Create schema
            if not self.db.create_schema():
                return False
                
            # Load data into main table
            if not self.db.insert_dataframe(self.df, 'social_media_posts'):
                return False
                
            print("     ✓ Database schema created")
            print("     ✓ Data loaded into social_media_posts")
            return True
        except Exception as e:
            print(f"     ❌ Database setup failed: {e}")
            return False
            
    def run_analysis(self):
        """Run comprehensive analysis"""
        print("   📊 Running analysis...")
        
        try:
            # Initialize analysis engine
            analyzer = AnalysisEngine(self.db)
            
            # Populate all analysis tables
            if not analyzer.populate_aggregated_tables():
                return False
                
            # Run analysis queries
            analysis_results = analyzer.run_analysis_queries()
            
            if not analysis_results:
                return False
                
            print("     ✅ Analysis completed successfully")
            return True
        except Exception as e:
            print(f"     ❌ Analysis failed: {e}")
            return False
            
    def create_visualizations(self):
        """Create visualization charts"""
        print("   🎨 Creating visualizations...")
        
        try:
            viz_dir = self.project_root / "reports" / "visualizations"
            viz_engine = VisualizationEngine(viz_dir)
            
            if not viz_engine.create_all_visualizations(self.db):
                return False
                
            print("     ✅ Visualizations created successfully")
            return True
        except Exception as e:
            print(f"     ❌ Visualization creation failed: {e}")
            return False
            
    def create_features(self):
        """Create feature sets for ML"""
        print("   🤖 Creating feature sets...")
        
        try:
            features_dir = self.project_root / "data" / "feature_sets"
            feature_engineer = FeatureEngineer(features_dir, self.db)
            
            if not feature_engineer.create_feature_sets():
                return False
                
            print("     ✅ Feature sets created successfully")
            return True
        except Exception as e:
            print(f"     ❌ Feature creation failed: {e}")
            return False
            
    def generate_reports(self):
        """Generate reports and dashboard"""
        print("   📋 Generating reports...")
        
        try:
            reports_dir = self.project_root / "reports"
            report_generator = ReportGenerator(reports_dir, self.db)
            
            if not report_generator.generate_dashboard():
                return False
                
            print("     ✅ Reports generated successfully")
            return True
        except Exception as e:
            print(f"     ❌ Report generation failed: {e}")
            return False

# ==================== MAIN EXECUTION ====================
def main():
    """Main entry point"""
    try:
        # Clean up project
        project_root = Path(__file__).parent
        
        # Remove unwanted directories
        unwanted = [".ipynb_checkpoints", "__pycache__"]
        for pattern in unwanted:
            for path in project_root.rglob(pattern):
                try:
                    if path.is_dir():
                        import shutil
                        shutil.rmtree(path, ignore_errors=True)
                except:
                    pass
        
        # Create required directories
        required_dirs = [
            "reports/visualizations",
            "data/feature_sets"
        ]
        
        for dir_path in required_dirs:
            (project_root / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        analyzer = SocialMediaAnalyzer()
        success = analyzer.run_pipeline()
        
        if success:
            print("\n" + "="*70)
            print("🎯 NEXT STEPS:")
            print("="*70)
            print("1. Open the interactive dashboard:")
            print("   📊 file://" + str(project_root / "reports" / "dashboard.html"))
            print("\n2. Explore feature sets for machine learning:")
            print("   🤖 data/feature_sets/")
            print("\n3. Check visualizations:")
            print("   🎨 reports/visualizations/")
            print("\n4. Query the database:")
            print("   💾 data/processed/social_media.db")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())