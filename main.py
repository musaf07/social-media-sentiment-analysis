#!/usr/bin/env python3
"""
Social Media Sentiment Analysis Pipeline
Complete end-to-end analysis with reporting
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sqlite3
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== SQLITE HANDLER ====================
class SQLiteHandler:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
    
    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    def create_tables(self, schema_file):
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            cursor = self.connection.cursor()
            cursor.executescript(schema_sql)
            self.connection.commit()
            return True
        except Exception as e:
            print(f"Table creation failed: {e}")
            return False
    
    def insert_dataframe(self, df, table_name, if_exists='replace'):
        try:
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            return True
        except Exception as e:
            print(f"Data insertion failed: {e}")
            return False
    
    def execute_query(self, query, return_df=True):
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
            print(f"Query execution failed: {e}")
            return None
    
    def close(self):
        if self.connection:
            self.connection.close()

# ==================== MAIN ANALYZER ====================
class SocialMediaAnalyzer:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.raw_data_path = self.data_dir / "raw" / "social_media_data.csv"
        self.processed_data_path = self.data_dir / "processed" / "cleaned_data.csv"
        
        # Create required directories
        for path in [self.data_dir, self.data_dir / "raw", 
                     self.data_dir / "processed", self.data_dir / "feature_sets"]:
            path.mkdir(exist_ok=True)
        
        # Initialize database
        self.db = SQLiteHandler(self.data_dir / "processed" / "social_media.db")
        
        print("=" * 60)
        print("SOCIAL MEDIA SENTIMENT ANALYZER")
        print("=" * 60)
    
    def check_environment(self):
        print("\n1. Checking environment...")
        
        # Check Python version
        python_version = sys.version_info
        print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check packages
        required = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'sqlite3']
        missing = []
        
        for package in required:
            try:
                __import__(package)
                print(f"   ✓ {package}")
            except ImportError:
                missing.append(package)
                print(f"   ✗ {package}")
        
        if missing:
            print(f"\n   Missing: {missing}")
            print("   Install: pip install " + " ".join(missing))
            return False
        
        # Check data
        if not self.raw_data_path.exists():
            print(f"   Creating sample data...")
            self._create_sample_data()
        
        print("   ✓ Environment OK")
        return True
    
    def _create_sample_data(self):
        """Create sample data if none exists"""
        np.random.seed(42)
        sample_data = {
            'post_id': range(1, 101),
            'user_id': [f'user_{i:03d}' for i in range(1, 101)],
            'platform': np.random.choice(['Twitter', 'Instagram', 'Facebook'], 100),
            'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], 100, p=[0.5, 0.3, 0.2]),
            'content': [f'Sample post {i}' for i in range(1, 101)],
            'hashtags': np.random.choice(['#tech', '#news', '#update', '#review', ''], 100),
            'retweets': np.random.randint(0, 1000, 100),
            'likes': np.random.randint(0, 5000, 100),
            'country': np.random.choice(['USA', 'UK', 'Canada', 'Australia', 'India'], 100),
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        }
        
        df = pd.DataFrame(sample_data)
        df['created_year'] = df['timestamp'].dt.year
        df['created_month'] = df['timestamp'].dt.month
        df['created_day'] = df['timestamp'].dt.day
        df['created_hour'] = df['timestamp'].dt.hour
        
        df.to_csv(self.raw_data_path, index=False)
        print(f"   Created sample data: {len(df)} records")
    
    def load_data(self):
        print("\n2. Loading data...")
        
        try:
            self.raw_df = pd.read_csv(self.raw_data_path)
            print(f"   ✓ Loaded {len(self.raw_df)} records")
            print(f"   Columns: {', '.join(self.raw_df.columns[:5])}...")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def clean_data(self):
        print("\n3. Cleaning data...")
        
        self.df = self.raw_df.copy()
        initial_rows = len(self.df)
        
        # Fill missing values
        if 'retweets' in self.df.columns:
            self.df['retweets'] = pd.to_numeric(self.df['retweets'], errors='coerce')
            self.df['retweets'] = self.df['retweets'].fillna(self.df['retweets'].median())
        
        if 'likes' in self.df.columns:
            self.df['likes'] = pd.to_numeric(self.df['likes'], errors='coerce')
            self.df['likes'] = self.df['likes'].fillna(self.df['likes'].median())
        
        self.df['country'] = self.df['country'].fillna('Unknown')
        self.df['hashtags'] = self.df['hashtags'].fillna('')
        
        # Convert timestamp
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
        
        # Remove duplicates
        self.df = self.df.drop_duplicates(subset=['post_id'], keep='first')
        
        # Create features
        if 'retweets' in self.df.columns and 'likes' in self.df.columns:
            max_val = self.df['retweets'].max() + self.df['likes'].max() + 1e-10
            self.df['engagement_score'] = (self.df['retweets'] + self.df['likes']) / max_val
        
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        self.df['sentiment_score'] = self.df['sentiment'].map(sentiment_map)
        
        # Save cleaned data
        self.df.to_csv(self.processed_data_path, index=False)
        
        print(f"   ✓ Cleaned {len(self.df)} records (removed {initial_rows - len(self.df)} duplicates)")
        print(f"   Saved: {self.processed_data_path}")
        return True
    
    def setup_database(self):
        print("\n4. Setting up database...")
        
        try:
            self.db.connect()
            
            # Create SQL directory
            sql_dir = self.project_root / "sql" / "sqlite"
            sql_dir.mkdir(parents=True, exist_ok=True)
            
            # Create schema
            schema_file = sql_dir / "create_tables.sql"
            if not schema_file.exists():
                schema = """
                CREATE TABLE IF NOT EXISTS social_media_posts (
                    post_id INTEGER PRIMARY KEY,
                    user_id TEXT,
                    platform TEXT,
                    sentiment TEXT,
                    content TEXT,
                    hashtags TEXT,
                    retweets INTEGER,
                    likes INTEGER,
                    country TEXT,
                    timestamp TIMESTAMP,
                    created_year INTEGER,
                    created_month INTEGER,
                    created_day INTEGER,
                    created_hour INTEGER,
                    engagement_score REAL,
                    sentiment_score INTEGER,
                    is_peak_hour INTEGER DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date DATE PRIMARY KEY,
                    total_posts INTEGER,
                    positive_count INTEGER,
                    negative_count INTEGER,
                    neutral_count INTEGER,
                    avg_retweets REAL,
                    avg_likes REAL,
                    total_engagement INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_timestamp ON social_media_posts(timestamp);
                CREATE INDEX IF NOT EXISTS idx_platform ON social_media_posts(platform);
                CREATE INDEX IF NOT EXISTS idx_sentiment ON social_media_posts(sentiment);
                """
                schema_file.write_text(schema)
            
            # Create tables and load data
            self.db.create_tables(str(schema_file))
            self.db.insert_dataframe(self.df, 'social_media_posts')
            
            # Set peak hours
            self.db.execute_query("""
                UPDATE social_media_posts 
                SET is_peak_hour = CASE 
                    WHEN created_hour BETWEEN 9 AND 17 THEN 1 
                    ELSE 0 
                END
            """, return_df=False)
            
            # Create daily metrics
            self.db.execute_query("DELETE FROM daily_metrics", return_df=False)
            self.db.execute_query("""
                INSERT INTO daily_metrics
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_posts,
                    SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) as neutral_count,
                    AVG(retweets) as avg_retweets,
                    AVG(likes) as avg_likes,
                    SUM(retweets + likes) as total_engagement
                FROM social_media_posts
                WHERE timestamp IS NOT NULL
                GROUP BY DATE(timestamp)
            """, return_df=False)
            
            print("   ✓ Database setup completed")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def analyze_data(self):
        print("\n5. Analyzing data...")
        
        try:
            # Create queries file
            sql_dir = self.project_root / "sql" / "sqlite"
            queries_file = sql_dir / "analysis_queries.sql"
            
            if not queries_file.exists():
                queries = """-- Query 1: Sentiment Summary
SELECT 
    sentiment,
    COUNT(*) as count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM social_media_posts), 1) as percentage,
    AVG(retweets) as avg_retweets,
    AVG(likes) as avg_likes
FROM social_media_posts
GROUP BY sentiment
ORDER BY count DESC;

-- Query 2: Platform Performance
SELECT 
    platform,
    COUNT(*) as posts,
    ROUND(100.0 * SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) / COUNT(*), 1) as positive_pct,
    AVG(retweets + likes) as avg_engagement
FROM social_media_posts
GROUP BY platform
ORDER BY posts DESC;

-- Query 3: Daily Trends
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as daily_posts,
    SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive_posts,
    AVG(retweets + likes) as avg_engagement
FROM social_media_posts
GROUP BY DATE(timestamp)
ORDER BY date;

-- Query 4: Top Posts
SELECT 
    post_id,
    platform,
    sentiment,
    content,
    retweets,
    likes,
    retweets + likes as total_engagement
FROM social_media_posts
ORDER BY total_engagement DESC
LIMIT 5;

-- Query 5: Country Analysis
SELECT 
    country,
    COUNT(*) as posts,
    ROUND(100.0 * SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) / COUNT(*), 1) as positive_pct
FROM social_media_posts
WHERE country != 'Unknown'
GROUP BY country
ORDER BY posts DESC;
"""
                queries_file.write_text(queries)
            
            # Execute queries
            with open(queries_file, 'r') as f:
                queries = f.read().split('-- Query')
            
            for i, query in enumerate(queries[1:], 1):
                query = '-- Query' + query.strip()
                if query:
                    print(f"\n   Query {i}:")
                    result = self.db.execute_query(query.split('\n', 1)[1])
                    if result is not None and not result.empty:
                        print(f"   Shape: {result.shape}")
                        print(result.head().to_string())
            
            # Basic statistics
            print("\n   Basic Statistics:")
            print(f"   - Total posts: {len(self.df)}")
            print(f"   - Unique users: {self.df['user_id'].nunique()}")
            print(f"   - Platforms: {', '.join(self.df['platform'].unique())}")
            
            if 'sentiment' in self.df.columns:
                counts = self.df['sentiment'].value_counts()
                for sentiment, count in counts.items():
                    pct = (count / len(self.df)) * 100
                    print(f"   - {sentiment}: {count} ({pct:.1f}%)")
            
            print("\n   ✓ Analysis completed")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def create_visualizations(self):
        print("\n6. Creating visualizations...")
        
        try:
            viz_dir = self.project_root / "reports" / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
            
            # 1. Sentiment Pie Chart
            plt.figure(figsize=(8, 6))
            sentiment_counts = self.df['sentiment'].value_counts()
            colors = ['#4CAF50', '#FF9800', '#F44336']  # Green, Orange, Red
            plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Sentiment Distribution', fontsize=14, fontweight='bold')
            plt.savefig(viz_dir / 'sentiment_pie.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ sentiment_pie.png")
            
            # 2. Platform Bar Chart
            plt.figure(figsize=(10, 6))
            platform_data = self.df.groupby('platform').size()
            platform_data.plot(kind='bar', color='#2196F3')
            plt.title('Posts by Platform', fontsize=14, fontweight='bold')
            plt.xlabel('Platform')
            plt.ylabel('Number of Posts')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(viz_dir / 'platform_bars.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ platform_bars.png")
            
            # 3. Engagement by Sentiment
            if 'retweets' in self.df.columns and 'likes' in self.df.columns:
                plt.figure(figsize=(10, 6))
                engagement = self.df.groupby('sentiment')[['retweets', 'likes']].mean()
                engagement.plot(kind='bar')
                plt.title('Average Engagement by Sentiment', fontsize=14, fontweight='bold')
                plt.xlabel('Sentiment')
                plt.ylabel('Average Count')
                plt.legend(['Retweets', 'Likes'])
                plt.tight_layout()
                plt.savefig(viz_dir / 'engagement_bars.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("   ✓ engagement_bars.png")
            
            # 4. Daily Trends
            if 'timestamp' in self.df.columns:
                plt.figure(figsize=(12, 5))
                self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
                daily = self.df.groupby('date').size()
                plt.plot(daily.index, daily.values, marker='o', linewidth=2, color='#9C27B0')
                plt.title('Daily Post Volume', fontsize=14, fontweight='bold')
                plt.xlabel('Date')
                plt.ylabel('Posts')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(viz_dir / 'daily_trends.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("   ✓ daily_trends.png")
            
            print(f"   All visualizations saved to: {viz_dir}")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def prepare_features(self):
        print("\n7. Preparing features for modeling...")
        
        try:
            features_dir = self.project_root / "data" / "feature_sets"
            features_dir.mkdir(exist_ok=True)
            
            # Feature 1: Daily aggregates
            daily_query = """
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total_posts,
                SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive_count,
                SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as negative_count,
                AVG(retweets) as avg_retweets,
                AVG(likes) as avg_likes,
                SUM(retweets + likes) as total_engagement
            FROM social_media_posts
            WHERE timestamp IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date
            """
            
            daily_features = self.db.execute_query(daily_query)
            if daily_features is not None:
                daily_features.to_csv(features_dir / "daily_features.csv", index=False)
                print(f"   ✓ daily_features.csv ({len(daily_features)} rows)")
            
            # Feature 2: Platform daily
            platform_query = """
            SELECT 
                DATE(timestamp) as date,
                platform,
                COUNT(*) as posts,
                AVG(retweets + likes) as avg_engagement
            FROM social_media_posts
            GROUP BY DATE(timestamp), platform
            ORDER BY date, platform
            """
            
            platform_features = self.db.execute_query(platform_query)
            if platform_features is not None:
                platform_features.to_csv(features_dir / "platform_features.csv", index=False)
                print(f"   ✓ platform_features.csv ({len(platform_features)} rows)")
            
            print("\n   Feature sets ready for ML modeling")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def generate_report(self):
        print("\n8. Generating report...")
        
        try:
            report_dir = self.project_root / "reports"
            report_dir.mkdir(exist_ok=True)
            
            # Calculate metrics
            total_posts = len(self.df)
            unique_users = self.df['user_id'].nunique()
            platforms = self.df['platform'].nunique()
            
            if 'sentiment' in self.df.columns:
                sentiment_counts = self.df['sentiment'].value_counts()
                positive = sentiment_counts.get('Positive', 0)
                negative = sentiment_counts.get('Negative', 0)
                neutral = sentiment_counts.get('Neutral', 0)
                positive_pct = (positive / total_posts) * 100 if total_posts > 0 else 0
            else:
                positive = negative = neutral = positive_pct = 0
            
            # Get date range
            if 'timestamp' in self.df.columns:
                try:
                    start_date = self.df['timestamp'].min().strftime('%Y-%m-%d')
                    end_date = self.df['timestamp'].max().strftime('%Y-%m-%d')
                except:
                    start_date = end_date = "N/A"
            else:
                start_date = end_date = "N/A"
            
            # HTML Report
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Social Media Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0; }}
        .metric-box {{ background: #f8f9fa; border: 1px solid #ddd; border-radius: 8px; padding: 20px; min-width: 200px; }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
        .insights {{ background: #e8f4fc; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .viz {{ text-align: center; margin: 30px 0; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
        footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #7f8c8d; }}
    </style>
</head>
<body>
    <h1>Social Media Sentiment Analysis Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="metrics">
        <div class="metric-box">
            <div class="metric-value">{total_posts}</div>
            <div class="metric-label">Total Posts</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{unique_users}</div>
            <div class="metric-label">Unique Users</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{positive_pct:.1f}%</div>
            <div class="metric-label">Positive Sentiment</div>
        </div>
        <div class="metric-box">
            <div class="metric-value">{platforms}</div>
            <div class="metric-label">Platforms</div>
        </div>
    </div>
    
    <div class="insights">
        <h3>Key Insights</h3>
        <p>• Period: {start_date} to {end_date}</p>
        <p>• Positive posts: {positive} ({positive_pct:.1f}%)</p>
        <p>• Negative posts: {negative} ({(negative/total_posts*100):.1f}%)</p>
        <p>• Neutral posts: {neutral} ({(neutral/total_posts*100):.1f}%)</p>
    </div>
    
    <h3>Visualizations</h3>
    <div class="viz">
        <img src="visualizations/sentiment_pie.png" alt="Sentiment Distribution">
        <p><em>Figure 1: Sentiment Distribution</em></p>
    </div>
    
    <div class="viz">
        <img src="visualizations/platform_bars.png" alt="Platform Distribution">
        <p><em>Figure 2: Posts by Platform</em></p>
    </div>
    
    <div class="insights">
        <h3>Next Steps</h3>
        <ol>
            <li>Use feature sets in <code>data/feature_sets/</code> for ML modeling</li>
            <li>Correlate sentiment with sales data</li>
            <li>Monitor negative sentiment trends</li>
            <li>Optimize posting times based on engagement</li>
        </ol>
    </div>
    
    <footer>
        <p>Analysis completed successfully. Ready for predictive modeling.</p>
    </footer>
</body>
</html>"""
            
            # Save HTML report
            report_path = report_dir / "analysis_report.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Simple text summary
            summary = f"""# Analysis Summary
Date: {datetime.now().strftime('%Y-%m-%d')}
Total Posts: {total_posts}
Positive: {positive} ({positive_pct:.1f}%)
Negative: {negative} ({(negative/total_posts*100):.1f}%)
Neutral: {neutral} ({(neutral/total_posts*100):.1f}%)
Period: {start_date} to {end_date}
Platforms: {platforms}
Unique Users: {unique_users}
"""
            
            summary_path = report_dir / "summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"   ✓ analysis_report.html")
            print(f"   ✓ summary.txt")
            return True
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_analysis(self):
        print("\n" + "=" * 60)
        print("STARTING ANALYSIS PIPELINE")
        print("=" * 60)
        
        steps = [
            ("Environment", self.check_environment),
            ("Data Loading", self.load_data),
            ("Data Cleaning", self.clean_data),
            ("Database", self.setup_database),
            ("Analysis", self.analyze_data),
            ("Visualization", self.create_visualizations),
            ("Feature Prep", self.prepare_features),
            ("Reporting", self.generate_report)
        ]
        
        results = []
        for name, func in steps:
            print(f"\n[{name}]")
            try:
                if func():
                    results.append((name, True, "✓"))
                else:
                    results.append((name, False, "✗"))
            except Exception as e:
                print(f"   Error: {e}")
                results.append((name, False, "✗"))
        
        # Summary
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        success_count = 0
        for name, success, symbol in results:
            status = "PASS" if success else "FAIL"
            print(f"{symbol} {name}: {status}")
            if success:
                success_count += 1
        
        self.db.close()
        
        print(f"\nSuccess: {success_count}/{len(steps)} steps completed")
        
        if success_count == len(steps):
            print("\n✅ Analysis completed successfully!")
            print("📊 Open: reports/analysis_report.html")
            print("📈 Check: reports/visualizations/")
            print("🤖 Use: data/feature_sets/ for ML")
        else:
            print("\n⚠️ Analysis completed with some errors")

# ==================== MAIN EXECUTION ====================
def clean_project():
    """Remove unwanted files and directories"""
    project_root = Path(__file__).parent
    
    # Remove unwanted directories
    unwanted = [".ipynb_checkpoints", "__pycache__", ".vscode", ".idea"]
    for pattern in unwanted:
        for path in project_root.rglob(pattern):
            try:
                if path.is_dir():
                    import shutil
                    shutil.rmtree(path)
            except:
                pass
    
    # Create clean structure
    dirs = [
        "data/raw",
        "data/processed",
        "data/feature_sets",
        "reports/visualizations",
        "sql/sqlite"
    ]
    
    for dir_path in dirs:
        (project_root / dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create requirements if missing
    req_file = project_root / "requirements.txt"
    if not req_file.exists():
        req_file.write_text("pandas\nnumpy\nmatplotlib\nseaborn\n")

if __name__ == "__main__":
    clean_project()
    analyzer = SocialMediaAnalyzer()
    analyzer.run_analysis()