
import pandas as pd
import numpy as np

class AnalysisTools:
    def __init__(self):
        pass
    
    def calculate_engagement_score(self, df):
        if 'retweets' in df.columns and 'likes' in df.columns:
            df['engagement_score'] = (df['retweets'] + df['likes']) / (
                df['retweets'].max() + df['likes'].max() + 1e-10
            )
        return df
    
    def get_sentiment_distribution(self, df):
        return df['sentiment'].value_counts()
    
    def get_platform_stats(self, df):
        return df.groupby('platform').agg({
            'post_id': 'count',
            'retweets': 'mean',
            'likes': 'mean'
        }).round(2)
