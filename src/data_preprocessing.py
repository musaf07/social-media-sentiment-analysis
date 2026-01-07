
import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self):
        pass
    
    def clean_data(self, df):
        # Make a copy
        cleaned_df = df.copy()
        
        # Handle missing values
        if 'retweets' in cleaned_df.columns:
            cleaned_df['retweets'] = pd.to_numeric(cleaned_df['retweets'], errors='coerce')
            cleaned_df['retweets'] = cleaned_df['retweets'].fillna(cleaned_df['retweets'].median())
        
        if 'likes' in cleaned_df.columns:
            cleaned_df['likes'] = pd.to_numeric(cleaned_df['likes'], errors='coerce')
            cleaned_df['likes'] = cleaned_df['likes'].fillna(cleaned_df['likes'].median())
        
        # Fill other missing values
        cleaned_df['country'] = cleaned_df['country'].fillna('Unknown')
        cleaned_df['hashtags'] = cleaned_df['hashtags'].fillna('')
        
        # Convert timestamp
        if 'timestamp' in cleaned_df.columns:
            cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'], errors='coerce')
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates(subset=['post_id'], keep='first')
        
        return cleaned_df
