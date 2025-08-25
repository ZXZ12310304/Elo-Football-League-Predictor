import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# config
YEAR = 2025  


# Defining English Column Name Mapping
COLUMN_MAPPING = {
    '轮次': 'round',
    '场序': 'match_id',
    '日期': 'date',
    '星期': 'weekday',
    '开球时间': 'time',
    '主队': 'home_team',
    '客队': 'away_team',
    '体育场': 'stadium',
    '主队得分': 'home_score',
    '客队得分': 'away_score'
}

def process_data(df):
    """precess data"""
    df = df.copy()
    
    df = df.rename(columns=COLUMN_MAPPING)
    
    # Delete unneeded columns
    columns_to_drop = ['weekday', 'time', 'stadium']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Process round number
    df['round'] = df['round'].str.extract('(\d+)').astype(int)
    
    # Process date format
    def convert_date(date_str):
        if isinstance(date_str, str):
            if '月' in date_str and '日' in date_str:
                month = int(date_str.split('月')[0])
                day = int(date_str.split('月')[1].split('日')[0])
                return f"{YEAR}-{month:02d}-{day:02d}"
        return date_str
    
    df['date'] = df['date'].apply(convert_date)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create team ID mapping
    all_teams = sorted(set(df['home_team'].unique()) | set(df['away_team'].unique()))
    team_mapping = {team: idx for idx, team in enumerate(all_teams)}
    team_mapping_reverse = {idx: team for team, idx in team_mapping.items()}
    
    # Save team mapping dictionary
    with open('data/team_mapping.json', 'w', encoding='utf-8') as f:
        json.dump({'team_to_id': team_mapping, 'id_to_team': team_mapping_reverse}, 
                 f, ensure_ascii=False, indent=2)
    
    df['home_team_id'] = df['home_team'].map(team_mapping)
    df['away_team_id'] = df['away_team'].map(team_mapping)
    
    df['result'] = ''  
    # Only process games where both scores are not empty
    mask = df['home_score'].notna() & df['away_score'].notna()
    df.loc[mask & (df['home_score'] > df['away_score']), 'result'] = '0'  
    df.loc[mask & (df['home_score'] < df['away_score']), 'result'] = '1'  
    df.loc[mask & (df['home_score'] == df['away_score']), 'result'] = '2' 
    
    columns_order = [
        'round', 'match_id', 'date',
        'home_team_id', 'away_team_id',
        'home_score', 'away_score',
        'result'
    ]
    
    df = df[columns_order]
    
    return df

def split_and_save_data(processed_df):
    """split data into train and test set and save"""
    test_mask = processed_df['home_score'].isna() | processed_df['away_score'].isna()
    
    test_df = processed_df[test_mask]
    train_df = processed_df[~test_mask]
    
    train_df.to_csv('data/processed_train.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('data/processed_test.csv', index=False, encoding='utf-8-sig')
    
    return train_df, test_df

def main():
    input_file = 'data/data.csv'
    
    try:
        print(f"Reading file: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"Error: Original data file not found: {input_file}")
        print("Please ensure 'data/data.csv' file exists in the project.")
        return
    
    processed_df = process_data(df)
    
    train_df, test_df = split_and_save_data(processed_df)
    
    print(f"\nData has been processed and saved to data/ directory:")
    print(f"1. Train set: data/processed_train.csv")
    print(f"2. Test set: data/processed_test.csv")
    print(f"3. Team ID mapping: data/team_mapping.json")
    
    # Show data statistics
    print("\nData statistics:")
    print(f"Total games: {len(processed_df)}")
    print(f"Train set games: {len(train_df)}")
    print(f"Test set games: {len(test_df)}")
    print(f"Rounds: {processed_df['round'].nunique()}")
    print(f"Teams: {len(set(processed_df['home_team_id']) | set(processed_df['away_team_id']))}")
    
    # Show train set game result statistics
    print("\nTrain set game result statistics:")
    result_counts = train_df['result'].value_counts().sort_index()
    print(f"Home team wins: {result_counts.get('0', 0)} games")
    print(f"Away team wins: {result_counts.get('1', 0)} games")
    print(f"Draws: {result_counts.get('2', 0)} games")

if __name__ == '__main__':
    main() 