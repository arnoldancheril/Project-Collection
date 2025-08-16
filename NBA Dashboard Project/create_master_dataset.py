#!/usr/bin/env python3
"""
NBA Master Dataset Script

This script combines all 21 NBA shot CSV files (2004-2024) into a single dataset

"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import json

def validate_csv_structure(file_path):
    """Validate that CSV has expected columns"""
    try:
        df = pd.read_csv(file_path, nrows=1)
        expected_columns = [
            'TEAM_NAME', 'PLAYER_NAME', 'POSITION_GROUP', 'POSITION', 
            'HOME_TEAM', 'AWAY_TEAM', 'SEASON_1', 'SEASON_2',
            'TEAM_ID', 'PLAYER_ID', 'GAME_DATE', 'GAME_ID',
            'EVENT_TYPE', 'SHOT_MADE', 'ACTION_TYPE', 'SHOT_TYPE',
            'BASIC_ZONE', 'ZONE_NAME', 'ZONE_ABB', 'ZONE_RANGE',
            'LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'QUARTER',
            'MINS_LEFT', 'SECS_LEFT'
        ]
        
        # Check if at least the core columns exist
        core_columns = ['PLAYER_NAME', 'SHOT_TYPE', 'SHOT_MADE', 'SEASON_1', 'SEASON_2']
        missing_core = [col for col in core_columns if col not in df.columns]
        
        if missing_core:
            print(f"Warning: {file_path} missing core columns: {missing_core}")
            return False, df.columns.tolist()
        
        return True, df.columns.tolist()
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, []

def combine_nba_datasets():
    """Combine all NBA shot CSV files into a master dataset"""
    
    print("🏀 NBA Master Dataset Creation")
    print("=" * 50)
    
    # Find all NBA CSV files
    data_files = glob.glob("Data/NBA_*_Shots.csv")
    data_files.sort()
    
    if not data_files:
        print("❌ No NBA CSV files found in Data/ directory!")
        return None
    
    print(f"📁 Found {len(data_files)} NBA shot files")
    
    # Validate all files first
    print("\n🔍 Validating file structures...")
    valid_files = []
    all_columns = set()
    
    for file_path in data_files:
        year = file_path.split('_')[1]  # Extract year from filename
        print(f"  Checking {year}...", end=' ')
        
        is_valid, columns = validate_csv_structure(file_path)
        if is_valid:
            valid_files.append(file_path)
            all_columns.update(columns)
            print("✅")
        else:
            print("❌")
    
    if not valid_files:
        print("❌ No valid CSV files found!")
        return None
    
    print(f"\n✅ {len(valid_files)} files validated successfully")
    print(f"📊 Total unique columns found: {len(all_columns)}")
    
    # Combine all files
    print("\n📊 Combining datasets...")
    master_df = pd.DataFrame()
    
    for file_path in tqdm(valid_files, desc="Processing files"):
        try:
            # Read CSV
            df = pd.read_csv(file_path)
            
            # Add metadata
            year = int(file_path.split('_')[1])
            df['FILE_YEAR'] = year
            df['DATA_SOURCE'] = f"NBA_{year}_Shots.csv"
            
            # Standardize season format
            if 'SEASON_1' in df.columns and 'SEASON_2' in df.columns:
                df['SEASON'] = df['SEASON_1'].astype(str) + '-' + df['SEASON_2'].astype(str).str[-2:]
            else:
                df['SEASON'] = f"{year}-{str(year+1)[-2:]}"
            
            # Combine with master
            if master_df.empty:
                master_df = df.copy()
            else:
                master_df = pd.concat([master_df, df], ignore_index=True, sort=False)
                
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue
    
    if master_df.empty:
        print("❌ Failed to create master dataset!")
        return None
    
    print(f"\n🎉 Master dataset created successfully!")
    print(f"📊 Total shots: {len(master_df):,}")
    print(f"📅 Years covered: {master_df['FILE_YEAR'].min()} - {master_df['FILE_YEAR'].max()}")
    print(f"🏀 Unique players: {master_df['PLAYER_NAME'].nunique():,}")
    print(f"🏟️ Unique teams: {master_df['TEAM_NAME'].nunique() if 'TEAM_NAME' in master_df.columns else 'N/A'}")
    
    return master_df

def create_enhanced_analysis_datasets(master_df):
    """Create additional analysis-ready datasets for the explorer"""
    
    print("\n🔬 Creating enhanced analysis datasets...")
    
    # 1. Player Career Analytics
    print("  Creating player career analytics...")
    player_career = master_df.groupby(['PLAYER_NAME', 'FILE_YEAR']).agg({
        'SHOT_MADE': ['sum', 'count'],
        'SHOT_TYPE': lambda x: (x == '3PT Field Goal').sum(),
        'BASIC_ZONE': lambda x: {
            'three_point': (x.isin(['Left Corner 3', 'Right Corner 3', 'Above the Break 3'])).sum(),
            'mid_range': (x == 'Mid-Range').sum(),
            'paint': (x.isin(['Restricted Area', 'In The Paint (Non-RA)'])).sum()
        },
        'SHOT_DISTANCE': 'mean',
        'QUARTER': 'count'  # Total shots per quarter
    }).round(2)
    
    player_career.columns = ['makes', 'attempts', 'three_point_attempts', 'zone_breakdown', 'avg_distance', 'total_shots']
    player_career = player_career.reset_index()
    
    # 2. Team Season Analytics
    print("  Creating team season analytics...")
    if 'TEAM_NAME' in master_df.columns:
        team_season = master_df.groupby(['TEAM_NAME', 'FILE_YEAR']).agg({
            'SHOT_MADE': ['sum', 'count'],
            'SHOT_TYPE': lambda x: (x == '3PT Field Goal').sum(),
            'PLAYER_NAME': 'nunique',
            'GAME_ID': 'nunique' if 'GAME_ID' in master_df.columns else 'count'
        }).round(2)
        
        team_season.columns = ['makes', 'attempts', 'three_point_attempts', 'unique_players', 'games_played']
        team_season = team_season.reset_index()
    else:
        team_season = pd.DataFrame()
    
    # 3. Advanced Shot Analytics
    print("  Creating advanced shot analytics...")
    shot_analytics = master_df.groupby(['SHOT_TYPE', 'BASIC_ZONE', 'FILE_YEAR']).agg({
        'SHOT_MADE': ['sum', 'count', 'mean'],
        'SHOT_DISTANCE': ['mean', 'std'],
        'PLAYER_NAME': 'nunique'
    }).round(3)
    
    shot_analytics.columns = ['makes', 'attempts', 'fg_percentage', 'avg_distance', 'distance_std', 'unique_players']
    shot_analytics = shot_analytics.reset_index()
    
    # 4. Game Situation Analytics
    print("  Creating game situation analytics...")
    if all(col in master_df.columns for col in ['QUARTER', 'MINS_LEFT', 'SECS_LEFT']):
        master_df['TIME_REMAINING'] = master_df['MINS_LEFT'] * 60 + master_df['SECS_LEFT']
        master_df['GAME_PERIOD'] = master_df['QUARTER'].apply(lambda x: 
            'Q1' if x == 1 else 'Q2' if x == 2 else 'Q3' if x == 3 else 'Q4' if x == 4 else 'OT'
        )
        
        situation_analytics = master_df.groupby(['GAME_PERIOD', 'SHOT_TYPE', 'FILE_YEAR']).agg({
            'SHOT_MADE': ['sum', 'count', 'mean'],
            'TIME_REMAINING': 'mean'
        }).round(3)
        
        situation_analytics.columns = ['makes', 'attempts', 'fg_percentage', 'avg_time_remaining']
        situation_analytics = situation_analytics.reset_index()
    else:
        situation_analytics = pd.DataFrame()
    
    return {
        'player_career': player_career,
        'team_season': team_season,
        'shot_analytics': shot_analytics,
        'situation_analytics': situation_analytics
    }

def save_datasets(master_df, analysis_datasets):
    """Save all datasets in multiple formats"""
    
    print("\n💾 Saving datasets...")
    
    # Create output directory
    os.makedirs('data/master', exist_ok=True)
    
    # Save master dataset (CSV for full data)
    print("  Saving master CSV...")
    master_df.to_csv('data/master/nba_master_shots_2004_2024.csv', index=False)
    
    # Save compressed version for web
    print("  Saving compressed master dataset...")
    master_sample = master_df.sample(n=min(100000, len(master_df)), random_state=42)
    master_sample.to_csv('data/master/nba_master_shots_sample.csv', index=False)
    
    # Save analysis datasets as JSON for web consumption
    print("  Saving analysis datasets...")
    for name, df in analysis_datasets.items():
        if not df.empty:
            # Convert to JSON format
            json_data = df.to_dict('records')
            
            # Save as JSON
            with open(f'data/master/{name}.json', 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            # Save as CSV
            df.to_csv(f'data/master/{name}.csv', index=False)
    
    # Create metadata file
    print("  Creating metadata...")
    metadata = {
        'creation_date': pd.Timestamp.now().isoformat(),
        'total_shots': len(master_df),
        'date_range': {
            'start_year': int(master_df['FILE_YEAR'].min()),
            'end_year': int(master_df['FILE_YEAR'].max())
        },
        'unique_players': int(master_df['PLAYER_NAME'].nunique()),
        'unique_teams': int(master_df['TEAM_NAME'].nunique()) if 'TEAM_NAME' in master_df.columns else 0,
        'columns': master_df.columns.tolist(),
        'file_sizes': {
            'master_csv_mb': round(os.path.getsize('data/master/nba_master_shots_2004_2024.csv') / 1024 / 1024, 2),
            'sample_csv_mb': round(os.path.getsize('data/master/nba_master_shots_sample.csv') / 1024 / 1024, 2)
        },
        'analysis_datasets': list(analysis_datasets.keys())
    }
    
    with open('data/master/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ All datasets saved to data/master/")
    print(f"📁 Master CSV: {metadata['file_sizes']['master_csv_mb']} MB")
    print(f"📁 Sample CSV: {metadata['file_sizes']['sample_csv_mb']} MB")
    
    return metadata

def main():
    """Main execution function"""
    
    print("🚀 Starting NBA Master Dataset Creation")
    print("This will combine all NBA shot data (2004-2024) into comprehensive datasets")
    print("for enhanced exploration and analysis.\n")
    
    # Step 1: Combine all CSV files
    master_df = combine_nba_datasets()
    if master_df is None:
        print("❌ Failed to create master dataset!")
        return
    
    # Step 2: Create enhanced analysis datasets
    analysis_datasets = create_enhanced_analysis_datasets(master_df)
    
    # Step 3: Save all datasets
    metadata = save_datasets(master_df, analysis_datasets)
    
    print("\n🎉 NBA Master Dataset Creation Complete!")
    print(f"🏀 Total shots processed: {metadata['total_shots']:,}")
    print(f"📅 Years: {metadata['date_range']['start_year']}-{metadata['date_range']['end_year']}")
    print(f"🏀 Players: {metadata['unique_players']:,}")
    print(f"🏟️ Teams: {metadata['unique_teams']}")
    
    print("\n📁 Files created:")
    print("  • data/master/nba_master_shots_2004_2024.csv (Complete dataset)")
    print("  • data/master/nba_master_shots_sample.csv (Sample for web)")
    print("  • data/master/player_career.json (Player analytics)")
    print("  • data/master/team_season.json (Team analytics)")
    print("  • data/master/shot_analytics.json (Shot type analytics)")
    print("  • data/master/situation_analytics.json (Game situation analytics)")
    print("  • data/master/metadata.json (Dataset information)")
    
    print("\n🎯 Ready for enhanced NBA visualization exploration!")

if __name__ == "__main__":
    main()
