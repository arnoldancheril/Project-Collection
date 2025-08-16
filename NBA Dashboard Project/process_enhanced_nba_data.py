#!/usr/bin/env python3
"""
Enhanced NBA Data Processor
Creates comprehensive data with top 100 players and conference-organized teams
"""

import pandas as pd
import json
import glob
from collections import defaultdict

# NBA Conference structure (2024 alignment)
NBA_CONFERENCES = {
    "Eastern Conference": {
        "Atlantic": ["Boston Celtics", "Brooklyn Nets", "New York Knicks", "Philadelphia 76ers", "Toronto Raptors"],
        "Central": ["Chicago Bulls", "Cleveland Cavaliers", "Detroit Pistons", "Indiana Pacers", "Milwaukee Bucks"],
        "Southeast": ["Atlanta Hawks", "Charlotte Hornets", "Miami Heat", "Orlando Magic", "Washington Wizards"]
    },
    "Western Conference": {
        "Northwest": ["Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder", "Portland Trail Blazers", "Utah Jazz"],
        "Pacific": ["Golden State Warriors", "LA Clippers", "Los Angeles Lakers", "Phoenix Suns", "Sacramento Kings"],
        "Southwest": ["Dallas Mavericks", "Houston Rockets", "Memphis Grizzlies", "New Orleans Pelicans", "San Antonio Spurs"]
    }
}

def process_enhanced_player_data():
    """Process player data to get top 100 three-point shooters."""
    print("🏀 Processing enhanced player data for top 100 shooters...")
    
    # Load existing comprehensive data
    with open('data/comprehensive_player_data.json', 'r') as f:
        player_data = json.load(f)
    
    # Calculate career metrics for all players
    enhanced_players = []
    for player in player_data:
        seasons = player['seasons']
        if len(seasons) >= 1:  # At least 1 season
            total_threes = sum(s.get('made_threes', 0) for s in seasons)
            total_attempts = sum(s.get('three_pt_shots', 0) for s in seasons)
            career_accuracy = (total_threes / total_attempts * 100) if total_attempts > 0 else 0
            peak_season = max(seasons, key=lambda x: x.get('made_threes', 0))
            
            enhanced_players.append({
                'player': player['player'],
                'seasons': seasons,
                'career_threes': total_threes,
                'career_attempts': total_attempts,
                'career_accuracy': round(career_accuracy, 1),
                'peak_threes': peak_season.get('made_threes', 0),
                'peak_season': peak_season.get('season', 0)
            })
    
    # Sort by career threes and take top 100
    enhanced_players.sort(key=lambda x: x['career_threes'], reverse=True)
    top_100_players = enhanced_players[:100]
    
    print(f"✅ Enhanced data for top {len(top_100_players)} players")
    return top_100_players

def organize_teams_by_conference():
    """Organize teams by conference structure."""
    print("🏟️ Organizing teams by conference...")
    
    # Load existing team data
    with open('data/comprehensive_team_data.json', 'r') as f:
        team_data = json.load(f)
    
    # Create conference-organized structure
    organized_teams = {
        "Eastern Conference": {
            "Atlantic": [],
            "Central": [],
            "Southeast": []
        },
        "Western Conference": {
            "Northwest": [],
            "Pacific": [],
            "Southwest": []
        }
    }
    
    # Map teams to their divisions
    team_lookup = {team['team']: team for team in team_data}
    
    for conference, divisions in NBA_CONFERENCES.items():
        for division, teams in divisions.items():
            for team_name in teams:
                if team_name in team_lookup:
                    organized_teams[conference][division].append(team_lookup[team_name])
                else:
                    print(f"⚠️  Team not found in data: {team_name}")
    
    print("✅ Teams organized by conference")
    return organized_teams, team_data

def create_enhanced_scene4_data():
    """Create enhanced scene4 data with top 100 players and conference organization."""
    
    # Get enhanced player data
    top_100_players = process_enhanced_player_data()
    
    # Get conference-organized teams
    conference_teams, all_teams = organize_teams_by_conference()
    
    # Load existing league data
    with open('data/comprehensive_league_data.json', 'r') as f:
        league_data = json.load(f)
    
    # Create enhanced scene4 data
    enhanced_data = {
        'league_trends': league_data,
        'team_data': all_teams,  # Keep flat structure for backward compatibility
        'team_conferences': conference_teams,  # New conference organization
        'player_data': top_100_players,
        'efficiency_comparison': {
            'mid_range_efficiency': 0.8,
            'three_point_efficiency': 1.1,
            'restricted_area_efficiency': 1.2
        },
        'metadata': {
            'total_teams': len(all_teams),
            'total_players': len(top_100_players),
            'conferences': list(NBA_CONFERENCES.keys()),
            'divisions_per_conference': 3,
            'teams_per_division': 5
        }
    }
    
    return enhanced_data

def save_enhanced_data():
    """Save enhanced data with better organization."""
    print("💾 Creating enhanced NBA data structure...")
    
    enhanced_data = create_enhanced_scene4_data()
    
    # Save enhanced data
    with open('data/scene4_data.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    # Also save individual components for easier debugging
    with open('data/top_100_players.json', 'w') as f:
        json.dump(enhanced_data['player_data'], f, indent=2)
    
    with open('data/teams_by_conference.json', 'w') as f:
        json.dump(enhanced_data['team_conferences'], f, indent=2)
    
    print("✅ Enhanced data saved successfully!")
    
    # Print summary
    print(f"\n📊 Enhanced Data Summary:")
    print(f"   Teams: {enhanced_data['metadata']['total_teams']}")
    print(f"   Players: {enhanced_data['metadata']['total_players']}")
    print(f"   Conferences: {len(enhanced_data['metadata']['conferences'])}")
    
    print(f"\n🏆 Top 10 Players by Career 3-Pointers:")
    for i, player in enumerate(enhanced_data['player_data'][:10]):
        print(f"   {i+1:2d}. {player['player']:<25} ({player['career_threes']:,} 3PM, {player['career_accuracy']:.1f}%)")
    
    print(f"\n🏟️  Conference Organization:")
    for conf_name, divisions in enhanced_data['team_conferences'].items():
        print(f"   {conf_name}:")
        for div_name, teams in divisions.items():
            print(f"     {div_name}: {len(teams)} teams")

if __name__ == "__main__":
    save_enhanced_data()
