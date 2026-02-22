"""
Optimized Line Fetching System

This module provides an optimized approach to fetching sportsbook lines
that minimizes API usage while still providing accurate line data.

Strategy:
1. Use our line projection system for initial estimates
2. Only fetch API lines for validation/calibration purposes (limited)
3. Store projected lines alongside actual lines for tracking

API Cost Optimization:
- Original: ~15 requests per day (5 games × 3 markets)
- Optimized: ~5 requests per day (only 1-2 games for validation)
- Projected: 0 requests (use projection system)

Author: PropAI Line Optimization
Date: 2026-02-04
"""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Import existing modules
try:
    from .odds_api_client import (
        fetch_nba_events,
        fetch_player_props_for_event,
        _deduplicate_props,
        store_player_props,
        OddsAPIPlayerProp,
        OddsAPIUsage,
        PLAYER_PROP_MARKETS,
        PREFERRED_BOOKS,
    )
    HAS_ODDS_API = True
except ImportError:
    HAS_ODDS_API = False

from ..engine.line_projector import (
    project_all_lines_for_date,
    store_projected_lines,
    ProjectedLine,
    ProjectionConfig,
)


@dataclass
class OptimizedLineResult:
    """Combined result from optimized line fetching."""
    projected_lines: List[ProjectedLine]
    api_lines: List[OddsAPIPlayerProp]
    api_usage: Optional[OddsAPIUsage]
    mode: str  # 'projected_only', 'api_only', 'hybrid'
    

def fetch_lines_optimized(
    conn: sqlite3.Connection,
    for_date: str,
    mode: str = 'projected_only',
    validation_sample_size: int = 1,  # Number of games to validate with API
    prop_types: List[str] = None,
    verbose: bool = False,
) -> OptimizedLineResult:
    """
    Fetch player prop lines using optimized strategy.
    
    Modes:
    - 'projected_only': Only use projection system (0 API calls)
    - 'api_only': Only use API (original behavior)
    - 'hybrid': Use projections, validate with limited API calls
    
    Args:
        conn: Database connection
        for_date: Date to fetch lines for (YYYY-MM-DD)
        mode: Fetching mode
        validation_sample_size: Number of games to fetch from API for validation
        prop_types: List of prop types (default: PTS, REB)
        verbose: Print progress
        
    Returns:
        OptimizedLineResult with projections and any API lines
    """
    if prop_types is None:
        prop_types = ['PTS', 'REB']
    
    result = OptimizedLineResult(
        projected_lines=[],
        api_lines=[],
        api_usage=None,
        mode=mode,
    )
    
    # Step 1: Always generate projections (no API cost)
    if verbose:
        print(f"\nGenerating projected lines for {for_date}...")
    
    projections = project_all_lines_for_date(
        conn, for_date, prop_types=prop_types
    )
    result.projected_lines = projections
    
    if verbose:
        print(f"  Generated {len(projections)} projected lines")
    
    # Store projections
    stored = store_projected_lines(conn, projections, for_date)
    if verbose:
        print(f"  Stored {stored} projections")
    
    # Step 2: Optionally fetch from API for validation
    if mode in ['api_only', 'hybrid'] and HAS_ODDS_API:
        if verbose:
            print(f"\nFetching API lines for validation...")
        
        # Get events
        events_response = fetch_nba_events(date_filter=for_date)
        
        if not events_response.success:
            if verbose:
                print(f"  Error: {events_response.error_message}")
        elif events_response.data:
            events = events_response.data
            
            # Limit to validation sample size for hybrid mode
            if mode == 'hybrid':
                events = events[:validation_sample_size]
            
            if verbose:
                print(f"  Found {len(events_response.data)} games, fetching {len(events)}")
            
            # Convert prop types to API market names
            markets = []
            for pt in prop_types:
                if pt == 'PTS':
                    markets.append('player_points')
                elif pt == 'REB':
                    markets.append('player_rebounds')
                elif pt == 'AST':
                    markets.append('player_assists')
            
            all_api_props = []
            for event in events:
                if verbose:
                    print(f"    Fetching: {event.away_team} @ {event.home_team}")
                
                props_response = fetch_player_props_for_event(
                    event_id=event.id,
                    markets=markets,
                    bookmakers=['draftkings'],  # Only fetch from one book to save quota
                )
                
                result.api_usage = props_response.usage
                
                if props_response.success and props_response.data:
                    all_api_props.extend(props_response.data)
            
            # Deduplicate
            result.api_lines = _deduplicate_props(all_api_props)
            
            if verbose:
                print(f"  Fetched {len(result.api_lines)} API lines")
                if result.api_usage:
                    print(f"  API quota remaining: {result.api_usage.requests_remaining}")
    
    return result


def get_consensus_lines(
    conn: sqlite3.Connection,
    for_date: str,
    prop_types: List[str] = None,
    prefer_api: bool = True,
) -> List[Dict]:
    """
    Get consensus lines combining projections and API data.
    
    Priority:
    1. If API line exists and prefer_api=True, use API line
    2. Otherwise use projected line
    
    Args:
        conn: Database connection
        for_date: Date to get lines for
        prop_types: List of prop types
        prefer_api: Whether to prefer API lines when available
        
    Returns:
        List of dicts with player_name, prop_type, line, source
    """
    if prop_types is None:
        prop_types = ['PTS', 'REB']
    
    lines = []
    
    # Get projected lines
    projections = {}
    rows = conn.execute("""
        SELECT player_id, player_name, prop_type, projected_line, confidence
        FROM projected_lines
        WHERE as_of_date = ?
    """, (for_date,)).fetchall()
    
    for row in rows:
        key = (row['player_name'], row['prop_type'])
        projections[key] = {
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'prop_type': row['prop_type'],
            'line': row['projected_line'],
            'confidence': row['confidence'],
            'source': 'projected',
        }
    
    # Get API lines (if available)
    api_lines = {}
    rows = conn.execute("""
        SELECT 
            sl.player_id,
            p.name as player_name,
            sl.prop_type,
            AVG(sl.line) as line,
            sl.book
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = ?
        GROUP BY sl.player_id, sl.prop_type
    """, (for_date,)).fetchall()
    
    for row in rows:
        key = (row['player_name'], row['prop_type'])
        api_lines[key] = {
            'player_id': row['player_id'],
            'player_name': row['player_name'],
            'prop_type': row['prop_type'],
            'line': row['line'],
            'confidence': 'API',
            'source': 'api',
        }
    
    # Merge
    all_keys = set(projections.keys()) | set(api_lines.keys())
    
    for key in all_keys:
        player_name, prop_type = key
        
        if prop_type not in prop_types:
            continue
        
        if prefer_api and key in api_lines:
            lines.append(api_lines[key])
        elif key in projections:
            lines.append(projections[key])
        elif key in api_lines:
            lines.append(api_lines[key])
    
    return lines


def estimate_api_savings(
    conn: sqlite3.Connection,
    for_date: str,
) -> Dict:
    """
    Estimate API quota savings from using projections.
    
    Returns:
        Dict with savings statistics
    """
    # Count projections
    proj_count = conn.execute("""
        SELECT COUNT(*) FROM projected_lines WHERE as_of_date = ?
    """, (for_date,)).fetchone()[0]
    
    # Get games for the date
    events_response = fetch_nba_events(date_filter=for_date) if HAS_ODDS_API else None
    num_games = len(events_response.data) if events_response and events_response.success else 5
    
    # Calculate costs
    # Each game × each market × each region = 1 request
    markets_per_game = 2  # PTS and REB
    original_cost = num_games * markets_per_game
    
    # Projected cost
    projected_cost = 0  # Free
    
    # Hybrid cost (1-2 games for validation)
    hybrid_cost = 1 * markets_per_game
    
    return {
        'date': for_date,
        'num_games': num_games,
        'lines_projected': proj_count,
        'original_api_cost': original_cost,
        'projected_only_cost': projected_cost,
        'hybrid_cost': hybrid_cost,
        'savings_projected': original_cost,
        'savings_hybrid': original_cost - hybrid_cost,
        'savings_percent': 100 * (original_cost - hybrid_cost) / original_cost if original_cost > 0 else 0,
    }


def print_line_comparison(
    conn: sqlite3.Connection,
    for_date: str,
    limit: int = 30,
):
    """Print comparison of projected vs API lines."""
    
    # Get consensus lines
    lines = get_consensus_lines(conn, for_date)
    
    # Separate by source
    projected = [l for l in lines if l['source'] == 'projected']
    api = [l for l in lines if l['source'] == 'api']
    
    print(f"\n{'='*70}")
    print(f"LINE COMPARISON FOR {for_date}")
    print(f"{'='*70}")
    print(f"\nProjected lines: {len(projected)}")
    print(f"API lines: {len(api)}")
    
    # Find overlap
    proj_keys = {(l['player_name'], l['prop_type']): l['line'] for l in projected}
    api_keys = {(l['player_name'], l['prop_type']): l['line'] for l in api}
    
    overlap = set(proj_keys.keys()) & set(api_keys.keys())
    
    if overlap:
        print(f"\nOverlap comparisons: {len(overlap)}")
        print(f"\n{'Player':<28} {'Type':>4} {'Proj':>6} {'API':>6} {'Diff':>6}")
        print("-" * 55)
        
        errors = []
        for key in sorted(overlap):
            proj_line = proj_keys[key]
            api_line = api_keys[key]
            diff = proj_line - api_line
            errors.append(abs(diff))
            
            print(f"{key[0]:<28} {key[1]:>4} {proj_line:>6.1f} {api_line:>6.1f} {diff:>+6.1f}")
        
        if errors:
            mae = sum(errors) / len(errors)
            print(f"\nMean Absolute Error: {mae:.2f}")
    
    # Show savings
    savings = estimate_api_savings(conn, for_date)
    print(f"\n{'='*40}")
    print(f"API QUOTA SAVINGS")
    print(f"{'='*40}")
    print(f"Original API cost: {savings['original_api_cost']} requests")
    print(f"Projected-only cost: {savings['projected_only_cost']} requests")
    print(f"Hybrid cost: {savings['hybrid_cost']} requests")
    print(f"Savings (hybrid): {savings['savings_hybrid']} requests ({savings['savings_percent']:.0f}%)")


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Test optimized fetching
    test_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Testing optimized line fetching for {test_date}")
    
    # Generate projections (no API cost)
    result = fetch_lines_optimized(conn, test_date, mode='projected_only', verbose=True)
    
    print(f"\nGenerated {len(result.projected_lines)} projections")
    
    # Show comparison if we have existing API lines
    print_line_comparison(conn, "2026-02-03")
    
    conn.commit()
    conn.close()
