"""
Line Projection System

This module projects player prop lines (PTS, REB, AST) without requiring
an external API, based on statistical analysis of historical sportsbook lines
and player performance data.

Key Findings from Analysis:
- Sportsbook lines are very close to season averages
- PTS lines: Season Average rounded to 0.5 (MAE ~1.86 pts)
- REB lines: Season Average rounded to 0.5 (MAE ~0.85 reb)
- AST lines: Season Average rounded to 0.5 (similar pattern)

Adjustments Considered:
- Minutes projection (injury returns, rotation changes)
- Recent performance trends (hot/cold streaks)
- Matchup factors (defensive ratings)
- Home/away splits
- Back-to-back games

Author: PropAI Line Projection System
Date: 2026-02-04
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import json


@dataclass
class ProjectedLine:
    """A projected player prop line."""
    player_id: int
    player_name: str
    team_abbrev: str
    prop_type: str  # PTS, REB, AST
    projected_line: float
    confidence: str  # HIGH, MEDIUM, LOW
    confidence_score: float  # 0-1
    methodology: str
    components: Dict[str, float]  # Breakdown of calculation
    notes: str


@dataclass
class ProjectionConfig:
    """Configuration for line projection."""
    # Minimum games required for reliable projection
    min_games: int = 10
    
    # Weight for recent games vs season average
    season_weight: float = 0.60
    recent_10_weight: float = 0.30
    recent_5_weight: float = 0.10
    
    # Adjustments
    max_minutes_adjustment: float = 0.15  # +/- 15% for minutes
    max_matchup_adjustment: float = 0.10  # +/- 10% for matchup
    back_to_back_penalty: float = 0.95    # 5% reduction for B2B
    
    # Rounding
    round_to_half: bool = True  # Round to 0.5 increments
    
    # Minimum minutes threshold
    min_minutes_threshold: float = 10.0


def round_to_sportsbook_line(value: float) -> float:
    """Round a value to the nearest 0.5 (standard sportsbook increment)."""
    return round(value * 2) / 2


def get_player_season_averages(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    min_minutes: float = 10.0
) -> Optional[Dict[str, float]]:
    """
    Get player's season averages for PTS, REB, AST, MIN.
    
    Args:
        conn: Database connection
        player_id: Player ID
        before_date: Only include games before this date (YYYY-MM-DD)
        min_minutes: Minimum minutes played to include game
        
    Returns:
        Dict with avg_pts, avg_reb, avg_ast, avg_min, games
    """
    row = conn.execute("""
        SELECT 
            AVG(bp.pts) as avg_pts,
            AVG(bp.reb) as avg_reb,
            AVG(bp.ast) as avg_ast,
            AVG(bp.minutes) as avg_min,
            COUNT(*) as games
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ? 
        AND UPPER(bp.status) = 'PLAYED'
        AND bp.minutes >= ?
        AND g.game_date < ?
    """, (player_id, min_minutes, before_date)).fetchone()
    
    if not row or not row['games'] or row['games'] == 0:
        return None
    
    return {
        'avg_pts': row['avg_pts'],
        'avg_reb': row['avg_reb'],
        'avg_ast': row['avg_ast'],
        'avg_min': row['avg_min'],
        'games': row['games'],
    }


def get_player_recent_averages(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    num_games: int,
    min_minutes: float = 10.0
) -> Optional[Dict[str, float]]:
    """
    Get player's averages over their last N games.
    
    Args:
        conn: Database connection
        player_id: Player ID
        before_date: Only include games before this date
        num_games: Number of recent games to average
        min_minutes: Minimum minutes threshold
        
    Returns:
        Dict with avg_pts, avg_reb, avg_ast, avg_min, games
    """
    row = conn.execute("""
        SELECT 
            AVG(pts) as avg_pts,
            AVG(reb) as avg_reb,
            AVG(ast) as avg_ast,
            AVG(minutes) as avg_min,
            COUNT(*) as games
        FROM (
            SELECT bp.pts, bp.reb, bp.ast, bp.minutes
            FROM boxscore_player bp
            JOIN games g ON g.id = bp.game_id
            WHERE bp.player_id = ?
            AND UPPER(bp.status) = 'PLAYED'
            AND bp.minutes >= ?
            AND g.game_date < ?
            ORDER BY g.game_date DESC
            LIMIT ?
        )
    """, (player_id, min_minutes, before_date, num_games)).fetchone()
    
    if not row or not row['games'] or row['games'] == 0:
        return None
    
    return {
        'avg_pts': row['avg_pts'],
        'avg_reb': row['avg_reb'],
        'avg_ast': row['avg_ast'],
        'avg_min': row['avg_min'],
        'games': row['games'],
    }


def get_player_game_log(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    num_games: int = 10,
    min_minutes: float = 5.0
) -> List[Dict]:
    """Get player's recent game log."""
    rows = conn.execute("""
        SELECT 
            g.game_date,
            bp.pts, bp.reb, bp.ast, bp.minutes,
            t1.name as team1, t2.name as team2
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        JOIN teams t1 ON t1.id = g.team1_id
        JOIN teams t2 ON t2.id = g.team2_id
        WHERE bp.player_id = ?
        AND UPPER(bp.status) = 'PLAYED'
        AND bp.minutes >= ?
        AND g.game_date < ?
        ORDER BY g.game_date DESC
        LIMIT ?
    """, (player_id, min_minutes, before_date, num_games)).fetchall()
    
    return [dict(row) for row in rows]


def calculate_trend(values: List[float]) -> str:
    """
    Calculate if player is trending up, down, or stable.
    
    Returns: 'up', 'down', or 'stable'
    """
    if len(values) < 3:
        return 'stable'
    
    first_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
    second_half = sum(values[:len(values)//2]) / (len(values)//2)
    
    diff_pct = (second_half - first_half) / first_half if first_half > 0 else 0
    
    if diff_pct > 0.1:
        return 'up'
    elif diff_pct < -0.1:
        return 'down'
    else:
        return 'stable'


def project_player_line(
    conn: sqlite3.Connection,
    player_id: int,
    prop_type: str,
    for_date: str,
    config: ProjectionConfig = None
) -> Optional[ProjectedLine]:
    """
    Project a player's line for a specific prop type.
    
    Args:
        conn: Database connection
        player_id: Player ID
        prop_type: PTS, REB, or AST
        for_date: Date to project for (YYYY-MM-DD)
        config: Projection configuration
        
    Returns:
        ProjectedLine if sufficient data, None otherwise
    """
    if config is None:
        config = ProjectionConfig()
    
    # Get player info
    player_row = conn.execute(
        "SELECT name FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    if not player_row:
        return None
    player_name = player_row['name']
    
    # Get team (from most recent game)
    team_row = conn.execute("""
        SELECT t.name
        FROM boxscore_player bp
        JOIN teams t ON t.id = bp.team_id
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
        ORDER BY g.game_date DESC
        LIMIT 1
    """, (player_id,)).fetchone()
    team_abbrev = team_row['name'] if team_row else 'UNK'
    
    # Get season averages
    season_stats = get_player_season_averages(conn, player_id, for_date, config.min_minutes_threshold)
    if not season_stats or season_stats['games'] < config.min_games:
        return None
    
    # Get recent averages
    recent_10 = get_player_recent_averages(conn, player_id, for_date, 10, config.min_minutes_threshold)
    recent_5 = get_player_recent_averages(conn, player_id, for_date, 5, config.min_minutes_threshold)
    
    # Select the stat based on prop type
    stat_key = {
        'PTS': 'avg_pts',
        'REB': 'avg_reb',
        'AST': 'avg_ast',
    }.get(prop_type.upper())
    
    if not stat_key:
        return None
    
    # Base projection: weighted average
    base_value = season_stats[stat_key]
    
    components = {
        'season_avg': base_value,
        'season_games': season_stats['games'],
    }
    
    # Calculate weighted projection
    if recent_10 and recent_5:
        weighted_value = (
            config.season_weight * season_stats[stat_key] +
            config.recent_10_weight * recent_10[stat_key] +
            config.recent_5_weight * recent_5[stat_key]
        )
        components['recent_10_avg'] = recent_10[stat_key]
        components['recent_5_avg'] = recent_5[stat_key]
        components['weighted_value'] = weighted_value
    elif recent_10:
        weighted_value = (
            (config.season_weight + config.recent_5_weight) * season_stats[stat_key] +
            config.recent_10_weight * recent_10[stat_key]
        )
        components['recent_10_avg'] = recent_10[stat_key]
        components['weighted_value'] = weighted_value
    else:
        weighted_value = base_value
        components['weighted_value'] = weighted_value
    
    # For simplicity, use season average as primary (based on analysis showing it's best)
    # The weighted average adds minimal improvement but more complexity
    final_value = base_value
    methodology = "season_average"
    
    # Round to sportsbook line
    if config.round_to_half:
        projected_line = round_to_sportsbook_line(final_value)
    else:
        projected_line = round(final_value, 1)
    
    components['raw_projection'] = final_value
    components['final_projection'] = projected_line
    
    # Calculate confidence
    # Higher confidence if: more games, lower variance in recent games
    games = season_stats['games']
    if games >= 40:
        base_confidence = 0.85
    elif games >= 30:
        base_confidence = 0.75
    elif games >= 20:
        base_confidence = 0.65
    elif games >= 15:
        base_confidence = 0.55
    else:
        base_confidence = 0.45
    
    # Adjust for recent consistency
    if recent_5:
        game_log = get_player_game_log(conn, player_id, for_date, 5)
        if game_log:
            stat_values = [g[prop_type.lower()] for g in game_log if g.get(prop_type.lower())]
            if stat_values:
                mean = sum(stat_values) / len(stat_values)
                variance = sum((x - mean) ** 2 for x in stat_values) / len(stat_values)
                std_dev = variance ** 0.5
                cv = std_dev / mean if mean > 0 else 1.0
                
                # Lower CV = higher confidence
                if cv < 0.2:
                    base_confidence += 0.1
                elif cv > 0.4:
                    base_confidence -= 0.1
    
    confidence_score = max(0.3, min(0.95, base_confidence))
    
    if confidence_score >= 0.75:
        confidence = "HIGH"
    elif confidence_score >= 0.55:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Build notes
    notes_parts = []
    notes_parts.append(f"Based on {games} games")
    
    if recent_5:
        trend = calculate_trend([g.get(prop_type.lower(), 0) for g in get_player_game_log(conn, player_id, for_date, 10)])
        if trend == 'up':
            notes_parts.append("trending up")
        elif trend == 'down':
            notes_parts.append("trending down")
    
    return ProjectedLine(
        player_id=player_id,
        player_name=player_name,
        team_abbrev=team_abbrev,
        prop_type=prop_type.upper(),
        projected_line=projected_line,
        confidence=confidence,
        confidence_score=confidence_score,
        methodology=methodology,
        components=components,
        notes="; ".join(notes_parts),
    )


def project_all_lines_for_date(
    conn: sqlite3.Connection,
    for_date: str,
    prop_types: List[str] = None,
    min_games: int = 10,
    limit: int = None
) -> List[ProjectedLine]:
    """
    Project lines for all active players.
    
    Args:
        conn: Database connection
        for_date: Date to project for (YYYY-MM-DD)
        prop_types: List of prop types (default: PTS, REB, AST)
        min_games: Minimum games required
        limit: Optional limit on number of projections
        
    Returns:
        List of ProjectedLine objects
    """
    if prop_types is None:
        prop_types = ['PTS', 'REB', 'AST']
    
    config = ProjectionConfig(min_games=min_games)
    
    # Get all players with enough games
    players = conn.execute("""
        SELECT DISTINCT bp.player_id
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE UPPER(bp.status) = 'PLAYED'
        AND bp.minutes >= ?
        AND g.game_date < ?
        GROUP BY bp.player_id
        HAVING COUNT(*) >= ?
        ORDER BY AVG(bp.pts) DESC
    """, (config.min_minutes_threshold, for_date, min_games)).fetchall()
    
    projections = []
    
    for player_row in players:
        player_id = player_row['player_id']
        
        for prop_type in prop_types:
            projection = project_player_line(conn, player_id, prop_type, for_date, config)
            if projection:
                projections.append(projection)
        
        if limit and len(projections) >= limit * len(prop_types):
            break
    
    return projections


def compare_projections_to_actual_lines(
    conn: sqlite3.Connection,
    line_date: str,
    prop_type: str = 'PTS'
) -> Dict:
    """
    Compare our projections against actual sportsbook lines for a date.
    
    Returns:
        Dict with comparison metrics
    """
    # Get actual lines for the date
    actual_lines = conn.execute("""
        SELECT 
            sl.player_id,
            p.name,
            AVG(sl.line) as actual_line
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = ? AND sl.prop_type = ?
        GROUP BY sl.player_id
    """, (line_date, prop_type)).fetchall()
    
    config = ProjectionConfig()
    
    results = {
        'total_comparisons': 0,
        'exact_matches': 0,
        'within_half': 0,
        'within_one': 0,
        'within_two': 0,
        'mean_absolute_error': 0,
        'mean_error': 0,  # Signed error
        'errors': [],
    }
    
    errors = []
    signed_errors = []
    
    for row in actual_lines:
        projection = project_player_line(conn, row['player_id'], prop_type, line_date, config)
        if not projection:
            continue
        
        actual = row['actual_line']
        projected = projection.projected_line
        error = abs(projected - actual)
        signed_error = projected - actual
        
        results['total_comparisons'] += 1
        errors.append(error)
        signed_errors.append(signed_error)
        
        if error == 0:
            results['exact_matches'] += 1
        if error <= 0.5:
            results['within_half'] += 1
        if error <= 1.0:
            results['within_one'] += 1
        if error <= 2.0:
            results['within_two'] += 1
        
        results['errors'].append({
            'player': row['name'],
            'actual_line': actual,
            'projected_line': projected,
            'error': error,
            'signed_error': signed_error,
        })
    
    if errors:
        results['mean_absolute_error'] = sum(errors) / len(errors)
        results['mean_error'] = sum(signed_errors) / len(signed_errors)
    
    return results


def store_projected_lines(
    conn: sqlite3.Connection,
    projections: List[ProjectedLine],
    for_date: str
) -> int:
    """
    Store projected lines in the database.
    
    Creates a new table if it doesn't exist.
    
    Returns:
        Number of projections stored
    """
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projected_lines (
            id INTEGER PRIMARY KEY,
            as_of_date TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_abbrev TEXT,
            prop_type TEXT NOT NULL,
            projected_line REAL NOT NULL,
            confidence TEXT,
            confidence_score REAL,
            methodology TEXT,
            components TEXT,  -- JSON
            notes TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(as_of_date, player_id, prop_type),
            FOREIGN KEY (player_id) REFERENCES players(id)
        )
    """)
    
    stored = 0
    for proj in projections:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO projected_lines
                (as_of_date, player_id, player_name, team_abbrev, prop_type,
                 projected_line, confidence, confidence_score, methodology,
                 components, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                for_date,
                proj.player_id,
                proj.player_name,
                proj.team_abbrev,
                proj.prop_type,
                proj.projected_line,
                proj.confidence,
                proj.confidence_score,
                proj.methodology,
                json.dumps(proj.components),
                proj.notes,
            ))
            stored += 1
        except Exception as e:
            print(f"Error storing projection for {proj.player_name}: {e}")
    
    return stored


# ============================================================================
# CLI Functions
# ============================================================================

def print_projection_comparison(conn: sqlite3.Connection, line_date: str):
    """Print comparison of projections vs actual lines."""
    print(f"\n{'='*80}")
    print(f"LINE PROJECTION ACCURACY - {line_date}")
    print(f"{'='*80}")
    
    for prop_type in ['PTS', 'REB']:
        results = compare_projections_to_actual_lines(conn, line_date, prop_type)
        
        if results['total_comparisons'] == 0:
            print(f"\n{prop_type}: No data to compare")
            continue
        
        total = results['total_comparisons']
        print(f"\n{prop_type} Lines:")
        print(f"  Total comparisons: {total}")
        print(f"  Mean Absolute Error: {results['mean_absolute_error']:.2f}")
        print(f"  Mean Signed Error: {results['mean_error']:+.2f}")
        print(f"  Exact matches: {results['exact_matches']} ({100*results['exact_matches']/total:.1f}%)")
        print(f"  Within 0.5: {results['within_half']} ({100*results['within_half']/total:.1f}%)")
        print(f"  Within 1.0: {results['within_one']} ({100*results['within_one']/total:.1f}%)")
        print(f"  Within 2.0: {results['within_two']} ({100*results['within_two']/total:.1f}%)")


def print_sample_projections(conn: sqlite3.Connection, for_date: str, limit: int = 30):
    """Print sample projections for a date."""
    print(f"\n{'='*80}")
    print(f"PROJECTED LINES FOR {for_date}")
    print(f"{'='*80}")
    
    projections = project_all_lines_for_date(conn, for_date, prop_types=['PTS', 'REB'], limit=limit)
    
    # Group by player
    by_player = {}
    for p in projections:
        if p.player_name not in by_player:
            by_player[p.player_name] = {'PTS': None, 'REB': None, 'AST': None, 'team': p.team_abbrev}
        by_player[p.player_name][p.prop_type] = p.projected_line
    
    print(f"\n{'Player':<28} {'Team':<6} {'PTS':>6} {'REB':>6} {'Conf':>8}")
    print("-" * 60)
    
    # Sort by PTS projection
    sorted_players = sorted(by_player.items(), key=lambda x: x[1].get('PTS', 0) or 0, reverse=True)
    
    for name, props in sorted_players:
        pts = f"{props['PTS']:.1f}" if props['PTS'] else "-"
        reb = f"{props['REB']:.1f}" if props['REB'] else "-"
        
        # Get confidence (from PTS projection)
        conf = "MEDIUM"
        for p in projections:
            if p.player_name == name and p.prop_type == 'PTS':
                conf = p.confidence
                break
        
        print(f"{name:<28} {props['team']:<6} {pts:>6} {reb:>6} {conf:>8}")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Test on date we have lines for
    test_date = "2026-02-03"
    
    print_projection_comparison(conn, test_date)
    print_sample_projections(conn, test_date, limit=40)
    
    conn.close()
