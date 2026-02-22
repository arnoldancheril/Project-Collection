"""
Projected Line Integration Module

This module integrates the new line projection system with existing models.
It provides a unified interface for models to use either:
1. Sportsbook lines (from API)
2. Projected lines (from our projection system)
3. Legacy derived lines (L10 average * 1.05)

The key benefit is that projected lines can be used when sportsbook lines
are not available, providing more accurate lines than the legacy derived approach.

Key Finding from Analysis:
- Legacy derived lines: L10 avg * 1.05 → MAE ~2.2
- Projected lines: Season avg rounded to 0.5 → MAE ~1.8 for PTS
- Sportsbook lines: Perfect (we're matching against them) → MAE ~0

Author: PropAI Team
Date: 2026-02-04
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from .line_projector import (
    ProjectedLine,
    ProjectionConfig,
    project_player_line,
    project_all_lines_for_date,
    round_to_sportsbook_line,
    get_player_season_averages,
)


@dataclass
class UnifiedLineInfo:
    """
    Unified line information from any source.
    
    This standardizes line data regardless of whether it came from:
    - sportsbook (API)
    - projected (our projection system)
    - derived (legacy L10 * 1.05 method)
    """
    line: float
    source: str  # "sportsbook", "projected", "derived"
    book: Optional[str] = None
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    confidence_score: float = 0.5  # 0-1 scale
    methodology: str = ""
    components: Optional[Dict] = None


def get_projected_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    config: Optional[ProjectionConfig] = None
) -> Optional[UnifiedLineInfo]:
    """
    Get a projected line using our line projection system.
    
    This is the recommended alternative to derived lines when
    sportsbook lines are not available.
    
    Args:
        conn: Database connection
        player_id: Player ID
        player_name: Player name (for logging)
        prop_type: PTS, REB, or AST
        game_date: Date to project for
        config: Optional projection configuration
        
    Returns:
        UnifiedLineInfo if projection successful, None otherwise
    """
    if config is None:
        config = ProjectionConfig()
    
    projection = project_player_line(conn, player_id, prop_type, game_date, config)
    
    if projection is None:
        return None
    
    return UnifiedLineInfo(
        line=projection.projected_line,
        source="projected",
        book=None,
        confidence=projection.confidence,
        confidence_score=projection.confidence_score,
        methodology=projection.methodology,
        components=projection.components,
    )


def get_enhanced_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    stats: Dict = None,
    prefer_projected: bool = True,
    derived_adjustment: float = 1.05
) -> UnifiedLineInfo:
    """
    Get enhanced line using the best available source.
    
    Priority order:
    1. Sportsbook line (if available)
    2. Projected line (if prefer_projected=True)
    3. Derived line (legacy fallback)
    
    This function ALWAYS returns a line, making it safe to use
    in model calculations without null checks.
    
    Args:
        conn: Database connection
        player_id: Player ID  
        player_name: Player name
        prop_type: PTS, REB, or AST
        game_date: Date for the line
        stats: Player stats dict (needed for derived line fallback)
        prefer_projected: Use projected over derived when no sportsbook
        derived_adjustment: Multiplier for derived lines (legacy)
        
    Returns:
        UnifiedLineInfo from best available source
    """
    # 1. Try sportsbook line first
    sportsbook = _get_sportsbook_line(conn, player_id, player_name, prop_type, game_date)
    if sportsbook:
        return sportsbook
    
    # 2. Try projected line (new system)
    if prefer_projected:
        projected = get_projected_line(conn, player_id, player_name, prop_type, game_date)
        if projected:
            return projected
    
    # 3. Fall back to derived line (legacy)
    return _get_derived_line(stats, prop_type, derived_adjustment)


def _get_sportsbook_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Optional[UnifiedLineInfo]:
    """Get sportsbook line from database."""
    row = conn.execute(
        """
        SELECT line, book
        FROM sportsbook_lines
        WHERE player_id = ? AND prop_type = ? AND as_of_date = ?
        ORDER BY created_at DESC LIMIT 1
        """,
        (player_id, prop_type.upper(), game_date)
    ).fetchone()
    
    if row:
        return UnifiedLineInfo(
            line=row["line"],
            source="sportsbook",
            book=row["book"] or "unknown",
            confidence="HIGH",
            confidence_score=0.95,
            methodology="sportsbook_api",
        )
    
    return None


def _get_derived_line(
    stats: Dict,
    prop_type: str,
    adjustment: float = 1.05
) -> UnifiedLineInfo:
    """
    Calculate legacy derived line based on L10 average.
    
    This is the fallback method when sportsbook and projected
    lines are not available.
    """
    pt = prop_type.lower()
    
    # Try to get L10 from stats
    if stats and hasattr(stats, 'l10'):
        l10_avg = stats.l10.get(pt, 0)
    elif stats and isinstance(stats, dict):
        l10_avg = stats.get('l10', {}).get(pt, 0)
    else:
        l10_avg = 0
    
    if l10_avg <= 0:
        # Last resort: use a default based on prop type
        defaults = {'pts': 15.0, 'reb': 5.0, 'ast': 3.0}
        l10_avg = defaults.get(pt, 10.0)
    
    derived = l10_avg * adjustment
    derived = round_to_sportsbook_line(derived)
    
    return UnifiedLineInfo(
        line=derived,
        source="derived",
        book=None,
        confidence="LOW",
        confidence_score=0.4,
        methodology=f"l10_avg_{adjustment}",
    )


# ============================================================================
# Model Integration Functions
# ============================================================================

def get_line_for_model(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    stats: Dict = None,
    use_projected_lines: bool = True
) -> Tuple[float, str, str]:
    """
    Get line for use in model predictions.
    
    This is the main integration point for existing models.
    
    Args:
        conn: Database connection
        player_id: Player ID
        player_name: Player name
        prop_type: PTS, REB, or AST
        game_date: Date for the line
        stats: Player stats (for derived fallback)
        use_projected_lines: Whether to use new projection system
        
    Returns:
        Tuple of (line_value, source, confidence)
    """
    line_info = get_enhanced_line(
        conn=conn,
        player_id=player_id,
        player_name=player_name,
        prop_type=prop_type,
        game_date=game_date,
        stats=stats,
        prefer_projected=use_projected_lines,
    )
    
    return line_info.line, line_info.source, line_info.confidence


def batch_project_lines(
    conn: sqlite3.Connection,
    game_date: str,
    player_ids: List[int] = None,
    prop_types: List[str] = None
) -> Dict[Tuple[int, str], UnifiedLineInfo]:
    """
    Batch project lines for multiple players.
    
    This is more efficient than calling get_projected_line 
    for each player individually.
    
    Args:
        conn: Database connection
        game_date: Date to project for
        player_ids: Optional list of player IDs (None = all)
        prop_types: List of prop types (default: PTS, REB, AST)
        
    Returns:
        Dict mapping (player_id, prop_type) to UnifiedLineInfo
    """
    if prop_types is None:
        prop_types = ['PTS', 'REB', 'AST']
    
    projections = project_all_lines_for_date(
        conn=conn,
        for_date=game_date,
        prop_types=prop_types,
        min_games=10,
        limit=None
    )
    
    result = {}
    for proj in projections:
        if player_ids is None or proj.player_id in player_ids:
            key = (proj.player_id, proj.prop_type)
            result[key] = UnifiedLineInfo(
                line=proj.projected_line,
                source="projected",
                book=None,
                confidence=proj.confidence,
                confidence_score=proj.confidence_score,
                methodology=proj.methodology,
                components=proj.components,
            )
    
    return result


# ============================================================================
# Statistics and Comparison
# ============================================================================

def compare_line_sources(
    conn: sqlite3.Connection,
    game_date: str
) -> Dict[str, Dict]:
    """
    Compare different line sources for a given date.
    
    This shows the accuracy and availability of each line source.
    
    Returns:
        Dict with statistics for each source (sportsbook, projected, derived)
    """
    # Get all sportsbook lines for the date
    sportsbook_lines = conn.execute("""
        SELECT player_id, prop_type, line as sportsbook_line
        FROM sportsbook_lines
        WHERE as_of_date = ?
    """, (game_date,)).fetchall()
    
    results = {
        'sportsbook': {'count': 0, 'available': True},
        'projected': {'count': 0, 'mae': 0.0, 'within_half': 0, 'within_one': 0},
        'derived': {'count': 0, 'mae': 0.0, 'within_half': 0, 'within_one': 0},
    }
    
    if not sportsbook_lines:
        results['sportsbook']['available'] = False
        return results
    
    results['sportsbook']['count'] = len(sportsbook_lines)
    
    projected_errors = []
    derived_errors = []
    
    for row in sportsbook_lines:
        player_id = row['player_id']
        prop_type = row['prop_type']
        actual_line = row['sportsbook_line']
        
        # Get projected line
        projected = get_projected_line(conn, player_id, "", prop_type, game_date)
        if projected:
            error = abs(projected.line - actual_line)
            projected_errors.append(error)
            results['projected']['count'] += 1
            if error <= 0.5:
                results['projected']['within_half'] += 1
            if error <= 1.0:
                results['projected']['within_one'] += 1
        
        # Get season stats for derived
        season_stats = get_player_season_averages(conn, player_id, game_date)
        if season_stats:
            pt = prop_type.lower()
            stat_key = {'pts': 'avg_pts', 'reb': 'avg_reb', 'ast': 'avg_ast'}.get(pt)
            if stat_key and season_stats.get(stat_key):
                l10_avg = season_stats[stat_key]  # Use season as proxy
                derived = round_to_sportsbook_line(l10_avg * 1.05)
                error = abs(derived - actual_line)
                derived_errors.append(error)
                results['derived']['count'] += 1
                if error <= 0.5:
                    results['derived']['within_half'] += 1
                if error <= 1.0:
                    results['derived']['within_one'] += 1
    
    # Calculate MAE
    if projected_errors:
        results['projected']['mae'] = sum(projected_errors) / len(projected_errors)
        results['projected']['within_half_pct'] = results['projected']['within_half'] / len(projected_errors) * 100
        results['projected']['within_one_pct'] = results['projected']['within_one'] / len(projected_errors) * 100
    
    if derived_errors:
        results['derived']['mae'] = sum(derived_errors) / len(derived_errors)
        results['derived']['within_half_pct'] = results['derived']['within_half'] / len(derived_errors) * 100
        results['derived']['within_one_pct'] = results['derived']['within_one'] / len(derived_errors) * 100
    
    return results


def print_line_comparison_report(conn: sqlite3.Connection, dates: List[str] = None):
    """
    Print a comprehensive comparison report of line sources.
    """
    if dates is None:
        # Get all dates with sportsbook lines
        dates = [row['as_of_date'] for row in conn.execute(
            "SELECT DISTINCT as_of_date FROM sportsbook_lines ORDER BY as_of_date"
        ).fetchall()]
    
    print("=" * 70)
    print("LINE SOURCE COMPARISON REPORT")
    print("=" * 70)
    
    total_projected = {'count': 0, 'mae_sum': 0, 'within_half': 0, 'within_one': 0}
    total_derived = {'count': 0, 'mae_sum': 0, 'within_half': 0, 'within_one': 0}
    
    for date in dates:
        results = compare_line_sources(conn, date)
        
        print(f"\nDate: {date}")
        print(f"  Sportsbook lines: {results['sportsbook']['count']}")
        
        if results['projected']['count'] > 0:
            print(f"  Projected: {results['projected']['count']} lines, "
                  f"MAE={results['projected']['mae']:.2f}, "
                  f"Within 0.5={results['projected'].get('within_half_pct', 0):.1f}%")
            total_projected['count'] += results['projected']['count']
            total_projected['mae_sum'] += results['projected']['mae'] * results['projected']['count']
            total_projected['within_half'] += results['projected']['within_half']
            total_projected['within_one'] += results['projected']['within_one']
        
        if results['derived']['count'] > 0:
            print(f"  Derived:   {results['derived']['count']} lines, "
                  f"MAE={results['derived']['mae']:.2f}, "
                  f"Within 0.5={results['derived'].get('within_half_pct', 0):.1f}%")
            total_derived['count'] += results['derived']['count']
            total_derived['mae_sum'] += results['derived']['mae'] * results['derived']['count']
            total_derived['within_half'] += results['derived']['within_half']
            total_derived['within_one'] += results['derived']['within_one']
    
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    if total_projected['count'] > 0:
        overall_mae = total_projected['mae_sum'] / total_projected['count']
        within_half_pct = total_projected['within_half'] / total_projected['count'] * 100
        within_one_pct = total_projected['within_one'] / total_projected['count'] * 100
        print(f"\nPROJECTED LINES:")
        print(f"  Total comparisons: {total_projected['count']}")
        print(f"  Mean Absolute Error: {overall_mae:.2f}")
        print(f"  Within 0.5 points: {within_half_pct:.1f}%")
        print(f"  Within 1.0 points: {within_one_pct:.1f}%")
    
    if total_derived['count'] > 0:
        overall_mae = total_derived['mae_sum'] / total_derived['count']
        within_half_pct = total_derived['within_half'] / total_derived['count'] * 100
        within_one_pct = total_derived['within_one'] / total_derived['count'] * 100
        print(f"\nDERIVED LINES (Legacy):")
        print(f"  Total comparisons: {total_derived['count']}")
        print(f"  Mean Absolute Error: {overall_mae:.2f}")
        print(f"  Within 0.5 points: {within_half_pct:.1f}%")
        print(f"  Within 1.0 points: {within_one_pct:.1f}%")
    
    # Calculate improvement
    if total_projected['count'] > 0 and total_derived['count'] > 0:
        proj_mae = total_projected['mae_sum'] / total_projected['count']
        der_mae = total_derived['mae_sum'] / total_derived['count']
        improvement = (der_mae - proj_mae) / der_mae * 100
        print(f"\nPROJECTED vs DERIVED:")
        print(f"  MAE improvement: {improvement:.1f}%")
        print(f"  Projected lines are {improvement:.1f}% more accurate than derived")


# ============================================================================
# Main entry point for testing
# ============================================================================

if __name__ == "__main__":
    import os
    
    db_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "data", "db", "nba_props.sqlite3"
    )
    db_path = os.path.abspath(db_path)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print_line_comparison_report(conn)
    
    conn.close()
