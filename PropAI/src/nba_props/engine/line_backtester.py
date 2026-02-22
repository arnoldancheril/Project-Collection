"""
Line Projection Backtesting System

This module tests the accuracy of our line projections by comparing them
against actual sportsbook lines and game results.

Backtesting Methodology:
1. For each date with sportsbook lines, project lines using only data
   available BEFORE that date
2. Compare projections to actual sportsbook lines
3. Compare both projections and actual lines to game results
4. Track accuracy metrics over time

Metrics:
- Mean Absolute Error (MAE): Average absolute difference from actual lines
- Signed Error: Average over/under bias
- Exact Match Rate: % of projections matching the exact line
- Within-X Rate: % of projections within X points of actual

Author: PropAI Backtesting System
Date: 2026-02-04
"""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from .line_projector import (
    project_player_line,
    ProjectionConfig,
    round_to_sportsbook_line,
)


@dataclass
class BacktestResult:
    """Results from backtesting on a single date."""
    date: str
    prop_type: str
    total_comparisons: int
    mean_absolute_error: float
    mean_signed_error: float
    exact_matches: int
    within_half: int
    within_one: int
    within_two: int
    vs_results_over_rate: float  # % of lines where actual result was OVER
    projected_over_rate: float   # % of projections that would have been OVER
    
    @property
    def exact_match_pct(self) -> float:
        return 100 * self.exact_matches / self.total_comparisons if self.total_comparisons else 0
    
    @property
    def within_half_pct(self) -> float:
        return 100 * self.within_half / self.total_comparisons if self.total_comparisons else 0
    
    @property
    def within_one_pct(self) -> float:
        return 100 * self.within_one / self.total_comparisons if self.total_comparisons else 0


@dataclass
class BacktestSummary:
    """Aggregated backtest results across multiple dates."""
    prop_type: str
    num_dates: int
    total_comparisons: int
    avg_mae: float
    avg_signed_error: float
    overall_exact_match_pct: float
    overall_within_half_pct: float
    overall_within_one_pct: float
    overall_within_two_pct: float
    overall_vs_results_over_rate: float
    dates_tested: List[str]


def backtest_single_date(
    conn: sqlite3.Connection,
    test_date: str,
    prop_type: str = 'PTS',
    config: ProjectionConfig = None
) -> Optional[BacktestResult]:
    """
    Backtest projections for a single date.
    
    Args:
        conn: Database connection
        test_date: Date to test (YYYY-MM-DD)
        prop_type: PTS, REB, or AST
        config: Projection configuration
        
    Returns:
        BacktestResult with metrics
    """
    if config is None:
        config = ProjectionConfig()
    
    # Get actual sportsbook lines for this date
    actual_lines = conn.execute("""
        SELECT 
            sl.player_id,
            p.name as player_name,
            AVG(sl.line) as actual_line
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = ? AND sl.prop_type = ?
        GROUP BY sl.player_id
    """, (test_date, prop_type)).fetchall()
    
    if not actual_lines:
        return None
    
    # Get game results for this date
    results_map = {}
    stat_col = prop_type.lower()
    game_results = conn.execute(f"""
        SELECT 
            bp.player_id,
            bp.{stat_col} as actual_stat
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE g.game_date = ? AND UPPER(bp.status) = 'PLAYED'
    """, (test_date,)).fetchall()
    
    for row in game_results:
        results_map[row['player_id']] = row['actual_stat']
    
    # Calculate metrics
    errors = []
    signed_errors = []
    exact = 0
    within_half = 0
    within_one = 0
    within_two = 0
    
    results_over = 0  # Actual result was over the line
    results_under = 0
    projected_over = 0  # Our projection was over the line
    projected_under = 0
    
    for row in actual_lines:
        projection = project_player_line(conn, row['player_id'], prop_type, test_date, config)
        if not projection:
            continue
        
        actual_line = row['actual_line']
        projected_line = projection.projected_line
        
        error = abs(projected_line - actual_line)
        signed_error = projected_line - actual_line
        
        errors.append(error)
        signed_errors.append(signed_error)
        
        if error == 0:
            exact += 1
        if error <= 0.5:
            within_half += 1
        if error <= 1.0:
            within_one += 1
        if error <= 2.0:
            within_two += 1
        
        # Compare to actual game results
        if row['player_id'] in results_map:
            actual_stat = results_map[row['player_id']]
            if actual_stat is not None:
                if actual_stat > actual_line:
                    results_over += 1
                elif actual_stat < actual_line:
                    results_under += 1
                
                if actual_stat > projected_line:
                    projected_over += 1
                elif actual_stat < projected_line:
                    projected_under += 1
    
    if not errors:
        return None
    
    total = len(errors)
    
    # Calculate over rates
    results_total = results_over + results_under
    vs_results_over_rate = 100 * results_over / results_total if results_total > 0 else 50.0
    
    proj_total = projected_over + projected_under
    projected_over_rate = 100 * projected_over / proj_total if proj_total > 0 else 50.0
    
    return BacktestResult(
        date=test_date,
        prop_type=prop_type,
        total_comparisons=total,
        mean_absolute_error=sum(errors) / total,
        mean_signed_error=sum(signed_errors) / total,
        exact_matches=exact,
        within_half=within_half,
        within_one=within_one,
        within_two=within_two,
        vs_results_over_rate=vs_results_over_rate,
        projected_over_rate=projected_over_rate,
    )


def backtest_all_dates(
    conn: sqlite3.Connection,
    prop_type: str = 'PTS',
    min_comparisons: int = 10,
    config: ProjectionConfig = None
) -> BacktestSummary:
    """
    Backtest projections across all dates with sportsbook lines.
    
    Args:
        conn: Database connection
        prop_type: PTS, REB, or AST
        min_comparisons: Minimum comparisons required per date
        config: Projection configuration
        
    Returns:
        BacktestSummary with aggregated metrics
    """
    # Get all dates with lines
    dates_with_lines = conn.execute("""
        SELECT DISTINCT as_of_date 
        FROM sportsbook_lines 
        WHERE prop_type = ?
        ORDER BY as_of_date
    """, (prop_type,)).fetchall()
    
    all_results = []
    dates_tested = []
    
    for row in dates_with_lines:
        date = row['as_of_date']
        result = backtest_single_date(conn, date, prop_type, config)
        if result and result.total_comparisons >= min_comparisons:
            all_results.append(result)
            dates_tested.append(date)
    
    if not all_results:
        return BacktestSummary(
            prop_type=prop_type,
            num_dates=0,
            total_comparisons=0,
            avg_mae=0,
            avg_signed_error=0,
            overall_exact_match_pct=0,
            overall_within_half_pct=0,
            overall_within_one_pct=0,
            overall_within_two_pct=0,
            overall_vs_results_over_rate=50.0,
            dates_tested=[],
        )
    
    # Aggregate metrics
    total_comparisons = sum(r.total_comparisons for r in all_results)
    weighted_mae = sum(r.mean_absolute_error * r.total_comparisons for r in all_results) / total_comparisons
    weighted_signed = sum(r.mean_signed_error * r.total_comparisons for r in all_results) / total_comparisons
    
    total_exact = sum(r.exact_matches for r in all_results)
    total_half = sum(r.within_half for r in all_results)
    total_one = sum(r.within_one for r in all_results)
    total_two = sum(r.within_two for r in all_results)
    
    # Weighted average of over rates
    over_rates = [r.vs_results_over_rate * r.total_comparisons for r in all_results if r.vs_results_over_rate > 0]
    avg_over_rate = sum(over_rates) / total_comparisons if over_rates else 50.0
    
    return BacktestSummary(
        prop_type=prop_type,
        num_dates=len(all_results),
        total_comparisons=total_comparisons,
        avg_mae=weighted_mae,
        avg_signed_error=weighted_signed,
        overall_exact_match_pct=100 * total_exact / total_comparisons,
        overall_within_half_pct=100 * total_half / total_comparisons,
        overall_within_one_pct=100 * total_one / total_comparisons,
        overall_within_two_pct=100 * total_two / total_comparisons,
        overall_vs_results_over_rate=avg_over_rate,
        dates_tested=dates_tested,
    )


def compare_projection_accuracy_vs_betting(
    conn: sqlite3.Connection,
    test_date: str,
    prop_type: str = 'PTS'
) -> Dict:
    """
    Compare how our projections would have performed vs the sportsbook lines.
    
    This simulates if we bet on our projections vs the actual lines.
    
    Returns:
        Dict with betting simulation results
    """
    actual_lines = conn.execute("""
        SELECT 
            sl.player_id,
            p.name as player_name,
            AVG(sl.line) as actual_line
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = ? AND sl.prop_type = ?
        GROUP BY sl.player_id
    """, (test_date, prop_type)).fetchall()
    
    # Get game results
    stat_col = prop_type.lower()
    results_map = {}
    game_results = conn.execute(f"""
        SELECT 
            bp.player_id,
            bp.{stat_col} as actual_stat
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE g.game_date = ? AND UPPER(bp.status) = 'PLAYED'
    """, (test_date,)).fetchall()
    
    for row in game_results:
        if row['actual_stat'] is not None:
            results_map[row['player_id']] = row['actual_stat']
    
    # Calculate betting outcomes
    config = ProjectionConfig()
    
    results = {
        'date': test_date,
        'prop_type': prop_type,
        'total_bets': 0,
        'our_wins': 0,
        'our_losses': 0,
        'pushes': 0,
        'sportsbook_wins': 0,
        'sportsbook_losses': 0,
        'our_edge_bets': [],  # Where our projection differs significantly
    }
    
    for row in actual_lines:
        player_id = row['player_id']
        actual_line = row['actual_line']
        
        if player_id not in results_map:
            continue
        
        actual_stat = results_map[player_id]
        
        projection = project_player_line(conn, player_id, prop_type, test_date, config)
        if not projection:
            continue
        
        projected_line = projection.projected_line
        
        results['total_bets'] += 1
        
        # Our "bet" based on projection vs actual line
        our_bet = None
        if projected_line > actual_line + 0.5:
            our_bet = 'OVER'  # We project higher, so bet over
        elif projected_line < actual_line - 0.5:
            our_bet = 'UNDER'  # We project lower, so bet under
        
        # Outcome
        if actual_stat > actual_line:
            actual_outcome = 'OVER'
        elif actual_stat < actual_line:
            actual_outcome = 'UNDER'
        else:
            actual_outcome = 'PUSH'
        
        if actual_outcome == 'PUSH':
            results['pushes'] += 1
            continue
        
        # Track sportsbook performance (ideally 50/50)
        if actual_outcome == 'UNDER':
            results['sportsbook_wins'] += 1
        else:
            results['sportsbook_losses'] += 1
        
        # Track our bets if we had an opinion
        if our_bet:
            if our_bet == actual_outcome:
                results['our_wins'] += 1
            else:
                results['our_losses'] += 1
            
            results['our_edge_bets'].append({
                'player': row['player_name'],
                'our_projection': projected_line,
                'actual_line': actual_line,
                'our_bet': our_bet,
                'actual_result': actual_stat,
                'outcome': 'WIN' if our_bet == actual_outcome else 'LOSS',
            })
    
    return results


def store_backtest_results(
    conn: sqlite3.Connection,
    summary: BacktestSummary
) -> None:
    """Store backtest results in database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY,
            run_date TEXT NOT NULL,
            prop_type TEXT NOT NULL,
            num_dates INTEGER,
            total_comparisons INTEGER,
            avg_mae REAL,
            avg_signed_error REAL,
            exact_match_pct REAL,
            within_half_pct REAL,
            within_one_pct REAL,
            within_two_pct REAL,
            vs_results_over_rate REAL,
            dates_tested TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    
    conn.execute("""
        INSERT INTO backtest_results
        (run_date, prop_type, num_dates, total_comparisons, avg_mae, avg_signed_error,
         exact_match_pct, within_half_pct, within_one_pct, within_two_pct,
         vs_results_over_rate, dates_tested)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d"),
        summary.prop_type,
        summary.num_dates,
        summary.total_comparisons,
        summary.avg_mae,
        summary.avg_signed_error,
        summary.overall_exact_match_pct,
        summary.overall_within_half_pct,
        summary.overall_within_one_pct,
        summary.overall_within_two_pct,
        summary.overall_vs_results_over_rate,
        json.dumps(summary.dates_tested),
    ))


def print_backtest_report(conn: sqlite3.Connection):
    """Print comprehensive backtest report."""
    print("\n" + "=" * 80)
    print("LINE PROJECTION BACKTESTING REPORT")
    print("=" * 80)
    
    for prop_type in ['PTS', 'REB']:
        summary = backtest_all_dates(conn, prop_type)
        
        print(f"\n{prop_type} PROJECTIONS")
        print("-" * 60)
        
        if summary.num_dates == 0:
            print("No data available for backtesting")
            continue
        
        print(f"Dates tested: {summary.num_dates}")
        print(f"Total comparisons: {summary.total_comparisons}")
        print(f"\nAccuracy vs Sportsbook Lines:")
        print(f"  Mean Absolute Error: {summary.avg_mae:.2f}")
        print(f"  Mean Signed Error: {summary.avg_signed_error:+.2f}")
        print(f"  Exact matches: {summary.overall_exact_match_pct:.1f}%")
        print(f"  Within 0.5: {summary.overall_within_half_pct:.1f}%")
        print(f"  Within 1.0: {summary.overall_within_one_pct:.1f}%")
        print(f"  Within 2.0: {summary.overall_within_two_pct:.1f}%")
        print(f"\nActual Results vs Lines:")
        print(f"  Over rate: {summary.overall_vs_results_over_rate:.1f}%")
        print(f"  (50% = perfectly calibrated lines)")
        
        # Store results
        store_backtest_results(conn, summary)
    
    # Compare betting simulation for most recent date
    dates = conn.execute("""
        SELECT DISTINCT as_of_date FROM sportsbook_lines 
        JOIN games g ON g.game_date = as_of_date
        ORDER BY as_of_date DESC LIMIT 1
    """).fetchone()
    
    if dates:
        recent_date = dates['as_of_date']
        print(f"\n\nBETTING SIMULATION FOR {recent_date}")
        print("-" * 60)
        
        for prop_type in ['PTS', 'REB']:
            betting = compare_projection_accuracy_vs_betting(conn, recent_date, prop_type)
            
            if betting['total_bets'] == 0:
                continue
            
            print(f"\n{prop_type}:")
            print(f"  Total props: {betting['total_bets']}")
            print(f"  Sportsbook vs Results: {betting['sportsbook_wins']}W-{betting['sportsbook_losses']}L")
            
            our_total = betting['our_wins'] + betting['our_losses']
            if our_total > 0:
                win_rate = 100 * betting['our_wins'] / our_total
                print(f"  Our Edge Bets: {betting['our_wins']}W-{betting['our_losses']}L ({win_rate:.1f}%)")
                
                if betting['our_edge_bets']:
                    print(f"\n  Our Edge Bet Details:")
                    for bet in betting['our_edge_bets'][:10]:
                        proj_vs_line = bet['our_projection'] - bet['actual_line']
                        print(f"    {bet['player']}: proj {bet['our_projection']:.1f} vs line {bet['actual_line']:.1f} "
                              f"({proj_vs_line:+.1f}) → {bet['our_bet']} = {bet['outcome']} "
                              f"(actual: {bet['actual_result']})")
    
    conn.commit()
    print("\n" + "=" * 80)
    print("Backtest results saved to database")


if __name__ == "__main__":
    db_path = Path(__file__).parent.parent.parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print_backtest_report(conn)
    
    conn.close()
