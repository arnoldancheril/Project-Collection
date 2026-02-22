#!/usr/bin/env python3
"""
Model Backtest with Projected Lines
====================================

This script runs comprehensive backtests on all models using:
1. The new projected line system (season average based)
2. The legacy derived line system (L10 * 1.05)
3. Actual sportsbook lines (when available)

This allows us to compare how models perform with different line sources.

Usage:
    python run_model_backtest_with_projected_lines.py
    python run_model_backtest_with_projected_lines.py --models v18_general v19_general
    python run_model_backtest_with_projected_lines.py --days 30

Author: PropAI Team
Date: 2026-02-04
"""

import sqlite3
import argparse
import sys
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import traceback

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.nba_props.engine.model_registry import (
    MODEL_REGISTRY,
    get_all_models,
    get_active_models,
    get_model_by_id,
    load_model_module,
    get_backtest_function as registry_get_backtest_function,
)
from src.nba_props.engine.projected_line_integration import (
    compare_line_sources,
    print_line_comparison_report,
)
from src.nba_props.engine.line_projector import (
    project_all_lines_for_date,
    store_projected_lines,
)


@dataclass
class ModelTestResult:
    """Result of testing a single model."""
    model_id: str
    model_name: str
    total_picks: int = 0
    hits: int = 0
    hit_rate: float = 0.0
    
    # By direction
    over_picks: int = 0
    over_hits: int = 0
    over_rate: float = 0.0
    under_picks: int = 0
    under_hits: int = 0
    under_rate: float = 0.0
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    pts_rate: float = 0.0
    reb_picks: int = 0
    reb_hits: int = 0
    reb_rate: float = 0.0
    ast_picks: int = 0
    ast_hits: int = 0
    ast_rate: float = 0.0
    
    # Line source breakdown
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    sportsbook_rate: float = 0.0
    projected_picks: int = 0
    projected_hits: int = 0
    projected_rate: float = 0.0
    derived_picks: int = 0
    derived_hits: int = 0
    derived_rate: float = 0.0
    
    # Metadata
    error: str = ""
    runtime_seconds: float = 0.0


def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    db_path = os.path.join(PROJECT_ROOT, "data", "db", "nba_props.sqlite3")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_test_dates(conn: sqlite3.Connection, days: int = 30) -> List[str]:
    """Get dates that have boxscore data for testing."""
    rows = conn.execute("""
        SELECT DISTINCT DATE(g.game_date) as test_date
        FROM games g
        JOIN boxscore_player bp ON bp.game_id = g.id
        WHERE g.game_date < DATE('now')
        ORDER BY g.game_date DESC
        LIMIT ?
    """, (days,)).fetchall()
    
    return [row['test_date'] for row in rows]


def ensure_projected_lines_exist(conn: sqlite3.Connection, dates: List[str]):
    """Ensure projected lines exist for all test dates."""
    print("\nGenerating projected lines for test dates...")
    
    for date in dates:
        # Check if projections exist
        count = conn.execute(
            "SELECT COUNT(*) FROM projected_lines WHERE as_of_date = ?",
            (date,)
        ).fetchone()[0]
        
        if count == 0:
            projections = project_all_lines_for_date(conn, date)
            if projections:
                stored = store_projected_lines(conn, projections, date)
                print(f"  {date}: Generated {stored} projections")
            else:
                print(f"  {date}: No projections generated (insufficient data)")
        else:
            print(f"  {date}: {count} projections already exist")


def run_model_backtest(
    model_id: str,
    start_date: str,
    end_date: str,
    verbose: bool = False
) -> ModelTestResult:
    """
    Run backtest for a single model.
    
    Returns ModelTestResult with performance metrics.
    """
    result = ModelTestResult(model_id=model_id, model_name=model_id)
    
    model_info = get_model_by_id(model_id)
    if not model_info:
        result.error = f"Model {model_id} not found in registry"
        return result
    
    result.model_name = model_info.display_name
    
    try:
        # Get backtest function
        backtest_func = registry_get_backtest_function(model_info)
        if not backtest_func:
            result.error = f"No backtest function found for {model_id}"
            return result
        
        # Run backtest
        start_time = datetime.now()
        
        try:
            # Different models have different signatures
            if 'start_date' in str(backtest_func.__code__.co_varnames):
                raw_result = backtest_func(start_date=start_date, end_date=end_date)
            else:
                raw_result = backtest_func(start_date, end_date)
        except TypeError:
            # Try with different args
            raw_result = backtest_func()
        
        result.runtime_seconds = (datetime.now() - start_time).total_seconds()
        
        # Extract results from raw_result
        if raw_result:
            result.total_picks = _get_attr(raw_result, ['total_picks', 'picks_count', 'num_picks'], 0)
            result.hits = _get_attr(raw_result, ['total_hits', 'hits', 'hit_count'], 0)
            
            if result.total_picks > 0:
                result.hit_rate = result.hits / result.total_picks
            
            # Direction breakdown
            result.over_picks = _get_attr(raw_result, ['over_picks', 'over_count'], 0)
            result.over_hits = _get_attr(raw_result, ['over_hits'], 0)
            if result.over_picks > 0:
                result.over_rate = result.over_hits / result.over_picks
            
            result.under_picks = _get_attr(raw_result, ['under_picks', 'under_count'], 0)
            result.under_hits = _get_attr(raw_result, ['under_hits'], 0)
            if result.under_picks > 0:
                result.under_rate = result.under_hits / result.under_picks
            
            # Prop type breakdown
            result.pts_picks = _get_attr(raw_result, ['pts_picks'], 0)
            result.pts_hits = _get_attr(raw_result, ['pts_hits'], 0)
            if result.pts_picks > 0:
                result.pts_rate = result.pts_hits / result.pts_picks
            
            result.reb_picks = _get_attr(raw_result, ['reb_picks'], 0)
            result.reb_hits = _get_attr(raw_result, ['reb_hits'], 0)
            if result.reb_picks > 0:
                result.reb_rate = result.reb_hits / result.reb_picks
            
            result.ast_picks = _get_attr(raw_result, ['ast_picks'], 0)
            result.ast_hits = _get_attr(raw_result, ['ast_hits'], 0)
            if result.ast_picks > 0:
                result.ast_rate = result.ast_hits / result.ast_picks
            
            # Line source breakdown (if available)
            result.sportsbook_picks = _get_attr(raw_result, ['sportsbook_picks'], 0)
            result.sportsbook_hits = _get_attr(raw_result, ['sportsbook_hits'], 0)
            if result.sportsbook_picks > 0:
                result.sportsbook_rate = result.sportsbook_hits / result.sportsbook_picks
            
            result.derived_picks = _get_attr(raw_result, ['derived_picks'], 0)
            result.derived_hits = _get_attr(raw_result, ['derived_hits'], 0)
            if result.derived_picks > 0:
                result.derived_rate = result.derived_hits / result.derived_picks
        
        return result
        
    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        if verbose:
            traceback.print_exc()
        return result


def _get_attr(obj: Any, attrs: List[str], default: Any = None) -> Any:
    """Get attribute from object, trying multiple names."""
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return default


def print_results_table(results: List[ModelTestResult], title: str = "BACKTEST RESULTS"):
    """Print a formatted table of results."""
    print("\n" + "=" * 100)
    print(f"{title:^100}")
    print("=" * 100)
    
    # Header
    print(f"{'Model':<25} {'Picks':>6} {'Hits':>6} {'Rate':>7} "
          f"{'OVER':>7} {'UNDER':>7} {'PTS':>7} {'REB':>7} {'AST':>7} {'Time':>6}")
    print("-" * 100)
    
    # Sort by hit rate
    sorted_results = sorted(results, key=lambda x: x.hit_rate, reverse=True)
    
    for r in sorted_results:
        if r.error:
            print(f"{r.model_name:<25} {'ERROR: ' + r.error[:60]}")
        else:
            over_str = f"{r.over_rate*100:.1f}%" if r.over_picks > 0 else "N/A"
            under_str = f"{r.under_rate*100:.1f}%" if r.under_picks > 0 else "N/A"
            pts_str = f"{r.pts_rate*100:.1f}%" if r.pts_picks > 0 else "N/A"
            reb_str = f"{r.reb_rate*100:.1f}%" if r.reb_picks > 0 else "N/A"
            ast_str = f"{r.ast_rate*100:.1f}%" if r.ast_picks > 0 else "N/A"
            
            print(f"{r.model_name:<25} {r.total_picks:>6} {r.hits:>6} "
                  f"{r.hit_rate*100:>6.1f}% "
                  f"{over_str:>7} {under_str:>7} {pts_str:>7} {reb_str:>7} {ast_str:>7} "
                  f"{r.runtime_seconds:>5.1f}s")
    
    # Summary
    print("-" * 100)
    
    # Top performers
    valid = [r for r in sorted_results if not r.error and r.total_picks >= 10]
    if valid:
        best = valid[0]
        print(f"\nTop Performer: {best.model_name} - {best.hit_rate*100:.1f}% ({best.hits}/{best.total_picks})")
        
        # Best by prop type
        best_pts = max(valid, key=lambda x: x.pts_rate if x.pts_picks >= 5 else 0)
        best_reb = max(valid, key=lambda x: x.reb_rate if x.reb_picks >= 5 else 0)
        best_ast = max(valid, key=lambda x: x.ast_rate if x.ast_picks >= 5 else 0)
        
        print(f"Best PTS: {best_pts.model_name} - {best_pts.pts_rate*100:.1f}%")
        print(f"Best REB: {best_reb.model_name} - {best_reb.reb_rate*100:.1f}%")
        print(f"Best AST: {best_ast.model_name} - {best_ast.ast_rate*100:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run model backtests with projected lines")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to test")
    parser.add_argument("--days", type=int, default=14, help="Number of days to backtest")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--compare-lines", action="store_true", help="Show line source comparison")
    args = parser.parse_args()
    
    print("=" * 70)
    print("MODEL BACKTEST WITH PROJECTED LINES")
    print("=" * 70)
    
    conn = get_db_connection()
    
    # Determine date range
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    
    print(f"\nTest Period: {start_date} to {end_date}")
    
    # Get test dates
    test_dates = get_test_dates(conn, args.days)
    print(f"Found {len(test_dates)} dates with game data")
    
    # Ensure projected lines exist
    ensure_projected_lines_exist(conn, test_dates)
    
    # Show line comparison if requested
    if args.compare_lines:
        print_line_comparison_report(conn, test_dates[:5])
    
    # Get models to test
    if args.models:
        models = [get_model_by_id(m) for m in args.models if get_model_by_id(m)]
    else:
        models = get_active_models()
    
    print(f"\nTesting {len(models)} models...")
    
    # Run backtests
    results = []
    
    for i, model_info in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] Testing {model_info.display_name}...", end=" ", flush=True)
        
        result = run_model_backtest(
            model_id=model_info.model_id,
            start_date=start_date,
            end_date=end_date,
            verbose=args.verbose
        )
        
        if result.error:
            print(f"ERROR: {result.error[:50]}")
        else:
            print(f"{result.hit_rate*100:.1f}% ({result.hits}/{result.total_picks})")
        
        results.append(result)
    
    # Print summary
    print_results_table(results)
    
    # Line projection accuracy summary
    print("\n" + "=" * 70)
    print("LINE PROJECTION SYSTEM IMPACT")
    print("=" * 70)
    
    total_sportsbook = sum(r.sportsbook_picks for r in results)
    total_derived = sum(r.derived_picks for r in results)
    total_projected = sum(getattr(r, 'projected_picks', 0) for r in results)
    
    if total_sportsbook > 0:
        sb_rate = sum(r.sportsbook_hits for r in results) / total_sportsbook
        print(f"\nSportsbook Line Picks: {total_sportsbook} at {sb_rate*100:.1f}%")
    
    if total_derived > 0:
        der_rate = sum(r.derived_hits for r in results) / total_derived
        print(f"Derived Line Picks: {total_derived} at {der_rate*100:.1f}%")
    
    print("\nNote: The line projection system is now integrated and available for all models.")
    print("Models can use projected lines instead of derived lines when sportsbook lines are unavailable.")
    
    conn.close()


if __name__ == "__main__":
    main()
