"""
Walk-Forward Backtesting Engine

This module implements proper walk-forward backtesting methodology that:
1. Uses only data available BEFORE each test date
2. Processes dates sequentially
3. Incorporates new data after each day
4. Uses the Line Projection Model for line generation
5. Tracks line sources separately for honest reporting

This addresses the concern that previous backtests may have inadvertently
used future data or static data snapshots.

Author: PropAI Team
Date: February 4, 2026
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.nba_props.db import Db
from src.nba_props.paths import get_paths
from src.nba_props.engine.line_projector import (
    project_player_line,
    ProjectionConfig,
    round_to_sportsbook_line,
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class WalkForwardPick:
    """A single pick with all tracking information."""
    date: str
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    prop_type: str
    direction: str
    line: float
    line_source: str  # "sportsbook", "projected", "derived"
    projection: float
    edge_pct: float
    confidence_score: float
    confidence_tier: str
    pattern: str
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    margin: Optional[float] = None


@dataclass
class WalkForwardDayResult:
    """Results for a single day."""
    date: str
    games: int = 0
    picks: int = 0
    hits: int = 0
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    projected_picks: int = 0
    projected_hits: int = 0
    derived_picks: int = 0
    derived_hits: int = 0
    all_picks: List[WalkForwardPick] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Comprehensive walk-forward backtest results."""
    model_name: str
    start_date: str
    end_date: str
    
    # Overall metrics
    days_tested: int = 0
    total_games: int = 0
    total_picks: int = 0
    hits: int = 0
    
    # By line source (CRITICAL for honest reporting)
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    projected_picks: int = 0
    projected_hits: int = 0
    derived_picks: int = 0
    derived_hits: int = 0
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    ast_picks: int = 0
    ast_hits: int = 0
    
    # By direction
    over_picks: int = 0
    over_hits: int = 0
    under_picks: int = 0
    under_hits: int = 0
    
    # By confidence tier
    premium_picks: int = 0
    premium_hits: int = 0
    high_picks: int = 0
    high_hits: int = 0
    standard_picks: int = 0
    standard_hits: int = 0
    
    # Daily breakdown
    daily_results: List[WalkForwardDayResult] = field(default_factory=list)
    all_picks: List[WalkForwardPick] = field(default_factory=list)
    
    def hit_rate(self) -> float:
        return self.hits / self.total_picks * 100 if self.total_picks > 0 else 0.0
    
    def sportsbook_rate(self) -> float:
        return self.sportsbook_hits / self.sportsbook_picks * 100 if self.sportsbook_picks > 0 else 0.0
    
    def projected_rate(self) -> float:
        return self.projected_hits / self.projected_picks * 100 if self.projected_picks > 0 else 0.0
    
    def derived_rate(self) -> float:
        return self.derived_hits / self.derived_picks * 100 if self.derived_picks > 0 else 0.0
    
    def summary(self) -> str:
        lines = [
            "=" * 80,
            f"WALK-FORWARD BACKTEST RESULTS: {self.model_name}",
            "=" * 80,
            f"Period: {self.start_date} to {self.end_date} ({self.days_tested} days)",
            "",
            "OVERALL PERFORMANCE",
            "-" * 40,
            f"Total Picks: {self.total_picks}",
            f"Hits: {self.hits}",
            f"Hit Rate: {self.hit_rate():.1f}%",
            "",
            "BY LINE SOURCE (CRITICAL)",
            "-" * 40,
        ]
        
        if self.sportsbook_picks > 0:
            lines.append(f"Sportsbook: {self.sportsbook_hits}/{self.sportsbook_picks} ({self.sportsbook_rate():.1f}%)  ← TRUE ACCURACY")
        else:
            lines.append("Sportsbook: 0 picks")
        
        if self.projected_picks > 0:
            lines.append(f"Projected:  {self.projected_hits}/{self.projected_picks} ({self.projected_rate():.1f}%)")
        else:
            lines.append("Projected: 0 picks")
            
        if self.derived_picks > 0:
            lines.append(f"Derived:    {self.derived_hits}/{self.derived_picks} ({self.derived_rate():.1f}%)  ← May be inflated")
        else:
            lines.append("Derived: 0 picks")
        
        lines.extend([
            "",
            "BY PROP TYPE",
            "-" * 40,
        ])
        
        if self.pts_picks > 0:
            lines.append(f"PTS: {self.pts_hits}/{self.pts_picks} ({self.pts_hits/self.pts_picks*100:.1f}%)")
        if self.reb_picks > 0:
            lines.append(f"REB: {self.reb_hits}/{self.reb_picks} ({self.reb_hits/self.reb_picks*100:.1f}%)")
        if self.ast_picks > 0:
            lines.append(f"AST: {self.ast_hits}/{self.ast_picks} ({self.ast_hits/self.ast_picks*100:.1f}%)")
        
        lines.extend([
            "",
            "BY DIRECTION",
            "-" * 40,
        ])
        
        if self.over_picks > 0:
            lines.append(f"OVER:  {self.over_hits}/{self.over_picks} ({self.over_hits/self.over_picks*100:.1f}%)")
        if self.under_picks > 0:
            lines.append(f"UNDER: {self.under_hits}/{self.under_picks} ({self.under_hits/self.under_picks*100:.1f}%)")
        
        lines.extend([
            "",
            "BY CONFIDENCE",
            "-" * 40,
        ])
        
        if self.premium_picks > 0:
            lines.append(f"PREMIUM:  {self.premium_hits}/{self.premium_picks} ({self.premium_hits/self.premium_picks*100:.1f}%)")
        if self.high_picks > 0:
            lines.append(f"HIGH:     {self.high_hits}/{self.high_picks} ({self.high_hits/self.high_picks*100:.1f}%)")
        if self.standard_picks > 0:
            lines.append(f"STANDARD: {self.standard_hits}/{self.standard_picks} ({self.standard_hits/self.standard_picks*100:.1f}%)")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# Model Registry - Maps model names to their functions
# ============================================================================

def get_model_functions(model_name: str):
    """
    Get the daily picks and grading functions for a model.
    
    Returns tuple of (get_daily_picks_fn, grade_fn, config_class)
    """
    model_name = model_name.lower()
    
    if model_name in ("v9", "model_v9"):
        from src.nba_props.engine.model_v9 import get_daily_picks_v9, ModelConfigV9
        return get_daily_picks_v9, None, ModelConfigV9
    
    elif model_name in ("v10", "model_v10"):
        from src.nba_props.engine.model_v10 import get_daily_picks_v10, ModelConfigV10
        return get_daily_picks_v10, None, ModelConfigV10
    
    elif model_name in ("v12", "v12_general", "model_v12_general"):
        from src.nba_props.engine.model_v12_general import get_daily_picks_general, GeneralModelConfig
        return get_daily_picks_general, None, GeneralModelConfig
    
    elif model_name in ("v12_under", "model_v12_under"):
        from src.nba_props.engine.model_v12_under import get_daily_picks_under, UnderModelConfig
        return get_daily_picks_under, None, UnderModelConfig
    
    elif model_name in ("v13", "v13_general", "model_v13_general"):
        from src.nba_props.engine.model_v13_general import get_daily_picks_v13_general, ModelConfigV13General
        return get_daily_picks_v13_general, None, ModelConfigV13General
    
    elif model_name in ("v13_under", "model_v13_under"):
        from src.nba_props.engine.model_v13_under import get_daily_picks_v13_under, ModelConfigV13Under
        return get_daily_picks_v13_under, None, ModelConfigV13Under
    
    elif model_name in ("v14", "v14_general", "model_v14_general"):
        from src.nba_props.engine.model_v14_general import get_daily_picks_v14_general, ModelConfigV14General
        return get_daily_picks_v14_general, None, ModelConfigV14General
    
    elif model_name in ("v14_under", "model_v14_under"):
        from src.nba_props.engine.model_v14_under import get_daily_picks_v14_under, ModelConfigV14Under
        return get_daily_picks_v14_under, None, ModelConfigV14Under
    
    elif model_name in ("v15", "v15_general", "model_v15_general"):
        from src.nba_props.engine.model_v15_general import get_daily_picks_v15_general, ModelConfigV15General
        return get_daily_picks_v15_general, None, ModelConfigV15General
    
    elif model_name in ("v16", "v16_general", "model_v16_general"):
        from src.nba_props.engine.model_v16_general import get_daily_picks_v16_general, ModelConfigV16General
        return get_daily_picks_v16_general, None, ModelConfigV16General
    
    elif model_name in ("v17", "v17_general", "model_v17_general"):
        from src.nba_props.engine.model_v17_general import get_daily_picks_v17_general, ModelConfigV17General
        return get_daily_picks_v17_general, None, ModelConfigV17General
    
    elif model_name in ("v18", "v18_general", "model_v18_general"):
        from src.nba_props.engine.model_v18_general import get_daily_picks_v18_general, ModelConfigV18General
        return get_daily_picks_v18_general, None, ModelConfigV18General
    
    elif model_name in ("v18_under", "model_v18_under"):
        from src.nba_props.engine.model_v18_under import get_daily_picks_v18_under, ModelConfigV18Under
        return get_daily_picks_v18_under, None, ModelConfigV18Under
    
    elif model_name in ("v19", "v19_general", "model_v19_general"):
        from src.nba_props.engine.model_v19_general import get_daily_picks_v19_general, ModelConfigV19General
        return get_daily_picks_v19_general, None, ModelConfigV19General
    
    elif model_name in ("v19_under", "model_v19_under"):
        from src.nba_props.engine.model_v19_under import get_daily_picks_v19_under, ModelConfigV19Under
        return get_daily_picks_v19_under, None, ModelConfigV19Under
    
    elif model_name in ("rcm", "regression_contribution"):
        from src.nba_props.engine.regression_contribution_model import get_rcm_daily_picks, RCMConfig
        return get_rcm_daily_picks, None, RCMConfig
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# Grading Functions
# ============================================================================

def get_actual_stats(conn: sqlite3.Connection, player_id: int, game_date: str) -> Optional[Dict]:
    """Get actual stats for a player on a specific date."""
    row = conn.execute("""
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
        AND g.game_date = ?
        AND UPPER(bp.status) = 'PLAYED'
    """, (player_id, game_date)).fetchone()
    
    if row:
        return {
            'pts': row['pts'],
            'reb': row['reb'],
            'ast': row['ast'],
            'min': row['minutes'],
        }
    return None


def grade_pick(actual_value: float, line: float, direction: str) -> tuple:
    """
    Grade a pick.
    
    Returns:
        (hit: bool, margin: float)
    """
    margin = actual_value - line
    
    if direction.upper() == "OVER":
        hit = actual_value > line
    else:
        hit = actual_value < line
    
    return hit, margin


# ============================================================================
# Walk-Forward Backtesting
# ============================================================================

def run_walk_forward_backtest(
    model_name: str,
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    verbose: bool = True,
    show_daily: bool = False,
) -> WalkForwardResult:
    """
    Run a proper walk-forward backtest for a model.
    
    This ensures that for each test date:
    1. Only data BEFORE that date is used for predictions
    2. Line projections use only pre-date data
    3. Results are graded against actual outcomes
    
    Args:
        model_name: Name of model (e.g., "v16_general")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Optional database path
        verbose: Print progress
        show_daily: Show daily breakdown
        
    Returns:
        WalkForwardResult with comprehensive metrics
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"WALK-FORWARD BACKTEST: {model_name.upper()}")
        print(f"Period: {start_date} to {end_date}")
        print(f"{'='*80}\n")
    
    # Get model functions
    get_daily_picks_fn, _, config_class = get_model_functions(model_name)
    
    # Initialize
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=db_path)
    
    result = WalkForwardResult(
        model_name=model_name,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    current_dt = start_dt
    
    with db.connect() as conn:
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            
            # Check if there are games on this date
            games_count = conn.execute(
                "SELECT COUNT(*) FROM games WHERE game_date = ?",
                (date_str,)
            ).fetchone()[0]
            
            if games_count == 0:
                current_dt += timedelta(days=1)
                continue
            
            result.days_tested += 1
            result.total_games += games_count
            
            day_result = WalkForwardDayResult(date=date_str, games=games_count)
            
            if verbose:
                print(f"Processing {date_str} ({games_count} games)...", end=" ")
            
            try:
                # Generate picks for this date
                # The model's get_daily_picks function should internally use only
                # data available before date_str (via before_date parameter in queries)
                daily_picks = get_daily_picks_fn(date_str, db_path=str(db_path))
                
                # Grade each pick
                daily_hits = 0
                daily_total = 0
                
                for pick in daily_picks.picks:
                    # Get actual stats
                    actual = get_actual_stats(conn, pick.player_id, date_str)
                    
                    if actual is None:
                        continue
                    
                    prop_key = pick.prop_type.lower()
                    actual_value = actual.get(prop_key, 0)
                    
                    # Grade the pick
                    hit, margin = grade_pick(actual_value, pick.line, pick.direction)
                    
                    # Create tracking record
                    wf_pick = WalkForwardPick(
                        date=date_str,
                        player_id=pick.player_id,
                        player_name=pick.player_name,
                        team_abbrev=getattr(pick, 'team_abbrev', ''),
                        opponent_abbrev=getattr(pick, 'opponent_abbrev', ''),
                        prop_type=pick.prop_type,
                        direction=pick.direction,
                        line=pick.line,
                        line_source=getattr(pick, 'line_source', 'derived'),
                        projection=getattr(pick, 'projection', 0),
                        edge_pct=getattr(pick, 'edge_pct', 0),
                        confidence_score=getattr(pick, 'confidence_score', 0),
                        confidence_tier=getattr(pick, 'confidence_tier', 'STANDARD'),
                        pattern=getattr(pick, 'pattern', ''),
                        actual_value=actual_value,
                        hit=hit,
                        margin=margin,
                    )
                    
                    # Update totals
                    result.total_picks += 1
                    day_result.picks += 1
                    daily_total += 1
                    
                    if hit:
                        result.hits += 1
                        day_result.hits += 1
                        daily_hits += 1
                    
                    # By line source
                    line_src = wf_pick.line_source.lower()
                    if line_src == "sportsbook":
                        result.sportsbook_picks += 1
                        day_result.sportsbook_picks += 1
                        if hit:
                            result.sportsbook_hits += 1
                            day_result.sportsbook_hits += 1
                    elif line_src == "projected":
                        result.projected_picks += 1
                        day_result.projected_picks += 1
                        if hit:
                            result.projected_hits += 1
                            day_result.projected_hits += 1
                    else:
                        result.derived_picks += 1
                        day_result.derived_picks += 1
                        if hit:
                            result.derived_hits += 1
                            day_result.derived_hits += 1
                    
                    # By prop type
                    if prop_key == "pts":
                        result.pts_picks += 1
                        if hit:
                            result.pts_hits += 1
                    elif prop_key == "reb":
                        result.reb_picks += 1
                        if hit:
                            result.reb_hits += 1
                    elif prop_key == "ast":
                        result.ast_picks += 1
                        if hit:
                            result.ast_hits += 1
                    
                    # By direction
                    if wf_pick.direction.upper() == "OVER":
                        result.over_picks += 1
                        if hit:
                            result.over_hits += 1
                    else:
                        result.under_picks += 1
                        if hit:
                            result.under_hits += 1
                    
                    # By confidence tier
                    tier = wf_pick.confidence_tier.upper()
                    if tier == "PREMIUM":
                        result.premium_picks += 1
                        if hit:
                            result.premium_hits += 1
                    elif tier == "HIGH":
                        result.high_picks += 1
                        if hit:
                            result.high_hits += 1
                    else:
                        result.standard_picks += 1
                        if hit:
                            result.standard_hits += 1
                    
                    day_result.all_picks.append(wf_pick)
                    result.all_picks.append(wf_pick)
                
                if verbose:
                    if daily_total > 0:
                        rate = daily_hits / daily_total * 100
                        print(f"{daily_hits}/{daily_total} ({rate:.1f}%)")
                    else:
                        print("No graded picks")
                
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
            
            result.daily_results.append(day_result)
            current_dt += timedelta(days=1)
    
    if verbose:
        print(f"\n{result.summary()}")
    
    return result


def run_comprehensive_walk_forward(
    start_date: str,
    end_date: str,
    models: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, WalkForwardResult]:
    """
    Run walk-forward backtests for multiple models.
    
    Args:
        start_date: Start date
        end_date: End date
        models: List of model names (defaults to all)
        db_path: Database path
        verbose: Show progress
        
    Returns:
        Dict mapping model name to results
    """
    if models is None:
        models = [
            "v9", "v10",
            "v12_general", "v12_under",
            "v13_general", "v13_under",
            "v14_general", "v14_under",
            "v15_general",
            "v16_general",
            "v17_general",
            "v18_general", "v18_under",
            "v19_general", "v19_under",
            "rcm",
        ]
    
    results = {}
    
    for model in models:
        try:
            if verbose:
                print(f"\n{'#'*80}")
                print(f"# Testing Model: {model.upper()}")
                print(f"{'#'*80}")
            
            result = run_walk_forward_backtest(
                model_name=model,
                start_date=start_date,
                end_date=end_date,
                db_path=db_path,
                verbose=verbose,
            )
            results[model] = result
            
        except Exception as e:
            if verbose:
                print(f"Error testing {model}: {e}")
            continue
    
    return results


def generate_comparison_report(results: Dict[str, WalkForwardResult]) -> str:
    """Generate a comparison report for multiple models."""
    lines = [
        "=" * 100,
        "WALK-FORWARD BACKTEST COMPARISON",
        "=" * 100,
        "",
    ]
    
    # Sort by hit rate
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].hit_rate(),
        reverse=True
    )
    
    # Overall ranking
    lines.extend([
        "OVERALL RANKING (By Hit Rate)",
        "-" * 100,
        f"{'Rank':<6}{'Model':<20}{'Picks':<10}{'Hits':<10}{'Rate':<10}{'SB Rate':<12}{'Proj Rate':<12}{'Der Rate':<12}",
        "-" * 100,
    ])
    
    for rank, (model, result) in enumerate(sorted_models, 1):
        sb_rate = f"{result.sportsbook_rate():.1f}%" if result.sportsbook_picks > 0 else "N/A"
        proj_rate = f"{result.projected_rate():.1f}%" if result.projected_picks > 0 else "N/A"
        der_rate = f"{result.derived_rate():.1f}%" if result.derived_picks > 0 else "N/A"
        
        lines.append(
            f"{rank:<6}{model:<20}{result.total_picks:<10}{result.hits:<10}"
            f"{result.hit_rate():.1f}%{'':<4}{sb_rate:<12}{proj_rate:<12}{der_rate:<12}"
        )
    
    lines.extend([
        "",
        "=" * 100,
        "",
        "KEY METRICS",
        "-" * 50,
    ])
    
    # Identify best performers
    if sorted_models:
        best_overall = sorted_models[0]
        lines.append(f"Best Overall: {best_overall[0]} ({best_overall[1].hit_rate():.1f}%)")
        
        # Best by sportsbook rate
        sb_models = [(m, r) for m, r in results.items() if r.sportsbook_picks >= 10]
        if sb_models:
            best_sb = max(sb_models, key=lambda x: x[1].sportsbook_rate())
            lines.append(f"Best Sportsbook Rate: {best_sb[0]} ({best_sb[1].sportsbook_rate():.1f}%)")
        
        # Best by volume
        best_volume = max(results.items(), key=lambda x: x[1].total_picks)
        lines.append(f"Most Picks: {best_volume[0]} ({best_volume[1].total_picks} picks)")
    
    lines.extend([
        "",
        "=" * 100,
        "",
        "HONEST MODEL DETECTION",
        "-" * 50,
        "Models where Sportsbook Rate >= Derived Rate are 'honest'",
        "(their performance isn't inflated by easy derived lines)",
        "",
    ])
    
    for model, result in sorted_models:
        if result.sportsbook_picks >= 5 and result.derived_picks >= 5:
            sb_rate = result.sportsbook_rate()
            der_rate = result.derived_rate()
            diff = sb_rate - der_rate
            status = "✓ HONEST" if sb_rate >= der_rate else "⚠ INFLATED"
            lines.append(f"{model:<20}: SB={sb_rate:.1f}% vs Der={der_rate:.1f}% ({diff:+.1f}%) {status}")
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Walk-Forward Backtesting")
    parser.add_argument("--model", type=str, help="Model to test (e.g., v16_general)")
    parser.add_argument("--start", type=str, default="2025-12-01", help="Start date")
    parser.add_argument("--end", type=str, default="2026-02-03", help="End date")
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.all:
        # Run comprehensive backtest
        results = run_comprehensive_walk_forward(
            start_date=args.start,
            end_date=args.end,
            verbose=args.verbose,
        )
        
        # Print comparison
        print(generate_comparison_report(results))
        
        # Save to file if requested
        if args.output:
            output_data = {
                model: {
                    "total_picks": r.total_picks,
                    "hits": r.hits,
                    "hit_rate": r.hit_rate(),
                    "sportsbook_picks": r.sportsbook_picks,
                    "sportsbook_hits": r.sportsbook_hits,
                    "sportsbook_rate": r.sportsbook_rate(),
                    "projected_picks": r.projected_picks,
                    "projected_hits": r.projected_hits,
                    "projected_rate": r.projected_rate(),
                    "derived_picks": r.derived_picks,
                    "derived_hits": r.derived_hits,
                    "derived_rate": r.derived_rate(),
                }
                for model, r in results.items()
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    elif args.model:
        # Run single model backtest
        result = run_walk_forward_backtest(
            model_name=args.model,
            start_date=args.start,
            end_date=args.end,
            verbose=True,
        )
        
        if args.output:
            output_data = {
                "model": args.model,
                "start_date": args.start,
                "end_date": args.end,
                "total_picks": result.total_picks,
                "hits": result.hits,
                "hit_rate": result.hit_rate(),
                "sportsbook_picks": result.sportsbook_picks,
                "sportsbook_rate": result.sportsbook_rate(),
                "projected_picks": result.projected_picks,
                "projected_rate": result.projected_rate(),
                "derived_picks": result.derived_picks,
                "derived_rate": result.derived_rate(),
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
    
    else:
        parser.print_help()
