"""
Model V12 Combined - Dual Model Orchestrator
==============================================

This module provides the COMBINED interface for the V12 dual-model system.
It orchestrates both the General and Under models to produce a unified
set of picks for daily betting.

ARCHITECTURE:
-------------
The V12 system uses TWO specialized models:

1. V12_GENERAL (model_v12_general.py)
   - Handles OVER picks primarily (cold bounce, hot sustained, weak defense)
   - Can suggest UNDER picks as secondary picks
   - Uses pattern-based detection

2. V12_UNDER (model_v12_under.py)
   - Specialized for UNDER picks ONLY
   - Uses factor-based scoring system
   - More sophisticated UNDER analysis

COMBINATION STRATEGY:
--------------------
1. Run both models independently
2. UNDER picks from V12_Under take priority (more specialized)
3. OVER picks come from V12_General
4. Deduplicate by player/prop combination
5. Sort by confidence and apply daily limits

PICK PRIORITIZATION:
-------------------
1. PREMIUM tier picks from either model
2. HIGH tier picks from either model
3. STANDARD tier picks (with caution)

LINE SOURCE TRACKING:
--------------------
All picks track whether they used sportsbook or derived lines.
This enables honest reporting of model performance.

USAGE:
------
    from src.nba_props.engine.model_v12_combined import (
        get_daily_picks_v12,
        run_backtest_v12,
        CombinedConfig,
    )
    
    # Get combined picks for today
    picks = get_daily_picks_v12("2026-02-03")
    
    # Run combined backtest
    result = run_backtest_v12("2025-12-01", "2026-02-02")
    
    # Run backtests for individual models
    from src.nba_props.engine.model_v12_general import run_backtest_general
    from src.nba_props.engine.model_v12_under import run_backtest_under

Author: PropAI Development Team
Created: February 2026
Version: 12.0
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Set

from ..db import Db
from ..team_aliases import abbrev_from_team_name

from .model_v12_shared import (
    PropPickV12,
    DailyPicksV12,
    BacktestResultV12,
    grade_pick,
)

from .model_v12_general import (
    GeneralModelConfig,
    get_daily_picks_general,
    run_backtest_general,
)

from .model_v12_under import (
    UnderModelConfig,
    get_daily_picks_under,
    run_backtest_under,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CombinedConfig:
    """
    Configuration for the combined V12 model system.
    """
    # Model name
    model_name: str = "V12_COMBINED"
    
    # Individual model configs
    general_config: Optional[GeneralModelConfig] = None
    under_config: Optional[UnderModelConfig] = None
    
    # Combination strategy
    under_priority: bool = True          # UNDER picks from Under model take priority
    allow_general_under: bool = True     # Allow UNDER picks from General model
    
    # Pick limits
    max_picks_per_player: int = 1        # Only 1 pick per player across both models
    max_picks_per_day: int = 20
    max_over_picks: int = 12             # Cap OVER picks
    max_under_picks: int = 12            # Cap UNDER picks
    
    # Tier mixing
    tier_priority: List[str] = field(default_factory=lambda: ["PREMIUM", "HIGH", "STANDARD"])
    
    def get_general_config(self) -> GeneralModelConfig:
        """Get or create General model config."""
        if self.general_config is None:
            return GeneralModelConfig()
        return self.general_config
    
    def get_under_config(self) -> UnderModelConfig:
        """Get or create Under model config."""
        if self.under_config is None:
            return UnderModelConfig()
        return self.under_config


# ============================================================================
# Combination Logic
# ============================================================================

def combine_picks(
    general_picks: List[PropPickV12],
    under_picks: List[PropPickV12],
    config: CombinedConfig,
) -> List[PropPickV12]:
    """
    Combine picks from both models with proper deduplication.
    
    Strategy:
    1. Start with Under model's UNDER picks (more specialized)
    2. Add General model's OVER picks
    3. Optionally add General model's UNDER picks if no conflict
    4. Deduplicate by player/prop
    5. Apply limits and sort by confidence
    """
    combined = []
    seen_player_props: Set[str] = set()  # "player_id_prop_type"
    
    # Helper to check if pick already exists
    def pick_key(pick: PropPickV12) -> str:
        return f"{pick.player_id}_{pick.prop_type}"
    
    # PHASE 1: Add Under model's picks (priority for UNDER direction)
    if config.under_priority:
        under_sorted = sorted(under_picks, key=lambda p: p.confidence_score, reverse=True)
        under_added = 0
        
        for pick in under_sorted:
            if under_added >= config.max_under_picks:
                break
            
            key = pick_key(pick)
            if key not in seen_player_props:
                combined.append(pick)
                seen_player_props.add(key)
                under_added += 1
    
    # PHASE 2: Add General model's OVER picks
    general_over = [p for p in general_picks if p.direction == "OVER"]
    general_over_sorted = sorted(general_over, key=lambda p: p.confidence_score, reverse=True)
    over_added = 0
    
    for pick in general_over_sorted:
        if over_added >= config.max_over_picks:
            break
        
        key = pick_key(pick)
        if key not in seen_player_props:
            combined.append(pick)
            seen_player_props.add(key)
            over_added += 1
    
    # PHASE 3: Optionally add General model's UNDER picks if no Under model conflict
    if config.allow_general_under:
        general_under = [p for p in general_picks if p.direction == "UNDER"]
        general_under_sorted = sorted(general_under, key=lambda p: p.confidence_score, reverse=True)
        
        # Count current UNDER picks
        current_under = sum(1 for p in combined if p.direction == "UNDER")
        
        for pick in general_under_sorted:
            if current_under >= config.max_under_picks:
                break
            
            key = pick_key(pick)
            if key not in seen_player_props:
                combined.append(pick)
                seen_player_props.add(key)
                current_under += 1
    
    # PHASE 4: Enforce per-player limit
    player_counts: Dict[int, int] = {}
    filtered = []
    
    for pick in sorted(combined, key=lambda p: p.confidence_score, reverse=True):
        player_id = pick.player_id
        current = player_counts.get(player_id, 0)
        
        if current < config.max_picks_per_player:
            filtered.append(pick)
            player_counts[player_id] = current + 1
    
    # PHASE 5: Apply daily limit
    filtered = filtered[:config.max_picks_per_day]
    
    # Sort by tier priority then confidence
    def sort_key(pick: PropPickV12) -> tuple:
        tier_order = {t: i for i, t in enumerate(config.tier_priority)}
        return (tier_order.get(pick.confidence_tier, 99), -pick.confidence_score)
    
    filtered.sort(key=sort_key)
    
    return filtered


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_v12(
    game_date: str,
    config: Optional[CombinedConfig] = None,
    db_path: str = "data/db/nba_props.sqlite3",
) -> DailyPicksV12:
    """
    Generate combined picks from both V12 models.
    
    This is the MAIN ENTRY POINT for V12 predictions.
    
    Args:
        game_date: Date in YYYY-MM-DD format
        config: Combined configuration
        db_path: Path to database
        
    Returns:
        DailyPicksV12 with combined picks
    """
    if config is None:
        config = CombinedConfig()
    
    # Get picks from both models
    general_picks = get_daily_picks_general(
        game_date,
        config=config.get_general_config(),
        db_path=db_path,
    )
    
    under_picks = get_daily_picks_under(
        game_date,
        config=config.get_under_config(),
        db_path=db_path,
    )
    
    # Combine picks
    combined_picks = combine_picks(
        general_picks.picks,
        under_picks.picks,
        config,
    )
    
    # Build result
    daily = DailyPicksV12(date=game_date, games=general_picks.games)
    daily.picks = combined_picks
    
    # Aggregate line source stats
    daily.players_with_sportsbook_lines = (
        general_picks.players_with_sportsbook_lines +
        under_picks.players_with_sportsbook_lines
    )
    daily.players_with_derived_lines = (
        general_picks.players_with_derived_lines +
        under_picks.players_with_derived_lines
    )
    
    return daily


def run_backtest_v12(
    start_date: str,
    end_date: str,
    config: Optional[CombinedConfig] = None,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> BacktestResultV12:
    """
    Run comprehensive backtest for the combined V12 model.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Combined configuration
        db_path: Path to database
        verbose: Whether to print progress
        
    Returns:
        BacktestResultV12 with combined results
    """
    if config is None:
        config = CombinedConfig()
    
    db = Db(db_path)
    result = BacktestResultV12(
        start_date=start_date,
        end_date=end_date,
        model_name=config.model_name,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL V12 COMBINED BACKTEST: {start_date} to {end_date}")
        print(f"{'='*70}")
        print()
    
    with db.connect() as conn:
        dates = conn.execute(
            """
            SELECT DISTINCT game_date
            FROM games
            WHERE game_date >= ? AND game_date <= ?
            ORDER BY game_date
            """,
            (start_date, end_date),
        ).fetchall()
        
        if verbose:
            print(f"Found {len(dates)} game days to test")
        
        for date_row in dates:
            game_date = date_row["game_date"]
            result.days_tested += 1
            
            game_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM games WHERE game_date = ?",
                (game_date,),
            ).fetchone()
            result.total_games += game_count["cnt"] if game_count else 0
            
            # Get combined picks for this date
            daily_picks = get_daily_picks_v12(game_date, config, db_path)
            
            daily_hits = 0
            daily_total = 0
            
            for pick in daily_picks.picks:
                # Get actual result
                actual = conn.execute(
                    """
                    SELECT bp.pts, bp.reb, bp.ast
                    FROM boxscore_player bp
                    JOIN games g ON g.id = bp.game_id
                    WHERE bp.player_id = ?
                      AND g.game_date = ?
                      AND bp.minutes > 0
                    """,
                    (pick.player_id, game_date),
                ).fetchone()
                
                if not actual:
                    continue
                
                actual_value = actual[pick.prop_type.lower()]
                if actual_value is None:
                    continue
                
                pick = grade_pick(pick, actual_value)
                
                result.total_picks += 1
                daily_total += 1
                
                if pick.hit:
                    result.hits += 1
                    daily_hits += 1
                
                # By line source
                if pick.line_source == "sportsbook":
                    result.sportsbook_picks += 1
                    if pick.hit:
                        result.sportsbook_hits += 1
                else:
                    result.derived_picks += 1
                    if pick.hit:
                        result.derived_hits += 1
                
                # By tier
                if pick.confidence_tier == "PREMIUM":
                    result.premium_picks += 1
                    if pick.hit:
                        result.premium_hits += 1
                elif pick.confidence_tier == "HIGH":
                    result.high_picks += 1
                    if pick.hit:
                        result.high_hits += 1
                
                # By direction
                if pick.direction == "OVER":
                    result.over_picks += 1
                    if pick.hit:
                        result.over_hits += 1
                else:
                    result.under_picks += 1
                    if pick.hit:
                        result.under_hits += 1
                
                # By prop type
                if pick.prop_type == "PTS":
                    result.pts_picks += 1
                    if pick.hit:
                        result.pts_hits += 1
                elif pick.prop_type == "REB":
                    result.reb_picks += 1
                    if pick.hit:
                        result.reb_hits += 1
                elif pick.prop_type == "AST":
                    result.ast_picks += 1
                    if pick.hit:
                        result.ast_hits += 1
                
                # By model source
                model_key = f"model_{pick.model}"
                if model_key not in result.pattern_stats:
                    result.pattern_stats[model_key] = {"picks": 0, "hits": 0}
                result.pattern_stats[model_key]["picks"] += 1
                if pick.hit:
                    result.pattern_stats[model_key]["hits"] += 1
                
                result.all_picks.append(pick)
            
            if daily_total > 0:
                daily_rate = daily_hits / daily_total * 100
                result.daily_results.append({
                    "date": game_date,
                    "picks": daily_total,
                    "hits": daily_hits,
                    "rate": daily_rate,
                })
                
                if verbose:
                    print(f"  {game_date}: {daily_hits}/{daily_total} ({daily_rate:.1f}%)")
    
    if verbose:
        print()
        print(result.summary())
    
    return result


def run_all_backtests(
    start_date: str,
    end_date: str,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> Dict[str, BacktestResultV12]:
    """
    Run backtests for all V12 models and return comparison.
    
    This runs:
    1. V12_GENERAL backtest
    2. V12_UNDER backtest  
    3. V12_COMBINED backtest
    
    Returns dict with all results for comparison.
    """
    results = {}
    
    if verbose:
        print("\n" + "="*70)
        print("RUNNING ALL V12 MODEL BACKTESTS")
        print("="*70)
        print(f"Period: {start_date} to {end_date}")
        print()
    
    # General model
    if verbose:
        print("\n--- V12_GENERAL ---")
    results["V12_GENERAL"] = run_backtest_general(
        start_date, end_date, db_path=db_path, verbose=verbose
    )
    
    # Under model
    if verbose:
        print("\n--- V12_UNDER ---")
    results["V12_UNDER"] = run_backtest_under(
        start_date, end_date, db_path=db_path, verbose=verbose
    )
    
    # Combined model
    if verbose:
        print("\n--- V12_COMBINED ---")
    results["V12_COMBINED"] = run_backtest_v12(
        start_date, end_date, db_path=db_path, verbose=verbose
    )
    
    # Print comparison
    if verbose:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Model':<15} {'Overall':<12} {'Sportsbook':<12} {'Derived':<12} {'PREMIUM':<12}")
        print("-"*63)
        
        for name, r in results.items():
            overall = f"{r.hit_rate:.1f}% ({r.total_picks})"
            sb = f"{r.sportsbook_rate:.1f}% ({r.sportsbook_picks})" if r.sportsbook_picks else "N/A"
            dr = f"{r.derived_rate:.1f}% ({r.derived_picks})" if r.derived_picks else "N/A"
            prem = f"{r.premium_rate:.1f}% ({r.premium_picks})" if r.premium_picks else "N/A"
            print(f"{name:<15} {overall:<12} {sb:<12} {dr:<12} {prem:<12}")
        
        print("="*70)
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for Model V12 Combined."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V12 Combined - Dual Model System")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--all", action="store_true", help="Run all model backtests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.all and args.backtest_start and args.backtest_end:
        run_all_backtests(
            args.backtest_start,
            args.backtest_end,
            verbose=True,
        )
    elif args.backtest_start and args.backtest_end:
        result = run_backtest_v12(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose or True,
        )
    elif args.date:
        picks = get_daily_picks_v12(args.date)
        print(picks.summary())
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_v12(today)
        print(picks.summary())


if __name__ == "__main__":
    main()
