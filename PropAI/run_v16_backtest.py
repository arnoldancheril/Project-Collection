#!/usr/bin/env python3
"""
Run Model V16 Backtest
"""
import sys
import os
from datetime import date

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nba_props.engine.model_v16_general import (
    run_backtest_v16_general,
    ModelConfigV16General,
)

def main():
    print("=" * 80)
    print("MODEL V16 GENERAL - COMPREHENSIVE BACKTEST")
    print("=" * 80)
    print()
    
    # Create config with default settings
    config = ModelConfigV16General()
    
    print("Configuration:")
    print(f"  Edge threshold (sportsbook): {config.min_edge_sportsbook:.1f}%")
    print(f"  Edge threshold (derived): {config.min_edge_derived:.1f}%")
    print(f"  Premium edge threshold: {config.min_edge_premium:.1f}%")
    print(f"  Min games: {config.min_games_required}")
    print(f"  Min minutes: {config.min_avg_minutes}")
    print(f"  Prop types: {config.prop_types}")
    print(f"  Include AST: {config.include_ast}")
    print()
    
    # Run backtest - use the available data range
    # Based on previous models, data should be available from approximately Oct 2025
    start_date = "2025-10-22"  # Start of NBA season
    end_date = "2026-02-05"      # Recent date
    
    print(f"Backtest period: {start_date} to {end_date}")
    print()
    
    try:
        result = run_backtest_v16_general(
            start_date=start_date,
            end_date=end_date,
            config=config,
            verbose=True
        )
        
        print()
        print("=" * 80)
        print("FINAL RESULTS SUMMARY")
        print("=" * 80)
        print()
        
        # Overall metrics
        print("OVERALL PERFORMANCE:")
        print(f"  Total picks: {result.total_picks}")
        print(f"  Hit rate: {result.hit_rate:.1f}%")
        print(f"  Hits: {result.hits}, Misses: {result.total_picks - result.hits}")
        print(f"  Days tested: {result.days_tested}")
        print(f"  Total games: {result.total_games}")
        print(f"  Avg picks/day: {result.total_picks / result.days_tested:.1f}" if result.days_tested > 0 else "")
        print()
        
        # By line source
        print("BY LINE SOURCE (KEY METRIC):")
        print(f"  Sportsbook lines: {result.sportsbook_hits}/{result.sportsbook_picks} = {result.sportsbook_rate:.1f}%")
        print(f"  Derived lines:    {result.derived_hits}/{result.derived_picks} = {result.derived_rate:.1f}%")
        print()
        
        # By direction
        print("BY DIRECTION:")
        over_rate = result.over_hits / result.over_picks * 100 if result.over_picks > 0 else 0
        under_rate = result.under_hits / result.under_picks * 100 if result.under_picks > 0 else 0
        print(f"  OVER:  {result.over_hits}/{result.over_picks} = {over_rate:.1f}%")
        print(f"  UNDER: {result.under_hits}/{result.under_picks} = {under_rate:.1f}%")
        print()
        
        # By prop type
        print("BY PROP TYPE:")
        pts_rate = result.pts_hits / result.pts_picks * 100 if result.pts_picks > 0 else 0
        reb_rate = result.reb_hits / result.reb_picks * 100 if result.reb_picks > 0 else 0
        ast_rate = result.ast_hits / result.ast_picks * 100 if result.ast_picks > 0 else 0
        print(f"  PTS: {result.pts_hits}/{result.pts_picks} = {pts_rate:.1f}%")
        print(f"  REB: {result.reb_hits}/{result.reb_picks} = {reb_rate:.1f}%")
        print(f"  AST: {result.ast_hits}/{result.ast_picks} = {ast_rate:.1f}%")
        print()
        
        # By prop type + direction
        print("BY PROP + DIRECTION:")
        pts_over_rate = result.pts_over_hits / result.pts_over_picks * 100 if result.pts_over_picks > 0 else 0
        pts_under_rate = result.pts_under_hits / result.pts_under_picks * 100 if result.pts_under_picks > 0 else 0
        reb_over_rate = result.reb_over_hits / result.reb_over_picks * 100 if result.reb_over_picks > 0 else 0
        reb_under_rate = result.reb_under_hits / result.reb_under_picks * 100 if result.reb_under_picks > 0 else 0
        print(f"  PTS OVER:  {result.pts_over_hits}/{result.pts_over_picks} = {pts_over_rate:.1f}%")
        print(f"  PTS UNDER: {result.pts_under_hits}/{result.pts_under_picks} = {pts_under_rate:.1f}%")
        print(f"  REB OVER:  {result.reb_over_hits}/{result.reb_over_picks} = {reb_over_rate:.1f}%")
        print(f"  REB UNDER: {result.reb_under_hits}/{result.reb_under_picks} = {reb_under_rate:.1f}%")
        print()
        
        # By pattern
        print("BY PATTERN:")
        cold_bounce_rate = result.cold_bounce_hits / result.cold_bounce_picks * 100 if result.cold_bounce_picks > 0 else 0
        usage_boost_rate = result.usage_boost_hits / result.usage_boost_picks * 100 if result.usage_boost_picks > 0 else 0
        elite_def_rate = result.elite_defense_under_hits / result.elite_defense_under_picks * 100 if result.elite_defense_under_picks > 0 else 0
        b2b_rate = result.b2b_under_hits / result.b2b_under_picks * 100 if result.b2b_under_picks > 0 else 0
        cold_streak_rate = result.cold_streak_under_hits / result.cold_streak_under_picks * 100 if result.cold_streak_under_picks > 0 else 0
        print(f"  Cold Bounce (OVER):    {result.cold_bounce_hits}/{result.cold_bounce_picks} = {cold_bounce_rate:.1f}%")
        print(f"  Usage Boost (OVER):    {result.usage_boost_hits}/{result.usage_boost_picks} = {usage_boost_rate:.1f}%")
        print(f"  Elite Defense (UNDER): {result.elite_defense_under_hits}/{result.elite_defense_under_picks} = {elite_def_rate:.1f}%")
        print(f"  B2B Fatigue (UNDER):   {result.b2b_under_hits}/{result.b2b_under_picks} = {b2b_rate:.1f}%")
        print(f"  Cold Streak (UNDER):   {result.cold_streak_under_hits}/{result.cold_streak_under_picks} = {cold_streak_rate:.1f}%")
        print()
        
        # By confidence tier
        print("BY CONFIDENCE TIER:")
        premium_rate = result.premium_hits / result.premium_picks * 100 if result.premium_picks > 0 else 0
        high_rate = result.high_hits / result.high_picks * 100 if result.high_picks > 0 else 0
        standard_rate = result.standard_hits / result.standard_picks * 100 if result.standard_picks > 0 else 0
        print(f"  PREMIUM:  {result.premium_hits}/{result.premium_picks} = {premium_rate:.1f}%")
        print(f"  HIGH:     {result.high_hits}/{result.high_picks} = {high_rate:.1f}%")
        print(f"  STANDARD: {result.standard_hits}/{result.standard_picks} = {standard_rate:.1f}%")
        
    except Exception as e:
        import traceback
        print(f"Error running backtest: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
