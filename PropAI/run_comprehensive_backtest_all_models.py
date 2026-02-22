#!/usr/bin/env python3
"""
Comprehensive Backtest Script - All Models
==========================================

This script runs backtests for all available models and generates
a comprehensive performance report using the LINE_PROJECTION_MODEL.

The backtest period is set to cover sufficient data for meaningful analysis.
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field

# Define backtest parameters
START_DATE = "2025-12-01"  # Good historical data range
END_DATE = "2026-02-03"    # Last date with box score data


@dataclass
class ModelResult:
    """Standardized result from a model backtest."""
    model_name: str
    model_id: str
    total_picks: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    
    # By line source
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    sportsbook_rate: float = 0.0
    derived_picks: int = 0
    derived_hits: int = 0
    derived_rate: float = 0.0
    projected_picks: int = 0
    projected_hits: int = 0
    projected_rate: float = 0.0
    
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
    
    # By direction
    over_picks: int = 0
    over_hits: int = 0
    over_rate: float = 0.0
    under_picks: int = 0
    under_hits: int = 0
    under_rate: float = 0.0
    
    # By tier
    premium_picks: int = 0
    premium_hits: int = 0
    premium_rate: float = 0.0
    high_picks: int = 0
    high_hits: int = 0
    high_rate: float = 0.0
    standard_picks: int = 0
    standard_hits: int = 0
    standard_rate: float = 0.0
    
    error: str = ""
    duration_seconds: float = 0.0
    
    def calculate_rates(self):
        """Calculate hit rates from pick/hit counts."""
        if self.total_picks > 0:
            self.hit_rate = (self.total_hits / self.total_picks) * 100
        if self.sportsbook_picks > 0:
            self.sportsbook_rate = (self.sportsbook_hits / self.sportsbook_picks) * 100
        if self.derived_picks > 0:
            self.derived_rate = (self.derived_hits / self.derived_picks) * 100
        if self.projected_picks > 0:
            self.projected_rate = (self.projected_hits / self.projected_picks) * 100
        if self.pts_picks > 0:
            self.pts_rate = (self.pts_hits / self.pts_picks) * 100
        if self.reb_picks > 0:
            self.reb_rate = (self.reb_hits / self.reb_picks) * 100
        if self.ast_picks > 0:
            self.ast_rate = (self.ast_hits / self.ast_picks) * 100
        if self.over_picks > 0:
            self.over_rate = (self.over_hits / self.over_picks) * 100
        if self.under_picks > 0:
            self.under_rate = (self.under_hits / self.under_picks) * 100
        if self.premium_picks > 0:
            self.premium_rate = (self.premium_hits / self.premium_picks) * 100
        if self.high_picks > 0:
            self.high_rate = (self.high_hits / self.high_picks) * 100
        if self.standard_picks > 0:
            self.standard_rate = (self.standard_hits / self.standard_picks) * 100


def safe_get(obj: Any, attrs: List[str], default: Any = 0) -> Any:
    """Safely get attribute from object, trying multiple names."""
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is not None:
            return val
        # Try dict-style access
        if hasattr(obj, '__getitem__'):
            try:
                return obj[attr]
            except (KeyError, TypeError):
                pass
    return default


def extract_result(raw_result: Any, model_name: str, model_id: str) -> ModelResult:
    """Extract standardized result from various model result formats."""
    result = ModelResult(model_name=model_name, model_id=model_id)
    
    if raw_result is None:
        result.error = "No result returned"
        return result
    
    # Total picks/hits - try various attribute names
    result.total_picks = safe_get(raw_result, ['total_picks', 'picks_count', 'total', 'num_picks', 'picks'], 0)
    result.total_hits = safe_get(raw_result, ['total_hits', 'hits_count', 'hits', 'num_hits'], 0)
    
    # Line source breakdown
    result.sportsbook_picks = safe_get(raw_result, ['sportsbook_picks', 'sb_picks'], 0)
    result.sportsbook_hits = safe_get(raw_result, ['sportsbook_hits', 'sb_hits'], 0)
    result.derived_picks = safe_get(raw_result, ['derived_picks', 'der_picks'], 0)
    result.derived_hits = safe_get(raw_result, ['derived_hits', 'der_hits'], 0)
    result.projected_picks = safe_get(raw_result, ['projected_picks', 'proj_picks'], 0)
    result.projected_hits = safe_get(raw_result, ['projected_hits', 'proj_hits'], 0)
    
    # Prop type breakdown
    result.pts_picks = safe_get(raw_result, ['pts_picks', 'pts_total'], 0)
    result.pts_hits = safe_get(raw_result, ['pts_hits'], 0)
    result.reb_picks = safe_get(raw_result, ['reb_picks', 'reb_total'], 0)
    result.reb_hits = safe_get(raw_result, ['reb_hits'], 0)
    result.ast_picks = safe_get(raw_result, ['ast_picks', 'ast_total'], 0)
    result.ast_hits = safe_get(raw_result, ['ast_hits'], 0)
    
    # Direction breakdown
    result.over_picks = safe_get(raw_result, ['over_picks', 'over_total'], 0)
    result.over_hits = safe_get(raw_result, ['over_hits'], 0)
    result.under_picks = safe_get(raw_result, ['under_picks', 'under_total'], 0)
    result.under_hits = safe_get(raw_result, ['under_hits'], 0)
    
    # Tier breakdown
    result.premium_picks = safe_get(raw_result, ['premium_picks', 'premium_total'], 0)
    result.premium_hits = safe_get(raw_result, ['premium_hits'], 0)
    result.high_picks = safe_get(raw_result, ['high_picks', 'high_total'], 0)
    result.high_hits = safe_get(raw_result, ['high_hits'], 0)
    result.standard_picks = safe_get(raw_result, ['standard_picks', 'std_picks', 'medium_picks'], 0)
    result.standard_hits = safe_get(raw_result, ['standard_hits', 'std_hits', 'medium_hits'], 0)
    
    # Check for by_prop_type dict
    by_prop = safe_get(raw_result, ['by_prop_type', 'by_prop', 'prop_breakdown'], None)
    if by_prop and isinstance(by_prop, dict):
        for key in ['PTS', 'pts']:
            if key in by_prop:
                pts_data = by_prop[key]
                result.pts_picks = safe_get(pts_data, ['picks', 'total'], result.pts_picks)
                result.pts_hits = safe_get(pts_data, ['hits'], result.pts_hits)
        for key in ['REB', 'reb']:
            if key in by_prop:
                reb_data = by_prop[key]
                result.reb_picks = safe_get(reb_data, ['picks', 'total'], result.reb_picks)
                result.reb_hits = safe_get(reb_data, ['hits'], result.reb_hits)
        for key in ['AST', 'ast']:
            if key in by_prop:
                ast_data = by_prop[key]
                result.ast_picks = safe_get(ast_data, ['picks', 'total'], result.ast_picks)
                result.ast_hits = safe_get(ast_data, ['hits'], result.ast_hits)
    
    # Check for by_direction dict
    by_dir = safe_get(raw_result, ['by_direction', 'direction_breakdown'], None)
    if by_dir and isinstance(by_dir, dict):
        for key in ['OVER', 'over', 'Over']:
            if key in by_dir:
                over_data = by_dir[key]
                result.over_picks = safe_get(over_data, ['picks', 'total'], result.over_picks)
                result.over_hits = safe_get(over_data, ['hits'], result.over_hits)
        for key in ['UNDER', 'under', 'Under']:
            if key in by_dir:
                under_data = by_dir[key]
                result.under_picks = safe_get(under_data, ['picks', 'total'], result.under_picks)
                result.under_hits = safe_get(under_data, ['hits'], result.under_hits)
    
    # Check for by_tier dict
    by_tier = safe_get(raw_result, ['by_tier', 'tier_breakdown', 'by_confidence'], None)
    if by_tier and isinstance(by_tier, dict):
        for key in ['PREMIUM', 'premium', 'Premium']:
            if key in by_tier:
                data = by_tier[key]
                result.premium_picks = safe_get(data, ['picks', 'total'], result.premium_picks)
                result.premium_hits = safe_get(data, ['hits'], result.premium_hits)
        for key in ['HIGH', 'high', 'High']:
            if key in by_tier:
                data = by_tier[key]
                result.high_picks = safe_get(data, ['picks', 'total'], result.high_picks)
                result.high_hits = safe_get(data, ['hits'], result.high_hits)
        for key in ['STANDARD', 'standard', 'Standard', 'MEDIUM', 'medium']:
            if key in by_tier:
                data = by_tier[key]
                result.standard_picks = safe_get(data, ['picks', 'total'], result.standard_picks)
                result.standard_hits = safe_get(data, ['hits'], result.standard_hits)
    
    # Check for by_line_source dict
    by_line = safe_get(raw_result, ['by_line_source', 'line_source_breakdown'], None)
    if by_line and isinstance(by_line, dict):
        for key in ['sportsbook', 'Sportsbook']:
            if key in by_line:
                data = by_line[key]
                result.sportsbook_picks = safe_get(data, ['picks', 'total'], result.sportsbook_picks)
                result.sportsbook_hits = safe_get(data, ['hits'], result.sportsbook_hits)
        for key in ['derived', 'Derived']:
            if key in by_line:
                data = by_line[key]
                result.derived_picks = safe_get(data, ['picks', 'total'], result.derived_picks)
                result.derived_hits = safe_get(data, ['hits'], result.derived_hits)
        for key in ['projected', 'Projected']:
            if key in by_line:
                data = by_line[key]
                result.projected_picks = safe_get(data, ['picks', 'total'], result.projected_picks)
                result.projected_hits = safe_get(data, ['hits'], result.projected_hits)
    
    result.calculate_rates()
    return result


def run_model_backtest(model_id: str, module_name: str, func_name: str, 
                       start_date: str, end_date: str) -> ModelResult:
    """Run backtest for a single model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_id}")
    print(f"  Module: {module_name}")
    print(f"  Function: {func_name}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Dynamic import
        module = __import__(f'src.nba_props.engine.{module_name}', fromlist=[func_name])
        backtest_func = getattr(module, func_name)
        
        # Run backtest
        raw_result = backtest_func(start_date, end_date)
        
        # Extract standardized result
        result = extract_result(raw_result, model_id, model_id)
        result.duration_seconds = time.time() - start_time
        
        print(f"\n  Results:")
        print(f"    Total: {result.total_hits}/{result.total_picks} ({result.hit_rate:.1f}%)")
        if result.sportsbook_picks > 0:
            print(f"    Sportsbook: {result.sportsbook_hits}/{result.sportsbook_picks} ({result.sportsbook_rate:.1f}%)")
        if result.derived_picks > 0:
            print(f"    Derived: {result.derived_hits}/{result.derived_picks} ({result.derived_rate:.1f}%)")
        if result.pts_picks > 0:
            print(f"    PTS: {result.pts_hits}/{result.pts_picks} ({result.pts_rate:.1f}%)")
        if result.reb_picks > 0:
            print(f"    REB: {result.reb_hits}/{result.reb_picks} ({result.reb_rate:.1f}%)")
        print(f"    Duration: {result.duration_seconds:.1f}s")
        
        return result
        
    except Exception as e:
        result = ModelResult(model_name=model_id, model_id=model_id)
        result.error = str(e)
        result.duration_seconds = time.time() - start_time
        print(f"  ERROR: {e}")
        traceback.print_exc()
        return result


def main():
    """Run comprehensive backtests for all models."""
    print("=" * 80)
    print("COMPREHENSIVE MODEL BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Define all models to test
    models = [
        # General models
        ("V9", "model_v9", "run_backtest_v9"),
        ("V10", "model_v10", "run_backtest_v10"),
        ("V12 General", "model_v12_general", "run_backtest_general"),
        ("V13 General", "model_v13_general", "run_backtest_v13_general"),
        ("V14 General", "model_v14_general", "run_backtest_v14_general"),
        ("V15 General", "model_v15_general", "run_backtest_v15_general"),
        ("V16 General", "model_v16_general", "run_backtest_v16_general"),
        ("V17 General", "model_v17_general", "run_backtest_v17_general"),
        ("V18 General", "model_v18_general", "run_backtest_v18_general"),
        ("V19 General", "model_v19_general", "run_backtest_v19_general"),
        
        # Under models
        ("V13 Under", "model_v13_under", "run_backtest_v13_under"),
        ("V14 Under", "model_v14_under", "run_backtest_v14_under"),
        ("V15 Under", "model_v15_under", "run_backtest_v15_under"),
        ("V17 Under", "model_v17_under", "run_backtest_v17_under"),
        ("V18 Under", "model_v18_under", "run_backtest_v18_under"),
        ("V19 Under", "model_v19_under", "run_backtest_v19_under"),
        
        # Specialized models
        ("Under Model V2", "under_model_v2", "backtest_under_model_v2"),
        ("RCM", "regression_contribution_model", "run_rcm_backtest"),
        ("Production", "model_production", "run_backtest"),
        ("Final", "model_final", "run_full_backtest"),
    ]
    
    results: List[ModelResult] = []
    
    for model_id, module_name, func_name in models:
        result = run_model_backtest(model_id, module_name, func_name, START_DATE, END_DATE)
        results.append(result)
    
    # Generate report
    print("\n\n")
    print("=" * 100)
    print("COMPREHENSIVE BACKTEST RESULTS SUMMARY")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("=" * 100)
    
    # Sort by hit rate
    valid_results = [r for r in results if r.total_picks > 0 and not r.error]
    valid_results.sort(key=lambda x: x.hit_rate, reverse=True)
    
    print("\n" + "=" * 100)
    print("OVERALL RANKINGS (by Hit Rate)")
    print("=" * 100)
    print(f"{'Rank':<5} {'Model':<20} {'Picks':>7} {'Hits':>7} {'Rate':>8} {'PTS':>10} {'REB':>10} {'OVER':>10} {'UNDER':>10}")
    print("-" * 100)
    
    for i, r in enumerate(valid_results, 1):
        pts_str = f"{r.pts_rate:.1f}%" if r.pts_picks > 0 else "-"
        reb_str = f"{r.reb_rate:.1f}%" if r.reb_picks > 0 else "-"
        over_str = f"{r.over_rate:.1f}%" if r.over_picks > 0 else "-"
        under_str = f"{r.under_rate:.1f}%" if r.under_picks > 0 else "-"
        print(f"{i:<5} {r.model_name:<20} {r.total_picks:>7} {r.total_hits:>7} {r.hit_rate:>7.1f}% {pts_str:>10} {reb_str:>10} {over_str:>10} {under_str:>10}")
    
    # Print errors
    error_results = [r for r in results if r.error]
    if error_results:
        print("\n" + "=" * 100)
        print("MODELS WITH ERRORS")
        print("=" * 100)
        for r in error_results:
            print(f"  {r.model_name}: {r.error[:80]}")
    
    # Print detailed breakdown
    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN BY LINE SOURCE")
    print("=" * 100)
    print(f"{'Model':<20} {'SB Picks':>10} {'SB Rate':>10} {'Der Picks':>10} {'Der Rate':>10} {'Proj Picks':>10} {'Proj Rate':>10}")
    print("-" * 100)
    
    for r in valid_results:
        sb_rate = f"{r.sportsbook_rate:.1f}%" if r.sportsbook_picks > 0 else "-"
        der_rate = f"{r.derived_rate:.1f}%" if r.derived_picks > 0 else "-"
        proj_rate = f"{r.projected_rate:.1f}%" if r.projected_picks > 0 else "-"
        print(f"{r.model_name:<20} {r.sportsbook_picks:>10} {sb_rate:>10} {r.derived_picks:>10} {der_rate:>10} {r.projected_picks:>10} {proj_rate:>10}")
    
    # Print confidence tier breakdown
    print("\n" + "=" * 100)
    print("CONFIDENCE TIER BREAKDOWN")
    print("=" * 100)
    print(f"{'Model':<20} {'Premium':>15} {'High':>15} {'Standard':>15}")
    print("-" * 100)
    
    for r in valid_results:
        prem_str = f"{r.premium_hits}/{r.premium_picks} ({r.premium_rate:.1f}%)" if r.premium_picks > 0 else "-"
        high_str = f"{r.high_hits}/{r.high_picks} ({r.high_rate:.1f}%)" if r.high_picks > 0 else "-"
        std_str = f"{r.standard_hits}/{r.standard_picks} ({r.standard_rate:.1f}%)" if r.standard_picks > 0 else "-"
        print(f"{r.model_name:<20} {prem_str:>15} {high_str:>15} {std_str:>15}")
    
    print("\n" + "=" * 100)
    print(f"Backtest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    results = main()
