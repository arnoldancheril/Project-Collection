#!/usr/bin/env python3
"""
Post-Trade-Deadline Backtest Script
====================================

This script runs a walk-forward backtest specifically for the post-trade-deadline
period (2026-02-06 onwards) to measure how well the V19.2 trade-aware model
handles roster disruptions compared to baseline performance.

Key Analyses:
1. Overall hit rate post-deadline vs pre-deadline
2. Performance on traded players specifically
3. Tank detection accuracy
4. Confidence tier distribution changes
5. Trade uncertainty factor impact

Usage:
    python run_post_deadline_backtest.py
    python run_post_deadline_backtest.py --pre-deadline   # Compare pre vs post
    python run_post_deadline_backtest.py --verbose
    python run_post_deadline_backtest.py --scan-trades     # Auto-detect trades first
"""
from __future__ import annotations

import sys
import time
import argparse
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

TRADE_DEADLINE_DATE = "2026-02-06"
PRE_DEADLINE_START = "2025-12-01"
PRE_DEADLINE_END = "2026-02-05"

# Auto-detect the most recent date with boxscore data
def _get_latest_game_date() -> str:
    """Find the latest game date in the database."""
    try:
        from src.nba_props.db import Db
        from src.nba_props.paths import get_paths
        paths = get_paths()
        db = Db(paths.db_path)
        with db.connect() as conn:
            row = conn.execute(
                "SELECT MAX(game_date) as max_date FROM games"
            ).fetchone()
            if row and row["max_date"]:
                return row["max_date"]
    except Exception:
        pass
    return "2026-02-15"  # Fallback


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TradeAwareBacktestResult:
    """Backtest result with trade-specific breakdowns."""
    period: str  # "pre-deadline" or "post-deadline"
    start_date: str = ""
    end_date: str = ""
    
    # Overall
    total_picks: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    
    # By direction
    under_picks: int = 0
    under_hits: int = 0
    under_rate: float = 0.0
    over_picks: int = 0
    over_hits: int = 0
    over_rate: float = 0.0
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    pts_rate: float = 0.0
    reb_picks: int = 0
    reb_hits: int = 0
    reb_rate: float = 0.0
    
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
    
    # By line source
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    sportsbook_rate: float = 0.0
    derived_picks: int = 0
    derived_hits: int = 0
    derived_rate: float = 0.0
    
    # Trade-specific (post-deadline only)
    traded_player_picks: int = 0
    traded_player_hits: int = 0
    traded_player_rate: float = 0.0
    non_traded_picks: int = 0
    non_traded_hits: int = 0
    non_traded_rate: float = 0.0
    
    # Tanking teams
    tanking_team_picks: int = 0
    tanking_team_hits: int = 0
    tanking_team_rate: float = 0.0
    
    # Skipped players
    players_skipped_trade: int = 0
    players_skipped_tank: int = 0
    
    # Factor analysis
    trade_factor_picks: int = 0
    trade_factor_hits: int = 0
    trade_factor_rate: float = 0.0
    
    # Daily breakdown
    daily_results: Dict[str, Dict] = field(default_factory=dict)
    
    # ROI
    theoretical_wagers: float = 0.0
    theoretical_profit: float = 0.0
    
    runtime_seconds: float = 0.0
    
    def calculate_rates(self):
        """Calculate all hit rates."""
        def rate(h, t):
            return (h / t * 100) if t > 0 else 0.0
        
        self.hit_rate = rate(self.total_hits, self.total_picks)
        self.under_rate = rate(self.under_hits, self.under_picks)
        self.over_rate = rate(self.over_hits, self.over_picks)
        self.pts_rate = rate(self.pts_hits, self.pts_picks)
        self.reb_rate = rate(self.reb_hits, self.reb_picks)
        self.premium_rate = rate(self.premium_hits, self.premium_picks)
        self.high_rate = rate(self.high_hits, self.high_picks)
        self.standard_rate = rate(self.standard_hits, self.standard_picks)
        self.sportsbook_rate = rate(self.sportsbook_hits, self.sportsbook_picks)
        self.derived_rate = rate(self.derived_hits, self.derived_picks)
        self.traded_player_rate = rate(self.traded_player_hits, self.traded_player_picks)
        self.non_traded_rate = rate(self.non_traded_hits, self.non_traded_picks)
        self.tanking_team_rate = rate(self.tanking_team_hits, self.tanking_team_picks)
        self.trade_factor_rate = rate(self.trade_factor_hits, self.trade_factor_picks)


# ============================================================================
# Core Backtest Logic
# ============================================================================

def run_post_deadline_backtest(
    start_date: str,
    end_date: str,
    period_label: str = "post-deadline",
    verbose: bool = True,
) -> TradeAwareBacktestResult:
    """
    Run a trade-aware backtest for the specified period.
    
    This uses the V19 General model with all trade-deadline enhancements
    (V19.2) and tracks trade-specific metrics.
    """
    from src.nba_props.db import Db
    from src.nba_props.paths import get_paths
    from src.nba_props.engine.model_v19_shared import (
        load_player_stats, get_games_for_date, get_players_in_game,
        get_game_context, get_sportsbook_line, get_injured_players,
        get_actual_stats, grade_pick,
    )
    from src.nba_props.engine.model_v19_general import (
        ModelConfigV19General, evaluate_player_for_prop,
        _select_best_picks,
    )
    from src.nba_props.engine.post_trade_adjustments import should_skip_player
    from src.nba_props.engine.trade_tracker import (
        get_player_trade_info_by_id, get_all_traded_players,
        auto_detect_trades_from_boxscores, auto_update_team_roster_status,
        init_trade_tables,
    )
    from src.nba_props.engine.tank_detector import (
        detect_all_tanking_teams, clear_tank_detection_cache,
    )
    from src.nba_props.team_aliases import abbrev_from_team_name

    start_time = time.time()
    paths = get_paths()
    db = Db(paths.db_path)
    config = ModelConfigV19General()
    
    result = TradeAwareBacktestResult(
        period=period_label,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    total_dates = len(dates)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  Post-Trade-Deadline Backtest: {period_label.upper()}")
        print(f"  Period: {start_date} to {end_date} ({total_dates} days)")
        print(f"{'='*70}")
    
    with db.connect() as conn:
        # V19.3: Auto-detect trades from boxscores for walk-forward compliance
        try:
            init_trade_tables(conn)
            auto_detect_trades_from_boxscores(conn, verbose=verbose)
            auto_update_team_roster_status(conn, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  [WARN] Trade auto-detection failed: {e}")

        # Pre-compute traded players set
        traded_players = set()
        try:
            all_traded = get_all_traded_players(conn)
            for t in all_traded:
                traded_players.add(t.player_id)
        except Exception:
            pass
        
        # Pre-compute tanking teams
        tanking_teams = set()
        try:
            tanking_results = detect_all_tanking_teams(conn, end_date)
            for result_item in tanking_results:
                if result_item.is_tanking:
                    tanking_teams.add(result_item.team_abbrev)
        except Exception:
            pass
        
        if verbose:
            print(f"  Traded players detected: {len(traded_players)}")
            print(f"  Tanking teams detected: {len(tanking_teams)}")
            if tanking_teams:
                print(f"    Teams: {', '.join(sorted(tanking_teams))}")
            print()
        
        for i, game_date in enumerate(dates):
            # V19.3: Clear tank detection cache for each new date
            clear_tank_detection_cache()

            # Progress
            if verbose and i % 5 == 0:
                pct = (i + 1) / total_dates * 100
                print(f"  Processing {game_date} ({pct:.0f}%)...", end="\r")
            
            games = get_games_for_date(conn, game_date)
            if not games:
                continue
            
            injured_set = get_injured_players(conn, game_date)
            daily_picks = []
            daily_skipped_trade = 0
            
            for game in games:
                team1_abbrev = abbrev_from_team_name(game["team1_name"]) or "UNK"
                team2_abbrev = abbrev_from_team_name(game["team2_name"]) or "UNK"
                
                game_context = get_game_context(conn, team1_abbrev, team2_abbrev, game_date)
                
                for team_abbrev, opp_abbrev in [
                    (team1_abbrev, team2_abbrev),
                    (team2_abbrev, team1_abbrev),
                ]:
                    player_ids = get_players_in_game(
                        conn, team_abbrev, game_date,
                        min_games=config.min_games_required // 2,
                        min_avg_minutes=15.0,
                    )
                    
                    for player_id in player_ids:
                        if player_id in injured_set:
                            continue
                        
                        # V19.1: Skip recently traded players with no data
                        skip, skip_reason = should_skip_player(
                            conn, player_id, "", team_abbrev, game_date
                        )
                        if skip:
                            daily_skipped_trade += 1
                            result.players_skipped_trade += 1
                            continue
                        
                        stats = load_player_stats(
                            conn, player_id, game_date, opp_abbrev,
                            min_games=config.min_games_required,
                            min_minutes=config.min_avg_minutes,
                            max_games=config.max_games_lookback,
                        )
                        
                        if not stats:
                            continue
                        
                        for pt in config.prop_types:
                            if pt.lower() == 'ast' and not config.include_ast:
                                continue
                            
                            picks = evaluate_player_for_prop(
                                conn, stats, pt, opp_abbrev, game_date,
                                config, game_context,
                            )
                            
                            # Tag picks with trade metadata
                            for pick in picks:
                                pick._is_traded = player_id in traded_players
                                pick._is_tanking_team = (
                                    team_abbrev in tanking_teams or
                                    opp_abbrev in tanking_teams
                                )
                                pick._has_trade_factor = any(
                                    "trade" in f.lower()
                                    for f in pick.active_factors
                                )
                            
                            daily_picks.extend(picks)
            
            # Select best picks
            selected = _select_best_picks(daily_picks, config)
            
            # Grade picks
            daily_total = 0
            daily_hits = 0
            
            for pick in selected:
                actual = get_actual_stats(conn, pick.player_id, game_date)
                if actual is None:
                    continue
                
                prop_key = pick.prop_type.lower()
                actual_value = actual.get(prop_key, 0)
                hit, margin = grade_pick(actual_value, pick.line, pick.direction)
                
                # Overall
                result.total_picks += 1
                daily_total += 1
                if hit:
                    result.total_hits += 1
                    daily_hits += 1
                
                # Direction
                if pick.direction == "UNDER":
                    result.under_picks += 1
                    if hit: result.under_hits += 1
                else:
                    result.over_picks += 1
                    if hit: result.over_hits += 1
                
                # Prop type
                if pick.prop_type.upper() == "PTS":
                    result.pts_picks += 1
                    if hit: result.pts_hits += 1
                else:
                    result.reb_picks += 1
                    if hit: result.reb_hits += 1
                
                # Tier
                if pick.confidence_tier == "PREMIUM":
                    result.premium_picks += 1
                    if hit: result.premium_hits += 1
                elif pick.confidence_tier == "HIGH":
                    result.high_picks += 1
                    if hit: result.high_hits += 1
                else:
                    result.standard_picks += 1
                    if hit: result.standard_hits += 1
                
                # Line source
                if pick.line_source == "sportsbook":
                    result.sportsbook_picks += 1
                    if hit: result.sportsbook_hits += 1
                else:
                    result.derived_picks += 1
                    if hit: result.derived_hits += 1
                
                # Trade-specific
                is_traded = getattr(pick, '_is_traded', False)
                if is_traded:
                    result.traded_player_picks += 1
                    if hit: result.traded_player_hits += 1
                else:
                    result.non_traded_picks += 1
                    if hit: result.non_traded_hits += 1
                
                is_tanking = getattr(pick, '_is_tanking_team', False)
                if is_tanking:
                    result.tanking_team_picks += 1
                    if hit: result.tanking_team_hits += 1
                
                has_trade_factor = getattr(pick, '_has_trade_factor', False)
                if has_trade_factor:
                    result.trade_factor_picks += 1
                    if hit: result.trade_factor_hits += 1
                
                # ROI (assuming -110 odds)
                result.theoretical_wagers += 100
                if hit:
                    result.theoretical_profit += 90.91
                else:
                    result.theoretical_profit -= 100
            
            # Daily breakdown
            if daily_total > 0:
                result.daily_results[game_date] = {
                    "picks": daily_total,
                    "hits": daily_hits,
                    "rate": daily_hits / daily_total * 100,
                    "skipped_trade": daily_skipped_trade,
                }
    
    result.calculate_rates()
    result.runtime_seconds = time.time() - start_time
    
    return result


# ============================================================================
# Reporting
# ============================================================================

def print_comparison_report(
    pre: Optional[TradeAwareBacktestResult],
    post: TradeAwareBacktestResult,
):
    """Print a detailed comparison report."""
    print("\n" + "=" * 80)
    print("POST-TRADE-DEADLINE BACKTEST REPORT")
    print("=" * 80)
    
    if pre:
        print(f"\n{'Metric':<35} {'Pre-Deadline':>18} {'Post-Deadline':>18} {'Delta':>10}")
        print("-" * 85)
        
        def row(label, pre_val, post_val, fmt=".1f"):
            delta = post_val - pre_val
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<33} {pre_val:>17{fmt}}% {post_val:>17{fmt}}% {sign}{delta:>8{fmt}}%")
        
        def row_count(label, pre_h, pre_t, post_h, post_t):
            pre_r = pre_h / pre_t * 100 if pre_t > 0 else 0
            post_r = post_h / post_t * 100 if post_t > 0 else 0
            delta = post_r - pre_r
            sign = "+" if delta >= 0 else ""
            pre_str = f"{pre_h}/{pre_t} ({pre_r:.1f}%)"
            post_str = f"{post_h}/{post_t} ({post_r:.1f}%)"
            print(f"  {label:<33} {pre_str:>18} {post_str:>18} {sign}{delta:>8.1f}%")
        
        print("\n  📊 OVERALL")
        row_count("Total", pre.total_hits, pre.total_picks, post.total_hits, post.total_picks)
        
        print("\n  📈 BY DIRECTION")
        row_count("UNDER", pre.under_hits, pre.under_picks, post.under_hits, post.under_picks)
        row_count("OVER", pre.over_hits, pre.over_picks, post.over_hits, post.over_picks)
        
        print("\n  🏀 BY PROP TYPE")
        row_count("PTS", pre.pts_hits, pre.pts_picks, post.pts_hits, post.pts_picks)
        row_count("REB", pre.reb_hits, pre.reb_picks, post.reb_hits, post.reb_picks)
        
        print("\n  ⭐ BY CONFIDENCE TIER")
        row_count("PREMIUM", pre.premium_hits, pre.premium_picks, post.premium_hits, post.premium_picks)
        row_count("HIGH", pre.high_hits, pre.high_picks, post.high_hits, post.high_picks)
        row_count("STANDARD", pre.standard_hits, pre.standard_picks, post.standard_hits, post.standard_picks)
        
        print("\n  📊 BY LINE SOURCE")
        row_count("Sportsbook", pre.sportsbook_hits, pre.sportsbook_picks, post.sportsbook_hits, post.sportsbook_picks)
        row_count("Derived", pre.derived_hits, pre.derived_picks, post.derived_hits, post.derived_picks)
    
    # Post-deadline specific metrics
    print(f"\n{'='*80}")
    print("TRADE-SPECIFIC ANALYSIS (Post-Deadline)")
    print("=" * 80)
    
    def detail(label, hits, total):
        rate = hits / total * 100 if total > 0 else 0
        print(f"  {label:<35} {hits:>4}/{total:<4} ({rate:.1f}%)")
    
    detail("Traded player picks", post.traded_player_hits, post.traded_player_picks)
    detail("Non-traded player picks", post.non_traded_hits, post.non_traded_picks)
    detail("Tanking team picks", post.tanking_team_hits, post.tanking_team_picks)
    detail("Trade factor active picks", post.trade_factor_hits, post.trade_factor_picks)
    
    print(f"\n  Players skipped (trade):   {post.players_skipped_trade}")
    print(f"  Players skipped (tank):    {post.players_skipped_tank}")
    
    # ROI
    if post.theoretical_wagers > 0:
        roi = post.theoretical_profit / post.theoretical_wagers * 100
        print(f"\n  💰 ROI: {roi:+.2f}% (${post.theoretical_profit:+.2f} on ${post.theoretical_wagers:.0f} wagered)")
    
    # Daily breakdown
    if post.daily_results:
        print(f"\n{'='*80}")
        print("DAILY BREAKDOWN (Post-Deadline)")
        print("=" * 80)
        print(f"  {'Date':<14} {'Picks':>6} {'Hits':>6} {'Rate':>8} {'Skipped':>10}")
        print(f"  {'-'*50}")
        
        for date_str in sorted(post.daily_results.keys()):
            day = post.daily_results[date_str]
            rate = day["rate"]
            emoji = "✅" if rate >= 55 else "⚠️" if rate >= 45 else "❌"
            print(f"  {date_str:<14} {day['picks']:>6} {day['hits']:>6} {rate:>7.1f}% {day['skipped_trade']:>8}  {emoji}")
    
    print(f"\n  Runtime: {post.runtime_seconds:.1f}s")
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-trade-deadline backtest for V19.2 trade-aware model"
    )
    parser.add_argument(
        "--pre-deadline", action="store_true",
        help="Also run pre-deadline backtest for comparison"
    )
    parser.add_argument(
        "--start-date", type=str, default=TRADE_DEADLINE_DATE,
        help=f"Start date for post-deadline backtest (default: {TRADE_DEADLINE_DATE})"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date for backtest (default: latest game in DB)"
    )
    parser.add_argument(
        "--scan-trades", action="store_true",
        help="Auto-detect trades from boxscores before backtesting"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Show detailed progress"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        args.verbose = False
    
    if args.end_date is None:
        args.end_date = _get_latest_game_date()
    
    print("=" * 80)
    print("NBA PROPS — POST-TRADE-DEADLINE BACKTEST (V19.2)")
    print(f"Trade Deadline: {TRADE_DEADLINE_DATE}")
    print(f"Post-Deadline Period: {args.start_date} to {args.end_date}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Auto-detect trades if requested
    if args.scan_trades:
        print("\n📡 Auto-detecting trades from boxscores...")
        try:
            from src.nba_props.db import Db
            from src.nba_props.paths import get_paths
            from src.nba_props.engine.trade_tracker import (
                auto_detect_trades_from_boxscores,
                auto_update_team_roster_status,
                update_post_trade_game_counts,
            )
            
            paths = get_paths()
            db = Db(paths.db_path)
            with db.connect() as conn:
                trades = auto_detect_trades_from_boxscores(
                    conn,
                    since_date=PRE_DEADLINE_START,
                    verbose=args.verbose,
                )
                if trades:
                    print(f"  Detected {len(trades)} trades")
                    update_post_trade_game_counts(conn)
                    auto_update_team_roster_status(conn, as_of_date=args.end_date)
                else:
                    print("  No new trades detected")
        except Exception as e:
            print(f"  ⚠️ Trade scan failed: {e}")
            traceback.print_exc()
    
    # Step 2: Run pre-deadline backtest (optional comparison)
    pre_result = None
    if args.pre_deadline:
        print(f"\n📊 Running pre-deadline backtest ({PRE_DEADLINE_START} to {PRE_DEADLINE_END})...")
        try:
            pre_result = run_post_deadline_backtest(
                PRE_DEADLINE_START, PRE_DEADLINE_END,
                period_label="pre-deadline",
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  ⚠️ Pre-deadline backtest failed: {e}")
            traceback.print_exc()
    
    # Step 3: Run post-deadline backtest
    print(f"\n📊 Running post-deadline backtest ({args.start_date} to {args.end_date})...")
    try:
        post_result = run_post_deadline_backtest(
            args.start_date, args.end_date,
            period_label="post-deadline",
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\n❌ Post-deadline backtest failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Print comparison report
    print_comparison_report(pre_result, post_result)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return post_result


if __name__ == "__main__":
    main()
