"""
Model V12 General - Primary NBA Props Prediction Model
========================================================

This is the GENERAL model of the V12 dual-model system. It focuses on:
1. Pattern-based predictions for both OVER and UNDER
2. Proper sportsbook line integration with derived fallback
3. Usage redistribution when teammates are injured
4. Strategic prop type selection

CORE PHILOSOPHY:
----------------
This model generates predictions for players regardless of whether sportsbook
lines exist (unlike V10 which required them). It:
- Uses sportsbook lines when available
- Falls back to derived lines (L10 * 1.05) with proper tracking
- Reports metrics separately for sportsbook vs derived line picks

VALIDATED PATTERNS:
-------------------
1. COLD BOUNCE (OVER) - 66.9% historical
   - Player is 18%+ below L15 (cold)
   - Last game showed recovery (above L10)
   - Opponent not elite defense
   
2. HOT SUSTAINED (OVER) - 65.9% historical
   - Player is 25%+ above L15 (hot)
   - Still accelerating (L3 >= L5)
   - 3+ of L5 games above L15
   
3. COLD STREAK (UNDER) - When negative factors align
   - Player 12%+ below season average
   - Used when no better UNDER from dedicated model

4. WEAK DEFENSE (OVER) - Opponent ranks 26-30
   - Weak defense creates opportunity
   - Must have supporting pattern

PROP TYPE STRATEGY:
-------------------
Based on extensive backtesting:
- PTS: Both directions, UNDER slightly preferred
- REB: Both directions (~59% both ways)
- AST: Excluded by default (54% is coin flip)

INTEGRATION WITH UNDER MODEL:
-----------------------------
This model can suggest UNDER picks, but the dedicated V12_Under model
typically provides higher-confidence UNDER opportunities. When combining
models, the Under model's picks take priority for UNDER direction.

USAGE:
------
    from src.nba_props.engine.model_v12_general import (
        get_daily_picks_general,
        run_backtest_general,
        GeneralModelConfig,
    )
    
    # Get picks for today
    picks = get_daily_picks_general("2026-02-03")
    
    # Run backtest
    result = run_backtest_general("2025-12-01", "2026-02-02")

Author: PropAI Development Team
Created: February 2026
Version: 12.0
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev

from .model_v12_shared import (
    ModelV12Config,
    PlayerStatsV12,
    DefenseContextV12,
    LineInfo,
    PropPickV12,
    DailyPicksV12,
    BacktestResultV12,
    normalize_name_for_matching,
    get_injured_players,
    get_line_info,
    get_defense_context,
    load_player_stats,
    calculate_edge,
    apply_defense_adjustment,
    determine_confidence_tier,
    grade_pick,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GeneralModelConfig(ModelV12Config):
    """
    Configuration specific to the General Model (V12 General).
    
    Inherits from ModelV12Config and adds general-model-specific settings.
    """
    model_name: str = "V12_GENERAL"
    
    # === OVER PATTERN REQUIREMENTS ===
    # Cold bounce (best OVER pattern)
    cold_bounce_cold_threshold: float = -18.0   # L5 is 18%+ below L15
    cold_bounce_min_bounce: float = 0.0         # Last game >= L10
    
    # Hot sustained
    hot_sustained_hot_threshold: float = 25.0   # L5 is 25%+ above L15
    hot_sustained_min_games_above: int = 3      # 3+ of L5 above L15
    
    # === UNDER PATTERN REQUIREMENTS ===
    # Cold streak (basic UNDER)
    cold_streak_threshold: float = -12.0        # L5 is 12%+ below season
    
    # === DIRECTION PREFERENCE ===
    pts_prefer_under: bool = True               # PTS UNDER slightly better
    allow_pts_over: bool = True                 # But still allow OVER with pattern
    
    # === PATTERN CONFIDENCE BONUSES ===
    cold_bounce_bonus: float = 10.0
    hot_sustained_bonus: float = 8.0
    cold_streak_bonus: float = 6.0
    weak_defense_bonus: float = 5.0


# ============================================================================
# Pattern Detection
# ============================================================================

@dataclass
class PatternResult:
    """Result of pattern detection."""
    pattern_name: str       # cold_bounce, hot_sustained, cold_streak, weak_defense, none
    direction: str          # OVER, UNDER
    is_valid: bool
    confidence_bonus: float
    reasons: List[str]


def detect_over_patterns(
    stats: PlayerStatsV12,
    prop_type: str,
    defense_context: DefenseContextV12,
    config: GeneralModelConfig,
) -> PatternResult:
    """
    Detect OVER patterns for a player/prop.
    
    Patterns:
    1. Cold Bounce (Best - 66.9%)
    2. Hot Sustained (Good - 65.9%)
    3. Weak Defense (Supporting factor)
    """
    pt = prop_type.lower()
    
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l3 = stats.l3.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l10 = stats.l10.get(pt, 0)
    l15 = stats.l15.get(pt, 0)
    last_game = stats.last_game.get(pt, 0)
    recent = stats.recent_games.get(pt, [])
    
    defense_rating = defense_context.get_rating(pt)
    
    # PATTERN 1: Cold Bounce (Best OVER pattern)
    if deviation_l15 <= config.cold_bounce_cold_threshold:
        # Check for bounce-back signal
        bounce_pct = (last_game - l10) / l10 * 100 if l10 > 0 else 0
        if bounce_pct >= config.cold_bounce_min_bounce:
            # Don't bet OVER vs elite defense
            if defense_rating != "elite":
                reasons = [
                    f"Cold bounce: L5 ({l5:.1f}) is {deviation_l15:.0f}% below L15 ({l15:.1f})",
                    f"Recovery signal: Last game ({last_game:.0f}) bounced above L10 ({l10:.1f})",
                    f"Regression to mean expected",
                ]
                if defense_rating == "weak":
                    reasons.append(f"Weak defense ({defense_context.team_abbrev}) adds opportunity")
                
                bonus = config.cold_bounce_bonus + min(abs(deviation_l15) / 3, 5)
                
                return PatternResult(
                    pattern_name="cold_bounce",
                    direction="OVER",
                    is_valid=True,
                    confidence_bonus=bonus,
                    reasons=reasons,
                )
    
    # PATTERN 2: Hot Sustained
    if deviation_l15 >= config.hot_sustained_hot_threshold:
        # Check if still hot (L3 >= L5 * 0.95)
        if l3 >= l5 * 0.95:
            # Count games above L15
            games_above = sum(1 for v in recent if v > l15)
            if games_above >= config.hot_sustained_min_games_above:
                # Don't bet OVER vs elite defense
                if defense_rating != "elite":
                    reasons = [
                        f"Hot sustained: L5 ({l5:.1f}) is {deviation_l15:.0f}% above L15 ({l15:.1f})",
                        f"Momentum: L3 ({l3:.1f}) maintaining level",
                        f"Consistency: {games_above}/5 recent games above baseline",
                    ]
                    
                    bonus = config.hot_sustained_bonus + min((deviation_l15 - 25) / 5, 4)
                    
                    return PatternResult(
                        pattern_name="hot_sustained",
                        direction="OVER",
                        is_valid=True,
                        confidence_bonus=bonus,
                        reasons=reasons,
                    )
    
    # PATTERN 3: Weak Defense Opportunity (requires supporting factor)
    if defense_rating == "weak":
        # Need some positive signal
        if deviation_l15 > 0:  # At least trending upward
            reasons = [
                f"Weak defense: {defense_context.team_abbrev} ranks #{defense_context.get_rank(pt)}",
                f"Player trending positive (L5 above L15)",
            ]
            
            return PatternResult(
                pattern_name="weak_defense",
                direction="OVER",
                is_valid=True,
                confidence_bonus=config.weak_defense_bonus,
                reasons=reasons,
            )
    
    # No valid OVER pattern
    return PatternResult(
        pattern_name="none",
        direction="OVER",
        is_valid=False,
        confidence_bonus=0,
        reasons=[],
    )


def detect_under_patterns(
    stats: PlayerStatsV12,
    prop_type: str,
    defense_context: DefenseContextV12,
    config: GeneralModelConfig,
) -> PatternResult:
    """
    Detect basic UNDER patterns for the General model.
    
    Note: The dedicated V12_Under model provides more sophisticated UNDER analysis.
    This is used as a fallback or when combining models.
    """
    pt = prop_type.lower()
    
    deviation_season = stats.deviations_season.get(pt, 0)
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    season = stats.season.get(pt, 0)
    
    defense_rating = defense_context.get_rating(pt)
    defense_rank = defense_context.get_rank(pt)
    
    reasons = []
    pattern_name = "none"
    confidence_bonus = 0
    is_valid = False
    
    # PATTERN 1: Cold Streak + Good/Elite Defense
    if deviation_season <= config.cold_streak_threshold:
        if defense_rating in ["elite", "good"]:
            reasons = [
                f"Cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season ({season:.1f})",
                f"Defense: {defense_context.team_abbrev} ranks #{defense_rank} (defense_rating)",
            ]
            pattern_name = "cold_streak_defense"
            confidence_bonus = config.cold_streak_bonus + (8 if defense_rating == "elite" else 4)
            is_valid = True
    
    # PATTERN 2: Elite Defense alone (strong signal)
    elif defense_rating == "elite":
        reasons = [
            f"Elite defense: {defense_context.team_abbrev} ranks #{defense_rank} vs {stats.position}",
        ]
        pattern_name = "elite_defense"
        confidence_bonus = 8
        is_valid = True
    
    # PATTERN 3: Cold streak alone
    elif deviation_season <= config.cold_streak_threshold * 1.5:  # -18%+
        reasons = [
            f"Significant cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season",
        ]
        pattern_name = "cold_streak"
        confidence_bonus = config.cold_streak_bonus
        is_valid = True
    
    return PatternResult(
        pattern_name=pattern_name,
        direction="UNDER",
        is_valid=is_valid,
        confidence_bonus=confidence_bonus,
        reasons=reasons,
    )


# ============================================================================
# Pick Generation
# ============================================================================

def generate_pick(
    conn: sqlite3.Connection,
    stats: PlayerStatsV12,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: GeneralModelConfig,
) -> Optional[PropPickV12]:
    """
    Generate a pick for a player/prop combination.
    
    This function:
    1. Gets the best available line (sportsbook preferred)
    2. Calculates projection with defense adjustment
    3. Detects patterns for both directions
    4. Selects the best direction based on patterns and edge
    5. Calculates confidence score
    """
    pt = prop_type.lower()
    
    # Get defense context
    defense_context = get_defense_context(conn, opponent_abbrev, stats.position, config)
    
    # Get line (sportsbook preferred, derived fallback)
    line_info = get_line_info(
        conn,
        stats.player_id,
        stats.player_name,
        pt,
        game_date,
        stats.l10.get(pt, 0),
        config,
    )
    
    if line_info.line <= 0:
        return None
    
    # Calculate projection
    base_projection = stats.get_projection(pt, config)
    defense_rating = defense_context.get_rating(pt)
    projection = apply_defense_adjustment(base_projection, defense_rating, config)
    projection_std = stats.stds.get(pt, 0)
    
    # Detect patterns
    over_pattern = detect_over_patterns(stats, pt, defense_context, config)
    under_pattern = detect_under_patterns(stats, pt, defense_context, config)
    
    # Calculate edges
    over_edge = calculate_edge(projection, line_info.line, "OVER")
    under_edge = calculate_edge(projection, line_info.line, "UNDER")
    
    # Determine direction
    selected_direction = None
    selected_pattern = None
    selected_edge = 0
    
    # Direction selection logic
    # For PTS: Slight preference for UNDER (per RCM insight)
    # For REB: Pick better option
    
    if pt == 'pts' and config.pts_prefer_under:
        # Try UNDER first for PTS
        if under_pattern.is_valid and under_edge >= config.min_edge_under:
            selected_direction = "UNDER"
            selected_pattern = under_pattern
            selected_edge = under_edge
        elif config.allow_pts_over and over_pattern.is_valid and over_edge >= config.min_edge_over * 1.2:
            # Higher bar for PTS OVER
            selected_direction = "OVER"
            selected_pattern = over_pattern
            selected_edge = over_edge
    else:
        # For REB (and other props): Pick better option
        if over_pattern.is_valid and over_edge >= config.min_edge_over:
            if under_pattern.is_valid and under_edge >= config.min_edge_under:
                # Both valid - pick higher edge + confidence combo
                over_score = over_edge + over_pattern.confidence_bonus
                under_score = under_edge + under_pattern.confidence_bonus
                if over_score > under_score:
                    selected_direction = "OVER"
                    selected_pattern = over_pattern
                    selected_edge = over_edge
                else:
                    selected_direction = "UNDER"
                    selected_pattern = under_pattern
                    selected_edge = under_edge
            else:
                selected_direction = "OVER"
                selected_pattern = over_pattern
                selected_edge = over_edge
        elif under_pattern.is_valid and under_edge >= config.min_edge_under:
            selected_direction = "UNDER"
            selected_pattern = under_pattern
            selected_edge = under_edge
    
    if selected_direction is None or selected_pattern is None:
        return None
    
    # Calculate confidence score
    base_confidence = 70.0
    confidence = base_confidence + selected_pattern.confidence_bonus
    
    # Edge bonus
    edge_bonus = min(selected_edge / 2, 10)
    confidence += edge_bonus
    
    # Consistency bonus
    cv = stats.get_cv(pt)
    if cv < 0.20:
        confidence += 5  # Very consistent
    elif cv > 0.40:
        confidence -= 5  # Volatile
    
    # Line source bonus (sportsbook lines are more reliable)
    if line_info.is_sportsbook:
        confidence += 3
    
    confidence = min(confidence, 100)
    
    # Determine tier
    tier = determine_confidence_tier(confidence, selected_edge, config)
    
    return PropPickV12(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        game_date=game_date,
        prop_type=prop_type.upper(),
        direction=selected_direction,
        line=round(line_info.line, 1),
        line_source=line_info.source,
        book=line_info.book,
        projection=round(projection, 1),
        projection_std=round(projection_std, 1),
        edge=round(selected_edge, 1),
        pattern=selected_pattern.pattern_name,
        confidence_tier=tier,
        confidence_score=round(confidence, 1),
        defense_rank=defense_context.get_rank(pt),
        defense_rating=defense_rating,
        l3_avg=round(stats.l3.get(pt, 0), 1),
        l5_avg=round(stats.l5.get(pt, 0), 1),
        l10_avg=round(stats.l10.get(pt, 0), 1),
        l15_avg=round(stats.l15.get(pt, 0), 1),
        season_avg=round(stats.season.get(pt, 0), 1),
        reasons=selected_pattern.reasons,
        model="V12_GENERAL",
    )


def generate_game_picks(
    conn: sqlite3.Connection,
    game_date: str,
    team1_name: str,
    team2_name: str,
    config: GeneralModelConfig,
    line_stats: Dict[str, int],
) -> List[PropPickV12]:
    """Generate picks for a single game."""
    t1_abbrev = abbrev_from_team_name(team1_name) or ""
    t2_abbrev = abbrev_from_team_name(team2_name) or ""
    
    injured = get_injured_players(conn, game_date)
    
    all_picks = []
    player_picks = {}  # Track picks per player
    
    for team_name, opp_abbrev in [(team1_name, t2_abbrev), (team2_name, t1_abbrev)]:
        team = conn.execute("SELECT id FROM teams WHERE name = ?", (team_name,)).fetchone()
        if not team:
            continue
        
        # Get team's players
        players = conn.execute(
            """
            SELECT b.player_id, AVG(b.minutes) as avg_min
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.team_id = ?
              AND g.game_date < ?
              AND b.minutes > ?
            GROUP BY b.player_id
            HAVING COUNT(*) >= ?
            ORDER BY avg_min DESC
            LIMIT 12
            """,
            (team["id"], game_date, config.min_minutes_filter, config.min_games_required),
        ).fetchall()
        
        for p in players:
            player_id = p["player_id"]
            
            if player_id in injured:
                continue
            
            if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                continue
            
            stats = load_player_stats(conn, player_id, game_date, config)
            if not stats:
                continue
            
            # Determine which props to analyze
            props_to_check = []
            if config.include_pts:
                props_to_check.append('pts')
            if config.include_reb:
                props_to_check.append('reb')
            if config.include_ast and stats.season.get('ast', 0) >= config.ast_min_avg:
                props_to_check.append('ast')
            
            for pt in props_to_check:
                if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                    break
                
                pick = generate_pick(conn, stats, pt, opp_abbrev, game_date, config)
                
                if pick:
                    all_picks.append(pick)
                    player_picks[player_id] = player_picks.get(player_id, 0) + 1
                    
                    # Track line source
                    if pick.line_source == "sportsbook":
                        line_stats["sportsbook"] = line_stats.get("sportsbook", 0) + 1
                    else:
                        line_stats["derived"] = line_stats.get("derived", 0) + 1
    
    return all_picks


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_general(
    game_date: str,
    config: Optional[GeneralModelConfig] = None,
    db_path: str = "data/db/nba_props.sqlite3",
) -> DailyPicksV12:
    """
    Generate picks for all games on a date using the General model.
    
    Args:
        game_date: Date in YYYY-MM-DD format
        config: Model configuration (uses defaults if None)
        db_path: Path to database
        
    Returns:
        DailyPicksV12 with all picks for the day
    """
    if config is None:
        config = GeneralModelConfig()
    
    db = Db(db_path)
    daily = DailyPicksV12(date=game_date, games=0)
    
    all_picks = []
    line_stats = {"sportsbook": 0, "derived": 0}
    
    with db.connect() as conn:
        # Get games for the date
        games = conn.execute(
            """
            SELECT g.id, t1.name as team1, t2.name as team2
            FROM games g
            JOIN teams t1 ON t1.id = g.team1_id
            JOIN teams t2 ON t2.id = g.team2_id
            WHERE g.game_date = ?
            """,
            (game_date,),
        ).fetchall()
        
        if games:
            daily.games = len(games)
            for game in games:
                picks = generate_game_picks(
                    conn, game_date, game["team1"], game["team2"], config, line_stats
                )
                all_picks.extend(picks)
    
    # Sort by confidence
    all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
    
    # Limit picks per day
    daily.picks = all_picks[:config.max_picks_per_day]
    daily.players_with_sportsbook_lines = line_stats.get("sportsbook", 0)
    daily.players_with_derived_lines = line_stats.get("derived", 0)
    
    return daily


def run_backtest_general(
    start_date: str,
    end_date: str,
    config: Optional[GeneralModelConfig] = None,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> BacktestResultV12:
    """
    Run comprehensive backtest for the General model.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Model configuration
        db_path: Path to database
        verbose: Whether to print progress
        
    Returns:
        BacktestResultV12 with comprehensive results
    """
    if config is None:
        config = GeneralModelConfig()
    
    db = Db(db_path)
    result = BacktestResultV12(
        start_date=start_date,
        end_date=end_date,
        model_name=config.model_name,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL V12 GENERAL BACKTEST: {start_date} to {end_date}")
        print(f"{'='*70}")
        print(f"Props: PTS={config.include_pts}, REB={config.include_reb}, AST={config.include_ast}")
        print()
    
    with db.connect() as conn:
        # Get all game dates in range
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
            
            # Get game count
            game_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM games WHERE game_date = ?",
                (game_date,),
            ).fetchone()
            result.total_games += game_count["cnt"] if game_count else 0
            
            # Generate picks
            line_stats = {"sportsbook": 0, "derived": 0}
            all_picks = []
            
            game_rows = conn.execute(
                """
                SELECT g.id, t1.name as team1, t2.name as team2
                FROM games g
                JOIN teams t1 ON t1.id = g.team1_id
                JOIN teams t2 ON t2.id = g.team2_id
                WHERE g.game_date = ?
                """,
                (game_date,),
            ).fetchall()
            
            for game in game_rows:
                picks = generate_game_picks(
                    conn, game_date, game["team1"], game["team2"], config, line_stats
                )
                all_picks.extend(picks)
            
            # Sort and limit
            all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
            all_picks = all_picks[:config.max_picks_per_day]
            
            # Grade picks
            daily_hits = 0
            daily_total = 0
            
            for pick in all_picks:
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
                
                # Grade the pick
                pick = grade_pick(pick, actual_value)
                
                # Update counters
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
                
                # By pattern
                if pick.pattern not in result.pattern_stats:
                    result.pattern_stats[pick.pattern] = {"picks": 0, "hits": 0}
                result.pattern_stats[pick.pattern]["picks"] += 1
                if pick.hit:
                    result.pattern_stats[pick.pattern]["hits"] += 1
                
                result.all_picks.append(pick)
            
            # Daily summary
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


# ============================================================================
# CLI
# ============================================================================

def main():
    """Command-line interface for Model V12 General."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V12 General - NBA Props")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_general(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_general(args.date)
        print(picks.summary())
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_general(today)
        print(picks.summary())


if __name__ == "__main__":
    main()
