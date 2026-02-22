"""
Post-Trade Adjustment Engine
=============================

The central engine that adjusts player projections to account for
trade deadline effects. This module combines trade tracking and 
tank detection to produce adjusted projections.

KEY ADJUSTMENTS:
1. Traded players → blend old/new team stats with declining weight on old
2. Teammates of departed players → usage boost (more shots/minutes)
3. Teammates of acquired players → usage reduction (fewer shots/minutes)
4. Tanking teams → minutes reduction for stars
5. New team chemistry → higher variance/lower confidence
6. Recently traded players → heavy warning flags

PROJECTION FLOW:
    Base Stats → Trade Check → Tank Check → Minutes Adj → Stat Adj → Final

Author: PropAI Team
Created: February 2026
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev, team_name_from_abbrev

from .trade_tracker import (
    TradeInfo, TeamRosterStatus, TradeAdjustedStats,
    get_player_trade_info, get_player_trade_info_by_id,
    get_team_roster_status, get_trades_affecting_team,
    init_trade_tables, update_post_trade_game_counts,
)
from .tank_detector import (
    TankDetectionResult, detect_tanking, detect_tanking_cached,
    get_tank_adjusted_minutes, KNOWN_TANKING_TEAMS,
)


# ============================================================================
# Constants
# ============================================================================

# Trade deadline date for 2025-26 season
TRADE_DEADLINE_DATE = "2026-02-06"

# How many games with new team before we trust the data
MIN_GAMES_NEW_TEAM_RELIABLE = 5
MIN_GAMES_NEW_TEAM_PARTIAL = 2

# Factor adjustments for traded players
TRADE_MINUTES_UNCERTAINTY = 0.92    # 8% minutes discount for uncertainty
TRADE_EFFICIENCY_DISCOUNT = 0.95    # 5% efficiency discount (new system)
NEW_TEAM_CHEMISTRY_PENALTY = 0.90   # 10% penalty for brand new teammates

# Usage redistribution when a star leaves
DEPARTED_STAR_USAGE_BOOST = 0.08     # 8% usage boost for remaining players
DEPARTED_STARTER_USAGE_BOOST = 0.04  # 4% for departed starters  
ARRIVED_STAR_USAGE_REDUCTION = 0.06  # 6% usage reduction for existing players

# Factor weights for the model
TRADE_FACTOR_WEIGHTS = {
    "recently_traded": 30,         # Just traded, high uncertainty
    "new_team_adjustment": 20,     # Settling into new role
    "departed_star_boost": 15,     # Teammate's star left → more usage
    "arrived_star_reduction": 10,  # New star arrived → less usage
    "tanking_team": 25,           # Team is tanking
    "roster_instability": 15,     # Multiple roster changes
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TradeContext:
    """
    Complete trade context for a player on a specific game date.
    Used by the model to adjust projections.
    """
    # Player trade status
    player_was_traded: bool = False
    trade_info: Optional[TradeInfo] = None
    
    # Team context
    team_status: Optional[TeamRosterStatus] = None
    tank_result: Optional[TankDetectionResult] = None
    team_had_changes: bool = False
    
    # Teammate changes
    departed_stars: List[str] = field(default_factory=list)
    departed_starters: List[str] = field(default_factory=list)
    arrived_players: List[str] = field(default_factory=list)
    
    # Calculated adjustments
    minutes_factor: float = 1.0        # Multiply projected minutes by this
    projection_factor: float = 1.0     # Multiply projected stats by this
    confidence_factor: float = 1.0     # Multiply confidence by this
    
    # Factors triggered (for model factor scoring)
    active_factors: Dict[str, float] = field(default_factory=dict)
    
    # Warnings for user
    warnings: List[str] = field(default_factory=list)
    
    @property
    def has_any_impact(self) -> bool:
        """Whether trade deadline has any impact on this player."""
        return (self.player_was_traded or self.team_had_changes or
                bool(self.departed_stars) or bool(self.arrived_players) or
                (self.tank_result and self.tank_result.is_tanking))
    
    @property
    def risk_level(self) -> str:
        """Risk level for betting on this player."""
        if self.player_was_traded and self.trade_info:
            if self.trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_PARTIAL:
                return "EXTREME"
            elif self.trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_RELIABLE:
                return "HIGH"
        if self.tank_result and self.tank_result.confidence >= 0.75:
            return "HIGH"
        if self.team_had_changes:
            return "MODERATE"
        return "LOW"


@dataclass
class AdjustedProjection:
    """
    A projection that has been adjusted for trade deadline effects.
    """
    # Original values
    original_minutes: float = 0.0
    original_pts: float = 0.0
    original_reb: float = 0.0
    original_ast: float = 0.0
    
    # Adjusted values
    adjusted_minutes: float = 0.0
    adjusted_pts: float = 0.0
    adjusted_reb: float = 0.0
    adjusted_ast: float = 0.0
    
    # Adjustment details
    trade_context: Optional[TradeContext] = None
    adjustments_applied: List[str] = field(default_factory=list)
    
    @property
    def pts_change_pct(self) -> float:
        if self.original_pts > 0:
            return (self.adjusted_pts - self.original_pts) / self.original_pts * 100
        return 0.0
    
    @property
    def minutes_change_pct(self) -> float:
        if self.original_minutes > 0:
            return (self.adjusted_minutes - self.original_minutes) / self.original_minutes * 100
        return 0.0


# ============================================================================
# Core Functions
# ============================================================================

def get_trade_context(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    game_date: str,
    deadline_date: str = TRADE_DEADLINE_DATE,
) -> TradeContext:
    """
    Build complete trade context for a player.
    
    This is the main entry point called by models to understand
    trade deadline effects on a specific player.
    
    Args:
        conn: Database connection
        player_id: Player ID
        player_name: Player name
        team_abbrev: Current team abbreviation
        game_date: Date of the game being projected
        deadline_date: Trade deadline date
    
    Returns:
        TradeContext with all adjustments calculated
    """
    ctx = TradeContext()
    
    # Ensure trade tables exist
    try:
        conn.execute("SELECT 1 FROM player_trades LIMIT 1")
    except sqlite3.OperationalError:
        init_trade_tables(conn)
        return ctx  # No trade data yet
    
    team_abbrev = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
    
    # =========================================================================
    # 1. Check if THIS player was traded
    # =========================================================================
    trade_info = get_player_trade_info_by_id(conn, player_id, as_of_date=game_date)
    if not trade_info:
        trade_info = get_player_trade_info(conn, player_name, as_of_date=game_date)
    
    if trade_info:
        ctx.player_was_traded = True
        ctx.trade_info = trade_info
        
        # Update game counts
        _update_games_count_for_player(conn, trade_info, game_date)
        
        # Calculate confidence discount
        ctx.confidence_factor *= trade_info.confidence_discount
        
        # Minutes uncertainty for recently traded players
        if trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_RELIABLE:
            ctx.minutes_factor *= TRADE_MINUTES_UNCERTAINTY
            ctx.projection_factor *= TRADE_EFFICIENCY_DISCOUNT
            
            if trade_info.games_with_new_team == 0:
                ctx.projection_factor *= NEW_TEAM_CHEMISTRY_PENALTY
                ctx.warnings.append(
                    f"🚨 TRADE ALERT: {player_name} traded from {trade_info.from_team} → "
                    f"{trade_info.to_team} on {trade_info.trade_date}. "
                    f"NO games with new team yet. HIGH UNCERTAINTY."
                )
                ctx.active_factors["recently_traded"] = TRADE_FACTOR_WEIGHTS["recently_traded"]
            elif trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_PARTIAL:
                ctx.warnings.append(
                    f"⚠️ TRADE WARNING: {player_name} has only {trade_info.games_with_new_team} "
                    f"games with {trade_info.to_team}. Limited data."
                )
                ctx.active_factors["recently_traded"] = TRADE_FACTOR_WEIGHTS["recently_traded"] * 0.7
            else:
                ctx.warnings.append(
                    f"ℹ️ TRADE NOTE: {player_name} settling into {trade_info.to_team} "
                    f"({trade_info.games_with_new_team} games). Adjusting projections."
                )
                ctx.active_factors["new_team_adjustment"] = TRADE_FACTOR_WEIGHTS["new_team_adjustment"]
    
    # =========================================================================
    # 2. Check team-level changes
    # =========================================================================
    team_status = get_team_roster_status(conn, team_abbrev)
    if team_status:
        ctx.team_status = team_status
        ctx.team_had_changes = team_status.had_significant_changes
        
        if team_status.had_significant_changes:
            ctx.confidence_factor *= team_status.confidence_impact
            ctx.active_factors["roster_instability"] = TRADE_FACTOR_WEIGHTS["roster_instability"]
    
    # =========================================================================
    # 3. Check for departed/arrived teammates
    # =========================================================================
    team_trades = get_trades_affecting_team(conn, team_abbrev)
    
    for departed in team_trades.get("departed", []):
        if departed.player_name.lower() == player_name.lower():
            continue  # Skip self
        
        if departed.old_team_role == "star":
            ctx.departed_stars.append(departed.player_name)
        elif departed.old_team_role == "starter":
            ctx.departed_starters.append(departed.player_name)
    
    for arrived in team_trades.get("arrived", []):
        if arrived.player_name.lower() == player_name.lower():
            continue  # Skip self
        ctx.arrived_players.append(arrived.player_name)
    
    # Usage boost from departed stars
    if ctx.departed_stars:
        boost = len(ctx.departed_stars) * DEPARTED_STAR_USAGE_BOOST
        ctx.projection_factor *= (1.0 + boost)
        ctx.minutes_factor *= (1.0 + boost * 0.5)  # Also get more minutes
        ctx.active_factors["departed_star_boost"] = TRADE_FACTOR_WEIGHTS["departed_star_boost"]
        
        names = ", ".join(ctx.departed_stars)
        ctx.warnings.append(
            f"📈 USAGE BOOST: {names} departed. "
            f"{player_name} should see {boost*100:.1f}% more usage."
        )
    elif ctx.departed_starters:
        boost = len(ctx.departed_starters) * DEPARTED_STARTER_USAGE_BOOST
        ctx.projection_factor *= (1.0 + boost)
        ctx.warnings.append(
            f"📈 Minor usage boost: {', '.join(ctx.departed_starters)} departed."
        )
    
    # Usage reduction from arrived stars
    arrived_stars = [a for a in ctx.arrived_players 
                     if _is_star_player(conn, a)]
    if arrived_stars:
        reduction = len(arrived_stars) * ARRIVED_STAR_USAGE_REDUCTION
        ctx.projection_factor *= (1.0 - reduction)
        ctx.active_factors["arrived_star_reduction"] = TRADE_FACTOR_WEIGHTS["arrived_star_reduction"]
        
        names = ", ".join(arrived_stars)
        ctx.warnings.append(
            f"📉 USAGE REDUCTION: {names} arrived. "
            f"{player_name} may see {reduction*100:.1f}% less usage."
        )
    
    # =========================================================================
    # 4. Tank detection (V19.3: use cached version for performance)
    # =========================================================================
    # Only run tank detection if game is after the deadline
    if game_date >= deadline_date:
        tank_result = detect_tanking_cached(conn, team_abbrev, deadline_date, game_date)
        ctx.tank_result = tank_result
        
        if tank_result.is_tanking:
            ctx.minutes_factor *= tank_result.star_minutes_factor
            ctx.confidence_factor *= tank_result.overall_confidence_impact
            ctx.active_factors["tanking_team"] = TRADE_FACTOR_WEIGHTS["tanking_team"]
            
            ctx.warnings.append(
                f"🏳️ TANK ALERT: {team_abbrev} showing tanking behavior "
                f"(confidence: {tank_result.confidence:.0%}). "
                f"Star minutes may be limited."
            )
    
    return ctx


def apply_trade_adjustments(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    game_date: str,
    base_pts: float,
    base_reb: float,
    base_ast: float,
    base_minutes: float,
    deadline_date: str = TRADE_DEADLINE_DATE,
) -> AdjustedProjection:
    """
    Apply trade deadline adjustments to base projections.
    
    This is the function models should call to get adjusted projections.
    
    Args:
        conn: Database connection
        player_id, player_name, team_abbrev: Player identification
        game_date: Date of the projected game
        base_pts/reb/ast/minutes: Base projections before trade adjustments
        deadline_date: Trade deadline date
    
    Returns:
        AdjustedProjection with original and adjusted values
    """
    ctx = get_trade_context(
        conn, player_id, player_name, team_abbrev, game_date, deadline_date
    )
    
    adj = AdjustedProjection(
        original_minutes=base_minutes,
        original_pts=base_pts,
        original_reb=base_reb,
        original_ast=base_ast,
        trade_context=ctx,
    )
    
    if not ctx.has_any_impact:
        # No trade effects — return original values
        adj.adjusted_minutes = base_minutes
        adj.adjusted_pts = base_pts
        adj.adjusted_reb = base_reb
        adj.adjusted_ast = base_ast
        return adj
    
    # =========================================================================
    # Apply minutes adjustment
    # =========================================================================
    adj.adjusted_minutes = base_minutes * ctx.minutes_factor
    
    # For traded players with new-team data, blend with observed minutes
    if ctx.player_was_traded and ctx.trade_info:
        new_team_mins = _get_new_team_avg_minutes(
            conn, ctx.trade_info, game_date
        )
        if new_team_mins is not None and ctx.trade_info.games_with_new_team >= MIN_GAMES_NEW_TEAM_PARTIAL:
            # Blend: weight toward observed new-team minutes
            w = ctx.trade_info.projection_weight_new_team
            adj.adjusted_minutes = (new_team_mins * w) + (adj.adjusted_minutes * (1 - w))
            adj.adjustments_applied.append(
                f"Minutes blended: {w:.0%} new-team ({new_team_mins:.1f}), "
                f"{1-w:.0%} projected ({base_minutes:.1f})"
            )
    
    # =========================================================================
    # Apply stat adjustments  
    # =========================================================================
    # Adjust based on minutes ratio
    minutes_ratio = adj.adjusted_minutes / base_minutes if base_minutes > 0 else 1.0
    
    # Per-minute rates stay similar, so scale by minutes + projection factor
    combined_factor = minutes_ratio * ctx.projection_factor
    
    adj.adjusted_pts = base_pts * combined_factor
    adj.adjusted_reb = base_reb * combined_factor
    adj.adjusted_ast = base_ast * combined_factor
    
    # For traded players with new-team data, blend with observed stats
    if ctx.player_was_traded and ctx.trade_info:
        new_team_stats = _get_new_team_avg_stats(conn, ctx.trade_info, game_date)
        if new_team_stats and ctx.trade_info.games_with_new_team >= MIN_GAMES_NEW_TEAM_PARTIAL:
            w = ctx.trade_info.projection_weight_new_team
            
            if new_team_stats.get("pts") is not None:
                adj.adjusted_pts = (new_team_stats["pts"] * w) + (adj.adjusted_pts * (1 - w))
            if new_team_stats.get("reb") is not None:
                adj.adjusted_reb = (new_team_stats["reb"] * w) + (adj.adjusted_reb * (1 - w))
            if new_team_stats.get("ast") is not None:
                adj.adjusted_ast = (new_team_stats["ast"] * w) + (adj.adjusted_ast * (1 - w))
            
            adj.adjustments_applied.append(
                f"Stats blended with {ctx.trade_info.games_with_new_team} "
                f"new-team games (weight: {w:.0%})"
            )
        
        # V19.3: Apply data-driven learning factor from post-trade performance
        for pt_key, attr_name in [("pts", "adjusted_pts"), ("reb", "adjusted_reb"), ("ast", "adjusted_ast")]:
            learning_factor = get_post_trade_learning_factor(
                conn, ctx.trade_info.player_id or 0, player_name,
                ctx.trade_info, pt_key, game_date,
            )
            if learning_factor is not None and learning_factor != 1.0:
                old_val = getattr(adj, attr_name)
                setattr(adj, attr_name, old_val * learning_factor)
                adj.adjustments_applied.append(
                    f"Post-trade learning ({pt_key}): {learning_factor:.2f}x "
                    f"({ctx.trade_info.games_with_new_team} games observed)"
                )
    
    # Track all adjustments
    if ctx.minutes_factor != 1.0:
        adj.adjustments_applied.append(f"Minutes factor: {ctx.minutes_factor:.2f}")
    if ctx.projection_factor != 1.0:
        adj.adjustments_applied.append(f"Projection factor: {ctx.projection_factor:.2f}")
    if ctx.confidence_factor != 1.0:
        adj.adjustments_applied.append(f"Confidence factor: {ctx.confidence_factor:.2f}")
    
    return adj


def get_trade_adjusted_stats_for_player(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    game_date: str,
    season_avg_pts: float,
    season_avg_reb: float,
    season_avg_ast: float,
    season_avg_min: float,
    deadline_date: str = TRADE_DEADLINE_DATE,
) -> TradeAdjustedStats:
    """
    Get comprehensive trade-adjusted stats for a player.
    
    Returns a TradeAdjustedStats object with blended projections
    and warnings. This is useful for the matchup advisor and 
    pick generation.
    """
    result = TradeAdjustedStats(player_name=player_name)
    
    # Get trade context
    ctx = get_trade_context(
        conn, player_id, player_name, team_abbrev, game_date, deadline_date
    )
    
    if not ctx.has_any_impact:
        result.adjusted_pts = season_avg_pts
        result.adjusted_reb = season_avg_reb
        result.adjusted_ast = season_avg_ast
        result.adjusted_minutes = season_avg_min
        return result
    
    result.was_traded = ctx.player_was_traded
    result.trade_info = ctx.trade_info
    result.team_status = ctx.team_status
    result.confidence_factor = ctx.confidence_factor
    result.warnings = ctx.warnings
    
    # Get new team data if traded
    if ctx.player_was_traded and ctx.trade_info:
        new_stats = _get_new_team_avg_stats(conn, ctx.trade_info, game_date)
        if new_stats:
            result.new_team_games = ctx.trade_info.games_with_new_team
            result.new_team_avg_pts = new_stats.get("pts", 0)
            result.new_team_avg_reb = new_stats.get("reb", 0)
            result.new_team_avg_ast = new_stats.get("ast", 0)
            result.new_team_avg_minutes = new_stats.get("minutes", 0)
        
        if ctx.trade_info.games_with_new_team == 0:
            result.data_quality = "no_data"
        elif ctx.trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_PARTIAL:
            result.data_quality = "poor"
        elif ctx.trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_RELIABLE:
            result.data_quality = "limited"
    
    # Apply adjustments
    adj = apply_trade_adjustments(
        conn, player_id, player_name, team_abbrev, game_date,
        season_avg_pts, season_avg_reb, season_avg_ast, season_avg_min,
        deadline_date,
    )
    
    result.adjusted_pts = adj.adjusted_pts
    result.adjusted_reb = adj.adjusted_reb
    result.adjusted_ast = adj.adjusted_ast
    result.adjusted_minutes = adj.adjusted_minutes
    
    return result


# ============================================================================
# Model Integration Helpers
# ============================================================================

def should_skip_player(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    game_date: str,
    deadline_date: str = TRADE_DEADLINE_DATE,
) -> Tuple[bool, str]:
    """
    Check if a player should be SKIPPED entirely due to trade effects.
    
    Returns (should_skip, reason)
    
    Skip criteria:
    1. Traded with 0 new-team games (no data at all)
    2. On a team that just had 5+ roster moves (too chaotic)
    """
    try:
        conn.execute("SELECT 1 FROM player_trades LIMIT 1")
    except sqlite3.OperationalError:
        return False, ""  # No trade tables
    
    trade_info = get_player_trade_info_by_id(conn, player_id, game_date)
    if not trade_info:
        trade_info = get_player_trade_info(conn, player_name, game_date)
    
    if trade_info and trade_info.games_with_new_team == 0:
        return True, (
            f"SKIP: {player_name} was traded from {trade_info.from_team} → "
            f"{trade_info.to_team} on {trade_info.trade_date} with "
            f"no games on new team yet."
        )
    
    return False, ""


def get_trade_factor_for_under(
    ctx: TradeContext,
    prop_type: str,
) -> Tuple[float, int, List[str]]:
    """
    Calculate UNDER-specific trade factor score.
    
    Returns (factor_score, factor_count, reasons)
    
    UNDER becomes MORE attractive when:
    - Player just traded (adjustment period)
    - New star teammate arrived (less usage)
    - Team is tanking (reduced minutes)
    """
    score = 0.0
    count = 0
    reasons = []
    
    if ctx.player_was_traded and ctx.trade_info:
        if ctx.trade_info.games_with_new_team < MIN_GAMES_NEW_TEAM_RELIABLE:
            score += TRADE_FACTOR_WEIGHTS["recently_traded"]
            count += 1
            reasons.append(
                f"Recently traded ({ctx.trade_info.games_with_new_team} new-team games)"
            )
    
    if ctx.arrived_players:
        arrived_star_names = [p for p in ctx.arrived_players 
                             if "star" in str(ctx.active_factors)]
        if arrived_star_names:
            score += TRADE_FACTOR_WEIGHTS["arrived_star_reduction"]
            count += 1
            reasons.append(f"New star(s) arrived: {', '.join(ctx.arrived_players[:2])}")
    
    if ctx.tank_result and ctx.tank_result.is_tanking:
        score += TRADE_FACTOR_WEIGHTS["tanking_team"]
        count += 1
        reasons.append(f"Team tanking ({ctx.tank_result.confidence:.0%} confidence)")
    
    return score, count, reasons


def get_trade_factor_for_over(
    ctx: TradeContext,
    prop_type: str,
) -> Tuple[float, int, List[str]]:
    """
    Calculate OVER-specific trade factor score.
    
    Returns (factor_score, factor_count, reasons)
    
    OVER becomes MORE attractive when:
    - A star teammate departed (more usage)
    """
    score = 0.0
    count = 0
    reasons = []
    
    if ctx.departed_stars:
        score += TRADE_FACTOR_WEIGHTS["departed_star_boost"]
        count += 1
        names = ", ".join(ctx.departed_stars[:2])
        reasons.append(f"Star teammate(s) departed: {names}")
    elif ctx.departed_starters:
        score += TRADE_FACTOR_WEIGHTS["departed_star_boost"] * 0.5
        count += 1
        names = ", ".join(ctx.departed_starters[:2])
        reasons.append(f"Starter(s) departed: {names}")
    
    # Reduce OVER score for tanking (minutes limited)
    if ctx.tank_result and ctx.tank_result.is_tanking:
        score *= 0.5  # Halve the OVER score for tanking teams
        reasons.append(f"⚠️ Tank risk reduces OVER confidence")
    
    return score, count, reasons


def get_opponent_tank_boost(
    conn: sqlite3.Connection,
    opponent_abbrev: str,
    game_date: str,
    deadline_date: str = TRADE_DEADLINE_DATE,
) -> Tuple[float, int, List[str]]:
    """
    V19.3: Calculate OVER boost when playing against a tanking opponent.
    
    When the opponent is tanking:
    - They play weaker lineups (benching stars, playing G-League players)
    - Their defensive effort is reduced
    - More scoring opportunities for the other team's players
    
    Returns (factor_score, factor_count, reasons)
    
    Weight: 12-15 depending on tank confidence (comparable to blowout_risk)
    """
    score = 0.0
    count = 0
    reasons = []
    
    if game_date < deadline_date:
        return score, count, reasons
    
    try:
        tank_result = detect_tanking_cached(conn, opponent_abbrev, deadline_date, game_date)
        
        if tank_result.is_tanking:
            # Scale weight by tank confidence: 0.30→8, 0.50→12, 0.75→15, 1.0→18
            base_weight = 8.0 + (tank_result.confidence * 13.0)
            weight = min(18.0, base_weight)
            
            score += weight
            count += 1
            reasons.append(
                f"📈 Opponent {opponent_abbrev} is tanking "
                f"({tank_result.confidence:.0%} confidence) — weaker defense/lineups"
            )
            
            # Extra boost if opponent has specific star minutes reductions
            affected_stars = [p for p in tank_result.player_analyses 
                             if p.is_star and p.has_significant_drop]
            if affected_stars:
                bonus = min(5.0, len(affected_stars) * 2.0)
                score += bonus
                names = ", ".join(p.player_name for p in affected_stars[:2])
                reasons.append(
                    f"📈 Opponent stars limited: {names} ({len(affected_stars)} star(s) affected)"
                )
    except Exception:
        pass  # Don't let tank detection errors block picks
    
    return score, count, reasons


def get_post_trade_learning_factor(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    trade_info: TradeInfo,
    prop_type: str,
    game_date: str,
) -> Optional[float]:
    """
    V19.3: Data-driven efficiency multiplier from post_trade_performance table.
    
    Instead of relying solely on the static formula-based confidence ramp-up
    (0→30 games curve in TradeInfo.confidence_discount), this reads actual
    post-trade performance data and computes how the player is performing
    relative to their pre-trade baseline.
    
    Returns:
        Multiplier (0.5-1.5) or None if insufficient data.
        - < 1.0: player underperforming on new team
        - > 1.0: player overperforming on new team
        - None: not enough data to compute
    
    Minimum 3 post-trade games required for a meaningful factor.
    """
    MIN_LEARNING_GAMES = 3
    
    if not trade_info.player_id or trade_info.games_with_new_team < MIN_LEARNING_GAMES:
        return None
    
    try:
        # Check if post_trade_performance table has data
        post_stats = conn.execute(
            """
            SELECT 
                AVG(pts) as avg_pts, AVG(reb) as avg_reb, AVG(ast) as avg_ast,
                AVG(minutes) as avg_min, COUNT(*) as games
            FROM post_trade_performance
            WHERE player_id = ?
              AND game_date < ?
              AND game_date > ?
            """,
            (trade_info.player_id, game_date, trade_info.trade_date),
        ).fetchone()
        
        if not post_stats or not post_stats["games"] or post_stats["games"] < MIN_LEARNING_GAMES:
            # Fall back to boxscore data on new team
            to_team_name = team_name_from_abbrev(trade_info.to_team)
            if not to_team_name:
                return None
            team_row = conn.execute(
                "SELECT id FROM teams WHERE name = ?", (to_team_name,)
            ).fetchone()
            if not team_row:
                return None
            
            post_stats = conn.execute(
                """
                SELECT 
                    AVG(b.pts) as avg_pts, AVG(b.reb) as avg_reb,
                    AVG(b.ast) as avg_ast, AVG(b.minutes) as avg_min,
                    COUNT(*) as games
                FROM boxscore_player b
                JOIN games g ON g.id = b.game_id
                WHERE b.player_id = ?
                  AND b.team_id = ?
                  AND g.game_date > ?
                  AND g.game_date < ?
                  AND b.minutes > 5
                """,
                (trade_info.player_id, team_row["id"],
                 trade_info.trade_date, game_date),
            ).fetchone()
            
            if not post_stats or not post_stats["games"] or post_stats["games"] < MIN_LEARNING_GAMES:
                return None
        
        # Get pre-trade baseline stats
        from_team_name = team_name_from_abbrev(trade_info.from_team)
        if not from_team_name:
            return None
        from_team_row = conn.execute(
            "SELECT id FROM teams WHERE name = ?", (from_team_name,)
        ).fetchone()
        if not from_team_row:
            return None
        
        pre_stats = conn.execute(
            """
            SELECT 
                AVG(b.pts) as avg_pts, AVG(b.reb) as avg_reb,
                AVG(b.ast) as avg_ast, AVG(b.minutes) as avg_min
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.player_id = ?
              AND b.team_id = ?
              AND g.game_date < ?
              AND g.game_date >= date(?, '-45 days')
              AND b.minutes > 5
            """,
            (trade_info.player_id, from_team_row["id"],
             trade_info.trade_date, trade_info.trade_date),
        ).fetchone()
        
        if not pre_stats or not pre_stats["avg_pts"]:
            return None
        
        # Calculate ratio for the requested prop type
        pt = prop_type.lower()
        stat_map = {"pts": "avg_pts", "reb": "avg_reb", "ast": "avg_ast"}
        stat_key = stat_map.get(pt, "avg_pts")
        
        pre_val = pre_stats[stat_key] or 0
        post_val = post_stats[stat_key] or 0
        
        if pre_val <= 0:
            return None
        
        raw_ratio = post_val / pre_val
        
        # Clamp to reasonable range and apply dampening
        # Don't let small samples create extreme multipliers
        games = post_stats["games"]
        dampening = min(1.0, games / 10.0)  # Full weight at 10+ games
        
        # Blend toward 1.0 based on sample size
        factor = 1.0 + (raw_ratio - 1.0) * dampening
        
        return max(0.5, min(1.5, factor))
        
    except (sqlite3.OperationalError, Exception):
        return None


# ============================================================================
# Internal Helpers
# ============================================================================

def _update_games_count_for_player(
    conn: sqlite3.Connection,
    trade_info: TradeInfo,
    game_date: str,
) -> None:
    """Update the games_with_new_team count for a specific trade."""
    if not trade_info.player_id:
        return
    
    to_team = trade_info.to_team
    full_name = team_name_from_abbrev(to_team)
    if not full_name:
        return
    
    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    if not team_row:
        return
    
    count = conn.execute(
        """
        SELECT COUNT(*) as cnt FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND b.team_id = ?
          AND g.game_date > ?
          AND g.game_date < ?
          AND b.minutes > 0
        """,
        (trade_info.player_id, team_row["id"], 
         trade_info.trade_date, game_date),
    ).fetchone()["cnt"]
    
    if count != trade_info.games_with_new_team:
        try:
            conn.execute(
                """UPDATE player_trades SET games_with_new_team = ? 
                   WHERE player_id = ? AND trade_date = ?""",
                (count, trade_info.player_id, trade_info.trade_date),
            )
            conn.commit()
            trade_info.games_with_new_team = count
        except Exception:
            pass


def _get_new_team_avg_stats(
    conn: sqlite3.Connection,
    trade_info: TradeInfo,
    before_date: str,
) -> Optional[Dict[str, float]]:
    """Get average stats for a player on their new team."""
    if not trade_info.player_id:
        return None
    
    full_name = team_name_from_abbrev(trade_info.to_team)
    if not full_name:
        return None
    
    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    if not team_row:
        return None
    
    row = conn.execute(
        """
        SELECT 
            AVG(b.pts) as pts, AVG(b.reb) as reb, AVG(b.ast) as ast,
            AVG(b.minutes) as minutes, COUNT(*) as games,
            AVG(b.fgm) as fgm, AVG(b.fga) as fga
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND b.team_id = ?
          AND g.game_date > ?
          AND g.game_date < ?
          AND b.minutes > 5
        """,
        (trade_info.player_id, team_row["id"], 
         trade_info.trade_date, before_date),
    ).fetchone()
    
    if not row or not row["games"]:
        return None
    
    return {
        "pts": row["pts"],
        "reb": row["reb"],
        "ast": row["ast"],
        "minutes": row["minutes"],
        "games": row["games"],
        "fgm": row["fgm"],
        "fga": row["fga"],
    }


def _get_new_team_avg_minutes(
    conn: sqlite3.Connection,
    trade_info: TradeInfo,
    before_date: str,
) -> Optional[float]:
    """Get average minutes for a player on their new team."""
    stats = _get_new_team_avg_stats(conn, trade_info, before_date)
    if stats:
        return stats.get("minutes")
    return None


def _is_star_player(
    conn: sqlite3.Connection,
    player_name: str,
) -> bool:
    """Check if a player is a star (high minutes/scoring)."""
    # Check archetypes table first
    row = conn.execute(
        """SELECT tier, bet_status FROM player_archetypes 
           WHERE LOWER(player_name) = LOWER(?)""",
        (player_name,),
    ).fetchone()
    
    if row:
        return row["tier"] <= 2 or row["bet_status"] == 2
    
    # Fall back to checking recent stats
    player_row = conn.execute(
        "SELECT id FROM players WHERE LOWER(name) = LOWER(?)",
        (player_name,),
    ).fetchone()
    
    if not player_row:
        return False
    
    stats = conn.execute(
        """
        SELECT AVG(b.minutes) as avg_min, AVG(b.pts) as avg_pts
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND b.minutes > 5
        ORDER BY g.game_date DESC
        LIMIT 15
        """,
        (player_row["id"],),
    ).fetchone()
    
    if stats and stats["avg_min"]:
        return stats["avg_min"] >= 28 or (stats["avg_pts"] and stats["avg_pts"] >= 18)
    
    return False


# ============================================================================
# Reporting
# ============================================================================

def generate_trade_deadline_report(
    conn: sqlite3.Connection,
    deadline_date: str = TRADE_DEADLINE_DATE,
    as_of_date: Optional[str] = None,
) -> str:
    """
    Generate a comprehensive trade deadline impact report.
    
    Shows all trades, tanking teams, and projected impacts.
    """
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    
    lines = []
    lines.append("=" * 70)
    lines.append("NBA TRADE DEADLINE IMPACT REPORT")
    lines.append(f"Deadline: {deadline_date} | As of: {as_of_date}")
    lines.append("=" * 70)
    
    # =========================================================================
    # Section 1: All Trades
    # =========================================================================
    try:
        from .trade_tracker import get_all_traded_players
        trades = get_all_traded_players(conn, since_date=deadline_date)
        
        if trades:
            lines.append(f"\n📋 TRADES ({len(trades)} total)")
            lines.append("-" * 40)
            
            for t in sorted(trades, key=lambda x: x.trade_date):
                games_str = f"{t.games_with_new_team} games" if t.games_with_new_team else "no games yet"
                role_str = f"[{t.old_team_role}]" if t.old_team_role else ""
                conf = f"({t.confidence_discount:.0%} confidence)" 
                lines.append(
                    f"  {t.player_name} {role_str}: {t.from_team} → {t.to_team} "
                    f"({t.trade_date}) — {games_str} {conf}"
                )
        else:
            lines.append("\n📋 No trades recorded yet.")
    except sqlite3.OperationalError:
        lines.append("\n📋 Trade tables not initialized. Run 'record-trade' first.")
    
    # =========================================================================
    # Section 2: Tanking Teams
    # =========================================================================
    lines.append(f"\n🏳️  TANK DETECTION")
    lines.append("-" * 40)
    
    tank_results = detect_all_tanking_teams(conn, deadline_date, as_of_date)
    
    if tank_results:
        for tr in tank_results:
            lines.append(tr.summary())
    else:
        lines.append("  No tanking signals detected.")
    
    # =========================================================================
    # Section 3: Affected Teams Summary
    # =========================================================================
    lines.append(f"\n📊 TEAM ROSTER STATUS")
    lines.append("-" * 40)
    
    try:
        statuses = conn.execute(
            "SELECT * FROM team_roster_status ORDER BY roster_stability_score ASC"
        ).fetchall()
        
        if statuses:
            for s in statuses:
                stability = s["roster_stability_score"]
                bar = "█" * int(stability * 10) + "░" * (10 - int(stability * 10))
                tank = " 🏳️ TANKING" if s["is_tanking"] else ""
                lines.append(
                    f"  {s['team_abbrev']}: [{bar}] stability: {stability:.2f}{tank}"
                )
        else:
            lines.append("  No team roster status data. Run 'record-team-status' first.")
    except sqlite3.OperationalError:
        lines.append("  Team roster status table not initialized.")
    
    # =========================================================================
    # Section 4: Betting Guidance
    # =========================================================================
    lines.append(f"\n💡 BETTING GUIDANCE")
    lines.append("-" * 40)
    lines.append("  • AVOID: Players with 0 games on new team (no data)")
    lines.append("  • CAUTION: Players with <5 games on new team (limited data)")
    lines.append("  • LEAN UNDER: Recently traded players (new system/chemistry)")
    lines.append("  • LEAN UNDER: Stars on tanking teams (minutes limited)")
    lines.append("  • LEAN OVER: Remaining stars after teammate traded away (more usage)")
    lines.append("  • SKIP ASSISTS: Post-trade AST props are extremely volatile")
    lines.append("")
    
    from .tank_detector import detect_all_tanking_teams
    
    return "\n".join(lines)
