"""
Model V18 General - Holistic Multi-Factor NBA Props Prediction Model
====================================================================

This is the GENERAL model of a dual-model approach:
- Model V18 General (this file): Holistic multi-factor approach for all picks
- Model V18 Under (separate file): Specialized UNDER model (Phase 2)

=============================================================================
MODEL V18 KEY INNOVATIONS
=============================================================================

1. **COMPREHENSIVE BOX SCORE ANALYSIS**:
   - Analyzes Plus/Minus (+/-) as consistency indicator
   - Tracks Shooting Efficiency (FG%, TS%) trends
   - Monitors minutes trends for role changes
   - This addresses: "analyzing box scores and prior game data"

2. **HOLISTIC MULTI-FACTOR SCORING** (Not Just Cold Bounce):
   - Combines 15+ factors with validated weights
   - Considers: defense, fatigue, efficiency, matchup history, usage
   - Minimum combined factor score required
   - This addresses: "do not just suggest based on cold bounces"

3. **HYBRID LINE APPROACH**:
   - Use sportsbook lines when available
   - ALWAYS generate picks (use derived with stricter edge when no sportsbook)
   - Track line source for honest reporting

4. **STRATEGIC DIRECTION SELECTION**:
   - PTS: UNDER preferred (63.9% vs 48.3% OVER from RCM)
   - PTS OVER: Only with cold bounce + NOT elite defense
   - REB: Both directions (~59% each)
   - AST: EXCLUDED (~54% is coin flip)

5. **THOROUGH BACKTESTING**:
   - Progress bar in terminal
   - Track by: line source, tier, direction, prop type, factor
   - Honest reporting with sportsbook vs derived separation

=============================================================================

USAGE:
------
    from src.nba_props.engine.model_v18_general import (
        get_daily_picks_v18_general,
        run_backtest_v18_general,
        ModelConfigV18General,
    )
    
    # Get picks for today
    picks = get_daily_picks_v18_general("2026-02-03")
    print(picks.summary())
    
    # Run backtest with progress bar
    result = run_backtest_v18_general(
        "2025-10-22", "2026-02-02", 
        verbose=True, 
        show_progress=True
    )
    print(result.summary())

Author: PropAI Team - Model V18
Created: February 2026
Version: 18.0
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths

from .model_v18_shared import (
    # Data classes
    LineInfo,
    PlayerStatsV18,
    DefenseContextV18,
    BackToBackInfo,
    InjuryImpact,
    HolisticFactorScore,
    HistoricalMatchup,
    EfficiencyStats,
    
    # Functions
    normalize_name,
    map_position,
    get_injured_players,
    get_injured_players_for_team,
    get_line,
    get_sportsbook_line,
    get_derived_line,
    get_defense_context,
    get_back_to_back_status,
    get_historical_matchup,
    load_player_stats,
    get_games_for_date,
    get_players_in_game,
    calculate_usage_boost,
    calculate_edge,
    detect_cold_bounce_pattern,
    detect_cold_streak_pattern,
    calculate_holistic_factor_score_under,
    calculate_holistic_factor_score_over,
    grade_pick,
    get_actual_stats,
    print_progress_bar,
    
    # Constants
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
    MIN_FACTOR_SCORE_PREMIUM,
    MIN_FACTOR_SCORE_HIGH,
    MIN_FACTOR_SCORE_STANDARD,
    FACTOR_PROJECTION_ADJUSTMENTS,
    MODEL_VERSION,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV18General:
    """
    Model V18 General Configuration.
    
    This model uses HOLISTIC multi-factor analysis with comprehensive
    box score data including +/-, efficiency, and historical matchups.
    
    KEY FEATURES:
    - Analyzes full box score data (not just PTS/REB/AST)
    - Holistic factor scoring (not single-pattern triggers)
    - Hybrid line handling (sportsbook preferred, derived allowed)
    - Different edge requirements by line source
    """
    # === VERSION INFO ===
    model_name: str = "Model V18 General"
    model_version: str = MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False  # ALWAYS generate picks
    derived_line_adjustment: float = 1.05  # +5% adjustment for derived lines
    sportsbook_confidence_boost: float = 10.0  # Higher confidence with real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Established players only (from Idea.txt)
    max_games_lookback: int = 20
    
    # === PROJECTION WEIGHTS ===
    weight_l3: float = 0.10
    weight_l5: float = 0.20
    weight_l10: float = 0.30
    weight_l15: float = 0.20
    weight_season: float = 0.20
    
    # === FACTOR SCORE THRESHOLDS ===
    min_factor_score_premium: float = MIN_FACTOR_SCORE_PREMIUM  # 60
    min_factor_score_high: float = MIN_FACTOR_SCORE_HIGH  # 45
    min_factor_score_standard: float = MIN_FACTOR_SCORE_STANDARD  # 35
    
    # === EDGE REQUIREMENTS ===
    # KEY: Different thresholds for sportsbook vs derived
    min_edge_sportsbook: float = 6.0   # Lower for real lines
    min_edge_derived: float = 12.0     # MUCH stricter for derived
    min_edge_premium: float = 15.0     # Premium tier needs high edge
    min_edge_over: float = 15.0        # OVERs need higher edge (riskier)
    
    # === STRATEGIC DIRECTION (From Backtest Analysis) ===
    # PTS: UNDER preferred (63.9% vs 48.3% OVER)
    pts_over_allowed: bool = True
    pts_over_min_factor_score: float = 50.0  # High bar for PTS OVER
    pts_over_require_cold_bounce: bool = True  # Must have cold bounce pattern
    pts_over_block_elite_defense: bool = True  # Block PTS OVER vs elite defense
    
    # REB: Both directions allowed
    reb_over_allowed: bool = True
    reb_under_allowed: bool = True
    reb_over_min_factor_score: float = 45.0
    reb_under_min_factor_score: float = 40.0
    
    # AST: EXCLUDED by default
    include_ast: bool = False
    min_ast_avg: float = 8.5  # Must average 8.5+ if enabled
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 6
    max_picks_per_day: int = 40
    max_picks_per_player: int = 1  # Focus on best prop per player
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence: float = 85.0
    high_confidence: float = 75.0
    standard_confidence: float = 68.0
    
    def get_weights(self) -> Dict[str, float]:
        return {
            'l3': self.weight_l3,
            'l5': self.weight_l5,
            'l10': self.weight_l10,
            'l15': self.weight_l15,
            'season': self.weight_season,
        }
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PropPickV18General:
    """A pick generated by Model V18 General."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details
    prop_type: str  # PTS, REB
    direction: str  # OVER, UNDER
    
    # Line information (KEY: tracks sportsbook vs derived)
    line: float
    line_source: str  # "sportsbook" or "derived"
    
    # Projection
    projection: float
    projection_adj: float  # After factor adjustments
    
    # Edge calculation
    edge_pct: float
    
    # Holistic factor scoring (V18 Core)
    factor_score: float
    primary_factor: str
    
    # Confidence
    confidence_score: float
    confidence_tier: str  # PREMIUM, HIGH, STANDARD
    
    # Context
    defense_rank: int
    defense_rating: str
    
    # Supporting data
    l5_avg: float
    l10_avg: float
    l15_avg: float
    season_avg: float
    
    # Fields with defaults MUST come after non-default fields
    active_factors: List[str] = field(default_factory=list)
    
    # V18 NEW: Efficiency metrics
    l5_plus_minus: float = 0.0
    l5_fg_pct: float = 0.0
    
    # Fields with defaults
    book: Optional[str] = None
    is_b2b: bool = False
    historical_games: int = 0
    historical_avg: float = 0.0
    
    # Reasons for the pick
    reasons: List[str] = field(default_factory=list)
    
    # Results (filled after game for backtesting)
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    margin: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "player": self.player_name,
            "team": self.team_abbrev,
            "opponent": self.opponent_abbrev,
            "date": self.game_date,
            "prop": self.prop_type.upper(),
            "direction": self.direction,
            "line": round(self.line, 1),
            "line_source": self.line_source,
            "book": self.book,
            "projection": round(self.projection_adj, 1),
            "edge": f"{self.edge_pct:.1f}%",
            "factor_score": round(self.factor_score, 1),
            "primary_factor": self.primary_factor,
            "active_factors": self.active_factors,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "b2b": self.is_b2b,
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "l15": round(self.l15_avg, 1),
            "season": round(self.season_avg, 1),
            "l5_plus_minus": round(self.l5_plus_minus, 1),
            "l5_fg_pct": round(self.l5_fg_pct * 100, 1) if self.l5_fg_pct else 0,
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV18General:
    """All picks for a day from Model V18 General."""
    date: str
    games: int
    config: ModelConfigV18General = field(default_factory=ModelConfigV18General)
    picks: List[PropPickV18General] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV18General]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV18General]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def over_picks(self) -> List[PropPickV18General]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV18General]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V18 GENERAL PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games} | Players analyzed: {self.players_analyzed}",
            f"Sportsbook lines available: {self.players_with_sportsbook_lines}",
            f"Using derived lines: {self.players_with_derived_lines}",
            "",
            f"Total picks: {self.total_picks}",
            f"  OVER: {len(self.over_picks)} | UNDER: {len(self.under_picks)}",
            f"  Sportsbook: {len(self.sportsbook_picks)} | Derived: {len(self.derived_picks)}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    emoji = "📈" if p.direction == "OVER" else "📉"
                    src = f"[{p.book}]" if p.line_source == "sportsbook" else "[derived]"
                    lines.append(
                        f"  {emoji} {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.direction} {p.line} {p.prop_type.upper()} {src}"
                    )
                    lines.append(
                        f"      Proj: {p.projection_adj:.1f} | Edge: {p.edge_pct:.1f}% | "
                        f"Factor: {p.factor_score:.0f} ({p.primary_factor})"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV18:
    """Results from a Model V18 backtest run."""
    start_date: str
    end_date: str
    config: ModelConfigV18General
    
    # Overall metrics
    total_picks: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    
    # By line source (KEY for honest reporting)
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    derived_picks: int = 0
    derived_hits: int = 0
    
    # By confidence tier
    premium_picks: int = 0
    premium_hits: int = 0
    high_picks: int = 0
    high_hits: int = 0
    standard_picks: int = 0
    standard_hits: int = 0
    
    # By direction
    over_picks: int = 0
    over_hits: int = 0
    under_picks: int = 0
    under_hits: int = 0
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    
    # By primary factor
    by_factor: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By factor score range
    by_score_range: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By edge range
    by_edge_range: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Individual picks (for detailed analysis)
    all_picks: List[PropPickV18General] = field(default_factory=list)
    
    # ROI calculation
    theoretical_profit: float = 0.0
    theoretical_wagers: float = 0.0
    
    @property
    def theoretical_roi(self) -> float:
        if self.theoretical_wagers == 0:
            return 0.0
        return (self.theoretical_profit / self.theoretical_wagers) * 100
    
    def _safe_rate(self, hits: int, total: int) -> str:
        if total == 0:
            return "N/A"
        return f"{hits/total*100:.1f}%"
    
    def summary(self) -> str:
        lines = [
            "",
            "=" * 80,
            f"MODEL V18 GENERAL BACKTEST RESULTS",
            f"Period: {self.start_date} to {self.end_date}",
            "=" * 80,
            "",
            f"OVERALL: {self.hit_rate:.1%} ({self.total_hits}/{self.total_picks})",
            "",
            "BY LINE SOURCE (Honest Reporting):",
            f"  Sportsbook: {self._safe_rate(self.sportsbook_hits, self.sportsbook_picks)} ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived:    {self._safe_rate(self.derived_hits, self.derived_picks)} ({self.derived_hits}/{self.derived_picks})",
            "",
            "BY CONFIDENCE TIER:",
            f"  PREMIUM:  {self._safe_rate(self.premium_hits, self.premium_picks)} ({self.premium_hits}/{self.premium_picks})",
            f"  HIGH:     {self._safe_rate(self.high_hits, self.high_picks)} ({self.high_hits}/{self.high_picks})",
            f"  STANDARD: {self._safe_rate(self.standard_hits, self.standard_picks)} ({self.standard_hits}/{self.standard_picks})",
            "",
            "BY DIRECTION:",
            f"  OVER:  {self._safe_rate(self.over_hits, self.over_picks)} ({self.over_hits}/{self.over_picks})",
            f"  UNDER: {self._safe_rate(self.under_hits, self.under_picks)} ({self.under_hits}/{self.under_picks})",
            "",
            "BY PROP TYPE:",
            f"  PTS: {self._safe_rate(self.pts_hits, self.pts_picks)} ({self.pts_hits}/{self.pts_picks})",
            f"  REB: {self._safe_rate(self.reb_hits, self.reb_picks)} ({self.reb_hits}/{self.reb_picks})",
            "",
        ]
        
        # Factor breakdown
        if self.by_factor:
            lines.append("BY PRIMARY FACTOR:")
            for factor, data in sorted(self.by_factor.items(), key=lambda x: x[1].get('total', 0), reverse=True):
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {factor}: {self._safe_rate(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # Score range breakdown
        if self.by_score_range:
            lines.append("BY FACTOR SCORE RANGE:")
            for range_name, data in sorted(self.by_score_range.items()):
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {range_name}: {self._safe_rate(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # Edge range breakdown
        if self.by_edge_range:
            lines.append("BY EDGE RANGE:")
            for range_name, data in sorted(self.by_edge_range.items()):
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {range_name}: {self._safe_rate(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # ROI
        lines.append(f"THEORETICAL ROI: {self.theoretical_roi:.1f}%")
        lines.append(f"(Assuming $100 wagers at -110 odds)")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def evaluate_player_for_prop(
    conn: sqlite3.Connection,
    stats: PlayerStatsV18,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: ModelConfigV18General,
) -> List[PropPickV18General]:
    """
    Evaluate a player for a specific prop type.
    
    Returns picks for both OVER and UNDER if they meet thresholds.
    The caller should select the best one.
    """
    picks = []
    pt = prop_type.lower()
    
    # Check minimum prop average
    prop_avg = stats.season.get(pt, 0)
    if prop_avg < MIN_PROP_AVERAGES.get(pt, 0):
        return []
    
    # Get context
    defense = get_defense_context(conn, opponent_abbrev, stats.position)
    b2b = get_back_to_back_status(conn, stats.team_abbrev, game_date)
    
    # Get injured teammates for usage redistribution
    injured = get_injured_players_for_team(conn, game_date, stats.team_abbrev)
    usage_boost, usage_reasons = calculate_usage_boost(injured)
    
    injury_impact = InjuryImpact(
        injured_teammates=injured,
        usage_boost_pct=usage_boost * 100,  # Convert to percentage
    )
    
    # Get line (sportsbook if available, derived otherwise)
    line_info = get_line(
        conn, stats.player_id, stats.player_name, pt, game_date, stats,
        config.derived_line_adjustment
    )
    
    # Get base projection
    projection = stats.get_projection(pt, config.get_weights())
    
    # =========================================================================
    # Evaluate UNDER
    # =========================================================================
    under_score = calculate_holistic_factor_score_under(
        stats, pt, defense, b2b, injury_impact
    )
    
    if under_score.total_score >= config.min_factor_score_standard:
        # Apply projection adjustment from factors
        proj_adj = projection * under_score.projection_adjustment
        
        # Calculate edge
        edge = calculate_edge(proj_adj, line_info.line, "UNDER")
        
        # Check edge requirements (different for sportsbook vs derived)
        min_edge = config.min_edge_sportsbook if line_info.is_sportsbook else config.min_edge_derived
        
        # Check prop-specific requirements
        if pt == 'reb' and under_score.total_score < config.reb_under_min_factor_score:
            pass  # Skip
        elif edge >= min_edge:
            # Determine confidence tier
            tier = under_score.get_tier()
            
            # Calculate confidence score
            confidence = 65.0
            confidence += min(under_score.total_score / 3, 15)  # Factor score bonus
            confidence += min(edge / 2, 10)  # Edge bonus
            if line_info.is_sportsbook:
                confidence += config.sportsbook_confidence_boost
            if defense.is_elite(pt):
                confidence += 5  # Elite defense bonus
            if b2b.is_second_of_b2b:
                confidence += 3  # B2B bonus
            
            confidence = min(confidence, 95)  # Cap at 95
            
            # Determine final tier based on confidence
            if confidence >= config.premium_confidence and tier == "PREMIUM":
                final_tier = "PREMIUM"
            elif confidence >= config.high_confidence and tier in ["PREMIUM", "HIGH"]:
                final_tier = "HIGH"
            else:
                final_tier = "STANDARD"
            
            pick = PropPickV18General(
                player_id=stats.player_id,
                player_name=stats.player_name,
                team_abbrev=stats.team_abbrev,
                opponent_abbrev=opponent_abbrev,
                game_date=game_date,
                prop_type=pt.upper(),
                direction="UNDER",
                line=line_info.line,
                line_source=line_info.source,
                book=line_info.book,
                projection=projection,
                projection_adj=proj_adj,
                edge_pct=edge,
                factor_score=under_score.total_score,
                primary_factor=under_score.primary_factor,
                active_factors=list(under_score.factors.keys()),
                confidence_score=confidence,
                confidence_tier=final_tier,
                defense_rank=defense.get_rank(pt),
                defense_rating=defense.get_rating(pt),
                l5_avg=stats.l5.get(pt, 0),
                l10_avg=stats.l10.get(pt, 0),
                l15_avg=stats.l15.get(pt, 0),
                season_avg=stats.season.get(pt, 0),
                l5_plus_minus=stats.efficiency.l5_plus_minus_avg,
                l5_fg_pct=stats.efficiency.l5_fg_pct,
                is_b2b=b2b.is_second_of_b2b,
                historical_games=stats.vs_opponent.games_played if stats.vs_opponent else 0,
                historical_avg=getattr(stats.vs_opponent, f"avg_{pt}", 0) if stats.vs_opponent else 0,
                reasons=under_score.reasons.copy(),
            )
            picks.append(pick)
    
    # =========================================================================
    # Evaluate OVER
    # =========================================================================
    over_score = calculate_holistic_factor_score_over(
        stats, pt, defense, b2b, injury_impact
    )
    
    if over_score.total_score >= config.min_factor_score_standard:
        # Check strategic direction requirements
        skip_over = False
        
        if pt == 'pts':
            # PTS OVER has high bar
            if over_score.total_score < config.pts_over_min_factor_score:
                skip_over = True
            if config.pts_over_require_cold_bounce and "cold_bounce" not in over_score.factors:
                skip_over = True
            if config.pts_over_block_elite_defense and defense.is_elite(pt):
                skip_over = True
        
        if pt == 'reb':
            if over_score.total_score < config.reb_over_min_factor_score:
                skip_over = True
        
        if not skip_over:
            # Apply projection adjustment
            proj_adj = projection * over_score.projection_adjustment
            
            # Calculate edge (OVERs need higher edge)
            edge = calculate_edge(proj_adj, line_info.line, "OVER")
            
            min_edge = max(
                config.min_edge_over,
                config.min_edge_sportsbook if line_info.is_sportsbook else config.min_edge_derived
            )
            
            if edge >= min_edge:
                tier = over_score.get_tier()
                
                # Calculate confidence
                confidence = 60.0  # Lower base for OVERs
                confidence += min(over_score.total_score / 3, 15)
                confidence += min(edge / 2, 10)
                if line_info.is_sportsbook:
                    confidence += config.sportsbook_confidence_boost
                if "cold_bounce" in over_score.factors:
                    confidence += 8  # Cold bounce is reliable
                
                confidence = min(confidence, 92)  # Cap slightly lower for OVERs
                
                if confidence >= config.premium_confidence and tier == "PREMIUM":
                    final_tier = "PREMIUM"
                elif confidence >= config.high_confidence and tier in ["PREMIUM", "HIGH"]:
                    final_tier = "HIGH"
                else:
                    final_tier = "STANDARD"
                
                pick = PropPickV18General(
                    player_id=stats.player_id,
                    player_name=stats.player_name,
                    team_abbrev=stats.team_abbrev,
                    opponent_abbrev=opponent_abbrev,
                    game_date=game_date,
                    prop_type=pt.upper(),
                    direction="OVER",
                    line=line_info.line,
                    line_source=line_info.source,
                    book=line_info.book,
                    projection=projection,
                    projection_adj=proj_adj,
                    edge_pct=edge,
                    factor_score=over_score.total_score,
                    primary_factor=over_score.primary_factor,
                    active_factors=list(over_score.factors.keys()),
                    confidence_score=confidence,
                    confidence_tier=final_tier,
                    defense_rank=defense.get_rank(pt),
                    defense_rating=defense.get_rating(pt),
                    l5_avg=stats.l5.get(pt, 0),
                    l10_avg=stats.l10.get(pt, 0),
                    l15_avg=stats.l15.get(pt, 0),
                    season_avg=stats.season.get(pt, 0),
                    l5_plus_minus=stats.efficiency.l5_plus_minus_avg,
                    l5_fg_pct=stats.efficiency.l5_fg_pct,
                    is_b2b=b2b.is_second_of_b2b,
                    historical_games=stats.vs_opponent.games_played if stats.vs_opponent else 0,
                    historical_avg=getattr(stats.vs_opponent, f"avg_{pt}", 0) if stats.vs_opponent else 0,
                    reasons=over_score.reasons.copy(),
                )
                picks.append(pick)
    
    return picks


def select_best_pick_for_player(
    picks: List[PropPickV18General],
    config: ModelConfigV18General,
) -> Optional[PropPickV18General]:
    """
    Select the best pick for a player from multiple candidates.
    
    Priority:
    1. Highest confidence tier
    2. Highest factor score
    3. Highest edge
    """
    if not picks:
        return None
    
    if len(picks) == 1:
        return picks[0]
    
    # Sort by tier priority, then factor score, then edge
    tier_order = {"PREMIUM": 3, "HIGH": 2, "STANDARD": 1}
    
    sorted_picks = sorted(
        picks,
        key=lambda p: (
            tier_order.get(p.confidence_tier, 0),
            p.factor_score,
            p.edge_pct,
        ),
        reverse=True
    )
    
    return sorted_picks[0]


# ============================================================================
# Main Entry Points
# ============================================================================

def get_daily_picks_v18_general(
    game_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV18General] = None,
    verbose: bool = False,
) -> DailyPicksV18General:
    """
    Generate picks for all games on a specific date.
    
    Args:
        game_date: Date in YYYY-MM-DD format
        db_path: Path to database (uses default if not provided)
        config: Model configuration (uses defaults if not provided)
        verbose: Print progress information
    
    Returns:
        DailyPicksV18General object with all picks
    """
    if config is None:
        config = ModelConfigV18General()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=Path(db_path))
    conn = db.connect()
    
    try:
        result = DailyPicksV18General(date=game_date, games=0, config=config)
        
        # Get games for the date
        games = get_games_for_date(conn, game_date)
        result.games = len(games)
        
        if verbose:
            print(f"Found {len(games)} games for {game_date}")
        
        # Get injured players
        injured_ids = get_injured_players(conn, game_date)
        
        all_candidates = []
        
        for game in games:
            team1_abbrev = abbrev_from_team_name(game["team1_name"]) or "UNK"
            team2_abbrev = abbrev_from_team_name(game["team2_name"]) or "UNK"
            
            # Process both teams
            for team_abbrev, opp_abbrev in [(team1_abbrev, team2_abbrev), (team2_abbrev, team1_abbrev)]:
                # Get players for this team
                player_ids = get_players_in_game(
                    conn, team_abbrev, game_date,
                    min_games=config.min_games_required,
                    min_avg_minutes=config.min_avg_minutes - 5  # Slightly lower to catch more
                )
                
                for player_id in player_ids:
                    if player_id in injured_ids:
                        continue
                    
                    # Load player stats
                    stats = load_player_stats(
                        conn, player_id, game_date,
                        opponent_abbrev=opp_abbrev,
                        min_games=config.min_games_required,
                        min_minutes=config.min_avg_minutes,
                        max_games=config.max_games_lookback,
                        min_game_minutes=config.min_minutes_filter,
                    )
                    
                    if not stats:
                        continue
                    
                    result.players_analyzed += 1
                    
                    # Track line availability
                    has_sportsbook = False
                    has_derived = False
                    
                    # Evaluate for each prop type
                    player_picks = []
                    
                    for pt in config.prop_types:
                        picks = evaluate_player_for_prop(
                            conn, stats, pt, opp_abbrev, game_date, config
                        )
                        player_picks.extend(picks)
                        
                        for p in picks:
                            if p.line_source == "sportsbook":
                                has_sportsbook = True
                            else:
                                has_derived = True
                    
                    if has_sportsbook:
                        result.players_with_sportsbook_lines += 1
                    if has_derived and not has_sportsbook:
                        result.players_with_derived_lines += 1
                    
                    # Select best pick per player
                    if player_picks:
                        best = select_best_pick_for_player(player_picks, config)
                        if best:
                            all_candidates.append(best)
        
        # Sort all candidates by confidence
        all_candidates.sort(
            key=lambda p: (
                {"PREMIUM": 3, "HIGH": 2, "STANDARD": 1}.get(p.confidence_tier, 0),
                p.factor_score,
                p.edge_pct,
            ),
            reverse=True
        )
        
        # Apply pick limits
        picks_per_game = {}
        final_picks = []
        
        for pick in all_candidates:
            if len(final_picks) >= config.max_picks_per_day:
                break
            
            game_key = f"{pick.team_abbrev}_vs_{pick.opponent_abbrev}"
            if picks_per_game.get(game_key, 0) >= config.max_picks_per_game:
                continue
            
            final_picks.append(pick)
            picks_per_game[game_key] = picks_per_game.get(game_key, 0) + 1
        
        result.picks = final_picks
        
        if verbose:
            print(f"Generated {len(final_picks)} picks")
        
        return result
        
    finally:
        conn.close()


def run_backtest_v18_general(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV18General] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> BacktestResultV18:
    """
    Run a backtest of Model V18 General over a date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database
        config: Model configuration
        verbose: Print detailed information
        show_progress: Show progress bar in terminal
    
    Returns:
        BacktestResultV18 with detailed metrics
    """
    if config is None:
        config = ModelConfigV18General()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=Path(db_path))
    conn = db.connect()
    
    try:
        result = BacktestResultV18(
            start_date=start_date,
            end_date=end_date,
            config=config,
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
            print(f"\nRunning backtest from {start_date} to {end_date} ({total_dates} days)")
            print(f"Config: {config.model_name} v{config.model_version}")
            print("")
        
        # Process each date
        for idx, game_date in enumerate(dates):
            if show_progress:
                print_progress_bar(
                    idx + 1, total_dates,
                    prefix=f"Backtesting",
                    suffix=f"{game_date}",
                    length=40
                )
            
            # Generate picks for this date
            daily = get_daily_picks_v18_general(
                game_date, db_path=db_path, config=config, verbose=False
            )
            
            # Grade each pick
            for pick in daily.picks:
                actual = get_actual_stats(conn, pick.player_id, game_date)
                
                if not actual:
                    continue  # Player didn't play
                
                pt = pick.prop_type.lower()
                actual_value = actual.get(pt, 0)
                
                hit, margin = grade_pick(actual_value, pick.line, pick.direction)
                
                pick.actual_value = actual_value
                pick.hit = hit
                pick.margin = margin
                
                # Update totals
                result.total_picks += 1
                if hit:
                    result.total_hits += 1
                
                # By line source
                if pick.line_source == "sportsbook":
                    result.sportsbook_picks += 1
                    if hit:
                        result.sportsbook_hits += 1
                else:
                    result.derived_picks += 1
                    if hit:
                        result.derived_hits += 1
                
                # By tier
                if pick.confidence_tier == "PREMIUM":
                    result.premium_picks += 1
                    if hit:
                        result.premium_hits += 1
                elif pick.confidence_tier == "HIGH":
                    result.high_picks += 1
                    if hit:
                        result.high_hits += 1
                else:
                    result.standard_picks += 1
                    if hit:
                        result.standard_hits += 1
                
                # By direction
                if pick.direction == "OVER":
                    result.over_picks += 1
                    if hit:
                        result.over_hits += 1
                else:
                    result.under_picks += 1
                    if hit:
                        result.under_hits += 1
                
                # By prop type
                if pt == "pts":
                    result.pts_picks += 1
                    if hit:
                        result.pts_hits += 1
                elif pt == "reb":
                    result.reb_picks += 1
                    if hit:
                        result.reb_hits += 1
                
                # By primary factor
                factor = pick.primary_factor
                if factor not in result.by_factor:
                    result.by_factor[factor] = {"total": 0, "hits": 0}
                result.by_factor[factor]["total"] += 1
                if hit:
                    result.by_factor[factor]["hits"] += 1
                
                # By factor score range
                score = pick.factor_score
                if score >= 70:
                    range_key = "70+"
                elif score >= 60:
                    range_key = "60-69"
                elif score >= 50:
                    range_key = "50-59"
                elif score >= 40:
                    range_key = "40-49"
                else:
                    range_key = "35-39"
                
                if range_key not in result.by_score_range:
                    result.by_score_range[range_key] = {"total": 0, "hits": 0}
                result.by_score_range[range_key]["total"] += 1
                if hit:
                    result.by_score_range[range_key]["hits"] += 1
                
                # By edge range
                edge = pick.edge_pct
                if edge >= 30:
                    edge_key = "30%+"
                elif edge >= 25:
                    edge_key = "25-29%"
                elif edge >= 20:
                    edge_key = "20-24%"
                elif edge >= 15:
                    edge_key = "15-19%"
                elif edge >= 10:
                    edge_key = "10-14%"
                else:
                    edge_key = "6-9%"
                
                if edge_key not in result.by_edge_range:
                    result.by_edge_range[edge_key] = {"total": 0, "hits": 0}
                result.by_edge_range[edge_key]["total"] += 1
                if hit:
                    result.by_edge_range[edge_key]["hits"] += 1
                
                # ROI calculation (assuming -110 odds)
                wager = 100.0
                result.theoretical_wagers += wager
                if hit:
                    result.theoretical_profit += wager * (100/110)  # Win at -110
                else:
                    result.theoretical_profit -= wager
                
                # Store pick
                result.all_picks.append(pick)
        
        # Calculate overall hit rate
        if result.total_picks > 0:
            result.hit_rate = result.total_hits / result.total_picks
        
        if show_progress:
            print()  # Newline after progress bar
        
        return result
        
    finally:
        conn.close()


# ============================================================================
# CLI Support
# ============================================================================

def main():
    """Command line interface for Model V18 General."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V18 General - NBA Props Prediction")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Daily picks command
    picks_parser = subparsers.add_parser("picks", help="Generate daily picks")
    picks_parser.add_argument("--date", required=True, help="Date (YYYY-MM-DD)")
    picks_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    backtest_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    args = parser.parse_args()
    
    if args.command == "picks":
        result = get_daily_picks_v18_general(args.date, verbose=args.verbose)
        print(result.summary())
    
    elif args.command == "backtest":
        result = run_backtest_v18_general(
            args.start, args.end,
            verbose=args.verbose,
            show_progress=not args.no_progress
        )
        print(result.summary())
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
