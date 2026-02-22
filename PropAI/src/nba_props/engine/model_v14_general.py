"""
Model V14 General - Market-Aware NBA Props Prediction Model (Over/Value Focus)
===============================================================================

This is the GENERAL model of a dual-model approach:
- Model V14 General: Focuses on OVER picks and REB in both directions
- Model V14 Under: Focuses EXCLUSIVELY on UNDER opportunities (separate file)

KEY INNOVATIONS IN V14:
-----------------------
1. HYBRID LINE HANDLING:
   - Use sportsbook lines when available (PREMIUM confidence)
   - Fall back to derived lines when sportsbook unavailable (LOWER confidence)
   - Track and report hit rates separately for honest validation
   
2. VALIDATED PATTERNS FROM PREVIOUS MODELS:
   - Cold Bounce: 66.9% (from Production model)
   - Hot Sustained: 65.9% (from Production model)
   - Usage Redistribution when teammates injured

3. STRATEGIC DIRECTION SELECTION:
   - PTS OVER: Highly selective (only with cold bounce + weak defense)
   - REB: Both directions (~59% each)
   - AST: Only for elite playmakers (8.5+ avg)

4. STRICT FILTERING:
   - 23+ minutes average (established players only)
   - 10+ games history
   - Pattern confirmation REQUIRED for all picks
   - No generic "edge" picks

THE DERIVED LINE AWARENESS:
---------------------------
This model addresses the "Derived Line Fallacy" by:
1. Tracking whether each pick used sportsbook or derived line
2. Applying a 5% upward adjustment to derived lines
3. Requiring HIGHER edge for derived line picks (10% vs 6%)
4. Reporting separate hit rates for honest validation

USAGE:
------
    from src.nba_props.engine.model_v14_general import (
        get_daily_picks_v14_general,
        run_backtest_v14_general,
        ModelConfigV14General,
    )
    
    # Get picks for today
    picks = get_daily_picks_v14_general("2026-02-03")
    
    # Run backtest
    result = run_backtest_v14_general("2025-12-01", "2026-02-02")

Author: NBA Props Team - Model V14
Created: February 2026
Version: 14.0
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths

from .model_v14_shared import (
    LineInfo,
    PlayerStatsV14,
    DefenseContextV14,
    BackToBackInfo,
    InjuryImpact,
    normalize_name,
    map_position,
    get_injured_players,
    get_injured_players_for_team,
    get_line,
    get_sportsbook_line,
    get_derived_line,
    get_defense_context,
    get_back_to_back_status,
    load_player_stats,
    get_games_for_date,
    get_players_in_game,
    calculate_usage_boost,
    calculate_edge,
    detect_cold_bounce_pattern,
    detect_hot_sustained_pattern,
    grade_pick,
    get_actual_stats,
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV14General:
    """
    Model V14 General Configuration.
    
    This model focuses on OVER picks and general value, with hybrid line handling.
    
    KEY DIFFERENCES FROM V13:
    - Stricter filtering (23 min avg vs 20)
    - Pattern-only picks (no generic edge)
    - Higher edge thresholds for derived lines
    - Better usage redistribution integration
    """
    # === VERSION INFO ===
    model_name: str = "Model V14 General"
    model_version: str = "14.0"
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False  # Allow derived lines
    derived_line_adjustment: float = 1.05  # +5% adjustment for derived lines
    sportsbook_confidence_boost: float = 12.0  # Higher confidence with real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Increased - established players only
    max_games_lookback: int = 20
    
    # === PROJECTION WEIGHTS ===
    # More recent-weighted for OVER picks (momentum matters)
    weight_l3: float = 0.10
    weight_l5: float = 0.25
    weight_l10: float = 0.30
    weight_l15: float = 0.20
    weight_season: float = 0.15
    
    # === PATTERN THRESHOLDS ===
    # Cold bounce - BEST OVER pattern (66.9% from Production)
    cold_deviation_threshold: float = -20.0  # L5 is 20%+ below L15
    bounce_threshold: float = 0.0  # Last game must be above L10
    
    # Hot sustained - Good OVER pattern (65.9% from Production)
    hot_deviation_threshold: float = 30.0  # L5 is 30%+ above L15
    sustained_games_above: int = 3  # 3+ of last 5 above baseline
    
    # === USAGE REDISTRIBUTION ===
    # When a star is out, remaining players get boost
    usage_boost_threshold: float = 15.0  # Teammate avg 15+ pts = significant
    usage_boost_per_player: float = 0.03  # 3% boost per star out
    max_usage_boost: float = 0.12  # Cap at 12%
    
    # === EDGE REQUIREMENTS ===
    min_edge_sportsbook: float = 6.0  # 6%+ edge vs sportsbook line
    min_edge_derived: float = 10.0  # 10%+ edge vs derived line (stricter)
    min_edge_premium: float = 12.0  # Premium needs 12%+ edge
    
    # === DEFENSE ADJUSTMENTS ===
    elite_defense_adj: float = 0.88  # -12% vs elite defense
    good_defense_adj: float = 0.94  # -6% vs good defense
    neutral_defense_adj: float = 1.00
    weak_defense_adj: float = 1.08  # +8% vs weak defense
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # PTS OVER: Only with cold bounce + weak/average defense
    pts_over_require_cold_bounce: bool = True
    pts_over_block_elite_defense: bool = True
    
    # REB: Both directions allowed
    reb_allow_under: bool = True  # General model can suggest REB UNDER too
    
    # AST: Only for elite playmakers
    include_ast: bool = True
    min_ast_avg: float = 8.5  # High bar
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 5
    max_picks_per_day: int = 20
    max_picks_per_player: int = 1  # Focus on best prop per player
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence: float = 85.0
    high_confidence: float = 75.0
    standard_confidence: float = 65.0
    
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
class PropPickV14General:
    """A pick generated by Model V14 General."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details
    prop_type: str  # PTS, REB, AST
    direction: str  # OVER, UNDER
    
    # Line information (KEY FIELD)
    line: float
    line_source: str  # "sportsbook" or "derived"
    
    # Projection
    projection: float
    projection_adj: float  # After defense/usage adjustments
    
    # Edge calculation
    edge_pct: float
    
    # Pattern and confidence
    pattern: str  # cold_bounce, hot_sustained, usage_boost, reb_under
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
    
    # === Fields with defaults ===
    book: Optional[str] = None
    usage_boost: float = 0.0
    is_b2b: bool = False
    
    # Reasons
    reasons: List[str] = field(default_factory=list)
    
    # Results (filled after game)
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
            "pattern": self.pattern,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "usage_boost": f"+{self.usage_boost*100:.1f}%" if self.usage_boost > 0 else None,
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "l15": round(self.l15_avg, 1),
            "season": round(self.season_avg, 1),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV14General:
    """All picks for a day from Model V14 General."""
    date: str
    games: int
    config: ModelConfigV14General = field(default_factory=ModelConfigV14General)
    picks: List[PropPickV14General] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV14General]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV14General]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def over_picks(self) -> List[PropPickV14General]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV14General]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V14 GENERAL PICKS - {self.date}",
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
                        f"{p.prop_type} {p.direction} {p.line:.1f} {src}"
                    )
                    lines.append(
                        f"      Proj: {p.projection_adj:.1f} | Edge: {p.edge_pct:.1f}% | "
                        f"Pattern: {p.pattern}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV14General:
    """Comprehensive backtest results for Model V14 General."""
    start_date: str
    end_date: str
    config: ModelConfigV14General
    
    # Overall
    total_picks: int = 0
    hits: int = 0
    
    # By line source (KEY METRIC)
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    derived_picks: int = 0
    derived_hits: int = 0
    
    # By tier
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
    ast_picks: int = 0
    ast_hits: int = 0
    
    # By pattern
    cold_bounce_picks: int = 0
    cold_bounce_hits: int = 0
    hot_sustained_picks: int = 0
    hot_sustained_hits: int = 0
    usage_boost_picks: int = 0
    usage_boost_hits: int = 0
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # All picks
    all_picks: List[PropPickV14General] = field(default_factory=list)
    daily_results: List[Dict] = field(default_factory=list)
    
    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_picks * 100 if self.total_picks > 0 else 0.0
    
    @property
    def sportsbook_rate(self) -> float:
        return self.sportsbook_hits / self.sportsbook_picks * 100 if self.sportsbook_picks > 0 else 0.0
    
    @property
    def derived_rate(self) -> float:
        return self.derived_hits / self.derived_picks * 100 if self.derived_picks > 0 else 0.0
    
    def summary(self) -> str:
        def pct(h, t):
            return f"{h/t*100:.1f}%" if t > 0 else "N/A"
        
        lines = [
            f"{'='*70}",
            f"MODEL V14 GENERAL - BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            "",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            "",
            f"BY LINE SOURCE (KEY METRIC):",
            f"  Sportsbook lines: {pct(self.sportsbook_hits, self.sportsbook_picks)} ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived lines:    {pct(self.derived_hits, self.derived_picks)} ({self.derived_hits}/{self.derived_picks})",
            "",
            f"BY TIER:",
            f"  PREMIUM:  {pct(self.premium_hits, self.premium_picks)} ({self.premium_hits}/{self.premium_picks})",
            f"  HIGH:     {pct(self.high_hits, self.high_picks)} ({self.high_hits}/{self.high_picks})",
            f"  STANDARD: {pct(self.standard_hits, self.standard_picks)} ({self.standard_hits}/{self.standard_picks})",
            "",
            f"BY DIRECTION:",
            f"  OVER:  {pct(self.over_hits, self.over_picks)} ({self.over_hits}/{self.over_picks})",
            f"  UNDER: {pct(self.under_hits, self.under_picks)} ({self.under_hits}/{self.under_picks})",
            "",
            f"BY PROP TYPE:",
            f"  PTS: {pct(self.pts_hits, self.pts_picks)} ({self.pts_hits}/{self.pts_picks})",
            f"  REB: {pct(self.reb_hits, self.reb_picks)} ({self.reb_hits}/{self.reb_picks})",
            f"  AST: {pct(self.ast_hits, self.ast_picks)} ({self.ast_hits}/{self.ast_picks})",
            "",
            f"BY PATTERN:",
            f"  Cold Bounce:   {pct(self.cold_bounce_hits, self.cold_bounce_picks)} ({self.cold_bounce_hits}/{self.cold_bounce_picks})",
            f"  Hot Sustained: {pct(self.hot_sustained_hits, self.hot_sustained_picks)} ({self.hot_sustained_hits}/{self.hot_sustained_picks})",
            f"  Usage Boost:   {pct(self.usage_boost_hits, self.usage_boost_picks)} ({self.usage_boost_hits}/{self.usage_boost_picks})",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ============================================================================
# Core Model Functions
# ============================================================================

def _apply_defense_adjustment(
    projection: float,
    defense_context: DefenseContextV14,
    prop_type: str,
    config: ModelConfigV14General,
) -> float:
    """Apply defensive adjustment to projection."""
    rating = defense_context.get_rating(prop_type)
    
    if rating == "elite":
        return projection * config.elite_defense_adj
    elif rating == "good":
        return projection * config.good_defense_adj
    elif rating == "weak":
        return projection * config.weak_defense_adj
    else:
        return projection * config.neutral_defense_adj


def _calculate_confidence(
    edge_pct: float,
    pattern: str,
    line_source: str,
    defense_rating: str,
    cv: float,
    usage_boost: float,
    config: ModelConfigV14General,
) -> float:
    """Calculate confidence score for a pick."""
    # Base confidence
    base = 65.0
    
    # Pattern bonus (validated from Production model)
    pattern_bonus = {
        "cold_bounce": 12.0,  # Best pattern
        "hot_sustained": 8.0,
        "usage_boost": 6.0,
        "reb_both": 4.0,
    }
    base += pattern_bonus.get(pattern, 0)
    
    # Edge bonus (capped)
    edge_bonus = min(edge_pct / 2, 10.0)
    base += edge_bonus
    
    # Sportsbook line bonus (IMPORTANT)
    if line_source == "sportsbook":
        base += config.sportsbook_confidence_boost
    
    # Defense context bonus
    if defense_rating == "weak" and pattern in ["cold_bounce", "hot_sustained"]:
        base += 4.0
    
    # Consistency bonus
    if cv < 0.20:
        base += 4.0
    elif cv > 0.40:
        base -= 4.0
    
    # Usage boost bonus
    if usage_boost > 0:
        base += min(usage_boost * 30, 5.0)
    
    return min(base, 100.0)


def _analyze_player_for_picks(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    game_date: str,
    injured_teammates: List[Dict[str, Any]],
    config: ModelConfigV14General,
) -> List[PropPickV14General]:
    """
    Analyze a player for potential picks.
    
    Returns list of picks that meet criteria.
    """
    picks = []
    
    # Load player stats
    stats = load_player_stats(
        conn, player_id, game_date,
        min_games=config.min_games_required,
        min_minutes=config.min_avg_minutes,
        max_games=config.max_games_lookback,
        min_game_minutes=config.min_minutes_filter,
    )
    
    if not stats:
        return []
    
    # Get defense context
    defense = get_defense_context(conn, opponent_abbrev, stats.position)
    
    # Get B2B status
    b2b = get_back_to_back_status(conn, stats.team_abbrev, game_date)
    
    # Calculate usage boost
    usage_boost = calculate_usage_boost(
        injured_teammates,
        boost_per_player=config.usage_boost_per_player,
        max_boost=config.max_usage_boost,
        min_pts_threshold=config.usage_boost_threshold,
    )
    
    # Analyze each prop type
    for prop_type in config.prop_types:
        pt = prop_type.lower()
        
        # Check minimum average
        season_avg = stats.season.get(pt, 0)
        min_avg = MIN_PROP_AVERAGES.get(pt, 0)
        
        # Special AST handling
        if pt == 'ast':
            if not config.include_ast or season_avg < config.min_ast_avg:
                continue
        elif season_avg < min_avg:
            continue
        
        # Get line (sportsbook or derived)
        line_info = get_line(
            conn, player_id, stats.player_name, prop_type, game_date,
            stats, config.derived_line_adjustment
        )
        
        # Calculate base projection
        projection = stats.get_projection(prop_type, config.get_weights())
        
        # Apply defense adjustment
        projection_adj = _apply_defense_adjustment(
            projection, defense, prop_type, config
        )
        
        # Apply usage boost
        if usage_boost > 0:
            projection_adj *= (1 + usage_boost)
        
        # === PATTERN DETECTION FOR OVER PICKS ===
        
        # Cold Bounce pattern
        is_cold_bounce, cold_reasons = detect_cold_bounce_pattern(
            stats, prop_type,
            cold_threshold=config.cold_deviation_threshold,
            bounce_threshold=config.bounce_threshold,
        )
        
        # Hot Sustained pattern
        is_hot_sustained, hot_reasons = detect_hot_sustained_pattern(
            stats, prop_type,
            hot_threshold=config.hot_deviation_threshold,
            sustained_games=config.sustained_games_above,
        )
        
        # === EVALUATE OVER PICKS ===
        
        # Determine if we should evaluate OVER for this prop
        should_eval_over = False
        
        if is_cold_bounce:
            # Cold bounce is the ONLY reliable OVER pattern (84.6% hit rate)
            if pt == 'pts':
                # PTS OVER: Only if NOT vs elite defense
                if not (config.pts_over_block_elite_defense and defense.is_elite(pt)):
                    should_eval_over = True
            else:
                # REB/AST cold bounce
                should_eval_over = True
        
        # HOT SUSTAINED DISABLED - backtest shows only 25.8% hit rate
        # if is_hot_sustained and not should_eval_over:
        #     # Hot sustained has shown poor performance
        #     if pt == 'reb' and not defense.is_elite(pt):
        #         should_eval_over = True
        
        if should_eval_over:
            edge = calculate_edge(projection_adj, line_info.line, "OVER")
            min_edge = (config.min_edge_sportsbook if line_info.is_sportsbook 
                       else config.min_edge_derived)
            
            if edge >= min_edge:
                pattern = "cold_bounce" if is_cold_bounce else "hot_sustained"
                reasons = cold_reasons if is_cold_bounce else hot_reasons
                
                # Add defense context to reasons
                reasons.append(f"Opponent defense: {defense.get_rating(pt)} (#{defense.get_rank(pt)})")
                
                if usage_boost > 0:
                    reasons.append(f"Usage boost: +{usage_boost*100:.1f}% from injured teammates")
                
                # Calculate confidence
                confidence = _calculate_confidence(
                    edge, pattern, line_info.source,
                    defense.get_rating(pt), stats.get_cv(pt),
                    usage_boost, config
                )
                
                # Determine tier
                if confidence >= config.premium_confidence and edge >= config.min_edge_premium:
                    tier = "PREMIUM"
                elif confidence >= config.high_confidence:
                    tier = "HIGH"
                elif confidence >= config.standard_confidence:
                    tier = "STANDARD"
                else:
                    continue  # Skip low confidence
                
                pick = PropPickV14General(
                    player_id=player_id,
                    player_name=stats.player_name,
                    team_abbrev=stats.team_abbrev,
                    opponent_abbrev=opponent_abbrev,
                    game_date=game_date,
                    prop_type=prop_type.upper(),
                    direction="OVER",
                    line=line_info.line,
                    line_source=line_info.source,
                    book=line_info.book,
                    projection=projection,
                    projection_adj=projection_adj,
                    edge_pct=edge,
                    pattern=pattern,
                    confidence_score=confidence,
                    confidence_tier=tier,
                    defense_rank=defense.get_rank(pt),
                    defense_rating=defense.get_rating(pt),
                    usage_boost=usage_boost,
                    is_b2b=b2b.is_second_of_b2b,
                    l5_avg=stats.l5.get(pt, 0),
                    l10_avg=stats.l10.get(pt, 0),
                    l15_avg=stats.l15.get(pt, 0),
                    season_avg=season_avg,
                    reasons=reasons,
                )
                picks.append(pick)
        
        # === EVALUATE REB UNDER PICKS (General model can suggest these) ===
        
        if pt == 'reb' and config.reb_allow_under:
            # REB UNDER with elite defense
            if defense.is_elite(pt) or defense.is_good(pt):
                edge = calculate_edge(projection_adj, line_info.line, "UNDER")
                min_edge = (config.min_edge_sportsbook if line_info.is_sportsbook 
                           else config.min_edge_derived)
                
                if edge >= min_edge:
                    reasons = [
                        f"REB UNDER: Strong defense at position (#{defense.get_rank(pt)})",
                        f"Projection ({projection_adj:.1f}) below line ({line_info.line})",
                    ]
                    
                    confidence = _calculate_confidence(
                        edge, "reb_both", line_info.source,
                        defense.get_rating(pt), stats.get_cv(pt),
                        0, config
                    )
                    
                    if confidence >= config.standard_confidence:
                        if confidence >= config.premium_confidence:
                            tier = "PREMIUM"
                        elif confidence >= config.high_confidence:
                            tier = "HIGH"
                        else:
                            tier = "STANDARD"
                        
                        pick = PropPickV14General(
                            player_id=player_id,
                            player_name=stats.player_name,
                            team_abbrev=stats.team_abbrev,
                            opponent_abbrev=opponent_abbrev,
                            game_date=game_date,
                            prop_type="REB",
                            direction="UNDER",
                            line=line_info.line,
                            line_source=line_info.source,
                            book=line_info.book,
                            projection=projection,
                            projection_adj=projection_adj,
                            edge_pct=edge,
                            pattern="reb_both",
                            confidence_score=confidence,
                            confidence_tier=tier,
                            defense_rank=defense.get_rank(pt),
                            defense_rating=defense.get_rating(pt),
                            usage_boost=0,
                            is_b2b=b2b.is_second_of_b2b,
                            l5_avg=stats.l5.get(pt, 0),
                            l10_avg=stats.l10.get(pt, 0),
                            l15_avg=stats.l15.get(pt, 0),
                            season_avg=season_avg,
                            reasons=reasons,
                        )
                        picks.append(pick)
        
        # === USAGE BOOST SPECIAL PATTERN ===
        
        if usage_boost >= 0.06 and pt in ['pts', 'reb']:  # 6%+ boost
            edge = calculate_edge(projection_adj, line_info.line, "OVER")
            min_edge = (config.min_edge_sportsbook if line_info.is_sportsbook 
                       else config.min_edge_derived) - 2  # Lower bar for usage boost
            
            if edge >= min_edge:
                # Check not already covered by other patterns
                existing = [p for p in picks if p.prop_type.lower() == pt and p.direction == "OVER"]
                if not existing:
                    reasons = [
                        f"Usage boost pattern: +{usage_boost*100:.1f}% projected increase",
                        f"Injured teammates: {', '.join(t['player_name'] for t in injured_teammates[:3])}",
                        f"Expected additional opportunities",
                    ]
                    
                    confidence = _calculate_confidence(
                        edge, "usage_boost", line_info.source,
                        defense.get_rating(pt), stats.get_cv(pt),
                        usage_boost, config
                    )
                    
                    if confidence >= config.standard_confidence:
                        if confidence >= config.premium_confidence:
                            tier = "PREMIUM"
                        elif confidence >= config.high_confidence:
                            tier = "HIGH"
                        else:
                            tier = "STANDARD"
                        
                        pick = PropPickV14General(
                            player_id=player_id,
                            player_name=stats.player_name,
                            team_abbrev=stats.team_abbrev,
                            opponent_abbrev=opponent_abbrev,
                            game_date=game_date,
                            prop_type=prop_type.upper(),
                            direction="OVER",
                            line=line_info.line,
                            line_source=line_info.source,
                            book=line_info.book,
                            projection=projection,
                            projection_adj=projection_adj,
                            edge_pct=edge,
                            pattern="usage_boost",
                            confidence_score=confidence,
                            confidence_tier=tier,
                            defense_rank=defense.get_rank(pt),
                            defense_rating=defense.get_rating(pt),
                            usage_boost=usage_boost,
                            is_b2b=b2b.is_second_of_b2b,
                            l5_avg=stats.l5.get(pt, 0),
                            l10_avg=stats.l10.get(pt, 0),
                            l15_avg=stats.l15.get(pt, 0),
                            season_avg=season_avg,
                            reasons=reasons,
                        )
                        picks.append(pick)
    
    return picks


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_v14_general(
    game_date: str,
    config: Optional[ModelConfigV14General] = None,
    db_path: Optional[Path] = None,
) -> DailyPicksV14General:
    """
    Generate picks for a specific date using Model V14 General.
    
    Args:
        game_date: Date string (YYYY-MM-DD)
        config: Optional model configuration
        db_path: Optional database path
    
    Returns:
        DailyPicksV14General with all picks for the date
    """
    if config is None:
        config = ModelConfigV14General()
    
    if db_path is None:
        db_path = get_paths().db_path
    
    db = Db(path=db_path)
    result = DailyPicksV14General(date=game_date, games=0, config=config)
    
    with db.connect() as conn:
        # Get games for date
        games = get_games_for_date(conn, game_date)
        result.games = len(games)
        
        if not games:
            return result
        
        # Get injured players
        injured_ids = get_injured_players(conn, game_date)
        
        all_picks = []
        player_pick_counts = {}
        
        for game in games:
            home_abbrev = abbrev_from_team_name(game["home_team"]) or ""
            away_abbrev = abbrev_from_team_name(game["away_team"]) or ""
            
            # Analyze away team players (vs home defense)
            injured_away = get_injured_players_for_team(conn, game_date, away_abbrev)
            away_players = get_players_in_game(conn, away_abbrev, game_date)
            
            for player_id in away_players:
                if player_id in injured_ids:
                    continue
                
                result.players_analyzed += 1
                
                picks = _analyze_player_for_picks(
                    conn, player_id, home_abbrev, game_date,
                    injured_away, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                
                all_picks.extend(picks)
            
            # Analyze home team players (vs away defense)
            injured_home = get_injured_players_for_team(conn, game_date, home_abbrev)
            home_players = get_players_in_game(conn, home_abbrev, game_date)
            
            for player_id in home_players:
                if player_id in injured_ids:
                    continue
                
                result.players_analyzed += 1
                
                picks = _analyze_player_for_picks(
                    conn, player_id, away_abbrev, game_date,
                    injured_home, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                
                all_picks.extend(picks)
        
        # Sort by confidence and apply limits
        all_picks.sort(key=lambda p: (-p.confidence_score, -p.edge_pct))
        
        final_picks = []
        for pick in all_picks:
            # Check player limit
            if player_pick_counts.get(pick.player_id, 0) >= config.max_picks_per_player:
                continue
            
            # Check daily limit
            if len(final_picks) >= config.max_picks_per_day:
                break
            
            final_picks.append(pick)
            player_pick_counts[pick.player_id] = player_pick_counts.get(pick.player_id, 0) + 1
        
        result.picks = final_picks
    
    return result


def run_backtest_v14_general(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV14General] = None,
    db_path: Optional[Path] = None,
    verbose: bool = False,
) -> BacktestResultV14General:
    """
    Run comprehensive backtest for Model V14 General.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Optional model configuration
        db_path: Optional database path
        verbose: Print progress
    
    Returns:
        BacktestResultV14General with comprehensive metrics
    """
    if config is None:
        config = ModelConfigV14General()
    
    if db_path is None:
        db_path = get_paths().db_path
    
    db = Db(path=db_path)
    result = BacktestResultV14General(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        if verbose:
            print(f"Processing {date_str}...")
        
        # Get picks for this date
        daily = get_daily_picks_v14_general(date_str, config, db_path)
        
        if daily.games == 0:
            current += timedelta(days=1)
            continue
        
        result.days_tested += 1
        result.total_games += daily.games
        
        daily_result = {
            "date": date_str,
            "games": daily.games,
            "picks": len(daily.picks),
            "hits": 0,
        }
        
        with db.connect() as conn:
            for pick in daily.picks:
                # Get actual result
                actual = get_actual_stats(conn, pick.player_id, date_str)
                
                if not actual:
                    continue  # Player didn't play
                
                actual_val = actual.get(pick.prop_type.lower(), 0)
                hit, margin = grade_pick(actual_val, pick.line, pick.direction)
                
                pick.actual_value = actual_val
                pick.hit = hit
                pick.margin = margin
                
                # Update counters
                result.total_picks += 1
                if hit:
                    result.hits += 1
                    daily_result["hits"] += 1
                
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
                pt = pick.prop_type.lower()
                if pt == "pts":
                    result.pts_picks += 1
                    if hit:
                        result.pts_hits += 1
                elif pt == "reb":
                    result.reb_picks += 1
                    if hit:
                        result.reb_hits += 1
                elif pt == "ast":
                    result.ast_picks += 1
                    if hit:
                        result.ast_hits += 1
                
                # By pattern
                if pick.pattern == "cold_bounce":
                    result.cold_bounce_picks += 1
                    if hit:
                        result.cold_bounce_hits += 1
                elif pick.pattern == "hot_sustained":
                    result.hot_sustained_picks += 1
                    if hit:
                        result.hot_sustained_hits += 1
                elif pick.pattern == "usage_boost":
                    result.usage_boost_picks += 1
                    if hit:
                        result.usage_boost_hits += 1
                
                result.all_picks.append(pick)
        
        result.daily_results.append(daily_result)
        current += timedelta(days=1)
    
    if verbose:
        print("\n" + result.summary())
    
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V14 General - NBA Props")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_v14_general(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_v14_general(args.date)
        print(picks.summary())
    else:
        # Default: today's date
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        picks = get_daily_picks_v14_general(today)
        print(picks.summary())
