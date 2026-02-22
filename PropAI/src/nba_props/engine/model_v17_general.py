"""
Model V17 General - Holistic Multi-Factor NBA Props Prediction Model
====================================================================

This is the GENERAL model of a dual-model approach:
- Model V17 General (this file): Holistic multi-factor approach for all picks
- Model V17 Under (separate file): Specialized UNDER model (placeholder for Phase 2)

=============================================================================
MODEL V17 KEY INNOVATIONS (Addressing ALL Previous Model Shortcomings)
=============================================================================

1. **HOLISTIC MULTI-FACTOR ANALYSIS** (NOT Just Cold Bounce):
   - Previous models over-relied on single patterns
   - V17 combines MULTIPLE factors into weighted scores
   - This addresses: "do not just suggest based on cold bounces"
   - Considers: injuries, trades, game plan changes, minutes trends

2. **HYBRID LINE APPROACH** (Sportsbook When Available, Projections Otherwise):
   - Use actual sportsbook lines when available
   - STILL GENERATE PICKS without lines (lines come late!)
   - Apply STRICTER edge requirements for derived lines (10% vs 6%)

3. **STRATEGIC DIRECTION SELECTION** (From RCM v1.4 Analysis):
   - PTS: UNDER strongly preferred (63.9% vs 48.3% OVER)
   - REB: OVER preferred with patterns (~61%)
   - AST: EXCLUDED entirely (~54% is coin flip)

4. **COMPREHENSIVE BACKTESTING**:
   - Progress bar for terminal feedback
   - Track by: line source, tier, direction, prop type, factor combination
   - Honest reporting with sportsbook vs derived separation

5. **STRICT FILTERING** (Quality Over Quantity):
   - 23+ minute average for established players
   - 10+ games history required
   - Minimum factor score required (not single pattern)

=============================================================================

USAGE:
------
    from src.nba_props.engine.model_v17_general import (
        get_daily_picks_v17_general,
        run_backtest_v17_general,
        ModelConfigV17General,
    )
    
    # Get picks for today
    picks = get_daily_picks_v17_general("2026-02-03")
    print(picks.summary())
    
    # Run backtest with progress bar
    result = run_backtest_v17_general(
        "2025-10-22", "2026-02-02", 
        verbose=True, 
        show_progress=True
    )
    print(result.summary())

Author: PropAI Team - Model V17
Created: February 2026
Version: 17.0
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

from .model_v17_shared import (
    LineInfo,
    PlayerStatsV17,
    DefenseContextV17,
    BackToBackInfo,
    InjuryImpact,
    HolisticFactorScore,
    HistoricalMatchup,
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
    calculate_holistic_factor_score_under,
    calculate_holistic_factor_score_over,
    calculate_edge,
    calculate_injury_impact,
    grade_pick,
    get_actual_stats,
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
    MIN_FACTOR_SCORE_PREMIUM,
    MIN_FACTOR_SCORE_HIGH,
    MIN_FACTOR_SCORE_STANDARD,
    MODEL_VERSION,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV17General:
    """
    Model V17 General Configuration.
    
    This model uses HOLISTIC multi-factor analysis rather than single patterns.
    Both OVER and UNDER picks are generated based on combined factor scores.
    
    KEY CHANGES FROM V16:
    ---------------------
    1. Holistic factor scoring instead of single-pattern triggers
    2. Minimum factor score required (30+) for picks
    3. Multiple factors combined for projection adjustment
    4. Historical matchup data considered when available
    5. Minutes trends factored in
    6. Stronger penalties for conflicting signals
    """
    # === VERSION INFO ===
    model_name: str = "Model V17 General"
    model_version: str = MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    # KEY: We do NOT require sportsbook lines
    # We use them when available, but still generate picks with projections
    require_sportsbook_line: bool = False
    derived_line_adjustment: float = 1.05  # +5% adjustment for derived lines
    sportsbook_confidence_boost: float = 10.0  # Higher confidence with real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Established players only (from Idea.txt)
    max_games_lookback: int = 20
    
    # === PROJECTION WEIGHTS ===
    # Validated from V14/V15/V16 backtesting
    weight_l3: float = 0.10
    weight_l5: float = 0.20
    weight_l10: float = 0.30
    weight_l15: float = 0.20
    weight_season: float = 0.20
    
    # === FACTOR SCORE THRESHOLDS (TUNED FROM BACKTEST) ===
    # Backtest: 50-60 score: 55%, 60-70: 66.7%, 70-80: 75%
    min_factor_score_premium: float = MIN_FACTOR_SCORE_PREMIUM  # 60 (tuned up)
    min_factor_score_high: float = MIN_FACTOR_SCORE_HIGH  # 45 (tuned up)
    min_factor_score_standard: float = MIN_FACTOR_SCORE_STANDARD  # 35 (tuned up)
    
    # === EDGE REQUIREMENTS (TUNED FROM BACKTEST) ===
    # Backtest: 15-20% edge: 54.5%, 20-25%: 52%, 25-30%: 54%, 30-35%: 78%!
    min_edge_sportsbook: float = 6.0  # 6%+ edge vs sportsbook line
    min_edge_derived: float = 12.0    # Raised from 10% - need higher edge for derived
    min_edge_premium: float = 15.0    # Premium needs 15%+ edge
    min_edge_over: float = 15.0       # NEW: OVERs need higher edge (45.5% backtest!)
    
    # === STRATEGIC DIRECTION (TUNED FROM BACKTEST) ===
    # Backtest: UNDER 55.3% vs OVER 45.5% - SIGNIFICANTLY favor UNDERS
    # PTS: UNDER 55.6% vs OVER 44.4%
    pts_over_allowed: bool = True  # Allow but with high bar
    pts_over_min_factor_score: float = 50.0  # NEW: High bar for PTS OVER
    pts_over_block_elite_defense: bool = True  # Block PTS OVER vs elite defense
    
    # REB: UNDER 54.1% vs OVER 47.0%  
    reb_over_allowed: bool = True    # Allowed but stricter
    reb_over_min_factor_score: float = 45.0  # NEW: Higher bar for REB OVER
    reb_under_allowed: bool = True
    reb_under_min_score: float = 40.0  # Slightly lower since UNDERs perform better
    
    # AST: Excluded by default (54% is coin flip)
    include_ast: bool = False
    min_ast_avg: float = 8.5  # Must average 8.5+ AST if enabled
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 6    # Reduced from 8 for quality
    max_picks_per_day: int = 30    # Reduced from 40 for quality
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
class PropPickV17General:
    """A pick generated by Model V17 General."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details
    prop_type: str  # PTS, REB, AST
    direction: str  # OVER, UNDER
    
    # Line information (KEY FIELD - tracks sportsbook vs derived)
    line: float
    line_source: str  # "sportsbook" or "derived"
    
    # Projection
    projection: float
    projection_adj: float  # After factor adjustments
    
    # Edge calculation (vs line, not player avg!)
    edge_pct: float
    
    # Holistic factor scoring
    factor_score: float
    primary_factor: str  # Main factor driving the pick
    
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
    
    # === Fields with defaults ===
    active_factors: List[str] = field(default_factory=list)
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
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "b2b": self.is_b2b,
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "l15": round(self.l15_avg, 1),
            "season": round(self.season_avg, 1),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV17General:
    """All picks for a day from Model V17 General."""
    date: str
    games: int
    config: ModelConfigV17General = field(default_factory=ModelConfigV17General)
    picks: List[PropPickV17General] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV17General]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV17General]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def over_picks(self) -> List[PropPickV17General]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV17General]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V17 GENERAL PICKS - {self.date}",
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
                for p in sorted(tier_picks, key=lambda x: -x.factor_score):
                    emoji = "📈" if p.direction == "OVER" else "📉"
                    src = f"[{p.book}]" if p.line_source == "sportsbook" else "[derived]"
                    lines.append(
                        f"  {emoji} {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.prop_type} {p.direction} {p.line:.1f} {src}"
                    )
                    lines.append(
                        f"      Proj: {p.projection_adj:.1f} | Edge: {p.edge_pct:.1f}% | "
                        f"Score: {p.factor_score:.0f} | Factors: {p.primary_factor}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV17General:
    """Comprehensive backtest results for Model V17 General."""
    start_date: str
    end_date: str
    config: ModelConfigV17General
    
    # Overall
    total_picks: int = 0
    hits: int = 0
    
    # By line source (KEY METRIC - honest reporting)
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
    
    # By prop type + direction
    pts_over_picks: int = 0
    pts_over_hits: int = 0
    pts_under_picks: int = 0
    pts_under_hits: int = 0
    reb_over_picks: int = 0
    reb_over_hits: int = 0
    reb_under_picks: int = 0
    reb_under_hits: int = 0
    
    # By primary factor
    factor_results: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # All picks for detailed analysis
    all_picks: List[PropPickV17General] = field(default_factory=list)
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
            f"MODEL V17 GENERAL - BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"Avg picks/day: {self.total_picks / max(self.days_tested, 1):.1f}",
            "",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            "",
            f"BY LINE SOURCE (KEY METRIC - Honest Reporting):",
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
            f"BY PROP + DIRECTION:",
            f"  PTS OVER:  {pct(self.pts_over_hits, self.pts_over_picks)} ({self.pts_over_hits}/{self.pts_over_picks})",
            f"  PTS UNDER: {pct(self.pts_under_hits, self.pts_under_picks)} ({self.pts_under_hits}/{self.pts_under_picks})",
            f"  REB OVER:  {pct(self.reb_over_hits, self.reb_over_picks)} ({self.reb_over_hits}/{self.reb_over_picks})",
            f"  REB UNDER: {pct(self.reb_under_hits, self.reb_under_picks)} ({self.reb_under_hits}/{self.reb_under_picks})",
            "",
        ]
        
        # Factor results
        if self.factor_results:
            lines.append("BY PRIMARY FACTOR:")
            for factor, data in sorted(self.factor_results.items(), key=lambda x: -x[1].get('picks', 0)):
                picks = data.get('picks', 0)
                hits = data.get('hits', 0)
                if picks > 0:
                    lines.append(f"  {factor}: {pct(hits, picks)} ({hits}/{picks})")
            lines.append("")
        
        lines.append(f"{'='*70}")
        return "\n".join(lines)
    
    def detailed_report(self) -> str:
        """Generate detailed breakdown report."""
        lines = [self.summary(), "", "DETAILED BREAKDOWN:", ""]
        
        # Edge analysis
        if self.all_picks:
            hits_by_edge = {}
            for p in self.all_picks:
                bucket = int(p.edge_pct // 5) * 5  # 0-5, 5-10, etc.
                key = f"{bucket}-{bucket+5}%"
                if key not in hits_by_edge:
                    hits_by_edge[key] = {"picks": 0, "hits": 0}
                hits_by_edge[key]["picks"] += 1
                if p.hit:
                    hits_by_edge[key]["hits"] += 1
            
            lines.append("BY EDGE BUCKET:")
            for key in sorted(hits_by_edge.keys()):
                data = hits_by_edge[key]
                pct = data["hits"]/data["picks"]*100 if data["picks"] > 0 else 0
                lines.append(f"  {key}: {pct:.1f}% ({data['hits']}/{data['picks']})")
            
            # Factor score analysis
            lines.append("")
            lines.append("BY FACTOR SCORE BUCKET:")
            hits_by_score = {}
            for p in self.all_picks:
                bucket = int(p.factor_score // 10) * 10
                key = f"{bucket}-{bucket+10}"
                if key not in hits_by_score:
                    hits_by_score[key] = {"picks": 0, "hits": 0}
                hits_by_score[key]["picks"] += 1
                if p.hit:
                    hits_by_score[key]["hits"] += 1
            
            for key in sorted(hits_by_score.keys()):
                data = hits_by_score[key]
                pct = data["hits"]/data["picks"]*100 if data["picks"] > 0 else 0
                lines.append(f"  {key}: {pct:.1f}% ({data['hits']}/{data['picks']})")
        
        return "\n".join(lines)


# ============================================================================
# Core Model Functions
# ============================================================================

def _calculate_confidence(
    factor_score: float,
    edge_pct: float,
    line_source: str,
    direction: str,
    config: ModelConfigV17General,
) -> float:
    """
    Calculate confidence score for a pick.
    
    Based on:
    - Factor score (primary driver)
    - Edge size
    - Line source (sportsbook gets boost)
    """
    # Base confidence from factor score
    # Score of 30 -> ~68 confidence
    # Score of 55 -> ~85 confidence
    base = 55.0 + (factor_score * 0.55)
    
    # Edge bonus (capped at 10)
    edge_bonus = min(edge_pct / 2.5, 10.0)
    base += edge_bonus
    
    # Sportsbook line bonus
    if line_source == "sportsbook":
        base += config.sportsbook_confidence_boost
    
    return min(base, 100.0)


def _determine_tier(
    factor_score: float,
    edge: float,
    config: ModelConfigV17General,
) -> str:
    """Determine confidence tier based on factor score and edge."""
    if factor_score >= config.min_factor_score_premium and edge >= config.min_edge_premium:
        return "PREMIUM"
    elif factor_score >= config.min_factor_score_high:
        return "HIGH"
    elif factor_score >= config.min_factor_score_standard:
        return "STANDARD"
    else:
        return "LOW"  # Will be filtered out


def _analyze_player_for_picks(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    game_date: str,
    injured_teammates: List[Dict[str, Any]],
    config: ModelConfigV17General,
) -> List[PropPickV17General]:
    """
    Analyze a player for potential picks using holistic factor scoring.
    
    Returns list of picks that meet all criteria.
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
    
    # Calculate injury impact
    injury_impact = calculate_injury_impact(injured_teammates)
    
    # Track best pick per player (we only want 1)
    best_pick = None
    best_combined_score = 0.0  # factor_score + edge
    
    # Analyze each prop type
    for prop_type in config.prop_types:
        pt = prop_type.lower()
        
        # Check minimum average (filter low-volume players)
        season_avg = stats.season.get(pt, 0)
        min_avg = MIN_PROP_AVERAGES.get(pt, 0)
        
        # Special AST handling (very high bar)
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
        
        # Get historical matchup data
        historical = get_historical_matchup(
            conn, player_id, opponent_abbrev, game_date, stats.season
        )
        
        # Calculate base projection
        projection = stats.get_projection(prop_type, config.get_weights())
        
        # Determine minimum edge based on line source
        min_edge = (config.min_edge_sportsbook if line_info.is_sportsbook 
                   else config.min_edge_derived)
        
        # ================================================================
        # ANALYZE UNDER POTENTIAL
        # ================================================================
        under_score = calculate_holistic_factor_score_under(
            stats, defense, b2b, historical, prop_type
        )
        
        # Apply projection adjustment
        under_proj_adj = projection * under_score.projection_adj_multiplier
        
        # Calculate edge
        under_edge = calculate_edge(under_proj_adj, line_info.line, "UNDER")
        
        # Check if pick qualifies
        # For REB UNDER, apply higher minimum score
        min_score = config.reb_under_min_score if pt == 'reb' else config.min_factor_score_standard
        
        # VALIDATION: cold_streak_severe as PRIMARY factor is problematic (47.6% hit rate!)
        # Either skip entirely or require VERY high supporting factors
        active_factors = [f for f, v in under_score.factors.items() if v]
        skip_pick = False
        
        if under_score.primary_factor == "cold_streak_severe":
            # Only allow if there's a strong supporting factor
            strong_factors = {"defense_elite", "defense_good", "b2b_fatigue", "injury_rust_first"}
            has_strong_support = any(f in strong_factors for f in active_factors if f != "cold_streak_severe")
            
            if not has_strong_support:
                skip_pick = True  # Skip - cold_streak_severe without strong support
        
        if not skip_pick and (under_score.total_score >= min_score and 
            under_edge >= min_edge):
            
            # Check additional REB UNDER restrictions
            if pt == 'reb' and not config.reb_under_allowed:
                pass  # Skip
            else:
                tier = _determine_tier(under_score.total_score, under_edge, config)
                
                if tier != "LOW":
                    confidence = _calculate_confidence(
                        under_score.total_score, under_edge, 
                        line_info.source, "UNDER", config
                    )
                    
                    combined = under_score.total_score + under_edge
                    
                    if combined > best_combined_score:
                        best_combined_score = combined
                        best_pick = PropPickV17General(
                            player_id=player_id,
                            player_name=stats.player_name,
                            team_abbrev=stats.team_abbrev,
                            opponent_abbrev=opponent_abbrev,
                            game_date=game_date,
                            prop_type=prop_type.upper(),
                            direction="UNDER",
                            line=line_info.line,
                            line_source=line_info.source,
                            book=line_info.book,
                            projection=projection,
                            projection_adj=under_proj_adj,
                            edge_pct=under_edge,
                            factor_score=under_score.total_score,
                            primary_factor=under_score.primary_factor,
                            active_factors=[f for f, v in under_score.factors.items() if v],
                            confidence_score=confidence,
                            confidence_tier=tier,
                            defense_rank=defense.get_rank(pt),
                            defense_rating=defense.get_rating(pt),
                            is_b2b=b2b.is_second_of_b2b,
                            historical_games=historical.games_played if historical else 0,
                            historical_avg=getattr(historical, f"avg_{pt}", 0) if historical else 0,
                            l5_avg=stats.l5.get(pt, 0),
                            l10_avg=stats.l10.get(pt, 0),
                            l15_avg=stats.l15.get(pt, 0),
                            season_avg=stats.season.get(pt, 0),
                            reasons=under_score.factor_reasons,
                        )
        
        # ================================================================
        # ANALYZE OVER POTENTIAL
        # (CAUTIOUS - OVERs hit only 45.5% in backtest!)
        # ================================================================
        over_score = calculate_holistic_factor_score_over(
            stats, defense, b2b, historical, injury_impact, prop_type
        )
        
        # Apply projection adjustment
        over_proj_adj = projection * over_score.projection_adj_multiplier
        
        # Calculate edge
        over_edge = calculate_edge(over_proj_adj, line_info.line, "OVER")
        
        # OVERs require HIGHER edge (backtest: 45.5% overall)
        min_edge_over = max(min_edge, config.min_edge_over)
        
        # Check if pick qualifies
        # Skip PTS OVER if not allowed or blocked by elite defense
        if pt == 'pts':
            if not config.pts_over_allowed:
                continue
            if config.pts_over_block_elite_defense and defense.is_elite(pt):
                continue
            # NEW: PTS OVER requires higher factor score (backtest: 44.4%)
            min_score_over = config.pts_over_min_factor_score
        elif pt == 'reb':
            if not config.reb_over_allowed:
                continue
            # NEW: REB OVER requires higher factor score (backtest: 47.0%)
            min_score_over = config.reb_over_min_factor_score
        else:
            min_score_over = config.min_factor_score_standard
        
        if (over_score.total_score >= min_score_over and 
            over_edge >= min_edge_over):
            
            tier = _determine_tier(over_score.total_score, over_edge, config)
            
            if tier != "LOW":
                confidence = _calculate_confidence(
                    over_score.total_score, over_edge, 
                    line_info.source, "OVER", config
                )
                
                combined = over_score.total_score + over_edge
                
                if combined > best_combined_score:
                    best_combined_score = combined
                    best_pick = PropPickV17General(
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
                        projection_adj=over_proj_adj,
                        edge_pct=over_edge,
                        factor_score=over_score.total_score,
                        primary_factor=over_score.primary_factor,
                        active_factors=[f for f, v in over_score.factors.items() if v],
                        confidence_score=confidence,
                        confidence_tier=tier,
                        defense_rank=defense.get_rank(pt),
                        defense_rating=defense.get_rating(pt),
                        is_b2b=b2b.is_second_of_b2b,
                        historical_games=historical.games_played if historical else 0,
                        historical_avg=getattr(historical, f"avg_{pt}", 0) if historical else 0,
                        l5_avg=stats.l5.get(pt, 0),
                        l10_avg=stats.l10.get(pt, 0),
                        l15_avg=stats.l15.get(pt, 0),
                        season_avg=stats.season.get(pt, 0),
                        reasons=over_score.factor_reasons,
                    )
    
    if best_pick:
        picks.append(best_pick)
    
    return picks


def get_daily_picks_v17_general(
    game_date: str,
    config: Optional[ModelConfigV17General] = None,
    db_path: Optional[Path] = None,
) -> DailyPicksV17General:
    """
    Generate picks for a specific date using Model V17 General.
    
    Uses holistic multi-factor scoring instead of single-pattern triggers.
    
    Args:
        game_date: Date string (YYYY-MM-DD)
        config: Optional configuration override
        db_path: Optional database path override
    
    Returns:
        DailyPicksV17General with all generated picks
    """
    if config is None:
        config = ModelConfigV17General()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=db_path)
    result = DailyPicksV17General(date=game_date, games=0, config=config)
    
    with db.connect() as conn:
        # Get games for the date
        games = get_games_for_date(conn, game_date)
        result.games = len(games)
        
        if not games:
            return result
        
        # Get injured players
        injured_ids = get_injured_players(conn, game_date)
        
        all_picks = []
        
        # Process each game
        for game in games:
            away_team = game["away_team"]
            home_team = game["home_team"]
            away_abbrev = abbrev_from_team_name(away_team) or ""
            home_abbrev = abbrev_from_team_name(home_team) or ""
            
            # Get injured teammates for each team
            away_injured = get_injured_players_for_team(conn, game_date, away_abbrev)
            home_injured = get_injured_players_for_team(conn, game_date, home_abbrev)
            
            # Get players for away team
            away_players = get_players_in_game(
                conn, away_abbrev, game_date,
                min_games=config.min_games_required,
                min_avg_minutes=config.min_avg_minutes
            )
            
            # Get players for home team
            home_players = get_players_in_game(
                conn, home_abbrev, game_date,
                min_games=config.min_games_required,
                min_avg_minutes=config.min_avg_minutes
            )
            
            # Analyze away team players (playing against home team)
            for player_id in away_players:
                if player_id in injured_ids:
                    continue
                
                result.players_analyzed += 1
                
                picks = _analyze_player_for_picks(
                    conn, player_id, home_abbrev, game_date,
                    away_injured, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                    all_picks.append(pick)
            
            # Analyze home team players (playing against away team)
            for player_id in home_players:
                if player_id in injured_ids:
                    continue
                
                result.players_analyzed += 1
                
                picks = _analyze_player_for_picks(
                    conn, player_id, away_abbrev, game_date,
                    home_injured, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                    all_picks.append(pick)
        
        # Sort by factor_score + edge (combined strength)
        all_picks.sort(key=lambda p: -(p.factor_score + p.edge_pct))
        
        # Apply limits
        result.picks = all_picks[:config.max_picks_per_day]
    
    return result


def run_backtest_v17_general(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV17General] = None,
    db_path: Optional[Path] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> BacktestResultV17General:
    """
    Run comprehensive backtest for Model V17 General.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Optional configuration object
        db_path: Optional database path override
        verbose: Print daily details
        show_progress: Show progress bar in terminal
    
    Returns:
        BacktestResultV17General with all statistics
    """
    if config is None:
        config = ModelConfigV17General()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=db_path)
    result = BacktestResultV17General(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days + 1
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False
        if show_progress:
            print("Note: Install 'tqdm' for progress bar (pip install tqdm)")
    
    print(f"\n{'='*70}")
    print(f"MODEL V17 GENERAL - BACKTESTING")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date} ({total_days} days)")
    print(f"Configuration: {config.model_name} v{config.model_version}")
    print(f"{'='*70}\n")
    
    current = start
    day_num = 0
    
    # Create iterator with optional progress bar
    if use_tqdm:
        date_iterator = tqdm(
            range(total_days),
            desc="Processing",
            unit="day",
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
    else:
        date_iterator = range(total_days)
    
    with db.connect() as conn:
        for _ in date_iterator:
            date_str = current.strftime("%Y-%m-%d")
            
            if verbose and not use_tqdm:
                print(f"Processing {date_str}...", end=" ")
            
            # Get picks for this date
            daily_picks = get_daily_picks_v17_general(date_str, config, db_path)
            
            if daily_picks.games > 0:
                result.days_tested += 1
                result.total_games += daily_picks.games
            
            daily_hits = 0
            daily_picks_count = 0
            
            # Grade each pick against actual results
            for pick in daily_picks.picks:
                actual = get_actual_stats(conn, pick.player_id, date_str)
                
                if actual is None:
                    continue
                
                actual_value = actual.get(pick.prop_type.lower(), 0)
                hit, margin = grade_pick(actual_value, pick.line, pick.direction)
                
                pick.actual_value = actual_value
                pick.hit = hit
                pick.margin = margin
                
                # Update totals
                result.total_picks += 1
                daily_picks_count += 1
                if hit:
                    result.hits += 1
                    daily_hits += 1
                
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
                elif pick.confidence_tier == "STANDARD":
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
                    if pick.direction == "OVER":
                        result.pts_over_picks += 1
                        if hit:
                            result.pts_over_hits += 1
                    else:
                        result.pts_under_picks += 1
                        if hit:
                            result.pts_under_hits += 1
                elif pt == "reb":
                    result.reb_picks += 1
                    if hit:
                        result.reb_hits += 1
                    if pick.direction == "OVER":
                        result.reb_over_picks += 1
                        if hit:
                            result.reb_over_hits += 1
                    else:
                        result.reb_under_picks += 1
                        if hit:
                            result.reb_under_hits += 1
                elif pt == "ast":
                    result.ast_picks += 1
                    if hit:
                        result.ast_hits += 1
                
                # By primary factor
                factor = pick.primary_factor or "unknown"
                if factor not in result.factor_results:
                    result.factor_results[factor] = {"picks": 0, "hits": 0}
                result.factor_results[factor]["picks"] += 1
                if hit:
                    result.factor_results[factor]["hits"] += 1
                
                result.all_picks.append(pick)
            
            if verbose and not use_tqdm and daily_picks_count > 0:
                pct = daily_hits / daily_picks_count * 100
                print(f"{daily_hits}/{daily_picks_count} ({pct:.1f}%)")
            elif verbose and not use_tqdm:
                print("No picks")
            
            result.daily_results.append({
                "date": date_str,
                "games": daily_picks.games,
                "picks": daily_picks_count,
                "hits": daily_hits,
            })
            
            current += timedelta(days=1)
            day_num += 1
            
            # Progress update for non-tqdm
            if not use_tqdm and show_progress and day_num % 10 == 0:
                pct_complete = day_num / total_days * 100
                running_rate = result.hit_rate
                print(f"  Progress: {pct_complete:.0f}% | Running hit rate: {running_rate:.1f}%")
    
    print("\n" + result.summary())
    
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V17 General - NBA Props Prediction")
    parser.add_argument("--date", type=str, help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", type=str, help="Backtest start date")
    parser.add_argument("--backtest-end", type=str, help="Backtest end date")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        print(f"Running backtest from {args.backtest_start} to {args.backtest_end}...")
        result = run_backtest_v17_general(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
            show_progress=not args.no_progress
        )
        print(result.detailed_report())
    elif args.date:
        print(f"Generating picks for {args.date}...")
        picks = get_daily_picks_v17_general(args.date)
        print(picks.summary())
    else:
        # Default to today
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"Generating picks for {today}...")
        picks = get_daily_picks_v17_general(today)
        print(picks.summary())
