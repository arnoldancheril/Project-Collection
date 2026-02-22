"""
Model V19 General - Holistic Multi-Factor NBA Props Prediction Model
====================================================================

This is the GENERAL model of a dual-model approach:
- Model V19 General (this file): Holistic multi-factor approach for all picks
- Model V19 Under (separate file): Specialized UNDER model (Phase 2 placeholder)

=============================================================================
MODEL V19 KEY IMPROVEMENTS OVER V18
=============================================================================

1. **STRICTER MULTI-FACTOR REQUIREMENTS**:
   - V19 requires multiple factors to align before making a pick
   - No single-factor picks allowed (learned from "cold bounce alone" issues)
   - Alignment score tracks how well factors reinforce each other

2. **COMPREHENSIVE BOX SCORE ANALYSIS**:
   - Analyzes Plus/Minus (+/-) as consistency indicator
   - Tracks Shooting Efficiency (FG%, TS%) trends
   - Monitors FTA (aggression indicator) trends
   - This addresses: "analyzing box scores and prior game data"

3. **HYBRID LINE APPROACH** (Always Generate Picks):
   - Use sportsbook lines when available (accurate edge)
   - ALWAYS generate picks (lines come late) - use derived with stricter edge
   - Track line source for honest reporting
   - 6% edge for sportsbook, 15% for derived

4. **STRATEGIC DIRECTION SELECTION** (Data-Driven):
   - PTS: UNDER strongly preferred (63.9% vs 48.3% OVER from RCM)
   - PTS OVER: Only with cold bounce + NOT vs elite defense + high edge
   - REB: Both directions (~59% each)
   - AST: EXCLUDED (~54% is coin flip after juice)

5. **GAME CONTEXT INTEGRATION**:
   - Blowout risk detection (large spreads)
   - Pace factors (O/U-based)
   - Home/away context

6. **HONEST REPORTING**:
   - Track sportsbook vs derived picks separately
   - Report hit rates by line source
   - No inflated metrics

=============================================================================

USAGE:
------
    from src.nba_props.engine.model_v19_general import (
        get_daily_picks_v19_general,
        run_backtest_v19_general,
        ModelConfigV19General,
    )
    
    # Get picks for today
    picks = get_daily_picks_v19_general("2026-02-03")
    print(picks.summary())
    
    # Run backtest with progress bar
    result = run_backtest_v19_general(
        "2025-10-22", "2026-02-02", 
        verbose=True, 
        show_progress=True
    )
    print(result.summary())

Author: PropAI Team - Model V19
Created: February 2026
Version: 19.0
"""
from __future__ import annotations

import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths

from .model_v19_shared import (
    # Data classes
    LineInfo,
    PlayerStatsV19,
    DefenseContextV19,
    BackToBackInfo,
    InjuryImpact,
    HolisticFactorScore,
    HistoricalMatchup,
    EfficiencyStats,
    GameContext,
    
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
    get_game_context,
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
    format_time_remaining,
    
    # Constants
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
    MIN_FACTOR_SCORE_PREMIUM,
    MIN_FACTOR_SCORE_HIGH,
    MIN_FACTOR_SCORE_STANDARD,
    MIN_FACTOR_SCORE_OVER_PREMIUM,
    MIN_FACTOR_SCORE_OVER_HIGH,
    MIN_FACTOR_SCORE_OVER_STANDARD,
    FACTOR_PROJECTION_ADJUSTMENTS,
    MODEL_VERSION,
)

# Trade deadline handling
from .post_trade_adjustments import (
    get_trade_context,
    apply_trade_adjustments,
    should_skip_player,
    get_trade_factor_for_under,
    get_trade_factor_for_over,
    get_opponent_tank_boost,
    TradeContext,
    TRADE_DEADLINE_DATE,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV19General:
    """
    Model V19 General Configuration.
    
    This model uses HOLISTIC multi-factor analysis with comprehensive
    box score data including +/-, efficiency, and historical matchups.
    
    KEY V19 IMPROVEMENTS:
    - Stricter multi-factor requirements
    - No single-factor picks
    - Higher edge requirements for derived lines
    - Game context integration
    """
    # === VERSION INFO ===
    model_name: str = "Model V19 General"
    model_version: str = MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False  # ALWAYS generate picks
    derived_line_adjustment: float = 1.05  # +5% adjustment for derived lines
    sportsbook_confidence_boost: float = 12.0  # Higher confidence with real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Established players only (from Idea.txt)
    max_games_lookback: int = 25  # Increased for better trends
    
    # === PROJECTION WEIGHTS ===
    weight_l3: float = 0.10
    weight_l5: float = 0.20
    weight_l10: float = 0.30
    weight_l15: float = 0.20
    weight_season: float = 0.20
    
    # === FACTOR SCORE THRESHOLDS (STRICTER in V19) ===
    min_factor_score_premium: float = MIN_FACTOR_SCORE_PREMIUM  # 65
    min_factor_score_high: float = MIN_FACTOR_SCORE_HIGH  # 50
    min_factor_score_standard: float = MIN_FACTOR_SCORE_STANDARD  # 40
    
    # For OVER picks (even stricter)
    min_factor_score_over_premium: float = MIN_FACTOR_SCORE_OVER_PREMIUM  # 55
    min_factor_score_over_high: float = MIN_FACTOR_SCORE_OVER_HIGH  # 45
    min_factor_score_over_standard: float = MIN_FACTOR_SCORE_OVER_STANDARD  # 35
    
    # === MULTI-FACTOR REQUIREMENTS (V19 KEY) ===
    require_multiple_factors_under: bool = True
    min_factors_required_under: int = 2  # Need at least 2 factors
    require_multiple_factors_over: bool = True
    min_factors_required_over: int = 2  # OVERs also need 2 factors
    
    # === EDGE REQUIREMENTS (STRICTER in V19) ===
    # KEY: Much higher requirements for derived lines
    min_edge_sportsbook: float = 6.0   # Lower for real lines
    min_edge_derived: float = 15.0     # MUCH stricter for derived (was 12%)
    min_edge_premium: float = 18.0     # Premium tier needs high edge
    min_edge_over: float = 18.0        # OVERs need higher edge (riskier)
    
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
    reb_under_min_factor_score: float = 42.0
    
    # AST: EXCLUDED by default
    include_ast: bool = False
    min_ast_avg: float = 8.5  # Must average 8.5+ if enabled
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 6
    max_picks_per_day: int = 35  # Reduced for quality
    max_picks_per_player: int = 1  # Focus on best prop per player
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence: float = 88.0
    high_confidence: float = 78.0
    standard_confidence: float = 70.0
    
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


def create_post_deadline_config(
    deadline_date: str = TRADE_DEADLINE_DATE,
    extra_cautious: bool = False,
) -> ModelConfigV19General:
    """
    Return a ModelConfigV19General tuned for post-trade-deadline games.

    V19.4 insight: After Feb 6, 2026:
    - Sportsbook lines hit at 84.6% (trust them, keep low edge bar)
    - Derived lines hit at 51.3% (need a larger cushion)
    - Roster changes create higher variance → require more factors

    Args:
        deadline_date: ISO date string of the trade deadline (informational).
        extra_cautious: If True, apply even stricter thresholds (20+ edge,
                        PREMIUM-only picks). Useful for the first 1-2 weeks
                        post-deadline before new-team data stabilises.

    Returns:
        ModelConfigV19General with adjusted thresholds.
    """
    base = ModelConfigV19General()

    # Derived lines: raise edge requirement to compensate for higher variance
    base.min_edge_derived = 22.0 if extra_cautious else 18.0

    # Sportsbook lines: keep low (they're highly accurate post-deadline)
    base.min_edge_sportsbook = 5.0

    # Factor score: require HIGH+ picks; skip STANDARD (lower signal when
    # rosters are new and chemistry is unformed)
    base.min_factor_score_standard = 50.0 if extra_cautious else 48.0

    # OVER picks: even more conservative post-deadline (usage patterns unclear)
    base.min_edge_over = 22.0 if extra_cautious else 20.0
    base.min_factor_score_over_standard = 45.0

    # Require at least 3 factors post-deadline (extra confirmation)
    if extra_cautious:
        base.min_factors_required_under = 3
        base.min_factors_required_over = 3

    return base


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PropPickV19General:
    """A pick generated by Model V19 General."""
    # Identity - required fields
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details - required fields
    prop_type: str  # PTS, REB
    direction: str  # OVER, UNDER
    
    # Line information (KEY: tracks sportsbook vs derived)
    line: float
    line_source: str  # "sportsbook" or "derived"
    
    # Projection - required
    projection: float
    projection_adj: float  # After factor adjustments
    
    # Edge calculation - required
    edge_pct: float
    
    # Holistic factor scoring (V19 Core) - required
    factor_score: float
    primary_factor: str
    
    # Confidence - required
    confidence_score: float
    confidence_tier: str  # PREMIUM, HIGH, STANDARD
    
    # Context - required
    defense_rank: int
    defense_rating: str
    
    # Supporting data - required
    l5_avg: float
    l10_avg: float
    l15_avg: float
    season_avg: float
    
    # ========== OPTIONAL FIELDS WITH DEFAULTS BELOW ==========
    
    # V19: Track secondary factor
    secondary_factor: str = ""
    factor_count: int = 0
    alignment_score: float = 0.0
    
    # Active factors list
    active_factors: List[str] = field(default_factory=list)
    
    # V19: Enhanced efficiency metrics
    l5_plus_minus: float = 0.0
    l5_fg_pct: float = 0.0
    l5_ts_pct: float = 0.0
    consistency_cv: float = 0.0
    
    # Optional context fields
    book: Optional[str] = None
    is_b2b: bool = False
    is_blowout_risk: bool = False  # V19: Game context
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
            "factor_count": self.factor_count,
            "primary_factor": self.primary_factor,
            "secondary_factor": self.secondary_factor,
            "alignment": round(self.alignment_score, 1),
            "active_factors": self.active_factors,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "b2b": self.is_b2b,
            "blowout_risk": self.is_blowout_risk,
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "l15": round(self.l15_avg, 1),
            "season": round(self.season_avg, 1),
            "l5_plus_minus": round(self.l5_plus_minus, 1),
            "l5_fg_pct": round(self.l5_fg_pct * 100, 1) if self.l5_fg_pct else 0,
            "cv": round(self.consistency_cv, 2),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV19General:
    """All picks for a day from Model V19 General."""
    date: str
    games: int
    config: ModelConfigV19General = field(default_factory=ModelConfigV19General)
    picks: List[PropPickV19General] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV19General]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV19General]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def over_picks(self) -> List[PropPickV19General]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV19General]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    @property
    def premium_picks(self) -> List[PropPickV19General]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV19General]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V19 GENERAL PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games} | Players analyzed: {self.players_analyzed}",
            f"Sportsbook lines available: {self.players_with_sportsbook_lines}",
            f"Using derived lines: {self.players_with_derived_lines}",
            "",
            f"Total picks: {self.total_picks}",
            f"  OVER: {len(self.over_picks)} | UNDER: {len(self.under_picks)}",
            f"  Sportsbook: {len(self.sportsbook_picks)} | Derived: {len(self.derived_picks)}",
            f"  PREMIUM: {len(self.premium_picks)} | HIGH: {len(self.high_picks)}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in sorted(tier_picks, key=lambda x: x.factor_score, reverse=True):
                    emoji = "📈" if p.direction == "OVER" else "📉"
                    src = f"[{p.book}]" if p.line_source == "sportsbook" else "[derived]"
                    lines.append(
                        f"  {emoji} {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.direction} {p.line} {p.prop_type.upper()} {src}"
                    )
                    lines.append(
                        f"      Proj: {p.projection_adj:.1f} | Edge: {p.edge_pct:.1f}% | "
                        f"Score: {p.factor_score:.0f} ({p.factor_count} factors)"
                    )
                    lines.append(
                        f"      Primary: {p.primary_factor} | Secondary: {p.secondary_factor or 'N/A'}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV19:
    """Results from a Model V19 backtest run."""
    start_date: str
    end_date: str
    config: ModelConfigV19General
    
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
    
    # By prop + direction (V19 detailed)
    pts_over_picks: int = 0
    pts_over_hits: int = 0
    pts_under_picks: int = 0
    pts_under_hits: int = 0
    reb_over_picks: int = 0
    reb_over_hits: int = 0
    reb_under_picks: int = 0
    reb_under_hits: int = 0
    
    # By primary factor
    by_factor: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By factor count (V19)
    by_factor_count: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    # By factor score range
    by_score_range: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By edge range
    by_edge_range: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Individual picks (for detailed analysis)
    all_picks: List[PropPickV19General] = field(default_factory=list)
    
    # ROI calculation
    theoretical_profit: float = 0.0
    theoretical_wagers: float = 0.0
    
    # Timing
    days_tested: int = 0
    games_tested: int = 0
    runtime_seconds: float = 0.0
    
    @property
    def theoretical_roi(self) -> float:
        if self.theoretical_wagers == 0:
            return 0.0
        return (self.theoretical_profit / self.theoretical_wagers) * 100
    
    @property
    def picks_per_day(self) -> float:
        if self.days_tested == 0:
            return 0.0
        return self.total_picks / self.days_tested
    
    def _safe_rate(self, hits: int, total: int) -> str:
        if total == 0:
            return "N/A"
        return f"{hits/total*100:.1f}%"
    
    def summary(self) -> str:
        lines = [
            "",
            "=" * 80,
            f"MODEL V19 GENERAL BACKTEST RESULTS",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.games_tested} | Runtime: {self.runtime_seconds:.1f}s",
            "=" * 80,
            "",
            f"OVERALL: {self.hit_rate:.1%} ({self.total_hits}/{self.total_picks})",
            f"Picks per day: {self.picks_per_day:.1f}",
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
            "BY PROP + DIRECTION:",
            f"  PTS OVER:  {self._safe_rate(self.pts_over_hits, self.pts_over_picks)} ({self.pts_over_hits}/{self.pts_over_picks})",
            f"  PTS UNDER: {self._safe_rate(self.pts_under_hits, self.pts_under_picks)} ({self.pts_under_hits}/{self.pts_under_picks})",
            f"  REB OVER:  {self._safe_rate(self.reb_over_hits, self.reb_over_picks)} ({self.reb_over_hits}/{self.reb_over_picks})",
            f"  REB UNDER: {self._safe_rate(self.reb_under_hits, self.reb_under_picks)} ({self.reb_under_hits}/{self.reb_under_picks})",
            "",
        ]
        
        # Factor count breakdown (V19)
        if self.by_factor_count:
            lines.append("BY FACTOR COUNT (V19 Multi-Factor Analysis):")
            for count in sorted(self.by_factor_count.keys()):
                data = self.by_factor_count[count]
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {count} factors: {self._safe_rate(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # Factor breakdown
        if self.by_factor:
            lines.append("BY PRIMARY FACTOR:")
            sorted_factors = sorted(
                self.by_factor.items(), 
                key=lambda x: x[1].get('total', 0), 
                reverse=True
            )
            for factor, data in sorted_factors[:15]:  # Top 15
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {factor}: {self._safe_rate(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # Score range breakdown
        if self.by_score_range:
            lines.append("BY FACTOR SCORE RANGE:")
            for range_name in sorted(self.by_score_range.keys()):
                data = self.by_score_range[range_name]
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {range_name}: {self._safe_rate(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # Edge range breakdown
        if self.by_edge_range:
            lines.append("BY EDGE RANGE:")
            for range_name in sorted(self.by_edge_range.keys()):
                data = self.by_edge_range[range_name]
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
    stats: PlayerStatsV19,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: ModelConfigV19General,
    game_context: Optional[GameContext] = None,
) -> List[PropPickV19General]:
    """
    Evaluate a player for a specific prop type.
    
    V19 KEY CHANGES:
    - Requires multiple factors
    - Higher edge requirements for derived
    - Game context integration
    
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
    boost_pct, boost_reasons = calculate_usage_boost(injured)
    
    injury_impact = InjuryImpact(
        injured_teammates=injured,
        usage_boost_pct=boost_pct * 100,  # Convert to percentage
    )
    
    # Get game context if not provided
    if game_context is None:
        game_context = get_game_context(conn, stats.team_abbrev, opponent_abbrev, game_date)
    
    # =========================================================================
    # V19.1: Trade Deadline Adjustments
    # =========================================================================
    trade_ctx = get_trade_context(
        conn, stats.player_id, stats.player_name,
        stats.team_abbrev, game_date
    )
    
    # Get line (sportsbook if available, derived otherwise)
    line_info = get_line(
        conn, stats.player_id, stats.player_name, pt, game_date, stats,
        config.derived_line_adjustment
    )

    # =========================================================================
    # V19.4: Dynamic lookback weights
    # Previously: config.get_weights() was ALWAYS passed to get_projection(),
    # which bypassed the trade-aware weight logic inside get_projection() (it only
    # activates when weights=None).
    #
    # Now:
    # 1. Traded players (stats.was_traded) → pass None so get_projection() selects
    #    its built-in trade-aware weight schedule (heavily recent).
    # 2. Non-traded players on disrupted teams (roster_stability < 0.60, post-deadline)
    #    → shift toward recent games to reflect new team context.
    # 3. All others → standard config weights.
    # =========================================================================
    if stats.was_traded and stats.new_team_games < 10:
        # Trade-aware weights handled inside get_projection() when weights=None
        projection_weights = None
    elif (
        game_date >= TRADE_DEADLINE_DATE
        and trade_ctx.team_status is not None
        and trade_ctx.team_status.roster_stability_score < 0.60
    ):
        stability = trade_ctx.team_status.roster_stability_score
        if stability <= 0.35:
            # Extreme disruption (e.g. CHI 0.30 — 6 in / 6 out):
            # very heavy recent bias, season/L15 data reflects a completely
            # different roster.
            projection_weights = {
                'l3': 0.35, 'l5': 0.35, 'l10': 0.20, 'l15': 0.05, 'season': 0.05,
            }
        elif stability <= 0.50:
            # High disruption (e.g. MEM 0.44 — 4 out / 3 in):
            projection_weights = {
                'l3': 0.25, 'l5': 0.30, 'l10': 0.25, 'l15': 0.10, 'season': 0.10,
            }
        else:
            # Moderate disruption (stability 0.50–0.59, e.g. UTA 0.52):
            projection_weights = {
                'l3': 0.15, 'l5': 0.25, 'l10': 0.30, 'l15': 0.15, 'season': 0.15,
            }
    else:
        projection_weights = config.get_weights()

    # Get base projection
    projection = stats.get_projection(pt, projection_weights)
    
    # Apply trade deadline adjustments to projection
    if trade_ctx.has_any_impact:
        trade_adj = apply_trade_adjustments(
            conn, stats.player_id, stats.player_name,
            stats.team_abbrev, game_date,
            projection, stats.season.get('reb', 0),
            stats.season.get('ast', 0), stats.avg_minutes,
        )
        # Adjust projection based on trade context
        if pt == 'pts':
            projection = trade_adj.adjusted_pts
        elif pt == 'reb':
            projection = trade_adj.adjusted_reb
        elif pt == 'ast':
            projection = trade_adj.adjusted_ast
    
    # =========================================================================
    # V19.4 FIX: Trade uncertainty — penalize confidence for recently traded players
    # Previously used stats.was_traded (never set). Now correctly uses trade_ctx.
    # =========================================================================
    trade_uncertainty_active = (
        trade_ctx.player_was_traded and
        trade_ctx.trade_info is not None and
        game_date >= TRADE_DEADLINE_DATE
    )
    new_team_games = (
        trade_ctx.trade_info.games_with_new_team
        if (trade_uncertainty_active and trade_ctx.trade_info)
        else 99  # Non-traded player, no uncertainty
    )

    if trade_uncertainty_active and new_team_games < 3:
        # Too uncertain — skip entirely for general model
        return []

    # V19.4: Set trade_confidence_discount on stats object so _create_under_pick
    # and _create_over_pick can apply it to confidence scores.
    # Previously this was checking stats.trade_confidence_discount which was never set.
    if trade_uncertainty_active and trade_ctx.trade_info:
        # These are normal Python objects so we can set attrs even on dataclasses
        try:
            stats.trade_confidence_discount = trade_ctx.trade_info.confidence_discount
        except Exception:
            pass  # Ignore if frozen dataclass
    if trade_ctx.tank_result and trade_ctx.tank_result.is_tanking:
        tank_confidence_penalty = trade_ctx.tank_result.overall_confidence_impact
        try:
            # Compound tank penalty with any existing trade discount
            existing = getattr(stats, 'trade_confidence_discount', 1.0)
            stats.trade_confidence_discount = existing * tank_confidence_penalty
        except Exception:
            pass

    # =========================================================================
    # Evaluate UNDER
    # =========================================================================
    under_score = calculate_holistic_factor_score_under(
        stats, pt, defense, b2b, injury_impact, game_context
    )
    
    # V19.4: Add trade uncertainty as a supporting UNDER factor (now correctly triggered)
    # Newly traded players tend to underperform in new environment
    if trade_uncertainty_active and new_team_games < 10:
        uncertainty_weight = max(5, 25 - (new_team_games * 2))  # 25 at 0 games, 5 at 10
        under_score.total_score += uncertainty_weight
        under_score.factor_count += 1
        under_score.factors["trade_role_uncertainty"] = uncertainty_weight
        under_score.reasons.append(
            f"Traded player ({new_team_games} new-team games) — role/chemistry uncertainty"
        )
    
    # V19.4: Player-specific tanking team signal — add UNDER when player on tanking team
    # This catches star players being selectively benched, not just team-level flags
    if (trade_ctx.tank_result and trade_ctx.tank_result.is_tanking and
            game_date >= TRADE_DEADLINE_DATE):
        tank_conf = trade_ctx.tank_result.confidence
        tank_minutes_factor = trade_ctx.tank_result.star_minutes_factor
        
        # Only generate Tank UNDER factor when we're confident about tanking
        if tank_conf >= 0.60:
            tank_weight = 20 if tank_conf >= 0.80 else (12 if tank_conf >= 0.70 else 8)
            under_score.total_score += tank_weight
            under_score.factor_count += 1
            under_score.factors["player_on_tanking_team"] = tank_weight
            under_score.reasons.append(
                f"Player on tanking team {stats.team_abbrev} "
                f"(confidence: {tank_conf:.0%}, minutes_factor: {tank_minutes_factor:.2f})"
            )

    # V19.1: Add trade deadline factors to UNDER score
    if trade_ctx.has_any_impact:
        trade_under_score, trade_under_count, trade_under_reasons = (
            get_trade_factor_for_under(trade_ctx, pt)
        )
        if trade_under_score > 0:
            under_score.total_score += trade_under_score
            under_score.factor_count += trade_under_count
            for reason in trade_under_reasons:
                under_score.factors[f"trade_{reason[:30]}"] = trade_under_score / max(trade_under_count, 1)
    
    if under_score.total_score >= config.min_factor_score_standard:
        # V19: Check multi-factor requirement
        if config.require_multiple_factors_under:
            if under_score.factor_count < config.min_factors_required_under:
                pass  # Skip - not enough factors
            else:
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
                    # Create pick
                    pick = _create_under_pick(
                        stats, pt, opponent_abbrev, game_date, line_info,
                        projection, proj_adj, edge, under_score, defense, b2b,
                        game_context, config
                    )
                    picks.append(pick)
        else:
            # No multi-factor requirement (legacy)
            proj_adj = projection * under_score.projection_adjustment
            edge = calculate_edge(proj_adj, line_info.line, "UNDER")
            min_edge = config.min_edge_sportsbook if line_info.is_sportsbook else config.min_edge_derived
            
            if pt == 'reb' and under_score.total_score < config.reb_under_min_factor_score:
                pass
            elif edge >= min_edge:
                pick = _create_under_pick(
                    stats, pt, opponent_abbrev, game_date, line_info,
                    projection, proj_adj, edge, under_score, defense, b2b,
                    game_context, config
                )
                picks.append(pick)
    
    # =========================================================================
    # Evaluate OVER (STRICTER in V19)
    # =========================================================================
    over_score = calculate_holistic_factor_score_over(
        stats, pt, defense, b2b, injury_impact, game_context
    )
    
    # V19.1: Add trade deadline factors to OVER score
    if trade_ctx.has_any_impact:
        trade_over_score, trade_over_count, trade_over_reasons = (
            get_trade_factor_for_over(trade_ctx, pt)
        )
        if trade_over_score > 0:
            over_score.total_score += trade_over_score
            over_score.factor_count += trade_over_count
            for reason in trade_over_reasons:
                over_score.factors[f"trade_{reason[:30]}"] = trade_over_score / max(trade_over_count, 1)
    
    # V19.3: Opponent tanking boost — playing against tanking teams = easier scoring
    opp_tank_score, opp_tank_count, opp_tank_reasons = get_opponent_tank_boost(
        conn, opponent_abbrev, game_date
    )
    if opp_tank_score > 0:
        over_score.total_score += opp_tank_score
        over_score.factor_count += opp_tank_count
        for reason in opp_tank_reasons:
            over_score.factors[f"opp_tank_{reason[:30]}"] = opp_tank_score / max(opp_tank_count, 1)
            over_score.reasons.append(reason)
    
    # V19.4 FIX: Traded players with few new-team games — block OVER picks
    # Not enough data to trust OVER projections for new environment
    # (previously used stats.new_team_games which was never set — now uses trade_ctx)
    if trade_uncertainty_active and new_team_games < 8:
        # Block OVER — too uncertain about upside in new role
        return picks  # Return any UNDER picks already collected

    # V19: Stricter thresholds for OVER
    min_over_score = config.min_factor_score_over_standard
    
    if over_score.total_score >= min_over_score:
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
        
        # V19: Check multi-factor requirement
        if config.require_multiple_factors_over:
            if over_score.factor_count < config.min_factors_required_over:
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
                pick = _create_over_pick(
                    stats, pt, opponent_abbrev, game_date, line_info,
                    projection, proj_adj, edge, over_score, defense, b2b,
                    game_context, config
                )
                picks.append(pick)
    
    return picks


def _create_under_pick(
    stats: PlayerStatsV19,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    line_info: LineInfo,
    projection: float,
    proj_adj: float,
    edge: float,
    factor_score: HolisticFactorScore,
    defense: DefenseContextV19,
    b2b: BackToBackInfo,
    game_context: Optional[GameContext],
    config: ModelConfigV19General,
) -> PropPickV19General:
    """Create an UNDER pick."""
    pt = prop_type.lower()
    
    # Determine confidence tier
    tier = factor_score.get_tier()
    
    # Calculate confidence score
    confidence = 68.0
    confidence += min(factor_score.total_score / 3, 15)  # Factor score bonus
    confidence += min(edge / 2, 10)  # Edge bonus
    confidence += factor_score.factor_count * 2  # Multi-factor bonus
    if line_info.is_sportsbook:
        confidence += config.sportsbook_confidence_boost
    if defense.is_elite(pt):
        confidence += 5  # Elite defense bonus
    if b2b.is_second_of_b2b:
        confidence += 4  # B2B bonus
    
    # V19.2: Apply trade confidence discount
    # Recently traded players get reduced confidence based on new-team games
    if hasattr(stats, 'trade_confidence_discount') and stats.trade_confidence_discount < 1.0:
        confidence *= stats.trade_confidence_discount
    
    confidence = min(confidence, 95)  # Cap at 95
    
    # Determine final tier based on confidence
    if confidence >= config.premium_confidence and tier == "PREMIUM":
        final_tier = "PREMIUM"
    elif confidence >= config.high_confidence and tier in ["PREMIUM", "HIGH"]:
        final_tier = "HIGH"
    else:
        final_tier = "STANDARD"
    
    # V19.2: Trade-limited players cannot be PREMIUM
    if hasattr(stats, 'was_traded') and stats.was_traded and stats.new_team_games < 10:
        if final_tier == "PREMIUM":
            final_tier = "HIGH"
    
    return PropPickV19General(
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
        factor_score=factor_score.total_score,
        primary_factor=factor_score.primary_factor,
        secondary_factor=factor_score.secondary_factor,
        factor_count=factor_score.factor_count,
        alignment_score=factor_score.alignment_score,
        active_factors=list(factor_score.factors.keys()),
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
        l5_ts_pct=stats.efficiency.l5_ts_pct,
        consistency_cv=stats.get_cv(pt),
        is_b2b=b2b.is_second_of_b2b,
        is_blowout_risk=game_context.is_blowout_risk if game_context else False,
        historical_games=stats.vs_opponent.games_played if stats.vs_opponent else 0,
        historical_avg=getattr(stats.vs_opponent, f"avg_{pt}", 0) if stats.vs_opponent else 0,
        reasons=factor_score.reasons.copy(),
    )


def _create_over_pick(
    stats: PlayerStatsV19,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    line_info: LineInfo,
    projection: float,
    proj_adj: float,
    edge: float,
    factor_score: HolisticFactorScore,
    defense: DefenseContextV19,
    b2b: BackToBackInfo,
    game_context: Optional[GameContext],
    config: ModelConfigV19General,
) -> PropPickV19General:
    """Create an OVER pick."""
    pt = prop_type.lower()
    
    # Determine confidence tier (lower for OVERs)
    tier = factor_score.get_tier()
    
    # Calculate confidence (lower base for OVERs)
    confidence = 62.0
    confidence += min(factor_score.total_score / 3, 12)
    confidence += min(edge / 2, 8)
    confidence += factor_score.factor_count * 2
    if line_info.is_sportsbook:
        confidence += config.sportsbook_confidence_boost
    if "cold_bounce" in factor_score.factors:
        confidence += 8  # Cold bounce is reliable
    
    # V19.2: Apply trade confidence discount
    if hasattr(stats, 'trade_confidence_discount') and stats.trade_confidence_discount < 1.0:
        confidence *= stats.trade_confidence_discount
    
    confidence = min(confidence, 90)  # Cap lower for OVERs
    
    # Determine final tier
    if confidence >= config.premium_confidence and tier == "PREMIUM":
        final_tier = "PREMIUM"
    elif confidence >= config.high_confidence and tier in ["PREMIUM", "HIGH"]:
        final_tier = "HIGH"
    else:
        final_tier = "STANDARD"
    
    # V19.2: Trade-limited players cannot be PREMIUM
    if hasattr(stats, 'was_traded') and stats.was_traded and stats.new_team_games < 10:
        if final_tier == "PREMIUM":
            final_tier = "HIGH"
    
    return PropPickV19General(
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
        factor_score=factor_score.total_score,
        primary_factor=factor_score.primary_factor,
        secondary_factor=factor_score.secondary_factor,
        factor_count=factor_score.factor_count,
        alignment_score=factor_score.alignment_score,
        active_factors=list(factor_score.factors.keys()),
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
        l5_ts_pct=stats.efficiency.l5_ts_pct,
        consistency_cv=stats.get_cv(pt),
        is_b2b=b2b.is_second_of_b2b,
        is_blowout_risk=game_context.is_blowout_risk if game_context else False,
        historical_games=stats.vs_opponent.games_played if stats.vs_opponent else 0,
        historical_avg=getattr(stats.vs_opponent, f"avg_{pt}", 0) if stats.vs_opponent else 0,
        reasons=factor_score.reasons.copy(),
    )


# ============================================================================
# Daily Picks Generation
# ============================================================================

def get_daily_picks_v19_general(
    game_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV19General] = None,
) -> DailyPicksV19General:
    """
    Generate picks for a specific date using Model V19 General.
    
    Args:
        game_date: Date to generate picks for (YYYY-MM-DD)
        db_path: Path to database (optional, uses default)
        config: Model configuration (optional, uses defaults)
    
    Returns:
        DailyPicksV19General with all picks for the date
    """
    if config is None:
        config = ModelConfigV19General()
    
    if db_path is None:
        db_path = get_paths().db_path
    
    db = Db(path=db_path)
    
    with db.connect() as conn:
        # Get games for the date
        games = get_games_for_date(conn, game_date)
        
        result = DailyPicksV19General(
            date=game_date,
            games=len(games),
            config=config,
        )
        
        if not games:
            return result
        
        # Get injured players
        injured_set = get_injured_players(conn, game_date)
        
        all_picks = []
        
        for game in games:
            team1_abbrev = abbrev_from_team_name(game["team1_name"]) or "UNK"
            team2_abbrev = abbrev_from_team_name(game["team2_name"]) or "UNK"
            
            # Get game context
            game_context = get_game_context(conn, team1_abbrev, team2_abbrev, game_date)
            
            # Process both teams
            for team_abbrev, opp_abbrev in [(team1_abbrev, team2_abbrev), (team2_abbrev, team1_abbrev)]:
                # Get players for this team
                player_ids = get_players_in_game(
                    conn, team_abbrev, game_date,
                    min_games=config.min_games_required // 2,  # Lower bar for player selection
                    min_avg_minutes=15.0  # Lower than config for selection
                )
                
                for player_id in player_ids:
                    if player_id in injured_set:
                        continue
                    
                    # V19.1: Skip recently-traded players with no new-team data
                    skip, skip_reason = should_skip_player(
                        conn, player_id, "", team_abbrev, game_date
                    )
                    if skip:
                        continue
                    
                    # Load player stats
                    stats = load_player_stats(
                        conn, player_id, game_date, opp_abbrev,
                        min_games=config.min_games_required,
                        min_minutes=config.min_avg_minutes,
                        max_games=config.max_games_lookback,
                    )
                    
                    if not stats:
                        continue
                    
                    result.players_analyzed += 1
                    
                    # Check if player has sportsbook lines
                    for pt in config.prop_types:
                        sb_line = get_sportsbook_line(conn, player_id, stats.player_name, pt, game_date)
                        if sb_line:
                            result.players_with_sportsbook_lines += 1
                            break
                    else:
                        result.players_with_derived_lines += 1
                    
                    # Evaluate for each prop type
                    for pt in config.prop_types:
                        if pt.lower() == 'ast' and not config.include_ast:
                            continue
                        
                        picks = evaluate_player_for_prop(
                            conn, stats, pt, opp_abbrev, game_date, config, game_context
                        )
                        all_picks.extend(picks)
        
        # Select best picks
        result.picks = _select_best_picks(all_picks, config)
        
        return result


def _select_best_picks(
    all_picks: List[PropPickV19General],
    config: ModelConfigV19General,
) -> List[PropPickV19General]:
    """
    Select the best picks from all candidates.
    
    Rules:
    - 1 pick per player (best one)
    - Max picks per day
    - Sort by factor score * edge
    """
    # Group by player
    by_player: Dict[int, List[PropPickV19General]] = {}
    for pick in all_picks:
        if pick.player_id not in by_player:
            by_player[pick.player_id] = []
        by_player[pick.player_id].append(pick)
    
    # Select best per player
    selected = []
    for player_id, player_picks in by_player.items():
        # Sort by composite score
        player_picks.sort(
            key=lambda p: (p.factor_score * 0.6 + p.edge_pct * 0.4 + p.factor_count * 5),
            reverse=True
        )
        # Take top N for this player
        selected.extend(player_picks[:config.max_picks_per_player])
    
    # Sort all selected by confidence
    selected.sort(
        key=lambda p: (p.factor_score * 0.5 + p.edge_pct * 0.3 + p.confidence_score * 0.2),
        reverse=True
    )
    
    # Apply daily limit
    return selected[:config.max_picks_per_day]


# ============================================================================
# Backtesting
# ============================================================================

def run_backtest_v19_general(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV19General] = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> BacktestResultV19:
    """
    Run a comprehensive backtest of Model V19 General.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database (optional)
        config: Model configuration (optional)
        verbose: Print detailed output
        show_progress: Show progress bar
    
    Returns:
        BacktestResultV19 with comprehensive results
    """
    start_time = time.time()
    
    if config is None:
        config = ModelConfigV19General()
    
    if db_path is None:
        db_path = get_paths().db_path
    
    db = Db(path=db_path)
    
    result = BacktestResultV19(
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
        print(f"\nStarting Model V19 General Backtest")
        print(f"Period: {start_date} to {end_date} ({total_dates} days)")
        print(f"Config: multi-factor={config.require_multiple_factors_under}, min_edge_derived={config.min_edge_derived}%")
        print("")
    
    with db.connect() as conn:
        for i, game_date in enumerate(dates):
            # Progress bar
            if show_progress:
                elapsed = time.time() - start_time
                if i > 0:
                    remaining = (elapsed / i) * (total_dates - i)
                    suffix = f"| {format_time_remaining(remaining)} remaining"
                else:
                    suffix = ""
                print_progress_bar(i + 1, total_dates, prefix='Backtesting:', suffix=suffix, length=40)
            
            # Get games for date
            games = get_games_for_date(conn, game_date)
            if not games:
                continue
            
            result.days_tested += 1
            result.games_tested += len(games)
            
            # Generate picks for this date
            daily = _get_picks_for_backtest_date(conn, game_date, config)
            
            # Grade each pick
            for pick in daily.picks:
                actual = get_actual_stats(conn, pick.player_id, game_date)
                if actual is None:
                    continue
                
                prop_key = pick.prop_type.lower()
                actual_value = actual.get(prop_key, 0)
                hit, margin = grade_pick(actual_value, pick.line, pick.direction)
                
                pick.actual_value = actual_value
                pick.hit = hit
                pick.margin = margin
                
                # Update result counters
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
                if pick.prop_type.upper() == "PTS":
                    result.pts_picks += 1
                    if hit:
                        result.pts_hits += 1
                    # By prop + direction
                    if pick.direction == "OVER":
                        result.pts_over_picks += 1
                        if hit:
                            result.pts_over_hits += 1
                    else:
                        result.pts_under_picks += 1
                        if hit:
                            result.pts_under_hits += 1
                else:
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
                
                # By primary factor
                factor = pick.primary_factor
                if factor not in result.by_factor:
                    result.by_factor[factor] = {'total': 0, 'hits': 0}
                result.by_factor[factor]['total'] += 1
                if hit:
                    result.by_factor[factor]['hits'] += 1
                
                # By factor count (V19)
                fc = pick.factor_count
                if fc not in result.by_factor_count:
                    result.by_factor_count[fc] = {'total': 0, 'hits': 0}
                result.by_factor_count[fc]['total'] += 1
                if hit:
                    result.by_factor_count[fc]['hits'] += 1
                
                # By score range
                score = pick.factor_score
                if score >= 80:
                    range_key = "80+"
                elif score >= 65:
                    range_key = "65-79"
                elif score >= 50:
                    range_key = "50-64"
                elif score >= 40:
                    range_key = "40-49"
                else:
                    range_key = "35-39"
                
                if range_key not in result.by_score_range:
                    result.by_score_range[range_key] = {'total': 0, 'hits': 0}
                result.by_score_range[range_key]['total'] += 1
                if hit:
                    result.by_score_range[range_key]['hits'] += 1
                
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
                else:
                    edge_key = "6-14%"
                
                if edge_key not in result.by_edge_range:
                    result.by_edge_range[edge_key] = {'total': 0, 'hits': 0}
                result.by_edge_range[edge_key]['total'] += 1
                if hit:
                    result.by_edge_range[edge_key]['hits'] += 1
                
                # ROI calculation (assuming -110 odds, $100 wagers)
                result.theoretical_wagers += 100
                if hit:
                    result.theoretical_profit += 90.91  # Win $90.91 on -110
                else:
                    result.theoretical_profit -= 100  # Lose $100
                
                result.all_picks.append(pick)
    
    # Calculate final hit rate
    if result.total_picks > 0:
        result.hit_rate = result.total_hits / result.total_picks
    
    result.runtime_seconds = time.time() - start_time
    
    if verbose:
        print(result.summary())
    
    return result


def _get_picks_for_backtest_date(
    conn: sqlite3.Connection,
    game_date: str,
    config: ModelConfigV19General,
) -> DailyPicksV19General:
    """
    Generate picks for a backtest date.
    """
    # Get games
    games = get_games_for_date(conn, game_date)
    
    result = DailyPicksV19General(
        date=game_date,
        games=len(games),
        config=config,
    )
    
    if not games:
        return result
    
    # Get injured players
    injured_set = get_injured_players(conn, game_date)
    
    all_picks = []
    
    for game in games:
        team1_abbrev = abbrev_from_team_name(game["team1_name"]) or "UNK"
        team2_abbrev = abbrev_from_team_name(game["team2_name"]) or "UNK"
        
        # Get game context
        game_context = get_game_context(conn, team1_abbrev, team2_abbrev, game_date)
        
        for team_abbrev, opp_abbrev in [(team1_abbrev, team2_abbrev), (team2_abbrev, team1_abbrev)]:
            player_ids = get_players_in_game(
                conn, team_abbrev, game_date,
                min_games=config.min_games_required // 2,
                min_avg_minutes=15.0
            )
            
            for player_id in player_ids:
                if player_id in injured_set:
                    continue
                
                # V19.1: Skip recently-traded players with no new-team data
                skip, skip_reason = should_skip_player(
                    conn, player_id, "", team_abbrev, game_date
                )
                if skip:
                    continue
                
                stats = load_player_stats(
                    conn, player_id, game_date, opp_abbrev,
                    min_games=config.min_games_required,
                    min_minutes=config.min_avg_minutes,
                    max_games=config.max_games_lookback,
                )
                
                if not stats:
                    continue
                
                result.players_analyzed += 1
                
                for pt in config.prop_types:
                    if pt.lower() == 'ast' and not config.include_ast:
                        continue
                    
                    picks = evaluate_player_for_prop(
                        conn, stats, pt, opp_abbrev, game_date, config, game_context
                    )
                    all_picks.extend(picks)
    
    result.picks = _select_best_picks(all_picks, config)
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V19 General")
    subparsers = parser.add_subparsers(dest="command")
    
    # Picks command
    picks_parser = subparsers.add_parser("picks", help="Generate daily picks")
    picks_parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    backtest_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    backtest_parser.add_argument("--verbose", "-v", action="store_true")
    backtest_parser.add_argument("--no-progress", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "picks":
        result = get_daily_picks_v19_general(args.date)
        print(result.summary())
    
    elif args.command == "backtest":
        result = run_backtest_v19_general(
            args.start,
            args.end,
            verbose=args.verbose,
            show_progress=not args.no_progress,
        )
    
    else:
        parser.print_help()
