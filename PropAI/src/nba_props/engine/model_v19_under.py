"""
Model V19 Under - Comprehensive Specialized UNDER NBA Props Prediction Model
=============================================================================

This is the SPECIALIZED UNDER model of the V19 dual-model approach:
- Model V19 General: Holistic multi-factor approach for all picks (prefers UNDER)
- Model V19 Under (this file): DEDICATED UNDER-only predictions with maximum precision

=============================================================================
MODEL V19 UNDER - KEY DESIGN PRINCIPLES (Addressing ALL Previous Shortcomings)
=============================================================================

1. **UNDER-ONLY SPECIALIZATION**:
   - UNDER picks are MORE PREDICTABLE than OVER picks
   - Negative factors compound more reliably than positive ones
   - Elite defenses consistently limit player production (68-74% hit rate)
   - Cold streaks alone are RISKY (~48%) - MUST pair with defense

2. **DEFENSE-ANCHORED APPROACH** (Primary Driver):
   - Defense DVP (vs position) is the PRIMARY factor
   - Elite Defense (Rank 1-3): 71-74% hit rate - HIGHEST weight
   - Good Defense (Rank 4-10): 60-67% hit rate - SOLID weight
   - NO UNDER picks vs weak defense (rank 20+) - too risky

3. **MULTI-FACTOR REQUIREMENT** (Avoiding Single-Pattern Trap):
   - Previous models failed with single factors (cold bounce alone)
   - V19 Under requires MINIMUM 2 FACTORS to align
   - Cold streak ALONE: ~48% (DO NOT USE)
   - Cold streak + Elite Defense: 83%+ (USE!)
   - B2B Fatigue + Defense: 75%+ (USE!)

4. **COMPREHENSIVE BOX SCORE ANALYSIS** (Not Just Averages):
   - Plus/Minus (+/-): Negative trend indicates struggling
   - FG% trends: Declining efficiency = UNDER signal
   - True Shooting %: Overall efficiency indicator
   - FTA trends: Less aggressive = fewer points
   - Minutes trends: Role reduction = less production

5. **HYBRID LINE APPROACH** (Always Generate Picks):
   - Use actual sportsbook lines when available (higher confidence)
   - ALWAYS generate picks even without lines (lines come late!)
   - Different edge thresholds:
     * Sportsbook lines: 5% minimum edge
     * Derived lines: 10% minimum edge (stricter due to inaccuracy)
   - Track line source for HONEST reporting

6. **VALIDATED FACTOR WEIGHTS** (From V16-V18 Extensive Backtesting):
   | Factor                  | Weight | Validated Hit Rate |
   |------------------------|--------|-------------------|
   | Elite Defense (Top 3)  | 55     | 71-74% |
   | B2B Fatigue (2nd game) | 45     | 69-75% |
   | Good Defense (Top 10)  | 30     | 60-67% |
   | Injury Rust (1st back) | 28     | 60-70% |
   | Third in Four Days     | 18     | 60-65% |
   | Cold Streak Mild       | 15     | 57-62% |
   | Poor H2H History       | 15     | 55-65% |
   | Minutes Decline        | 12     | 55-60% |
   | Negative +/- Trend     | 10     | ~55% |
   | Poor Efficiency Trend  | 10     | ~55% |
   | High Variance Player   | 8      | ~55% |
   | Cold Streak Severe     | 3      | ~48% (REQUIRES SUPPORT!) |
   | Blowout Risk           | 10     | ~60% |

7. **PROP SELECTION** (Data-Driven):
   - PTS UNDER: PRIMARY focus (63.9% UNDER rate from RCM v1.4)
   - REB UNDER: SECONDARY (52-57%, more volatile)
   - AST UNDER: EXCLUDED (~54% is coin flip after juice)

8. **STRICT QUALITY FILTERING**:
   - 23+ minute average (established players only)
   - 10+ games history required
   - Require defense data for the matchup
   - Minimum combined factor score: 45 (high bar)
   - Maximum defense rank for any pick: 20 (no weak defense UNDERs)

=============================================================================

USAGE:
------
    from src.nba_props.engine.model_v19_under import (
        get_daily_picks_v19_under,
        run_backtest_v19_under,
        ModelConfigV19Under,
    )
    
    # Get UNDER picks for today
    picks = get_daily_picks_v19_under("2026-02-03")
    print(picks.summary())
    
    # Run comprehensive backtest with progress bar
    result = run_backtest_v19_under(
        "2025-10-22", "2026-02-03", 
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
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
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
    detect_cold_streak_pattern,
    grade_pick,
    get_actual_stats,
    print_progress_bar,
    format_time_remaining,
    
    # Constants
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    AVERAGE_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
    FACTOR_PROJECTION_ADJUSTMENTS,
    MODEL_VERSION,
)

# Trade deadline handling
from .post_trade_adjustments import (
    get_trade_context,
    apply_trade_adjustments,
    should_skip_player,
    get_trade_factor_for_under,
    TradeContext,
    TRADE_DEADLINE_DATE,
)


# ============================================================================
# Version Info
# ============================================================================

UNDER_MODEL_VERSION = "19.0"
UNDER_MODEL_NAME = "Model V19 Under"


# ============================================================================
# V19 Under Factor Weights (Optimized for UNDER Prediction)
# ============================================================================

# Primary factors (defense-anchored) - HIGHEST WEIGHTS
# These are validated with 60%+ hit rates
V19_UNDER_FACTOR_WEIGHTS = {
    # =========================================================================
    # TIER 1: PRIMARY UNDER SIGNALS (Defense + Fatigue) - 60-75% hit rates
    # =========================================================================
    "defense_elite": 55,        # Top 3 DVP - 71-74% hit rate - STRONGEST
    "b2b_fatigue": 45,          # Second of B2B - 69-75% hit rate - VERY STRONG
    "defense_good": 30,         # Top 4-10 DVP - 60-67% hit rate
    "injury_rust_first": 28,    # First game back - 60-70% hit rate
    
    # =========================================================================
    # TIER 2: SUPPORTING SIGNALS - 55-62% hit rates
    # =========================================================================
    "third_in_four": 18,        # Third game in 4 days - 60-65%
    "cold_streak_mild": 15,     # L5 < 90% of season - 57-62%
    "poor_h2h_history": 15,     # Below avg vs opponent (3+ games) - 55-65%
    "minutes_decline": 12,      # Minutes declining by 10%+ - 55-60%
    "injury_rust_second": 12,   # Second game back - 55-62%
    
    # =========================================================================
    # TIER 3: BOX SCORE SIGNALS (V19 Enhanced) - ~55% hit rates
    # =========================================================================
    "negative_plus_minus": 10,  # L5 avg +/- < -5 - NEW
    "poor_efficiency_trend": 10,  # FG% declining by 5%+ - NEW
    "poor_ts_trend": 8,         # True Shooting % declining - NEW
    "low_fta_trend": 6,         # FTA declining (less aggressive) - NEW
    
    # =========================================================================
    # TIER 4: PLAYER PROFILE + CONTEXT - ~55% hit rates
    # =========================================================================
    "high_variance": 8,         # CV > 0.40 (inconsistent player)
    "blowout_risk": 10,         # Large spread (>10pts) - garbage time
    "pace_factor_slow": 6,      # Opponent plays slow pace
    "defense_average": 5,       # Top 11-15 DVP (minimal value)
    
    # =========================================================================
    # TIER 5: RISKY ALONE (REQUIRES SUPPORT) - <55% hit rates
    # =========================================================================
    "cold_streak_severe": 3,    # L5 < 80% of season - 48% ALONE (DANGEROUS!)
    # ^^ This factor is ONLY useful when combined with defense!
}

# Projection adjustments (multipliers) - values < 1.0 reduce projection
V19_UNDER_ADJUSTMENTS = {
    # Primary factors - larger adjustments
    "defense_elite": 0.84,      # -16% (STRONGER)
    "defense_good": 0.92,       # -8%
    "defense_average": 0.97,    # -3%
    "b2b_fatigue": 0.92,        # -8% (STRONGER - fatigue is real)
    "third_in_four": 0.95,      # -5%
    
    # Cold streak - be careful!
    "cold_streak_mild": 0.95,   # -5%
    "cold_streak_severe": 0.92, # -8% (but risky alone!)
    
    # Injury/rust
    "injury_rust_first": 0.80,  # -20% (first game is rough)
    "injury_rust_second": 0.90, # -10%
    
    # Box score factors
    "negative_plus_minus": 0.97, # -3%
    "poor_efficiency_trend": 0.97, # -3%
    "poor_ts_trend": 0.97,      # -3%
    "low_fta_trend": 0.98,      # -2%
    
    # Context factors
    "minutes_decline": 0.94,    # -6%
    "high_variance": 0.98,      # -2%
    "poor_h2h_history": 0.94,   # -6%
    "blowout_risk": 0.93,       # -7%
    "pace_factor_slow": 0.97,   # -3%
}


# ============================================================================
# V19 Under Thresholds (Calibrated from backtesting)
# ============================================================================

# Factor score thresholds for tiers
V19_UNDER_PREMIUM_THRESHOLD = 70    # Requires multiple strong factors
V19_UNDER_HIGH_THRESHOLD = 55       # Strong signals
V19_UNDER_STANDARD_THRESHOLD = 45   # Minimum for any pick

# Multi-factor requirements
V19_UNDER_MIN_FACTORS = 2           # Require at least 2 factors
V19_UNDER_MIN_STRONG_FACTOR = 20    # At least one factor with weight >= 20

# Edge requirements (different for line source)
V19_UNDER_EDGE_SPORTSBOOK = 5.0     # 5% edge for real lines
V19_UNDER_EDGE_DERIVED = 10.0       # 10% edge for derived (stricter)
V19_UNDER_EDGE_PREMIUM = 15.0       # 15% for premium picks

# Defense requirements
V19_UNDER_MAX_DEFENSE_RANK = 20     # No UNDER vs bottom 10 defenses


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV19Under:
    """
    Model V19 Under Configuration.
    
    SPECIALIZED for UNDER-only picks with focus on:
    - Defense as primary factor
    - Multi-factor requirements
    - Comprehensive box score analysis
    - Hybrid line handling
    """
    # === VERSION INFO ===
    model_name: str = UNDER_MODEL_NAME
    model_version: str = UNDER_MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False  # ALWAYS generate picks
    derived_line_adjustment: float = 1.05  # +5% for derived lines
    sportsbook_confidence_boost: float = 12.0  # Higher confidence with real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Established players only
    max_games_lookback: int = 25
    
    # === PROJECTION WEIGHTS ===
    weight_l3: float = 0.10
    weight_l5: float = 0.20
    weight_l10: float = 0.30
    weight_l15: float = 0.20
    weight_season: float = 0.20
    
    # === FACTOR SCORE THRESHOLDS ===
    min_factor_score_premium: float = V19_UNDER_PREMIUM_THRESHOLD
    min_factor_score_high: float = V19_UNDER_HIGH_THRESHOLD
    min_factor_score_standard: float = V19_UNDER_STANDARD_THRESHOLD
    
    # === MULTI-FACTOR REQUIREMENTS (KEY V19 INNOVATION) ===
    require_multiple_factors: bool = True
    min_factors_required: int = V19_UNDER_MIN_FACTORS
    min_strong_factor_weight: float = V19_UNDER_MIN_STRONG_FACTOR
    
    # === EDGE REQUIREMENTS ===
    min_edge_sportsbook: float = V19_UNDER_EDGE_SPORTSBOOK
    min_edge_derived: float = V19_UNDER_EDGE_DERIVED
    min_edge_premium: float = V19_UNDER_EDGE_PREMIUM
    
    # === DEFENSE REQUIREMENTS ===
    require_defense_data: bool = True
    max_defense_rank_for_under: int = V19_UNDER_MAX_DEFENSE_RANK
    
    # === COLD STREAK THRESHOLDS ===
    cold_streak_mild_threshold: float = -10.0   # L5 is 10%+ below season
    cold_streak_severe_threshold: float = -20.0  # L5 is 20%+ below season
    
    # === EFFICIENCY THRESHOLDS (V19 NEW) ===
    efficiency_decline_threshold: float = -5.0  # FG% decline by 5%+
    ts_decline_threshold: float = -5.0          # TS% decline by 5%+
    fta_decline_threshold: float = -15.0        # FTA decline by 15%+
    plus_minus_negative_threshold: float = -5.0 # L5 avg +/- < -5
    
    # === VARIANCE THRESHOLDS ===
    high_variance_cv_threshold: float = 0.40    # CV > 0.40 = inconsistent
    
    # === MINUTES THRESHOLDS ===
    minutes_decline_threshold: float = -10.0    # L5 min decline by 10%+
    
    # === HISTORICAL MATCHUP ===
    h2h_min_games: int = 3                      # Min games for H2H data
    h2h_poor_threshold: float = -10.0           # Below avg by 10%+
    
    # === BLOWOUT RISK ===
    blowout_spread_threshold: float = 10.0      # Spread > 10 = blowout risk
    
    # === PACE FACTOR ===
    slow_pace_ou_threshold: float = 215.0       # O/U < 215 = slow pace
    
    # === PROP SELECTION ===
    include_pts_under: bool = True   # PRIMARY - 63.9% UNDER rate
    include_reb_under: bool = True   # SECONDARY - more volatile
    include_ast_under: bool = False  # EXCLUDED - coin flip
    
    # REB UNDER requires additional checks
    reb_under_require_elite_defense: bool = True
    reb_under_min_factor_score: float = 60.0    # Higher bar for REB
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 4
    max_picks_per_day: int = 25
    max_picks_per_player: int = 1  # One prop per player
    
    # === CONFIDENCE MAPPING ===
    premium_confidence: float = 88.0
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
class UnderFactor:
    """A single factor contributing to an UNDER pick."""
    name: str
    weight: float
    adjustment: float
    reason: str
    tier: str  # "primary", "supporting", "boxscore", "context", "risky"


@dataclass
class PropPickV19Under:
    """A pick generated by Model V19 Under."""
    # === Identity ===
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    position: str
    
    # === Pick Details (Always UNDER) ===
    prop_type: str  # PTS, REB
    direction: str = "UNDER"
    
    # === Line Information ===
    line: float = 0.0
    line_source: str = "derived"  # "sportsbook" or "derived"
    book: Optional[str] = None
    
    # === Projection ===
    base_projection: float = 0.0       # Before factor adjustments
    adjusted_projection: float = 0.0    # After factor adjustments
    total_adjustment: float = 1.0       # Combined adjustment factor
    
    # === Edge Calculation ===
    edge_pct: float = 0.0
    
    # === Factor Scoring ===
    factor_score: float = 0.0
    factor_count: int = 0
    factors: List[UnderFactor] = field(default_factory=list)
    primary_factor: str = ""
    secondary_factor: str = ""
    
    # === Confidence ===
    confidence_score: float = 0.0
    confidence_tier: str = "STANDARD"  # PREMIUM, HIGH, STANDARD
    
    # === Defense Context ===
    defense_rank: int = 15
    defense_rating: str = "average"
    
    # === Fatigue Context ===
    is_b2b: bool = False
    is_third_in_four: bool = False
    
    # === Cold Streak Context ===
    cold_streak_severity: str = "none"  # none, mild, severe
    
    # === Box Score Metrics (V19 NEW) ===
    l5_plus_minus: float = 0.0
    l5_fg_pct: float = 0.0
    l5_ts_pct: float = 0.0
    fg_trend: float = 0.0  # L5 vs L15 FG% change
    
    # === Stats for Display ===
    l3_avg: float = 0.0
    l5_avg: float = 0.0
    l10_avg: float = 0.0
    l15_avg: float = 0.0
    season_avg: float = 0.0
    
    # === Historical Matchup ===
    h2h_games: int = 0
    h2h_avg: float = 0.0
    h2h_vs_season_pct: float = 0.0
    
    # === Reasoning ===
    reasons: List[str] = field(default_factory=list)
    
    # === Results (filled after game for backtesting) ===
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    margin: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "player": self.player_name,
            "team": self.team_abbrev,
            "opponent": self.opponent_abbrev,
            "position": self.position,
            "date": self.game_date,
            "prop": self.prop_type.upper(),
            "direction": "UNDER",
            "line": round(self.line, 1),
            "line_source": self.line_source,
            "book": self.book,
            "base_projection": round(self.base_projection, 1),
            "adj_projection": round(self.adjusted_projection, 1),
            "total_reduction": f"{(1 - self.total_adjustment) * 100:.1f}%",
            "edge": f"{self.edge_pct:.1f}%",
            "factor_score": round(self.factor_score, 1),
            "factor_count": self.factor_count,
            "primary_factor": self.primary_factor,
            "secondary_factor": self.secondary_factor,
            "all_factors": [f.name for f in self.factors],
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "b2b": self.is_b2b,
            "cold_streak": self.cold_streak_severity,
            "l5_plus_minus": round(self.l5_plus_minus, 1),
            "l5_fg_pct": round(self.l5_fg_pct * 100, 1) if self.l5_fg_pct else 0,
            "fg_trend": f"{self.fg_trend:.1f}%",
            "l3": round(self.l3_avg, 1),
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "l15": round(self.l15_avg, 1),
            "season": round(self.season_avg, 1),
            "h2h": f"{self.h2h_games} games, {self.h2h_avg:.1f} avg ({self.h2h_vs_season_pct:+.0f}%)" if self.h2h_games >= 3 else "N/A",
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }
    
    def summary_line(self) -> str:
        """One-line summary for display."""
        factors_str = ", ".join(f.name for f in self.factors[:3])
        src = f"[{self.book}]" if self.line_source == "sportsbook" else "[derived]"
        return (
            f"📉 {self.player_name} ({self.team_abbrev} vs {self.opponent_abbrev}) - "
            f"{self.prop_type.upper()} UNDER {self.line:.1f} {src} | "
            f"Proj: {self.adjusted_projection:.1f} | Edge: {self.edge_pct:.1f}% | "
            f"Score: {self.factor_score:.0f} ({self.confidence_tier}) | "
            f"Factors: {factors_str}"
        )


@dataclass
class DailyPicksV19Under:
    """All UNDER picks for a day from Model V19 Under."""
    date: str
    games: int
    config: ModelConfigV19Under = field(default_factory=ModelConfigV19Under)
    picks: List[PropPickV19Under] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    players_filtered_no_defense: int = 0
    players_filtered_weak_defense: int = 0
    players_filtered_low_score: int = 0
    players_filtered_single_factor: int = 0
    
    # Defense data status
    defense_data_available: bool = True
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV19Under]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV19Under]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def premium_picks(self) -> List[PropPickV19Under]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV19Under]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    @property
    def pts_picks(self) -> List[PropPickV19Under]:
        return [p for p in self.picks if p.prop_type.upper() == "PTS"]
    
    @property
    def reb_picks(self) -> List[PropPickV19Under]:
        return [p for p in self.picks if p.prop_type.upper() == "REB"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*75}",
            f"MODEL V19 UNDER PICKS - {self.date}",
            f"{'='*75}",
            f"Games: {self.games} | Players analyzed: {self.players_analyzed}",
            f"",
            f"LINE COVERAGE:",
            f"  Sportsbook lines available: {self.players_with_sportsbook_lines}",
            f"  Using derived lines: {self.players_with_derived_lines}",
            f"",
            f"FILTERING SUMMARY:",
            f"  Filtered (no defense data): {self.players_filtered_no_defense}",
            f"  Filtered (weak defense): {self.players_filtered_weak_defense}",
            f"  Filtered (low score): {self.players_filtered_low_score}",
            f"  Filtered (single factor): {self.players_filtered_single_factor}",
            f"",
            f"PICKS SUMMARY:",
            f"  Total UNDER picks: {self.total_picks}",
            f"  PREMIUM: {len(self.premium_picks)} | HIGH: {len(self.high_picks)}",
            f"  PTS UNDER: {len(self.pts_picks)} | REB UNDER: {len(self.reb_picks)}",
            f"  Sportsbook: {len(self.sportsbook_picks)} | Derived: {len(self.derived_picks)}",
            "",
        ]
        
        # Group by tier
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in sorted(tier_picks, key=lambda x: x.factor_score, reverse=True):
                    lines.append(p.summary_line())
                lines.append("")
        
        lines.append(f"{'='*75}")
        return "\n".join(lines)


@dataclass
class BacktestResultV19Under:
    """Comprehensive backtest results for Model V19 Under."""
    start_date: str
    end_date: str
    config: ModelConfigV19Under = field(default_factory=ModelConfigV19Under)
    
    # === Overall ===
    total_picks: int = 0
    hits: int = 0
    
    # === By Line Source (HONEST REPORTING) ===
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    derived_picks: int = 0
    derived_hits: int = 0
    
    # === By Confidence Tier ===
    premium_picks: int = 0
    premium_hits: int = 0
    high_picks: int = 0
    high_hits: int = 0
    standard_picks: int = 0
    standard_hits: int = 0
    
    # === By Prop Type ===
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    
    # === By Primary Factor ===
    by_primary_factor: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # === By Factor Combination ===
    by_factor_combo: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # === By Factor Score Range ===
    score_70_plus_picks: int = 0
    score_70_plus_hits: int = 0
    score_55_70_picks: int = 0
    score_55_70_hits: int = 0
    score_45_55_picks: int = 0
    score_45_55_hits: int = 0
    
    # === By Edge Range ===
    edge_15_plus_picks: int = 0
    edge_15_plus_hits: int = 0
    edge_10_15_picks: int = 0
    edge_10_15_hits: int = 0
    edge_5_10_picks: int = 0
    edge_5_10_hits: int = 0
    
    # === By Defense Rating ===
    elite_defense_picks: int = 0
    elite_defense_hits: int = 0
    good_defense_picks: int = 0
    good_defense_hits: int = 0
    average_defense_picks: int = 0
    average_defense_hits: int = 0
    
    # === By B2B Status ===
    b2b_picks: int = 0
    b2b_hits: int = 0
    non_b2b_picks: int = 0
    non_b2b_hits: int = 0
    
    # === Coverage ===
    days_tested: int = 0
    total_games: int = 0
    
    # === All picks for detailed analysis ===
    all_picks: List[PropPickV19Under] = field(default_factory=list)
    daily_results: List[Dict] = field(default_factory=list)
    
    # === ROI calculation ===
    theoretical_profit: float = 0.0
    theoretical_wagers: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_picks * 100 if self.total_picks > 0 else 0.0
    
    @property
    def sportsbook_rate(self) -> float:
        return self.sportsbook_hits / self.sportsbook_picks * 100 if self.sportsbook_picks > 0 else 0.0
    
    @property
    def derived_rate(self) -> float:
        return self.derived_hits / self.derived_picks * 100 if self.derived_picks > 0 else 0.0
    
    @property
    def theoretical_roi(self) -> float:
        if self.theoretical_wagers == 0:
            return 0.0
        return (self.theoretical_profit / self.theoretical_wagers) * 100
    
    def _pct(self, hits: int, total: int) -> str:
        if total == 0:
            return "N/A"
        return f"{hits/total*100:.1f}%"
    
    def summary(self) -> str:
        lines = [
            "",
            "=" * 80,
            "MODEL V19 UNDER - COMPREHENSIVE BACKTEST RESULTS",
            "=" * 80,
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"Avg picks/day: {self.total_picks / max(self.days_tested, 1):.1f}",
            "",
            f"OVERALL HIT RATE: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            "",
            "BY LINE SOURCE (Honest Reporting):",
            f"  Sportsbook: {self._pct(self.sportsbook_hits, self.sportsbook_picks)} ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived:    {self._pct(self.derived_hits, self.derived_picks)} ({self.derived_hits}/{self.derived_picks})",
            "",
            "BY CONFIDENCE TIER:",
            f"  PREMIUM (score ≥70): {self._pct(self.premium_hits, self.premium_picks)} ({self.premium_hits}/{self.premium_picks})",
            f"  HIGH (score 55-69):  {self._pct(self.high_hits, self.high_picks)} ({self.high_hits}/{self.high_picks})",
            f"  STANDARD (score 45-54): {self._pct(self.standard_hits, self.standard_picks)} ({self.standard_hits}/{self.standard_picks})",
            "",
            "BY PROP TYPE:",
            f"  PTS UNDER: {self._pct(self.pts_hits, self.pts_picks)} ({self.pts_hits}/{self.pts_picks})",
            f"  REB UNDER: {self._pct(self.reb_hits, self.reb_picks)} ({self.reb_hits}/{self.reb_picks})",
            "",
            "BY FACTOR SCORE RANGE:",
            f"  Score ≥70:  {self._pct(self.score_70_plus_hits, self.score_70_plus_picks)} ({self.score_70_plus_hits}/{self.score_70_plus_picks})",
            f"  Score 55-69: {self._pct(self.score_55_70_hits, self.score_55_70_picks)} ({self.score_55_70_hits}/{self.score_55_70_picks})",
            f"  Score 45-54: {self._pct(self.score_45_55_hits, self.score_45_55_picks)} ({self.score_45_55_hits}/{self.score_45_55_picks})",
            "",
            "BY EDGE RANGE:",
            f"  Edge ≥15%:  {self._pct(self.edge_15_plus_hits, self.edge_15_plus_picks)} ({self.edge_15_plus_hits}/{self.edge_15_plus_picks})",
            f"  Edge 10-15%: {self._pct(self.edge_10_15_hits, self.edge_10_15_picks)} ({self.edge_10_15_hits}/{self.edge_10_15_picks})",
            f"  Edge 5-10%: {self._pct(self.edge_5_10_hits, self.edge_5_10_picks)} ({self.edge_5_10_hits}/{self.edge_5_10_picks})",
            "",
            "BY DEFENSE RATING:",
            f"  Elite (Top 3):   {self._pct(self.elite_defense_hits, self.elite_defense_picks)} ({self.elite_defense_hits}/{self.elite_defense_picks})",
            f"  Good (Top 10):   {self._pct(self.good_defense_hits, self.good_defense_picks)} ({self.good_defense_hits}/{self.good_defense_picks})",
            f"  Average (11-20): {self._pct(self.average_defense_hits, self.average_defense_picks)} ({self.average_defense_hits}/{self.average_defense_picks})",
            "",
            "BY B2B STATUS:",
            f"  B2B Games:     {self._pct(self.b2b_hits, self.b2b_picks)} ({self.b2b_hits}/{self.b2b_picks})",
            f"  Non-B2B Games: {self._pct(self.non_b2b_hits, self.non_b2b_picks)} ({self.non_b2b_hits}/{self.non_b2b_picks})",
            "",
        ]
        
        # Primary Factor breakdown
        if self.by_primary_factor:
            lines.append("BY PRIMARY FACTOR:")
            sorted_factors = sorted(
                self.by_primary_factor.items(),
                key=lambda x: x[1].get('total', 0),
                reverse=True
            )
            for factor, data in sorted_factors[:10]:
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total > 0:
                    lines.append(f"  {factor}: {self._pct(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # Factor combination breakdown
        if self.by_factor_combo:
            lines.append("BY TOP FACTOR COMBINATIONS:")
            sorted_combos = sorted(
                self.by_factor_combo.items(),
                key=lambda x: x[1].get('total', 0),
                reverse=True
            )
            for combo, data in sorted_combos[:8]:
                total = data.get('total', 0)
                hits = data.get('hits', 0)
                if total >= 5:  # Only show combos with 5+ picks
                    lines.append(f"  {combo}: {self._pct(hits, total)} ({hits}/{total})")
            lines.append("")
        
        # ROI
        lines.append(f"THEORETICAL ROI: {self.theoretical_roi:.1f}%")
        lines.append("(Assuming $100 wagers at -110 odds)")
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# ============================================================================
# Core Factor Calculation
# ============================================================================

def calculate_v19_under_factors(
    stats: PlayerStatsV19,
    prop_type: str,
    defense: DefenseContextV19,
    b2b: BackToBackInfo,
    game_context: GameContext,
    config: ModelConfigV19Under,
) -> Tuple[List[UnderFactor], float, float]:
    """
    Calculate all applicable UNDER factors for V19.
    
    Returns: (factors, total_score, total_adjustment)
    """
    factors = []
    pt = prop_type.lower()
    
    # =========================================================================
    # TIER 1: PRIMARY SIGNALS - Defense + Fatigue (60-75% hit rates)
    # =========================================================================
    
    # === Defense Factors (PRIMARY DRIVER) ===
    if defense.data_available:
        defense_rank = defense.get_rank(pt)
        
        if defense_rank <= ELITE_DEFENSE_RANK:
            factors.append(UnderFactor(
                name="defense_elite",
                weight=V19_UNDER_FACTOR_WEIGHTS["defense_elite"],
                adjustment=V19_UNDER_ADJUSTMENTS["defense_elite"],
                reason=f"ELITE defense: {defense.team_abbrev} ranks #{defense_rank} vs {pt.upper()}",
                tier="primary"
            ))
        elif defense_rank <= GOOD_DEFENSE_RANK:
            factors.append(UnderFactor(
                name="defense_good",
                weight=V19_UNDER_FACTOR_WEIGHTS["defense_good"],
                adjustment=V19_UNDER_ADJUSTMENTS["defense_good"],
                reason=f"Good defense: {defense.team_abbrev} ranks #{defense_rank} vs {pt.upper()}",
                tier="primary"
            ))
        elif defense_rank <= AVERAGE_DEFENSE_RANK:
            factors.append(UnderFactor(
                name="defense_average",
                weight=V19_UNDER_FACTOR_WEIGHTS["defense_average"],
                adjustment=V19_UNDER_ADJUSTMENTS["defense_average"],
                reason=f"Average defense: {defense.team_abbrev} ranks #{defense_rank} vs {pt.upper()}",
                tier="context"
            ))
    
    # === Fatigue Factors (VALIDATED STRONG) ===
    if b2b.is_second_of_b2b:
        factors.append(UnderFactor(
            name="b2b_fatigue",
            weight=V19_UNDER_FACTOR_WEIGHTS["b2b_fatigue"],
            adjustment=V19_UNDER_ADJUSTMENTS["b2b_fatigue"],
            reason="B2B fatigue: Second game of back-to-back (69-75% hit rate)",
            tier="primary"
        ))
    elif b2b.is_third_in_four:
        factors.append(UnderFactor(
            name="third_in_four",
            weight=V19_UNDER_FACTOR_WEIGHTS["third_in_four"],
            adjustment=V19_UNDER_ADJUSTMENTS["third_in_four"],
            reason="Fatigue: Third game in four days",
            tier="supporting"
        ))
    
    # === Injury Rust Factor ===
    if stats.days_since_last_game >= 7:
        factors.append(UnderFactor(
            name="injury_rust_first",
            weight=V19_UNDER_FACTOR_WEIGHTS["injury_rust_first"],
            adjustment=V19_UNDER_ADJUSTMENTS["injury_rust_first"],
            reason=f"Injury rust: {stats.days_since_last_game} days since last game",
            tier="primary"
        ))
    elif stats.days_since_last_game >= 5:
        factors.append(UnderFactor(
            name="injury_rust_second",
            weight=V19_UNDER_FACTOR_WEIGHTS["injury_rust_second"],
            adjustment=V19_UNDER_ADJUSTMENTS["injury_rust_second"],
            reason=f"Extended rest: {stats.days_since_last_game} days off",
            tier="supporting"
        ))
    
    # =========================================================================
    # TIER 2: SUPPORTING SIGNALS - Cold streaks, H2H, minutes (55-62% hit rates)
    # =========================================================================
    
    # === Cold Streak Detection ===
    cold_severity, cold_reasons = detect_cold_streak_pattern(
        stats, pt,
        mild_threshold=config.cold_streak_mild_threshold,
        severe_threshold=config.cold_streak_severe_threshold,
    )
    
    if cold_severity == "severe":
        # SEVERE alone is risky (48%), but good as support
        factors.append(UnderFactor(
            name="cold_streak_severe",
            weight=V19_UNDER_FACTOR_WEIGHTS["cold_streak_severe"],  # Low weight!
            adjustment=V19_UNDER_ADJUSTMENTS["cold_streak_severe"],
            reason=cold_reasons[0] if cold_reasons else "Severe cold streak (L5 < 80% of season)",
            tier="risky"  # Mark as risky - needs support!
        ))
    elif cold_severity == "mild":
        factors.append(UnderFactor(
            name="cold_streak_mild",
            weight=V19_UNDER_FACTOR_WEIGHTS["cold_streak_mild"],
            adjustment=V19_UNDER_ADJUSTMENTS["cold_streak_mild"],
            reason=cold_reasons[0] if cold_reasons else "Mild cold streak (L5 < 90% of season)",
            tier="supporting"
        ))
    
    # === Historical Matchup ===
    if stats.vs_opponent and stats.vs_opponent.has_sufficient_data(config.h2h_min_games):
        if stats.vs_opponent.is_poor_matchup(pt, config.h2h_poor_threshold):
            factors.append(UnderFactor(
                name="poor_h2h_history",
                weight=V19_UNDER_FACTOR_WEIGHTS["poor_h2h_history"],
                adjustment=V19_UNDER_ADJUSTMENTS.get("poor_h2h_history", 0.94),
                reason=f"Poor H2H: {stats.vs_opponent.games_played} games vs {stats.vs_opponent.opponent_abbrev}, {stats.vs_opponent.pts_vs_season_pct:.0f}% vs season",
                tier="supporting"
            ))
    
    # === Minutes Decline ===
    minutes_trend = stats.get_minutes_trend()
    if minutes_trend < config.minutes_decline_threshold:
        factors.append(UnderFactor(
            name="minutes_decline",
            weight=V19_UNDER_FACTOR_WEIGHTS["minutes_decline"],
            adjustment=V19_UNDER_ADJUSTMENTS["minutes_decline"],
            reason=f"Minutes declining: {minutes_trend:.0f}% (L5 vs L15)",
            tier="supporting"
        ))
    
    # =========================================================================
    # TIER 3: BOX SCORE SIGNALS (V19 Enhanced) - ~55% hit rates
    # =========================================================================
    
    # === Negative Plus/Minus Trend ===
    if stats.efficiency.has_negative_plus_minus(config.plus_minus_negative_threshold):
        factors.append(UnderFactor(
            name="negative_plus_minus",
            weight=V19_UNDER_FACTOR_WEIGHTS["negative_plus_minus"],
            adjustment=V19_UNDER_ADJUSTMENTS["negative_plus_minus"],
            reason=f"Negative +/-: L5 avg {stats.efficiency.l5_plus_minus_avg:.1f}",
            tier="boxscore"
        ))
    
    # === Poor Efficiency Trend (FG%) ===
    if stats.efficiency.is_efficiency_declining(config.efficiency_decline_threshold):
        factors.append(UnderFactor(
            name="poor_efficiency_trend",
            weight=V19_UNDER_FACTOR_WEIGHTS["poor_efficiency_trend"],
            adjustment=V19_UNDER_ADJUSTMENTS["poor_efficiency_trend"],
            reason=f"FG% declining: {stats.efficiency.get_fg_trend():.1f}% (L5 vs L15)",
            tier="boxscore"
        ))
    
    # === Poor True Shooting % Trend ===
    if stats.efficiency.is_ts_declining(config.ts_decline_threshold):
        factors.append(UnderFactor(
            name="poor_ts_trend",
            weight=V19_UNDER_FACTOR_WEIGHTS["poor_ts_trend"],
            adjustment=V19_UNDER_ADJUSTMENTS["poor_ts_trend"],
            reason=f"TS% declining: {stats.efficiency.get_ts_trend():.1f}%",
            tier="boxscore"
        ))
    
    # === Less Aggressive (Lower FTA) ===
    if stats.efficiency.is_less_aggressive(config.fta_decline_threshold):
        factors.append(UnderFactor(
            name="low_fta_trend",
            weight=V19_UNDER_FACTOR_WEIGHTS["low_fta_trend"],
            adjustment=V19_UNDER_ADJUSTMENTS.get("low_fta_trend", 0.98),
            reason=f"FTA declining: {stats.efficiency.get_fta_trend():.0f}%",
            tier="boxscore"
        ))
    
    # =========================================================================
    # TIER 4: PLAYER PROFILE + CONTEXT - ~55% hit rates
    # =========================================================================
    
    # === High Variance Player ===
    cv = stats.get_cv(pt)
    if cv > config.high_variance_cv_threshold:
        factors.append(UnderFactor(
            name="high_variance",
            weight=V19_UNDER_FACTOR_WEIGHTS["high_variance"],
            adjustment=V19_UNDER_ADJUSTMENTS.get("high_variance", 0.98),
            reason=f"High variance: CV = {cv:.2f} (inconsistent performer)",
            tier="context"
        ))
    
    # === Blowout Risk ===
    if game_context.is_blowout_risk:
        factors.append(UnderFactor(
            name="blowout_risk",
            weight=V19_UNDER_FACTOR_WEIGHTS["blowout_risk"],
            adjustment=V19_UNDER_ADJUSTMENTS["blowout_risk"],
            reason=f"Blowout risk: Spread = {abs(game_context.spread):.1f} (garbage time)",
            tier="context"
        ))
    
    # === Slow Pace Factor ===
    if game_context.expected_pace == "slow":
        factors.append(UnderFactor(
            name="pace_factor_slow",
            weight=V19_UNDER_FACTOR_WEIGHTS["pace_factor_slow"],
            adjustment=V19_UNDER_ADJUSTMENTS["pace_factor_slow"],
            reason=f"Slow pace expected: O/U = {game_context.over_under:.1f}",
            tier="context"
        ))
    
    # =========================================================================
    # Calculate totals
    # =========================================================================
    total_score = sum(f.weight for f in factors)
    
    # Calculate combined adjustment (multiplicative)
    total_adjustment = 1.0
    for f in factors:
        total_adjustment *= f.adjustment
    
    return factors, total_score, total_adjustment


def evaluate_player_for_under(
    conn: sqlite3.Connection,
    stats: PlayerStatsV19,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: ModelConfigV19Under,
) -> Optional[PropPickV19Under]:
    """
    Evaluate a player for an UNDER pick.
    
    Returns PropPickV19Under if pick meets all criteria, None otherwise.
    """
    pt = prop_type.lower()
    
    # === Check minimum prop average ===
    prop_avg = stats.season.get(pt, 0)
    if prop_avg < MIN_PROP_AVERAGES.get(pt, 0):
        return None
    
    # === Get context ===
    defense = get_defense_context(conn, opponent_abbrev, stats.position)
    b2b = get_back_to_back_status(conn, stats.team_abbrev, game_date)
    game_context = get_game_context(conn, stats.team_abbrev, opponent_abbrev, game_date)
    
    # === V19.1: Trade Deadline Adjustments ===
    trade_ctx = get_trade_context(
        conn, stats.player_id, stats.player_name,
        stats.team_abbrev, game_date
    )
    
    # === V19.2: Trade uncertainty check ===
    trade_uncertainty_active = False
    if hasattr(stats, 'was_traded') and stats.was_traded:
        trade_uncertainty_active = True
        if stats.new_team_games < 3:
            return None  # Too uncertain — insufficient new-team data
    
    # === Check defense data requirement ===
    if config.require_defense_data and not defense.data_available:
        return None
    
    # === Check defense rank threshold ===
    defense_rank = defense.get_rank(pt) if defense.data_available else 15
    if defense_rank > config.max_defense_rank_for_under:
        return None  # Don't bet UNDER vs weak defense
    
    # === Calculate factors ===
    factors, total_score, total_adjustment = calculate_v19_under_factors(
        stats, pt, defense, b2b, game_context, config
    )
    
    # === V19.2: Add trade uncertainty as supporting UNDER factor ===
    if trade_uncertainty_active and stats.new_team_games < 10:
        uncertainty_weight = max(5, 25 - (stats.new_team_games * 2))
        total_score += uncertainty_weight
        factors.append(UnderFactor(
            name="trade_role_uncertainty",
            weight=uncertainty_weight,
            adjustment=0.97,  # Slight downward projection adjustment
            reason=f"Traded player ({stats.new_team_games} new-team games) — role uncertainty",
            tier="secondary",
        ))
    
    # === V19.1: Add trade deadline factors to UNDER score ===
    if trade_ctx.has_any_impact:
        trade_under_score, trade_under_count, trade_under_reasons = (
            get_trade_factor_for_under(trade_ctx, pt)
        )
        if trade_under_score > 0:
            total_score += trade_under_score
            for reason in trade_under_reasons:
                factors.append(UnderFactor(
                    name=f"trade_{reason[:30]}",
                    weight=trade_under_score / max(trade_under_count, 1),
                    adjustment=1.0,
                    reason=reason,
                    tier="primary",
                ))
    
    # === Check minimum factor score ===
    if total_score < config.min_factor_score_standard:
        return None
    
    # === Check multi-factor requirement ===
    if config.require_multiple_factors:
        if len(factors) < config.min_factors_required:
            return None
        
        # Check for at least one strong factor
        strong_factors = [f for f in factors if f.weight >= config.min_strong_factor_weight]
        if not strong_factors:
            return None
        
        # Special check: Don't allow cold_streak_severe alone as primary
        # It needs support from defense or fatigue
        primary_factors = [f for f in factors if f.tier == "primary"]
        risky_factors = [f for f in factors if f.tier == "risky"]
        
        if risky_factors and not primary_factors:
            # Only risky factors (like severe cold streak) - reject
            return None
    
    # === Get line (sportsbook preferred, derived fallback) ===
    line_info = get_line(
        conn, stats.player_id, stats.player_name, pt, game_date, stats,
        config.derived_line_adjustment
    )

    # === V19.4: Dynamic lookback weights (same logic as model_v19_general) ===
    # Traded players: pass None so get_projection() uses trade-aware weight schedule.
    # Non-traded players on disrupted teams: shift toward recent games.
    if stats.was_traded and stats.new_team_games < 10:
        projection_weights = None
    elif (
        game_date >= TRADE_DEADLINE_DATE
        and trade_ctx.team_status is not None
        and trade_ctx.team_status.roster_stability_score < 0.60
    ):
        stability = trade_ctx.team_status.roster_stability_score
        if stability <= 0.35:
            projection_weights = {
                'l3': 0.35, 'l5': 0.35, 'l10': 0.20, 'l15': 0.05, 'season': 0.05,
            }
        elif stability <= 0.50:
            projection_weights = {
                'l3': 0.25, 'l5': 0.30, 'l10': 0.25, 'l15': 0.10, 'season': 0.10,
            }
        else:
            projection_weights = {
                'l3': 0.15, 'l5': 0.25, 'l10': 0.30, 'l15': 0.15, 'season': 0.15,
            }
    else:
        projection_weights = config.get_weights()

    # === Calculate projection ===
    base_projection = stats.get_projection(pt, projection_weights)
    adjusted_projection = base_projection * total_adjustment
    
    # === V19.1: Apply trade deadline adjustments to projection ===
    if trade_ctx.has_any_impact:
        trade_adj = apply_trade_adjustments(
            conn, stats.player_id, stats.player_name,
            stats.team_abbrev, game_date,
            adjusted_projection, stats.season.get('reb', 0),
            stats.season.get('ast', 0), stats.avg_minutes,
        )
        if pt == 'pts':
            adjusted_projection = trade_adj.adjusted_pts
        elif pt == 'reb':
            adjusted_projection = trade_adj.adjusted_reb
        elif pt == 'ast':
            adjusted_projection = trade_adj.adjusted_ast
    
    # === Calculate edge ===
    edge = calculate_edge(adjusted_projection, line_info.line, "UNDER")
    
    # === Check edge requirements ===
    min_edge = config.min_edge_sportsbook if line_info.is_sportsbook else config.min_edge_derived
    if edge < min_edge:
        return None
    
    # === REB UNDER additional checks ===
    if pt == 'reb':
        if config.reb_under_require_elite_defense and defense_rank > GOOD_DEFENSE_RANK:
            return None
        if total_score < config.reb_under_min_factor_score:
            return None
    
    # === Determine confidence tier ===
    if total_score >= config.min_factor_score_premium:
        tier = "PREMIUM"
        confidence = config.premium_confidence
    elif total_score >= config.min_factor_score_high:
        tier = "HIGH"
        confidence = config.high_confidence
    else:
        tier = "STANDARD"
        confidence = config.standard_confidence
    
    # === Adjust confidence based on factors ===
    # Sportsbook line boost
    if line_info.is_sportsbook:
        confidence += config.sportsbook_confidence_boost
    
    # Elite defense boost
    if defense.is_elite(pt):
        confidence += 5
    
    # B2B boost
    if b2b.is_second_of_b2b:
        confidence += 3
    
    # Edge boost
    confidence += min(edge / 3, 5)
    
    # V19.2: Apply trade confidence discount
    if hasattr(stats, 'trade_confidence_discount') and stats.trade_confidence_discount < 1.0:
        confidence *= stats.trade_confidence_discount
    
    confidence = min(confidence, 95)  # Cap at 95
    
    # V19.2: Trade-limited players cannot be PREMIUM
    if trade_uncertainty_active and stats.new_team_games < 10:
        if tier == "PREMIUM":
            tier = "HIGH"
    
    # === Get primary and secondary factors ===
    sorted_factors = sorted(factors, key=lambda x: x.weight, reverse=True)
    primary_factor = sorted_factors[0].name if sorted_factors else ""
    secondary_factor = sorted_factors[1].name if len(sorted_factors) > 1 else ""
    
    # === Get cold streak severity ===
    cold_names = [f.name for f in factors]
    if "cold_streak_severe" in cold_names:
        cold_severity = "severe"
    elif "cold_streak_mild" in cold_names:
        cold_severity = "mild"
    else:
        cold_severity = "none"
    
    # === Build reasons ===
    reasons = [f.reason for f in factors]
    
    # === Create pick ===
    pick = PropPickV19Under(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        game_date=game_date,
        position=stats.position,
        prop_type=pt.upper(),
        direction="UNDER",
        line=line_info.line,
        line_source=line_info.source,
        book=line_info.book,
        base_projection=base_projection,
        adjusted_projection=adjusted_projection,
        total_adjustment=total_adjustment,
        edge_pct=edge,
        factor_score=total_score,
        factor_count=len(factors),
        factors=factors,
        primary_factor=primary_factor,
        secondary_factor=secondary_factor,
        confidence_score=confidence,
        confidence_tier=tier,
        defense_rank=defense_rank,
        defense_rating=defense.get_rating(pt),
        is_b2b=b2b.is_second_of_b2b,
        is_third_in_four=b2b.is_third_in_four,
        cold_streak_severity=cold_severity,
        l5_plus_minus=stats.efficiency.l5_plus_minus_avg,
        l5_fg_pct=stats.efficiency.l5_fg_pct,
        l5_ts_pct=stats.efficiency.l5_ts_pct,
        fg_trend=stats.efficiency.get_fg_trend(),
        l3_avg=stats.l3.get(pt, 0),
        l5_avg=stats.l5.get(pt, 0),
        l10_avg=stats.l10.get(pt, 0),
        l15_avg=stats.l15.get(pt, 0),
        season_avg=stats.season.get(pt, 0),
        h2h_games=stats.vs_opponent.games_played if stats.vs_opponent else 0,
        h2h_avg=getattr(stats.vs_opponent, f"avg_{pt}", 0) if stats.vs_opponent else 0,
        h2h_vs_season_pct=getattr(stats.vs_opponent, f"{pt}_vs_season_pct", 0) if stats.vs_opponent else 0,
        reasons=reasons,
    )
    
    return pick


# ============================================================================
# Main Entry Points
# ============================================================================

def get_daily_picks_v19_under(
    game_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV19Under] = None,
    verbose: bool = False,
) -> DailyPicksV19Under:
    """
    Generate UNDER picks for all games on a specific date.
    
    Args:
        game_date: Date in YYYY-MM-DD format
        db_path: Path to database (uses default if not provided)
        config: Model configuration (uses defaults if not provided)
        verbose: Print progress information
    
    Returns:
        DailyPicksV19Under object with all UNDER picks
    """
    if config is None:
        config = ModelConfigV19Under()
    
    if db_path is None:
        paths = get_paths()
        db_path = str(paths.db_path)
    
    db = Db(Path(db_path))
    
    result = DailyPicksV19Under(
        date=game_date,
        games=0,
        config=config,
    )
    
    with db.connect() as conn:
        # Get games for date
        games = get_games_for_date(conn, game_date)
        result.games = len(games)
        
        if verbose:
            print(f"\nModel V19 Under - Generating picks for {game_date}")
            print(f"Found {len(games)} games")
        
        # Get injured players for the date
        injured_players = get_injured_players(conn, game_date)
        
        # Process each game
        all_picks = []
        
        for game in games:
            team1_abbrev = abbrev_from_team_name(game["team1_name"]) or "UNK"
            team2_abbrev = abbrev_from_team_name(game["team2_name"]) or "UNK"
            
            # Process both teams
            for team_abbrev, opponent_abbrev in [
                (team1_abbrev, team2_abbrev),
                (team2_abbrev, team1_abbrev)
            ]:
                # Get players for this team
                player_ids = get_players_in_game(
                    conn, team_abbrev, game_date,
                    min_games=config.min_games_required,
                    min_avg_minutes=config.min_avg_minutes
                )
                
                for player_id in player_ids:
                    # Skip injured players
                    if player_id in injured_players:
                        continue
                    
                    # V19.1: Skip recently-traded players with insufficient new-team data
                    skip, skip_reason = should_skip_player(
                        conn, player_id, "", team_abbrev, game_date
                    )
                    if skip:
                        continue
                    
                    # Load player stats
                    stats = load_player_stats(
                        conn, player_id, game_date,
                        opponent_abbrev=opponent_abbrev,
                        min_games=config.min_games_required,
                        min_minutes=config.min_avg_minutes,
                        max_games=config.max_games_lookback,
                    )
                    
                    if stats is None:
                        continue
                    
                    result.players_analyzed += 1
                    
                    # Check sportsbook lines availability
                    pts_line = get_sportsbook_line(conn, player_id, stats.player_name, "pts", game_date)
                    if pts_line:
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                    
                    # Evaluate for each prop type
                    player_picks = []
                    
                    for pt in ['pts', 'reb']:
                        if pt == 'pts' and not config.include_pts_under:
                            continue
                        if pt == 'reb' and not config.include_reb_under:
                            continue
                        
                        pick = evaluate_player_for_under(
                            conn, stats, pt, opponent_abbrev, game_date, config
                        )
                        
                        if pick:
                            player_picks.append(pick)
                    
                    # Keep only best pick per player
                    if player_picks and config.max_picks_per_player == 1:
                        # Sort by factor_score, take best
                        best_pick = max(player_picks, key=lambda p: (p.factor_score, p.edge_pct))
                        all_picks.append(best_pick)
                    else:
                        all_picks.extend(player_picks[:config.max_picks_per_player])
        
        # Sort by confidence tier, then factor score
        tier_order = {"PREMIUM": 3, "HIGH": 2, "STANDARD": 1}
        all_picks.sort(
            key=lambda p: (tier_order.get(p.confidence_tier, 0), p.factor_score, p.edge_pct),
            reverse=True
        )
        
        # Apply daily limit
        result.picks = all_picks[:config.max_picks_per_day]
    
    if verbose:
        print(f"\nGenerated {len(result.picks)} UNDER picks")
        print(f"  PREMIUM: {len(result.premium_picks)}")
        print(f"  HIGH: {len(result.high_picks)}")
    
    return result


def run_backtest_v19_under(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV19Under] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> BacktestResultV19Under:
    """
    Run comprehensive backtest for Model V19 Under.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database
        config: Model configuration
        verbose: Print detailed output
        show_progress: Show progress bar
    
    Returns:
        BacktestResultV19Under with comprehensive analysis
    """
    if config is None:
        config = ModelConfigV19Under()
    
    if db_path is None:
        paths = get_paths()
        db_path = str(paths.db_path)
    
    db = Db(Path(db_path))
    
    result = BacktestResultV19Under(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    # Generate list of dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    total_dates = len(dates)
    
    if verbose:
        print("\n" + "=" * 80)
        print("MODEL V19 UNDER - COMPREHENSIVE BACKTEST")
        print("=" * 80)
        print(f"Period: {start_date} to {end_date}")
        print(f"Total dates to process: {total_dates}")
        print("")
    
    start_time = time.time()
    
    with db.connect() as conn:
        for i, game_date in enumerate(dates):
            # Progress bar
            if show_progress:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_dates - i - 1) / rate if rate > 0 else 0
                
                # Progress bar
                pct = (i + 1) / total_dates
                bar_width = 40
                filled = int(bar_width * pct)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                sys.stdout.write(f"\r[{bar}] {pct*100:.1f}% | Day {i+1}/{total_dates} | {game_date} | ETA: {format_time_remaining(remaining)}")
                sys.stdout.flush()
            
            # Get games for this date
            games = get_games_for_date(conn, game_date)
            if not games:
                continue
            
            result.days_tested += 1
            result.total_games += len(games)
            
            # Generate picks for this date
            daily_picks = get_daily_picks_v19_under(
                game_date, db_path, config, verbose=False
            )
            
            daily_result = {
                "date": game_date,
                "games": len(games),
                "picks": 0,
                "hits": 0,
            }
            
            # Grade each pick
            for pick in daily_picks.picks:
                # Get actual result
                actual = get_actual_stats(
                    conn, pick.player_id, game_date
                )
                
                if actual is None:
                    continue  # Player didn't play or no data
                
                pt = pick.prop_type.lower()
                actual_value = actual.get(pt, 0)
                
                if actual_value is None:
                    continue
                
                pick.actual_value = actual_value
                pick.hit = actual_value < pick.line
                pick.margin = pick.line - actual_value
                
                # Update totals
                result.total_picks += 1
                if pick.hit:
                    result.hits += 1
                
                # By line source
                if pick.line_source == "sportsbook":
                    result.sportsbook_picks += 1
                    if pick.hit:
                        result.sportsbook_hits += 1
                else:
                    result.derived_picks += 1
                    if pick.hit:
                        result.derived_hits += 1
                
                # By confidence tier
                if pick.confidence_tier == "PREMIUM":
                    result.premium_picks += 1
                    if pick.hit:
                        result.premium_hits += 1
                elif pick.confidence_tier == "HIGH":
                    result.high_picks += 1
                    if pick.hit:
                        result.high_hits += 1
                else:
                    result.standard_picks += 1
                    if pick.hit:
                        result.standard_hits += 1
                
                # By prop type
                if pt == "pts":
                    result.pts_picks += 1
                    if pick.hit:
                        result.pts_hits += 1
                elif pt == "reb":
                    result.reb_picks += 1
                    if pick.hit:
                        result.reb_hits += 1
                
                # By factor score range
                if pick.factor_score >= 70:
                    result.score_70_plus_picks += 1
                    if pick.hit:
                        result.score_70_plus_hits += 1
                elif pick.factor_score >= 55:
                    result.score_55_70_picks += 1
                    if pick.hit:
                        result.score_55_70_hits += 1
                else:
                    result.score_45_55_picks += 1
                    if pick.hit:
                        result.score_45_55_hits += 1
                
                # By edge range
                if pick.edge_pct >= 15:
                    result.edge_15_plus_picks += 1
                    if pick.hit:
                        result.edge_15_plus_hits += 1
                elif pick.edge_pct >= 10:
                    result.edge_10_15_picks += 1
                    if pick.hit:
                        result.edge_10_15_hits += 1
                else:
                    result.edge_5_10_picks += 1
                    if pick.hit:
                        result.edge_5_10_hits += 1
                
                # By defense rating
                if pick.defense_rank <= ELITE_DEFENSE_RANK:
                    result.elite_defense_picks += 1
                    if pick.hit:
                        result.elite_defense_hits += 1
                elif pick.defense_rank <= GOOD_DEFENSE_RANK:
                    result.good_defense_picks += 1
                    if pick.hit:
                        result.good_defense_hits += 1
                else:
                    result.average_defense_picks += 1
                    if pick.hit:
                        result.average_defense_hits += 1
                
                # By B2B status
                if pick.is_b2b:
                    result.b2b_picks += 1
                    if pick.hit:
                        result.b2b_hits += 1
                else:
                    result.non_b2b_picks += 1
                    if pick.hit:
                        result.non_b2b_hits += 1
                
                # By primary factor
                pf = pick.primary_factor
                if pf not in result.by_primary_factor:
                    result.by_primary_factor[pf] = {"total": 0, "hits": 0}
                result.by_primary_factor[pf]["total"] += 1
                if pick.hit:
                    result.by_primary_factor[pf]["hits"] += 1
                
                # By factor combination (top 2 factors)
                if pick.secondary_factor:
                    combo = f"{pick.primary_factor}+{pick.secondary_factor}"
                else:
                    combo = pick.primary_factor
                if combo not in result.by_factor_combo:
                    result.by_factor_combo[combo] = {"total": 0, "hits": 0}
                result.by_factor_combo[combo]["total"] += 1
                if pick.hit:
                    result.by_factor_combo[combo]["hits"] += 1
                
                # ROI calculation (assuming -110 odds)
                result.theoretical_wagers += 100
                if pick.hit:
                    result.theoretical_profit += 90.91  # Win $90.91 on $100 bet at -110
                else:
                    result.theoretical_profit -= 100  # Lose $100
                
                # Store pick
                result.all_picks.append(pick)
                
                daily_result["picks"] += 1
                if pick.hit:
                    daily_result["hits"] += 1
            
            result.daily_results.append(daily_result)
    
    if show_progress:
        print("")  # New line after progress bar
    
    if verbose:
        print(result.summary())
    
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Command line interface for Model V19 Under."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V19 Under - UNDER Predictions")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)", default=None)
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--start", help="Backtest start date", default="2025-10-22")
    parser.add_argument("--end", help="Backtest end date", default="2026-02-02")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress bar")
    
    args = parser.parse_args()
    
    if args.backtest:
        result = run_backtest_v19_under(
            args.start, args.end,
            verbose=args.verbose,
            show_progress=not args.no_progress
        )
        print(result.summary())
    else:
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_v19_under(date, verbose=args.verbose)
        print(picks.summary())


if __name__ == "__main__":
    main()
