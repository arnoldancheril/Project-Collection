"""
Model V18 Under - Specialized UNDER Model for NBA Props
========================================================

This is the specialized UNDER model of the V18 dual-model architecture:
- Model V18 General: Holistic multi-factor approach for all picks
- Model V18 Under (this file): Specialized UNDER model with enhanced factors

=============================================================================
V18 UNDER MODEL - KEY DESIGN PRINCIPLES
=============================================================================

1. **SPECIALIZED FOR UNDERS ONLY** (Why Unders Work Better)
   - UNDER picks are more predictable than OVER picks
   - Negative factors compound more reliably than positive ones
   - Elite defenses consistently limit player production
   - Cold streaks persist longer than hot streaks (psychology)
   - V16/V17/V19 showed PTS UNDER at 63.9% vs OVER at 48.3%

2. **HOLISTIC MULTI-FACTOR ANALYSIS** (Not Just Cold Bounce)
   - Combines 15+ factors with validated weights from V16-V19 backtesting
   - Requires 2+ factors minimum (V19 showed 3-factor picks at 74.4%)
   - Cold streak alone is only 52% - MUST pair with defense signal
   - Analyzes FULL box score: +/-, efficiency, minutes trends

3. **HYBRID LINE APPROACH** (Addressing Derived Line Fallacy)
   - Use sportsbook lines when available (6% min edge)
   - ALWAYS generate picks when lines not available (lines come late!)
   - Derived lines use L10 × 1.05 with stricter 12-15% edge requirement
   - Track line source for honest reporting

4. **VALIDATED FACTOR WEIGHTS** (From Extensive Backtesting)
   | Factor               | Weight | Validated Hit Rate | Notes                |
   |---------------------|--------|-------------------|----------------------|
   | Elite Defense (1-5) | 50     | 68-71%            | PRIMARY SIGNAL       |
   | B2B Fatigue         | 40     | 60-70%            | HIGH VALUE           |
   | Injury Rust (1st)   | 35     | 65-70%            | HIGH VALUE           |
   | Good Defense (6-10) | 30     | 60-65%            | SOLID                |
   | Cold Streak Mild    | 25     | 57-62%            | PAIRED WITH DEFENSE  |
   | Third in Four       | 20     | 55-60%            | FATIGUE COMPOUND     |
   | Poor +/- Trend      | 15     | NEW               | BOX SCORE ANALYSIS   |
   | Poor Efficiency     | 15     | NEW               | BOX SCORE ANALYSIS   |
   | Cold Streak Severe  | 12     | 48-52% ALONE      | REQUIRES SUPPORT!    |
   | High Variance       | 10     | 55%               | VOLATILITY           |
   | Poor H2H History    | 15     | 55-65%            | 3+ GAMES REQUIRED    |

5. **CONFIDENCE TIERS**
   | Tier     | Factor Score | Required Factors | Expected HR |
   |----------|--------------|------------------|-------------|
   | PREMIUM  | ≥65          | 3+               | 70-80%      |
   | HIGH     | ≥50          | 2+               | 60-70%      |
   | STANDARD | ≥40          | 2+               | 55-62%      |

6. **WHAT THIS MODEL EXCLUDES**
   - REB UNDER (too volatile ~52-54%) - optional via config
   - AST (coin flip ~54%)
   - UNDER vs weak defense (rank 25+)
   - Single-factor picks (except elite defense)
   - Players with <10 games or <23 min avg

=============================================================================

USAGE:
------
    from src.nba_props.engine.model_v18_under import (
        get_daily_picks_v18_under,
        run_backtest_v18_under,
        ModelConfigV18Under,
    )
    
    # Get UNDER picks for today
    picks = get_daily_picks_v18_under("2026-02-03")
    print(picks.summary())
    
    # Run backtest with progress bar
    result = run_backtest_v18_under(
        "2025-10-22", "2026-02-03",
        verbose=True,
        show_progress=True
    )
    print(result.summary())

Author: PropAI Team - Model V18 Under
Created: February 2026
Version: 18.5
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
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
    detect_cold_streak_pattern,
    
    # Constants
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
    MODEL_VERSION,
)


# ============================================================================
# Version Info
# ============================================================================

UNDER_MODEL_VERSION = "18.5"
UNDER_MODEL_NAME = "Model V18.5 Under"


# ============================================================================
# UNDER Factor Weights - VALIDATED FROM V16/V17/V19 BACKTESTING
# ============================================================================

# These weights are tuned based on actual backtest performance:
# - Elite defense: 68-71% HR (PRIMARY SIGNAL)
# - B2B fatigue: 60-70% HR 
# - Injury rust: 65-70% HR
# - Cold streak (combined with defense): 80%+ HR
# - Cold streak alone: only 48-52% HR (REQUIRES SUPPORT)

UNDER_FACTOR_WEIGHTS = {
    # PRIMARY FACTORS - Defense (highest validated hit rates)
    "defense_elite": 50,        # Top 5 DVP rank - 68-71% HR
    "defense_good": 30,         # Top 6-10 DVP rank - 60-65% HR
    
    # HIGH VALUE - Fatigue factors (consistently validated)
    "b2b_fatigue": 40,          # Second of back-to-back - 60-70% HR
    "third_in_four": 20,        # Third game in 4 nights - 55-60% HR
    
    # HIGH VALUE - Injury factors
    "injury_rust_first": 35,    # First game back from injury - 65-70% HR
    "injury_rust_second": 18,   # Second game back
    "injury_rust_third": 8,     # Third game back
    
    # MODERATE - Cold streak (CAREFUL: severe alone is risky!)
    "cold_streak_severe": 12,   # L5 < 80% of season - ONLY 48-52% alone!
    "cold_streak_mild": 25,     # L5 < 90% of season - 57-62% HR
    
    # MODERATE - V18 Box Score Analysis factors
    "negative_plus_minus": 15,  # L5 avg +/- < -5 - NEW
    "poor_efficiency_trend": 15, # FG% declining by 5%+ - NEW
    "minutes_decline": 12,      # Minutes trending down 10%+ - 55-60% HR
    
    # SUPPORTING - Player characteristics
    "high_variance": 10,        # CV > 0.40 (inconsistent player)
    
    # SUPPORTING - Historical matchup
    "poor_h2h_history": 15,     # Below avg vs this opponent (3+ games)
    
    # SITUATIONAL
    "blowout_risk": 10,         # Spread > 10 (garbage time risk)
    "home_disadvantage": 5,     # Away player vs strong home defense
}


# Factor adjustments (multipliers applied to projections for edge calculation)
# Values < 1.0 reduce the projection (supports UNDER)
UNDER_FACTOR_ADJUSTMENTS = {
    "defense_elite": 0.86,      # -14% (strongest)
    "defense_good": 0.92,       # -8%
    "cold_streak_severe": 0.88, # -12%
    "cold_streak_mild": 0.94,   # -6%
    "b2b_fatigue": 0.93,        # -7%
    "third_in_four": 0.96,      # -4%
    "injury_rust_first": 0.82,  # -18% (returning from injury is big)
    "injury_rust_second": 0.90, # -10%
    "injury_rust_third": 0.95,  # -5%
    "negative_plus_minus": 0.96, # -4%
    "poor_efficiency_trend": 0.96, # -4%
    "minutes_decline": 0.95,    # -5%
    "high_variance": 0.97,      # -3%
    "poor_h2h_history": 0.93,   # -7%
    "blowout_risk": 0.95,       # -5%
    "home_disadvantage": 0.99,  # -1%
}


# ============================================================================
# Confidence Tier Thresholds
# ============================================================================

PREMIUM_SCORE_THRESHOLD = 65   # Score ≥65 = PREMIUM tier (70-80% HR expected)
HIGH_SCORE_THRESHOLD = 50      # Score ≥50 = HIGH tier (60-70% HR expected)
STANDARD_SCORE_THRESHOLD = 40  # Score ≥40 = STANDARD tier (55-62% HR expected)

# Minimum factors required for each tier
PREMIUM_MIN_FACTORS = 3
HIGH_MIN_FACTORS = 2
STANDARD_MIN_FACTORS = 2


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV18Under:
    """
    Model V18.5 Under Configuration.
    
    Specialized UNDER model with holistic multi-factor analysis.
    Addresses all shortcomings from V2-V17 models.
    """
    # === VERSION INFO ===
    model_name: str = UNDER_MODEL_NAME
    model_version: str = UNDER_MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False  # NEVER require - lines come late!
    derived_line_adjustment: float = 1.05  # +5% for derived lines (sportsbook typically higher)
    sportsbook_confidence_boost: float = 10.0  # Boost score when using real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10          # Need history for reliable analysis
    min_minutes_filter: int = 5           # Filter garbage time games
    min_avg_minutes: float = 23.0         # Established players only
    max_games_lookback: int = 20          # Recent games for analysis
    
    # === FACTOR THRESHOLDS ===
    # Cold streak thresholds (L5 vs Season deviation %)
    cold_streak_mild_threshold: float = -10.0   # L5 is 10%+ below season
    cold_streak_severe_threshold: float = -20.0  # L5 is 20%+ below season
    
    # Variance threshold (coefficient of variation)
    high_variance_threshold: float = 0.40  # CV > 0.40 = inconsistent
    
    # Efficiency thresholds
    efficiency_decline_threshold: float = -5.0  # FG% down 5%+
    plus_minus_negative_threshold: float = -5.0  # L5 avg +/- < -5
    
    # Minutes decline threshold
    minutes_decline_threshold: float = -10.0  # L5 min 10%+ below L15
    
    # Historical struggle threshold
    historical_struggle_threshold: float = -10.0  # vs_opp 10%+ below season
    
    # Blowout risk (spread threshold)
    blowout_spread_threshold: float = 10.0  # If spread > 10, blowout risk
    
    # Injury rust windows (days since last game)
    injury_rust_first_days: int = 7        # 7+ days = first game back
    injury_rust_second_days: int = 5       # 5-6 days = extended rest
    injury_rust_third_days: int = 4        # 4 days = moderate rest
    
    # === EDGE REQUIREMENTS ===
    min_edge_sportsbook: float = 6.0       # 6%+ edge vs sportsbook line
    min_edge_derived: float = 12.0         # 12%+ edge vs derived line (stricter!)
    premium_edge_threshold: float = 18.0   # Premium picks need higher edge
    
    # === SCORE THRESHOLDS ===
    premium_score_threshold: float = PREMIUM_SCORE_THRESHOLD
    high_score_threshold: float = HIGH_SCORE_THRESHOLD
    min_score_threshold: float = STANDARD_SCORE_THRESHOLD
    
    # === MULTI-FACTOR REQUIREMENTS ===
    # V19 showed 3-factor picks hit at 74.4%!
    require_multiple_factors: bool = True
    min_factors_for_pick: int = 2          # Require at least 2 factors
    min_factors_for_premium: int = 3       # Premium needs 3+ factors
    
    # Cold streak protection: severe alone is only 48-52%
    cold_streak_requires_defense: bool = True  # Severe cold needs defense support
    
    # === DEFENSE REQUIREMENTS ===
    require_defense_data: bool = True       # Need DVP data for picks
    max_defense_rank_for_under: int = 20    # No UNDER vs weak defense (rank 25+)
    elite_defense_solo_allowed: bool = True # Allow elite defense as solo factor
    
    # === PROP SELECTION ===
    include_pts_under: bool = True         # PTS UNDER is primary (63.9% HR)
    include_reb_under: bool = False        # REB UNDER disabled by default (~52-54%)
    include_ast_under: bool = False        # AST excluded (~54% = coin flip)
    
    # If REB enabled, require elite defense only
    reb_under_require_elite_only: bool = True
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 4            # Focused picks
    max_picks_per_day: int = 25            # Quality over quantity
    max_picks_per_player: int = 1          # One prop per player
    
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
    validated_hr: Optional[float] = None  # Historical hit rate for this factor


@dataclass 
class PropPickV18Under:
    """A pick generated by Model V18.5 Under."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    position: str
    
    # Pick details (always UNDER for this model)
    prop_type: str  # PTS, REB
    direction: str = "UNDER"
    
    # Line information
    line: float = 0.0
    line_source: str = "derived"  # "sportsbook" or "derived"
    book: Optional[str] = None
    
    # Projections
    base_projection: float = 0.0      # Before factor adjustments
    adjusted_projection: float = 0.0   # After factor adjustments
    total_adjustment: float = 1.0      # Combined adjustment factor
    
    # Edge calculation
    edge_pct: float = 0.0
    
    # Factor scoring (V18 holistic approach)
    factor_score: float = 0.0
    factors: List[UnderFactor] = field(default_factory=list)
    factor_count: int = 0
    primary_factor: str = ""
    
    # Confidence
    confidence_score: float = 0.0
    confidence_tier: str = "STANDARD"  # PREMIUM, HIGH, STANDARD
    
    # Defense context
    defense_rank: int = 15
    defense_rating: str = "average"
    
    # Situation context
    is_b2b: bool = False
    is_third_in_four: bool = False
    cold_streak_severity: str = "none"  # none, mild, severe
    days_since_injury: Optional[int] = None
    
    # Box score analysis (V18 NEW)
    l5_plus_minus: float = 0.0
    efficiency_trend: float = 0.0  # FG% trend %
    minutes_trend: float = 0.0     # Minutes trend %
    
    # Stats for display
    l3_avg: float = 0.0
    l5_avg: float = 0.0
    l10_avg: float = 0.0
    l15_avg: float = 0.0
    season_avg: float = 0.0
    variance_cv: float = 0.0
    
    # Historical matchup
    h2h_games: int = 0
    h2h_avg: float = 0.0
    h2h_vs_season_pct: float = 0.0
    
    # Reasoning
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
            "position": self.position,
            "date": self.game_date,
            "prop": self.prop_type.upper(),
            "direction": "UNDER",
            "line": round(self.line, 1),
            "line_source": self.line_source,
            "book": self.book,
            "base_proj": round(self.base_projection, 1),
            "adj_proj": round(self.adjusted_projection, 1),
            "adjustment": f"{(1 - self.total_adjustment) * 100:.1f}% reduction",
            "edge": f"{self.edge_pct:.1f}%",
            "factor_score": round(self.factor_score, 1),
            "factor_count": self.factor_count,
            "factors": [f.name for f in self.factors],
            "primary_factor": self.primary_factor,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "b2b": self.is_b2b,
            "cold_streak": self.cold_streak_severity,
            "plus_minus_l5": round(self.l5_plus_minus, 1),
            "efficiency_trend": f"{self.efficiency_trend:.1f}%",
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "season": round(self.season_avg, 1),
            "reasons": self.reasons[:5],  # Top 5 reasons
            "actual": self.actual_value,
            "hit": self.hit,
        }
    
    def summary_line(self) -> str:
        """One-line summary for display."""
        factors_str = ", ".join(f.name for f in self.factors[:3])
        hit_str = ""
        if self.hit is not None:
            hit_str = " ✅" if self.hit else " ❌"
        return (
            f"📉 {self.player_name} ({self.team_abbrev} vs {self.opponent_abbrev}) - "
            f"{self.prop_type} UNDER {self.line:.1f} [{self.line_source[:5]}] | "
            f"Proj: {self.adjusted_projection:.1f} | Edge: {self.edge_pct:.1f}% | "
            f"Score: {self.factor_score:.0f} ({self.confidence_tier}) | "
            f"{factors_str}{hit_str}"
        )


@dataclass
class DailyPicksV18Under:
    """All UNDER picks for a day from Model V18.5."""
    date: str
    games: int
    config: ModelConfigV18Under = field(default_factory=ModelConfigV18Under)
    picks: List[PropPickV18Under] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    players_filtered: int = 0
    
    # Defense data status
    defense_data_available: bool = True
    defense_data_freshness: str = ""
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV18Under]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV18Under]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def premium_picks(self) -> List[PropPickV18Under]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV18Under]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    @property
    def standard_picks(self) -> List[PropPickV18Under]:
        return [p for p in self.picks if p.confidence_tier == "STANDARD"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*75}",
            f"MODEL V18.5 UNDER - DAILY PICKS FOR {self.date}",
            f"{'='*75}",
            f"Games: {self.games} | Players analyzed: {self.players_analyzed}",
            f"  Sportsbook lines available: {self.players_with_sportsbook_lines}",
            f"  Using derived lines: {self.players_with_derived_lines}",
            f"  Filtered out: {self.players_filtered}",
            f"Defense data: {'Available' if self.defense_data_available else 'NOT AVAILABLE'}",
            "",
            f"Total UNDER picks: {self.total_picks}",
            f"  PREMIUM (score≥{int(self.config.premium_score_threshold)}): {len(self.premium_picks)}",
            f"  HIGH (score≥{int(self.config.high_score_threshold)}): {len(self.high_picks)}",
            f"  STANDARD (score≥{int(self.config.min_score_threshold)}): {len(self.standard_picks)}",
            f"  By line source: Sportsbook={len(self.sportsbook_picks)}, Derived={len(self.derived_picks)}",
            "",
        ]
        
        # Group by tier
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} TIER ({len(tier_picks)} picks) ---")
                for p in sorted(tier_picks, key=lambda x: -x.factor_score):
                    lines.append(p.summary_line())
                lines.append("")
        
        lines.append(f"{'='*75}")
        return "\n".join(lines)


@dataclass
class BacktestResultV18Under:
    """Comprehensive backtest results for Model V18.5 Under."""
    start_date: str
    end_date: str
    config: ModelConfigV18Under = field(default_factory=ModelConfigV18Under)
    
    # Overall
    total_picks: int = 0
    hits: int = 0
    
    # By line source (CRITICAL for honest reporting)
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
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    
    # By primary factor (KEY for validation)
    by_primary_factor: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By factor count
    by_factor_count: Dict[int, Dict[str, int]] = field(default_factory=dict)
    
    # By factor score bucket
    score_65_plus_picks: int = 0
    score_65_plus_hits: int = 0
    score_50_65_picks: int = 0
    score_50_65_hits: int = 0
    score_40_50_picks: int = 0
    score_40_50_hits: int = 0
    
    # By edge range
    edge_20_plus_picks: int = 0
    edge_20_plus_hits: int = 0
    edge_15_20_picks: int = 0
    edge_15_20_hits: int = 0
    edge_10_15_picks: int = 0
    edge_10_15_hits: int = 0
    edge_6_10_picks: int = 0
    edge_6_10_hits: int = 0
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # All picks for detailed analysis
    all_picks: List[PropPickV18Under] = field(default_factory=list)
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
    
    @property
    def theoretical_roi(self) -> float:
        """Calculate theoretical ROI assuming -110 juice."""
        if self.total_picks == 0:
            return 0.0
        wins = self.hits
        losses = self.total_picks - self.hits
        profit = wins * 0.909 - losses  # Win pays 0.909 units, loss costs 1 unit
        return profit / self.total_picks * 100
    
    def summary(self) -> str:
        def pct(h, t):
            return f"{h/t*100:.1f}%" if t > 0 else "N/A"
        
        lines = [
            f"{'='*75}",
            f"MODEL V18.5 UNDER - BACKTEST RESULTS",
            f"{'='*75}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"Avg picks/day: {self.total_picks / max(self.days_tested, 1):.1f}",
            "",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            f"Theoretical ROI: {self.theoretical_roi:.1f}%",
            "",
            f"BY LINE SOURCE (Honest Reporting):",
            f"  Sportsbook: {pct(self.sportsbook_hits, self.sportsbook_picks)} ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived:    {pct(self.derived_hits, self.derived_picks)} ({self.derived_hits}/{self.derived_picks})",
            "",
            f"BY CONFIDENCE TIER:",
            f"  PREMIUM (score ≥65): {pct(self.premium_hits, self.premium_picks)} ({self.premium_hits}/{self.premium_picks})",
            f"  HIGH (score 50-64):  {pct(self.high_hits, self.high_picks)} ({self.high_hits}/{self.high_picks})",
            f"  STANDARD (score 40-49): {pct(self.standard_hits, self.standard_picks)} ({self.standard_hits}/{self.standard_picks})",
            "",
            f"BY PROP TYPE:",
            f"  PTS UNDER: {pct(self.pts_hits, self.pts_picks)} ({self.pts_hits}/{self.pts_picks})",
            f"  REB UNDER: {pct(self.reb_hits, self.reb_picks)} ({self.reb_hits}/{self.reb_picks})",
            "",
            f"BY FACTOR SCORE BUCKET:",
            f"  Score ≥65 (Premium):  {pct(self.score_65_plus_hits, self.score_65_plus_picks)} ({self.score_65_plus_hits}/{self.score_65_plus_picks})",
            f"  Score 50-64 (High):   {pct(self.score_50_65_hits, self.score_50_65_picks)} ({self.score_50_65_hits}/{self.score_50_65_picks})",
            f"  Score 40-49 (Std):    {pct(self.score_40_50_hits, self.score_40_50_picks)} ({self.score_40_50_hits}/{self.score_40_50_picks})",
            "",
            f"BY EDGE RANGE:",
            f"  Edge 20%+:   {pct(self.edge_20_plus_hits, self.edge_20_plus_picks)} ({self.edge_20_plus_hits}/{self.edge_20_plus_picks})",
            f"  Edge 15-19%: {pct(self.edge_15_20_hits, self.edge_15_20_picks)} ({self.edge_15_20_hits}/{self.edge_15_20_picks})",
            f"  Edge 10-14%: {pct(self.edge_10_15_hits, self.edge_10_15_picks)} ({self.edge_10_15_hits}/{self.edge_10_15_picks})",
            f"  Edge 6-9%:   {pct(self.edge_6_10_hits, self.edge_6_10_picks)} ({self.edge_6_10_hits}/{self.edge_6_10_picks})",
            "",
        ]
        
        # Primary factor breakdown
        if self.by_primary_factor:
            lines.append("BY PRIMARY FACTOR:")
            sorted_factors = sorted(
                self.by_primary_factor.items(),
                key=lambda x: x[1].get("picks", 0),
                reverse=True
            )
            for factor, data in sorted_factors:
                picks = data.get("picks", 0)
                hits = data.get("hits", 0)
                if picks > 0:
                    lines.append(f"  {factor:25s} {pct(hits, picks):>6s} ({hits}/{picks})")
        
        lines.append("")
        
        # Factor count breakdown
        if self.by_factor_count:
            lines.append("BY FACTOR COUNT:")
            for count in sorted(self.by_factor_count.keys()):
                data = self.by_factor_count[count]
                picks = data.get("picks", 0)
                hits = data.get("hits", 0)
                if picks > 0:
                    lines.append(f"  {count} factors: {pct(hits, picks):>6s} ({hits}/{picks})")
        
        lines.append(f"{'='*75}")
        return "\n".join(lines)


# ============================================================================
# Core Factor Calculation
# ============================================================================

def calculate_under_factors(
    stats: PlayerStatsV18,
    prop_type: str,
    defense_context: DefenseContextV18,
    b2b_info: BackToBackInfo,
    config: ModelConfigV18Under,
    spread: Optional[float] = None,
) -> Tuple[List[UnderFactor], float, float]:
    """
    Calculate all applicable factors for an UNDER pick.
    
    This is the CORE of the V18 Under model - holistic multi-factor analysis.
    
    Returns: (factors, total_score, total_adjustment)
    """
    factors = []
    pt = prop_type.lower()
    
    # =========================================================================
    # DEFENSE FACTORS (PRIMARY - Highest validated hit rates)
    # =========================================================================
    defense_rank = defense_context.get_rank(pt)
    
    if defense_context.data_available:
        if defense_rank <= ELITE_DEFENSE_RANK:  # Top 3-5
            factors.append(UnderFactor(
                name="defense_elite",
                weight=UNDER_FACTOR_WEIGHTS["defense_elite"],
                adjustment=UNDER_FACTOR_ADJUSTMENTS["defense_elite"],
                reason=f"Elite defense: {defense_context.team_abbrev} #{defense_rank} vs {pt.upper()}",
                validated_hr=0.70  # 68-71% historical
            ))
        elif defense_rank <= GOOD_DEFENSE_RANK:  # Top 6-10
            factors.append(UnderFactor(
                name="defense_good",
                weight=UNDER_FACTOR_WEIGHTS["defense_good"],
                adjustment=UNDER_FACTOR_ADJUSTMENTS["defense_good"],
                reason=f"Good defense: {defense_context.team_abbrev} #{defense_rank} vs {pt.upper()}",
                validated_hr=0.62  # 60-65% historical
            ))
    
    # =========================================================================
    # FATIGUE FACTORS (HIGH VALUE - Consistently validated)
    # =========================================================================
    if b2b_info.is_second_of_b2b:
        factors.append(UnderFactor(
            name="b2b_fatigue",
            weight=UNDER_FACTOR_WEIGHTS["b2b_fatigue"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["b2b_fatigue"],
            reason="B2B fatigue: Second game of back-to-back",
            validated_hr=0.65  # 60-70% historical
        ))
    elif b2b_info.is_third_in_four:
        factors.append(UnderFactor(
            name="third_in_four",
            weight=UNDER_FACTOR_WEIGHTS["third_in_four"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["third_in_four"],
            reason="Fatigue: Third game in four nights",
            validated_hr=0.58  # 55-60% historical
        ))
    
    # =========================================================================
    # INJURY RUST FACTORS (HIGH VALUE)
    # =========================================================================
    days_rest = stats.days_since_last_game
    if days_rest >= config.injury_rust_first_days:
        factors.append(UnderFactor(
            name="injury_rust_first",
            weight=UNDER_FACTOR_WEIGHTS["injury_rust_first"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["injury_rust_first"],
            reason=f"Injury rust: {days_rest} days since last game (first back)",
            validated_hr=0.68  # 65-70% historical
        ))
    elif days_rest >= config.injury_rust_second_days:
        factors.append(UnderFactor(
            name="injury_rust_second",
            weight=UNDER_FACTOR_WEIGHTS["injury_rust_second"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["injury_rust_second"],
            reason=f"Extended rest: {days_rest} days since last game",
            validated_hr=0.60
        ))
    elif days_rest >= config.injury_rust_third_days:
        factors.append(UnderFactor(
            name="injury_rust_third",
            weight=UNDER_FACTOR_WEIGHTS["injury_rust_third"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["injury_rust_third"],
            reason=f"Moderate rest: {days_rest} days off",
            validated_hr=0.55
        ))
    
    # =========================================================================
    # COLD STREAK FACTORS (MODERATE - BE CAREFUL!)
    # Severe cold streak alone is only 48-52% - must pair with defense
    # =========================================================================
    cold_severity, cold_reasons = detect_cold_streak_pattern(
        stats, pt,
        mild_threshold=config.cold_streak_mild_threshold,
        severe_threshold=config.cold_streak_severe_threshold,
    )
    
    # Check if we have defense support for cold streak
    has_defense_support = any(f.name.startswith("defense") for f in factors)
    
    if cold_severity == "severe":
        # Severe cold streak - only include if paired with defense or config allows
        if has_defense_support or not config.cold_streak_requires_defense:
            factors.append(UnderFactor(
                name="cold_streak_severe",
                weight=UNDER_FACTOR_WEIGHTS["cold_streak_severe"],
                adjustment=UNDER_FACTOR_ADJUSTMENTS["cold_streak_severe"],
                reason=cold_reasons[0] if cold_reasons else "Severe cold streak (L5 20%+ below season)",
                validated_hr=0.50 if not has_defense_support else 0.75  # Much better with defense
            ))
    elif cold_severity == "mild":
        factors.append(UnderFactor(
            name="cold_streak_mild",
            weight=UNDER_FACTOR_WEIGHTS["cold_streak_mild"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["cold_streak_mild"],
            reason=cold_reasons[0] if cold_reasons else "Mild cold streak (L5 10%+ below season)",
            validated_hr=0.60  # 57-62% historical
        ))
    
    # =========================================================================
    # V18 NEW: BOX SCORE ANALYSIS FACTORS
    # =========================================================================
    
    # Negative Plus/Minus trend
    if stats.efficiency.has_negative_plus_minus(config.plus_minus_negative_threshold):
        factors.append(UnderFactor(
            name="negative_plus_minus",
            weight=UNDER_FACTOR_WEIGHTS["negative_plus_minus"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["negative_plus_minus"],
            reason=f"Poor +/- trend: L5 avg {stats.efficiency.l5_plus_minus_avg:.1f}",
            validated_hr=0.55  # NEW - to be validated
        ))
    
    # Poor efficiency (FG% declining)
    if stats.efficiency.is_efficiency_declining(config.efficiency_decline_threshold):
        fg_trend = stats.efficiency.get_fg_trend()
        factors.append(UnderFactor(
            name="poor_efficiency_trend",
            weight=UNDER_FACTOR_WEIGHTS["poor_efficiency_trend"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["poor_efficiency_trend"],
            reason=f"Efficiency declining: FG% {fg_trend:.1f}% trend",
            validated_hr=0.57  # NEW - to be validated
        ))
    
    # Minutes declining
    minutes_trend = stats.get_minutes_trend()
    if minutes_trend < config.minutes_decline_threshold:
        factors.append(UnderFactor(
            name="minutes_decline",
            weight=UNDER_FACTOR_WEIGHTS["minutes_decline"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["minutes_decline"],
            reason=f"Minutes declining: {minutes_trend:.0f}% trend",
            validated_hr=0.55
        ))
    
    # =========================================================================
    # PLAYER CHARACTERISTICS
    # =========================================================================
    
    # High variance (inconsistent player)
    cv = stats.get_cv(pt)
    if cv > config.high_variance_threshold:
        factors.append(UnderFactor(
            name="high_variance",
            weight=UNDER_FACTOR_WEIGHTS["high_variance"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["high_variance"],
            reason=f"High variance: CV={cv:.2f} (inconsistent)",
            validated_hr=0.55
        ))
    
    # =========================================================================
    # HISTORICAL MATCHUP
    # =========================================================================
    if stats.vs_opponent and stats.vs_opponent.is_poor_matchup(pt, config.historical_struggle_threshold):
        h2h = stats.vs_opponent
        factors.append(UnderFactor(
            name="poor_h2h_history",
            weight=UNDER_FACTOR_WEIGHTS["poor_h2h_history"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["poor_h2h_history"],
            reason=f"Poor H2H: {h2h.avg_pts if pt=='pts' else h2h.avg_reb:.1f} avg vs {h2h.opponent_abbrev} ({h2h.games_played} games)",
            validated_hr=0.60  # 55-65% historical
        ))
    
    # =========================================================================
    # SITUATIONAL FACTORS
    # =========================================================================
    
    # Blowout risk (garbage time)
    if spread is not None and abs(spread) > config.blowout_spread_threshold:
        factors.append(UnderFactor(
            name="blowout_risk",
            weight=UNDER_FACTOR_WEIGHTS["blowout_risk"],
            adjustment=UNDER_FACTOR_ADJUSTMENTS["blowout_risk"],
            reason=f"Blowout risk: Spread is {spread:.1f}",
            validated_hr=0.55
        ))
    
    # =========================================================================
    # CALCULATE TOTALS
    # =========================================================================
    total_score = sum(f.weight for f in factors)
    
    # Calculate total adjustment (multiplicative)
    total_adjustment = 1.0
    for f in factors:
        total_adjustment *= f.adjustment
    
    return factors, total_score, total_adjustment


# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_player_for_under(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    opponent_abbrev: str,
    team_abbrev: str,
    config: ModelConfigV18Under,
    spread: Optional[float] = None,
) -> List[PropPickV18Under]:
    """
    Evaluate a player for UNDER picks.
    
    Returns list of PropPickV18Under (usually 0 or 1, max 1 per player).
    """
    picks = []
    
    # Load player stats
    stats = load_player_stats(
        conn,
        player_id,
        game_date,
        opponent_abbrev=opponent_abbrev,
        min_games=config.min_games_required,
        min_minutes=config.min_avg_minutes,
        max_games=config.max_games_lookback,
        min_game_minutes=config.min_minutes_filter,
    )
    
    if not stats:
        return []
    
    # Get context data
    defense_context = get_defense_context(conn, opponent_abbrev, stats.position)
    b2b_info = get_back_to_back_status(conn, team_abbrev, game_date)
    
    # Check defense data requirement
    if config.require_defense_data and not defense_context.data_available:
        return []
    
    # Determine prop types to evaluate
    prop_types = []
    if config.include_pts_under:
        prop_types.append('pts')
    if config.include_reb_under:
        prop_types.append('reb')
    
    best_pick = None
    best_score = 0
    
    for prop_type in prop_types:
        pt = prop_type.lower()
        
        # Check minimum average threshold
        avg = stats.l10.get(pt, 0)
        if avg < MIN_PROP_AVERAGES.get(pt, 0):
            continue
        
        # For REB, may require elite defense only
        if pt == 'reb' and config.reb_under_require_elite_only:
            if defense_context.get_rank(pt) > ELITE_DEFENSE_RANK:
                continue
        
        # Check defense rank limit (no UNDER vs weak defense)
        if defense_context.data_available:
            if defense_context.get_rank(pt) > config.max_defense_rank_for_under:
                continue
        
        # Calculate factors
        factors, total_score, total_adjustment = calculate_under_factors(
            stats, pt, defense_context, b2b_info, config, spread
        )
        
        # Check minimum score
        if total_score < config.min_score_threshold:
            continue
        
        # Check multi-factor requirement
        factor_count = len(factors)
        if config.require_multiple_factors:
            # Elite defense alone is allowed (68-71% HR)
            has_elite_defense = any(f.name == "defense_elite" for f in factors)
            if factor_count < config.min_factors_for_pick:
                if not (config.elite_defense_solo_allowed and has_elite_defense):
                    continue
        
        # Calculate projection and edge
        base_projection = stats.get_projection(pt)
        adjusted_projection = base_projection * total_adjustment
        
        # Get line
        line_info = get_line(
            conn, player_id, stats.player_name, pt, game_date, stats,
            config.derived_line_adjustment
        )
        
        # Calculate edge
        edge = calculate_edge(adjusted_projection, line_info.line, "UNDER")
        
        # Apply edge requirements based on line source
        min_edge = config.min_edge_sportsbook if line_info.is_sportsbook else config.min_edge_derived
        if edge < min_edge:
            continue
        
        # Determine confidence tier
        if total_score >= config.premium_score_threshold and factor_count >= config.min_factors_for_premium:
            tier = "PREMIUM"
            confidence = 90.0
        elif total_score >= config.high_score_threshold and factor_count >= config.min_factors_for_pick:
            tier = "HIGH"
            confidence = 75.0
        else:
            tier = "STANDARD"
            confidence = 65.0
        
        # Boost confidence for sportsbook lines
        if line_info.is_sportsbook:
            confidence += config.sportsbook_confidence_boost
        
        # Identify primary factor
        primary_factor = max(factors, key=lambda f: f.weight).name if factors else ""
        
        # Build pick
        pick = PropPickV18Under(
            player_id=player_id,
            player_name=stats.player_name,
            team_abbrev=team_abbrev,
            opponent_abbrev=opponent_abbrev,
            game_date=game_date,
            position=stats.position,
            prop_type=pt.upper(),
            line=line_info.line,
            line_source=line_info.source,
            book=line_info.book,
            base_projection=base_projection,
            adjusted_projection=adjusted_projection,
            total_adjustment=total_adjustment,
            edge_pct=edge,
            factor_score=total_score,
            factors=factors,
            factor_count=factor_count,
            primary_factor=primary_factor,
            confidence_score=confidence,
            confidence_tier=tier,
            defense_rank=defense_context.get_rank(pt),
            defense_rating=defense_context.get_rating(pt),
            is_b2b=b2b_info.is_second_of_b2b,
            is_third_in_four=b2b_info.is_third_in_four,
            cold_streak_severity=detect_cold_streak_pattern(stats, pt)[0],
            days_since_injury=stats.days_since_last_game if stats.days_since_last_game >= 4 else None,
            l5_plus_minus=stats.efficiency.l5_plus_minus_avg,
            efficiency_trend=stats.efficiency.get_fg_trend(),
            minutes_trend=stats.get_minutes_trend(),
            l3_avg=stats.l3.get(pt, 0),
            l5_avg=stats.l5.get(pt, 0),
            l10_avg=stats.l10.get(pt, 0),
            l15_avg=stats.l15.get(pt, 0),
            season_avg=stats.season.get(pt, 0),
            variance_cv=stats.get_cv(pt),
            h2h_games=stats.vs_opponent.games_played if stats.vs_opponent else 0,
            h2h_avg=stats.vs_opponent.avg_pts if stats.vs_opponent and pt == 'pts' else (stats.vs_opponent.avg_reb if stats.vs_opponent else 0),
            h2h_vs_season_pct=stats.vs_opponent.pts_vs_season_pct if stats.vs_opponent and pt == 'pts' else (stats.vs_opponent.reb_vs_season_pct if stats.vs_opponent else 0),
            reasons=[f.reason for f in factors],
        )
        
        # Keep best pick per player
        if total_score > best_score:
            best_score = total_score
            best_pick = pick
    
    if best_pick:
        picks.append(best_pick)
    
    return picks


# ============================================================================
# Daily Picks Generation
# ============================================================================

def get_daily_picks_v18_under(
    game_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV18Under] = None,
    verbose: bool = False,
) -> DailyPicksV18Under:
    """
    Generate UNDER picks for a specific date.
    
    Args:
        game_date: Date in YYYY-MM-DD format
        db_path: Path to database (uses default if None)
        config: Model configuration (uses defaults if None)
        verbose: Print progress
    
    Returns:
        DailyPicksV18Under with all recommended picks
    """
    if config is None:
        config = ModelConfigV18Under()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(Path(db_path))
    
    daily = DailyPicksV18Under(
        date=game_date,
        games=0,
        config=config,
    )
    
    with db.connect() as conn:
        # Get games for date
        games = get_games_for_date(conn, game_date)
        daily.games = len(games)
        
        if verbose:
            print(f"\nAnalyzing {len(games)} games for {game_date}...")
        
        # Get injured players
        injured_ids = get_injured_players(conn, game_date)
        
        all_picks = []
        
        for game in games:
            team1_abbrev = abbrev_from_team_name(game["team1_name"]) or "UNK"
            team2_abbrev = abbrev_from_team_name(game["team2_name"]) or "UNK"
            
            # Evaluate players from both teams
            for team_abbrev, opp_abbrev in [(team1_abbrev, team2_abbrev), (team2_abbrev, team1_abbrev)]:
                # Get players for this team
                player_ids = get_players_in_game(
                    conn, team_abbrev, game_date,
                    min_games=config.min_games_required,
                    min_avg_minutes=config.min_avg_minutes,
                )
                
                for player_id in player_ids:
                    if player_id in injured_ids:
                        continue
                    
                    daily.players_analyzed += 1
                    
                    picks = evaluate_player_for_under(
                        conn, player_id, game_date, opp_abbrev, team_abbrev, config
                    )
                    
                    for pick in picks:
                        if pick.line_source == "sportsbook":
                            daily.players_with_sportsbook_lines += 1
                        else:
                            daily.players_with_derived_lines += 1
                        all_picks.append(pick)
        
        # Sort by factor score and apply limits
        all_picks.sort(key=lambda p: (-p.factor_score, -p.edge_pct))
        
        # Limit per game
        game_counts = {}
        filtered_picks = []
        for pick in all_picks:
            game_key = f"{pick.team_abbrev}_{pick.opponent_abbrev}"
            game_counts[game_key] = game_counts.get(game_key, 0) + 1
            if game_counts[game_key] <= config.max_picks_per_game:
                filtered_picks.append(pick)
        
        # Limit total per day
        daily.picks = filtered_picks[:config.max_picks_per_day]
        daily.players_filtered = len(all_picks) - len(daily.picks)
    
    if verbose:
        print(daily.summary())
    
    return daily


# ============================================================================
# Backtesting
# ============================================================================

def run_backtest_v18_under(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    config: Optional[ModelConfigV18Under] = None,
    verbose: bool = False,
    show_progress: bool = True,
) -> BacktestResultV18Under:
    """
    Run comprehensive backtest for Model V18.5 Under.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Path to database
        config: Model configuration
        verbose: Print detailed output
        show_progress: Show progress bar in terminal
    
    Returns:
        BacktestResultV18Under with comprehensive metrics
    """
    if config is None:
        config = ModelConfigV18Under()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(Path(db_path))
    
    result = BacktestResultV18Under(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    with db.connect() as conn:
        # Get all dates with games in range
        rows = conn.execute(
            """
            SELECT DISTINCT game_date
            FROM games
            WHERE game_date >= ? AND game_date <= ?
            ORDER BY game_date
            """,
            (start_date, end_date),
        ).fetchall()
        
        dates = [r["game_date"] for r in rows]
        total_dates = len(dates)
        
        if verbose:
            print(f"\n{'='*75}")
            print(f"MODEL V18.5 UNDER - BACKTEST")
            print(f"{'='*75}")
            print(f"Period: {start_date} to {end_date}")
            print(f"Testing {total_dates} days...")
            print()
        
        for i, game_date in enumerate(dates):
            # Progress bar
            if show_progress:
                pct = (i + 1) / total_dates * 100
                bar_len = 50
                filled = int(bar_len * (i + 1) // total_dates)
                bar = "█" * filled + "░" * (bar_len - filled)
                sys.stdout.write(f"\r[{bar}] {pct:5.1f}% ({i+1}/{total_dates}) - {game_date}")
                sys.stdout.flush()
            
            # Generate picks for this date
            daily = get_daily_picks_v18_under(
                game_date,
                db_path=db_path,
                config=config,
                verbose=False,
            )
            
            result.total_games += daily.games
            result.days_tested += 1
            
            daily_hits = 0
            daily_picks = 0
            
            for pick in daily.picks:
                # Get actual result
                actual = get_actual_stats(conn, pick.player_id, game_date, pick.prop_type.lower())
                
                if actual is None:
                    continue
                
                pick.actual_value = actual
                pick.hit = actual < pick.line  # UNDER hits when actual < line
                pick.margin = pick.line - actual
                
                # Update totals
                result.total_picks += 1
                daily_picks += 1
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
                else:
                    result.standard_picks += 1
                    if pick.hit:
                        result.standard_hits += 1
                
                # By prop type
                if pick.prop_type == "PTS":
                    result.pts_picks += 1
                    if pick.hit:
                        result.pts_hits += 1
                elif pick.prop_type == "REB":
                    result.reb_picks += 1
                    if pick.hit:
                        result.reb_hits += 1
                
                # By primary factor
                pf = pick.primary_factor
                if pf not in result.by_primary_factor:
                    result.by_primary_factor[pf] = {"picks": 0, "hits": 0}
                result.by_primary_factor[pf]["picks"] += 1
                if pick.hit:
                    result.by_primary_factor[pf]["hits"] += 1
                
                # By factor count
                fc = pick.factor_count
                if fc not in result.by_factor_count:
                    result.by_factor_count[fc] = {"picks": 0, "hits": 0}
                result.by_factor_count[fc]["picks"] += 1
                if pick.hit:
                    result.by_factor_count[fc]["hits"] += 1
                
                # By factor score bucket
                if pick.factor_score >= 65:
                    result.score_65_plus_picks += 1
                    if pick.hit:
                        result.score_65_plus_hits += 1
                elif pick.factor_score >= 50:
                    result.score_50_65_picks += 1
                    if pick.hit:
                        result.score_50_65_hits += 1
                else:
                    result.score_40_50_picks += 1
                    if pick.hit:
                        result.score_40_50_hits += 1
                
                # By edge range
                if pick.edge_pct >= 20:
                    result.edge_20_plus_picks += 1
                    if pick.hit:
                        result.edge_20_plus_hits += 1
                elif pick.edge_pct >= 15:
                    result.edge_15_20_picks += 1
                    if pick.hit:
                        result.edge_15_20_hits += 1
                elif pick.edge_pct >= 10:
                    result.edge_10_15_picks += 1
                    if pick.hit:
                        result.edge_10_15_hits += 1
                else:
                    result.edge_6_10_picks += 1
                    if pick.hit:
                        result.edge_6_10_hits += 1
                
                result.all_picks.append(pick)
            
            # Daily summary
            result.daily_results.append({
                "date": game_date,
                "games": daily.games,
                "picks": daily_picks,
                "hits": daily_hits,
                "hit_rate": daily_hits / daily_picks * 100 if daily_picks > 0 else 0,
            })
        
        if show_progress:
            print()  # New line after progress bar
    
    if verbose:
        print(result.summary())
    
    return result


# ============================================================================
# Utility Functions
# ============================================================================

def get_actual_stats(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    prop_type: str,
) -> Optional[float]:
    """Get actual stat value for a player on a specific date."""
    row = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ? AND g.game_date = ?
        LIMIT 1
        """,
        (player_id, game_date),
    ).fetchone()
    
    if not row:
        return None
    
    pt = prop_type.lower()
    return row[pt] if pt in ['pts', 'reb', 'ast'] else None


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V18.5 Under - NBA Props UNDER Prediction")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Picks command
    picks_parser = subparsers.add_parser("picks", help="Generate daily UNDER picks")
    picks_parser.add_argument("--date", "-d", required=True, help="Date (YYYY-MM-DD)")
    picks_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument("--start", "-s", required=True, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", "-e", required=True, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    bt_parser.add_argument("--no-progress", action="store_true", help="Hide progress bar")
    
    args = parser.parse_args()
    
    if args.command == "picks":
        result = get_daily_picks_v18_under(args.date, verbose=True)
    elif args.command == "backtest":
        result = run_backtest_v18_under(
            args.start, args.end,
            verbose=args.verbose,
            show_progress=not args.no_progress,
        )
        print(result.summary())
    else:
        parser.print_help()
