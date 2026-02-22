"""
Model V19 Shared Utilities
===========================

Common functions, data classes, and utilities shared between:
- Model V19 General (Holistic multi-factor approach for all picks)
- Model V19 Under (Specialized UNDER model - to be developed in Phase 2)

=============================================================================
MODEL V19 KEY INNOVATIONS (Addressing ALL Previous Model Shortcomings)
=============================================================================

1. **TRUE HOLISTIC ANALYSIS** (Not Just Cold Bounce or Single Patterns):
   - Previous models over-relied on single patterns (cold bounce alone)
   - V19 requires MULTIPLE factors to align before making a pick
   - Considers: game context, player trends, efficiency, defense, fatigue, 
     historical matchups, team dynamics, usage shifts
   - CRITICAL: Cold bounce is just ONE factor, not THE factor

2. **COMPREHENSIVE BOX SCORE ANALYSIS**:
   - Previous models only looked at PTS/REB/AST averages
   - V19 analyzes: Plus/Minus (+/-), FG%, TS%, FTA, minutes trends, usage
   - This addresses: "analyzing box scores and prior game data"

3. **HYBRID LINE APPROACH** (Always Generate Picks):
   - Use actual sportsbook lines when available (accurate edge)
   - ALWAYS generate picks even without lines (lines come late)
   - Different edge thresholds: 6% for sportsbook, 15% for derived
   - Track line source for honest reporting

4. **MARKET-AWARE VALIDATION**:
   - Previous models had "Derived Line Fallacy" - testing against averages
   - V19 uses actual sportsbook lines when available for accurate edge
   - Higher edge requirements when using derived lines (buffer for inaccuracy)

5. **STRATEGIC DIRECTION SELECTION** (Data-Driven from RCM v1.4):
   - PTS: UNDER strongly preferred (63.9% vs 48.3% OVER)
   - PTS OVER: Only with cold bounce + NOT vs elite defense
   - REB: Both directions (~59% each)
   - AST: EXCLUDED entirely (~54% is coin flip after juice)

6. **MULTI-WINDOW PROJECTION SYSTEM**:
   - L3: Very recent form (10% weight - volatile)
   - L5: Recent form (20% weight)
   - L10: Primary baseline (30% weight - most stable)
   - L15: Extended baseline (20% weight)
   - Season: True talent level (20% weight)

7. **VALIDATED FACTOR WEIGHTS** (From V16-V18 Backtesting):
   - Elite Defense: 71-74% hit rate → High weight
   - B2B Fatigue: 69-75% hit rate → High weight
   - Cold Bounce: 64-84% hit rate → Primary OVER trigger ONLY
   - Hot Form: 43% hit rate → ELIMINATED
   - Weak Defense OVERs: 43% hit rate → HEAVILY reduced weight

8. **HISTORICAL MATCHUP ANALYSIS**:
   - Track player performance vs specific opponents
   - Require 3+ games for H2H data to be meaningful
   - Identify consistent over/under performers against specific teams

9. **STRICT QUALITY FILTERING**:
   - 23+ minute average (established players only)
   - 10+ games history required
   - Minimum combined factor score required
   - Exclude volatile situations (injuries, trades, role changes)

10. **HONEST REPORTING**:
    - Track sportsbook vs derived picks separately
    - Report hit rates by line source
    - No inflated metrics from derived line fallacy

=============================================================================
HOLISTIC FACTOR SCORING SYSTEM (V19 Refined)
=============================================================================

FOR UNDER PICKS:
| Factor                  | Weight | Validated Hit Rate | Notes |
|------------------------|--------|-------------------|-------|
| Elite Defense (Top 3)  | 50     | 71-74% | PRIMARY DRIVER |
| B2B Fatigue (2nd game) | 40     | 69-75% | STRONG |
| Good Defense (Top 10)  | 28     | 60-67% | SOLID |
| Injury Rust (1st back) | 25     | 60-70% | VALIDATED |
| Cold Streak Mild       | 18     | 57-62% | SECONDARY |
| Minutes Decline        | 15     | 55-60% | CONTEXT |
| Third in Four Days     | 15     | 60-65% | FATIGUE |
| Poor H2H History       | 15     | 55-65% | HISTORICAL |
| Negative +/- Trend     | 12     | ~55% | BOX SCORE |
| Poor Efficiency Trend  | 12     | ~55% | BOX SCORE |
| High Variance Player   | 8      | ~55% | PLAYER PROFILE |
| Cold Streak Severe     | 5      | 48% | REQUIRES support! |

FOR OVER PICKS (MUCH STRICTER):
| Factor                  | Weight | Validated Hit Rate | Notes |
|------------------------|--------|-------------------|-------|
| Cold Bounce Recovery   | 40     | 64-84% | ONLY PRIMARY TRIGGER |
| Good H2H History       | 18     | 55-65% | HISTORICAL |
| Consistent Player (CV) | 15     | 55-60% | PREDICTABLE |
| Usage Boost (Star OUT) | 12     | 52-55% | USAGE SHIFT |
| Minutes Increase       | 10     | 55-60% | ROLE EXPANSION |
| Positive +/- Trend     | 8      | ~55% | BOX SCORE |
| Weak Defense (Bot 5)   | 5      | 43% | HEAVILY REDUCED |
| Hot Form               | 0      | 43% | ELIMINATED |

=============================================================================

Author: PropAI Team - Model V19
Created: February 2026
Version: 19.0
"""
from __future__ import annotations

import sqlite3
import statistics
import unicodedata
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any, Set
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev


# ============================================================================
# Version Info
# ============================================================================

MODEL_VERSION = "19.0"
MODEL_NAME = "Model V19"


# ============================================================================
# Constants
# ============================================================================

# Position mapping for defense vs position data (Hashtag Basketball)
POSITION_MAP = {
    'G': 'PG', 'PG': 'PG', 'SG': 'SG',
    'F': 'SF', 'SF': 'SF', 'PF': 'PF',
    'C': 'C', 'F-C': 'PF', 'C-F': 'PF',
    'G-F': 'SG', 'F-G': 'SG',
    'GUARD': 'PG', 'FORWARD': 'SF', 'CENTER': 'C',
    '': 'SF',  # Default fallback
}

# Defense rating thresholds (1 = best defense, 30 = worst)
ELITE_DEFENSE_RANK = 3      # Top 3 = elite defense (STRICTEST)
GOOD_DEFENSE_RANK = 10      # Top 10 = good defense
AVERAGE_DEFENSE_RANK = 15   # Top 15 = average
POOR_DEFENSE_RANK = 26      # Bottom 5 = weak defense

# Minimum thresholds for including props (filter low-volume players)
MIN_PROP_AVERAGES = {
    'pts': 8.0,     # Min 8 PPG to consider
    'reb': 4.0,     # Min 4 RPG to consider
    'ast': 8.5,     # Min 8.5 APG for AST picks (VERY high bar - usually excluded)
}

# Prop types supported (AST excluded by default - 54% is coin flip)
PROP_TYPES = ['pts', 'reb']


# ============================================================================
# Factor Weights - UNDER (Refined from V16/V17/V18 Backtesting)
# ============================================================================

UNDER_FACTOR_WEIGHTS = {
    # Defense factors - VALIDATED STRONG (Primary drivers)
    "defense_elite": 50,        # Top 3 DVP - 71-74% hit rate (INCREASED)
    "defense_good": 28,         # Top 4-10 DVP - 60-67% hit rate
    
    # Fatigue factors - VALIDATED STRONG
    "b2b_fatigue": 40,          # Second of back-to-back - 69-75% hit rate (INCREASED)
    "third_in_four": 15,        # Third game in 4 days - 60-65%
    
    # Form/trend factors - MIXED (be careful)
    "cold_streak_mild": 18,     # L5 < 90% of season - 57-62% hit rate
    "cold_streak_severe": 5,    # L5 < 80% of season - 48% (REQUIRES support!)
    "minutes_decline": 15,      # L5 min < L15 min by 10%+
    
    # Special situations - VALIDATED
    "injury_rust_first": 25,    # First game back from injury - 60-70%
    "injury_rust_second": 10,   # Second game back
    
    # V19: Enhanced Box Score Analysis Factors
    "negative_plus_minus": 12,  # Consistently negative +/- (L5 avg < -5)
    "poor_efficiency_trend": 12,  # FG% declining (L5 < L15 by 5%+)
    "poor_ts_trend": 10,        # True Shooting % declining
    "low_fta_trend": 8,         # FTA declining (less aggressive)
    
    # Player characteristics
    "high_variance": 8,         # CV > 0.40 (inconsistent player)
    
    # Historical matchup - MEANINGFUL with enough data
    "poor_h2h_history": 15,     # Below avg vs this opponent (3+ games)
    
    # Team context (NEW in V19)
    "blowout_risk": 10,         # Large spread games - garbage time risk
    "pace_factor_slow": 8,      # Opponent plays slow pace
}

# ============================================================================
# Factor Weights - OVER (MUCH STRICTER - OVERs are generally riskier)
# ============================================================================

OVER_FACTOR_WEIGHTS = {
    # Pattern factors - PRIMARY OVER TRIGGER (ONLY reliable pattern)
    "cold_bounce": 40,          # L5 < L15 but last game > L10 - 64-84% hit rate
    
    # Defense factors - CAUTION: weak defense showed only 43%
    "defense_weak": 5,          # Bottom 5 DVP - HEAVILY REDUCED from 20
    "defense_poor": 3,          # Bottom 10 DVP - MINIMAL weight
    
    # Usage factors - theoretical but inconsistent
    "usage_boost_major": 12,    # Star teammate OUT (15+ PPG)
    "usage_boost_minor": 5,     # Role player OUT (10+ PPG)
    
    # V19: Enhanced Box Score Analysis Factors
    "positive_plus_minus": 8,   # Consistently positive +/- (L5 avg > +5)
    "good_efficiency_trend": 8, # FG% improving (L5 > L15 by 5%+)
    "high_fta_trend": 8,        # FTA increasing (more aggressive)
    
    # Player characteristics - IMPORTANT for OVERs
    "consistent_player": 15,    # CV < 0.20 (very predictable)
    "minutes_increase": 10,     # L5 min > L15 min by 5%+
    
    # Historical matchup - MEANINGFUL
    "good_h2h_history": 18,     # Above avg vs opponent (3+ games)
    
    # Team context (NEW in V19)
    "pace_factor_fast": 8,      # Opponent plays fast pace
    
    # ELIMINATED - proved unreliable
    "hot_form": 0,              # L3 > L10 - 43% hit rate = NEGATIVE edge
}


# ============================================================================
# Factor Score Thresholds (Tuned from V17/V18 backtesting)
# ============================================================================

MIN_FACTOR_SCORE_PREMIUM = 65   # High bar for best picks (INCREASED)
MIN_FACTOR_SCORE_HIGH = 50      # Good picks (INCREASED)
MIN_FACTOR_SCORE_STANDARD = 40  # Minimum acceptable (INCREASED)

# For OVER picks, even stricter thresholds
MIN_FACTOR_SCORE_OVER_PREMIUM = 55
MIN_FACTOR_SCORE_OVER_HIGH = 45
MIN_FACTOR_SCORE_OVER_STANDARD = 35

# Projection adjustments based on factors
FACTOR_PROJECTION_ADJUSTMENTS = {
    # UNDER adjustments (reduce projection)
    "defense_elite": 0.85,      # -15% (STRONGER adjustment)
    "defense_good": 0.93,       # -7%
    "cold_streak_severe": 0.90, # -10%
    "cold_streak_mild": 0.95,   # -5%
    "b2b_fatigue": 0.93,        # -7% (STRONGER - fatigue is real)
    "third_in_four": 0.96,      # -4%
    "injury_rust_first": 0.82,  # -18% (STRONGER - rust is significant)
    "injury_rust_second": 0.92, # -8%
    "negative_plus_minus": 0.97, # -3%
    "poor_efficiency_trend": 0.97, # -3%
    "poor_ts_trend": 0.97,      # -3%
    "minutes_decline": 0.95,    # -5%
    "blowout_risk": 0.94,       # -6%
    "pace_factor_slow": 0.97,   # -3%
    
    # OVER adjustments (increase projection) - CONSERVATIVE
    "defense_weak": 1.03,       # +3% (HEAVILY reduced from 10%)
    "defense_poor": 1.02,       # +2%
    "usage_boost_major": 1.06,  # +6% (reduced)
    "usage_boost_minor": 1.03,  # +3%
    "cold_bounce": 1.03,        # +3% (regression expected)
    "positive_plus_minus": 1.02, # +2%
    "good_efficiency_trend": 1.02, # +2%
    "minutes_increase": 1.03,   # +3%
    "pace_factor_fast": 1.03,   # +3%
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LineInfo:
    """
    Information about a betting line (sportsbook or derived).
    
    KEY FIELD: source tells us if this is a real betting line or derived.
    This is critical for honest model validation.
    """
    line: float
    source: str  # "sportsbook" or "derived"
    book: Optional[str] = None  # e.g., "draftkings", "fanduel"
    odds_american: Optional[int] = None  # e.g., -110
    
    @property
    def is_sportsbook(self) -> bool:
        return self.source == "sportsbook"
    
    @property
    def is_derived(self) -> bool:
        return self.source == "derived"


@dataclass
class EfficiencyStats:
    """
    V19: Enhanced shooting efficiency metrics for deeper analysis.
    """
    # Field Goals
    l5_fg_pct: float = 0.0
    l10_fg_pct: float = 0.0
    l15_fg_pct: float = 0.0
    season_fg_pct: float = 0.0
    
    # True Shooting %: PTS / (2 * (FGA + 0.44 * FTA))
    l5_ts_pct: float = 0.0
    l10_ts_pct: float = 0.0
    season_ts_pct: float = 0.0
    
    # Free Throw Attempts (aggression indicator)
    l5_fta: float = 0.0
    l10_fta: float = 0.0
    season_fta: float = 0.0
    
    # Plus/Minus trends
    l5_plus_minus_avg: float = 0.0
    l10_plus_minus_avg: float = 0.0
    season_plus_minus_avg: float = 0.0
    
    # Usage proxy (FGA + 0.44*FTA per minute)
    l5_usage_proxy: float = 0.0
    l10_usage_proxy: float = 0.0
    
    def get_fg_trend(self) -> float:
        """Return FG% trend: (L5 - L15) / L15 * 100"""
        if self.l15_fg_pct <= 0:
            return 0.0
        return (self.l5_fg_pct - self.l15_fg_pct) / self.l15_fg_pct * 100
    
    def get_ts_trend(self) -> float:
        """Return TS% trend: (L5 - Season) / Season * 100"""
        if self.season_ts_pct <= 0:
            return 0.0
        return (self.l5_ts_pct - self.season_ts_pct) / self.season_ts_pct * 100
    
    def get_fta_trend(self) -> float:
        """Return FTA trend: (L5 - Season) / Season * 100"""
        if self.season_fta <= 0:
            return 0.0
        return (self.l5_fta - self.season_fta) / self.season_fta * 100
    
    def is_efficiency_declining(self, threshold: float = -5.0) -> bool:
        """Check if FG% is declining by threshold %"""
        return self.get_fg_trend() < threshold
    
    def is_efficiency_improving(self, threshold: float = 5.0) -> bool:
        """Check if FG% is improving by threshold %"""
        return self.get_fg_trend() > threshold
    
    def is_ts_declining(self, threshold: float = -5.0) -> bool:
        """Check if TS% is declining"""
        return self.get_ts_trend() < threshold
    
    def has_negative_plus_minus(self, threshold: float = -5.0) -> bool:
        """Check if L5 avg +/- is below threshold"""
        return self.l5_plus_minus_avg < threshold
    
    def has_positive_plus_minus(self, threshold: float = 5.0) -> bool:
        """Check if L5 avg +/- is above threshold"""
        return self.l5_plus_minus_avg > threshold
    
    def is_less_aggressive(self, threshold: float = -15.0) -> bool:
        """Check if FTA is declining (less aggressive)"""
        return self.get_fta_trend() < threshold
    
    def is_more_aggressive(self, threshold: float = 15.0) -> bool:
        """Check if FTA is increasing (more aggressive)"""
        return self.get_fta_trend() > threshold


@dataclass
class HistoricalMatchup:
    """
    Historical performance vs a specific opponent.
    CRITICAL: Require 3+ games for meaningful data.
    """
    opponent_abbrev: str
    games_played: int = 0
    avg_pts: float = 0.0
    avg_reb: float = 0.0
    avg_ast: float = 0.0
    avg_minutes: float = 0.0
    
    # Comparison to overall averages
    pts_vs_season_pct: float = 0.0  # +10 means 10% above season avg
    reb_vs_season_pct: float = 0.0
    ast_vs_season_pct: float = 0.0
    
    def has_sufficient_data(self, min_games: int = 3) -> bool:
        """Check if we have enough games for meaningful analysis."""
        return self.games_played >= min_games
    
    def is_good_matchup(self, prop_type: str, threshold: float = 10.0) -> bool:
        """Check if player performs above average vs this opponent."""
        if not self.has_sufficient_data():
            return False
        mapping = {'pts': self.pts_vs_season_pct, 'reb': self.reb_vs_season_pct, 'ast': self.ast_vs_season_pct}
        return mapping.get(prop_type.lower(), 0) >= threshold
    
    def is_poor_matchup(self, prop_type: str, threshold: float = -10.0) -> bool:
        """Check if player performs below average vs this opponent."""
        if not self.has_sufficient_data():
            return False
        mapping = {'pts': self.pts_vs_season_pct, 'reb': self.reb_vs_season_pct, 'ast': self.ast_vs_season_pct}
        return mapping.get(prop_type.lower(), 0) <= threshold


@dataclass
class GameContext:
    """
    V19 NEW: Game-level context for better predictions.
    """
    spread: float = 0.0          # Point spread (negative = home favored)
    over_under: float = 0.0      # Total points O/U
    is_home: bool = False
    
    # Pace factors (derived from O/U and team pace data)
    expected_pace: str = "normal"  # "fast", "normal", "slow"
    
    # Blowout risk
    is_blowout_risk: bool = False  # Spread > 10 points
    
    def __post_init__(self):
        # Determine blowout risk
        if abs(self.spread) >= 10:
            self.is_blowout_risk = True
        
        # Determine expected pace from O/U
        if self.over_under >= 235:
            self.expected_pace = "fast"
        elif self.over_under <= 215:
            self.expected_pace = "slow"


@dataclass
class PlayerStatsV19:
    """
    Comprehensive player statistics for Model V19.
    
    ENHANCED from V18 to include:
    - More granular efficiency tracking
    - Usage proxy calculation
    - Better game-by-game data storage
    """
    player_id: int
    player_name: str
    team_abbrev: str
    position: str
    games_played: int
    avg_minutes: float
    
    # Averages at different windows (keys: pts, reb, ast, min)
    l3: Dict[str, float] = field(default_factory=dict)
    l5: Dict[str, float] = field(default_factory=dict)
    l10: Dict[str, float] = field(default_factory=dict)
    l15: Dict[str, float] = field(default_factory=dict)
    l20: Dict[str, float] = field(default_factory=dict)
    season: Dict[str, float] = field(default_factory=dict)
    
    # Deviations: L5 vs L15, L5 vs Season (percentage)
    deviations_l15: Dict[str, float] = field(default_factory=dict)
    deviations_season: Dict[str, float] = field(default_factory=dict)
    
    # L3 deviations (NEW in V19 - for detecting very recent trends)
    deviations_l3_vs_l10: Dict[str, float] = field(default_factory=dict)
    
    # Last game values
    last_game: Dict[str, float] = field(default_factory=dict)
    second_last_game: Dict[str, float] = field(default_factory=dict)
    
    # Standard deviations (L10 window) - for CV calculation
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Recent game values (last 5) for pattern analysis
    recent_games: Dict[str, List[float]] = field(default_factory=dict)
    
    # V19: Enhanced Efficiency stats
    efficiency: EfficiencyStats = field(default_factory=EfficiencyStats)
    
    # Historical vs specific opponent
    vs_opponent: Optional[HistoricalMatchup] = None
    
    # Minutes trends
    l5_minutes: float = 0.0
    l15_minutes: float = 0.0
    
    # Game dates for injury detection
    last_game_date: Optional[str] = None
    days_since_last_game: int = 1
    
    # V19: Raw game data for deeper analysis
    recent_game_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # V19: Consistency metrics
    games_above_avg: int = 0  # Out of last 10
    games_below_avg: int = 0
    
    # V19.1: Trade deadline awareness
    was_traded: bool = False
    new_team_games: int = 0  # Number of games with current (new) team
    trade_date: Optional[str] = None
    old_team_abbrev: Optional[str] = None
    trade_confidence_discount: float = 1.0  # 0.3-1.0 based on new-team games
    
    def get_projection(
        self, 
        prop_type: str, 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted projection for a prop type.
        
        Default weights (validated from backtesting):
        - L3: 0.10 (very recent form - volatile but recent)
        - L5: 0.20 (recent form)
        - L10: 0.30 (primary baseline - most stable)
        - L15: 0.20 (extended baseline)
        - Season: 0.20 (true talent level)
        
        V19.1 TRADE-AWARE: For traded players, shift weight heavily toward
        recent (new-team) games. Season/L15/L20 data includes old-team
        performance which may not be representative of new context.
        
        Trade-aware weight shift by new_team_games:
        - 1-2 games: L3=0.40, L5=0.30, L10=0.15, L15=0.10, Season=0.05
        - 3-5 games: L3=0.20, L5=0.35, L10=0.25, L15=0.10, Season=0.10
        - 6-10 games: L3=0.15, L5=0.25, L10=0.30, L15=0.15, Season=0.15
        - 10+ games: standard weights
        """
        if weights is None:
            if self.was_traded and self.new_team_games < 10:
                # Shift weight toward most recent games (new-team data)
                if self.new_team_games <= 2:
                    weights = {
                        'l3': 0.40, 'l5': 0.30, 'l10': 0.15,
                        'l15': 0.10, 'season': 0.05,
                    }
                elif self.new_team_games <= 5:
                    weights = {
                        'l3': 0.20, 'l5': 0.35, 'l10': 0.25,
                        'l15': 0.10, 'season': 0.10,
                    }
                else:  # 6-9 games
                    weights = {
                        'l3': 0.15, 'l5': 0.25, 'l10': 0.30,
                        'l15': 0.15, 'season': 0.15,
                    }
            else:
                weights = {
                    'l3': 0.10, 'l5': 0.20, 'l10': 0.30,
                    'l15': 0.20, 'season': 0.20,
                }
        
        pt = prop_type.lower()
        
        values = {
            'l3': self.l3.get(pt, 0),
            'l5': self.l5.get(pt, 0),
            'l10': self.l10.get(pt, 0),
            'l15': self.l15.get(pt, 0),
            'season': self.season.get(pt, 0),
        }
        
        total_weight = sum(weights.values())
        if total_weight <= 0:
            return values.get('season', 0)
        
        projection = sum(values[k] * weights.get(k, 0) for k in values)
        return projection / total_weight
    
    def get_cv(self, prop_type: str) -> float:
        """
        Get coefficient of variation (std/mean) for consistency analysis.
        
        CV < 0.20 = Very consistent player (confidence boost for OVER)
        CV > 0.40 = Volatile player (confidence boost for UNDER)
        """
        pt = prop_type.lower()
        mean = self.l10.get(pt, 0)
        std = self.stds.get(pt, 0)
        if mean <= 0:
            return 1.0
        return std / mean
    
    def get_deviation_l15(self, prop_type: str) -> float:
        """Get L5 vs L15 deviation percentage."""
        return self.deviations_l15.get(prop_type.lower(), 0)
    
    def get_deviation_season(self, prop_type: str) -> float:
        """Get L5 vs Season deviation percentage."""
        return self.deviations_season.get(prop_type.lower(), 0)
    
    def get_deviation_l3_vs_l10(self, prop_type: str) -> float:
        """Get L3 vs L10 deviation percentage (very recent trend)."""
        return self.deviations_l3_vs_l10.get(prop_type.lower(), 0)
    
    def get_minutes_trend(self) -> float:
        """
        Get minutes trend as percentage change.
        Returns: (L5_min - L15_min) / L15_min * 100
        """
        if self.l15_minutes <= 0:
            return 0.0
        return (self.l5_minutes - self.l15_minutes) / self.l15_minutes * 100
    
    def get_consistency_score(self, prop_type: str) -> float:
        """
        Get a 0-100 consistency score.
        Higher = more consistent player.
        """
        cv = self.get_cv(prop_type)
        # CV of 0 = 100, CV of 0.5+ = ~0
        score = max(0, min(100, (0.5 - cv) * 200))
        return score


@dataclass
class DefenseContextV19:
    """
    Defense vs position context for an opponent team.
    """
    team_abbrev: str
    position: str
    data_available: bool = False
    
    # Ranks (1 = best defense, 30 = worst)
    pts_rank: int = 15
    reb_rank: int = 15
    ast_rank: int = 15
    
    # Allowed values (per game)
    pts_allowed: float = 0.0
    reb_allowed: float = 0.0
    ast_allowed: float = 0.0
    
    # Ratings (derived from rank)
    pts_rating: str = "average"
    reb_rating: str = "average"
    ast_rating: str = "average"
    
    def get_rank(self, prop_type: str) -> int:
        """Get defense rank for a prop type (1=best, 30=worst)."""
        mapping = {'pts': self.pts_rank, 'reb': self.reb_rank, 'ast': self.ast_rank}
        return mapping.get(prop_type.lower(), 15)
    
    def get_rating(self, prop_type: str) -> str:
        """Get defense rating for a prop type."""
        mapping = {'pts': self.pts_rating, 'reb': self.reb_rating, 'ast': self.ast_rating}
        return mapping.get(prop_type.lower(), "average")
    
    def is_elite(self, prop_type: str) -> bool:
        """Check if defense is elite (top 3) for this prop type."""
        return self.get_rank(prop_type) <= ELITE_DEFENSE_RANK
    
    def is_good(self, prop_type: str) -> bool:
        """Check if defense is good (top 10) for this prop type."""
        rank = self.get_rank(prop_type)
        return ELITE_DEFENSE_RANK < rank <= GOOD_DEFENSE_RANK
    
    def is_weak(self, prop_type: str) -> bool:
        """Check if defense is weak (bottom 5) for this prop type."""
        return self.get_rank(prop_type) >= POOR_DEFENSE_RANK
    
    def is_poor(self, prop_type: str) -> bool:
        """Check if defense is poor (bottom 10 but not bottom 5)."""
        rank = self.get_rank(prop_type)
        return 20 <= rank < POOR_DEFENSE_RANK


@dataclass
class BackToBackInfo:
    """
    Information about team's rest/fatigue status.
    """
    is_b2b: bool = False
    is_second_of_b2b: bool = False
    is_third_in_four: bool = False
    is_fourth_in_six: bool = False  # NEW in V19
    days_rest: int = 1
    
    def has_fatigue_factor(self) -> bool:
        """Check if team has any fatigue factor."""
        return self.is_second_of_b2b or self.is_third_in_four or self.is_fourth_in_six


@dataclass
class InjuryImpact:
    """
    Impact of injured teammates on a player's projection.
    """
    injured_teammates: List[Dict[str, Any]] = field(default_factory=list)
    total_pts_out: float = 0.0
    total_reb_out: float = 0.0
    total_ast_out: float = 0.0
    total_minutes_out: float = 0.0
    usage_boost_pct: float = 0.0
    
    def has_significant_impact(self) -> bool:
        """Check if injuries create meaningful usage boost."""
        return self.usage_boost_pct >= 5.0 or self.total_pts_out >= 15.0


@dataclass
class HolisticFactorScore:
    """
    V19 Holistic Factor Score for a pick.
    
    Combines multiple factors into a single score with detailed breakdown.
    KEY DIFFERENCE from V18: Requires multiple factors to align.
    """
    total_score: float = 0.0
    direction: str = ""  # "OVER" or "UNDER"
    
    # Individual factor contributions
    factors: Dict[str, float] = field(default_factory=dict)
    
    # Primary factor (highest weight contributor)
    primary_factor: str = ""
    primary_factor_weight: float = 0.0
    
    # Secondary factor (second highest)
    secondary_factor: str = ""
    secondary_factor_weight: float = 0.0
    
    # Confidence based on number and strength of factors
    factor_count: int = 0
    
    # Projection adjustment from factors
    projection_adjustment: float = 1.0
    
    # Reasons/explanations for each factor
    reasons: List[str] = field(default_factory=list)
    
    # V19: Factor alignment score (0-100)
    # Higher when factors reinforce each other
    alignment_score: float = 0.0
    
    def get_tier(self) -> str:
        """Get confidence tier based on total score."""
        if self.direction == "OVER":
            if self.total_score >= MIN_FACTOR_SCORE_OVER_PREMIUM:
                return "PREMIUM"
            elif self.total_score >= MIN_FACTOR_SCORE_OVER_HIGH:
                return "HIGH"
            elif self.total_score >= MIN_FACTOR_SCORE_OVER_STANDARD:
                return "STANDARD"
        else:
            if self.total_score >= MIN_FACTOR_SCORE_PREMIUM:
                return "PREMIUM"
            elif self.total_score >= MIN_FACTOR_SCORE_HIGH:
                return "HIGH"
            elif self.total_score >= MIN_FACTOR_SCORE_STANDARD:
                return "STANDARD"
        return "BELOW_THRESHOLD"
    
    def has_multiple_strong_factors(self, min_factors: int = 2, min_score: float = 15.0) -> bool:
        """
        V19 KEY: Check if pick has multiple strong factors.
        This is critical for avoiding single-factor picks.
        """
        strong_factors = [w for w in self.factors.values() if w >= min_score]
        return len(strong_factors) >= min_factors


# ============================================================================
# Name Normalization
# ============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize player name for matching.
    
    Handles accents, suffixes, case differences.
    """
    if not name:
        return ""
    
    # Normalize Unicode (remove accents)
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    # Remove suffixes
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']:
        if ascii_name.lower().endswith(suffix):
            ascii_name = ascii_name[:-len(suffix)]
    
    return ascii_name.lower().strip()


def map_position(pos: str) -> str:
    """Map various position formats to standard DVP positions (PG, SG, SF, PF, C)."""
    if not pos:
        return "SF"
    return POSITION_MAP.get(pos.upper().strip(), "SF")


# ============================================================================
# Database Query Functions
# ============================================================================

def get_injured_players(conn: sqlite3.Connection, game_date: str) -> Set[int]:
    """
    Get set of player IDs who are OUT or DOUBTFUL for the date.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT COALESCE(ir.player_id, p.id) as pid
        FROM injury_report ir
        LEFT JOIN players p ON LOWER(p.name) = LOWER(ir.player_name)
        WHERE ir.game_date = ?
          AND ir.status IN ('OUT', 'DOUBTFUL')
        """,
        (game_date,),
    ).fetchall()
    
    return {row["pid"] for row in rows if row["pid"]}


def get_injured_players_for_team(
    conn: sqlite3.Connection, 
    game_date: str, 
    team_abbrev: str
) -> List[Dict[str, Any]]:
    """
    Get injured players for a specific team with their stats.
    Used for usage redistribution calculations.
    """
    # Get team ID
    team_row = conn.execute(
        """
        SELECT id FROM teams 
        WHERE name LIKE ? OR name LIKE ?
        LIMIT 1
        """,
        (f"%{team_abbrev}%", f"{team_abbrev}%")
    ).fetchone()
    
    if not team_row:
        return []
    
    team_id = team_row["id"]
    
    # Get injured players
    rows = conn.execute(
        """
        SELECT DISTINCT p.id as player_id, p.name as player_name, ir.status
        FROM injury_report ir
        LEFT JOIN players p ON ir.player_id = p.id OR LOWER(p.name) = LOWER(ir.player_name)
        WHERE ir.game_date = ?
          AND ir.status IN ('OUT', 'DOUBTFUL')
        """,
        (game_date,),
    ).fetchall()
    
    injured = []
    for row in rows:
        if not row["player_id"]:
            continue
        
        # Get player's average stats
        stats = conn.execute(
            """
            SELECT 
                AVG(bp.pts) as avg_pts,
                AVG(bp.reb) as avg_reb,
                AVG(bp.ast) as avg_ast,
                AVG(bp.minutes) as avg_min,
                MAX(bp.team_id) as last_team_id
            FROM boxscore_player bp
            JOIN games g ON g.id = bp.game_id
            WHERE bp.player_id = ?
              AND g.game_date < ?
              AND bp.minutes > 10
            ORDER BY g.game_date DESC
            LIMIT 15
            """,
            (row["player_id"], game_date),
        ).fetchone()
        
        # Check if player is on the target team
        if stats and stats["last_team_id"] == team_id and stats["avg_pts"]:
            injured.append({
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "status": row["status"],
                "avg_pts": stats["avg_pts"] or 0,
                "avg_reb": stats["avg_reb"] or 0,
                "avg_ast": stats["avg_ast"] or 0,
                "avg_min": stats["avg_min"] or 0,
            })
    
    return injured


def get_sportsbook_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Optional[LineInfo]:
    """
    Get sportsbook line for a player/prop/date.
    
    Returns LineInfo with source="sportsbook" if found, None otherwise.
    """
    # Try by player_id first
    if player_id:
        row = conn.execute(
            """
            SELECT line, book, odds_american
            FROM sportsbook_lines
            WHERE player_id = ? AND prop_type = ? AND as_of_date = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (player_id, prop_type.upper(), game_date)
        ).fetchone()
        
        if row:
            return LineInfo(
                line=row["line"],
                source="sportsbook",
                book=row["book"] or "unknown",
                odds_american=row["odds_american"]
            )
    
    # Try by fuzzy name match
    rows = conn.execute(
        """
        SELECT sl.line, sl.book, sl.odds_american, p.name
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.prop_type = ? AND sl.as_of_date = ?
        """,
        (prop_type.upper(), game_date)
    ).fetchall()
    
    norm_target = normalize_name(player_name)
    for row in rows:
        if normalize_name(row["name"]) == norm_target:
            return LineInfo(
                line=row["line"],
                source="sportsbook",
                book=row["book"] or "unknown",
                odds_american=row["odds_american"]
            )
    
    return None


def get_derived_line(
    stats: PlayerStatsV19,
    prop_type: str,
    adjustment: float = 1.05
) -> LineInfo:
    """
    Calculate a derived line based on player's L10 average.
    
    We apply a 5% adjustment upward since sportsbook lines
    tend to be slightly higher than player averages.
    """
    pt = prop_type.lower()
    l10_avg = stats.l10.get(pt, 0)
    
    # Apply adjustment
    derived = l10_avg * adjustment
    
    # Round to nearest 0.5 (standard for prop lines)
    derived = round(derived * 2) / 2
    
    return LineInfo(line=derived, source="derived", book=None)


def get_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    stats: PlayerStatsV19,
    derived_adjustment: float = 1.05
) -> LineInfo:
    """
    Get line for a player - sportsbook if available, derived otherwise.
    
    ALWAYS returns a line (sportsbook or derived) so picks can be generated
    even when lines aren't available yet.
    """
    # Try sportsbook first
    sportsbook = get_sportsbook_line(conn, player_id, player_name, prop_type, game_date)
    if sportsbook:
        return sportsbook
    
    # Fall back to derived
    return get_derived_line(stats, prop_type, derived_adjustment)


def get_defense_context(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
) -> DefenseContextV19:
    """
    Get defense vs position context for an opponent team.
    """
    context = DefenseContextV19(
        team_abbrev=team_abbrev,
        position=position,
    )
    
    # Map to DVP position
    dvp_position = map_position(position)
    
    # Query DVP data
    row = conn.execute(
        """
        SELECT pts_rank, reb_rank, ast_rank, pts_allowed, reb_allowed, ast_allowed
        FROM team_defense_vs_position
        WHERE team_abbrev = ? AND position = ?
        ORDER BY updated_at DESC LIMIT 1
        """,
        (team_abbrev.upper(), dvp_position)
    ).fetchone()
    
    if not row:
        # Try with normalized abbrev
        normalized = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
        row = conn.execute(
            """
            SELECT pts_rank, reb_rank, ast_rank, pts_allowed, reb_allowed, ast_allowed
            FROM team_defense_vs_position
            WHERE UPPER(team_abbrev) = ? AND position = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (normalized, dvp_position)
        ).fetchone()
    
    if row:
        context.data_available = True
        context.pts_rank = row["pts_rank"] or 15
        context.reb_rank = row["reb_rank"] or 15
        context.ast_rank = row["ast_rank"] or 15
        context.pts_allowed = row["pts_allowed"] or 0
        context.reb_allowed = row["reb_allowed"] or 0
        context.ast_allowed = row["ast_allowed"] or 0
        
        # Determine ratings based on rank thresholds
        for stat, rank in [('pts', context.pts_rank), ('reb', context.reb_rank), ('ast', context.ast_rank)]:
            if rank <= ELITE_DEFENSE_RANK:
                rating = "elite"
            elif rank <= GOOD_DEFENSE_RANK:
                rating = "good"
            elif rank <= AVERAGE_DEFENSE_RANK:
                rating = "average"
            else:
                rating = "weak"
            
            if stat == 'pts':
                context.pts_rating = rating
            elif stat == 'reb':
                context.reb_rating = rating
            else:
                context.ast_rating = rating
    
    return context


def get_back_to_back_status(
    conn: sqlite3.Connection,
    team_abbrev: str,
    game_date: str,
) -> BackToBackInfo:
    """
    Check if team is on back-to-back or has other fatigue factors.
    """
    info = BackToBackInfo()
    
    # Get team ID
    team_row = conn.execute(
        """
        SELECT t.id FROM teams t
        WHERE t.name LIKE ?
        LIMIT 1
        """,
        (f"%{team_abbrev}%",)
    ).fetchone()
    
    if not team_row:
        return info
    
    team_id = team_row["id"]
    
    # Parse game date
    try:
        gd = datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError:
        return info
    
    # Check last 6 days of games (for fourth in six check)
    start_date = (gd - timedelta(days=6)).strftime("%Y-%m-%d")
    
    rows = conn.execute(
        """
        SELECT DISTINCT g.game_date
        FROM games g
        WHERE (g.team1_id = ? OR g.team2_id = ?)
          AND g.game_date >= ?
          AND g.game_date < ?
        ORDER BY g.game_date DESC
        """,
        (team_id, team_id, start_date, game_date)
    ).fetchall()
    
    if not rows:
        return info
    
    game_dates = [r["game_date"] for r in rows]
    
    # Check if played yesterday (B2B)
    yesterday = (gd - timedelta(days=1)).strftime("%Y-%m-%d")
    info.is_b2b = yesterday in game_dates
    info.is_second_of_b2b = info.is_b2b
    
    # Check third in four days
    four_days_ago = (gd - timedelta(days=4)).strftime("%Y-%m-%d")
    recent_4_games = [d for d in game_dates if d >= four_days_ago]
    if len(recent_4_games) >= 2:
        info.is_third_in_four = True
    
    # Check fourth in six days (NEW in V19)
    if len(game_dates) >= 3:
        info.is_fourth_in_six = True
    
    # Calculate rest days
    if game_dates:
        last_game = datetime.strptime(game_dates[0], "%Y-%m-%d")
        info.days_rest = (gd - last_game).days
    
    return info


def get_game_context(
    conn: sqlite3.Connection,
    team_abbrev: str,
    opponent_abbrev: str,
    game_date: str,
) -> GameContext:
    """
    V19 NEW: Get game-level context (spread, O/U, pace).
    """
    context = GameContext()
    
    # Try to get from scheduled_games or game_lines
    row = conn.execute(
        """
        SELECT gl.spread, gl.over_under
        FROM game_lines gl
        JOIN teams t1 ON t1.id = gl.away_team_id
        JOIN teams t2 ON t2.id = gl.home_team_id
        WHERE gl.game_date = ?
          AND (t1.name LIKE ? OR t2.name LIKE ?)
        LIMIT 1
        """,
        (game_date, f"%{team_abbrev}%", f"%{team_abbrev}%")
    ).fetchone()
    
    if row:
        context.spread = row["spread"] or 0
        context.over_under = row["over_under"] or 0
    
    # Try scheduled_games as fallback
    if context.over_under == 0:
        row2 = conn.execute(
            """
            SELECT sg.spread, sg.over_under
            FROM scheduled_games sg
            JOIN teams t1 ON t1.id = sg.away_team_id
            JOIN teams t2 ON t2.id = sg.home_team_id
            WHERE sg.game_date = ?
              AND (t1.name LIKE ? OR t2.name LIKE ?)
            LIMIT 1
            """,
            (game_date, f"%{team_abbrev}%", f"%{team_abbrev}%")
        ).fetchone()
        
        if row2:
            context.spread = row2["spread"] or 0
            context.over_under = row2["over_under"] or 0
    
    # Determine home/away
    home_row = conn.execute(
        """
        SELECT t.name FROM scheduled_games sg
        JOIN teams t ON t.id = sg.home_team_id
        WHERE sg.game_date = ?
          AND t.name LIKE ?
        LIMIT 1
        """,
        (game_date, f"%{team_abbrev}%")
    ).fetchone()
    
    context.is_home = home_row is not None
    
    return context


def get_historical_matchup(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    before_date: str,
    season_stats: Dict[str, float],
    max_games: int = 10,
) -> Optional[HistoricalMatchup]:
    """
    Get player's historical performance vs a specific opponent.
    """
    # Get opponent team ID
    opp_row = conn.execute(
        """
        SELECT id FROM teams WHERE name LIKE ?
        LIMIT 1
        """,
        (f"%{opponent_abbrev}%",)
    ).fetchone()
    
    if not opp_row:
        return None
    
    opp_id = opp_row["id"]
    
    # Get games against this opponent
    rows = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
          AND g.game_date < ?
          AND (g.team1_id = ? OR g.team2_id = ?)
          AND bp.minutes > 5
        ORDER BY g.game_date DESC
        LIMIT ?
        """,
        (player_id, before_date, opp_id, opp_id, max_games)
    ).fetchall()
    
    if len(rows) < 2:  # Need at least 2 games for any H2H data
        return None
    
    games = [dict(r) for r in rows]
    n = len(games)
    
    matchup = HistoricalMatchup(
        opponent_abbrev=opponent_abbrev,
        games_played=n,
        avg_pts=sum(g["pts"] or 0 for g in games) / n,
        avg_reb=sum(g["reb"] or 0 for g in games) / n,
        avg_ast=sum(g["ast"] or 0 for g in games) / n,
        avg_minutes=sum(g["minutes"] or 0 for g in games) / n,
    )
    
    # Calculate vs season averages
    if season_stats.get("pts", 0) > 0:
        matchup.pts_vs_season_pct = (matchup.avg_pts - season_stats["pts"]) / season_stats["pts"] * 100
    if season_stats.get("reb", 0) > 0:
        matchup.reb_vs_season_pct = (matchup.avg_reb - season_stats["reb"]) / season_stats["reb"] * 100
    if season_stats.get("ast", 0) > 0:
        matchup.ast_vs_season_pct = (matchup.avg_ast - season_stats["ast"]) / season_stats["ast"] * 100
    
    return matchup


def load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    opponent_abbrev: Optional[str] = None,
    min_games: int = 10,
    min_minutes: float = 20.0,
    max_games: int = 25,  # Increased to capture more data
    min_game_minutes: int = 5,
) -> Optional[PlayerStatsV19]:
    """
    Load comprehensive player statistics for analysis.
    
    V19 ENHANCED: Now includes more efficiency metrics and better trends.
    V19.1 TRADE-AWARE: Detects traded players, tags new-team vs old-team
    games, and adjusts projection weights accordingly. For traded players,
    the min_games requirement is relaxed to 3 games (new-team) if they
    have enough historical data combined.
    
    Returns None if player doesn't meet requirements.
    """
    # Get player info
    player = conn.execute(
        "SELECT id, name FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    
    if not player:
        return None
    
    # Get game history with FULL box score data
    rows = conn.execute(
        """
        SELECT 
            g.game_date, 
            b.pts, b.reb, b.ast, b.minutes, b.pos,
            b.fgm, b.fga, b.fg_pct,
            b.tpm, b.tpa, b.tp_pct,
            b.ftm, b.fta, b.ft_pct,
            b.plus_minus,
            b.oreb, b.dreb, b.stl, b.blk, b.tov,
            t.name as team_name,
            b.team_id
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        JOIN teams t ON t.id = b.team_id
        WHERE b.player_id = ?
          AND g.game_date < ?
          AND b.minutes IS NOT NULL
          AND b.minutes > ?
        ORDER BY g.game_date DESC
        LIMIT ?
        """,
        (player_id, before_date, min_game_minutes, max_games),
    ).fetchall()
    
    # =========================================================================
    # V19.1: Trade detection — check if team changed between games
    # =========================================================================
    was_traded = False
    trade_date = None
    old_team_abbrev = None
    new_team_games = 0
    trade_confidence_discount = 1.0
    
    if len(rows) >= 2:
        current_team_id = rows[0]["team_id"]
        current_team_name = rows[0]["team_name"]
        
        # Count how many recent games are with the current team
        for i, r in enumerate(rows):
            if r["team_id"] == current_team_id:
                new_team_games = i + 1
            else:
                # Found a team change
                was_traded = True
                old_team_abbrev = abbrev_from_team_name(r["team_name"]) or "UNK"
                # Trade date is approximately between this game and the last new-team game
                if i > 0:
                    trade_date = rows[i - 1]["game_date"]  # Last new-team game date as proxy
                break
        else:
            # No team change found — all games with same team
            new_team_games = len(rows)
        
        # Also check the trade_tracker table for more precise trade info
        try:
            from .trade_tracker import get_player_trade_info_by_id
            trade_info = get_player_trade_info_by_id(conn, player_id, as_of_date=before_date)
            if trade_info:
                was_traded = True
                trade_date = trade_info.trade_date
                old_team_abbrev = trade_info.from_team
                new_team_games = trade_info.games_with_new_team
                trade_confidence_discount = trade_info.confidence_discount
        except Exception:
            pass  # trade_tracker tables may not exist yet
    
    # For traded players, relax min_games: require at least 3 new-team games
    # OR 10 total games (including old team) for blending
    effective_min_games = min_games
    if was_traded and new_team_games >= 3:
        effective_min_games = 3  # Accept with just 3 new-team games
    elif was_traded and len(rows) >= min_games:
        effective_min_games = min(min_games, len(rows))  # Use whatever we have
    
    if len(rows) < effective_min_games:
        return None
    
    games = [dict(r) for r in rows]
    n = len(games)
    
    # Calculate average minutes
    all_minutes = [g["minutes"] for g in games if g["minutes"]]
    avg_minutes = statistics.mean(all_minutes) if all_minutes else 0
    
    # V19.1: For traded players, use new-team minutes as the baseline
    if was_traded and new_team_games >= 2:
        current_team_id = games[0].get("team_id")
        new_team_minutes = [
            g["minutes"] for g in games
            if g.get("team_id") == current_team_id and g["minutes"]
        ]
        if new_team_minutes:
            avg_minutes = statistics.mean(new_team_minutes)
    
    if avg_minutes < min_minutes:
        return None
    
    # Get team abbrev from most recent game
    team_name = games[0].get("team_name", "")
    team_abbrev = abbrev_from_team_name(team_name) or "UNK"
    position = games[0].get("pos", "") or "SF"
    
    # Initialize stats object
    stats = PlayerStatsV19(
        player_id=player_id,
        player_name=player["name"],
        team_abbrev=team_abbrev,
        position=position,
        games_played=n,
        avg_minutes=avg_minutes,
        # V19.1: Trade awareness fields
        was_traded=was_traded,
        new_team_games=new_team_games if was_traded else n,
        trade_date=trade_date,
        old_team_abbrev=old_team_abbrev,
        trade_confidence_discount=trade_confidence_discount,
    )
    
    # Store recent game details for deeper analysis
    stats.recent_game_details = games[:5]
    
    # Calculate averages for different windows
    for prop in ['pts', 'reb', 'ast']:
        values = [g[prop] for g in games if g.get(prop) is not None]
        
        if not values:
            continue
        
        # L3, L5, L10, L15, L20, Season
        stats.l3[prop] = statistics.mean(values[:3]) if len(values) >= 3 else statistics.mean(values)
        stats.l5[prop] = statistics.mean(values[:5]) if len(values) >= 5 else statistics.mean(values)
        stats.l10[prop] = statistics.mean(values[:10]) if len(values) >= 10 else statistics.mean(values)
        stats.l15[prop] = statistics.mean(values[:15]) if len(values) >= 15 else statistics.mean(values)
        stats.l20[prop] = statistics.mean(values[:20]) if len(values) >= 20 else statistics.mean(values)
        stats.season[prop] = statistics.mean(values)
        
        # Standard deviation (L10 window)
        if len(values) >= 5:
            stats.stds[prop] = statistics.stdev(values[:min(10, len(values))])
        
        # Recent games for pattern analysis
        stats.recent_games[prop] = values[:5]
        
        # Last game and second-to-last
        if len(values) >= 1:
            stats.last_game[prop] = values[0]
        if len(values) >= 2:
            stats.second_last_game[prop] = values[1]
        
        # Deviations
        l3 = stats.l3.get(prop, 0)
        l5 = stats.l5.get(prop, 0)
        l10 = stats.l10.get(prop, 0)
        l15 = stats.l15.get(prop, 0)
        season = stats.season.get(prop, 0)
        
        if l15 > 0:
            stats.deviations_l15[prop] = (l5 - l15) / l15 * 100
        if season > 0:
            stats.deviations_season[prop] = (l5 - season) / season * 100
        if l10 > 0:
            stats.deviations_l3_vs_l10[prop] = (l3 - l10) / l10 * 100
        
        # Consistency tracking (above/below season avg in L10)
        season_val = stats.season.get(prop, 0)
        if prop == 'pts' and season_val > 0:
            recent_10 = values[:10] if len(values) >= 10 else values
            stats.games_above_avg = sum(1 for v in recent_10 if v > season_val)
            stats.games_below_avg = sum(1 for v in recent_10 if v < season_val)
    
    # Minutes trends
    minutes_values = [g["minutes"] for g in games if g.get("minutes")]
    if len(minutes_values) >= 5:
        stats.l5_minutes = statistics.mean(minutes_values[:5])
    if len(minutes_values) >= 15:
        stats.l15_minutes = statistics.mean(minutes_values[:15])
    else:
        stats.l15_minutes = statistics.mean(minutes_values) if minutes_values else 0
    
    # Last game date for injury detection
    stats.last_game_date = games[0].get("game_date")
    if stats.last_game_date:
        try:
            last_dt = datetime.strptime(stats.last_game_date, "%Y-%m-%d")
            before_dt = datetime.strptime(before_date, "%Y-%m-%d")
            stats.days_since_last_game = (before_dt - last_dt).days
        except:
            pass
    
    # =========================================================================
    # V19 ENHANCED: Calculate Efficiency Stats
    # =========================================================================
    efficiency = EfficiencyStats()
    
    # Plus/Minus
    pm_values = [g["plus_minus"] for g in games if g.get("plus_minus") is not None]
    if pm_values:
        efficiency.l5_plus_minus_avg = statistics.mean(pm_values[:5]) if len(pm_values) >= 5 else statistics.mean(pm_values)
        efficiency.l10_plus_minus_avg = statistics.mean(pm_values[:10]) if len(pm_values) >= 10 else statistics.mean(pm_values)
        efficiency.season_plus_minus_avg = statistics.mean(pm_values)
    
    # FG%
    fg_pct_values = [g["fg_pct"] for g in games if g.get("fg_pct") is not None and g["fg_pct"] > 0]
    if fg_pct_values:
        efficiency.l5_fg_pct = statistics.mean(fg_pct_values[:5]) if len(fg_pct_values) >= 5 else statistics.mean(fg_pct_values)
        efficiency.l10_fg_pct = statistics.mean(fg_pct_values[:10]) if len(fg_pct_values) >= 10 else statistics.mean(fg_pct_values)
        efficiency.l15_fg_pct = statistics.mean(fg_pct_values[:15]) if len(fg_pct_values) >= 15 else statistics.mean(fg_pct_values)
        efficiency.season_fg_pct = statistics.mean(fg_pct_values)
    
    # FTA (Free Throw Attempts - aggression indicator)
    fta_values = [g["fta"] for g in games if g.get("fta") is not None]
    if fta_values:
        efficiency.l5_fta = statistics.mean(fta_values[:5]) if len(fta_values) >= 5 else statistics.mean(fta_values)
        efficiency.l10_fta = statistics.mean(fta_values[:10]) if len(fta_values) >= 10 else statistics.mean(fta_values)
        efficiency.season_fta = statistics.mean(fta_values)
    
    # True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
    ts_values = []
    for g in games:
        pts = g.get("pts", 0) or 0
        fga = g.get("fga", 0) or 0
        fta = g.get("fta", 0) or 0
        if fga > 0 or fta > 0:
            ts = pts / (2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0
            ts_values.append(ts)
    
    if ts_values:
        efficiency.l5_ts_pct = statistics.mean(ts_values[:5]) if len(ts_values) >= 5 else statistics.mean(ts_values)
        efficiency.l10_ts_pct = statistics.mean(ts_values[:10]) if len(ts_values) >= 10 else statistics.mean(ts_values)
        efficiency.season_ts_pct = statistics.mean(ts_values)
    
    # Usage proxy: (FGA + 0.44*FTA) per minute
    usage_values = []
    for g in games:
        fga = g.get("fga", 0) or 0
        fta = g.get("fta", 0) or 0
        minutes = g.get("minutes", 0) or 0
        if minutes > 0:
            usage = (fga + 0.44 * fta) / minutes
            usage_values.append(usage)
    
    if usage_values:
        efficiency.l5_usage_proxy = statistics.mean(usage_values[:5]) if len(usage_values) >= 5 else statistics.mean(usage_values)
        efficiency.l10_usage_proxy = statistics.mean(usage_values[:10]) if len(usage_values) >= 10 else statistics.mean(usage_values)
    
    stats.efficiency = efficiency
    
    # Historical matchup vs opponent (if provided)
    if opponent_abbrev:
        stats.vs_opponent = get_historical_matchup(
            conn, player_id, opponent_abbrev, before_date, stats.season
        )
    
    return stats


def get_games_for_date(
    conn: sqlite3.Connection,
    game_date: str,
) -> List[Dict[str, Any]]:
    """
    Get all games scheduled for a date.
    """
    rows = conn.execute(
        """
        SELECT 
            g.id as game_id,
            g.game_date,
            t1.id as team1_id, t1.name as team1_name,
            t2.id as team2_id, t2.name as team2_name
        FROM games g
        JOIN teams t1 ON t1.id = g.team1_id
        JOIN teams t2 ON t2.id = g.team2_id
        WHERE g.game_date = ?
        """,
        (game_date,),
    ).fetchall()
    
    return [dict(r) for r in rows]


def get_players_in_game(
    conn: sqlite3.Connection,
    team_abbrev: str,
    before_date: str,
    min_games: int = 5,
    min_avg_minutes: float = 15.0,
) -> List[int]:
    """
    Get player IDs for players likely to play in a game.
    """
    # Get team ID
    team_row = conn.execute(
        """
        SELECT t.id FROM teams t
        WHERE t.name LIKE ?
        LIMIT 1
        """,
        (f"%{team_abbrev}%",)
    ).fetchone()
    
    if not team_row:
        return []
    
    team_id = team_row["id"]
    
    # Get players with recent minutes for team
    rows = conn.execute(
        """
        SELECT bp.player_id, COUNT(*) as games, AVG(bp.minutes) as avg_min
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.team_id = ?
          AND g.game_date < ?
          AND bp.minutes IS NOT NULL
          AND bp.minutes > 5
        GROUP BY bp.player_id
        HAVING COUNT(*) >= ? AND AVG(bp.minutes) >= ?
        ORDER BY avg_min DESC
        """,
        (team_id, before_date, min_games, min_avg_minutes),
    ).fetchall()
    
    return [row["player_id"] for row in rows]


# ============================================================================
# Usage Redistribution
# ============================================================================

def calculate_usage_boost(
    injured_teammates: List[Dict[str, Any]],
    boost_per_major_player: float = 0.04,  # 4% per 15+ PPG player
    boost_per_minor_player: float = 0.02,  # 2% per 10+ PPG player
    max_boost: float = 0.12,  # Cap at 12%
) -> Tuple[float, List[str]]:
    """
    Calculate usage boost percentage when teammates are injured.
    
    V19 REFINED: More conservative boost estimates.
    
    Returns: (boost_decimal, reasons_list)
    """
    major_out = [p for p in injured_teammates if p.get("avg_pts", 0) >= 15.0]
    minor_out = [p for p in injured_teammates if 10.0 <= p.get("avg_pts", 0) < 15.0]
    
    if not major_out and not minor_out:
        return 0.0, []
    
    boost = len(major_out) * boost_per_major_player + len(minor_out) * boost_per_minor_player
    boost = min(boost, max_boost)
    
    reasons = []
    for p in major_out:
        reasons.append(f"MAJOR: {p['player_name']} OUT ({p['avg_pts']:.1f} PPG)")
    for p in minor_out[:2]:  # List up to 2 minor
        reasons.append(f"Minor: {p['player_name']} OUT ({p['avg_pts']:.1f} PPG)")
    
    return boost, reasons


# ============================================================================
# Edge Calculation
# ============================================================================

def calculate_edge(
    projection: float,
    line: float,
    direction: str,
) -> float:
    """
    Calculate edge percentage.
    
    OVER: (projection - line) / line * 100
    UNDER: (line - projection) / line * 100
    """
    if line <= 0:
        return 0.0
    
    if direction.upper() == "OVER":
        return (projection - line) / line * 100
    else:
        return (line - projection) / line * 100


# ============================================================================
# Pattern Detection Functions
# ============================================================================

def detect_cold_bounce_pattern(
    stats: PlayerStatsV19,
    prop_type: str,
    cold_threshold: float = -15.0,
    bounce_threshold: float = 5.0,
) -> Tuple[bool, List[str]]:
    """
    Detect Cold Bounce pattern for OVER picks.
    
    This is the ONLY reliable OVER pattern - 64-84% hit rate!
    
    Conditions:
    1. L5 is cold_threshold% or more BELOW L15 (player is cold)
    2. Last game was bounce_threshold% or more ABOVE L10 (showing recovery)
    
    V19: Added validation that bounce isn't due to weak opponent
    """
    pt = prop_type.lower()
    
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l10 = stats.l10.get(pt, 0)
    l15 = stats.l15.get(pt, 0)
    last_game = stats.last_game.get(pt, 0)
    
    # Check if player is cold (L5 significantly below L15)
    if deviation_l15 > cold_threshold:
        return False, []
    
    # Check bounce
    if l10 <= 0:
        return False, []
    
    bounce_pct = (last_game - l10) / l10 * 100
    if bounce_pct < bounce_threshold:
        return False, []
    
    reasons = [
        f"Cold bounce: L5 ({l5:.1f}) is {deviation_l15:.0f}% below L15 ({l15:.1f})",
        f"Recovery signal: Last game ({last_game:.0f}) bounced {bounce_pct:.0f}% above L10",
        f"Regression to baseline ({l15:.1f}) expected",
    ]
    
    return True, reasons


def detect_cold_streak_pattern(
    stats: PlayerStatsV19,
    prop_type: str,
    mild_threshold: float = -10.0,
    severe_threshold: float = -20.0,
) -> Tuple[str, List[str]]:
    """
    Detect Cold Streak pattern for UNDER picks.
    
    Returns: (severity, reasons)
    - severity: "none", "mild", "severe"
    
    V19 NOTE: Severe alone is RISKY (48% hit rate) - needs other factors!
    """
    pt = prop_type.lower()
    
    deviation_season = stats.deviations_season.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    season = stats.season.get(pt, 0)
    
    if deviation_season > mild_threshold:
        return "none", []
    
    if deviation_season <= severe_threshold:
        reasons = [
            f"Severe cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season ({season:.1f})",
            f"⚠️ Severe cold alone is risky - needs supporting factors",
        ]
        return "severe", reasons
    else:
        reasons = [
            f"Mild cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season ({season:.1f})",
        ]
        return "mild", reasons


def detect_hot_form(
    stats: PlayerStatsV19,
    prop_type: str,
    hot_threshold: float = 10.0,
) -> Tuple[bool, List[str]]:
    """
    Detect Hot Form pattern.
    
    ⚠️ WARNING: This pattern showed only 43% hit rate - NOT RELIABLE!
    Included for analysis only, weight is 0 in V19.
    """
    pt = prop_type.lower()
    
    deviation = stats.deviations_l3_vs_l10.get(pt, 0)
    l3 = stats.l3.get(pt, 0)
    l10 = stats.l10.get(pt, 0)
    
    if deviation >= hot_threshold:
        reasons = [
            f"⚠️ Hot form detected but UNRELIABLE: L3 ({l3:.1f}) is {deviation:.0f}% above L10 ({l10:.1f})",
            f"Hot form showed only 43% hit rate - WEIGHT=0",
        ]
        return True, reasons
    
    return False, []


# ============================================================================
# Holistic Factor Scoring (V19 Core Innovation)
# ============================================================================

def calculate_holistic_factor_score_under(
    stats: PlayerStatsV19,
    prop_type: str,
    defense: DefenseContextV19,
    b2b: BackToBackInfo,
    injury_impact: InjuryImpact,
    game_context: Optional[GameContext] = None,
) -> HolisticFactorScore:
    """
    Calculate holistic factor score for UNDER picks.
    
    V19 ENHANCED:
    - Requires multiple factors to align
    - More conservative with single-factor picks
    - Includes game context factors
    """
    score = HolisticFactorScore(direction="UNDER")
    pt = prop_type.lower()
    
    # =========================================================================
    # Defense Factors (PRIMARY DRIVERS for UNDER)
    # =========================================================================
    if defense.data_available:
        if defense.is_elite(pt):
            weight = UNDER_FACTOR_WEIGHTS["defense_elite"]
            score.factors["defense_elite"] = weight
            score.reasons.append(f"🛡️ ELITE defense: {defense.team_abbrev} ranks #{defense.get_rank(pt)} vs {pt.upper()}")
        elif defense.is_good(pt):
            weight = UNDER_FACTOR_WEIGHTS["defense_good"]
            score.factors["defense_good"] = weight
            score.reasons.append(f"🛡️ Good defense: {defense.team_abbrev} ranks #{defense.get_rank(pt)} vs {pt.upper()}")
    
    # =========================================================================
    # Fatigue Factors (VALIDATED STRONG)
    # =========================================================================
    if b2b.is_second_of_b2b:
        weight = UNDER_FACTOR_WEIGHTS["b2b_fatigue"]
        score.factors["b2b_fatigue"] = weight
        score.reasons.append("😴 B2B fatigue: Second game of back-to-back")
    elif b2b.is_third_in_four:
        weight = UNDER_FACTOR_WEIGHTS["third_in_four"]
        score.factors["third_in_four"] = weight
        score.reasons.append("😴 Fatigue: Third game in four days")
    elif b2b.is_fourth_in_six:
        weight = UNDER_FACTOR_WEIGHTS["third_in_four"] // 2  # Half weight
        score.factors["fourth_in_six"] = weight
        score.reasons.append("😴 Heavy schedule: Fourth game in six days")
    
    # =========================================================================
    # Cold Streak Factors (BE CAREFUL - severe alone is risky)
    # =========================================================================
    cold_severity, cold_reasons = detect_cold_streak_pattern(stats, pt)
    if cold_severity == "severe":
        weight = UNDER_FACTOR_WEIGHTS["cold_streak_severe"]
        score.factors["cold_streak_severe"] = weight
        score.reasons.extend(cold_reasons)
    elif cold_severity == "mild":
        weight = UNDER_FACTOR_WEIGHTS["cold_streak_mild"]
        score.factors["cold_streak_mild"] = weight
        score.reasons.extend(cold_reasons)
    
    # =========================================================================
    # Injury Rust Factor
    # =========================================================================
    if stats.days_since_last_game >= 7:
        weight = UNDER_FACTOR_WEIGHTS["injury_rust_first"]
        score.factors["injury_rust_first"] = weight
        score.reasons.append(f"🤕 Injury rust: {stats.days_since_last_game} days since last game")
    elif stats.days_since_last_game >= 5:
        weight = UNDER_FACTOR_WEIGHTS["injury_rust_second"]
        score.factors["injury_rust_second"] = weight
        score.reasons.append(f"🤕 Extended rest: {stats.days_since_last_game} days off")
    
    # =========================================================================
    # Minutes Decline Factor
    # =========================================================================
    minutes_trend = stats.get_minutes_trend()
    if minutes_trend < -10.0:  # L5 minutes 10%+ below L15
        weight = UNDER_FACTOR_WEIGHTS["minutes_decline"]
        score.factors["minutes_decline"] = weight
        score.reasons.append(f"📉 Minutes declining: {minutes_trend:.0f}% trend (role change?)")
    
    # =========================================================================
    # V19: Plus/Minus Factor (Box Score Analysis)
    # =========================================================================
    if stats.efficiency.has_negative_plus_minus(-5.0):
        weight = UNDER_FACTOR_WEIGHTS["negative_plus_minus"]
        score.factors["negative_plus_minus"] = weight
        score.reasons.append(f"📊 Negative +/-: L5 avg {stats.efficiency.l5_plus_minus_avg:.1f}")
    
    # =========================================================================
    # V19: Efficiency Trend Factors (Box Score Analysis)
    # =========================================================================
    if stats.efficiency.is_efficiency_declining(-5.0):
        weight = UNDER_FACTOR_WEIGHTS["poor_efficiency_trend"]
        score.factors["poor_efficiency_trend"] = weight
        fg_trend = stats.efficiency.get_fg_trend()
        score.reasons.append(f"📊 Efficiency declining: FG% {fg_trend:.1f}% below baseline")
    
    if stats.efficiency.is_ts_declining(-5.0):
        weight = UNDER_FACTOR_WEIGHTS["poor_ts_trend"]
        score.factors["poor_ts_trend"] = weight
        ts_trend = stats.efficiency.get_ts_trend()
        score.reasons.append(f"📊 TS% declining: {ts_trend:.1f}% below season")
    
    # =========================================================================
    # V19: Aggression Factor (FTA trend)
    # =========================================================================
    if stats.efficiency.is_less_aggressive(-15.0):
        weight = UNDER_FACTOR_WEIGHTS["low_fta_trend"]
        score.factors["low_fta_trend"] = weight
        fta_trend = stats.efficiency.get_fta_trend()
        score.reasons.append(f"📊 Less aggressive: FTA {fta_trend:.0f}% below season")
    
    # =========================================================================
    # Player Variance Factor
    # =========================================================================
    cv = stats.get_cv(pt)
    if cv > 0.40:
        weight = UNDER_FACTOR_WEIGHTS["high_variance"]
        score.factors["high_variance"] = weight
        score.reasons.append(f"🎲 High variance player: CV={cv:.2f} (unpredictable)")
    
    # =========================================================================
    # Historical Matchup Factor
    # =========================================================================
    if stats.vs_opponent and stats.vs_opponent.is_poor_matchup(pt, -10.0):
        weight = UNDER_FACTOR_WEIGHTS["poor_h2h_history"]
        score.factors["poor_h2h_history"] = weight
        pct = getattr(stats.vs_opponent, f"{pt}_vs_season_pct", 0)
        score.reasons.append(f"📜 Poor H2H: {pct:.0f}% vs season avg ({stats.vs_opponent.games_played} games)")
    
    # =========================================================================
    # V19: Game Context Factors
    # =========================================================================
    if game_context:
        if game_context.is_blowout_risk:
            weight = UNDER_FACTOR_WEIGHTS["blowout_risk"]
            score.factors["blowout_risk"] = weight
            score.reasons.append(f"🏀 Blowout risk: Spread is {abs(game_context.spread):.1f} points")
        
        if game_context.expected_pace == "slow":
            weight = UNDER_FACTOR_WEIGHTS["pace_factor_slow"]
            score.factors["pace_factor_slow"] = weight
            score.reasons.append(f"🐢 Slow pace game: O/U {game_context.over_under:.1f}")
    
    # =========================================================================
    # Calculate totals
    # =========================================================================
    score.total_score = sum(score.factors.values())
    score.factor_count = len(score.factors)
    
    # Find primary and secondary factors
    if score.factors:
        sorted_factors = sorted(score.factors.items(), key=lambda x: x[1], reverse=True)
        score.primary_factor = sorted_factors[0][0]
        score.primary_factor_weight = sorted_factors[0][1]
        if len(sorted_factors) >= 2:
            score.secondary_factor = sorted_factors[1][0]
            score.secondary_factor_weight = sorted_factors[1][1]
    
    # Calculate projection adjustment
    adj = 1.0
    for factor in score.factors.keys():
        if factor in FACTOR_PROJECTION_ADJUSTMENTS:
            adj *= FACTOR_PROJECTION_ADJUSTMENTS[factor]
    score.projection_adjustment = adj
    
    # Calculate alignment score (how well factors reinforce each other)
    if score.factor_count >= 2:
        score.alignment_score = min(100, score.factor_count * 20 + score.total_score / 2)
    else:
        score.alignment_score = score.total_score / 2  # Single factor = lower alignment
    
    return score


def calculate_holistic_factor_score_over(
    stats: PlayerStatsV19,
    prop_type: str,
    defense: DefenseContextV19,
    b2b: BackToBackInfo,
    injury_impact: InjuryImpact,
    game_context: Optional[GameContext] = None,
) -> HolisticFactorScore:
    """
    Calculate holistic factor score for OVER picks.
    
    V19 STRICTER:
    - Cold Bounce is the PRIMARY and practically ONLY reliable trigger
    - Weak defense weight is HEAVILY reduced (showed 43%)
    - Hot Form is ELIMINATED (showed 43%)
    - Requires higher edge thresholds
    """
    score = HolisticFactorScore(direction="OVER")
    pt = prop_type.lower()
    
    # =========================================================================
    # Cold Bounce Pattern (PRIMARY OVER TRIGGER - 64-84% hit rate)
    # =========================================================================
    is_cold_bounce, bounce_reasons = detect_cold_bounce_pattern(stats, pt)
    if is_cold_bounce:
        weight = OVER_FACTOR_WEIGHTS["cold_bounce"]
        score.factors["cold_bounce"] = weight
        score.reasons.extend(bounce_reasons)
    
    # =========================================================================
    # Defense Factors (CAUTION: weak defense showed only 43% for OVERs)
    # =========================================================================
    if defense.data_available:
        if defense.is_weak(pt):
            weight = OVER_FACTOR_WEIGHTS["defense_weak"]
            score.factors["defense_weak"] = weight
            score.reasons.append(f"⚠️ Weak defense ({defense.team_abbrev} #{defense.get_rank(pt)}) - LOW WEIGHT due to 43% hit rate")
        elif defense.is_poor(pt):
            weight = OVER_FACTOR_WEIGHTS["defense_poor"]
            score.factors["defense_poor"] = weight
            score.reasons.append(f"⚠️ Poor defense ({defense.team_abbrev} #{defense.get_rank(pt)}) - MINIMAL weight")
    
    # =========================================================================
    # Usage Boost Factor (when stars are OUT)
    # =========================================================================
    if injury_impact.usage_boost_pct >= 6.0:  # Significant boost
        weight = OVER_FACTOR_WEIGHTS["usage_boost_major"]
        score.factors["usage_boost_major"] = weight
        for inj in injury_impact.injured_teammates[:2]:  # List top 2
            if inj.get("avg_pts", 0) >= 15:
                score.reasons.append(f"📈 Usage boost: {inj['player_name']} OUT ({inj['avg_pts']:.1f} PPG)")
    elif injury_impact.usage_boost_pct >= 3.0:
        weight = OVER_FACTOR_WEIGHTS["usage_boost_minor"]
        score.factors["usage_boost_minor"] = weight
        score.reasons.append(f"📈 Minor usage boost: {injury_impact.usage_boost_pct:.1f}%")
    
    # =========================================================================
    # V19: Plus/Minus Factor (Box Score Analysis)
    # =========================================================================
    if stats.efficiency.has_positive_plus_minus(5.0):
        weight = OVER_FACTOR_WEIGHTS["positive_plus_minus"]
        score.factors["positive_plus_minus"] = weight
        score.reasons.append(f"📊 Positive +/-: L5 avg +{stats.efficiency.l5_plus_minus_avg:.1f}")
    
    # =========================================================================
    # V19: Efficiency Trend Factors (Box Score Analysis)
    # =========================================================================
    if stats.efficiency.is_efficiency_improving(5.0):
        weight = OVER_FACTOR_WEIGHTS["good_efficiency_trend"]
        score.factors["good_efficiency_trend"] = weight
        fg_trend = stats.efficiency.get_fg_trend()
        score.reasons.append(f"📊 Efficiency improving: FG% +{fg_trend:.1f}% above baseline")
    
    # =========================================================================
    # V19: Aggression Factor (FTA trend)
    # =========================================================================
    if stats.efficiency.is_more_aggressive(15.0):
        weight = OVER_FACTOR_WEIGHTS["high_fta_trend"]
        score.factors["high_fta_trend"] = weight
        fta_trend = stats.efficiency.get_fta_trend()
        score.reasons.append(f"📊 More aggressive: FTA +{fta_trend:.0f}% above season")
    
    # =========================================================================
    # Player Consistency Factor (Important for OVERs)
    # =========================================================================
    cv = stats.get_cv(pt)
    if cv < 0.20:
        weight = OVER_FACTOR_WEIGHTS["consistent_player"]
        score.factors["consistent_player"] = weight
        score.reasons.append(f"✅ Very consistent player: CV={cv:.2f}")
    
    # =========================================================================
    # Minutes Increase Factor
    # =========================================================================
    minutes_trend = stats.get_minutes_trend()
    if minutes_trend > 5.0:  # L5 minutes 5%+ above L15
        weight = OVER_FACTOR_WEIGHTS["minutes_increase"]
        score.factors["minutes_increase"] = weight
        score.reasons.append(f"📈 Minutes increasing: +{minutes_trend:.0f}% trend")
    
    # =========================================================================
    # Historical Matchup Factor
    # =========================================================================
    if stats.vs_opponent and stats.vs_opponent.is_good_matchup(pt, 10.0):
        weight = OVER_FACTOR_WEIGHTS["good_h2h_history"]
        score.factors["good_h2h_history"] = weight
        pct = getattr(stats.vs_opponent, f"{pt}_vs_season_pct", 0)
        score.reasons.append(f"📜 Good H2H: +{pct:.0f}% vs season avg ({stats.vs_opponent.games_played} games)")
    
    # =========================================================================
    # V19: Game Context Factors
    # =========================================================================
    if game_context:
        if game_context.expected_pace == "fast":
            weight = OVER_FACTOR_WEIGHTS["pace_factor_fast"]
            score.factors["pace_factor_fast"] = weight
            score.reasons.append(f"🏃 Fast pace game: O/U {game_context.over_under:.1f}")
    
    # =========================================================================
    # Hot Form - ELIMINATED (0 weight) but tracked for analysis
    # =========================================================================
    is_hot, hot_reasons = detect_hot_form(stats, pt)
    if is_hot:
        # Weight is 0, just log for reference
        score.reasons.extend(hot_reasons)
    
    # =========================================================================
    # Calculate totals
    # =========================================================================
    score.total_score = sum(score.factors.values())
    score.factor_count = len(score.factors)
    
    # Find primary and secondary factors
    if score.factors:
        sorted_factors = sorted(score.factors.items(), key=lambda x: x[1], reverse=True)
        score.primary_factor = sorted_factors[0][0]
        score.primary_factor_weight = sorted_factors[0][1]
        if len(sorted_factors) >= 2:
            score.secondary_factor = sorted_factors[1][0]
            score.secondary_factor_weight = sorted_factors[1][1]
    
    # Calculate projection adjustment
    adj = 1.0
    for factor in score.factors.keys():
        if factor in FACTOR_PROJECTION_ADJUSTMENTS:
            adj *= FACTOR_PROJECTION_ADJUSTMENTS[factor]
    score.projection_adjustment = adj
    
    # Calculate alignment score
    if score.factor_count >= 2:
        score.alignment_score = min(100, score.factor_count * 20 + score.total_score / 2)
    else:
        score.alignment_score = score.total_score / 2
    
    return score


# ============================================================================
# Result Grading
# ============================================================================

def grade_pick(
    actual_value: float,
    line: float,
    direction: str,
) -> Tuple[bool, float]:
    """
    Grade a pick against actual result.
    
    Returns: (hit, margin)
    """
    margin = actual_value - line
    
    if direction.upper() == "OVER":
        hit = actual_value > line
    else:
        hit = actual_value < line
    
    return hit, margin


def get_actual_stats(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
) -> Optional[Dict[str, float]]:
    """Get actual stats for a player on a specific date (for backtesting)."""
    row = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes, bp.plus_minus, bp.fg_pct
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
          AND g.game_date = ?
          AND bp.minutes IS NOT NULL
        LIMIT 1
        """,
        (player_id, game_date),
    ).fetchone()
    
    if not row:
        return None
    
    return {
        'pts': row["pts"] or 0,
        'reb': row["reb"] or 0,
        'ast': row["ast"] or 0,
        'min': row["minutes"] or 0,
        'plus_minus': row["plus_minus"] or 0,
        'fg_pct': row["fg_pct"] or 0,
    }


# ============================================================================
# Progress Bar Utility
# ============================================================================

def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    decimals: int = 1,
    length: int = 50,
    fill: str = '█',
    print_end: str = "\r"
):
    """
    Print a progress bar to terminal.
    
    Args:
        iteration: Current iteration (0 to total-1)
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places in percent
        length: Character length of bar
        fill: Bar fill character
        print_end: End character (\\r for same line)
    """
    if total <= 0:
        return
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    
    # Print newline on complete
    if iteration >= total:
        print()


def format_time_remaining(seconds: float) -> str:
    """Format seconds into human readable time."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"
