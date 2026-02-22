"""
Model V17 Shared Utilities
===========================

Common functions, data classes, and utilities shared between:
- Model V17 General (Holistic multi-factor approach for all picks)
- Model V17 Under (Specialized UNDER model - placeholder for Phase 2)

=============================================================================
MODEL V17 KEY INNOVATIONS (Addressing ALL Previous Model Shortcomings)
=============================================================================

1. **HOLISTIC MULTI-FACTOR ANALYSIS** (NOT Just Cold Bounce):
   - Previous models over-relied on single patterns (cold bounce)
   - V17 combines MULTIPLE factors into a weighted score
   - Factors: Defense, Form Trends, Usage Redistribution, B2B Fatigue,
     Historical vs Opponent, Player Consistency (CV), Minutes Trends
   - Minimum combined factor score required (not single pattern)
   - This addresses the concern: "do not just suggest based on cold bounces"

2. **HYBRID LINE APPROACH** (Sportsbook When Available, Projections Otherwise):
   - Use actual sportsbook lines when available (accurate edge)
   - STILL GENERATE PICKS when lines aren't available
   - Track line source for honest reporting
   - Apply stricter edge requirements for derived lines (10% vs 6%)

3. **STRATEGIC DIRECTION SELECTION** (From RCM v1.4 Analysis):
   - PTS: UNDER strongly preferred (63.9% vs 48.3% OVER)
   - REB: OVER preferred with cold bounce (~61%)
   - AST: EXCLUDED entirely (~54% is coin flip after juice)

4. **CONSIDER ALL FACTORS HOLISTICALLY**:
   - Player injuries/trades/game plan changes are considered
   - Minutes trends analyzed (not just cold = bad)
   - Multiple context signals combined
   - Historical H2H matchup data when available

5. **STRICT FILTERING** (Quality Over Quantity):
   - 23+ minute average for established players
   - 10+ games history required
   - Minimum factor score required
   - Exclude volatile archetypes (scoring guards have 51.5% hit rate)

6. **HONEST REPORTING**:
   - Track sportsbook vs derived line picks separately
   - Report hit rates by line source
   - No inflated metrics from derived line fallacy

=============================================================================
HOLISTIC FACTOR SCORING SYSTEM
=============================================================================

Instead of single-pattern triggers, V17 uses weighted factor scoring:

FOR UNDER PICKS:
| Factor                  | Weight | Description                        |
|------------------------|--------|------------------------------------|
| Elite Defense (Top 3)  | +30    | Team ranks 1-3 in DVP for position|
| Good Defense (Top 10)  | +15    | Team ranks 4-10 in DVP            |
| Severe Cold Streak     | +22    | L5 < 80% of season average        |
| Mild Cold Streak       | +12    | L5 < 90% of season average        |
| B2B Fatigue            | +18    | Second game of back-to-back       |
| Third in Four Days     | +10    | Third game in four days           |
| Injury Rust (1st Back) | +20    | First game back from injury       |
| High Variance (CV>0.4) | +8     | Inconsistent player               |
| Poor Matchup History   | +12    | Below average vs this opponent    |
| Minutes Decline        | +8     | L5 minutes < L15 by 10%+          |

FOR OVER PICKS:
| Factor                  | Weight | Description                        |
|------------------------|--------|------------------------------------|
| Cold Bounce Recovery   | +25    | L5 < L15 but last game above L10  |
| Weak Defense (Bot 5)   | +20    | Team ranks 26-30 in DVP           |
| Usage Boost (Star OUT) | +18    | Significant teammate injured      |
| Hot Form (L3 > L10)    | +10    | Recent uptick in performance      |
| Good Matchup History   | +15    | Above average vs this opponent    |
| Consistent (CV < 0.2)  | +10    | Very predictable player           |
| Minutes Increase       | +8     | L5 minutes > L15 by 5%+           |

THRESHOLDS:
- PREMIUM: Score >= 55 AND Edge >= 15%
- HIGH: Score >= 40 AND Edge >= 10%
- STANDARD: Score >= 30 AND Edge >= required minimum

=============================================================================

Author: PropAI Team - Model V17
Created: February 2026
Version: 17.0
"""
from __future__ import annotations

import sqlite3
import statistics
import unicodedata
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any, Set
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev


# ============================================================================
# Version Info
# ============================================================================

MODEL_VERSION = "17.0"
MODEL_NAME = "Model V17"


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
ELITE_DEFENSE_RANK = 3      # Top 3 = elite defense (STRICTER than V16's 5)
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
# Factor Weights for Holistic Scoring
# ============================================================================

# UNDER factor weights
# ============================================================================
# Factor Weights for Holistic Scoring (TUNED FROM BACKTEST 2025-10-22 to 2026-02-02)
# ============================================================================
# Backtest Results (used to tune weights):
#   defense_elite: 73.9% (51/69) - EXCELLENT, high weight
#   b2b_fatigue: 70.0% (7/10) - EXCELLENT, high weight
#   cold_bounce: 64.5% (40/62) - GOOD, moderate-high weight
#   cold_streak_mild: 61.7% (116/188) - GOOD, moderate weight
#   defense_good: 60.3% (41/68) - GOOD, moderate weight
#   injury_rust_first: 60.0% (12/20) - GOOD, moderate weight
#   cold_streak_severe: 48.0% (201/419) - BELOW 50%, reduce weight
#   hot_form: 43.3% (205/473) - NEGATIVE EDGE, ELIMINATED
#   defense_weak: 43.3% (26/60) - NEGATIVE EDGE for OVERS
# ============================================================================

UNDER_FACTOR_WEIGHTS = {
    # Defense factors - VALIDATED STRONG
    "defense_elite": 45,        # Top 3 DVP - 71.1% hit rate! Primary driver
    "defense_good": 25,         # Top 4-10 DVP - 64.9% hit rate. Increased
    
    # Form/trend factors - TUNED ROUND 2
    "cold_streak_severe": 8,    # L5 < 80% of season - 48.1% hit rate, further reduced!
    "cold_streak_mild": 20,     # L5 < 90% of season - 59.3% hit rate! Increased
    "minutes_decline": 12,      # L5 min < L15 min by 10%+
    
    # Fatigue factors - VALIDATED STRONG
    "b2b_fatigue": 35,          # Second of back-to-back - 69.6% hit rate! Primary driver
    "third_in_four": 15,        # Third game in 4 days
    
    # Special situations - VALIDATED
    "injury_rust_first": 25,    # First game back - 60% hit rate. Increased
    "injury_rust_second": 10,   # Second game back
    
    # Player characteristics
    "high_variance": 8,         # CV > 0.40
    
    # Historical matchup
    "poor_matchup_history": 12, # Below avg vs opponent (3+ games)
}

# OVER factor weights (CAUTIOUS - OVERs underperformed at 45.5%)
# Backtest shows OVERs are risky. Use high thresholds.
OVER_FACTOR_WEIGHTS = {
    # Pattern factors
    "cold_bounce": 35,          # L5 < L15 but last game > L10 - 64.5% hit rate! Increased
    "hot_form": 0,              # L3 > L10 - 43.3% REMOVED (negative edge)
    
    # Defense factors - CAUTION: defense_weak showed 43.3%, be selective
    "defense_weak": 10,         # Bottom 5 DVP - reduced from 20 (43.3% hit rate!)
    "defense_poor": 5,          # Bottom 10 DVP - reduced from 10
    
    # Usage factors - theoretical, keep moderate
    "usage_boost_major": 15,    # Star teammate OUT (15+ PPG)
    "usage_boost_minor": 5,     # Role player OUT (10+ PPG)
    
    # Player characteristics
    "consistent_player": 12,    # CV < 0.20 (more important for OVERs)
    "minutes_increase": 10,     # L5 min > L15 min by 5%+
    
    # Historical matchup
    "good_matchup_history": 15, # Above avg vs opponent (3+ games)
}

# Minimum factor scores for tiers (TUNED FROM BACKTEST)
# Backtest showed: 50-60 score: 55%, 60-70: 66.7%, 70-80: 75%
# Higher factor scores = better hit rates. Raising thresholds for quality.
MIN_FACTOR_SCORE_PREMIUM = 60   # Raised from 55 (was 66.7% hit rate at 60-70)
MIN_FACTOR_SCORE_HIGH = 45      # Raised from 40 (was 55% at 50-60)
MIN_FACTOR_SCORE_STANDARD = 35  # Raised from 30 (filter out weakest picks)

# Projection adjustments based on factors
FACTOR_PROJECTION_ADJUSTMENTS = {
    # UNDER adjustments (reduce projection)
    "defense_elite": 0.88,      # -12%
    "defense_good": 0.94,       # -6%
    "cold_streak_severe": 0.90, # -10%
    "cold_streak_mild": 0.95,   # -5%
    "b2b_fatigue": 0.94,        # -6%
    "injury_rust_first": 0.85,  # -15%
    "injury_rust_second": 0.93, # -7%
    
    # OVER adjustments (increase projection)
    "defense_weak": 1.10,       # +10%
    "defense_poor": 1.05,       # +5%
    "usage_boost_major": 1.08,  # +8%
    "usage_boost_minor": 1.04,  # +4%
    "cold_bounce": 1.02,        # +2% (regression expected)
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LineInfo:
    """
    Information about a betting line (sportsbook or derived).
    
    KEY FIELD: source tells us if this is a real betting line or derived from player avg.
    This is critical for honest model validation - addresses the "Derived Line Fallacy".
    """
    line: float
    source: str  # "sportsbook" or "derived"
    book: Optional[str] = None  # e.g., "draftkings", "fanduel"
    
    @property
    def is_sportsbook(self) -> bool:
        return self.source == "sportsbook"
    
    @property
    def is_derived(self) -> bool:
        return self.source == "derived"


@dataclass
class PlayerStatsV17:
    """
    Comprehensive player statistics for Model V17.
    
    Includes all averaging windows, deviations, variance metrics,
    and recent game values needed for holistic factor analysis.
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
    
    # Last game values
    last_game: Dict[str, float] = field(default_factory=dict)
    
    # Second-to-last game values (for trend analysis)
    second_last_game: Dict[str, float] = field(default_factory=dict)
    
    # Standard deviations (L10 window)
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Recent game values (last 5) for pattern analysis
    recent_games: Dict[str, List[float]] = field(default_factory=dict)
    
    # Historical vs specific opponent (if available)
    vs_opponent: Dict[str, float] = field(default_factory=dict)
    vs_opponent_games: int = 0
    
    # Minutes trends
    l5_minutes: float = 0.0
    l15_minutes: float = 0.0
    
    # Game dates for injury detection
    last_game_date: Optional[str] = None
    days_since_last_game: int = 1
    
    def get_projection(
        self, 
        prop_type: str, 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted projection for a prop type.
        
        Default weights (validated from V14/V15/V16 backtesting):
        - L3: 0.10 (very recent form)
        - L5: 0.20 (recent form)
        - L10: 0.30 (primary baseline)
        - L15: 0.20 (extended baseline)
        - Season: 0.20 (true talent level)
        """
        if weights is None:
            weights = {
                'l3': 0.10,
                'l5': 0.20,
                'l10': 0.30,
                'l15': 0.20,
                'season': 0.20,
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
            return values['season']
        
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
    
    def get_minutes_trend(self) -> float:
        """
        Get minutes trend as percentage change.
        
        Returns: (L5_min - L15_min) / L15_min * 100
        Positive = increasing minutes, Negative = decreasing
        """
        if self.l15_minutes <= 0:
            return 0.0
        return (self.l5_minutes - self.l15_minutes) / self.l15_minutes * 100


@dataclass
class DefenseContextV17:
    """
    Defense vs position context for an opponent team.
    
    This is from Hashtag Basketball DVP data and is CRITICAL for picks.
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
        return self.get_rank(prop_type) <= GOOD_DEFENSE_RANK
    
    def is_poor(self, prop_type: str) -> bool:
        """Check if defense is poor (21-25) for this prop type."""
        return self.get_rank(prop_type) >= 21 and self.get_rank(prop_type) < POOR_DEFENSE_RANK
    
    def is_weak(self, prop_type: str) -> bool:
        """Check if defense is weak (bottom 5) for this prop type."""
        return self.get_rank(prop_type) >= POOR_DEFENSE_RANK


@dataclass
class BackToBackInfo:
    """
    Information about team's rest/fatigue status.
    
    B2B fatigue showed 60.5%+ UNDER hit rate in previous testing.
    """
    is_b2b: bool = False
    is_second_of_b2b: bool = False
    is_third_in_four: bool = False
    days_rest: int = 1
    
    def has_fatigue_factor(self) -> bool:
        """Check if team has any fatigue factor."""
        return self.is_second_of_b2b or self.is_third_in_four


@dataclass
class InjuryImpact:
    """
    Impact of injured teammates on a player's projection.
    
    When stars are OUT, remaining players get usage boost.
    This is key for OVER picks per RCM/Idea.txt.
    """
    injured_teammates: List[Dict[str, Any]] = field(default_factory=list)
    total_pts_out: float = 0.0
    total_reb_out: float = 0.0
    total_ast_out: float = 0.0
    has_major_injury: bool = False  # Star player out (15+ PPG)
    has_minor_injury: bool = False  # Role player out (10+ PPG)
    usage_boost_pct: float = 0.0
    
    def has_significant_impact(self) -> bool:
        """Check if injuries create meaningful usage boost."""
        return self.has_major_injury or self.usage_boost_pct >= 5.0


@dataclass
class HolisticFactorScore:
    """
    Holistic factor scoring for Model V17.
    
    This replaces single-pattern detection with a multi-factor approach
    that considers ALL relevant signals and weights them appropriately.
    """
    # Factor flags (which factors are present)
    factors: Dict[str, bool] = field(default_factory=dict)
    
    # Weighted score
    total_score: float = 0.0
    
    # Direction (OVER or UNDER)
    direction: str = ""
    
    # Projection adjustment multiplier
    projection_adj_multiplier: float = 1.0
    
    # Factor explanations for reporting
    factor_reasons: List[str] = field(default_factory=list)
    
    # Primary factor (highest weighted one that's active)
    primary_factor: str = ""
    
    def get_tier(self, edge: float, min_edge: float) -> str:
        """Determine confidence tier based on score and edge."""
        if edge < min_edge:
            return "LOW"
        
        if self.total_score >= MIN_FACTOR_SCORE_PREMIUM and edge >= 15.0:
            return "PREMIUM"
        elif self.total_score >= MIN_FACTOR_SCORE_HIGH and edge >= 10.0:
            return "HIGH"
        elif self.total_score >= MIN_FACTOR_SCORE_STANDARD:
            return "STANDARD"
        else:
            return "LOW"
    
    def summary(self) -> str:
        """Get human-readable summary of factors."""
        active = [f for f, active in self.factors.items() if active]
        return f"{self.direction} (Score: {self.total_score:.0f}, Factors: {', '.join(active)})"


@dataclass
class HistoricalMatchup:
    """Historical performance vs a specific opponent."""
    opponent_abbrev: str
    games_played: int = 0
    avg_pts: float = 0.0
    avg_reb: float = 0.0
    avg_ast: float = 0.0
    pts_vs_season_pct: float = 0.0  # % difference vs season avg
    reb_vs_season_pct: float = 0.0
    ast_vs_season_pct: float = 0.0


# ============================================================================
# Name Normalization
# ============================================================================

def normalize_name(name: str) -> str:
    """
    Normalize player name for matching.
    
    Handles:
    - Accents/diacritics (Jokić -> Jokic)
    - Suffixes (Jr., Sr., III, etc.)
    - Case differences
    - Extra whitespace
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
    
    These players should be excluded from picks.
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
    
    Returns list of dicts with player info and their average stats.
    Used for usage redistribution calculations.
    """
    # Get team ID by various methods
    team_row = conn.execute(
        """
        SELECT id FROM teams 
        WHERE name LIKE ? 
           OR name LIKE ?
           OR name LIKE ?
        LIMIT 1
        """,
        (f"%{team_abbrev}%", f"{team_abbrev}%", f"% {team_abbrev}")
    ).fetchone()
    
    if not team_row:
        return []
    
    team_id = team_row["id"]
    
    # Get injured players for this team
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
        
        # Get player's average stats and check if they're on this team
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
    
    Priority:
    1. Match by player_id (most reliable)
    2. Fuzzy name match (handles Jokić vs Jokic etc.)
    """
    # Try by player_id first
    if player_id:
        row = conn.execute(
            """
            SELECT line, book
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
                book=row["book"] or "unknown"
            )
    
    # Try by fuzzy name match
    rows = conn.execute(
        """
        SELECT sl.line, sl.book, p.name
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
                book=row["book"] or "unknown"
            )
    
    return None


def get_derived_line(
    stats: PlayerStatsV17,
    prop_type: str,
    adjustment: float = 1.05
) -> LineInfo:
    """
    Calculate a derived line based on player's L10 average.
    
    We apply a 5% adjustment upward since sportsbook lines
    tend to be slightly higher than player averages.
    This addresses the "Derived Line Fallacy" identified in V10.
    
    Args:
        stats: Player stats object
        prop_type: pts, reb, or ast
        adjustment: Multiplier (1.05 = +5% adjustment)
    
    Returns:
        LineInfo with source="derived"
    """
    pt = prop_type.lower()
    l10_avg = stats.l10.get(pt, 0)
    
    # Apply adjustment (derived lines tend to underestimate actual sportsbook lines)
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
    stats: PlayerStatsV17,
    derived_adjustment: float = 1.05
) -> LineInfo:
    """
    Get line for a player - sportsbook if available, derived otherwise.
    
    This is the KEY FUNCTION for hybrid line handling.
    Always try sportsbook first for honest edge calculation.
    
    KEY V17: We ALWAYS return a line (sportsbook or derived)
    because lines come late, so we should still generate picks with projections.
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
) -> DefenseContextV17:
    """
    Get defense vs position context for an opponent team.
    
    This queries the team_defense_vs_position table (from Hashtag Basketball).
    """
    context = DefenseContextV17(
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
            elif rank < POOR_DEFENSE_RANK:
                rating = "poor"
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
    
    # Check last 4 days of games
    start_date = (gd - timedelta(days=4)).strftime("%Y-%m-%d")
    
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
    if len(game_dates) >= 2:
        info.is_third_in_four = True
    
    # Calculate rest days
    if game_dates:
        last_game = datetime.strptime(game_dates[0], "%Y-%m-%d")
        info.days_rest = (gd - last_game).days
    
    return info


def get_historical_matchup(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    before_date: str,
    season_stats: Dict[str, float],
    max_games: int = 5,
) -> HistoricalMatchup:
    """
    Get player's historical performance vs specific opponent.
    
    This helps identify players who consistently over/underperform vs certain teams.
    """
    matchup = HistoricalMatchup(opponent_abbrev=opponent_abbrev)
    
    # Get opponent team ID
    opp_row = conn.execute(
        """
        SELECT id FROM teams WHERE name LIKE ? LIMIT 1
        """,
        (f"%{opponent_abbrev}%",)
    ).fetchone()
    
    if not opp_row:
        return matchup
    
    opp_id = opp_row["id"]
    
    # Get games vs opponent
    rows = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
          AND (g.team1_id = ? OR g.team2_id = ?)
          AND g.game_date < ?
          AND bp.minutes > 10
        ORDER BY g.game_date DESC
        LIMIT ?
        """,
        (player_id, opp_id, opp_id, before_date, max_games)
    ).fetchall()
    
    if not rows:
        return matchup
    
    matchup.games_played = len(rows)
    matchup.avg_pts = sum(r["pts"] or 0 for r in rows) / len(rows)
    matchup.avg_reb = sum(r["reb"] or 0 for r in rows) / len(rows)
    matchup.avg_ast = sum(r["ast"] or 0 for r in rows) / len(rows)
    
    # Compare to season averages
    season_pts = season_stats.get('pts', 0)
    season_reb = season_stats.get('reb', 0)
    season_ast = season_stats.get('ast', 0)
    
    if season_pts > 0:
        matchup.pts_vs_season_pct = (matchup.avg_pts - season_pts) / season_pts * 100
    if season_reb > 0:
        matchup.reb_vs_season_pct = (matchup.avg_reb - season_reb) / season_reb * 100
    if season_ast > 0:
        matchup.ast_vs_season_pct = (matchup.avg_ast - season_ast) / season_ast * 100
    
    return matchup


def load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    min_games: int = 10,
    min_minutes: float = 20.0,
    max_games: int = 20,
    min_game_minutes: int = 5,
) -> Optional[PlayerStatsV17]:
    """
    Load comprehensive player statistics for analysis.
    
    Returns None if player doesn't meet requirements:
    - Less than min_games games
    - Average minutes below min_minutes
    - Filters out garbage time (<5 min games)
    """
    # Get player info
    player = conn.execute(
        "SELECT id, name FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    
    if not player:
        return None
    
    # Get game history
    rows = conn.execute(
        """
        SELECT 
            g.game_date, b.pts, b.reb, b.ast, b.minutes, b.pos,
            t.name as team_name
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
    
    if len(rows) < min_games:
        return None
    
    games = [dict(r) for r in rows]
    n = len(games)
    
    # Check minimum average minutes
    avg_min = sum(g["minutes"] or 0 for g in games) / n
    if avg_min < min_minutes:
        return None
    
    # Extract stats into lists
    stat_data = {
        'pts': [g["pts"] or 0 for g in games],
        'reb': [g["reb"] or 0 for g in games],
        'ast': [g["ast"] or 0 for g in games],
        'min': [g["minutes"] or 0 for g in games],
    }
    
    def avg(vals, limit=None):
        subset = vals[:limit] if limit else vals
        return sum(subset) / len(subset) if subset else 0.0
    
    def safe_std(vals, limit=10):
        subset = vals[:limit]
        return statistics.stdev(subset) if len(subset) >= 2 else 0.0
    
    # Build stats object
    stats = PlayerStatsV17(
        player_id=player_id,
        player_name=player["name"],
        team_abbrev=abbrev_from_team_name(games[0]["team_name"]) or "",
        position=map_position(games[0].get("pos") or ""),
        games_played=n,
        avg_minutes=avg_min,
    )
    
    for stat in ['pts', 'reb', 'ast', 'min']:
        vals = stat_data[stat]
        stats.l3[stat] = avg(vals, 3)
        stats.l5[stat] = avg(vals, 5)
        stats.l10[stat] = avg(vals, 10)
        stats.l15[stat] = avg(vals, 15) if n >= 15 else avg(vals)
        stats.l20[stat] = avg(vals, 20) if n >= 20 else avg(vals)
        stats.season[stat] = avg(vals)
        stats.stds[stat] = safe_std(vals)
        stats.last_game[stat] = vals[0] if vals else 0
        stats.second_last_game[stat] = vals[1] if len(vals) > 1 else 0
        stats.recent_games[stat] = vals[:5]
        
        # Calculate deviations (percentage difference)
        l5_val = stats.l5[stat]
        l15_val = stats.l15[stat]
        season_val = stats.season[stat]
        
        if l15_val > 0:
            stats.deviations_l15[stat] = (l5_val - l15_val) / l15_val * 100
        else:
            stats.deviations_l15[stat] = 0.0
        
        if season_val > 0:
            stats.deviations_season[stat] = (l5_val - season_val) / season_val * 100
        else:
            stats.deviations_season[stat] = 0.0
    
    # Minutes trends
    stats.l5_minutes = stats.l5.get('min', 0)
    stats.l15_minutes = stats.l15.get('min', 0)
    
    # Last game date and days since
    stats.last_game_date = games[0]["game_date"] if games else None
    if stats.last_game_date:
        try:
            last_dt = datetime.strptime(stats.last_game_date, "%Y-%m-%d")
            before_dt = datetime.strptime(before_date, "%Y-%m-%d")
            stats.days_since_last_game = (before_dt - last_dt).days
        except ValueError:
            pass
    
    return stats


def get_games_for_date(
    conn: sqlite3.Connection,
    game_date: str,
) -> List[Dict[str, Any]]:
    """Get all games scheduled for a date."""
    # Try scheduled_games first (for future dates)
    rows = conn.execute(
        """
        SELECT 
            sg.id, sg.game_date,
            t1.name as away_team, t1.id as away_team_id,
            t2.name as home_team, t2.id as home_team_id
        FROM scheduled_games sg
        JOIN teams t1 ON t1.id = sg.away_team_id
        JOIN teams t2 ON t2.id = sg.home_team_id
        WHERE sg.game_date = ?
        """,
        (game_date,),
    ).fetchall()
    
    if rows:
        return [dict(r) for r in rows]
    
    # Fallback to games table (for historical dates)
    rows = conn.execute(
        """
        SELECT 
            g.id, g.game_date,
            t1.name as away_team, t1.id as away_team_id,
            t2.name as home_team, t2.id as home_team_id
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
    
    Based on recent playing history for the team.
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
# Holistic Factor Scoring
# ============================================================================

def calculate_holistic_factor_score_under(
    stats: PlayerStatsV17,
    defense: DefenseContextV17,
    b2b: BackToBackInfo,
    historical: Optional[HistoricalMatchup],
    prop_type: str,
) -> HolisticFactorScore:
    """
    Calculate holistic factor score for UNDER picks.
    
    This combines multiple signals into a weighted score rather than
    relying on a single pattern trigger.
    """
    score = HolisticFactorScore(direction="UNDER")
    pt = prop_type.lower()
    
    total = 0.0
    adj_multiplier = 1.0
    
    # ===== DEFENSE FACTORS =====
    rank = defense.get_rank(pt)
    
    if defense.is_elite(pt):
        score.factors["defense_elite"] = True
        total += UNDER_FACTOR_WEIGHTS["defense_elite"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["defense_elite"]
        score.factor_reasons.append(f"Elite defense (#{rank} vs {pt.upper()})")
        if not score.primary_factor:
            score.primary_factor = "defense_elite"
    elif defense.is_good(pt):
        score.factors["defense_good"] = True
        total += UNDER_FACTOR_WEIGHTS["defense_good"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["defense_good"]
        score.factor_reasons.append(f"Good defense (#{rank} vs {pt.upper()})")
        if not score.primary_factor:
            score.primary_factor = "defense_good"
    
    # ===== FORM/TREND FACTORS =====
    dev_season = stats.get_deviation_season(pt)
    
    # Severe cold streak: L5 < 80% of season (deviation < -20%)
    if dev_season <= -20:
        score.factors["cold_streak_severe"] = True
        total += UNDER_FACTOR_WEIGHTS["cold_streak_severe"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["cold_streak_severe"]
        score.factor_reasons.append(f"Severe cold streak (L5 at {dev_season:.1f}% vs season)")
        if not score.primary_factor:
            score.primary_factor = "cold_streak_severe"
    # Mild cold streak: L5 < 90% of season (deviation < -10%)
    elif dev_season <= -10:
        score.factors["cold_streak_mild"] = True
        total += UNDER_FACTOR_WEIGHTS["cold_streak_mild"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["cold_streak_mild"]
        score.factor_reasons.append(f"Mild cold streak (L5 at {dev_season:.1f}% vs season)")
        if not score.primary_factor:
            score.primary_factor = "cold_streak_mild"
    
    # Minutes decline
    min_trend = stats.get_minutes_trend()
    if min_trend <= -10:
        score.factors["minutes_decline"] = True
        total += UNDER_FACTOR_WEIGHTS["minutes_decline"]
        score.factor_reasons.append(f"Minutes declining ({min_trend:.1f}%)")
    
    # ===== FATIGUE FACTORS =====
    if b2b.is_second_of_b2b:
        score.factors["b2b_fatigue"] = True
        total += UNDER_FACTOR_WEIGHTS["b2b_fatigue"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["b2b_fatigue"]
        score.factor_reasons.append("Second of back-to-back")
        if not score.primary_factor:
            score.primary_factor = "b2b_fatigue"
    elif b2b.is_third_in_four:
        score.factors["third_in_four"] = True
        total += UNDER_FACTOR_WEIGHTS["third_in_four"]
        score.factor_reasons.append("Third game in four days")
    
    # ===== INJURY RUST FACTORS =====
    # First game back from extended absence (7+ days)
    if stats.days_since_last_game >= 7:
        score.factors["injury_rust_first"] = True
        total += UNDER_FACTOR_WEIGHTS["injury_rust_first"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["injury_rust_first"]
        score.factor_reasons.append(f"First game back ({stats.days_since_last_game} days off)")
        if not score.primary_factor:
            score.primary_factor = "injury_rust_first"
    elif stats.days_since_last_game >= 5:
        score.factors["injury_rust_second"] = True
        total += UNDER_FACTOR_WEIGHTS["injury_rust_second"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["injury_rust_second"]
        score.factor_reasons.append(f"Returning from absence ({stats.days_since_last_game} days off)")
    
    # ===== PLAYER CHARACTERISTICS =====
    cv = stats.get_cv(pt)
    if cv > 0.40:
        score.factors["high_variance"] = True
        total += UNDER_FACTOR_WEIGHTS["high_variance"]
        score.factor_reasons.append(f"High variance player (CV={cv:.2f})")
    
    # ===== HISTORICAL MATCHUP =====
    if historical and historical.games_played >= 3:
        vs_season_pct = getattr(historical, f"{pt}_vs_season_pct", 0)
        if vs_season_pct <= -10:  # 10%+ below average vs this opponent
            score.factors["poor_matchup_history"] = True
            total += UNDER_FACTOR_WEIGHTS["poor_matchup_history"]
            score.factor_reasons.append(f"Poor history vs opponent ({vs_season_pct:.1f}% in {historical.games_played} games)")
    
    score.total_score = total
    score.projection_adj_multiplier = adj_multiplier
    
    return score


def calculate_holistic_factor_score_over(
    stats: PlayerStatsV17,
    defense: DefenseContextV17,
    b2b: BackToBackInfo,
    historical: Optional[HistoricalMatchup],
    injury_impact: InjuryImpact,
    prop_type: str,
) -> HolisticFactorScore:
    """
    Calculate holistic factor score for OVER picks.
    
    This combines multiple signals into a weighted score.
    """
    score = HolisticFactorScore(direction="OVER")
    pt = prop_type.lower()
    
    total = 0.0
    adj_multiplier = 1.0
    
    # ===== COLD BOUNCE PATTERN =====
    # L5 below L15 (cold) but last game showing recovery (bounce)
    dev_l15 = stats.get_deviation_l15(pt)
    l10_val = stats.l10.get(pt, 0)
    last_val = stats.last_game.get(pt, 0)
    
    # Cold: L5 is 15%+ below L15, Bounce: Last game > L10
    is_cold = dev_l15 <= -15
    is_bouncing = l10_val > 0 and last_val > l10_val * 1.05  # 5% above L10
    
    if is_cold and is_bouncing:
        score.factors["cold_bounce"] = True
        total += OVER_FACTOR_WEIGHTS["cold_bounce"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS.get("cold_bounce", 1.02)
        score.factor_reasons.append(f"Cold bounce: L5 at {dev_l15:.1f}% vs L15, last game {last_val:.1f} (above L10 {l10_val:.1f})")
        score.primary_factor = "cold_bounce"
    
    # ===== HOT FORM =====
    # L3 > L10 indicates recent uptick
    l3_val = stats.l3.get(pt, 0)
    if l10_val > 0 and l3_val > l10_val * 1.10:  # 10%+ above L10
        score.factors["hot_form"] = True
        total += OVER_FACTOR_WEIGHTS["hot_form"]
        score.factor_reasons.append(f"Hot form: L3 ({l3_val:.1f}) > L10 ({l10_val:.1f})")
        if not score.primary_factor:
            score.primary_factor = "hot_form"
    
    # ===== DEFENSE FACTORS =====
    rank = defense.get_rank(pt)
    
    if defense.is_weak(pt):
        score.factors["defense_weak"] = True
        total += OVER_FACTOR_WEIGHTS["defense_weak"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["defense_weak"]
        score.factor_reasons.append(f"Weak defense (#{rank} vs {pt.upper()})")
        if not score.primary_factor:
            score.primary_factor = "defense_weak"
    elif defense.is_poor(pt):
        score.factors["defense_poor"] = True
        total += OVER_FACTOR_WEIGHTS["defense_poor"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["defense_poor"]
        score.factor_reasons.append(f"Poor defense (#{rank} vs {pt.upper()})")
        if not score.primary_factor:
            score.primary_factor = "defense_poor"
    
    # ===== USAGE BOOST =====
    if injury_impact.has_major_injury:
        score.factors["usage_boost_major"] = True
        total += OVER_FACTOR_WEIGHTS["usage_boost_major"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["usage_boost_major"]
        score.factor_reasons.append(f"Major usage boost: Star teammate(s) OUT")
        if not score.primary_factor:
            score.primary_factor = "usage_boost_major"
    elif injury_impact.has_minor_injury:
        score.factors["usage_boost_minor"] = True
        total += OVER_FACTOR_WEIGHTS["usage_boost_minor"]
        adj_multiplier *= FACTOR_PROJECTION_ADJUSTMENTS["usage_boost_minor"]
        score.factor_reasons.append(f"Minor usage boost: Rotation player(s) OUT")
    
    # ===== PLAYER CHARACTERISTICS =====
    cv = stats.get_cv(pt)
    if cv < 0.20:
        score.factors["consistent_player"] = True
        total += OVER_FACTOR_WEIGHTS["consistent_player"]
        score.factor_reasons.append(f"Very consistent player (CV={cv:.2f})")
    
    # Minutes increase
    min_trend = stats.get_minutes_trend()
    if min_trend >= 5:
        score.factors["minutes_increase"] = True
        total += OVER_FACTOR_WEIGHTS["minutes_increase"]
        score.factor_reasons.append(f"Minutes increasing ({min_trend:.1f}%)")
    
    # ===== HISTORICAL MATCHUP =====
    if historical and historical.games_played >= 3:
        vs_season_pct = getattr(historical, f"{pt}_vs_season_pct", 0)
        if vs_season_pct >= 10:  # 10%+ above average vs this opponent
            score.factors["good_matchup_history"] = True
            total += OVER_FACTOR_WEIGHTS["good_matchup_history"]
            score.factor_reasons.append(f"Good history vs opponent (+{vs_season_pct:.1f}% in {historical.games_played} games)")
    
    # ===== PENALTIES FOR OVER PICKS =====
    # B2B is bad for OVER
    if b2b.is_second_of_b2b:
        total -= 10  # Penalty
        score.factor_reasons.append("PENALTY: B2B fatigue (-10)")
    
    # Elite defense is bad for OVER
    if defense.is_elite(pt):
        total -= 15  # Bigger penalty
        score.factor_reasons.append("PENALTY: Elite defense (-15)")
    
    score.total_score = max(total, 0)  # Floor at 0
    score.projection_adj_multiplier = adj_multiplier
    
    return score


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


def calculate_injury_impact(
    injured_teammates: List[Dict[str, Any]],
    major_threshold: float = 15.0,
    minor_threshold: float = 10.0,
) -> InjuryImpact:
    """
    Calculate impact of injured teammates.
    
    Args:
        injured_teammates: List of injured player dicts with avg_pts
        major_threshold: PPG for "star" player (15+)
        minor_threshold: PPG for "rotation" player (10+)
    """
    impact = InjuryImpact(injured_teammates=injured_teammates)
    
    for player in injured_teammates:
        avg_pts = player.get("avg_pts", 0)
        avg_reb = player.get("avg_reb", 0)
        avg_ast = player.get("avg_ast", 0)
        
        impact.total_pts_out += avg_pts
        impact.total_reb_out += avg_reb
        impact.total_ast_out += avg_ast
        
        if avg_pts >= major_threshold:
            impact.has_major_injury = True
        elif avg_pts >= minor_threshold:
            impact.has_minor_injury = True
    
    # Calculate usage boost (rough estimate)
    # Each star out adds ~5% boost, each rotation player ~2%
    boost = 0.0
    for player in injured_teammates:
        avg_pts = player.get("avg_pts", 0)
        if avg_pts >= major_threshold:
            boost += 0.05
        elif avg_pts >= minor_threshold:
            boost += 0.02
    
    impact.usage_boost_pct = min(boost * 100, 15.0)  # Cap at 15%
    
    return impact


# ============================================================================
# Result Grading
# ============================================================================

def grade_pick(
    actual_value: float,
    line: float,
    direction: str,
) -> Tuple[bool, float]:
    """
    Grade a pick based on actual result.
    
    Returns:
        (hit: bool, margin: float)
    """
    if direction.upper() == "OVER":
        hit = actual_value > line
        margin = actual_value - line
    else:
        hit = actual_value < line
        margin = line - actual_value
    
    return hit, margin


def get_actual_stats(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
) -> Optional[Dict[str, float]]:
    """
    Get actual stats for a player on a specific date.
    
    Used for backtesting to grade picks.
    """
    row = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
          AND g.game_date = ?
          AND bp.minutes > 0
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
    }
