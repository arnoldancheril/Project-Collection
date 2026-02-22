"""
Model V16.5 Under - Specialized UNDER NBA Props Prediction Model
=================================================================

This is the specialized UNDER model of a dual-model approach:
- Model V16 General: Handles OVER picks + can suggest UNDER when strongest
- Model V16 Under (this file): Specialized UNDER predictions for dedicated UNDER targeting

=============================================================================
MODEL V16.5 UNDER - KEY DESIGN PRINCIPLES
=============================================================================

1. **SPECIALIZED FOR UNDERS ONLY**
   - UNDER picks are more predictable than OVER picks
   - Negative factors compound more reliably than positive ones
   - Elite defenses consistently limit player production
   - Cold streaks persist longer than hot streaks (psychology)

2. **HYBRID LINE APPROACH** (Addressing Derived Line Fallacy):
   - Use actual sportsbook lines when available (accurate edge calculation)
   - STILL GENERATE PICKS without lines (use projections with +5% adjustment)
   - Track line source for honest reporting
   - Apply LOWER edge requirements for derived (UNDER has better base odds)

3. **FACTOR-BASED SCORING SYSTEM** (From Under Model V2):
   - Elite Defense (DVP 1-5): +30 points
   - Good Defense (DVP 6-10): +15 points
   - Severe Cold Streak (L5 < Season by 20%+): +22 points
   - Mild Cold Streak (L5 < Season by 10%+): +12 points
   - B2B Fatigue (second game): +8 points
   - Injury Rust (first game back): +18 points
   - High Variance Player: +6 points

4. **VALIDATED PATTERNS** (From Model V16.5 Backtesting - 2025-10-22 to 2026-02-03):
   | Pattern                | Hit Rate | Sample | Priority |
   |-----------------------|----------|--------|----------|
   | PREMIUM (score ≥50)   | 84.6%    | 13     | HIGHEST  |
   | Combined Elite+Cold   | 83.3%    | 12     | HIGHEST  |
   | Elite Defense (1-5)   | 68.9%    | 61     | HIGH     |
   | Injury Rust           | 66.7%    | 9      | HIGH     |
   | Good Defense (6-10)   | 65.9%    | 41     | HIGH     |
   | B2B Fatigue           | 63.6%    | 44     | MEDIUM   |
   | Cold Streak Mild      | 60.0%    | 35     | MEDIUM   |
   | Cold Streak Severe    | 51.7%    | 58     | LOW*     |
   
   *Cold Streak Severe alone is only ~52%, but when combined with Elite Defense
    it jumps to 83.3%! Always pair cold streak with defense.

5. **WHAT THIS MODEL EXCLUDES**:
   - REB UNDER (too volatile, ~52-54%)
   - AST UNDER for non-elite passers (coin flip)
   - UNDER vs weak defense (rank 25-30)
   - Players with insufficient games (<10)
   - Garbage time players (<23 min avg)

6. **KEY METRIC: Defense vs Position (DVP)**
   - Primary data source: Hashtag Basketball
   - Rank 1-5: Elite defense (STRONG UNDER signal)
   - Rank 6-10: Good defense (moderate signal)
   - Rank 11-20: Average defense (weak signal)
   - Rank 21-30: Weak defense (NO UNDER picks)

=============================================================================
USAGE:
------
    from src.nba_props.engine.model_v16_under import (
        get_daily_picks_v16_under,
        run_backtest_v16_under,
        ModelConfigV16Under,
    )
    
    # Get UNDER picks for today
    picks = get_daily_picks_v16_under("2026-02-03")
    print(picks.summary())
    
    # Run backtest
    result = run_backtest_v16_under("2025-10-22", "2026-02-03", verbose=True)
    print(result.summary())

Author: NBA Props Team - Model V16.5
Created: February 2026
Version: 16.5
"""
from __future__ import annotations

import sqlite3
import statistics
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths

from .model_v16_shared import (
    LineInfo,
    PlayerStatsV16,
    DefenseContextV16,
    BackToBackInfo,
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
    calculate_edge,
    detect_cold_streak_pattern,
    grade_pick,
    get_actual_stats,
    ELITE_DEFENSE_RANK,
    GOOD_DEFENSE_RANK,
    POOR_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
    MODEL_VERSION,
)


# ============================================================================
# Version Info
# ============================================================================

UNDER_MODEL_VERSION = "16.5"
UNDER_MODEL_NAME = "Model V16.5 Under"


# ============================================================================
# Factor Weights and Thresholds
# ============================================================================

# Factor weights for scoring (validated from Under Model V2 backtesting)
# Higher weight = stronger UNDER signal
FACTOR_WEIGHTS = {
    "defense_elite": 30,        # Top 5 defense at position (PRIMARY FACTOR)
    "defense_good": 15,         # Top 10 defense at position
    "defense_average": 5,       # Top 15 defense (minimal weight)
    "cold_streak_severe": 22,   # L5 < 80% of season (STRONG)
    "cold_streak_mild": 12,     # L5 < 90% of season
    "b2b_fatigue": 8,           # Second game of back-to-back
    "b2b_third_in_four": 5,     # Third game in 4 nights
    "injury_rust_first": 18,    # First game back from injury
    "injury_rust_second": 12,   # Second game back
    "injury_rust_third": 6,     # Third game back
    "high_variance": 6,         # CV > 0.35 (inconsistent player)
    "historical_struggle": 10,  # Poor history vs opponent
    "home_disadvantage": 3,     # Away player vs strong home defense
    "blowout_risk": 5,          # Large spread (garbage time risk)
}

# Factor adjustments (multipliers applied to projections)
# Values < 1.0 reduce the projection (good for UNDER)
FACTOR_ADJUSTMENTS = {
    "defense_elite": 0.88,      # 12% reduction
    "defense_good": 0.94,       # 6% reduction  
    "defense_average": 0.98,    # 2% reduction
    "cold_streak_severe": 0.86, # 14% reduction
    "cold_streak_mild": 0.93,   # 7% reduction
    "b2b_fatigue": 0.95,        # 5% reduction
    "b2b_third_in_four": 0.97,  # 3% reduction
    "injury_rust_first": 0.80,  # 20% reduction
    "injury_rust_second": 0.88, # 12% reduction
    "injury_rust_third": 0.94,  # 6% reduction
    "high_variance": 0.97,      # 3% reduction
    "historical_struggle": 0.93, # 7% reduction
    "home_disadvantage": 0.99,  # 1% reduction
    "blowout_risk": 0.96,       # 4% reduction
}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV16Under:
    """
    Model V16.5 Under Configuration.
    
    This model focuses EXCLUSIVELY on identifying high-confidence UNDER opportunities.
    It uses a factor-based scoring system to compound negative signals.
    
    KEY DIFFERENCES FROM V16 GENERAL:
    ---------------------------------
    1. UNDER picks ONLY (no OVER picks)
    2. Factor-based scoring instead of pattern detection
    3. Lower edge requirements (UNDER has better base odds)
    4. PTS-focused (REB UNDER is too volatile)
    5. Defense is PRIMARY factor (not just secondary)
    """
    # === VERSION INFO ===
    model_name: str = "Model V16.5 Under"
    model_version: str = UNDER_MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False  # NEVER require - lines come late!
    derived_line_adjustment: float = 1.05  # +5% for derived lines
    sportsbook_confidence_boost: float = 10.0  # Confidence boost for real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Established players only
    max_games_lookback: int = 20
    
    # === FACTOR THRESHOLDS ===
    # Cold streak thresholds (L5 vs Season deviation)
    cold_streak_mild_threshold: float = -10.0   # L5 is 10%+ below season
    cold_streak_severe_threshold: float = -20.0  # L5 is 20%+ below season
    
    # Variance threshold (coefficient of variation)
    high_variance_threshold: float = 0.35  # CV > 0.35 = inconsistent
    
    # Historical struggle threshold
    historical_struggle_threshold: float = -15.0  # vs_opp is 15%+ below season
    
    # Blowout risk (spread threshold)
    blowout_spread_threshold: float = 10.0  # If spread > 10, blowout risk
    
    # Injury rust windows (games since return)
    injury_rust_first_max: int = 1   # First game back
    injury_rust_second_max: int = 2  # Second game back
    injury_rust_third_max: int = 3   # Third game back
    
    # === EDGE REQUIREMENTS ===
    # UNDER-specific: Lower than General because UNDER has better base rate
    min_edge_sportsbook: float = 5.0   # 5%+ edge vs sportsbook line
    min_edge_derived: float = 8.0      # 8%+ edge vs derived line
    
    # === SCORE THRESHOLDS ===
    # Based on factor weights sum
    premium_score_threshold: float = 50.0  # Elite defense + cold streak severe
    high_score_threshold: float = 35.0     # Elite defense OR good + cold
    min_score_threshold: float = 25.0      # Minimum to generate pick
    
    # === CONFIDENCE MAPPING ===
    premium_confidence: float = 90.0
    high_confidence: float = 75.0
    standard_confidence: float = 65.0
    
    # === PROP SELECTION ===
    # PTS UNDER is PRIMARY (63.9% from RCM vs 48.3% OVER)
    include_pts_under: bool = True
    
    # REB UNDER is WEAK (~52-54%) - DISABLED by default
    include_reb_under: bool = False
    reb_under_require_elite_only: bool = True  # If enabled, elite defense only
    
    # AST UNDER is too volatile - DISABLED
    include_ast_under: bool = False
    
    # === DEFENSE REQUIREMENTS ===
    # Require at least average defense for any UNDER pick
    require_defense_data: bool = True
    max_defense_rank_for_under: int = 20  # Won't suggest UNDER vs weak defense
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 4   # Focused picks
    max_picks_per_day: int = 20   # Quality over quantity
    max_picks_per_player: int = 1 # One prop per player
    
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


@dataclass
class PropPickV16Under:
    """A pick generated by Model V16.5 Under."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    position: str
    
    # Pick details (always UNDER)
    prop_type: str  # PTS, REB
    direction: str = "UNDER"
    
    # Line information (tracks sportsbook vs derived)
    line: float = 0.0
    line_source: str = "derived"  # "sportsbook" or "derived"
    book: Optional[str] = None
    
    # Projection
    base_projection: float = 0.0  # Before factor adjustments
    adjusted_projection: float = 0.0  # After factor adjustments
    total_adjustment: float = 1.0  # Combined adjustment factor
    
    # Edge calculation
    edge_pct: float = 0.0
    
    # Factor scoring
    factor_score: float = 0.0
    factors: List[UnderFactor] = field(default_factory=list)
    factor_count: int = 0
    
    # Confidence
    confidence_score: float = 0.0
    confidence_tier: str = "HIGH"  # PREMIUM, HIGH, STANDARD
    
    # Defense context
    defense_rank: int = 15
    defense_rating: str = "average"
    
    # Additional context
    is_b2b: bool = False
    cold_streak_severity: str = "none"  # none, mild, severe
    games_since_injury: Optional[int] = None
    variance_cv: float = 0.0
    
    # Stats for display
    l5_avg: float = 0.0
    l10_avg: float = 0.0
    season_avg: float = 0.0
    
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
            "base_projection": round(self.base_projection, 1),
            "adj_projection": round(self.adjusted_projection, 1),
            "adjustment": f"{(1 - self.total_adjustment) * 100:.1f}% reduction",
            "edge": f"{self.edge_pct:.1f}%",
            "factor_score": round(self.factor_score, 1),
            "factor_count": self.factor_count,
            "factors": [f.name for f in self.factors],
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "b2b": self.is_b2b,
            "cold_streak": self.cold_streak_severity,
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "season": round(self.season_avg, 1),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }
    
    def summary_line(self) -> str:
        """One-line summary for display."""
        factors_str = ", ".join(f.name for f in self.factors[:3])
        return (
            f"📉 {self.player_name} ({self.team_abbrev} vs {self.opponent_abbrev}) - "
            f"{self.prop_type} UNDER {self.line:.1f} [{self.line_source}] | "
            f"Proj: {self.adjusted_projection:.1f} | Edge: {self.edge_pct:.1f}% | "
            f"Score: {self.factor_score:.0f} ({self.confidence_tier}) | "
            f"Factors: {factors_str}"
        )


@dataclass
class DailyPicksV16Under:
    """All UNDER picks for a day from Model V16.5."""
    date: str
    games: int
    config: ModelConfigV16Under = field(default_factory=ModelConfigV16Under)
    picks: List[PropPickV16Under] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    # Defense data status
    defense_data_available: bool = True
    defense_data_freshness: str = ""
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV16Under]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV16Under]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def premium_picks(self) -> List[PropPickV16Under]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV16Under]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V16.5 UNDER PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games} | Players analyzed: {self.players_analyzed}",
            f"Sportsbook lines available: {self.players_with_sportsbook_lines}",
            f"Using derived lines: {self.players_with_derived_lines}",
            f"Defense data: {'Available' if self.defense_data_available else 'NOT AVAILABLE'}",
            "",
            f"Total UNDER picks: {self.total_picks}",
            f"  PREMIUM: {len(self.premium_picks)} | HIGH: {len(self.high_picks)}",
            f"  Sportsbook: {len(self.sportsbook_picks)} | Derived: {len(self.derived_picks)}",
            "",
        ]
        
        # Group by tier
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    lines.append(p.summary_line())
                lines.append("")
        
        lines.append(f"{'='*70}")
        return "\n".join(lines)


@dataclass
class BacktestResultV16Under:
    """Comprehensive backtest results for Model V16.5 Under."""
    start_date: str
    end_date: str
    config: ModelConfigV16Under
    
    # Overall
    total_picks: int = 0
    hits: int = 0
    
    # By line source
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
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    
    # By primary factor
    elite_defense_picks: int = 0
    elite_defense_hits: int = 0
    good_defense_picks: int = 0
    good_defense_hits: int = 0
    cold_streak_severe_picks: int = 0
    cold_streak_severe_hits: int = 0
    cold_streak_mild_picks: int = 0
    cold_streak_mild_hits: int = 0
    b2b_fatigue_picks: int = 0
    b2b_fatigue_hits: int = 0
    injury_rust_picks: int = 0
    injury_rust_hits: int = 0
    combined_elite_cold_picks: int = 0
    combined_elite_cold_hits: int = 0
    
    # By factor score bucket
    score_50_plus_picks: int = 0
    score_50_plus_hits: int = 0
    score_35_50_picks: int = 0
    score_35_50_hits: int = 0
    score_25_35_picks: int = 0
    score_25_35_hits: int = 0
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # All picks for analysis
    all_picks: List[PropPickV16Under] = field(default_factory=list)
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
            f"MODEL V16.5 UNDER - BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"Avg picks/day: {self.total_picks / max(self.days_tested, 1):.1f}",
            "",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            "",
            f"BY LINE SOURCE (Honest Reporting):",
            f"  Sportsbook lines: {pct(self.sportsbook_hits, self.sportsbook_picks)} ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived lines:    {pct(self.derived_hits, self.derived_picks)} ({self.derived_hits}/{self.derived_picks})",
            "",
            f"BY CONFIDENCE TIER:",
            f"  PREMIUM (score ≥50): {pct(self.premium_hits, self.premium_picks)} ({self.premium_hits}/{self.premium_picks})",
            f"  HIGH (score 35-49):  {pct(self.high_hits, self.high_picks)} ({self.high_hits}/{self.high_picks})",
            f"  STANDARD (score 25-34): {pct(self.standard_hits, self.standard_picks)} ({self.standard_hits}/{self.standard_picks})",
            "",
            f"BY PROP TYPE:",
            f"  PTS UNDER: {pct(self.pts_hits, self.pts_picks)} ({self.pts_hits}/{self.pts_picks})",
            f"  REB UNDER: {pct(self.reb_hits, self.reb_picks)} ({self.reb_hits}/{self.reb_picks})",
            "",
            f"BY PRIMARY FACTOR:",
            f"  Elite Defense (rank 1-5):  {pct(self.elite_defense_hits, self.elite_defense_picks)} ({self.elite_defense_hits}/{self.elite_defense_picks})",
            f"  Good Defense (rank 6-10):  {pct(self.good_defense_hits, self.good_defense_picks)} ({self.good_defense_hits}/{self.good_defense_picks})",
            f"  Cold Streak Severe (-20%): {pct(self.cold_streak_severe_hits, self.cold_streak_severe_picks)} ({self.cold_streak_severe_hits}/{self.cold_streak_severe_picks})",
            f"  Cold Streak Mild (-10%):   {pct(self.cold_streak_mild_hits, self.cold_streak_mild_picks)} ({self.cold_streak_mild_hits}/{self.cold_streak_mild_picks})",
            f"  B2B Fatigue:               {pct(self.b2b_fatigue_hits, self.b2b_fatigue_picks)} ({self.b2b_fatigue_hits}/{self.b2b_fatigue_picks})",
            f"  Injury Rust:               {pct(self.injury_rust_hits, self.injury_rust_picks)} ({self.injury_rust_hits}/{self.injury_rust_picks})",
            f"  Combined Elite+Cold:       {pct(self.combined_elite_cold_hits, self.combined_elite_cold_picks)} ({self.combined_elite_cold_hits}/{self.combined_elite_cold_picks})",
            "",
            f"BY FACTOR SCORE BUCKET:",
            f"  Score ≥50 (Premium):  {pct(self.score_50_plus_hits, self.score_50_plus_picks)} ({self.score_50_plus_hits}/{self.score_50_plus_picks})",
            f"  Score 35-49 (High):   {pct(self.score_35_50_hits, self.score_35_50_picks)} ({self.score_35_50_hits}/{self.score_35_50_picks})",
            f"  Score 25-34 (Std):    {pct(self.score_25_35_hits, self.score_25_35_picks)} ({self.score_25_35_hits}/{self.score_25_35_picks})",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ============================================================================
# Core Model Functions
# ============================================================================

def _get_games_since_injury(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
    lookback_days: int = 30,
) -> Optional[int]:
    """
    Determine how many games the player has played since returning from injury.
    
    Returns None if no recent injury found.
    Returns 1, 2, 3, etc. for games since return.
    """
    # Look for injury reports in past 30 days
    start_date = (datetime.strptime(game_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    # Find most recent OUT status
    injury_row = conn.execute(
        """
        SELECT MAX(game_date) as last_out_date
        FROM injury_report
        WHERE (player_id = ? OR player_name IN (SELECT name FROM players WHERE id = ?))
          AND status IN ('OUT', 'DOUBTFUL')
          AND game_date >= ?
          AND game_date < ?
        """,
        (player_id, player_id, start_date, game_date),
    ).fetchone()
    
    if not injury_row or not injury_row["last_out_date"]:
        return None
    
    last_out_date = injury_row["last_out_date"]
    
    # Count games played since return
    games_since = conn.execute(
        """
        SELECT COUNT(*) as games
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
          AND g.game_date > ?
          AND g.game_date < ?
          AND bp.minutes IS NOT NULL
          AND bp.minutes > 5
        """,
        (player_id, last_out_date, game_date),
    ).fetchone()
    
    if games_since and games_since["games"] > 0:
        return games_since["games"]
    
    return None


def _calculate_factors(
    stats: PlayerStatsV16,
    prop_type: str,
    defense_context: DefenseContextV16,
    b2b_info: BackToBackInfo,
    config: ModelConfigV16Under,
    games_since_injury: Optional[int] = None,
    spread: Optional[float] = None,
) -> Tuple[List[UnderFactor], float, float]:
    """
    Calculate all applicable factors for an UNDER pick.
    
    Returns: (factors, total_score, total_adjustment)
    """
    factors = []
    pt = prop_type.lower()
    
    # === DEFENSE FACTORS (PRIMARY) ===
    defense_rank = defense_context.get_rank(pt)
    
    if defense_rank <= ELITE_DEFENSE_RANK:
        factors.append(UnderFactor(
            name="defense_elite",
            weight=FACTOR_WEIGHTS["defense_elite"],
            adjustment=FACTOR_ADJUSTMENTS["defense_elite"],
            reason=f"Elite defense at {stats.position} (#{defense_rank})"
        ))
    elif defense_rank <= GOOD_DEFENSE_RANK:
        factors.append(UnderFactor(
            name="defense_good",
            weight=FACTOR_WEIGHTS["defense_good"],
            adjustment=FACTOR_ADJUSTMENTS["defense_good"],
            reason=f"Good defense at {stats.position} (#{defense_rank})"
        ))
    elif defense_rank <= 15:
        factors.append(UnderFactor(
            name="defense_average",
            weight=FACTOR_WEIGHTS["defense_average"],
            adjustment=FACTOR_ADJUSTMENTS["defense_average"],
            reason=f"Average defense at {stats.position} (#{defense_rank})"
        ))
    
    # === COLD STREAK FACTORS ===
    cold_severity, cold_reasons = detect_cold_streak_pattern(
        stats, pt,
        mild_threshold=config.cold_streak_mild_threshold,
        severe_threshold=config.cold_streak_severe_threshold,
    )
    
    if cold_severity == "severe":
        factors.append(UnderFactor(
            name="cold_streak_severe",
            weight=FACTOR_WEIGHTS["cold_streak_severe"],
            adjustment=FACTOR_ADJUSTMENTS["cold_streak_severe"],
            reason=cold_reasons[0] if cold_reasons else "Severe cold streak"
        ))
    elif cold_severity == "mild":
        factors.append(UnderFactor(
            name="cold_streak_mild",
            weight=FACTOR_WEIGHTS["cold_streak_mild"],
            adjustment=FACTOR_ADJUSTMENTS["cold_streak_mild"],
            reason=cold_reasons[0] if cold_reasons else "Mild cold streak"
        ))
    
    # === FATIGUE FACTORS ===
    if b2b_info.is_second_of_b2b:
        factors.append(UnderFactor(
            name="b2b_fatigue",
            weight=FACTOR_WEIGHTS["b2b_fatigue"],
            adjustment=FACTOR_ADJUSTMENTS["b2b_fatigue"],
            reason="Second game of back-to-back"
        ))
    elif b2b_info.is_third_in_four:
        factors.append(UnderFactor(
            name="b2b_third_in_four",
            weight=FACTOR_WEIGHTS["b2b_third_in_four"],
            adjustment=FACTOR_ADJUSTMENTS["b2b_third_in_four"],
            reason="Third game in four nights"
        ))
    
    # === INJURY RUST FACTORS ===
    if games_since_injury is not None:
        if games_since_injury <= config.injury_rust_first_max:
            factors.append(UnderFactor(
                name="injury_rust_first",
                weight=FACTOR_WEIGHTS["injury_rust_first"],
                adjustment=FACTOR_ADJUSTMENTS["injury_rust_first"],
                reason=f"First game back from injury"
            ))
        elif games_since_injury <= config.injury_rust_second_max:
            factors.append(UnderFactor(
                name="injury_rust_second",
                weight=FACTOR_WEIGHTS["injury_rust_second"],
                adjustment=FACTOR_ADJUSTMENTS["injury_rust_second"],
                reason=f"Second game back from injury"
            ))
        elif games_since_injury <= config.injury_rust_third_max:
            factors.append(UnderFactor(
                name="injury_rust_third",
                weight=FACTOR_WEIGHTS["injury_rust_third"],
                adjustment=FACTOR_ADJUSTMENTS["injury_rust_third"],
                reason=f"Third game back from injury"
            ))
    
    # === VARIANCE FACTOR ===
    cv = stats.get_cv(pt)
    if cv > config.high_variance_threshold:
        factors.append(UnderFactor(
            name="high_variance",
            weight=FACTOR_WEIGHTS["high_variance"],
            adjustment=FACTOR_ADJUSTMENTS["high_variance"],
            reason=f"High variance player (CV: {cv:.2f})"
        ))
    
    # === HISTORICAL STRUGGLE FACTOR ===
    if stats.vs_opponent_games >= 3:
        vs_opp = stats.vs_opponent.get(pt, 0)
        season = stats.season.get(pt, 0)
        if season > 0 and vs_opp > 0:
            vs_opp_pct = (vs_opp - season) / season * 100
            if vs_opp_pct <= config.historical_struggle_threshold:
                factors.append(UnderFactor(
                    name="historical_struggle",
                    weight=FACTOR_WEIGHTS["historical_struggle"],
                    adjustment=FACTOR_ADJUSTMENTS["historical_struggle"],
                    reason=f"Struggles vs opponent ({vs_opp:.1f} vs {season:.1f} season)"
                ))
    
    # === BLOWOUT RISK FACTOR ===
    if spread is not None and abs(spread) > config.blowout_spread_threshold:
        factors.append(UnderFactor(
            name="blowout_risk",
            weight=FACTOR_WEIGHTS["blowout_risk"],
            adjustment=FACTOR_ADJUSTMENTS["blowout_risk"],
            reason=f"Blowout risk (spread: {spread:.1f})"
        ))
    
    # Calculate totals
    total_score = sum(f.weight for f in factors)
    total_adjustment = 1.0
    for f in factors:
        total_adjustment *= f.adjustment
    
    return factors, total_score, total_adjustment


def _map_score_to_confidence(
    score: float,
    line_source: str,
    config: ModelConfigV16Under,
) -> Tuple[float, str]:
    """
    Map factor score to confidence score and tier.
    
    Score >= 50: PREMIUM (elite defense + cold streak)
    Score 35-49: HIGH (elite defense OR good + cold)
    Score 25-34: STANDARD (baseline picks)
    """
    # Add bonus for sportsbook lines (more reliable)
    if line_source == "sportsbook":
        score += config.sportsbook_confidence_boost
    
    if score >= config.premium_score_threshold:
        # Premium tier
        confidence = config.premium_confidence + min(10, (score - 50) * 0.5)
        tier = "PREMIUM"
    elif score >= config.high_score_threshold:
        # High tier
        confidence = config.high_confidence + (score - 35) * 0.67
        tier = "HIGH"
    elif score >= config.min_score_threshold:
        # Standard tier
        confidence = config.standard_confidence + (score - 25) * 1.0
        tier = "STANDARD"
    else:
        # Below threshold - won't be used
        confidence = 50 + score
        tier = "LOW"
    
    return min(100, confidence), tier


def _generate_player_under_pick(
    conn: sqlite3.Connection,
    stats: PlayerStatsV16,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: ModelConfigV16Under,
    spread: Optional[float] = None,
) -> Optional[PropPickV16Under]:
    """
    Generate an UNDER pick for a specific player and prop type.
    
    Returns None if:
    - Player doesn't meet requirements
    - Defense is too weak
    - Factor score too low
    - Edge too small
    """
    pt = prop_type.lower()
    
    # Check minimum prop average
    if stats.l10.get(pt, 0) < MIN_PROP_AVERAGES.get(pt, 0):
        return None
    
    # Get defense context
    defense_context = get_defense_context(conn, opponent_abbrev, stats.position)
    
    # Require defense data if configured
    if config.require_defense_data and not defense_context.data_available:
        return None
    
    # Check defense isn't too weak for UNDER
    defense_rank = defense_context.get_rank(pt)
    if defense_rank > config.max_defense_rank_for_under:
        return None
    
    # Get B2B status
    b2b_info = get_back_to_back_status(conn, stats.team_abbrev, game_date)
    
    # Check for injury rust
    games_since_injury = _get_games_since_injury(conn, stats.player_id, game_date)
    
    # Calculate factors
    factors, factor_score, total_adjustment = _calculate_factors(
        stats, pt, defense_context, b2b_info, config,
        games_since_injury=games_since_injury,
        spread=spread,
    )
    
    # Check minimum factor score
    if factor_score < config.min_score_threshold:
        return None
    
    # Calculate base projection
    base_projection = stats.get_projection(pt)
    
    # Apply factor adjustments
    adjusted_projection = base_projection * total_adjustment
    
    # Get line (sportsbook preferred, derived fallback)
    line_info = get_line(
        conn, stats.player_id, stats.player_name, pt, game_date, stats,
        derived_adjustment=config.derived_line_adjustment
    )
    
    # Calculate edge (UNDER = line - projection)
    edge_pct = calculate_edge(adjusted_projection, line_info.line, "UNDER")
    
    # Check minimum edge
    min_edge = (
        config.min_edge_sportsbook if line_info.is_sportsbook 
        else config.min_edge_derived
    )
    if edge_pct < min_edge:
        return None
    
    # Map score to confidence
    confidence_score, confidence_tier = _map_score_to_confidence(
        factor_score, line_info.source, config
    )
    
    # Determine cold streak severity for tracking
    cold_severity = "none"
    for f in factors:
        if f.name == "cold_streak_severe":
            cold_severity = "severe"
            break
        elif f.name == "cold_streak_mild":
            cold_severity = "mild"
    
    # Build reasons list
    reasons = [f.reason for f in factors]
    reasons.append(f"Projection reduced by {(1 - total_adjustment) * 100:.1f}%")
    reasons.append(f"Edge vs line: {edge_pct:.1f}%")
    
    return PropPickV16Under(
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
        edge_pct=edge_pct,
        factor_score=factor_score,
        factors=factors,
        factor_count=len(factors),
        confidence_score=confidence_score,
        confidence_tier=confidence_tier,
        defense_rank=defense_rank,
        defense_rating=defense_context.get_rating(pt),
        is_b2b=b2b_info.is_second_of_b2b,
        cold_streak_severity=cold_severity,
        games_since_injury=games_since_injury,
        variance_cv=stats.get_cv(pt),
        l5_avg=stats.l5.get(pt, 0),
        l10_avg=stats.l10.get(pt, 0),
        season_avg=stats.season.get(pt, 0),
        reasons=reasons,
    )


def get_daily_picks_v16_under(
    game_date: str,
    config: Optional[ModelConfigV16Under] = None,
    db_path: Optional[Path] = None,
) -> DailyPicksV16Under:
    """
    Generate UNDER picks for all games on a given date.
    
    This is the main entry point for the Model V16.5 Under.
    
    Args:
        game_date: Date string in YYYY-MM-DD format
        config: Optional configuration override
        db_path: Optional database path override
    
    Returns:
        DailyPicksV16Under with all generated picks
    """
    if config is None:
        config = ModelConfigV16Under()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=db_path)
    
    result = DailyPicksV16Under(
        date=game_date,
        games=0,
        config=config,
    )
    
    with db.connect() as conn:
        # Get games for the date
        games = get_games_for_date(conn, game_date)
        result.games = len(games)
        
        if not games:
            return result
        
        # Get injured players to exclude
        injured_ids = get_injured_players(conn, game_date)
        
        # Check defense data availability
        defense_check = conn.execute(
            "SELECT COUNT(*) as cnt FROM team_defense_vs_position"
        ).fetchone()
        result.defense_data_available = defense_check["cnt"] > 0 if defense_check else False
        
        all_picks: List[PropPickV16Under] = []
        sportsbook_count = 0
        derived_count = 0
        
        for game in games:
            away_abbrev = abbrev_from_team_name(game["away_team"]) or ""
            home_abbrev = abbrev_from_team_name(game["home_team"]) or ""
            
            # Process both teams
            for team_abbrev, opponent_abbrev in [
                (away_abbrev, home_abbrev),
                (home_abbrev, away_abbrev),
            ]:
                if not team_abbrev or not opponent_abbrev:
                    continue
                
                # Get players for this team
                player_ids = get_players_in_game(
                    conn, team_abbrev, game_date,
                    min_games=5, min_avg_minutes=15.0
                )
                
                for player_id in player_ids:
                    if player_id in injured_ids:
                        continue
                    
                    result.players_analyzed += 1
                    
                    # Load player stats
                    stats = load_player_stats(
                        conn, player_id, game_date,
                        min_games=config.min_games_required,
                        min_minutes=config.min_avg_minutes,
                        max_games=config.max_games_lookback,
                        min_game_minutes=config.min_minutes_filter,
                    )
                    
                    if not stats:
                        continue
                    
                    # Generate picks for each prop type
                    for prop_type in ['pts', 'reb']:
                        # Check if this prop type is enabled
                        if prop_type == 'pts' and not config.include_pts_under:
                            continue
                        if prop_type == 'reb' and not config.include_reb_under:
                            continue
                        
                        pick = _generate_player_under_pick(
                            conn, stats, prop_type, opponent_abbrev,
                            game_date, config
                        )
                        
                        if pick:
                            all_picks.append(pick)
                            if pick.line_source == "sportsbook":
                                sportsbook_count += 1
                            else:
                                derived_count += 1
        
        # Sort by factor score (highest first)
        all_picks.sort(key=lambda p: (p.factor_score, p.edge_pct), reverse=True)
        
        # Apply limits
        game_picks: Dict[int, List[PropPickV16Under]] = {}
        player_picks: Dict[int, int] = {}
        final_picks: List[PropPickV16Under] = []
        
        for pick in all_picks:
            # Check per-player limit
            if player_picks.get(pick.player_id, 0) >= config.max_picks_per_player:
                continue
            
            # Check per-day limit
            if len(final_picks) >= config.max_picks_per_day:
                break
            
            final_picks.append(pick)
            player_picks[pick.player_id] = player_picks.get(pick.player_id, 0) + 1
        
        result.picks = final_picks
        result.players_with_sportsbook_lines = sportsbook_count
        result.players_with_derived_lines = derived_count
    
    return result


def run_backtest_v16_under(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV16Under] = None,
    db_path: Optional[Path] = None,
    verbose: bool = False,
) -> BacktestResultV16Under:
    """
    Run comprehensive backtest of Model V16.5 Under.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Optional configuration override
        db_path: Optional database path override
        verbose: Print progress updates
    
    Returns:
        BacktestResultV16Under with detailed metrics
    """
    if config is None:
        config = ModelConfigV16Under()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    result = BacktestResultV16Under(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    db = Db(path=db_path)
    
    # Generate all dates in range
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    with db.connect() as conn:
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            if verbose:
                print(f"Processing {date_str}...", end=" ")
            
            # Get picks for this day
            daily_picks = get_daily_picks_v16_under(date_str, config, db_path)
            
            if daily_picks.games > 0:
                result.days_tested += 1
                result.total_games += daily_picks.games
            
            # Grade each pick
            day_hits = 0
            day_picks = 0
            
            for pick in daily_picks.picks:
                # Get actual result
                actual = get_actual_stats(conn, pick.player_id, date_str)
                
                if not actual:
                    continue  # Skip if no actual result (player didn't play)
                
                pt = pick.prop_type.lower()
                actual_value = actual.get(pt, 0)
                
                # Grade the pick
                hit, margin = grade_pick(actual_value, pick.line, "UNDER")
                
                pick.actual_value = actual_value
                pick.hit = hit
                pick.margin = margin
                
                # Update counts
                result.total_picks += 1
                result.all_picks.append(pick)
                
                if hit:
                    result.hits += 1
                    day_hits += 1
                day_picks += 1
                
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
                
                # By prop type
                if pt == "pts":
                    result.pts_picks += 1
                    if hit:
                        result.pts_hits += 1
                elif pt == "reb":
                    result.reb_picks += 1
                    if hit:
                        result.reb_hits += 1
                
                # By factor score bucket
                if pick.factor_score >= 50:
                    result.score_50_plus_picks += 1
                    if hit:
                        result.score_50_plus_hits += 1
                elif pick.factor_score >= 35:
                    result.score_35_50_picks += 1
                    if hit:
                        result.score_35_50_hits += 1
                else:
                    result.score_25_35_picks += 1
                    if hit:
                        result.score_25_35_hits += 1
                
                # By primary factor
                factor_names = [f.name for f in pick.factors]
                
                has_elite_defense = "defense_elite" in factor_names
                has_good_defense = "defense_good" in factor_names
                has_cold_severe = "cold_streak_severe" in factor_names
                has_cold_mild = "cold_streak_mild" in factor_names
                has_b2b = "b2b_fatigue" in factor_names
                has_injury = any(f.startswith("injury_rust") for f in factor_names)
                
                if has_elite_defense:
                    result.elite_defense_picks += 1
                    if hit:
                        result.elite_defense_hits += 1
                
                if has_good_defense:
                    result.good_defense_picks += 1
                    if hit:
                        result.good_defense_hits += 1
                
                if has_cold_severe:
                    result.cold_streak_severe_picks += 1
                    if hit:
                        result.cold_streak_severe_hits += 1
                
                if has_cold_mild:
                    result.cold_streak_mild_picks += 1
                    if hit:
                        result.cold_streak_mild_hits += 1
                
                if has_b2b:
                    result.b2b_fatigue_picks += 1
                    if hit:
                        result.b2b_fatigue_hits += 1
                
                if has_injury:
                    result.injury_rust_picks += 1
                    if hit:
                        result.injury_rust_hits += 1
                
                # Combined elite + cold
                if has_elite_defense and (has_cold_severe or has_cold_mild):
                    result.combined_elite_cold_picks += 1
                    if hit:
                        result.combined_elite_cold_hits += 1
            
            if verbose and day_picks > 0:
                rate = day_hits / day_picks * 100
                print(f"{day_hits}/{day_picks} ({rate:.1f}%)")
            elif verbose:
                print("No picks")
            
            result.daily_results.append({
                "date": date_str,
                "picks": day_picks,
                "hits": day_hits,
            })
            
            current += timedelta(days=1)
    
    if verbose:
        print("\n" + result.summary())
    
    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V16.5 Under - UNDER picks")
    parser.add_argument("--date", type=str, help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--start", type=str, default="2025-10-22", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2026-02-03", help="Backtest end date")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.backtest:
        print(f"Running Model V16.5 Under backtest from {args.start} to {args.end}...")
        result = run_backtest_v16_under(args.start, args.end, verbose=args.verbose)
        print(result.summary())
    elif args.date:
        print(f"Generating Model V16.5 Under picks for {args.date}...")
        picks = get_daily_picks_v16_under(args.date)
        print(picks.summary())
    else:
        # Default to today
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"Generating Model V16.5 Under picks for {today}...")
        picks = get_daily_picks_v16_under(today)
        print(picks.summary())
