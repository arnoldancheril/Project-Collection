"""
Model V14 Under - Specialized UNDER NBA Props Prediction Model
===============================================================

This is the UNDER-specialized model of a dual-model approach:
- Model V14 General: Focuses on OVER picks and REB in both directions
- Model V14 Under: Focuses EXCLUSIVELY on UNDER opportunities

WHY A SEPARATE UNDER MODEL?
---------------------------
Analysis from extensive backtesting revealed:
1. PTS UNDER hits at 63.9% vs PTS OVER at 48.3% (HUGE difference!)
2. Negative factors compound more reliably than positive ones
3. Elite defense consistently limits production across positions
4. Cold streaks persist longer than hot streaks due to psychology

KEY INSIGHTS FROM PREVIOUS MODELS:
----------------------------------
1. DEFENSE VS POSITION (From Under Model V2):
   - Elite defense (top 5) = ~62% UNDER hit rate
   - Elite defense + Cold streak = ~66% UNDER hit rate
   
2. COLD STREAK PERSISTENCE:
   - Players don't immediately snap out of slumps
   - L5 < Season avg = continued underperformance expected
   
3. COMPOUNDING FACTORS:
   - Multiple negative factors = premium picks
   - Elite defense + Cold streak + B2B = highest confidence

4. PROP-SPECIFIC INSIGHTS:
   - PTS UNDER: 63.9% (primary focus)
   - REB UNDER: 59% (good secondary option)
   - AST UNDER: Volatile, only for specific players

MODEL V14 UNDER RULES:
----------------------
1. ONLY generate UNDER picks (leave OVER to General model)
2. Primary signal: Elite defense at position
3. Secondary signals: Cold streak, B2B fatigue, injury rust
4. Premium picks require: Elite defense + Cold streak
5. Track line source for honest reporting
6. Focus on PTS and REB (AST only for high-avg players)

FACTOR SCORING SYSTEM:
----------------------
Each factor contributes weight to a total score:
- Elite Defense (30): Top 5 DVP rank
- Good Defense (15): Top 10 DVP rank
- Cold Streak Severe (22): L5 < 80% of season
- Cold Streak Mild (12): L5 < 90% of season
- B2B Second Game (8): Back-to-back fatigue
- Injury First Back (18): Rust after absence
- High Variance (6): Inconsistent performer

CONFIDENCE TIERS:
-----------------
- PREMIUM: Elite defense + Cold streak severe (score >= 50)
- HIGH: Elite defense OR (Good defense + Cold streak) (score >= 35)
- STANDARD: Good defense + other factors (score >= 25)

USAGE:
------
    from src.nba_props.engine.model_v14_under import (
        get_daily_picks_v14_under,
        run_backtest_v14_under,
        ModelConfigV14Under,
    )
    
    # Get picks for today
    picks = get_daily_picks_v14_under("2026-02-03")
    
    # Run backtest
    result = run_backtest_v14_under("2025-12-01", "2026-02-02")

Author: NBA Props Team - Model V14
Created: February 2026
Version: 14.0
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths

from .model_v14_shared import (
    LineInfo,
    PlayerStatsV14,
    DefenseContextV14,
    BackToBackInfo,
    normalize_name,
    map_position,
    get_injured_players,
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
    AVERAGE_DEFENSE_RANK,
    MIN_PROP_AVERAGES,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV14Under:
    """
    Model V14 Under Configuration.
    
    Specialized for UNDER picks only with defense-first approach.
    
    VALIDATED INSIGHTS:
    - Elite defense (top 5) = ~62% UNDER hit rate
    - Elite defense + Cold streak severe = ~66% UNDER hit rate
    - PTS UNDER >> PTS OVER (63.9% vs 48.3%)
    """
    # === VERSION INFO ===
    model_name: str = "Model V14 Under"
    model_version: str = "14.0"
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False
    derived_line_adjustment: float = 1.05
    sportsbook_confidence_boost: float = 10.0
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5
    min_avg_minutes: float = 20.0  # Slightly lower than General for more coverage
    max_games_lookback: int = 20
    
    # === DEFENSE THRESHOLDS ===
    elite_defense_rank: int = 5
    good_defense_rank: int = 10
    average_defense_rank: int = 15
    
    # === COLD STREAK THRESHOLDS ===
    cold_streak_severe_pct: float = -20.0  # L5 is 20%+ below season
    cold_streak_mild_pct: float = -10.0    # L5 is 10%+ below season
    
    # === FACTOR WEIGHTS (for scoring) ===
    # Based on extensive backtest analysis
    weights: Dict[str, int] = field(default_factory=lambda: {
        # Primary factors
        "defense_elite": 30,
        "defense_good": 15,
        "defense_average": 5,
        # Cold streak factors
        "cold_streak_severe": 22,
        "cold_streak_mild": 12,
        # Fatigue factors
        "b2b_second": 8,
        "b2b_third_in_four": 5,
        # Injury factors
        "injury_first_back": 18,
        "injury_second_back": 12,
        "injury_third_back": 6,
        # Other factors
        "high_variance": 6,
        "historical_struggle": 10,
        "away_game": 3,
    })
    
    # === FACTOR ADJUSTMENTS (multipliers) ===
    adjustments: Dict[str, float] = field(default_factory=lambda: {
        "defense_elite": 0.86,      # -14% vs elite defense
        "defense_good": 0.93,       # -7% vs good defense
        "defense_average": 0.98,    # -2% vs average defense
        "cold_streak_severe": 0.85, # -15% when cold
        "cold_streak_mild": 0.92,   # -8% when mildly cold
        "b2b_second": 0.96,         # -4% on B2B
        "b2b_third_in_four": 0.98,  # -2%
        "injury_first_back": 0.78,  # -22% first game back
        "injury_second_back": 0.88, # -12%
        "injury_third_back": 0.94,  # -6%
        "high_variance": 0.97,      # -3%
        "historical_struggle": 0.93, # -7%
        "away_game": 0.99,          # -1%
    })
    
    # === EDGE REQUIREMENTS ===
    min_edge_sportsbook: float = 5.0   # Lower bar with real lines
    min_edge_derived: float = 10.0     # Higher bar with derived lines (was 8.0)
    
    # === CONFIDENCE THRESHOLDS ===
    # Based on factor score
    premium_score_threshold: int = 50   # Elite defense + Cold streak severe
    high_score_threshold: int = 35      # Elite defense OR Good + Cold
    standard_score_threshold: int = 35  # Raised from 25 for better quality
    
    premium_confidence: float = 88.0
    high_confidence: float = 75.0
    standard_confidence: float = 65.0
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb', 'ast'])
    
    # Minimum averages to consider (avoid garbage time players)
    min_avgs: Dict[str, float] = field(default_factory=lambda: {
        'pts': 8.0,
        'reb': 4.0,
        'ast': 5.5,  # Higher bar for AST
    })
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 6
    max_picks_per_day: int = 25
    max_picks_per_player: int = 2
    
    # === QUALITY REQUIREMENTS ===
    require_defense_factor: bool = True  # Must have at least good defense
    min_factors_for_standard: int = 2    # Need 2+ factors for non-elite defense
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class UnderFactors:
    """Factors detected for an UNDER pick."""
    factors: Dict[str, bool] = field(default_factory=dict)
    total_score: int = 0
    adjustment_factor: float = 1.0
    reasons: List[str] = field(default_factory=list)
    
    def add_factor(
        self, 
        name: str, 
        weight: int, 
        adjustment: float,
        reason: str,
    ):
        """Add a detected factor."""
        self.factors[name] = True
        self.total_score += weight
        self.adjustment_factor *= adjustment
        self.reasons.append(reason)
    
    @property
    def factor_count(self) -> int:
        return len(self.factors)
    
    def has_defense_factor(self) -> bool:
        return any(k.startswith("defense_") for k in self.factors)
    
    def has_elite_defense(self) -> bool:
        return self.factors.get("defense_elite", False)
    
    def has_cold_streak(self) -> bool:
        return (self.factors.get("cold_streak_severe", False) or 
                self.factors.get("cold_streak_mild", False))


@dataclass
class PropPickV14Under:
    """A pick generated by Model V14 Under."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details (always UNDER)
    prop_type: str  # PTS, REB, AST
    
    # Line information
    line: float
    line_source: str  # "sportsbook" or "derived"
    
    # Projection
    base_projection: float
    adjusted_projection: float
    adjustment_factor: float
    
    # Edge
    edge_pct: float
    
    # Confidence
    confidence_score: float
    confidence_tier: str  # PREMIUM, HIGH, STANDARD
    
    # Defense context
    defense_rank: int
    defense_rating: str
    
    # Supporting data
    l5_avg: float
    l10_avg: float
    season_avg: float
    
    # Factor score (required)
    factor_score: int
    factor_count: int
    
    # === Fields with defaults below ===
    direction: str = "UNDER"
    book: Optional[str] = None
    
    # Factors
    factors: Dict[str, bool] = field(default_factory=dict)
    
    # Context
    is_b2b: bool = False
    is_cold: bool = False
    
    # Reasons
    reasons: List[str] = field(default_factory=list)
    
    # Results
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
            "projection": round(self.adjusted_projection, 1),
            "edge": f"{self.edge_pct:.1f}%",
            "factors": list(self.factors.keys()),
            "factor_score": self.factor_score,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "season": round(self.season_avg, 1),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV14Under:
    """All picks for a day from Model V14 Under."""
    date: str
    games: int
    config: ModelConfigV14Under = field(default_factory=ModelConfigV14Under)
    picks: List[PropPickV14Under] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV14Under]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV14Under]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def pts_picks(self) -> List[PropPickV14Under]:
        return [p for p in self.picks if p.prop_type.upper() == "PTS"]
    
    @property
    def reb_picks(self) -> List[PropPickV14Under]:
        return [p for p in self.picks if p.prop_type.upper() == "REB"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V14 UNDER PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games} | Players analyzed: {self.players_analyzed}",
            f"Sportsbook lines: {self.players_with_sportsbook_lines}",
            f"Derived lines: {self.players_with_derived_lines}",
            "",
            f"Total UNDER picks: {self.total_picks}",
            f"  PTS: {len(self.pts_picks)} | REB: {len(self.reb_picks)}",
            f"  Sportsbook: {len(self.sportsbook_picks)} | Derived: {len(self.derived_picks)}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    src = f"[{p.book}]" if p.line_source == "sportsbook" else "[derived]"
                    factors_str = ", ".join(list(p.factors.keys())[:3])
                    lines.append(
                        f"  📉 {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.prop_type} UNDER {p.line:.1f} {src}"
                    )
                    lines.append(
                        f"      Proj: {p.adjusted_projection:.1f} | Edge: {p.edge_pct:.1f}% | "
                        f"Score: {p.factor_score} | Factors: {factors_str}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV14Under:
    """Backtest results for Model V14 Under."""
    start_date: str
    end_date: str
    config: ModelConfigV14Under
    
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
    ast_picks: int = 0
    ast_hits: int = 0
    
    # By key factors
    elite_defense_picks: int = 0
    elite_defense_hits: int = 0
    cold_streak_picks: int = 0
    cold_streak_hits: int = 0
    elite_and_cold_picks: int = 0
    elite_and_cold_hits: int = 0
    b2b_picks: int = 0
    b2b_hits: int = 0
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # All picks
    all_picks: List[PropPickV14Under] = field(default_factory=list)
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
    def elite_and_cold_rate(self) -> float:
        return self.elite_and_cold_hits / self.elite_and_cold_picks * 100 if self.elite_and_cold_picks > 0 else 0.0
    
    def summary(self) -> str:
        def pct(h, t):
            return f"{h/t*100:.1f}%" if t > 0 else "N/A"
        
        lines = [
            f"{'='*70}",
            f"MODEL V14 UNDER - BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            "",
            f"OVERALL UNDER: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            "",
            f"BY LINE SOURCE:",
            f"  Sportsbook: {pct(self.sportsbook_hits, self.sportsbook_picks)} ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived:    {pct(self.derived_hits, self.derived_picks)} ({self.derived_hits}/{self.derived_picks})",
            "",
            f"BY TIER:",
            f"  PREMIUM:  {pct(self.premium_hits, self.premium_picks)} ({self.premium_hits}/{self.premium_picks})",
            f"  HIGH:     {pct(self.high_hits, self.high_picks)} ({self.high_hits}/{self.high_picks})",
            f"  STANDARD: {pct(self.standard_hits, self.standard_picks)} ({self.standard_hits}/{self.standard_picks})",
            "",
            f"BY PROP TYPE:",
            f"  PTS UNDER: {pct(self.pts_hits, self.pts_picks)} ({self.pts_hits}/{self.pts_picks})",
            f"  REB UNDER: {pct(self.reb_hits, self.reb_picks)} ({self.reb_hits}/{self.reb_picks})",
            f"  AST UNDER: {pct(self.ast_hits, self.ast_picks)} ({self.ast_hits}/{self.ast_picks})",
            "",
            f"BY KEY FACTORS:",
            f"  Elite Defense:         {pct(self.elite_defense_hits, self.elite_defense_picks)} ({self.elite_defense_hits}/{self.elite_defense_picks})",
            f"  Cold Streak:           {pct(self.cold_streak_hits, self.cold_streak_picks)} ({self.cold_streak_hits}/{self.cold_streak_picks})",
            f"  Elite Defense + Cold:  {pct(self.elite_and_cold_hits, self.elite_and_cold_picks)} ({self.elite_and_cold_hits}/{self.elite_and_cold_picks}) *** PREMIUM ***",
            f"  B2B Fatigue:           {pct(self.b2b_hits, self.b2b_picks)} ({self.b2b_hits}/{self.b2b_picks})",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ============================================================================
# Core Model Functions
# ============================================================================

def _detect_under_factors(
    stats: PlayerStatsV14,
    prop_type: str,
    defense: DefenseContextV14,
    b2b: BackToBackInfo,
    config: ModelConfigV14Under,
) -> UnderFactors:
    """
    Detect all negative factors for an UNDER pick.
    
    Returns factors with cumulative score and adjustment.
    """
    factors = UnderFactors()
    pt = prop_type.lower()
    weights = config.weights
    adjs = config.adjustments
    
    # === DEFENSE FACTORS (PRIMARY) ===
    defense_rank = defense.get_rank(pt)
    
    if defense_rank <= config.elite_defense_rank:
        factors.add_factor(
            "defense_elite", weights["defense_elite"], adjs["defense_elite"],
            f"Elite defense at position: #{defense_rank} in league"
        )
    elif defense_rank <= config.good_defense_rank:
        factors.add_factor(
            "defense_good", weights["defense_good"], adjs["defense_good"],
            f"Good defense at position: #{defense_rank} in league"
        )
    elif defense_rank <= config.average_defense_rank:
        factors.add_factor(
            "defense_average", weights["defense_average"], adjs["defense_average"],
            f"Above-average defense at position: #{defense_rank}"
        )
    
    # === COLD STREAK FACTORS ===
    deviation = stats.deviations_season.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    season = stats.season.get(pt, 0)
    
    if deviation <= config.cold_streak_severe_pct:
        factors.add_factor(
            "cold_streak_severe", weights["cold_streak_severe"], adjs["cold_streak_severe"],
            f"Severe cold streak: L5 ({l5:.1f}) is {deviation:.0f}% below season ({season:.1f})"
        )
    elif deviation <= config.cold_streak_mild_pct:
        factors.add_factor(
            "cold_streak_mild", weights["cold_streak_mild"], adjs["cold_streak_mild"],
            f"Mild cold streak: L5 ({l5:.1f}) is {deviation:.0f}% below season ({season:.1f})"
        )
    
    # === B2B FATIGUE FACTORS ===
    if b2b.is_second_of_b2b:
        factors.add_factor(
            "b2b_second", weights["b2b_second"], adjs["b2b_second"],
            "Second game of back-to-back"
        )
    elif b2b.is_third_in_four:
        factors.add_factor(
            "b2b_third_in_four", weights["b2b_third_in_four"], adjs["b2b_third_in_four"],
            "Third game in four nights"
        )
    
    # === VARIANCE FACTOR ===
    cv = stats.get_cv(pt)
    if cv > 0.35:  # High variance player
        factors.add_factor(
            "high_variance", weights["high_variance"], adjs["high_variance"],
            f"High variance player (CV: {cv:.2f}) - more likely to underperform"
        )
    
    return factors


def _calculate_under_confidence(
    factors: UnderFactors,
    edge_pct: float,
    line_source: str,
    config: ModelConfigV14Under,
) -> Tuple[float, str]:
    """
    Calculate confidence score and tier for an UNDER pick.
    
    Returns: (confidence_score, tier)
    """
    score = factors.total_score
    
    # Base confidence from factor score
    if score >= config.premium_score_threshold:
        base = 90.0
    elif score >= config.high_score_threshold:
        base = 78.0
    elif score >= config.standard_score_threshold:
        base = 68.0
    else:
        base = 55.0
    
    # Edge bonus
    edge_bonus = min(edge_pct / 2, 8.0)
    base += edge_bonus
    
    # Sportsbook line bonus
    if line_source == "sportsbook":
        base += config.sportsbook_confidence_boost
    
    # Elite + Cold combo bonus (the best combination)
    if factors.has_elite_defense() and factors.has_cold_streak():
        base += 5.0
    
    confidence = min(base, 100.0)
    
    # Determine tier
    if confidence >= config.premium_confidence and score >= config.premium_score_threshold:
        tier = "PREMIUM"
    elif confidence >= config.high_confidence:
        tier = "HIGH"
    elif confidence >= config.standard_confidence:
        tier = "STANDARD"
    else:
        tier = "LOW"
    
    return confidence, tier


def _analyze_player_for_under(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    game_date: str,
    config: ModelConfigV14Under,
) -> List[PropPickV14Under]:
    """
    Analyze a player for potential UNDER picks.
    
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
    
    # Analyze each prop type
    for prop_type in config.prop_types:
        pt = prop_type.lower()
        
        # Check minimum average
        season_avg = stats.season.get(pt, 0)
        min_avg = config.min_avgs.get(pt, 0)
        
        if season_avg < min_avg:
            continue
        
        # Detect negative factors
        factors = _detect_under_factors(stats, prop_type, defense, b2b, config)
        
        # Check quality requirements
        if config.require_defense_factor and not factors.has_defense_factor():
            continue
        
        if not factors.has_elite_defense():
            if factors.factor_count < config.min_factors_for_standard:
                continue
        
        if factors.total_score < config.standard_score_threshold:
            continue
        
        # Get line
        line_info = get_line(
            conn, player_id, stats.player_name, prop_type, game_date,
            stats, config.derived_line_adjustment
        )
        
        # Calculate adjusted projection
        l10_avg = stats.l10.get(pt, 0)
        adjusted_projection = l10_avg * factors.adjustment_factor
        
        # Calculate edge
        edge = calculate_edge(adjusted_projection, line_info.line, "UNDER")
        min_edge = (config.min_edge_sportsbook if line_info.is_sportsbook 
                   else config.min_edge_derived)
        
        if edge < min_edge:
            continue
        
        # Calculate confidence
        confidence, tier = _calculate_under_confidence(
            factors, edge, line_info.source, config
        )
        
        if tier == "LOW":
            continue
        
        # Create pick
        pick = PropPickV14Under(
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
            base_projection=l10_avg,
            adjusted_projection=adjusted_projection,
            adjustment_factor=factors.adjustment_factor,
            edge_pct=edge,
            factors=factors.factors,
            factor_score=factors.total_score,
            factor_count=factors.factor_count,
            confidence_score=confidence,
            confidence_tier=tier,
            defense_rank=defense.get_rank(pt),
            defense_rating=defense.get_rating(pt),
            is_b2b=b2b.is_second_of_b2b,
            is_cold=factors.has_cold_streak(),
            l5_avg=stats.l5.get(pt, 0),
            l10_avg=l10_avg,
            season_avg=season_avg,
            reasons=factors.reasons,
        )
        picks.append(pick)
    
    return picks


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_v14_under(
    game_date: str,
    config: Optional[ModelConfigV14Under] = None,
    db_path: Optional[Path] = None,
) -> DailyPicksV14Under:
    """
    Generate UNDER picks for a specific date using Model V14 Under.
    
    Args:
        game_date: Date string (YYYY-MM-DD)
        config: Optional model configuration
        db_path: Optional database path
    
    Returns:
        DailyPicksV14Under with all UNDER picks for the date
    """
    if config is None:
        config = ModelConfigV14Under()
    
    if db_path is None:
        db_path = get_paths().db_path
    
    db = Db(path=db_path)
    result = DailyPicksV14Under(date=game_date, games=0, config=config)
    
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
            away_players = get_players_in_game(conn, away_abbrev, game_date)
            
            for player_id in away_players:
                if player_id in injured_ids:
                    continue
                
                result.players_analyzed += 1
                
                picks = _analyze_player_for_under(
                    conn, player_id, home_abbrev, game_date, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                
                all_picks.extend(picks)
            
            # Analyze home team players (vs away defense)
            home_players = get_players_in_game(conn, home_abbrev, game_date)
            
            for player_id in home_players:
                if player_id in injured_ids:
                    continue
                
                result.players_analyzed += 1
                
                picks = _analyze_player_for_under(
                    conn, player_id, away_abbrev, game_date, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        result.players_with_sportsbook_lines += 1
                    else:
                        result.players_with_derived_lines += 1
                
                all_picks.extend(picks)
        
        # Sort by factor score and confidence
        all_picks.sort(key=lambda p: (-p.factor_score, -p.confidence_score))
        
        # Apply limits
        final_picks = []
        for pick in all_picks:
            if player_pick_counts.get(pick.player_id, 0) >= config.max_picks_per_player:
                continue
            
            if len(final_picks) >= config.max_picks_per_day:
                break
            
            final_picks.append(pick)
            player_pick_counts[pick.player_id] = player_pick_counts.get(pick.player_id, 0) + 1
        
        result.picks = final_picks
    
    return result


def run_backtest_v14_under(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV14Under] = None,
    db_path: Optional[Path] = None,
    verbose: bool = False,
) -> BacktestResultV14Under:
    """
    Run comprehensive backtest for Model V14 Under.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Optional model configuration
        db_path: Optional database path
        verbose: Print progress
    
    Returns:
        BacktestResultV14Under with comprehensive metrics
    """
    if config is None:
        config = ModelConfigV14Under()
    
    if db_path is None:
        db_path = get_paths().db_path
    
    db = Db(path=db_path)
    result = BacktestResultV14Under(
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
        daily = get_daily_picks_v14_under(date_str, config, db_path)
        
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
                    continue
                
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
                
                # By key factors
                has_elite = pick.factors.get("defense_elite", False)
                has_cold = (pick.factors.get("cold_streak_severe", False) or 
                           pick.factors.get("cold_streak_mild", False))
                has_b2b = pick.factors.get("b2b_second", False)
                
                if has_elite:
                    result.elite_defense_picks += 1
                    if hit:
                        result.elite_defense_hits += 1
                
                if has_cold:
                    result.cold_streak_picks += 1
                    if hit:
                        result.cold_streak_hits += 1
                
                if has_elite and has_cold:
                    result.elite_and_cold_picks += 1
                    if hit:
                        result.elite_and_cold_hits += 1
                
                if has_b2b:
                    result.b2b_picks += 1
                    if hit:
                        result.b2b_hits += 1
                
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
    
    parser = argparse.ArgumentParser(description="Model V14 Under - NBA Props")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_v14_under(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_v14_under(args.date)
        print(picks.summary())
    else:
        from datetime import date
        today = date.today().strftime("%Y-%m-%d")
        picks = get_daily_picks_v14_under(today)
        print(picks.summary())
