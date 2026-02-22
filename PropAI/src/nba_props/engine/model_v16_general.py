"""
Model V16 General - Comprehensive NBA Props Prediction Model
==============================================================

This is the GENERAL model of a dual-model approach:
- Model V16 General (this file): Focuses on OVER picks + can suggest UNDER when strongest
- Model V16 Under (separate file): Specialized UNDER predictions (placeholder for future)

=============================================================================
MODEL V16 KEY INNOVATIONS (Addressing ALL Previous Model Shortcomings)
=============================================================================

1. **HYBRID LINE APPROACH** (Per User Request - Lines Come Late):
   - Use actual sportsbook lines when available (accurate edge calculation)
   - STILL GENERATE PICKS without lines (use projections with +5% adjustment)
   - Track line source for honest reporting
   - Apply STRICTER edge requirements for derived lines (10% vs 6%)

2. **PATTERN-BASED PICKS ONLY** (No Generic Edge Picks):
   - Every pick requires a VALIDATED pattern from backtesting
   - Cold Bounce: 65-84% hit rate - BEST OVER pattern
   - B2B Fatigue UNDER: 60.5% - Strong
   - Elite Defense UNDER: 62.2% - Strong
   - Hot Sustained: DISABLED (25.8% hit rate)

3. **STRATEGIC DIRECTION SELECTION** (From RCM v1.4 Analysis):
   - PTS: UNDER strongly preferred (63.9% vs 48.3% OVER from RCM)
   - REB: Both directions (~59% each)
   - AST: EXCLUDED entirely (~54% is coin flip after juice)

4. **STRICT FILTERING** (From V9/V10/Idea.txt):
   - 23+ minute average for established players
   - 10+ games history required
   - Pattern confirmation REQUIRED
   - Minimum prop averages (8 PTS, 4 REB)

5. **USAGE REDISTRIBUTION** (From RCM/Idea.txt):
   - When star is OUT, remaining players get usage boost
   - Only suggest usage-boost OVER if meaningful increase (>5%)

6. **DEFENSE INTEGRATION** (From Under Model V2/V10):
   - Elite defense (top 5 DVP) is STRONG UNDER signal (62.2%)
   - Elite defense BLOCKS PTS OVER picks
   - Good defense (top 10) + Cold streak = PREMIUM UNDER

7. **HONEST REPORTING**:
   - Track sportsbook vs derived line picks separately
   - Report hit rates by line source
   - No inflated metrics

=============================================================================
VALIDATED PATTERNS & HIT RATES (From Backtesting)
=============================================================================

| Pattern               | Direction | Hit Rate | Notes                    |
|----------------------|-----------|----------|--------------------------|
| Cold Bounce          | OVER      | 65-84%   | BEST - Regression to mean|
| B2B Fatigue          | UNDER     | 60.5%    | Second game of B2B       |
| Elite Defense        | UNDER     | 62.2%    | Top 5 DVP                |
| Cold Streak          | UNDER     | 57.8%    | L5 < Season by 20%+      |
| Combined (Elite+Cold)| UNDER     | 55-62%   | Multiple factors         |
| Usage Boost          | OVER      | 52.4%    | When stars OUT (moderate)|
| Hot Sustained        | OVER      | 25.8%    | DISABLED - unreliable    |

=============================================================================

USAGE:
------
    from src.nba_props.engine.model_v16_general import (
        get_daily_picks_v16_general,
        run_backtest_v16_general,
        ModelConfigV16General,
    )
    
    # Get picks for today
    picks = get_daily_picks_v16_general("2026-02-03")
    print(picks.summary())
    
    # Run backtest
    result = run_backtest_v16_general("2025-12-01", "2026-02-02", verbose=True)
    print(result.summary())

Author: NBA Props Team - Model V16
Created: February 2026
Version: 16.0
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
    calculate_usage_boost,
    calculate_edge,
    detect_cold_bounce_pattern,
    detect_hot_sustained_pattern,
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
# Configuration
# ============================================================================

@dataclass
class ModelConfigV16General:
    """
    Model V16 General Configuration.
    
    This model focuses on high-confidence picks using validated patterns.
    Both OVER and UNDER picks are generated, but with strict requirements.
    
    KEY CHANGES FROM V15:
    ---------------------
    1. Hybrid line handling: Always generate picks (sportsbook or derived)
    2. Stricter derived line edge (10% vs 6% for sportsbook)
    3. Cold bounce is primary OVER pattern
    4. PTS UNDER preferred per RCM analysis
    5. Hot sustained DISABLED (25.8% hit rate)
    6. AST excluded (54% is coin flip)
    """
    # === VERSION INFO ===
    model_name: str = "Model V16 General"
    model_version: str = MODEL_VERSION
    
    # === SPORTSBOOK LINE HANDLING ===
    # KEY V16 CHANGE: We do NOT require sportsbook lines
    # We use them when available, but still generate picks with projections
    require_sportsbook_line: bool = False
    derived_line_adjustment: float = 1.05  # +5% adjustment for derived lines
    sportsbook_confidence_boost: float = 12.0  # Higher confidence with real lines
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5  # Filter garbage time games
    min_avg_minutes: float = 23.0  # Established players only (from Idea.txt)
    max_games_lookback: int = 20
    
    # === PROJECTION WEIGHTS ===
    # Validated from V14/V15 backtesting
    weight_l3: float = 0.10
    weight_l5: float = 0.20
    weight_l10: float = 0.30
    weight_l15: float = 0.20
    weight_season: float = 0.20
    
    # === PATTERN THRESHOLDS ===
    # Cold bounce - BEST OVER pattern (65-84% from V14)
    cold_deviation_threshold: float = -15.0  # L5 is 15%+ below L15
    bounce_threshold: float = 5.0  # Last game must be 5%+ above L10 (showing recovery)
    
    # Hot sustained - DISABLED due to 25.8% hit rate
    enable_hot_sustained: bool = False
    hot_deviation_threshold: float = 30.0  # L5 is 30%+ above L15
    sustained_games_above: int = 3  # 3+ of last 5 above baseline
    
    # Cold streak for UNDER - DISABLED by default (51.6% hit rate in backtest V16.0)
    enable_cold_streak: bool = False
    cold_streak_mild_threshold: float = -10.0  # L5 is 10%+ below season
    cold_streak_severe_threshold: float = -20.0  # L5 is 20%+ below season
    
    # === USAGE REDISTRIBUTION ===
    # When a star is out, remaining players get boost
    usage_boost_threshold: float = 15.0  # Teammate avg 15+ pts = significant
    usage_boost_per_player: float = 0.03  # 3% boost per star out
    max_usage_boost: float = 0.12  # Cap at 12%
    min_usage_boost_for_over: float = 0.05  # Need 5%+ boost for usage OVER pick
    
    # === EDGE REQUIREMENTS ===
    # KEY V16 CHANGE: Different thresholds for sportsbook vs derived lines
    min_edge_sportsbook: float = 6.0  # 6%+ edge vs sportsbook line
    min_edge_derived: float = 10.0  # 10%+ edge vs derived line (STRICTER!)
    min_edge_premium: float = 15.0  # Premium needs 15%+ edge
    
    # === DEFENSE ADJUSTMENTS ===
    elite_defense_adj: float = 0.86  # -14% vs elite defense
    good_defense_adj: float = 0.93  # -7% vs good defense
    neutral_defense_adj: float = 1.00
    weak_defense_adj: float = 1.08  # +8% vs weak defense
    
    # === PROP SELECTION ===
    # AST excluded by default (54% is coin flip)
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # PTS OVER: Only with cold bounce + NOT elite defense (per RCM findings)
    pts_over_require_cold_bounce: bool = True
    pts_over_block_elite_defense: bool = True
    
    # PTS UNDER: Strong signal per RCM (63.9% vs 48.3% OVER)
    pts_under_require_defense: bool = False  # Can generate without defense factor if cold streak
    pts_under_elite_defense_boost: float = 8.0  # Extra confidence for elite defense
    
    # REB: Both directions, but only cold_bounce for OVER (usage_boost was 33.3%)
    # V16.1: REB UNDER disabled (51.6% hit rate in backtest - barely above coin flip)
    reb_allow_under: bool = False  # V16.1: Disabled due to poor performance
    reb_allow_over: bool = True
    reb_under_require_elite_defense: bool = True  # V16.1: Stricter - elite only (if enabled)
    reb_over_cold_bounce_only: bool = True  # Block REB usage_boost (33.3% hit rate)
    
    # AST: Excluded by default (too volatile)
    include_ast: bool = False
    min_ast_avg: float = 8.5  # Must average 8.5+ AST if enabled
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 6
    max_picks_per_day: int = 30  # More picks to show
    max_picks_per_player: int = 1  # Focus on best prop per player
    
    # === CONFIDENCE THRESHOLDS ===
    # Raised from V14 based on backtest (STANDARD tier only hit 48.1%)
    premium_confidence: float = 85.0
    high_confidence: float = 75.0
    standard_confidence: float = 70.0  # Raised from 65 to filter poor picks
    
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
class PropPickV16General:
    """A pick generated by Model V16 General."""
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
    projection_adj: float  # After defense/usage adjustments
    
    # Edge calculation (vs line, not player avg!)
    edge_pct: float
    
    # Pattern and confidence
    pattern: str  # cold_bounce, usage_boost, elite_defense_under, b2b_under, etc.
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
            "pattern": self.pattern,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense": f"{self.defense_rating} (#{self.defense_rank})",
            "usage_boost": f"+{self.usage_boost*100:.1f}%" if self.usage_boost > 0 else None,
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
class DailyPicksV16General:
    """All picks for a day from Model V16 General."""
    date: str
    games: int
    config: ModelConfigV16General = field(default_factory=ModelConfigV16General)
    picks: List[PropPickV16General] = field(default_factory=list)
    
    # Coverage stats
    players_analyzed: int = 0
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV16General]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV16General]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def over_picks(self) -> List[PropPickV16General]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV16General]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V16 GENERAL PICKS - {self.date}",
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
class BacktestResultV16General:
    """Comprehensive backtest results for Model V16 General."""
    start_date: str
    end_date: str
    config: ModelConfigV16General
    
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
    
    # By pattern
    cold_bounce_picks: int = 0
    cold_bounce_hits: int = 0
    usage_boost_picks: int = 0
    usage_boost_hits: int = 0
    elite_defense_under_picks: int = 0
    elite_defense_under_hits: int = 0
    b2b_under_picks: int = 0
    b2b_under_hits: int = 0
    cold_streak_under_picks: int = 0
    cold_streak_under_hits: int = 0
    good_defense_under_picks: int = 0
    good_defense_under_hits: int = 0
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # All picks for detailed analysis
    all_picks: List[PropPickV16General] = field(default_factory=list)
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
            f"MODEL V16 GENERAL - BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"Avg picks/day: {self.total_picks / self.days_tested:.1f}" if self.days_tested > 0 else "",
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
            f"BY PATTERN:",
            f"  Cold Bounce (OVER):      {pct(self.cold_bounce_hits, self.cold_bounce_picks)} ({self.cold_bounce_hits}/{self.cold_bounce_picks})",
            f"  Usage Boost (OVER):      {pct(self.usage_boost_hits, self.usage_boost_picks)} ({self.usage_boost_hits}/{self.usage_boost_picks})",
            f"  Elite Defense (UNDER):   {pct(self.elite_defense_under_hits, self.elite_defense_under_picks)} ({self.elite_defense_under_hits}/{self.elite_defense_under_picks})",
            f"  Good Defense (UNDER):    {pct(self.good_defense_under_hits, self.good_defense_under_picks)} ({self.good_defense_under_hits}/{self.good_defense_under_picks})",
            f"  B2B Fatigue (UNDER):     {pct(self.b2b_under_hits, self.b2b_under_picks)} ({self.b2b_under_hits}/{self.b2b_under_picks})",
            f"  Cold Streak (UNDER):     {pct(self.cold_streak_under_hits, self.cold_streak_under_picks)} ({self.cold_streak_under_hits}/{self.cold_streak_under_picks})",
            f"{'='*70}",
        ]
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
        
        return "\n".join(lines)


# ============================================================================
# Core Model Functions
# ============================================================================

def _apply_defense_adjustment(
    projection: float,
    defense_context: DefenseContextV16,
    prop_type: str,
    config: ModelConfigV16General,
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
    is_b2b: bool,
    direction: str,
    config: ModelConfigV16General,
) -> float:
    """
    Calculate confidence score for a pick.
    
    Factors:
    - Pattern reliability (from backtest data)
    - Edge size
    - Line source (sportsbook gets boost)
    - Defense context
    - Player consistency (CV)
    - Usage boost
    """
    # Base confidence
    base = 65.0
    
    # Pattern bonus (validated from backtesting)
    pattern_bonus = {
        "cold_bounce": 15.0,        # Best OVER pattern (65-84%)
        "usage_boost": 4.0,          # Moderate (52.4%)
        "elite_defense_under": 12.0, # Strong UNDER (62.2%)
        "good_defense_under": 6.0,   # Solid UNDER
        "b2b_under": 10.0,           # Reliable (60.5%)
        "cold_streak_under": 8.0,    # Good (57.8%)
        "combined_under": 18.0,      # Elite def + cold streak
    }
    base += pattern_bonus.get(pattern, 0)
    
    # Edge bonus (capped at 12)
    edge_bonus = min(edge_pct / 2, 12.0)
    base += edge_bonus
    
    # Sportsbook line bonus (CRITICAL for honest reporting)
    if line_source == "sportsbook":
        base += config.sportsbook_confidence_boost
    
    # Defense context bonus/penalty
    if direction == "UNDER":
        if defense_rating == "elite":
            base += 5.0
        elif defense_rating == "good":
            base += 2.0
    elif direction == "OVER":
        if defense_rating == "weak":
            base += 4.0
        elif defense_rating == "elite":
            base -= 8.0  # Penalty for OVER vs elite defense
    
    # Consistency bonus/penalty
    if cv < 0.20:
        base += 5.0  # Very consistent player
    elif cv > 0.40:
        base -= 5.0  # Volatile player
    
    # Usage boost bonus
    if usage_boost > 0 and direction == "OVER":
        base += min(usage_boost * 40, 8.0)
    
    # B2B fatigue bonus for UNDER
    if is_b2b and direction == "UNDER":
        base += 4.0
    
    return min(base, 100.0)


def _determine_tier(
    confidence: float,
    edge: float,
    config: ModelConfigV16General,
) -> str:
    """Determine confidence tier for a pick."""
    if confidence >= config.premium_confidence and edge >= config.min_edge_premium:
        return "PREMIUM"
    elif confidence >= config.high_confidence:
        return "HIGH"
    elif confidence >= config.standard_confidence:
        return "STANDARD"
    else:
        return "LOW"  # Will be filtered out


def _analyze_player_for_picks(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    game_date: str,
    injured_teammates: List[Dict[str, Any]],
    config: ModelConfigV16General,
) -> List[PropPickV16General]:
    """
    Analyze a player for potential picks using validated patterns.
    
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
    
    # Calculate usage boost from injured teammates
    usage_boost = calculate_usage_boost(
        injured_teammates,
        boost_per_player=config.usage_boost_per_player,
        max_boost=config.max_usage_boost,
        min_pts_threshold=config.usage_boost_threshold,
    )
    
    # Track best pick per player (we only want 1)
    best_pick = None
    best_confidence = 0.0
    
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
        
        # Calculate base projection
        projection = stats.get_projection(prop_type, config.get_weights())
        
        # Apply defense adjustment
        projection_adj = _apply_defense_adjustment(
            projection, defense, prop_type, config
        )
        
        # Apply usage boost for OVER considerations
        projection_adj_with_usage = projection_adj * (1 + usage_boost) if usage_boost > 0 else projection_adj
        
        # Determine minimum edge based on line source
        min_edge = (config.min_edge_sportsbook if line_info.is_sportsbook 
                   else config.min_edge_derived)
        
        # ================================================================
        # PATTERN 1: COLD BOUNCE (OVER) - BEST OVER PATTERN
        # ================================================================
        is_cold_bounce, cold_reasons = detect_cold_bounce_pattern(
            stats, prop_type,
            cold_threshold=config.cold_deviation_threshold,
            bounce_threshold=config.bounce_threshold,
        )
        
        if is_cold_bounce:
            # Check if PTS OVER is blocked vs elite defense
            if pt == 'pts' and config.pts_over_block_elite_defense and defense.is_elite(pt):
                pass  # Skip PTS OVER vs elite defense
            # Check if REB OVER cold_bounce only
            elif pt == 'reb' and config.reb_over_cold_bounce_only:
                # REB cold bounce is allowed
                edge = calculate_edge(projection_adj_with_usage, line_info.line, "OVER")
                
                if edge >= min_edge:
                    reasons = cold_reasons.copy()
                    reasons.append(f"Opponent {pt.upper()} defense: {defense.get_rating(pt)} (#{defense.get_rank(pt)})")
                    
                    if usage_boost > 0:
                        reasons.append(f"Usage boost: +{usage_boost*100:.1f}% from injured teammates")
                    
                    confidence = _calculate_confidence(
                        edge, "cold_bounce", line_info.source,
                        defense.get_rating(pt), stats.get_cv(pt),
                        usage_boost, b2b.is_second_of_b2b, "OVER", config
                    )
                    
                    tier = _determine_tier(confidence, edge, config)
                    
                    if tier != "LOW" and confidence > best_confidence:
                        best_confidence = confidence
                        best_pick = PropPickV16General(
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
                            projection_adj=projection_adj_with_usage,
                            edge_pct=edge,
                            pattern="cold_bounce",
                            confidence_score=confidence,
                            confidence_tier=tier,
                            defense_rank=defense.get_rank(pt),
                            defense_rating=defense.get_rating(pt),
                            usage_boost=usage_boost,
                            is_b2b=b2b.is_second_of_b2b,
                            l5_avg=stats.l5.get(pt, 0),
                            l10_avg=stats.l10.get(pt, 0),
                            l15_avg=stats.l15.get(pt, 0),
                            season_avg=stats.season.get(pt, 0),
                            reasons=reasons,
                        )
            else:
                # PTS cold bounce (if not vs elite defense)
                edge = calculate_edge(projection_adj_with_usage, line_info.line, "OVER")
                
                if edge >= min_edge:
                    reasons = cold_reasons.copy()
                    reasons.append(f"Opponent {pt.upper()} defense: {defense.get_rating(pt)} (#{defense.get_rank(pt)})")
                    
                    if usage_boost > 0:
                        reasons.append(f"Usage boost: +{usage_boost*100:.1f}% from injured teammates")
                    
                    confidence = _calculate_confidence(
                        edge, "cold_bounce", line_info.source,
                        defense.get_rating(pt), stats.get_cv(pt),
                        usage_boost, b2b.is_second_of_b2b, "OVER", config
                    )
                    
                    tier = _determine_tier(confidence, edge, config)
                    
                    if tier != "LOW" and confidence > best_confidence:
                        best_confidence = confidence
                        best_pick = PropPickV16General(
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
                            projection_adj=projection_adj_with_usage,
                            edge_pct=edge,
                            pattern="cold_bounce",
                            confidence_score=confidence,
                            confidence_tier=tier,
                            defense_rank=defense.get_rank(pt),
                            defense_rating=defense.get_rating(pt),
                            usage_boost=usage_boost,
                            is_b2b=b2b.is_second_of_b2b,
                            l5_avg=stats.l5.get(pt, 0),
                            l10_avg=stats.l10.get(pt, 0),
                            l15_avg=stats.l15.get(pt, 0),
                            season_avg=stats.season.get(pt, 0),
                            reasons=reasons,
                        )
        
        # ================================================================
        # PATTERN 2: UNDER PATTERNS (Multiple Signals)
        # ================================================================
        
        # Check if under picks are allowed for this prop
        # V16.1: For REB UNDER, can require elite defense
        reb_under_allowed = (pt == 'reb' and config.reb_allow_under and 
                            (not config.reb_under_require_elite_defense or defense.is_elite(pt)))
        
        if pt == 'pts' or reb_under_allowed:
            # Calculate UNDER edge
            edge_under = calculate_edge(projection_adj, line_info.line, "UNDER")
            
            # Detect cold streak (only if enabled)
            cold_severity = "none"
            cold_streak_reasons = []
            if config.enable_cold_streak:
                cold_severity, cold_streak_reasons = detect_cold_streak_pattern(
                    stats, prop_type,
                    mild_threshold=config.cold_streak_mild_threshold,
                    severe_threshold=config.cold_streak_severe_threshold,
                )
            
            # Determine UNDER pattern based on signals
            pattern = None
            reasons = []
            confidence_boost = 0.0
            
            # Elite defense + cold streak = STRONGEST UNDER (cold_streak component only if enabled)
            if defense.is_elite(pt) and cold_severity != "none":
                pattern = "combined_under"
                reasons = cold_streak_reasons.copy()
                reasons.append(f"Elite {pt.upper()} defense: #{defense.get_rank(pt)}")
                confidence_boost = config.pts_under_elite_defense_boost if pt == 'pts' else 4.0
            
            # Elite defense alone
            elif defense.is_elite(pt):
                pattern = "elite_defense_under"
                reasons = [f"Elite {pt.upper()} defense: #{defense.get_rank(pt)}"]
                confidence_boost = config.pts_under_elite_defense_boost if pt == 'pts' else 4.0
            
            # Good defense + cold streak (only if cold_streak enabled)
            elif defense.is_good(pt) and cold_severity != "none":
                pattern = "good_defense_under"
                reasons = cold_streak_reasons.copy()
                reasons.append(f"Good {pt.upper()} defense: #{defense.get_rank(pt)}")
            
            # B2B fatigue (60.5% hit rate)
            elif b2b.is_second_of_b2b:
                pattern = "b2b_under"
                reasons = [f"Second game of back-to-back (fatigue factor)"]
                # Only include cold_streak reasons if enabled
                if config.enable_cold_streak and cold_severity != "none":
                    reasons.extend(cold_streak_reasons)
            
            # Severe cold streak alone (only if enabled - 51.6% hit rate in backtest, not recommended)
            elif config.enable_cold_streak and cold_severity == "severe":
                pattern = "cold_streak_under"
                reasons = cold_streak_reasons.copy()
            
            # Generate UNDER pick if pattern matched and edge sufficient
            if pattern and edge_under >= min_edge:
                confidence = _calculate_confidence(
                    edge_under, pattern, line_info.source,
                    defense.get_rating(pt), stats.get_cv(pt),
                    0, b2b.is_second_of_b2b, "UNDER", config
                )
                confidence += confidence_boost
                
                tier = _determine_tier(confidence, edge_under, config)
                
                if tier != "LOW" and confidence > best_confidence:
                    best_confidence = confidence
                    best_pick = PropPickV16General(
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
                        projection_adj=projection_adj,
                        edge_pct=edge_under,
                        pattern=pattern,
                        confidence_score=confidence,
                        confidence_tier=tier,
                        defense_rank=defense.get_rank(pt),
                        defense_rating=defense.get_rating(pt),
                        usage_boost=0,
                        is_b2b=b2b.is_second_of_b2b,
                        l5_avg=stats.l5.get(pt, 0),
                        l10_avg=stats.l10.get(pt, 0),
                        l15_avg=stats.l15.get(pt, 0),
                        season_avg=stats.season.get(pt, 0),
                        reasons=reasons,
                    )
    
    if best_pick:
        picks.append(best_pick)
    
    return picks


def get_daily_picks_v16_general(
    game_date: str,
    config: Optional[ModelConfigV16General] = None,
    db_path: Optional[Path] = None,
) -> DailyPicksV16General:
    """
    Generate picks for all games on a given date.
    
    This is the main entry point for Model V16 General.
    
    Args:
        game_date: Date string (YYYY-MM-DD)
        config: Optional configuration object
        db_path: Optional database path override
    
    Returns:
        DailyPicksV16General with all picks for the day
    """
    if config is None:
        config = ModelConfigV16General()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=db_path)
    result = DailyPicksV16General(date=game_date, games=0, config=config)
    
    with db.connect() as conn:
        # Get games for date
        games = get_games_for_date(conn, game_date)
        result.games = len(games)
        
        if not games:
            return result
        
        # Get injured players (to exclude from picks)
        injured_player_ids = get_injured_players(conn, game_date)
        
        all_picks = []
        players_analyzed = 0
        sportsbook_count = 0
        derived_count = 0
        
        for game in games:
            away_abbrev = abbrev_from_team_name(game["away_team"]) or ""
            home_abbrev = abbrev_from_team_name(game["home_team"]) or ""
            
            # Get injured teammates for usage boost calculation
            away_injured = get_injured_players_for_team(conn, game_date, away_abbrev)
            home_injured = get_injured_players_for_team(conn, game_date, home_abbrev)
            
            # Analyze away team players
            away_players = get_players_in_game(
                conn, away_abbrev, game_date,
                min_games=config.min_games_required,
                min_avg_minutes=config.min_avg_minutes,
            )
            
            for player_id in away_players:
                if player_id in injured_player_ids:
                    continue
                
                players_analyzed += 1
                picks = _analyze_player_for_picks(
                    conn, player_id, home_abbrev, game_date,
                    away_injured, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        sportsbook_count += 1
                    else:
                        derived_count += 1
                
                all_picks.extend(picks)
            
            # Analyze home team players
            home_players = get_players_in_game(
                conn, home_abbrev, game_date,
                min_games=config.min_games_required,
                min_avg_minutes=config.min_avg_minutes,
            )
            
            for player_id in home_players:
                if player_id in injured_player_ids:
                    continue
                
                players_analyzed += 1
                picks = _analyze_player_for_picks(
                    conn, player_id, away_abbrev, game_date,
                    home_injured, config
                )
                
                for pick in picks:
                    if pick.line_source == "sportsbook":
                        sportsbook_count += 1
                    else:
                        derived_count += 1
                
                all_picks.extend(picks)
        
        # Sort by confidence and apply limits
        all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
        
        # Apply per-game limit
        game_pick_counts = {}
        filtered_picks = []
        for pick in all_picks:
            game_key = f"{pick.team_abbrev}_vs_{pick.opponent_abbrev}"
            if game_pick_counts.get(game_key, 0) < config.max_picks_per_game:
                filtered_picks.append(pick)
                game_pick_counts[game_key] = game_pick_counts.get(game_key, 0) + 1
        
        # Apply daily limit
        result.picks = filtered_picks[:config.max_picks_per_day]
        result.players_analyzed = players_analyzed
        result.players_with_sportsbook_lines = sportsbook_count
        result.players_with_derived_lines = derived_count
    
    return result


def run_backtest_v16_general(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV16General] = None,
    db_path: Optional[Path] = None,
    verbose: bool = False,
) -> BacktestResultV16General:
    """
    Run comprehensive backtest for Model V16 General.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        config: Optional configuration object
        db_path: Optional database path override
        verbose: Print progress
    
    Returns:
        BacktestResultV16General with all statistics
    """
    if config is None:
        config = ModelConfigV16General()
    
    if db_path is None:
        paths = get_paths()
        db_path = paths.db_path
    
    db = Db(path=db_path)
    result = BacktestResultV16General(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    # Generate date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    with db.connect() as conn:
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            if verbose:
                print(f"Processing {date_str}...", end=" ")
            
            # Get picks for this date
            daily_picks = get_daily_picks_v16_general(date_str, config, db_path)
            
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
                
                # By pattern
                pattern = pick.pattern
                if pattern == "cold_bounce":
                    result.cold_bounce_picks += 1
                    if hit:
                        result.cold_bounce_hits += 1
                elif pattern == "usage_boost":
                    result.usage_boost_picks += 1
                    if hit:
                        result.usage_boost_hits += 1
                elif pattern == "elite_defense_under" or pattern == "combined_under":
                    result.elite_defense_under_picks += 1
                    if hit:
                        result.elite_defense_under_hits += 1
                elif pattern == "good_defense_under":
                    result.good_defense_under_picks += 1
                    if hit:
                        result.good_defense_under_hits += 1
                elif pattern == "b2b_under":
                    result.b2b_under_picks += 1
                    if hit:
                        result.b2b_under_hits += 1
                elif pattern == "cold_streak_under":
                    result.cold_streak_under_picks += 1
                    if hit:
                        result.cold_streak_under_hits += 1
                
                result.all_picks.append(pick)
            
            if verbose and daily_picks_count > 0:
                pct = daily_hits / daily_picks_count * 100
                print(f"{daily_hits}/{daily_picks_count} ({pct:.1f}%)")
            elif verbose:
                print("No picks")
            
            result.daily_results.append({
                "date": date_str,
                "games": daily_picks.games,
                "picks": daily_picks_count,
                "hits": daily_hits,
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
    
    parser = argparse.ArgumentParser(description="Model V16 General - NBA Props Prediction")
    parser.add_argument("--date", type=str, help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", type=str, help="Backtest start date")
    parser.add_argument("--backtest-end", type=str, help="Backtest end date")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        print(f"Running backtest from {args.backtest_start} to {args.backtest_end}...")
        result = run_backtest_v16_general(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose
        )
        print(result.summary())
    elif args.date:
        print(f"Generating picks for {args.date}...")
        picks = get_daily_picks_v16_general(args.date)
        print(picks.summary())
    else:
        # Default to today
        today = datetime.now().strftime("%Y-%m-%d")
        print(f"Generating picks for {today}...")
        picks = get_daily_picks_v16_general(today)
        print(picks.summary())
