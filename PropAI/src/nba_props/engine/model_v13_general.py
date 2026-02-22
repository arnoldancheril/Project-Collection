"""
Model V13 General - Market-Aware NBA Props Prediction Model (General/Over Focus)
=================================================================================

This is the first half of a dual-model approach:
- Model V13 General: Focuses on OVER picks and general value opportunities
- Model V13 Under: Focuses exclusively on UNDER opportunities (separate file)

THE KEY INSIGHT:
----------------
Previous models had a critical flaw: testing predictions against "derived lines" 
(player averages) instead of actual sportsbook betting lines. This artificially 
inflated hit rates by 5-15%.

HOWEVER, we cannot ignore players without sportsbook lines entirely - lines are 
often published late, and we still need projections. So we:
1. PRIORITIZE picks with actual sportsbook lines (higher confidence)
2. ALLOW picks with derived lines (lower confidence, flagged as "derived")
3. Track and report hit rates separately for each category

VALIDATED INSIGHTS INCORPORATED:
--------------------------------
1. PATTERNS WORK (from Model Production):
   - Cold Bounce: ~67% when player is 20%+ below L15 but showing recovery
   - Hot Sustained: ~66% when player is 30%+ above L15 with momentum

2. DIRECTION MATTERS (from RCM v1.4):
   - PTS OVER: Only 48.3% (AVOID for this model - handle in Under model)
   - REB: ~59% both directions (INCLUDE)
   - AST: ~54% (EXCLUDE for players under 8.5 avg assists)
   
3. USAGE REDISTRIBUTION:
   - When a star is OUT, their usage goes somewhere
   - Model boosts projections for players who historically benefit

4. SPORTSBOOK LINE EDGE:
   - Only bet when projection differs significantly from actual line
   - Use derived lines when sportsbook lines unavailable, but flag them

MODEL V13 GENERAL RULES:
------------------------
1. Focus on OVER picks and REB in both directions
2. Use sportsbook lines when available (PREMIUM confidence)
3. Use derived lines as fallback (STANDARD confidence)
4. Pattern confirmation required for picks
5. Apply usage redistribution when teammates are injured
6. Strict filtering: 20+ minutes, 10+ games history
7. Exclude low-value props (AST under 5.5 avg)

USAGE:
------
    from src.nba_props.engine.model_v13_general import (
        get_daily_picks_v13_general,
        run_backtest_v13_general,
        ModelConfigV13General,
    )
    
    # Get picks for today
    picks = get_daily_picks_v13_general("2026-02-03")
    
    # Run backtest
    result = run_backtest_v13_general("2025-12-01", "2026-02-02")

Author: NBA Props Team - Model V13
Created: February 2026
Version: 13.0
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
# Configuration
# ============================================================================

@dataclass
class ModelConfigV13General:
    """
    Model V13 General Configuration.
    
    This model focuses on OVER picks and general value, with sportsbook line 
    integration when available but also supporting derived lines.
    """
    # === VERSION INFO ===
    model_name: str = "Model V13 General"
    model_version: str = "13.0"
    
    # === SPORTSBOOK LINE HANDLING ===
    # Unlike V10, we DON'T require sportsbook lines - but we track them
    require_sportsbook_line: bool = False  # Allow derived lines
    sportsbook_line_confidence_boost: float = 10.0  # Boost for having real line
    derived_line_adjustment: float = 1.05  # Derived lines are typically 5% below actual
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10           # Need sufficient history
    min_minutes_filter: int = 5            # Filter garbage time games
    min_avg_minutes: float = 20.0          # Focus on rotation players
    max_games_lookback: int = 20           # Use last 20 games
    
    # === PROJECTION WEIGHTS ===
    # Balanced weighting across windows
    weight_l3: float = 0.10                # Very recent form
    weight_l5: float = 0.20                # Recent form
    weight_l10: float = 0.30               # Primary baseline
    weight_l15: float = 0.20               # Extended baseline
    weight_season: float = 0.20            # Full season stability
    
    # === PATTERN THRESHOLDS ===
    # Cold bounce-back pattern - BEST OVER pattern
    cold_deviation_threshold: float = -20.0  # L5 is 20%+ below L15
    bounce_threshold: float = 0.0            # Last game > L10 (any amount)
    
    # Hot sustained pattern
    hot_deviation_threshold: float = 30.0    # L5 is 30%+ above L15
    sustained_games_above: int = 3           # 3+ of last 5 above L15
    
    # Usage redistribution pattern
    usage_boost_threshold: float = 15.0      # Teammate avg 15+ pts = star
    usage_boost_per_player: float = 0.03     # 3% boost per star out
    max_usage_boost: float = 0.12            # Cap at 12%
    
    # === EDGE REQUIREMENTS ===
    min_edge_sportsbook: float = 6.0        # 6%+ edge vs actual line
    min_edge_derived: float = 10.0          # 10%+ edge vs derived line (harder)
    min_edge_premium: float = 10.0          # Premium picks need 10%+ edge
    
    # === DEFENSE VS POSITION ===
    elite_defense_rank: int = 5             # Top 5 = elite
    good_defense_rank: int = 10             # Top 10 = good
    poor_defense_rank: int = 25             # Bottom 5 = weak
    
    # Defense adjustments
    elite_defense_adj: float = 0.90         # -10% vs elite
    good_defense_adj: float = 0.95          # -5% vs good
    neutral_defense_adj: float = 1.00       # No change
    weak_defense_adj: float = 1.06          # +6% vs weak defense
    
    # === PROP SELECTION ===
    # PTS OVER performs poorly (48.3%) - let Under model handle PTS
    # REB works both ways (~59%)
    # AST is volatile - only for high-assist players
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # PTS: Allow OVER only with strong patterns AND weak defense
    pts_over_require_weak_defense: bool = True
    
    # REB: Both directions allowed
    reb_direction: str = "BOTH"
    
    # AST: Only for 8.5+ avg assist players
    include_ast: bool = True
    min_ast_avg_for_pick: float = 8.5
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 5
    max_picks_per_day: int = 25
    max_picks_per_player: int = 2
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence_threshold: float = 85.0
    high_confidence_threshold: float = 75.0
    standard_confidence_threshold: float = 65.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlayerStatsV13:
    """Comprehensive player statistics for Model V13."""
    player_id: int
    player_name: str
    team_abbrev: str
    position: str
    games_played: int
    
    # Averages at different windows
    l3: Dict[str, float] = field(default_factory=dict)
    l5: Dict[str, float] = field(default_factory=dict)
    l10: Dict[str, float] = field(default_factory=dict)
    l15: Dict[str, float] = field(default_factory=dict)
    l20: Dict[str, float] = field(default_factory=dict)
    season: Dict[str, float] = field(default_factory=dict)
    
    # Deviations (L5 vs L15, L5 vs Season)
    deviations_l15: Dict[str, float] = field(default_factory=dict)
    deviations_season: Dict[str, float] = field(default_factory=dict)
    
    # Last game values
    last_game: Dict[str, float] = field(default_factory=dict)
    
    # Standard deviations (for consistency)
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Recent game values (for sustained pattern)
    recent_games: Dict[str, List[float]] = field(default_factory=dict)
    
    # Average minutes
    avg_minutes: float = 0.0
    
    def get_projection(self, prop_type: str, config: ModelConfigV13General) -> float:
        """Calculate weighted projection for a prop type."""
        pt = prop_type.lower()
        
        l3_val = self.l3.get(pt, 0)
        l5_val = self.l5.get(pt, 0)
        l10_val = self.l10.get(pt, 0)
        l15_val = self.l15.get(pt, 0)
        season_val = self.season.get(pt, 0)
        
        total_weight = (config.weight_l3 + config.weight_l5 + config.weight_l10 + 
                       config.weight_l15 + config.weight_season)
        if total_weight <= 0:
            return season_val
        
        projection = (
            l3_val * config.weight_l3 +
            l5_val * config.weight_l5 +
            l10_val * config.weight_l10 +
            l15_val * config.weight_l15 +
            season_val * config.weight_season
        ) / total_weight
        
        return projection
    
    def get_cv(self, prop_type: str) -> float:
        """Get coefficient of variation (std/mean) for consistency."""
        pt = prop_type.lower()
        mean = self.l10.get(pt, 0)
        std = self.stds.get(pt, 0)
        if mean <= 0:
            return 1.0
        return std / mean
    
    def get_derived_line(self, prop_type: str) -> float:
        """Get derived line (L10 average)."""
        return self.l10.get(prop_type.lower(), 0)


@dataclass
class DefenseContextV13:
    """Defense vs position context for opponent."""
    team_abbrev: str
    position: str
    data_available: bool = False
    
    # Ranks (1 = best defense, 30 = worst)
    pts_rank: int = 15
    reb_rank: int = 15
    ast_rank: int = 15
    
    # Ratings
    pts_rating: str = "average"  # elite, good, average, weak
    reb_rating: str = "average"
    ast_rating: str = "average"
    
    def get_rating(self, prop_type: str) -> str:
        """Get defense rating for a prop type."""
        mapping = {'pts': self.pts_rating, 'reb': self.reb_rating, 'ast': self.ast_rating}
        return mapping.get(prop_type.lower(), "average")
    
    def get_rank(self, prop_type: str) -> int:
        """Get defense rank for a prop type."""
        mapping = {'pts': self.pts_rank, 'reb': self.reb_rank, 'ast': self.ast_rank}
        return mapping.get(prop_type.lower(), 15)


@dataclass
class LineInfo:
    """Information about the betting line."""
    value: float
    source: str  # "sportsbook" or "derived"
    book: str = "unknown"  # draftkings, fanduel, etc.
    
    @property
    def is_sportsbook(self) -> bool:
        return self.source == "sportsbook"


@dataclass
class PatternResult:
    """Result of pattern detection."""
    pattern_name: str
    direction: str  # OVER, UNDER
    confidence_bonus: float
    reasons: List[str]
    is_valid: bool


@dataclass
class UsageBoostInfo:
    """Information about usage boost from injured teammates."""
    boost_pct: float = 0.0
    injured_teammates: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


@dataclass
class PropPickV13General:
    """A pick generated by Model V13 General."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details
    prop_type: str     # PTS, REB, AST
    direction: str     # OVER, UNDER
    
    # Line information
    line: float
    line_source: str   # "sportsbook" or "derived"
    book: str
    
    # Projection
    projection: float
    projection_raw: float  # Before defense adjustment
    defense_adjustment: float
    usage_boost: float
    
    # Edge calculation
    edge_vs_line: float
    
    # Pattern and confidence
    pattern: str
    confidence_tier: str    # PREMIUM, HIGH, STANDARD
    confidence_score: float
    
    # Defense context
    defense_rank: int
    defense_rating: str
    
    # Supporting data
    l5_avg: float
    l10_avg: float
    l15_avg: float
    season_avg: float
    
    # Analysis
    reasons: List[str] = field(default_factory=list)
    
    # Results (filled after game)
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    margin: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
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
            "projection": round(self.projection, 1),
            "edge": f"{self.edge_vs_line:.1f}%",
            "pattern": self.pattern,
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense_rank": self.defense_rank,
            "defense_rating": self.defense_rating,
            "l5": round(self.l5_avg, 1),
            "l10": round(self.l10_avg, 1),
            "l15": round(self.l15_avg, 1),
            "season": round(self.season_avg, 1),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV13General:
    """All picks for a day from Model V13 General."""
    date: str
    games: int
    picks: List[PropPickV13General] = field(default_factory=list)
    
    # Coverage stats
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def sportsbook_picks(self) -> List[PropPickV13General]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV13General]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    @property
    def premium_picks(self) -> List[PropPickV13General]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def over_picks(self) -> List[PropPickV13General]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV13General]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"{'='*70}",
            f"MODEL V13 GENERAL PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games}",
            f"Total Picks: {self.total_picks}",
            f"  - With sportsbook lines: {len(self.sportsbook_picks)}",
            f"  - With derived lines: {len(self.derived_picks)}",
            f"  - OVER: {len(self.over_picks)}, UNDER: {len(self.under_picks)}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    direction_emoji = "📈" if p.direction == "OVER" else "📉"
                    line_badge = "🎯" if p.line_source == "sportsbook" else "📊"
                    lines.append(
                        f"  {direction_emoji}{line_badge} {p.player_name} ({p.team_abbrev}): "
                        f"{p.prop_type} {p.direction} {p.line:.1f}"
                    )
                    lines.append(
                        f"      Proj: {p.projection:.1f} | Edge: {p.edge_vs_line:.1f}% | "
                        f"Pattern: {p.pattern}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV13General:
    """Comprehensive backtest results for Model V13 General."""
    start_date: str
    end_date: str
    config: ModelConfigV13General
    
    # Overall
    total_picks: int = 0
    hits: int = 0
    
    # By line source (CRITICAL - tracks derived vs sportsbook)
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
    
    # Coverage stats
    days_tested: int = 0
    total_games: int = 0
    
    # Detailed data
    all_picks: List[PropPickV13General] = field(default_factory=list)
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
    def premium_rate(self) -> float:
        return self.premium_hits / self.premium_picks * 100 if self.premium_picks > 0 else 0.0
    
    @property
    def over_rate(self) -> float:
        return self.over_hits / self.over_picks * 100 if self.over_picks > 0 else 0.0
    
    @property
    def under_rate(self) -> float:
        return self.under_hits / self.under_picks * 100 if self.under_picks > 0 else 0.0
    
    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            f"{'='*70}",
            f"MODEL V13 GENERAL - BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            "",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            "",
            "BY LINE SOURCE (Critical!):",
            f"  Sportsbook: {self.sportsbook_hits}/{self.sportsbook_picks} ({self.sportsbook_rate:.1f}%)" if self.sportsbook_picks else "  Sportsbook: N/A",
            f"  Derived:    {self.derived_hits}/{self.derived_picks} ({self.derived_rate:.1f}%)" if self.derived_picks else "  Derived: N/A",
            "",
            "BY TIER:",
            f"  Premium:  {self.premium_hits}/{self.premium_picks} ({self.premium_hits/self.premium_picks*100:.1f}%)" if self.premium_picks else "  Premium: N/A",
            f"  High:     {self.high_hits}/{self.high_picks} ({self.high_hits/self.high_picks*100:.1f}%)" if self.high_picks else "  High: N/A",
            f"  Standard: {self.standard_hits}/{self.standard_picks} ({self.standard_hits/self.standard_picks*100:.1f}%)" if self.standard_picks else "  Standard: N/A",
            "",
            "BY DIRECTION:",
            f"  OVER:  {self.over_hits}/{self.over_picks} ({self.over_rate:.1f}%)" if self.over_picks else "  OVER: N/A",
            f"  UNDER: {self.under_hits}/{self.under_picks} ({self.under_rate:.1f}%)" if self.under_picks else "  UNDER: N/A",
            "",
            "BY PROP TYPE:",
            f"  PTS: {self.pts_hits}/{self.pts_picks} ({self.pts_hits/self.pts_picks*100:.1f}%)" if self.pts_picks else "  PTS: N/A",
            f"  REB: {self.reb_hits}/{self.reb_picks} ({self.reb_hits/self.reb_picks*100:.1f}%)" if self.reb_picks else "  REB: N/A",
            f"  AST: {self.ast_hits}/{self.ast_picks} ({self.ast_hits/self.ast_picks*100:.1f}%)" if self.ast_picks else "  AST: N/A",
            "",
            "BY PATTERN:",
            f"  Cold Bounce:  {self.cold_bounce_hits}/{self.cold_bounce_picks} ({self.cold_bounce_hits/self.cold_bounce_picks*100:.1f}%)" if self.cold_bounce_picks else "  Cold Bounce: N/A",
            f"  Hot Sustained: {self.hot_sustained_hits}/{self.hot_sustained_picks} ({self.hot_sustained_hits/self.hot_sustained_picks*100:.1f}%)" if self.hot_sustained_picks else "  Hot Sustained: N/A",
            f"  Usage Boost:  {self.usage_boost_hits}/{self.usage_boost_picks} ({self.usage_boost_hits/self.usage_boost_picks*100:.1f}%)" if self.usage_boost_picks else "  Usage Boost: N/A",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        if ascii_name.lower().endswith(suffix):
            ascii_name = ascii_name[:-len(suffix)]
    return ascii_name.lower().strip()


def _get_injured_players(conn: sqlite3.Connection, game_date: str) -> Set[int]:
    """Get set of player IDs who are OUT or DOUBTFUL for the date."""
    rows = conn.execute(
        """
        SELECT DISTINCT player_id
        FROM injury_report
        WHERE game_date = ?
          AND status IN ('OUT', 'DOUBTFUL')
          AND player_id IS NOT NULL
        """,
        (game_date,),
    ).fetchall()
    return {row["player_id"] for row in rows}


def _get_injured_teammates(
    conn: sqlite3.Connection, 
    team_abbrev: str, 
    game_date: str,
    config: ModelConfigV13General,
) -> UsageBoostInfo:
    """
    Get information about injured high-usage teammates for usage redistribution.
    """
    boost_info = UsageBoostInfo()
    
    # Get players on this team who are OUT
    rows = conn.execute(
        """
        SELECT DISTINCT ir.player_name, p.id as player_id
        FROM injury_report ir
        LEFT JOIN players p ON ir.player_id = p.id
        JOIN teams t ON ir.team_id = t.id
        WHERE ir.game_date = ?
          AND ir.status = 'OUT'
          AND t.name LIKE ?
        """,
        (game_date, f"%{team_abbrev}%"),
    ).fetchall()
    
    for row in rows:
        player_id = row["player_id"]
        player_name = row["player_name"]
        
        if not player_id:
            continue
        
        # Check if this is a high-usage player
        avg_pts = conn.execute(
            """
            SELECT AVG(bp.pts) as avg_pts
            FROM boxscore_player bp
            JOIN games g ON g.id = bp.game_id
            WHERE bp.player_id = ?
              AND g.game_date < ?
              AND bp.minutes > 5
            ORDER BY g.game_date DESC
            LIMIT 15
            """,
            (player_id, game_date),
        ).fetchone()
        
        if avg_pts and avg_pts["avg_pts"] and avg_pts["avg_pts"] >= config.usage_boost_threshold:
            boost_info.injured_teammates.append(player_name)
            boost_info.boost_pct += config.usage_boost_per_player
            boost_info.reasons.append(f"Star out: {player_name} ({avg_pts['avg_pts']:.1f} PPG)")
    
    # Cap the boost
    boost_info.boost_pct = min(boost_info.boost_pct, config.max_usage_boost)
    
    return boost_info


def _get_sportsbook_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Optional[Tuple[float, str]]:
    """
    Get actual sportsbook line for a player/prop/date.
    Returns: (line, book) or None if no line exists.
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
            return (row["line"], row["book"] or "unknown")
    
    # Try by player name (fuzzy match)
    rows = conn.execute(
        """
        SELECT sl.line, sl.book, p.name
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.prop_type = ? AND sl.as_of_date = ?
        """,
        (prop_type.upper(), game_date)
    ).fetchall()
    
    norm_name = _normalize_name(player_name)
    for row in rows:
        if _normalize_name(row["name"]) == norm_name:
            return (row["line"], row["book"] or "unknown")
    
    return None


def _get_line_info(
    conn: sqlite3.Connection,
    stats: PlayerStatsV13,
    prop_type: str,
    game_date: str,
    config: ModelConfigV13General,
) -> LineInfo:
    """Get line information - sportsbook if available, else derived."""
    sportsbook_result = _get_sportsbook_line(
        conn, stats.player_id, stats.player_name, prop_type, game_date
    )
    
    if sportsbook_result:
        return LineInfo(
            value=sportsbook_result[0],
            source="sportsbook",
            book=sportsbook_result[1],
        )
    else:
        # Use derived line (L10 average with adjustment)
        derived = stats.get_derived_line(prop_type) * config.derived_line_adjustment
        return LineInfo(
            value=derived,
            source="derived",
            book="derived",
        )


def _get_defense_context(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
    config: ModelConfigV13General,
) -> DefenseContextV13:
    """Get defense vs position context for an opponent team."""
    context = DefenseContextV13(team_abbrev=team_abbrev, position=position)
    
    # Map position to DVP position
    pos_map = {
        'G': 'PG', 'PG': 'PG', 'SG': 'SG',
        'F': 'SF', 'SF': 'SF', 'PF': 'PF',
        'C': 'C', 'F-C': 'PF', 'G-F': 'SF'
    }
    dvp_position = pos_map.get(position.upper() if position else 'SF', 'SF')
    
    # Try to get DVP data
    row = conn.execute(
        """
        SELECT pts_rank, reb_rank, ast_rank
        FROM team_defense_vs_position
        WHERE team_abbrev = ? AND position = ?
        ORDER BY updated_at DESC LIMIT 1
        """,
        (team_abbrev, dvp_position)
    ).fetchone()
    
    if row:
        context.data_available = True
        context.pts_rank = row["pts_rank"] or 15
        context.reb_rank = row["reb_rank"] or 15
        context.ast_rank = row["ast_rank"] or 15
        
        # Determine ratings
        for stat, rank in [('pts', context.pts_rank), ('reb', context.reb_rank), ('ast', context.ast_rank)]:
            if rank <= config.elite_defense_rank:
                rating = "elite"
            elif rank <= config.good_defense_rank:
                rating = "good"
            elif rank <= config.poor_defense_rank:
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


# ============================================================================
# Core Model Functions
# ============================================================================

def _load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    config: ModelConfigV13General,
) -> Optional[PlayerStatsV13]:
    """Load comprehensive player statistics."""
    player = conn.execute(
        "SELECT id, name FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    
    if not player:
        return None
    
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
        (player_id, before_date, config.min_minutes_filter, config.max_games_lookback),
    ).fetchall()
    
    if len(rows) < config.min_games_required:
        return None
    
    games = [dict(r) for r in rows]
    n = len(games)
    
    # Check minimum average minutes
    avg_min = sum(g["minutes"] or 0 for g in games) / n
    if avg_min < config.min_avg_minutes:
        return None
    
    # Extract stats
    stats_data = {
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
    
    player_stats = PlayerStatsV13(
        player_id=player_id,
        player_name=player["name"],
        team_abbrev=abbrev_from_team_name(games[0]["team_name"]) or "",
        position=games[0].get("pos") or "G",
        games_played=n,
        avg_minutes=avg_min,
    )
    
    for stat in ['pts', 'reb', 'ast', 'min']:
        vals = stats_data[stat]
        player_stats.l3[stat] = avg(vals, 3)
        player_stats.l5[stat] = avg(vals, 5)
        player_stats.l10[stat] = avg(vals, 10)
        player_stats.l15[stat] = avg(vals, 15) if n >= 15 else avg(vals)
        player_stats.l20[stat] = avg(vals, 20) if n >= 20 else avg(vals)
        player_stats.season[stat] = avg(vals)
        player_stats.stds[stat] = safe_std(vals)
        player_stats.last_game[stat] = vals[0] if vals else 0
        player_stats.recent_games[stat] = vals[:5]
        
        # Deviations
        l15 = player_stats.l15[stat]
        season = player_stats.season[stat]
        l5 = player_stats.l5[stat]
        
        player_stats.deviations_l15[stat] = (l5 - l15) / l15 * 100 if l15 > 0 else 0.0
        player_stats.deviations_season[stat] = (l5 - season) / season * 100 if season > 0 else 0.0
    
    return player_stats


def _detect_cold_bounce_pattern(
    stats: PlayerStatsV13,
    prop_type: str,
    defense_context: DefenseContextV13,
    config: ModelConfigV13General,
) -> PatternResult:
    """
    Detect Cold Bounce pattern - BEST OVER pattern (~67%).
    Player is cold but last game shows recovery.
    """
    pt = prop_type.lower()
    
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l10 = stats.l10.get(pt, 0)
    l15 = stats.l15.get(pt, 0)
    last_game = stats.last_game.get(pt, 0)
    
    # Check if player is cold
    if deviation_l15 > config.cold_deviation_threshold:
        return PatternResult("none", "OVER", 0, [], False)
    
    # Check if last game shows bounce (above L10)
    bounce_pct = (last_game - l10) / l10 * 100 if l10 > 0 else 0
    if bounce_pct < config.bounce_threshold:
        return PatternResult("none", "OVER", 0, [], False)
    
    # For PTS OVER, require weak defense (based on analysis)
    if pt == 'pts' and config.pts_over_require_weak_defense:
        if defense_context.get_rating('pts') not in ['weak', 'average']:
            return PatternResult("none", "OVER", 0, [], False)
    
    # Don't bet against elite defense
    if defense_context.get_rating(pt) == "elite":
        return PatternResult("none", "OVER", 0, [], False)
    
    reasons = [
        f"Cold bounce: L5 ({l5:.1f}) is {deviation_l15:.0f}% below L15 ({l15:.1f})",
        f"Recovery signal: Last game ({last_game:.0f}) bounced above L10 ({l10:.1f})",
        f"Regression expected toward baseline",
    ]
    
    confidence_bonus = min(abs(deviation_l15) / 2, 12)
    
    return PatternResult("cold_bounce", "OVER", confidence_bonus, reasons, True)


def _detect_hot_sustained_pattern(
    stats: PlayerStatsV13,
    prop_type: str,
    defense_context: DefenseContextV13,
    config: ModelConfigV13General,
) -> PatternResult:
    """
    Detect Hot Sustained pattern (~66%).
    Player is hot and maintaining momentum.
    """
    pt = prop_type.lower()
    
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l3 = stats.l3.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l15 = stats.l15.get(pt, 0)
    recent = stats.recent_games.get(pt, [])
    
    # Check if player is hot
    if deviation_l15 < config.hot_deviation_threshold:
        return PatternResult("none", "OVER", 0, [], False)
    
    # Check if still hot (L3 >= L5 * 0.95)
    if l3 < l5 * 0.95:
        return PatternResult("none", "OVER", 0, [], False)
    
    # Count games above L15
    games_above = sum(1 for v in recent if v > l15)
    if games_above < config.sustained_games_above:
        return PatternResult("none", "OVER", 0, [], False)
    
    # Don't bet against elite defense
    if defense_context.get_rating(pt) == "elite":
        return PatternResult("none", "OVER", 0, [], False)
    
    # For PTS OVER, require weak defense
    if pt == 'pts' and config.pts_over_require_weak_defense:
        if defense_context.get_rating('pts') not in ['weak', 'average']:
            return PatternResult("none", "OVER", 0, [], False)
    
    reasons = [
        f"Hot sustained: L5 ({l5:.1f}) is {deviation_l15:.0f}% above L15 ({l15:.1f})",
        f"Momentum: L3 ({l3:.1f}) maintaining level",
        f"Consistency: {games_above}/5 recent games above baseline",
    ]
    
    confidence_bonus = min((deviation_l15 - config.hot_deviation_threshold) / 3, 10)
    
    return PatternResult("hot_sustained", "OVER", confidence_bonus, reasons, True)


def _detect_reb_under_pattern(
    stats: PlayerStatsV13,
    defense_context: DefenseContextV13,
    config: ModelConfigV13General,
) -> PatternResult:
    """
    Detect REB UNDER pattern.
    REB works both ways, so we can include UNDER for this model.
    """
    pt = 'reb'
    
    deviation_season = stats.deviations_season.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    season = stats.season.get(pt, 0)
    defense_rating = defense_context.get_rating(pt)
    defense_rank = defense_context.get_rank(pt)
    
    reasons = []
    confidence_bonus = 0
    is_valid = False
    
    # Elite defense for REB
    if defense_rating == "elite":
        reasons.append(f"Elite REB defense: {defense_context.team_abbrev} ranks #{defense_rank}")
        confidence_bonus += 10
        is_valid = True
    
    # Cold streak
    if deviation_season <= -15:
        reasons.append(f"Cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season ({season:.1f})")
        confidence_bonus += 8
        is_valid = True
    
    # Combined
    if defense_rating == "elite" and deviation_season <= -10:
        confidence_bonus += 5  # Bonus for combination
    
    if not is_valid:
        return PatternResult("none", "UNDER", 0, [], False)
    
    return PatternResult("reb_under", "UNDER", confidence_bonus, reasons, True)


def _generate_pick(
    conn: sqlite3.Connection,
    stats: PlayerStatsV13,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    usage_boost: UsageBoostInfo,
    config: ModelConfigV13General,
) -> Optional[PropPickV13General]:
    """Generate a pick for a player/prop combination."""
    pt = prop_type.lower()
    
    # Skip AST for low-assist players
    if pt == 'ast' and config.include_ast:
        if stats.season.get('ast', 0) < config.min_ast_avg_for_pick:
            return None
    elif pt == 'ast' and not config.include_ast:
        return None
    
    # Get line information
    line_info = _get_line_info(conn, stats, pt, game_date, config)
    
    # Skip if line is too low (not meaningful)
    if line_info.value < 3.0:
        return None
    
    # Get defense context
    defense_context = _get_defense_context(conn, opponent_abbrev, stats.position, config)
    
    # Calculate projection
    projection_raw = stats.get_projection(pt, config)
    
    # Apply defense adjustment
    defense_rating = defense_context.get_rating(pt)
    if defense_rating == "elite":
        defense_adj = config.elite_defense_adj
    elif defense_rating == "good":
        defense_adj = config.good_defense_adj
    elif defense_rating == "weak":
        defense_adj = config.weak_defense_adj
    else:
        defense_adj = config.neutral_defense_adj
    
    projection = projection_raw * defense_adj
    
    # Apply usage boost
    if usage_boost.boost_pct > 0:
        projection *= (1 + usage_boost.boost_pct)
    
    # Detect patterns
    cold_bounce = _detect_cold_bounce_pattern(stats, pt, defense_context, config)
    hot_sustained = _detect_hot_sustained_pattern(stats, pt, defense_context, config)
    
    # Also check REB UNDER pattern (REB works both ways)
    reb_under = None
    if pt == 'reb':
        reb_under = _detect_reb_under_pattern(stats, defense_context, config)
    
    # Select best pattern and direction
    selected_pattern = None
    selected_direction = None
    selected_edge = 0
    
    # Calculate edges
    over_edge = (projection - line_info.value) / line_info.value * 100 if line_info.value > 0 else 0
    under_edge = (line_info.value - projection) / line_info.value * 100 if line_info.value > 0 else 0
    
    # Determine min edge based on line source
    min_edge = config.min_edge_sportsbook if line_info.is_sportsbook else config.min_edge_derived
    
    # Check OVER patterns
    if cold_bounce.is_valid and over_edge >= min_edge:
        selected_pattern = cold_bounce
        selected_direction = "OVER"
        selected_edge = over_edge
    
    if hot_sustained.is_valid and over_edge >= min_edge:
        if not selected_pattern or hot_sustained.confidence_bonus > selected_pattern.confidence_bonus:
            selected_pattern = hot_sustained
            selected_direction = "OVER"
            selected_edge = over_edge
    
    # Check usage boost pattern (OVER)
    if usage_boost.boost_pct > 0.02 and over_edge >= min_edge:
        usage_pattern = PatternResult(
            "usage_boost", "OVER", 
            min(usage_boost.boost_pct * 100, 8),
            usage_boost.reasons,
            True
        )
        if not selected_pattern or (over_edge > selected_edge and usage_pattern.confidence_bonus >= 5):
            selected_pattern = usage_pattern
            selected_direction = "OVER"
            selected_edge = over_edge
    
    # Check REB UNDER pattern
    if reb_under and reb_under.is_valid and under_edge >= min_edge:
        if not selected_pattern or (reb_under.confidence_bonus > selected_pattern.confidence_bonus and under_edge > selected_edge):
            selected_pattern = reb_under
            selected_direction = "UNDER"
            selected_edge = under_edge
    
    if not selected_pattern or not selected_direction:
        return None
    
    # Calculate confidence
    base_confidence = 68.0
    
    # Pattern bonus
    confidence = base_confidence + selected_pattern.confidence_bonus
    
    # Sportsbook line bonus
    if line_info.is_sportsbook:
        confidence += config.sportsbook_line_confidence_boost
    
    # Edge bonus
    edge_bonus = min(selected_edge / 2, 8)
    confidence += edge_bonus
    
    # Consistency bonus/penalty
    cv = stats.get_cv(pt)
    if cv < 0.20:
        confidence += 4
    elif cv > 0.40:
        confidence -= 4
    
    confidence = min(confidence, 100)
    
    # Determine tier
    if confidence >= config.premium_confidence_threshold and selected_edge >= config.min_edge_premium:
        tier = "PREMIUM"
    elif confidence >= config.high_confidence_threshold:
        tier = "HIGH"
    elif confidence >= config.standard_confidence_threshold:
        tier = "STANDARD"
    else:
        return None  # Below threshold
    
    return PropPickV13General(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        game_date=game_date,
        prop_type=prop_type.upper(),
        direction=selected_direction,
        line=round(line_info.value, 1),
        line_source=line_info.source,
        book=line_info.book,
        projection=round(projection, 1),
        projection_raw=round(projection_raw, 1),
        defense_adjustment=defense_adj,
        usage_boost=usage_boost.boost_pct,
        edge_vs_line=round(selected_edge, 1),
        pattern=selected_pattern.pattern_name,
        confidence_tier=tier,
        confidence_score=round(confidence, 1),
        defense_rank=defense_context.get_rank(pt),
        defense_rating=defense_rating,
        l5_avg=round(stats.l5.get(pt, 0), 1),
        l10_avg=round(stats.l10.get(pt, 0), 1),
        l15_avg=round(stats.l15.get(pt, 0), 1),
        season_avg=round(stats.season.get(pt, 0), 1),
        reasons=selected_pattern.reasons,
    )


def _generate_game_picks(
    conn: sqlite3.Connection,
    game_date: str,
    team1_name: str,
    team2_name: str,
    config: ModelConfigV13General,
    coverage_stats: Dict[str, int],
) -> List[PropPickV13General]:
    """Generate picks for a single game."""
    t1_abbrev = abbrev_from_team_name(team1_name) or ""
    t2_abbrev = abbrev_from_team_name(team2_name) or ""
    
    injured = _get_injured_players(conn, game_date)
    
    all_picks = []
    player_picks = {}
    
    for team_name, opp_abbrev, team_abbrev in [
        (team1_name, t2_abbrev, t1_abbrev), 
        (team2_name, t1_abbrev, t2_abbrev)
    ]:
        team = conn.execute("SELECT id FROM teams WHERE name = ?", (team_name,)).fetchone()
        if not team:
            continue
        
        # Get usage boost for this team
        usage_boost = _get_injured_teammates(conn, team_abbrev, game_date, config)
        
        # Get team's players
        players = conn.execute(
            """
            SELECT b.player_id, AVG(b.minutes) as avg_min
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.team_id = ?
              AND g.game_date < ?
              AND b.minutes > ?
            GROUP BY b.player_id
            HAVING COUNT(*) >= ?
            ORDER BY avg_min DESC
            LIMIT 12
            """,
            (team["id"], game_date, config.min_minutes_filter, config.min_games_required),
        ).fetchall()
        
        for p in players:
            player_id = p["player_id"]
            
            if player_id in injured:
                continue
            
            if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                continue
            
            stats = _load_player_stats(conn, player_id, game_date, config)
            if not stats:
                continue
            
            # Generate picks for each prop type
            for pt in config.prop_types:
                if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                    break
                
                pick = _generate_pick(conn, stats, pt, opp_abbrev, game_date, usage_boost, config)
                
                if pick:
                    all_picks.append(pick)
                    player_picks[player_id] = player_picks.get(player_id, 0) + 1
                    
                    if pick.line_source == "sportsbook":
                        coverage_stats["sportsbook"] += 1
                    else:
                        coverage_stats["derived"] += 1
    
    return all_picks


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_v13_general(
    game_date: str,
    config: Optional[ModelConfigV13General] = None,
    db_path: str = "data/db/nba_props.sqlite3",
) -> DailyPicksV13General:
    """Generate picks for all games on a date."""
    if config is None:
        config = ModelConfigV13General()
    
    db = Db(Path(db_path))
    daily = DailyPicksV13General(date=game_date, games=0)
    
    all_picks = []
    coverage_stats = {"sportsbook": 0, "derived": 0}
    
    with db.connect() as conn:
        games = conn.execute(
            """
            SELECT g.id, t1.name as team1, t2.name as team2
            FROM games g
            JOIN teams t1 ON t1.id = g.team1_id
            JOIN teams t2 ON t2.id = g.team2_id
            WHERE g.game_date = ?
            """,
            (game_date,),
        ).fetchall()
        
        if games:
            daily.games = len(games)
            for game in games:
                picks = _generate_game_picks(
                    conn, game_date, game["team1"], game["team2"], config, coverage_stats
                )
                all_picks.extend(picks)
    
    # Sort by confidence
    all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
    
    # Limit picks
    daily.picks = all_picks[:config.max_picks_per_day]
    daily.players_with_sportsbook_lines = coverage_stats["sportsbook"]
    daily.players_with_derived_lines = coverage_stats["derived"]
    
    return daily


def run_backtest_v13_general(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV13General] = None,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> BacktestResultV13General:
    """Run comprehensive backtest for Model V13 General."""
    if config is None:
        config = ModelConfigV13General()
    
    db = Db(Path(db_path))
    result = BacktestResultV13General(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL V13 GENERAL BACKTEST: {start_date} to {end_date}")
        print(f"{'='*70}")
        print(f"Prop types: {config.prop_types}")
        print()
    
    with db.connect() as conn:
        dates = conn.execute(
            """
            SELECT DISTINCT game_date
            FROM games
            WHERE game_date >= ? AND game_date <= ?
            ORDER BY game_date
            """,
            (start_date, end_date),
        ).fetchall()
        
        if verbose:
            print(f"Found {len(dates)} game days to test")
        
        for date_row in dates:
            game_date = date_row["game_date"]
            result.days_tested += 1
            
            games = conn.execute(
                "SELECT COUNT(*) as cnt FROM games WHERE game_date = ?",
                (game_date,),
            ).fetchone()
            result.total_games += games["cnt"] if games else 0
            
            # Generate picks
            coverage_stats = {"sportsbook": 0, "derived": 0}
            all_picks = []
            
            game_rows = conn.execute(
                """
                SELECT g.id, t1.name as team1, t2.name as team2
                FROM games g
                JOIN teams t1 ON t1.id = g.team1_id
                JOIN teams t2 ON t2.id = g.team2_id
                WHERE g.game_date = ?
                """,
                (game_date,),
            ).fetchall()
            
            for game in game_rows:
                picks = _generate_game_picks(
                    conn, game_date, game["team1"], game["team2"], config, coverage_stats
                )
                all_picks.extend(picks)
            
            # Grade picks
            daily_hits = 0
            daily_total = 0
            
            for pick in all_picks[:config.max_picks_per_day]:
                actual = conn.execute(
                    """
                    SELECT bp.pts, bp.reb, bp.ast
                    FROM boxscore_player bp
                    JOIN games g ON g.id = bp.game_id
                    WHERE bp.player_id = ?
                      AND g.game_date = ?
                      AND bp.minutes > 0
                    """,
                    (pick.player_id, game_date),
                ).fetchone()
                
                if not actual:
                    continue
                
                actual_value = actual[pick.prop_type.lower()]
                if actual_value is None:
                    continue
                
                pick.actual_value = actual_value
                pick.margin = actual_value - pick.line
                
                if pick.direction == "OVER":
                    pick.hit = actual_value > pick.line
                else:
                    pick.hit = actual_value < pick.line
                
                # Update counters
                result.total_picks += 1
                daily_total += 1
                
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
                
                # By direction
                if pick.direction == "OVER":
                    result.over_picks += 1
                    if pick.hit:
                        result.over_hits += 1
                else:
                    result.under_picks += 1
                    if pick.hit:
                        result.under_hits += 1
                
                # By prop type
                if pick.prop_type == "PTS":
                    result.pts_picks += 1
                    if pick.hit:
                        result.pts_hits += 1
                elif pick.prop_type == "REB":
                    result.reb_picks += 1
                    if pick.hit:
                        result.reb_hits += 1
                elif pick.prop_type == "AST":
                    result.ast_picks += 1
                    if pick.hit:
                        result.ast_hits += 1
                
                # By pattern
                if pick.pattern == "cold_bounce":
                    result.cold_bounce_picks += 1
                    if pick.hit:
                        result.cold_bounce_hits += 1
                elif pick.pattern == "hot_sustained":
                    result.hot_sustained_picks += 1
                    if pick.hit:
                        result.hot_sustained_hits += 1
                elif pick.pattern == "usage_boost":
                    result.usage_boost_picks += 1
                    if pick.hit:
                        result.usage_boost_hits += 1
                
                result.all_picks.append(pick)
            
            if daily_total > 0:
                daily_rate = daily_hits / daily_total * 100
                result.daily_results.append({
                    "date": game_date,
                    "picks": daily_total,
                    "hits": daily_hits,
                    "rate": daily_rate,
                })
                
                if verbose:
                    print(f"  {game_date}: {daily_hits}/{daily_total} ({daily_rate:.1f}%)")
    
    if verbose:
        print()
        print(result.summary())
    
    return result


# ============================================================================
# CLI Integration
# ============================================================================

def main():
    """Command-line interface for Model V13 General."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V13 General - NBA Props")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_v13_general(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_v13_general(args.date)
        print(picks.summary())
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_v13_general(today)
        print(picks.summary())


if __name__ == "__main__":
    main()
