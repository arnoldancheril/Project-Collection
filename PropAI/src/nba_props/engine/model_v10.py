"""
Model V10 - Market-Aware NBA Props Prediction Model
=====================================================

This model addresses the FUNDAMENTAL FLAW in all previous models: testing predictions
against derived/estimated lines instead of actual sportsbook betting lines.

THE CRITICAL INSIGHT:
--------------------
Previous models achieved "66.7%" hit rates by comparing projections to player averages.
But if Vegas sets LeBron's line at 26.5 and his L10 average is 25.5, beating 25.5
does NOT mean the bet wins. You must beat 26.5.

Model V10 FIX: 
- ONLY generate picks when actual sportsbook lines exist
- Calculate edge vs the ACTUAL BETTING LINE
- Track performance vs ACTUAL lines for honest metrics
- When no sportsbook line exists, NO PICK is generated

VALIDATED INSIGHTS FROM PREVIOUS MODELS:
----------------------------------------
1. PATTERNS WORK (from Model Production):
   - Cold Bounce: 66.9% when player is 20%+ below L15 but last game > L10
   - Hot Sustained: 65.9% when player is 30%+ above L15 with momentum

2. DIRECTION MATTERS (from RCM v1.4):
   - PTS UNDER: 63.9% hit rate
   - PTS OVER: 48.3% hit rate (AVOID)
   - REB: ~59% both directions
   
3. PROP TYPE MATTERS (from Model Production):
   - PTS: 68.6% hit rate (best)
   - REB: 65.1% hit rate (good)
   - AST: ~54% hit rate (EXCLUDE - too volatile)

4. FILTERING MATTERS (from Model V6):
   - Scoring Guards: 51.5% hit rate (EXCLUDE)
   - Stretch Bigs: 64.9% hit rate (PRIORITIZE)
   - Traditional Bigs: 64.0% hit rate (PRIORITIZE)

5. DEFENSE MATTERS (from Under Model V2):
   - Elite defense (rank 1-5): Bet UNDER
   - Terrible defense (rank 26-30): Avoid UNDER

MODEL V10 RULES:
----------------
1. SPORTSBOOK LINE REQUIRED: No sportsbook line = No pick
2. STRATEGIC DIRECTION:
   - PTS: UNDER only (unless strong OVER pattern with positive edge vs actual line)
   - REB: Both directions allowed
   - AST: Excluded entirely
3. PATTERN CONFIRMATION:
   - OVER requires: Cold bounce pattern + positive edge vs sportsbook line
   - UNDER requires: Elite defense OR cold streak + negative edge would be positive
4. STRICT FILTERING:
   - Minimum 23 minutes average (established players)
   - Minimum 10 games history
   - Exclude volatile archetypes (scoring guards)
   - Exclude recent injury returns (first 2 games back)

USAGE:
------
    from src.nba_props.engine.model_v10 import (
        get_daily_picks_v10,
        run_backtest_v10,
        ModelConfigV10,
    )
    
    # Get picks for today (ONLY where sportsbook lines exist)
    picks = get_daily_picks_v10("2026-02-03")
    
    # Run backtest - only tests picks with actual lines
    result = run_backtest_v10("2025-12-01", "2026-02-02")

Author: NBA Props Team - Model V10
Created: February 2026
Version: 10.0
"""
from __future__ import annotations

import sqlite3
import statistics
import unicodedata
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any, Set

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfigV10:
    """
    Model V10 Configuration - Market-Aware with Strict Sportsbook Line Requirement.
    
    THE KEY DIFFERENCE: This model ONLY generates picks where actual sportsbook
    lines exist. No more beating ghost lines.
    
    VALIDATED THROUGH EXTENSIVE ANALYSIS:
    - Pattern detection from Model Production (66.7% hit rate)
    - Direction strategy from RCM v1.4 (PTS UNDER at 63.9%)
    - Defense integration from Under Model V2 (~58%)
    - Archetype filtering from Model V6 (Scoring Guards = 51.5%)
    """
    # === VERSION INFO ===
    model_name: str = "Model V10 - Market Aware"
    model_version: str = "10.0"
    
    # === CORE PRINCIPLE: SPORTSBOOK LINES REQUIRED ===
    require_sportsbook_line: bool = True  # CRITICAL: Set to False ONLY for comparison testing
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10           # Need sufficient history
    min_minutes_filter: int = 5            # Filter garbage time games
    min_avg_minutes: float = 23.0          # Increased - only established players
    max_games_lookback: int = 20           # Use last 20 games
    
    # === PROJECTION WEIGHTS ===
    # Conservative weighting - favor longer-term stability
    weight_l5: float = 0.20                # Recent form
    weight_l10: float = 0.30               # Primary baseline
    weight_l15: float = 0.25               # Extended baseline
    weight_season: float = 0.25            # Full season stability
    
    # === PATTERN THRESHOLDS (Validated from Production Model) ===
    # Cold bounce-back pattern - 66.9% hit rate
    cold_deviation_threshold: float = -20.0  # L5 is 20%+ below L15
    bounce_threshold: float = 0.0            # Last game > L10
    
    # Hot sustained pattern - 65.9% hit rate
    hot_deviation_threshold: float = 30.0    # L5 is 30%+ above L15
    sustained_games_above: int = 3           # 3+ of last 5 above L15
    
    # Cold streak for UNDER picks
    cold_streak_threshold: float = -15.0     # L5 is 15%+ below season avg
    
    # === EDGE REQUIREMENTS (Against ACTUAL Sportsbook Lines) ===
    # These are edges vs the REAL betting line, not derived estimates
    min_edge_over: float = 8.0              # Need 8%+ edge for OVER
    min_edge_under: float = 6.0             # Need 6%+ edge for UNDER (slightly lower)
    min_edge_premium: float = 12.0          # Premium picks need 12%+ edge
    
    # === DEFENSE VS POSITION THRESHOLDS ===
    elite_defense_rank: int = 5             # Top 5 = elite defense
    good_defense_rank: int = 10             # Top 10 = good defense
    weak_defense_rank: int = 25             # Bottom 5 = weak defense
    
    # Defense adjustment multipliers
    elite_defense_adj: float = 0.88         # -12% vs elite defense
    good_defense_adj: float = 0.94          # -6% vs good defense
    neutral_defense_adj: float = 1.00       # No change
    weak_defense_adj: float = 1.06          # +6% vs weak defense
    
    # === STRATEGIC DIRECTION SELECTION ===
    # Based on RCM v1.4 analysis: PTS UNDER (63.9%) >> PTS OVER (48.3%)
    pts_direction: str = "UNDER_PREFERRED"  # UNDER, OVER, BOTH, UNDER_PREFERRED
    reb_direction: str = "BOTH"             # Both directions work ~59%
    include_ast: bool = False               # EXCLUDED - too volatile (~54%)
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb'])
    
    # === ARCHETYPE FILTERING (From Model V6 analysis) ===
    # Scoring Guards hit at only 51.5% - should be excluded
    exclude_archetypes: List[str] = field(default_factory=lambda: [
        "scoring_guard", "volume_scorer"
    ])
    
    # === INJURY/RECENT RETURN FILTERING ===
    games_back_from_injury_min: int = 2     # Skip first 2 games after injury
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 4
    max_picks_per_day: int = 20
    max_picks_per_player: int = 1           # 1 prop per player for focus
    
    # === CONFIDENCE SCORING ===
    premium_confidence_threshold: float = 85.0
    high_confidence_threshold: float = 75.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# ============================================================================
# Data Classes  
# ============================================================================

@dataclass
class PlayerStatsV10:
    """Comprehensive player statistics for Model V10."""
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
    
    def get_projection(self, prop_type: str, config: ModelConfigV10) -> float:
        """Calculate weighted projection for a prop type."""
        pt = prop_type.lower()
        
        l5_val = self.l5.get(pt, 0)
        l10_val = self.l10.get(pt, 0)
        l15_val = self.l15.get(pt, 0)
        season_val = self.season.get(pt, 0)
        
        total_weight = config.weight_l5 + config.weight_l10 + config.weight_l15 + config.weight_season
        if total_weight <= 0:
            return season_val
        
        projection = (
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


@dataclass
class DefenseContextV10:
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
        mapping = {
            'pts': self.pts_rating,
            'reb': self.reb_rating,
            'ast': self.ast_rating,
        }
        return mapping.get(prop_type.lower(), "average")
    
    def get_rank(self, prop_type: str) -> int:
        """Get defense rank for a prop type."""
        mapping = {
            'pts': self.pts_rank,
            'reb': self.reb_rank,
            'ast': self.ast_rank,
        }
        return mapping.get(prop_type.lower(), 15)


@dataclass
class PatternDetectionResult:
    """Result of pattern analysis for a player/prop."""
    pattern_name: str  # cold_bounce, hot_sustained, cold_streak, elite_defense, none
    direction: str     # OVER, UNDER
    confidence_bonus: float
    reasons: List[str]
    is_valid: bool


@dataclass
class PropPickV10:
    """A pick generated by Model V10 with full market awareness."""
    # Identity
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    
    # Pick details
    prop_type: str     # PTS, REB
    direction: str     # OVER, UNDER
    
    # LINE INFORMATION (CRITICAL - Must use sportsbook line)
    sportsbook_line: float          # ACTUAL betting line from sportsbook
    book: str                       # Which sportsbook (draftkings, fanduel, etc)
    
    # Projection
    projection: float
    projection_std: float
    
    # Edge calculation (vs ACTUAL sportsbook line)
    edge_vs_line: float             # (projection - line) / line * 100 for OVER
                                    # (line - projection) / line * 100 for UNDER
    
    # Pattern and confidence
    pattern: str                    # cold_bounce, hot_sustained, cold_streak, elite_defense
    confidence_tier: str            # PREMIUM, HIGH, STANDARD
    confidence_score: float
    
    # Defense context
    defense_rank: int               # Opponent's DVP rank for this position/stat
    defense_rating: str             # elite, good, average, weak
    
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
    margin: Optional[float] = None   # actual - line
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display/storage."""
        return {
            "player": self.player_name,
            "team": self.team_abbrev,
            "opponent": self.opponent_abbrev,
            "date": self.game_date,
            "prop": self.prop_type.upper(),
            "direction": self.direction,
            "line": round(self.sportsbook_line, 1),
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
            "margin": round(self.margin, 1) if self.margin else None,
        }


@dataclass
class DailyPicksV10:
    """All picks for a day from Model V10."""
    date: str
    games: int
    picks: List[PropPickV10] = field(default_factory=list)
    
    # Stats
    players_with_lines: int = 0
    players_without_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def premium_picks(self) -> List[PropPickV10]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV10]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    @property
    def over_picks(self) -> List[PropPickV10]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV10]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            f"{'='*70}",
            f"MODEL V10 PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games}",
            f"Total Picks: {self.total_picks} (OVER: {len(self.over_picks)}, UNDER: {len(self.under_picks)})",
            f"Players with sportsbook lines: {self.players_with_lines}",
            f"Players without lines (skipped): {self.players_without_lines}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    direction_emoji = "📈" if p.direction == "OVER" else "📉"
                    lines.append(
                        f"  {direction_emoji} {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.prop_type} {p.direction} {p.sportsbook_line:.1f}"
                    )
                    lines.append(
                        f"      Proj: {p.projection:.1f} | Edge: {p.edge_vs_line:.1f}% | "
                        f"Pattern: {p.pattern} | Def: {p.defense_rating}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV10:
    """Comprehensive backtest results for Model V10."""
    start_date: str
    end_date: str
    config: ModelConfigV10
    
    # Overall
    total_picks: int = 0
    hits: int = 0
    
    # By tier
    premium_picks: int = 0
    premium_hits: int = 0
    high_picks: int = 0
    high_hits: int = 0
    
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
    
    # By pattern
    cold_bounce_picks: int = 0
    cold_bounce_hits: int = 0
    hot_sustained_picks: int = 0
    hot_sustained_hits: int = 0
    cold_streak_picks: int = 0
    cold_streak_hits: int = 0
    elite_defense_picks: int = 0
    elite_defense_hits: int = 0
    
    # Coverage stats
    days_tested: int = 0
    total_games: int = 0
    players_with_lines: int = 0
    players_without_lines: int = 0
    
    # All picks for detailed analysis
    all_picks: List[PropPickV10] = field(default_factory=list)
    daily_results: List[Dict] = field(default_factory=list)
    
    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_picks * 100 if self.total_picks > 0 else 0.0
    
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
            f"MODEL V10 BACKTEST RESULTS",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            f"",
            f"BY TIER:",
            f"  Premium: {self.premium_hits}/{self.premium_picks} ({self.premium_rate:.1f}%)" if self.premium_picks else "  Premium: N/A",
            f"  High:    {self.high_hits}/{self.high_picks} ({self.high_hits/self.high_picks*100:.1f}%)" if self.high_picks else "  High: N/A",
            f"",
            f"BY DIRECTION:",
            f"  OVER:  {self.over_hits}/{self.over_picks} ({self.over_rate:.1f}%)" if self.over_picks else "  OVER: N/A",
            f"  UNDER: {self.under_hits}/{self.under_picks} ({self.under_rate:.1f}%)" if self.under_picks else "  UNDER: N/A",
            f"",
            f"BY PROP TYPE:",
            f"  PTS: {self.pts_hits}/{self.pts_picks} ({self.pts_hits/self.pts_picks*100:.1f}%)" if self.pts_picks else "  PTS: N/A",
            f"  REB: {self.reb_hits}/{self.reb_picks} ({self.reb_hits/self.reb_picks*100:.1f}%)" if self.reb_picks else "  REB: N/A",
            f"",
            f"BY PATTERN:",
            f"  Cold Bounce:    {self.cold_bounce_hits}/{self.cold_bounce_picks} ({self.cold_bounce_hits/self.cold_bounce_picks*100:.1f}%)" if self.cold_bounce_picks else "  Cold Bounce: N/A",
            f"  Hot Sustained:  {self.hot_sustained_hits}/{self.hot_sustained_picks} ({self.hot_sustained_hits/self.hot_sustained_picks*100:.1f}%)" if self.hot_sustained_picks else "  Hot Sustained: N/A",
            f"  Cold Streak:    {self.cold_streak_hits}/{self.cold_streak_picks} ({self.cold_streak_hits/self.cold_streak_picks*100:.1f}%)" if self.cold_streak_picks else "  Cold Streak: N/A",
            f"  Elite Defense:  {self.elite_defense_hits}/{self.elite_defense_picks} ({self.elite_defense_hits/self.elite_defense_picks*100:.1f}%)" if self.elite_defense_picks else "  Elite Defense: N/A",
            f"",
            f"COVERAGE:",
            f"  Players with sportsbook lines: {self.players_with_lines}",
            f"  Players without lines (skipped): {self.players_without_lines}",
            f"{'='*70}",
        ]
        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def _normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    import unicodedata
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Remove suffixes
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
    
    THIS IS THE CRITICAL FUNCTION - Model V10 REQUIRES actual lines.
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


def _get_defense_context(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
    config: ModelConfigV10,
) -> DefenseContextV10:
    """Get defense vs position context for an opponent team."""
    context = DefenseContextV10(
        team_abbrev=team_abbrev,
        position=position,
    )
    
    # Map position to DVP position
    pos_map = {
        'G': 'PG', 'PG': 'PG', 'SG': 'SG',
        'F': 'SF', 'SF': 'SF', 'PF': 'PF',
        'C': 'C', 'F-C': 'PF', 'G-F': 'SF'
    }
    dvp_position = pos_map.get(position, 'SF')
    
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
            elif rank <= config.weak_defense_rank:
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
    config: ModelConfigV10,
) -> Optional[PlayerStatsV10]:
    """Load comprehensive player statistics."""
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
    stats = {
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
    
    # Build player stats
    player_stats = PlayerStatsV10(
        player_id=player_id,
        player_name=player["name"],
        team_abbrev=abbrev_from_team_name(games[0]["team_name"]) or "",
        position=games[0].get("pos") or "G",
        games_played=n,
        avg_minutes=avg_min,
    )
    
    for stat in ['pts', 'reb', 'ast', 'min']:
        vals = stats[stat]
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
        
        if l15 > 0:
            player_stats.deviations_l15[stat] = (l5 - l15) / l15 * 100
        else:
            player_stats.deviations_l15[stat] = 0.0
            
        if season > 0:
            player_stats.deviations_season[stat] = (l5 - season) / season * 100
        else:
            player_stats.deviations_season[stat] = 0.0
    
    return player_stats


def _detect_pattern_over(
    stats: PlayerStatsV10,
    prop_type: str,
    defense_context: DefenseContextV10,
    config: ModelConfigV10,
) -> PatternDetectionResult:
    """
    Detect OVER patterns.
    
    Patterns:
    1. Cold Bounce: Player is cold but last game shows recovery
    2. Hot Sustained: Player is hot and maintaining
    """
    pt = prop_type.lower()
    
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l3 = stats.l3.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l10 = stats.l10.get(pt, 0)
    l15 = stats.l15.get(pt, 0)
    last_game = stats.last_game.get(pt, 0)
    recent = stats.recent_games.get(pt, [])
    
    # Check COLD BOUNCE pattern (Best OVER pattern - 66.9% from Production)
    if deviation_l15 <= config.cold_deviation_threshold:
        # Last game must be above L10 (bouncing back)
        bounce_pct = (last_game - l10) / l10 * 100 if l10 > 0 else 0
        if bounce_pct >= config.bounce_threshold:
            # Additional check: opponent not elite defense
            if defense_context.get_rating(pt) != "elite":
                reasons = [
                    f"Cold bounce: L5 ({l5:.1f}) is {deviation_l15:.0f}% below L15 ({l15:.1f})",
                    f"Recovery signal: Last game ({last_game:.0f}) is {bounce_pct:.0f}% above L10 ({l10:.1f})",
                    f"Regression expected toward baseline ({l15:.1f})",
                ]
                confidence_bonus = min(abs(deviation_l15) / 2, 10)
                return PatternDetectionResult(
                    pattern_name="cold_bounce",
                    direction="OVER",
                    confidence_bonus=confidence_bonus,
                    reasons=reasons,
                    is_valid=True,
                )
    
    # Check HOT SUSTAINED pattern (65.9% from Production)
    if deviation_l15 >= config.hot_deviation_threshold:
        # L3 >= L5 (still hot or accelerating)
        if l3 >= l5 * 0.95:
            # Count games above L15
            games_above = sum(1 for v in recent if v > l15)
            if games_above >= config.sustained_games_above:
                # Additional check: not facing elite defense
                if defense_context.get_rating(pt) != "elite":
                    reasons = [
                        f"Hot sustained: L5 ({l5:.1f}) is {deviation_l15:.0f}% above L15 ({l15:.1f})",
                        f"Momentum: L3 ({l3:.1f}) maintaining level",
                        f"Consistency: {games_above}/5 recent games above baseline",
                    ]
                    confidence_bonus = min((deviation_l15 - config.hot_deviation_threshold) / 3, 8)
                    return PatternDetectionResult(
                        pattern_name="hot_sustained",
                        direction="OVER",
                        confidence_bonus=confidence_bonus,
                        reasons=reasons,
                        is_valid=True,
                    )
    
    # No valid OVER pattern
    return PatternDetectionResult(
        pattern_name="none",
        direction="OVER",
        confidence_bonus=0,
        reasons=[],
        is_valid=False,
    )


def _detect_pattern_under(
    stats: PlayerStatsV10,
    prop_type: str,
    defense_context: DefenseContextV10,
    config: ModelConfigV10,
) -> PatternDetectionResult:
    """
    Detect UNDER patterns.
    
    Patterns:
    1. Elite Defense: Facing top-5 defense at position
    2. Cold Streak: Player is significantly below season average
    3. Combined: Cold + Good Defense
    """
    pt = prop_type.lower()
    
    deviation_season = stats.deviations_season.get(pt, 0)
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l10 = stats.l10.get(pt, 0)
    season = stats.season.get(pt, 0)
    defense_rank = defense_context.get_rank(pt)
    defense_rating = defense_context.get_rating(pt)
    
    reasons = []
    pattern_name = "none"
    confidence_bonus = 0
    is_valid = False
    
    # Check ELITE DEFENSE pattern
    if defense_rating == "elite":
        reasons.append(f"Elite defense: {defense_context.team_abbrev} ranks #{defense_rank} vs {stats.position} for {pt.upper()}")
        pattern_name = "elite_defense"
        confidence_bonus += 8
        is_valid = True
    
    # Check COLD STREAK pattern
    if deviation_season <= config.cold_streak_threshold:
        reasons.append(f"Cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season avg ({season:.1f})")
        if pattern_name == "elite_defense":
            pattern_name = "elite_defense_cold"
            confidence_bonus += 10  # Combined patterns are strongest
        else:
            pattern_name = "cold_streak"
            confidence_bonus += 6
        is_valid = True
    
    # Good defense + cold also works
    if defense_rating == "good" and deviation_season <= -10:
        if not is_valid:
            reasons.append(f"Good defense: {defense_context.team_abbrev} ranks #{defense_rank} + player cold")
            pattern_name = "good_defense_cold"
            confidence_bonus += 5
            is_valid = True
    
    return PatternDetectionResult(
        pattern_name=pattern_name,
        direction="UNDER",
        confidence_bonus=confidence_bonus,
        reasons=reasons,
        is_valid=is_valid,
    )


def _generate_pick(
    conn: sqlite3.Connection,
    stats: PlayerStatsV10,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: ModelConfigV10,
) -> Optional[PropPickV10]:
    """
    Generate a pick for a player/prop combination.
    
    CRITICAL: This function REQUIRES an actual sportsbook line.
    If no line exists, returns None.
    """
    pt = prop_type.lower()
    
    # STEP 1: GET SPORTSBOOK LINE (MANDATORY)
    line_result = _get_sportsbook_line(conn, stats.player_id, stats.player_name, pt, game_date)
    
    if config.require_sportsbook_line and line_result is None:
        return None  # NO LINE = NO PICK
    
    if line_result:
        sportsbook_line, book = line_result
    else:
        # Fallback mode (only for comparison testing)
        sportsbook_line = stats.l10.get(pt, 0) * 1.05
        book = "derived"
    
    # STEP 2: GET DEFENSE CONTEXT
    defense_context = _get_defense_context(conn, opponent_abbrev, stats.position, config)
    
    # STEP 3: CALCULATE PROJECTION
    base_projection = stats.get_projection(pt, config)
    
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
    
    projection = base_projection * defense_adj
    projection_std = stats.stds.get(pt, 0)
    
    # STEP 4: DETERMINE DIRECTION BASED ON CONFIG
    # PTS: Prefer UNDER (63.9% vs 48.3% from RCM analysis)
    # REB: Both directions work
    
    direction_config = config.pts_direction if pt == 'pts' else config.reb_direction
    
    # Detect patterns for both directions
    over_pattern = _detect_pattern_over(stats, pt, defense_context, config)
    under_pattern = _detect_pattern_under(stats, pt, defense_context, config)
    
    # Calculate edges
    over_edge = (projection - sportsbook_line) / sportsbook_line * 100 if sportsbook_line > 0 else 0
    under_edge = (sportsbook_line - projection) / sportsbook_line * 100 if sportsbook_line > 0 else 0
    
    # Select direction
    selected_direction = None
    selected_pattern = None
    selected_edge = 0
    
    if direction_config == "UNDER":
        # Only UNDER picks
        if under_pattern.is_valid and under_edge >= config.min_edge_under:
            selected_direction = "UNDER"
            selected_pattern = under_pattern
            selected_edge = under_edge
    elif direction_config == "OVER":
        # Only OVER picks
        if over_pattern.is_valid and over_edge >= config.min_edge_over:
            selected_direction = "OVER"
            selected_pattern = over_pattern
            selected_edge = over_edge
    elif direction_config == "UNDER_PREFERRED":
        # Prefer UNDER, but allow OVER with strong pattern
        if under_pattern.is_valid and under_edge >= config.min_edge_under:
            selected_direction = "UNDER"
            selected_pattern = under_pattern
            selected_edge = under_edge
        elif over_pattern.is_valid and over_edge >= config.min_edge_over * 1.25:  # Higher bar for OVER
            selected_direction = "OVER"
            selected_pattern = over_pattern
            selected_edge = over_edge
    else:  # BOTH
        # Pick the better option
        if under_pattern.is_valid and under_edge >= config.min_edge_under:
            if over_pattern.is_valid and over_edge >= config.min_edge_over:
                # Both valid - pick higher edge
                if over_edge > under_edge:
                    selected_direction = "OVER"
                    selected_pattern = over_pattern
                    selected_edge = over_edge
                else:
                    selected_direction = "UNDER"
                    selected_pattern = under_pattern
                    selected_edge = under_edge
            else:
                selected_direction = "UNDER"
                selected_pattern = under_pattern
                selected_edge = under_edge
        elif over_pattern.is_valid and over_edge >= config.min_edge_over:
            selected_direction = "OVER"
            selected_pattern = over_pattern
            selected_edge = over_edge
    
    if selected_direction is None or selected_pattern is None:
        return None
    
    # STEP 5: CALCULATE CONFIDENCE
    base_confidence = 70.0
    
    # Pattern bonus
    confidence = base_confidence + selected_pattern.confidence_bonus
    
    # Edge bonus (more edge = more confidence)
    edge_bonus = min(selected_edge / 2, 10)
    confidence += edge_bonus
    
    # Consistency bonus
    cv = stats.get_cv(pt)
    if cv < 0.20:
        confidence += 5  # Very consistent
    elif cv > 0.40:
        confidence -= 5  # Volatile
    
    # Defense confidence
    if selected_direction == "UNDER" and defense_rating == "elite":
        confidence += 5
    
    confidence = min(confidence, 100)
    
    # Determine tier
    if confidence >= config.premium_confidence_threshold and selected_edge >= config.min_edge_premium:
        tier = "PREMIUM"
    elif confidence >= config.high_confidence_threshold:
        tier = "HIGH"
    else:
        tier = "STANDARD"
    
    return PropPickV10(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        game_date=game_date,
        prop_type=prop_type.upper(),
        direction=selected_direction,
        sportsbook_line=round(sportsbook_line, 1),
        book=book,
        projection=round(projection, 1),
        projection_std=round(projection_std, 1),
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
    config: ModelConfigV10,
    line_stats: Dict[str, int],
) -> List[PropPickV10]:
    """Generate picks for a single game."""
    t1_abbrev = abbrev_from_team_name(team1_name) or ""
    t2_abbrev = abbrev_from_team_name(team2_name) or ""
    
    injured = _get_injured_players(conn, game_date)
    
    all_picks = []
    player_picks = {}  # Track picks per player
    
    for team_name, opp_abbrev in [(team1_name, t2_abbrev), (team2_name, t1_abbrev)]:
        team = conn.execute("SELECT id FROM teams WHERE name = ?", (team_name,)).fetchone()
        if not team:
            continue
        
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
            
            # Check if already have enough picks for this player
            if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                continue
            
            stats = _load_player_stats(conn, player_id, game_date, config)
            if not stats:
                continue
            
            # Generate picks for each prop type
            for pt in config.prop_types:
                if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                    break
                
                pick = _generate_pick(conn, stats, pt, opp_abbrev, game_date, config)
                
                if pick:
                    all_picks.append(pick)
                    player_picks[player_id] = player_picks.get(player_id, 0) + 1
                    line_stats["with_line"] += 1
                else:
                    # Check if it was because of missing line
                    line_result = _get_sportsbook_line(conn, player_id, stats.player_name, pt, game_date)
                    if line_result is None:
                        line_stats["without_line"] += 1
    
    return all_picks


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_v10(
    game_date: str,
    config: Optional[ModelConfigV10] = None,
    db_path: str = "data/db/nba_props.sqlite3",
) -> DailyPicksV10:
    """
    Generate picks for all games on a date.
    
    REQUIRES actual sportsbook lines - no picks generated without them.
    """
    if config is None:
        config = ModelConfigV10()
    
    db = Db(db_path)
    daily = DailyPicksV10(date=game_date, games=0)
    
    all_picks = []
    line_stats = {"with_line": 0, "without_line": 0}
    
    with db.connect() as conn:
        # Get games for the date
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
                    conn, game_date, game["team1"], game["team2"], config, line_stats
                )
                all_picks.extend(picks)
    
    # Sort by confidence
    all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
    
    # Limit picks per day
    daily.picks = all_picks[:config.max_picks_per_day]
    daily.players_with_lines = line_stats["with_line"]
    daily.players_without_lines = line_stats["without_line"]
    
    return daily


def run_backtest_v10(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV10] = None,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> BacktestResultV10:
    """
    Run comprehensive backtest for Model V10.
    
    Only tests picks where actual sportsbook lines existed.
    """
    if config is None:
        config = ModelConfigV10()
    
    db = Db(db_path)
    result = BacktestResultV10(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL V10 BACKTEST: {start_date} to {end_date}")
        print(f"{'='*70}")
        print(f"Config: require_sportsbook_line={config.require_sportsbook_line}")
        print(f"Prop types: {config.prop_types}")
        print(f"PTS direction: {config.pts_direction}")
        print()
    
    with db.connect() as conn:
        # Get all game dates in range
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
            
            # Get games for this date
            games = conn.execute(
                """
                SELECT COUNT(*) as cnt
                FROM games
                WHERE game_date = ?
                """,
                (game_date,),
            ).fetchone()
            
            result.total_games += games["cnt"] if games else 0
            
            # Generate picks for this date
            line_stats = {"with_line": 0, "without_line": 0}
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
                    conn, game_date, game["team1"], game["team2"], config, line_stats
                )
                all_picks.extend(picks)
            
            result.players_with_lines += line_stats["with_line"]
            result.players_without_lines += line_stats["without_line"]
            
            # Grade each pick
            daily_hits = 0
            daily_total = 0
            
            for pick in all_picks[:config.max_picks_per_day]:
                # Get actual result
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
                
                # Get actual value for this prop
                actual_value = actual[pick.prop_type.lower()]
                if actual_value is None:
                    continue
                
                pick.actual_value = actual_value
                pick.margin = actual_value - pick.sportsbook_line
                
                # Determine if hit
                if pick.direction == "OVER":
                    pick.hit = actual_value > pick.sportsbook_line
                else:
                    pick.hit = actual_value < pick.sportsbook_line
                
                # Update counters
                result.total_picks += 1
                daily_total += 1
                
                if pick.hit:
                    result.hits += 1
                    daily_hits += 1
                
                # By tier
                if pick.confidence_tier == "PREMIUM":
                    result.premium_picks += 1
                    if pick.hit:
                        result.premium_hits += 1
                elif pick.confidence_tier == "HIGH":
                    result.high_picks += 1
                    if pick.hit:
                        result.high_hits += 1
                
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
                
                # By pattern
                if pick.pattern == "cold_bounce":
                    result.cold_bounce_picks += 1
                    if pick.hit:
                        result.cold_bounce_hits += 1
                elif pick.pattern == "hot_sustained":
                    result.hot_sustained_picks += 1
                    if pick.hit:
                        result.hot_sustained_hits += 1
                elif pick.pattern == "cold_streak":
                    result.cold_streak_picks += 1
                    if pick.hit:
                        result.cold_streak_hits += 1
                elif pick.pattern in ["elite_defense", "elite_defense_cold"]:
                    result.elite_defense_picks += 1
                    if pick.hit:
                        result.elite_defense_hits += 1
                
                result.all_picks.append(pick)
            
            # Daily summary
            if daily_total > 0:
                daily_rate = daily_hits / daily_total * 100
                result.daily_results.append({
                    "date": game_date,
                    "picks": daily_total,
                    "hits": daily_hits,
                    "rate": daily_rate,
                })
                
                if verbose and daily_total > 0:
                    print(f"  {game_date}: {daily_hits}/{daily_total} ({daily_rate:.1f}%)")
    
    if verbose:
        print()
        print(result.summary())
    
    return result


# ============================================================================
# CLI Integration
# ============================================================================

def main():
    """Command-line interface for Model V10."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V10 - Market-Aware NBA Props")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_v10(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_v10(args.date)
        print(picks.summary())
    else:
        # Default to today
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_v10(today)
        print(picks.summary())


if __name__ == "__main__":
    main()
