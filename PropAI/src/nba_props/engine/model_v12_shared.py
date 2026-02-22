"""
Model V12 Shared Components
============================

Shared data classes, utilities, and configuration used by both the 
Model V12 General and Model V12 Under systems.

This module provides:
- Common data classes (PlayerStats, GameContext, etc.)
- Sportsbook line fetching with proper fallback
- Defense vs position data access
- Injury checking utilities
- Player name normalization
- Edge calculation utilities

KEY ARCHITECTURAL DECISIONS:
----------------------------
1. SPORTSBOOK LINES: Use when available, fall back to derived with tracking
   - Unlike V10 which REQUIRES lines, we use them when available
   - Track whether pick used real or derived line for honest reporting
   - Apply conservative adjustment (+5%) to derived lines

2. PROJECTION METHODOLOGY:
   - Base projection from weighted historical averages
   - Defense adjustments from DVP data
   - Usage redistribution when teammates are injured

3. EDGE CALCULATION:
   - Always calculate edge vs the line being used
   - Track edge type (vs_sportsbook vs vs_derived)

Author: PropAI Development Team - Model V12
Created: February 2026
Version: 12.0
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
# Constants
# ============================================================================

# Defense rating thresholds
ELITE_DEFENSE_RANK = 5      # Top 5 = elite defense
GOOD_DEFENSE_RANK = 10      # Top 10 = good defense  
AVERAGE_DEFENSE_RANK = 20   # Top 20 = average
WEAK_DEFENSE_RANK = 25      # Bottom 5 = weak defense

# Minimum data requirements
MIN_GAMES_REQUIRED = 8       # Need 8+ game history (slightly lower for more coverage)
MIN_MINUTES_FILTER = 5       # Filter garbage time games
MIN_AVG_MINUTES = 22.0       # Minimum average minutes for consideration
MAX_GAMES_LOOKBACK = 20      # Use last 20 games

# Edge thresholds
MIN_EDGE_OVER = 7.0          # Minimum edge for OVER picks
MIN_EDGE_UNDER = 5.0         # Lower bar for UNDER (historically better)
MIN_EDGE_PREMIUM = 12.0      # Premium tier needs higher edge

# Derived line adjustment (sportsbook lines typically 5% higher than averages)
DERIVED_LINE_ADJUSTMENT = 1.05


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelV12Config:
    """
    Model V12 Configuration - Shared between General and Under models.
    
    KEY INSIGHT: This config allows using sportsbook lines when available
    but falls back to derived lines with appropriate tracking. This gives
    us more picks while maintaining honest reporting.
    """
    # === VERSION INFO ===
    model_name: str = "Model V12"
    model_version: str = "12.0"
    
    # === LINE STRATEGY ===
    prefer_sportsbook_lines: bool = True  # Use sportsbook lines when available
    derived_line_adjustment: float = 1.05  # Adjust derived lines up by 5%
    track_line_source: bool = True         # Always track where line came from
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 8
    min_minutes_filter: int = 5
    min_avg_minutes: float = 22.0
    max_games_lookback: int = 20
    
    # === PROJECTION WEIGHTS ===
    weight_l3: float = 0.10                # Very recent (momentum)
    weight_l5: float = 0.20                # Recent form
    weight_l10: float = 0.30               # Primary baseline
    weight_l15: float = 0.20               # Extended baseline
    weight_season: float = 0.20            # Full season stability
    
    # === PATTERN THRESHOLDS ===
    cold_deviation_threshold: float = -18.0   # L5 is 18%+ below L15 (for cold bounce)
    hot_deviation_threshold: float = 25.0     # L5 is 25%+ above L15 (for hot sustained)
    cold_streak_threshold: float = -12.0      # L5 is 12%+ below season (for under)
    
    # === EDGE REQUIREMENTS ===
    min_edge_over: float = 7.0
    min_edge_under: float = 5.0
    min_edge_premium: float = 12.0
    
    # === DEFENSE THRESHOLDS ===
    elite_defense_rank: int = 5
    good_defense_rank: int = 10
    weak_defense_rank: int = 25
    
    # === DEFENSE ADJUSTMENTS ===
    elite_defense_adj: float = 0.88         # -12% vs elite defense
    good_defense_adj: float = 0.94          # -6% vs good defense
    neutral_defense_adj: float = 1.00       # No change
    weak_defense_adj: float = 1.07          # +7% vs weak defense
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence: float = 85.0
    high_confidence: float = 75.0
    
    # === PICK LIMITS ===
    max_picks_per_player: int = 1           # Focus on best pick per player
    max_picks_per_game: int = 4
    max_picks_per_day: int = 20
    
    # === PROP TYPES ===
    include_pts: bool = True
    include_reb: bool = True
    include_ast: bool = False               # AST excluded by default (~54% hit rate)
    ast_min_avg: float = 6.0                # Only include AST for high-volume playmakers
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlayerStatsV12:
    """Comprehensive player statistics for Model V12."""
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
    
    # Deviations (for pattern detection)
    deviations_l15: Dict[str, float] = field(default_factory=dict)  # L5 vs L15
    deviations_season: Dict[str, float] = field(default_factory=dict)  # L5 vs Season
    
    # Last game values
    last_game: Dict[str, float] = field(default_factory=dict)
    
    # Standard deviations (for consistency)
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Recent game values (for pattern analysis)
    recent_games: Dict[str, List[float]] = field(default_factory=dict)
    
    # Average minutes
    avg_minutes: float = 0.0
    
    def get_projection(self, prop_type: str, config: ModelV12Config) -> float:
        """Calculate weighted projection for a prop type."""
        pt = prop_type.lower()
        
        l3_val = self.l3.get(pt, 0)
        l5_val = self.l5.get(pt, 0)
        l10_val = self.l10.get(pt, 0)
        l15_val = self.l15.get(pt, 0)
        season_val = self.season.get(pt, 0)
        
        total_weight = (
            config.weight_l3 + config.weight_l5 + 
            config.weight_l10 + config.weight_l15 + config.weight_season
        )
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
        """Get coefficient of variation (std/mean) for consistency measure."""
        pt = prop_type.lower()
        mean = self.l10.get(pt, 0)
        std = self.stds.get(pt, 0)
        if mean <= 0:
            return 1.0
        return std / mean
    
    def is_cold(self, prop_type: str, threshold: float = -15.0) -> bool:
        """Check if player is in a cold streak for this prop."""
        return self.deviations_l15.get(prop_type.lower(), 0) <= threshold
    
    def is_hot(self, prop_type: str, threshold: float = 20.0) -> bool:
        """Check if player is in a hot streak for this prop."""
        return self.deviations_l15.get(prop_type.lower(), 0) >= threshold


@dataclass
class DefenseContextV12:
    """Defense vs position context for opponent."""
    team_abbrev: str
    position: str
    data_available: bool = False
    
    # Ranks (1 = best defense, 30 = worst)
    pts_rank: int = 15
    reb_rank: int = 15
    ast_rank: int = 15
    
    # Ratings (elite, good, average, weak)
    pts_rating: str = "average"
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
    
    def is_elite(self, prop_type: str) -> bool:
        """Check if defense is elite for this prop."""
        return self.get_rating(prop_type.lower()) == "elite"
    
    def is_weak(self, prop_type: str) -> bool:
        """Check if defense is weak for this prop."""
        return self.get_rating(prop_type.lower()) == "weak"


@dataclass
class LineInfo:
    """Information about a betting line."""
    line: float
    source: str  # "sportsbook" or "derived"
    book: str    # Which sportsbook (if sportsbook line)
    odds_american: Optional[int] = None
    
    @property
    def is_sportsbook(self) -> bool:
        return self.source == "sportsbook"
    
    @property
    def is_derived(self) -> bool:
        return self.source == "derived"


@dataclass
class PropPickV12:
    """A pick generated by Model V12."""
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
    book: str          # Which sportsbook or "derived"
    
    # Projection
    projection: float
    projection_std: float
    
    # Edge calculation
    edge: float         # Edge vs the line being used
    
    # Pattern and confidence
    pattern: str        # cold_bounce, hot_sustained, cold_streak, elite_defense, etc.
    confidence_tier: str  # PREMIUM, HIGH, STANDARD
    confidence_score: float
    
    # Defense context
    defense_rank: int
    defense_rating: str
    
    # Supporting data
    l3_avg: float
    l5_avg: float
    l10_avg: float
    l15_avg: float
    season_avg: float
    
    # Analysis
    reasons: List[str] = field(default_factory=list)
    
    # Model source
    model: str = "V12"  # "V12_GENERAL" or "V12_UNDER"
    
    # Results (filled after game)
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    margin: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for display/storage."""
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
            "edge": f"{self.edge:.1f}%",
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
            "model": self.model,
            "actual": self.actual_value,
            "hit": self.hit,
            "margin": round(self.margin, 1) if self.margin is not None else None,
        }


@dataclass
class DailyPicksV12:
    """All picks for a day from Model V12."""
    date: str
    games: int
    picks: List[PropPickV12] = field(default_factory=list)
    
    # Coverage stats
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def premium_picks(self) -> List[PropPickV12]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV12]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    @property
    def over_picks(self) -> List[PropPickV12]:
        return [p for p in self.picks if p.direction == "OVER"]
    
    @property
    def under_picks(self) -> List[PropPickV12]:
        return [p for p in self.picks if p.direction == "UNDER"]
    
    @property
    def sportsbook_picks(self) -> List[PropPickV12]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV12]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    def summary(self) -> str:
        """Generate text summary."""
        sportsbook_count = len(self.sportsbook_picks)
        derived_count = len(self.derived_picks)
        
        lines = [
            f"{'='*70}",
            f"MODEL V12 PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games}",
            f"Total Picks: {self.total_picks}",
            f"  - OVER:  {len(self.over_picks)}",
            f"  - UNDER: {len(self.under_picks)}",
            f"Line Sources:",
            f"  - Sportsbook: {sportsbook_count}",
            f"  - Derived:    {derived_count}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "STANDARD"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    direction_emoji = "📈" if p.direction == "OVER" else "📉"
                    line_marker = "🎰" if p.line_source == "sportsbook" else "📊"
                    lines.append(
                        f"  {direction_emoji}{line_marker} {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.prop_type} {p.direction} {p.line:.1f}"
                    )
                    lines.append(
                        f"      Proj: {p.projection:.1f} | Edge: {p.edge:.1f}% | "
                        f"Pattern: {p.pattern} | Def: {p.defense_rating}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV12:
    """Comprehensive backtest results for Model V12."""
    start_date: str
    end_date: str
    model_name: str
    
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
    pattern_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Coverage stats
    days_tested: int = 0
    total_games: int = 0
    
    # All picks for detailed analysis
    all_picks: List[PropPickV12] = field(default_factory=list)
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
            f"MODEL V12 BACKTEST RESULTS - {self.model_name}",
            f"{'='*70}",
            f"Period: {self.start_date} to {self.end_date}",
            f"Days: {self.days_tested} | Games: {self.total_games}",
            f"",
            f"OVERALL: {self.hit_rate:.1f}% ({self.hits}/{self.total_picks})",
            f"",
            f"BY LINE SOURCE:",
            f"  Sportsbook: {self.sportsbook_hits}/{self.sportsbook_picks} ({self.sportsbook_rate:.1f}%)" if self.sportsbook_picks else "  Sportsbook: N/A",
            f"  Derived:    {self.derived_hits}/{self.derived_picks} ({self.derived_rate:.1f}%)" if self.derived_picks else "  Derived: N/A",
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
            f"  AST: {self.ast_hits}/{self.ast_picks} ({self.ast_hits/self.ast_picks*100:.1f}%)" if self.ast_picks else "  AST: N/A",
        ]
        
        # Pattern breakdown
        if self.pattern_stats:
            lines.append("")
            lines.append("BY PATTERN:")
            for pattern, stats in sorted(self.pattern_stats.items()):
                picks = stats.get("picks", 0)
                hits = stats.get("hits", 0)
                rate = hits / picks * 100 if picks > 0 else 0
                lines.append(f"  {pattern}: {hits}/{picks} ({rate:.1f}%)")
        
        lines.append(f"{'='*70}")
        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_name_for_matching(name: str) -> str:
    """Normalize a player name for matching."""
    # Remove accents
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    
    # Remove suffixes
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        if ascii_name.lower().endswith(suffix):
            ascii_name = ascii_name[:-len(suffix)]
    
    return ascii_name.lower().strip()


def get_injured_players(conn: sqlite3.Connection, game_date: str) -> Set[int]:
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


def get_line_info(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    l10_avg: float,
    config: ModelV12Config,
) -> LineInfo:
    """
    Get betting line with sportsbook preference and proper fallback.
    
    This is a key function - it tries to get sportsbook lines first,
    but falls back to derived lines with tracking.
    """
    # Try sportsbook line by player_id
    if config.prefer_sportsbook_lines and player_id:
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
                odds_american=row["odds_american"],
            )
    
    # Try sportsbook line by name match
    if config.prefer_sportsbook_lines:
        rows = conn.execute(
            """
            SELECT sl.line, sl.book, sl.odds_american, p.name
            FROM sportsbook_lines sl
            JOIN players p ON p.id = sl.player_id
            WHERE sl.prop_type = ? AND sl.as_of_date = ?
            """,
            (prop_type.upper(), game_date)
        ).fetchall()
        
        norm_name = normalize_name_for_matching(player_name)
        for row in rows:
            if normalize_name_for_matching(row["name"]) == norm_name:
                return LineInfo(
                    line=row["line"],
                    source="sportsbook",
                    book=row["book"] or "unknown",
                    odds_american=row["odds_american"],
                )
    
    # Fall back to derived line
    derived_line = l10_avg * config.derived_line_adjustment
    return LineInfo(
        line=round(derived_line, 1),
        source="derived",
        book="derived",
    )


def get_defense_context(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
    config: ModelV12Config,
) -> DefenseContextV12:
    """Get defense vs position context for an opponent team."""
    context = DefenseContextV12(
        team_abbrev=team_abbrev,
        position=position,
    )
    
    # Map position to DVP position
    pos_map = {
        'G': 'PG', 'PG': 'PG', 'SG': 'SG',
        'F': 'SF', 'SF': 'SF', 'PF': 'PF',
        'C': 'C', 'F-C': 'PF', 'G-F': 'SF'
    }
    dvp_position = pos_map.get(position.upper() if position else 'G', 'SF')
    
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
        
        # Determine ratings based on rank
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


def load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    config: ModelV12Config,
) -> Optional[PlayerStatsV12]:
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
    player_stats = PlayerStatsV12(
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
        
        # Calculate deviations
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


def calculate_edge(projection: float, line: float, direction: str) -> float:
    """
    Calculate edge percentage vs a line.
    
    For OVER: (projection - line) / line * 100
    For UNDER: (line - projection) / line * 100
    """
    if line <= 0:
        return 0.0
    
    if direction.upper() == "OVER":
        return (projection - line) / line * 100
    else:
        return (line - projection) / line * 100


def apply_defense_adjustment(
    base_projection: float,
    defense_rating: str,
    config: ModelV12Config,
) -> float:
    """Apply defense adjustment to projection."""
    if defense_rating == "elite":
        return base_projection * config.elite_defense_adj
    elif defense_rating == "good":
        return base_projection * config.good_defense_adj
    elif defense_rating == "weak":
        return base_projection * config.weak_defense_adj
    else:
        return base_projection * config.neutral_defense_adj


def determine_confidence_tier(
    confidence_score: float,
    edge: float,
    config: ModelV12Config,
) -> str:
    """Determine confidence tier based on score and edge."""
    if confidence_score >= config.premium_confidence and edge >= config.min_edge_premium:
        return "PREMIUM"
    elif confidence_score >= config.high_confidence:
        return "HIGH"
    else:
        return "STANDARD"


def grade_pick(pick: PropPickV12, actual_value: float) -> PropPickV12:
    """Grade a pick based on actual results."""
    pick.actual_value = actual_value
    pick.margin = actual_value - pick.line
    
    if pick.direction == "OVER":
        pick.hit = actual_value > pick.line
    else:
        pick.hit = actual_value < pick.line
    
    return pick
