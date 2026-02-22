"""
Model V11 Shared Utilities
===========================

Shared utilities between General Model V11 and Under Model V11.
Provides common functionality for:
- Line fetching (sportsbook + derived fallback with source tracking)
- Edge calculation (honest metrics vs actual lines)
- Pattern detection (cold bounce, hot sustained, cold streak)
- Injury checking
- Player stats loading
- Defense context lookup

CRITICAL DESIGN DECISIONS:
--------------------------
1. LINE HANDLING: Unlike Model V10, we don't require sportsbook lines.
   Instead, we TRACK the source (sportsbook vs derived) for honest metrics.
   This allows picks even when lines aren't available yet (late release).

2. HONEST METRICS: All backtests report metrics separately:
   - Overall hit rate
   - Hit rate with sportsbook lines
   - Hit rate with derived lines
   
3. DERIVED LINE CALCULATION: When no sportsbook line exists:
   - Use L10 average × 1.05 (Vegas typically sets 5% higher)
   - Track this as "derived" for reporting

Author: PropAI Team - Model V11
Created: February 2026
Version: 11.0
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

# Derived line adjustment factor (Vegas typically sets lines 5% above player averages)
DERIVED_LINE_FACTOR = 1.05

# Defense rating thresholds
ELITE_DEFENSE_RANK = 5      # Top 5 = elite defense
GOOD_DEFENSE_RANK = 10      # Top 10 = good defense
WEAK_DEFENSE_RANK = 25      # Bottom 5 = weak defense

# Pattern thresholds (validated from Model Production)
COLD_DEVIATION_THRESHOLD = -20.0    # L5 is 20%+ below L15
HOT_DEVIATION_THRESHOLD = 30.0      # L5 is 30%+ above L15
COLD_STREAK_THRESHOLD = -15.0       # L5 is 15%+ below season avg
BOUNCE_THRESHOLD = 0.0              # Last game > L10 (any amount)
SUSTAINED_GAMES_ABOVE = 3           # 3+ of last 5 above L15

# Data requirements
MIN_GAMES_REQUIRED = 10
MIN_MINUTES_FILTER = 5
MIN_AVG_MINUTES = 23.0
MAX_GAMES_LOOKBACK = 20

# Projection weights
WEIGHT_L5 = 0.20
WEIGHT_L10 = 0.30
WEIGHT_L15 = 0.25
WEIGHT_SEASON = 0.25


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LineResult:
    """Result of fetching a betting line."""
    line: float
    source: str  # "sportsbook" or "derived"
    book: str    # e.g., "draftkings", "derived_l10"
    
    @property
    def is_sportsbook(self) -> bool:
        return self.source == "sportsbook"
    
    @property
    def is_derived(self) -> bool:
        return self.source == "derived"


@dataclass
class PlayerStats:
    """Comprehensive player statistics for projections."""
    player_id: int
    player_name: str
    team_abbrev: str
    position: str
    games_played: int
    avg_minutes: float
    
    # Averages at different windows
    l3: Dict[str, float] = field(default_factory=dict)
    l5: Dict[str, float] = field(default_factory=dict)
    l10: Dict[str, float] = field(default_factory=dict)
    l15: Dict[str, float] = field(default_factory=dict)
    l20: Dict[str, float] = field(default_factory=dict)
    season: Dict[str, float] = field(default_factory=dict)
    
    # Deviations (percentage change)
    deviations_l15: Dict[str, float] = field(default_factory=dict)   # L5 vs L15
    deviations_season: Dict[str, float] = field(default_factory=dict)  # L5 vs Season
    
    # Last game values
    last_game: Dict[str, float] = field(default_factory=dict)
    
    # Standard deviations (for consistency)
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Recent game values (for sustained pattern)
    recent_games: Dict[str, List[float]] = field(default_factory=dict)
    
    def get_projection(
        self, 
        prop_type: str,
        weight_l5: float = WEIGHT_L5,
        weight_l10: float = WEIGHT_L10,
        weight_l15: float = WEIGHT_L15,
        weight_season: float = WEIGHT_SEASON,
    ) -> float:
        """Calculate weighted projection for a prop type."""
        pt = prop_type.lower()
        
        l5_val = self.l5.get(pt, 0)
        l10_val = self.l10.get(pt, 0)
        l15_val = self.l15.get(pt, 0)
        season_val = self.season.get(pt, 0)
        
        total_weight = weight_l5 + weight_l10 + weight_l15 + weight_season
        if total_weight <= 0:
            return season_val
        
        projection = (
            l5_val * weight_l5 +
            l10_val * weight_l10 +
            l15_val * weight_l15 +
            season_val * weight_season
        ) / total_weight
        
        return projection
    
    def get_cv(self, prop_type: str) -> float:
        """Get coefficient of variation (std/mean) for consistency check."""
        pt = prop_type.lower()
        mean = self.l10.get(pt, 0)
        std = self.stds.get(pt, 0)
        if mean <= 0:
            return 1.0  # High CV for zero/negative mean
        return std / mean


@dataclass
class DefenseContext:
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
class PatternResult:
    """Result of pattern detection."""
    pattern_name: str       # cold_bounce, hot_sustained, cold_streak, elite_defense, none
    direction: str          # OVER, UNDER
    confidence_bonus: float # Bonus to add to confidence score
    reasons: List[str]      # Human-readable explanations
    is_valid: bool          # Whether pattern was detected
    
    @property
    def is_over_pattern(self) -> bool:
        return self.direction == "OVER" and self.is_valid
    
    @property
    def is_under_pattern(self) -> bool:
        return self.direction == "UNDER" and self.is_valid


# ============================================================================
# Name Normalization
# ============================================================================

def normalize_name(name: str) -> str:
    """Normalize a player name for matching: lowercase, remove accents, strip."""
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.lower().strip()


# ============================================================================
# Line Fetching
# ============================================================================

def get_line_with_source(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    l10_average: float,
) -> LineResult:
    """
    Fetch sportsbook line if available, otherwise derive from L10 average.
    
    CRITICAL: This function ALWAYS returns a line and tracks its source.
    This allows picks even when lines aren't available yet.
    
    Args:
        conn: Database connection
        player_id: Player's database ID
        player_name: Player's name (for fuzzy matching)
        prop_type: PTS, REB, or AST
        game_date: Date of the game
        l10_average: Player's L10 average for this stat (used for derived line)
    
    Returns:
        LineResult with line value, source ("sportsbook" or "derived"), and book
    """
    # Try to get sportsbook line by player_id first
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
            return LineResult(
                line=row["line"],
                source="sportsbook",
                book=row["book"] or "unknown"
            )
    
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
    
    norm_name = normalize_name(player_name)
    for row in rows:
        if normalize_name(row["name"]) == norm_name:
            return LineResult(
                line=row["line"],
                source="sportsbook",
                book=row["book"] or "unknown"
            )
    
    # Fall back to derived line
    # Apply adjustment factor (Vegas typically sets 5% above player average)
    derived_line = l10_average * DERIVED_LINE_FACTOR
    
    return LineResult(
        line=round(derived_line, 1),
        source="derived",
        book="derived_l10"
    )


def get_sportsbook_line_only(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Optional[Tuple[float, str]]:
    """
    Get sportsbook line only (no fallback to derived).
    Returns (line, book) tuple or None if no sportsbook line exists.
    """
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
    
    # Try by player name
    rows = conn.execute(
        """
        SELECT sl.line, sl.book, p.name
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.prop_type = ? AND sl.as_of_date = ?
        """,
        (prop_type.upper(), game_date)
    ).fetchall()
    
    norm_name = normalize_name(player_name)
    for row in rows:
        if normalize_name(row["name"]) == norm_name:
            return (row["line"], row["book"] or "unknown")
    
    return None


# ============================================================================
# Edge Calculation
# ============================================================================

def calculate_edge(
    projection: float,
    line: float,
    direction: str,
) -> float:
    """
    Calculate edge percentage vs a line.
    
    For OVER: edge = (projection - line) / line * 100
    For UNDER: edge = (line - projection) / line * 100
    
    Positive edge = in our favor
    """
    if line <= 0:
        return 0.0
    
    if direction.upper() == "OVER":
        return (projection - line) / line * 100
    else:  # UNDER
        return (line - projection) / line * 100


def check_line_value(
    projection: float,
    sportsbook_line: Optional[float],
    derived_line: float,
    direction: str,
) -> Tuple[bool, str]:
    """
    Check if there's actual value vs the sportsbook line (if available).
    
    Returns:
        (has_value, reason): Whether pick has value and why
    """
    if sportsbook_line is None:
        return (True, "No sportsbook line to compare")
    
    # Compare projection to sportsbook line
    if direction.upper() == "OVER":
        has_value = projection > sportsbook_line
        if not has_value:
            return (False, f"Market already adjusted: line {sportsbook_line} > projection {projection:.1f}")
    else:
        has_value = projection < sportsbook_line
        if not has_value:
            return (False, f"Market already adjusted: line {sportsbook_line} < projection {projection:.1f}")
    
    return (True, "Value exists vs sportsbook line")


# ============================================================================
# Injury Checking
# ============================================================================

def get_injured_players_for_date(
    conn: sqlite3.Connection, 
    game_date: str
) -> Dict[int, str]:
    """
    Get players who are OUT or DOUBTFUL for the given date.
    
    Returns dict mapping player_id -> status for players to exclude.
    """
    rows = conn.execute(
        """
        SELECT ir.player_id, ir.player_name, ir.status, p.id as resolved_id, p.name as resolved_name
        FROM injury_report ir
        LEFT JOIN players p ON ir.player_id = p.id
        WHERE ir.game_date = ?
          AND ir.status IN ('OUT', 'DOUBTFUL')
        """,
        (game_date,),
    ).fetchall()
    
    result = {}
    
    for row in rows:
        status = row["status"].upper() if row["status"] else ""
        
        if row["player_id"]:
            result[row["player_id"]] = status
        
        if row["resolved_id"]:
            result[row["resolved_id"]] = status
    
    # For entries without player_id, try to match by name
    unmatched = [
        (row["player_name"], row["status"]) 
        for row in rows 
        if not row["player_id"] and row["player_name"]
    ]
    
    if unmatched:
        all_players = conn.execute("SELECT id, name FROM players").fetchall()
        for player_name, status in unmatched:
            norm_name = normalize_name(player_name)
            for p in all_players:
                if normalize_name(p["name"]) == norm_name:
                    result[p["id"]] = status
                    break
    
    return result


def get_injured_player_names(
    conn: sqlite3.Connection, 
    game_date: str
) -> Set[str]:
    """Get set of normalized player names who are OUT or DOUBTFUL."""
    rows = conn.execute(
        """
        SELECT DISTINCT COALESCE(p.name, ir.player_name) as player_name
        FROM injury_report ir
        LEFT JOIN players p ON ir.player_id = p.id
        WHERE ir.game_date = ?
          AND ir.status IN ('OUT', 'DOUBTFUL')
        """,
        (game_date,),
    ).fetchall()
    
    return {normalize_name(row["player_name"]) for row in rows if row["player_name"]}


def get_out_players_for_team(
    conn: sqlite3.Connection,
    team_abbrev: str,
    game_date: str,
) -> List[str]:
    """Get list of OUT player names for a specific team on a date."""
    rows = conn.execute(
        """
        SELECT DISTINCT COALESCE(p.name, ir.player_name) as player_name
        FROM injury_report ir
        LEFT JOIN players p ON ir.player_id = p.id
        LEFT JOIN teams t ON ir.team_id = t.id
        WHERE ir.game_date = ?
          AND ir.status = 'OUT'
          AND (t.name LIKE ? OR ir.player_name IS NOT NULL)
        """,
        (game_date, f"%{team_abbrev}%"),
    ).fetchall()
    
    return [row["player_name"] for row in rows if row["player_name"]]


# ============================================================================
# Defense Context
# ============================================================================

def get_defense_context(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
) -> DefenseContext:
    """Get defense vs position context for an opponent team."""
    context = DefenseContext(
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
            if rank <= ELITE_DEFENSE_RANK:
                rating = "elite"
            elif rank <= GOOD_DEFENSE_RANK:
                rating = "good"
            elif rank <= WEAK_DEFENSE_RANK:
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


def get_defense_adjustment(
    defense_rating: str,
    elite_adj: float = 0.88,
    good_adj: float = 0.94,
    neutral_adj: float = 1.00,
    weak_adj: float = 1.06,
) -> float:
    """Get projection adjustment multiplier based on defense rating."""
    mapping = {
        "elite": elite_adj,
        "good": good_adj,
        "average": neutral_adj,
        "weak": weak_adj,
    }
    return mapping.get(defense_rating, neutral_adj)


# ============================================================================
# Player Stats Loading
# ============================================================================

def load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    min_games: int = MIN_GAMES_REQUIRED,
    min_minutes_filter: int = MIN_MINUTES_FILTER,
    min_avg_minutes: float = MIN_AVG_MINUTES,
    max_lookback: int = MAX_GAMES_LOOKBACK,
) -> Optional[PlayerStats]:
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
        (player_id, before_date, min_minutes_filter, max_lookback),
    ).fetchall()
    
    if len(rows) < min_games:
        return None
    
    games = [dict(r) for r in rows]
    n = len(games)
    
    # Check minimum average minutes
    avg_min = sum(g["minutes"] or 0 for g in games) / n
    if avg_min < min_avg_minutes:
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
    
    # Build player stats
    player_stats = PlayerStats(
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
        
        if l15 > 0:
            player_stats.deviations_l15[stat] = (l5 - l15) / l15 * 100
        else:
            player_stats.deviations_l15[stat] = 0.0
            
        if season > 0:
            player_stats.deviations_season[stat] = (l5 - season) / season * 100
        else:
            player_stats.deviations_season[stat] = 0.0
    
    return player_stats


# ============================================================================
# Pattern Detection
# ============================================================================

def detect_over_pattern(
    stats: PlayerStats,
    prop_type: str,
    defense_context: DefenseContext,
    cold_threshold: float = COLD_DEVIATION_THRESHOLD,
    hot_threshold: float = HOT_DEVIATION_THRESHOLD,
    bounce_threshold: float = BOUNCE_THRESHOLD,
    sustained_games: int = SUSTAINED_GAMES_ABOVE,
) -> PatternResult:
    """
    Detect OVER patterns (cold bounce, hot sustained).
    
    Patterns:
    1. Cold Bounce: Player is cold but last game shows recovery
       - L5 is 20%+ below L15 (cold)
       - Last game > L10 (bouncing back)
       - NOT facing elite defense
       
    2. Hot Sustained: Player is hot and maintaining
       - L5 is 30%+ above L15 (hot)
       - L3 >= L5 (still hot or accelerating)
       - 3+ of last 5 above L15
       - NOT facing elite defense
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
    if deviation_l15 <= cold_threshold:
        # Last game must be above L10 (bouncing back)
        bounce_pct = (last_game - l10) / l10 * 100 if l10 > 0 else 0
        if bounce_pct >= bounce_threshold:
            # Additional check: opponent not elite defense
            if defense_context.get_rating(pt) != "elite":
                reasons = [
                    f"Cold bounce: L5 ({l5:.1f}) is {deviation_l15:.0f}% below L15 ({l15:.1f})",
                    f"Recovery signal: Last game ({last_game:.0f}) is {bounce_pct:.0f}% above L10 ({l10:.1f})",
                    f"Regression expected toward baseline ({l15:.1f})",
                ]
                confidence_bonus = min(abs(deviation_l15) / 2, 10)
                return PatternResult(
                    pattern_name="cold_bounce",
                    direction="OVER",
                    confidence_bonus=confidence_bonus,
                    reasons=reasons,
                    is_valid=True,
                )
    
    # Check HOT SUSTAINED pattern (65.9% from Production)
    if deviation_l15 >= hot_threshold:
        # L3 >= L5 (still hot or accelerating)
        if l3 >= l5 * 0.95:
            # Count games above L15
            games_above = sum(1 for v in recent if v > l15)
            if games_above >= sustained_games:
                # Additional check: not facing elite defense
                if defense_context.get_rating(pt) != "elite":
                    reasons = [
                        f"Hot sustained: L5 ({l5:.1f}) is {deviation_l15:.0f}% above L15 ({l15:.1f})",
                        f"Momentum: L3 ({l3:.1f}) maintaining level",
                        f"Consistency: {games_above}/5 recent games above baseline",
                    ]
                    confidence_bonus = min((deviation_l15 - hot_threshold) / 3, 8)
                    return PatternResult(
                        pattern_name="hot_sustained",
                        direction="OVER",
                        confidence_bonus=confidence_bonus,
                        reasons=reasons,
                        is_valid=True,
                    )
    
    # No valid OVER pattern
    return PatternResult(
        pattern_name="none",
        direction="OVER",
        confidence_bonus=0,
        reasons=[],
        is_valid=False,
    )


def detect_under_pattern(
    stats: PlayerStats,
    prop_type: str,
    defense_context: DefenseContext,
    cold_streak_threshold: float = COLD_STREAK_THRESHOLD,
) -> PatternResult:
    """
    Detect UNDER patterns (elite defense, cold streak, combined).
    
    Patterns:
    1. Elite Defense: Facing top-5 defense at position
    2. Cold Streak: Player is significantly below season average
    3. Combined: Cold + Good/Elite Defense (strongest)
    """
    pt = prop_type.lower()
    
    deviation_season = stats.deviations_season.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
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
    if deviation_season <= cold_streak_threshold:
        reasons.append(f"Cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season avg ({season:.1f})")
        if pattern_name == "elite_defense":
            pattern_name = "elite_defense_cold"
            confidence_bonus += 10  # Combined patterns are strongest
        else:
            pattern_name = "cold_streak"
            confidence_bonus += 6
        is_valid = True
    
    # Good defense + cold also works (weaker pattern)
    if defense_rating == "good" and deviation_season <= -10:
        if not is_valid:
            reasons.append(f"Good defense: {defense_context.team_abbrev} ranks #{defense_rank} + player in cold stretch")
            pattern_name = "good_defense_cold"
            confidence_bonus += 5
            is_valid = True
        elif pattern_name == "cold_streak":
            # Upgrade cold streak with good defense context
            reasons.append(f"Good defense support: {defense_context.team_abbrev} ranks #{defense_rank}")
            confidence_bonus += 3
    
    return PatternResult(
        pattern_name=pattern_name,
        direction="UNDER",
        confidence_bonus=confidence_bonus,
        reasons=reasons,
        is_valid=is_valid,
    )


# ============================================================================
# Back-to-Back Detection
# ============================================================================

def get_back_to_back_status(
    conn: sqlite3.Connection,
    team_abbrev: str,
    game_date: str,
) -> Dict[str, bool]:
    """Check if team is on back-to-back or has played 3 in 4 nights."""
    result = {
        "is_b2b": False,
        "is_third_in_four": False,
    }
    
    try:
        game_dt = datetime.strptime(game_date, "%Y-%m-%d")
        yesterday = (game_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        four_days_ago = (game_dt - timedelta(days=4)).strftime("%Y-%m-%d")
        
        # Check for game yesterday
        yesterday_game = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM games g
            JOIN teams t ON (t.id = g.team1_id OR t.id = g.team2_id)
            WHERE g.game_date = ?
              AND t.name LIKE ?
            """,
            (yesterday, f"%{team_abbrev}%"),
        ).fetchone()
        
        if yesterday_game and yesterday_game["cnt"] > 0:
            result["is_b2b"] = True
        
        # Check for 3 in 4 nights
        recent_games = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM games g
            JOIN teams t ON (t.id = g.team1_id OR t.id = g.team2_id)
            WHERE g.game_date BETWEEN ? AND ?
              AND g.game_date < ?
              AND t.name LIKE ?
            """,
            (four_days_ago, game_date, game_date, f"%{team_abbrev}%"),
        ).fetchone()
        
        if recent_games and recent_games["cnt"] >= 2:
            result["is_third_in_four"] = True
    except:
        pass
    
    return result


# ============================================================================
# Usage Redistribution
# ============================================================================

def get_usage_boost(
    conn: sqlite3.Connection,
    player_id: int,
    team_abbrev: str,
    game_date: str,
) -> Dict[str, float]:
    """
    Calculate usage boost for a player when stars are out.
    
    Returns dict with boost values for pts, reb, ast.
    """
    boosts = {"pts": 0.0, "reb": 0.0, "ast": 0.0}
    
    try:
        # Import here to avoid circular imports
        from .usage_redistribution import calculate_usage_redistribution, get_team_usage_profiles
        
        # Get OUT players for this team
        out_players = get_out_players_for_team(conn, team_abbrev, game_date)
        
        if not out_players:
            return boosts
        
        # Get usage profiles
        profiles = get_team_usage_profiles(conn, team_abbrev)
        
        for out_player in out_players:
            redistribution = calculate_usage_redistribution(conn, team_abbrev, out_player)
            
            if redistribution:
                for r in redistribution.redistributions:
                    if r["player_id"] == player_id:
                        boosts["pts"] += r.get("pts_boost", 0)
                        boosts["reb"] += r.get("reb_boost", 0)
                        boosts["ast"] += r.get("ast_boost", 0)
                        break
    except:
        pass
    
    return boosts


# ============================================================================
# Utility Functions
# ============================================================================

def map_position(pos: str) -> str:
    """Map raw position to standard position for DVP lookup."""
    if not pos:
        return "SF"
    
    pos = pos.upper()
    
    # Direct mappings
    if pos in ("PG", "SG", "SF", "PF", "C"):
        return pos
    
    # Common combinations
    mapping = {
        "G": "PG",
        "F": "SF",
        "G-F": "SG",
        "F-G": "SF",
        "F-C": "PF",
        "C-F": "PF",
    }
    
    return mapping.get(pos, "SF")


def get_games_for_date(
    conn: sqlite3.Connection,
    game_date: str,
) -> List[Dict[str, Any]]:
    """Get all games for a specific date."""
    rows = conn.execute(
        """
        SELECT g.id, t1.name as team1, t2.name as team2
        FROM games g
        JOIN teams t1 ON t1.id = g.team1_id
        JOIN teams t2 ON t2.id = g.team2_id
        WHERE g.game_date = ?
        """,
        (game_date,),
    ).fetchall()
    
    return [dict(r) for r in rows]


def get_actual_stats(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
) -> Optional[Dict[str, float]]:
    """Get actual stats for a player on a specific date."""
    row = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
          AND g.game_date = ?
          AND bp.minutes > 0
        """,
        (player_id, game_date),
    ).fetchone()
    
    if not row:
        return None
    
    return {
        "pts": row["pts"] or 0,
        "reb": row["reb"] or 0,
        "ast": row["ast"] or 0,
        "min": row["minutes"] or 0,
    }


def get_game_dates_in_range(
    conn: sqlite3.Connection,
    start_date: str,
    end_date: str,
) -> List[str]:
    """Get all game dates in a range."""
    rows = conn.execute(
        """
        SELECT DISTINCT game_date
        FROM games
        WHERE game_date >= ? AND game_date <= ?
        ORDER BY game_date
        """,
        (start_date, end_date),
    ).fetchall()
    
    return [row["game_date"] for row in rows]


def get_team_players(
    conn: sqlite3.Connection,
    team_id: int,
    game_date: str,
    min_games: int = MIN_GAMES_REQUIRED,
    min_minutes_filter: int = MIN_MINUTES_FILTER,
) -> List[Dict[str, Any]]:
    """Get top players for a team based on minutes played."""
    rows = conn.execute(
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
        (team_id, game_date, min_minutes_filter, min_games),
    ).fetchall()
    
    return [dict(r) for r in rows]
