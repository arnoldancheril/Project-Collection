"""
Model V16 Shared Utilities
===========================

Common functions, data classes, and utilities shared between:
- Model V16 General (Over/Value focused + can suggest UNDER when strongest)
- Model V16 Under (UNDER specialized - placeholder for future development)

This module provides:
1. Player statistics loading with comprehensive windows (L3/L5/L10/L15/Season)
2. HYBRID LINE HANDLING: Sportsbook lines when available, projections when not
3. Defense vs position context with proper position mapping
4. Injury tracking and back-to-back status checking
5. Pattern detection utilities (Cold Bounce, Cold Streak, etc.)
6. Usage redistribution calculations
7. Edge calculation and confidence scoring
8. Common data classes

=============================================================================
MODEL V16 KEY INNOVATIONS (Addressing ALL Previous Model Shortcomings)
=============================================================================

1. **HYBRID LINE APPROACH** (Per User Request):
   - Use actual sportsbook lines when available (for accurate edge calculation)
   - STILL GENERATE PICKS when lines aren't available (lines come late!)
   - Track line source for honest reporting
   - Apply stricter edge requirements for derived lines (10% vs 6%)

2. **PATTERN-BASED PICKS ONLY** (No Generic Edge Picks):
   - Every pick requires a VALIDATED pattern from backtesting
   - Cold Bounce: 84.6% (V14) / 65.9% (V10) - BEST OVER pattern
   - B2B Fatigue UNDER: 60.5% - STRONG
   - Elite Defense UNDER: 62.2% - STRONG
   - Hot Sustained: DISABLED (25.8% hit rate in V14)

3. **STRATEGIC DIRECTION SELECTION** (From RCM v1.4 Analysis):
   - PTS: UNDER strongly preferred (63.9% vs 48.3% OVER)
   - REB: Both directions (~59% each)
   - AST: EXCLUDED entirely (~54% is coin flip after juice)

4. **STRICT FILTERING** (From V9/V10/Idea.txt):
   - 23+ minute average for established players
   - 10+ games history required
   - Pattern confirmation REQUIRED
   - Exclude scoring guards when cold_bounce (volatile - 51.5%)

5. **USAGE REDISTRIBUTION** (From RCM/Idea.txt):
   - When star is OUT, remaining players get usage boost
   - Only suggest OVER if usage increases meaningfully (>5%)
   - Key for identifying OVER opportunities

6. **DEFENSE INTEGRATION** (From Under Model V2/V10):
   - Elite defense (top 5 DVP) is STRONG UNDER signal (62.2%)
   - Elite defense BLOCKS PTS OVER picks
   - Good defense (top 10) + Cold streak = PREMIUM UNDER

7. **HONEST REPORTING**:
   - Track sportsbook vs derived line picks separately
   - Report hit rates by line source
   - No inflated metrics

=============================================================================
VALIDATED PATTERNS & HIT RATES (From Extensive Backtesting)
=============================================================================

| Pattern               | Direction | Hit Rate | Source      | Notes                    |
|----------------------|-----------|----------|-------------|--------------------------|
| Cold Bounce          | OVER      | 65-84%   | V14/V10     | BEST - Regression to mean|
| B2B Fatigue          | UNDER     | 60.5%    | V14         | Second game of B2B       |
| Elite Defense        | UNDER     | 62.2%    | V10/V14     | Top 5 DVP                |
| Cold Streak          | UNDER     | 57.8%    | V14         | L5 < Season by 20%+      |
| Combined (Elite+Cold)| UNDER     | 55-62%   | V14         | Multiple factors         |
| Hot Sustained        | OVER      | 25.8%    | V14         | DISABLED - unreliable    |
| Usage Boost          | OVER      | 52.4%    | V14         | When stars OUT           |

=============================================================================

Author: NBA Props Team - Model V16
Created: February 2026
Version: 16.0
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

MODEL_VERSION = "16.1"
MODEL_NAME = "Model V16"


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
# V15/V14 testing showed Top 3 is elite (top 4-5 was only 27% UNDER)
ELITE_DEFENSE_RANK = 5      # Top 5 = elite defense
GOOD_DEFENSE_RANK = 10      # Top 10 = good defense
AVERAGE_DEFENSE_RANK = 15   # Top 15 = average
POOR_DEFENSE_RANK = 25      # Bottom 5 = weak defense

# Minimum thresholds for including props (filter low-volume players)
MIN_PROP_AVERAGES = {
    'pts': 8.0,     # Min 8 PPG to consider
    'reb': 4.0,     # Min 4 RPG to consider
    'ast': 8.5,     # Min 8.5 APG for AST picks (VERY high bar - usually excluded)
}

# Prop types supported
PROP_TYPES = ['pts', 'reb']  # AST excluded by default (54% is coin flip)


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
class PlayerStatsV16:
    """
    Comprehensive player statistics for Model V16.
    
    Includes all averaging windows, deviations, variance metrics,
    and recent game values needed for pattern detection.
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
    
    # Standard deviations (L10 window)
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Recent game values (last 5) for pattern analysis
    recent_games: Dict[str, List[float]] = field(default_factory=dict)
    
    # Historical vs specific opponent (if available)
    vs_opponent: Dict[str, float] = field(default_factory=dict)
    vs_opponent_games: int = 0
    
    def get_projection(
        self, 
        prop_type: str, 
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate weighted projection for a prop type.
        
        Default weights (validated from V14/V15 backtesting):
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
        
        CV < 0.20 = Very consistent player (confidence boost)
        CV > 0.40 = Volatile player (confidence penalty)
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


@dataclass
class DefenseContextV16:
    """
    Defense vs position context for an opponent team.
    
    This is from Hashtag Basketball DVP data and is CRITICAL for UNDER picks.
    Elite defense (top 5) = ~62% UNDER hit rate from V10 testing.
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
        """Check if defense is elite (top 5) for this prop type."""
        return self.get_rank(prop_type) <= ELITE_DEFENSE_RANK
    
    def is_good(self, prop_type: str) -> bool:
        """Check if defense is good (top 10) for this prop type."""
        return self.get_rank(prop_type) <= GOOD_DEFENSE_RANK
    
    def is_weak(self, prop_type: str) -> bool:
        """Check if defense is weak (bottom 5) for this prop type."""
        return self.get_rank(prop_type) >= POOR_DEFENSE_RANK


@dataclass
class BackToBackInfo:
    """
    Information about team's rest/fatigue status.
    
    B2B Fatigue showed 60.5% hit rate for UNDER in V14 testing.
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
    usage_boost_pct: float = 0.0
    
    def has_significant_impact(self) -> bool:
        """Check if injuries create meaningful usage boost."""
        return self.usage_boost_pct >= 3.0


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
    # Try to find team by abbreviation
    team_row = conn.execute(
        """
        SELECT id FROM teams 
        WHERE name LIKE ? 
           OR name LIKE ?
        LIMIT 1
        """,
        (f"%{team_abbrev}%", f"{team_abbrev}%")
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
    stats: PlayerStatsV16,
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
    stats: PlayerStatsV16,
    derived_adjustment: float = 1.05
) -> LineInfo:
    """
    Get line for a player - sportsbook if available, derived otherwise.
    
    This is the KEY FUNCTION for hybrid line handling per V16 design.
    Always try sportsbook first for honest edge calculation.
    
    KEY V16 CHANGE: We ALWAYS return a line (sportsbook or derived)
    because user specified that lines come late, so we should still
    generate picks with projections.
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
) -> DefenseContextV16:
    """
    Get defense vs position context for an opponent team.
    
    This queries the team_defense_vs_position table (from Hashtag Basketball).
    """
    context = DefenseContextV16(
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
    
    B2B fatigue showed 60.5% UNDER hit rate in V14 testing.
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


def load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    min_games: int = 10,
    min_minutes: float = 20.0,
    max_games: int = 20,
    min_game_minutes: int = 5,
) -> Optional[PlayerStatsV16]:
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
    stats = PlayerStatsV16(
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
# Usage Redistribution
# ============================================================================

def calculate_usage_boost(
    injured_teammates: List[Dict[str, Any]],
    boost_per_player: float = 0.03,
    max_boost: float = 0.12,
    min_pts_threshold: float = 15.0,
) -> float:
    """
    Calculate usage boost percentage when teammates are injured.
    
    From RCM and Idea.txt: When a star is OUT, remaining players get boost.
    Only considers teammates with avg_pts >= threshold (significant contributors).
    
    Args:
        injured_teammates: List of injured player dicts with avg_pts
        boost_per_player: Boost per significant injured player (3% default)
        max_boost: Maximum total boost (12% default)
        min_pts_threshold: Minimum avg_pts to be "significant" (15.0 default)
    
    Returns:
        Usage boost as decimal (e.g., 0.06 = 6% boost)
    """
    significant_out = [
        p for p in injured_teammates 
        if p.get("avg_pts", 0) >= min_pts_threshold
    ]
    
    if not significant_out:
        return 0.0
    
    boost = len(significant_out) * boost_per_player
    return min(boost, max_boost)


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
    
    Positive edge means favorable bet.
    """
    if line <= 0:
        return 0.0
    
    if direction.upper() == "OVER":
        return (projection - line) / line * 100
    else:  # UNDER
        return (line - projection) / line * 100


# ============================================================================
# Pattern Detection
# ============================================================================

def detect_cold_bounce_pattern(
    stats: PlayerStatsV16,
    prop_type: str,
    cold_threshold: float = -15.0,
    bounce_threshold: float = 5.0,
) -> Tuple[bool, List[str]]:
    """
    Detect Cold Bounce pattern for OVER picks.
    
    This is the BEST OVER pattern - 65-84% hit rate in testing!
    
    Conditions:
    1. L5 is cold_threshold% or more BELOW L15 (player is cold)
    2. Last game was bounce_threshold% or more ABOVE L10 (showing recovery)
    
    Logic: Regression to mean after cold streak
    
    Returns: (is_pattern, reasons)
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


def detect_hot_sustained_pattern(
    stats: PlayerStatsV16,
    prop_type: str,
    hot_threshold: float = 30.0,
    sustained_games: int = 3,
) -> Tuple[bool, List[str]]:
    """
    Detect Hot Sustained pattern for OVER picks.
    
    NOTE: V14 testing showed only 25.8% hit rate for hot sustained!
    This pattern is DISABLED by default in V16.
    
    Conditions:
    1. L5 is hot_threshold% or more ABOVE L15 (player is hot)
    2. L3 >= L5 (still hot, not cooling)
    3. sustained_games of last 5 games above L15
    
    Returns: (is_pattern, reasons)
    """
    pt = prop_type.lower()
    
    deviation_l15 = stats.deviations_l15.get(pt, 0)
    l3 = stats.l3.get(pt, 0)
    l5 = stats.l5.get(pt, 0)
    l15 = stats.l15.get(pt, 0)
    recent = stats.recent_games.get(pt, [])
    
    if deviation_l15 < hot_threshold:
        return False, []
    
    # Check not cooling off
    if l3 < l5 * 0.95:
        return False, []
    
    # Check sustained
    games_above = sum(1 for v in recent if v > l15)
    if games_above < sustained_games:
        return False, []
    
    reasons = [
        f"Hot sustained: L5 ({l5:.1f}) is {deviation_l15:.0f}% above L15 ({l15:.1f})",
        f"Momentum maintained: L3 ({l3:.1f}) still strong",
        f"Consistency: {games_above}/5 recent games above baseline",
    ]
    
    return True, reasons


def detect_cold_streak_pattern(
    stats: PlayerStatsV16,
    prop_type: str,
    mild_threshold: float = -10.0,
    severe_threshold: float = -20.0,
) -> Tuple[str, List[str]]:
    """
    Detect Cold Streak pattern for UNDER picks.
    
    Cold streaks persist longer than hot streaks (psychology).
    Used as UNDER signal.
    
    Returns: (severity, reasons)
    - severity: "none", "mild", "severe"
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
            f"Consistent underperformance - cold trend persisting",
        ]
        return "severe", reasons
    else:
        reasons = [
            f"Mild cold streak: L5 ({l5:.1f}) is {deviation_season:.0f}% below season ({season:.1f})",
        ]
        return "mild", reasons


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
    else:  # UNDER
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
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
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
    }
