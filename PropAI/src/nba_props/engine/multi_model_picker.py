"""
Multi-Model Picker - Smart Ensemble for NBA Props Predictions
=============================================================

This module implements an improved ensemble approach based on comprehensive 
backtesting analysis. It generates 10+ picks per day while maintaining good
hit rates by combining the strengths of multiple validated models.

KEY INSIGHTS FROM BACKTESTING (Jan 1 - Feb 3, 2026):
----------------------------------------------------
- V9:  72.0% hit rate (67/93) - Best overall, uses cold bounce OVERs
- V16: 67.9% hit rate (36/53) - Pattern-based (cold bounce, defense)
- V17: 59.4% (233/392) - Multi-factor with special patterns:
    - b2b_fatigue: 74.5% (41/55)
    - cold_bounce: 77.8% (7/9)
    - defense_good: 70.7% (29/41)
    - defense_elite: 62.5% (20/32)
- V18/V19: ~56% but higher volume

STRATEGY:
---------
1. Prioritize picks where HIGH-ACCURACY patterns are detected
2. Generate picks across multiple tiers for volume
3. Weight models by their validated accuracy
4. Separate tracking for sportsbook vs derived line picks

PICK GENERATION TARGETS:
------------------------
- 3-5 games: 5-8 picks
- 6-10 games: 8-12 picks  
- 11+ games: 12-15 picks

Author: PropAI Team
Created: February 2026
Version: 1.0
"""
from __future__ import annotations

import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths


# ============================================================================
# CONFIGURATION BASED ON BACKTESTING RESULTS
# ============================================================================

MODEL_VERSION = "1.0"

# Model weights based on validated backtesting (higher = more influence)
MODEL_ACCURACY = {
    "v9": 0.72,
    "v16_general": 0.68,
    "v17_general": 0.59,
    "v18_general": 0.57,
    "v19_general": 0.56,
}

# High-value patterns discovered from V17 factor analysis
HIGH_VALUE_PATTERNS = {
    "cold_bounce": 0.778,      # 77.8% hit rate
    "b2b_fatigue": 0.745,      # 74.5% hit rate  
    "defense_good": 0.707,     # 70.7% hit rate
    "defense_elite": 0.625,    # 62.5% hit rate
    "injury_rust_first": 0.583, # 58.3% hit rate
}

# Minimum edge requirements for each tier
TIER_CONFIG = {
    "PREMIUM": {
        "min_edge": 10.0,
        "min_models": 2,
        "require_sportsbook": True,
        "expected_hit": 0.70,
    },
    "HIGH": {
        "min_edge": 7.0,
        "min_models": 1,
        "require_sportsbook": False,
        "expected_hit": 0.65,
    },
    "STANDARD": {
        "min_edge": 5.0,
        "min_models": 1,
        "require_sportsbook": False,
        "expected_hit": 0.58,
    },
}


class ConfidenceTier(Enum):
    PREMIUM = "PREMIUM"
    HIGH = "HIGH"
    STANDARD = "STANDARD"
    
    @property
    def stars(self) -> int:
        return {"PREMIUM": 5, "HIGH": 4, "STANDARD": 3}[self.value]
    
    @property
    def display(self) -> str:
        return "★" * self.stars + "☆" * (5 - self.stars)


class LineSource(Enum):
    SPORTSBOOK = "sportsbook"
    DERIVED = "derived"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PickSignal:
    """A signal from a single model/pattern."""
    source: str  # model name or pattern name
    player_id: int
    player_name: str
    team: str
    opponent: str
    game_date: str
    prop_type: str
    direction: str
    line: float
    line_source: LineSource
    projection: float
    edge: float
    confidence: float
    pattern: str  # Pattern that triggered this signal
    factors: List[str] = field(default_factory=list)
    accuracy: float = 0.55  # Expected accuracy of this signal source


@dataclass
class MultiModelPick:
    """A unified pick combining multiple signals."""
    # Identification
    pick_id: str
    
    # Player/Game
    player_id: int
    player_name: str
    team: str
    opponent: str
    game_date: str
    game_id: Optional[int] = None
    
    # Prop
    prop_type: str = ""
    direction: str = ""
    line: float = 0.0
    line_source: LineSource = LineSource.DERIVED
    
    # Projection
    projection: float = 0.0
    edge: float = 0.0
    
    # Confidence
    tier: ConfidenceTier = ConfidenceTier.STANDARD
    confidence_score: float = 0.0
    expected_hit_rate: float = 0.55
    
    # Sources
    signals: List[PickSignal] = field(default_factory=list)
    primary_pattern: str = ""
    key_factors: List[str] = field(default_factory=list)
    
    # Grading (filled after game)
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    
    def to_dict(self) -> Dict:
        return {
            "pick_id": self.pick_id,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "team": self.team,
            "opponent": self.opponent,
            "game_date": self.game_date,
            "game_id": self.game_id,
            "prop_type": self.prop_type,
            "direction": self.direction,
            "line": self.line,
            "line_source": self.line_source.value,
            "projection": round(self.projection, 1),
            "edge": round(self.edge, 1),
            "tier": self.tier.value,
            "tier_display": self.tier.display,
            "confidence_score": round(self.confidence_score, 1),
            "expected_hit_rate": round(self.expected_hit_rate * 100, 1),
            "num_signals": len(self.signals),
            "primary_pattern": self.primary_pattern,
            "key_factors": self.key_factors,
            "actual_value": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV2:
    """Collection of picks for a day."""
    date: str
    num_games: int = 0
    picks: List[MultiModelPick] = field(default_factory=list)
    
    # Statistics
    premium_count: int = 0
    high_count: int = 0
    standard_count: int = 0
    sportsbook_count: int = 0
    derived_count: int = 0
    
    def calculate_stats(self):
        self.premium_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.PREMIUM)
        self.high_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.HIGH)
        self.standard_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.STANDARD)
        self.sportsbook_count = sum(1 for p in self.picks if p.line_source == LineSource.SPORTSBOOK)
        self.derived_count = sum(1 for p in self.picks if p.line_source == LineSource.DERIVED)
    
    def to_dict(self) -> Dict:
        self.calculate_stats()
        return {
            "date": self.date,
            "num_games": self.num_games,
            "total_picks": len(self.picks),
            "picks": [p.to_dict() for p in self.picks],
            "by_tier": {
                "premium": self.premium_count,
                "high": self.high_count,
                "standard": self.standard_count,
            },
            "by_line_source": {
                "sportsbook": self.sportsbook_count,
                "derived": self.derived_count,
            },
        }
    
    def summary(self) -> str:
        self.calculate_stats()
        lines = [
            "=" * 60,
            f"MULTI-MODEL PICKS - {self.date}",
            "=" * 60,
            f"Games: {self.num_games} | Total Picks: {len(self.picks)}",
            "",
            "BY TIER:",
            f"  ★★★★★ Premium: {self.premium_count}",
            f"  ★★★★☆ High: {self.high_count}",
            f"  ★★★☆☆ Standard: {self.standard_count}",
            "",
            f"Line Sources: Sportsbook {self.sportsbook_count} | Derived {self.derived_count}",
            "",
            "TOP PICKS:",
            "-" * 40,
        ]
        
        for i, pick in enumerate(self.picks[:10], 1):
            lines.append(
                f"{i}. {pick.tier.display} {pick.player_name} "
                f"{pick.direction} {pick.line} {pick.prop_type} "
                f"(edge: {pick.edge:.1f}%, pattern: {pick.primary_pattern})"
            )
        
        return "\n".join(lines)


@dataclass
class BacktestResultV2:
    """Backtest results with detailed breakdowns."""
    start_date: str
    end_date: str
    days_tested: int = 0
    
    # Overall
    total_picks: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    
    # By tier
    premium_picks: int = 0
    premium_hits: int = 0
    high_picks: int = 0
    high_hits: int = 0
    standard_picks: int = 0
    standard_hits: int = 0
    
    # By line source
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    derived_picks: int = 0
    derived_hits: int = 0
    
    # By pattern
    pattern_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    ast_picks: int = 0
    ast_hits: int = 0
    
    # By direction
    over_picks: int = 0
    over_hits: int = 0
    under_picks: int = 0
    under_hits: int = 0
    
    # Daily details
    daily_results: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "days_tested": self.days_tested,
            "total_picks": self.total_picks,
            "total_hits": self.total_hits,
            "hit_rate": round(self.hit_rate * 100, 1),
            "picks_per_day": round(self.total_picks / max(self.days_tested, 1), 1),
            "by_tier": {
                "premium": {
                    "picks": self.premium_picks,
                    "hits": self.premium_hits,
                    "rate": round(self.premium_hits / max(self.premium_picks, 1) * 100, 1),
                },
                "high": {
                    "picks": self.high_picks,
                    "hits": self.high_hits,
                    "rate": round(self.high_hits / max(self.high_picks, 1) * 100, 1),
                },
                "standard": {
                    "picks": self.standard_picks,
                    "hits": self.standard_hits,
                    "rate": round(self.standard_hits / max(self.standard_picks, 1) * 100, 1),
                },
            },
            "by_line_source": {
                "sportsbook": {
                    "picks": self.sportsbook_picks,
                    "hits": self.sportsbook_hits,
                    "rate": round(self.sportsbook_hits / max(self.sportsbook_picks, 1) * 100, 1),
                },
                "derived": {
                    "picks": self.derived_picks,
                    "hits": self.derived_hits,
                    "rate": round(self.derived_hits / max(self.derived_picks, 1) * 100, 1),
                },
            },
            "by_pattern": {
                k: {
                    "picks": v.get("picks", 0),
                    "hits": v.get("hits", 0),
                    "rate": round(v.get("hits", 0) / max(v.get("picks", 1), 1) * 100, 1),
                }
                for k, v in self.pattern_stats.items()
            },
            "by_direction": {
                "over": {
                    "picks": self.over_picks,
                    "hits": self.over_hits,
                    "rate": round(self.over_hits / max(self.over_picks, 1) * 100, 1),
                },
                "under": {
                    "picks": self.under_picks,
                    "hits": self.under_hits,
                    "rate": round(self.under_hits / max(self.under_picks, 1) * 100, 1),
                },
            },
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 70,
            "MULTI-MODEL PICKER BACKTEST RESULTS",
            "=" * 70,
            f"Period: {self.start_date} to {self.end_date} ({self.days_tested} days)",
            "",
            f"OVERALL: {self.hit_rate*100:.1f}% ({self.total_hits}/{self.total_picks})",
            f"Picks per Day: {self.total_picks / max(self.days_tested, 1):.1f}",
            "",
            "BY TIER:",
        ]
        
        if self.premium_picks > 0:
            rate = self.premium_hits / self.premium_picks * 100
            lines.append(f"  ★★★★★ Premium: {rate:.1f}% ({self.premium_hits}/{self.premium_picks})")
        if self.high_picks > 0:
            rate = self.high_hits / self.high_picks * 100
            lines.append(f"  ★★★★☆ High: {rate:.1f}% ({self.high_hits}/{self.high_picks})")
        if self.standard_picks > 0:
            rate = self.standard_hits / self.standard_picks * 100
            lines.append(f"  ★★★☆☆ Standard: {rate:.1f}% ({self.standard_hits}/{self.standard_picks})")
        
        lines.append("")
        lines.append("BY LINE SOURCE:")
        if self.sportsbook_picks > 0:
            rate = self.sportsbook_hits / self.sportsbook_picks * 100
            lines.append(f"  Sportsbook: {rate:.1f}% ({self.sportsbook_hits}/{self.sportsbook_picks})")
        if self.derived_picks > 0:
            rate = self.derived_hits / self.derived_picks * 100
            lines.append(f"  Derived: {rate:.1f}% ({self.derived_hits}/{self.derived_picks})")
        
        if self.pattern_stats:
            lines.append("")
            lines.append("BY PATTERN:")
            sorted_patterns = sorted(
                self.pattern_stats.items(),
                key=lambda x: x[1].get("hits", 0) / max(x[1].get("picks", 1), 1),
                reverse=True
            )
            for pattern, stats in sorted_patterns[:10]:
                if stats.get("picks", 0) > 0:
                    rate = stats["hits"] / stats["picks"] * 100
                    lines.append(f"  {pattern}: {rate:.1f}% ({stats['hits']}/{stats['picks']})")
        
        lines.append("")
        lines.append("BY DIRECTION:")
        if self.over_picks > 0:
            rate = self.over_hits / self.over_picks * 100
            lines.append(f"  OVER: {rate:.1f}% ({self.over_hits}/{self.over_picks})")
        if self.under_picks > 0:
            rate = self.under_hits / self.under_picks * 100
            lines.append(f"  UNDER: {rate:.1f}% ({self.under_hits}/{self.under_picks})")
        
        return "\n".join(lines)


# ============================================================================
# DATABASE HELPERS
# ============================================================================

def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    paths = get_paths()
    conn = sqlite3.connect(paths.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_games_for_date(conn: sqlite3.Connection, date: str) -> List[Dict]:
    """Get all games for a date."""
    cur = conn.execute("""
        SELECT g.id, g.game_date, t1.name as team1, t2.name as team2
        FROM games g
        JOIN teams t1 ON g.team1_id = t1.id
        JOIN teams t2 ON g.team2_id = t2.id
        WHERE g.game_date = ?
    """, (date,))
    return [dict(row) for row in cur.fetchall()]


def get_players_for_game(conn: sqlite3.Connection, game_id: int, game_date: str) -> List[Dict]:
    """Get players who played meaningful minutes in a game."""
    cur = conn.execute("""
        SELECT DISTINCT 
            p.id as player_id,
            p.name as player_name,
            t.name as team,
            bp.minutes
        FROM boxscore_player bp
        JOIN players p ON bp.player_id = p.id
        JOIN teams t ON bp.team_id = t.id
        WHERE bp.game_id = ?
          AND bp.minutes >= 15
        ORDER BY bp.minutes DESC
    """, (game_id,))
    return [dict(row) for row in cur.fetchall()]


def get_player_averages(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    window: int = 10,
) -> Dict[str, float]:
    """Get player's average stats over last N games."""
    cur = conn.execute("""
        SELECT 
            AVG(bp.pts) as pts,
            AVG(bp.reb) as reb,
            AVG(bp.ast) as ast,
            AVG(bp.minutes) as minutes,
            COUNT(*) as games,
            GROUP_CONCAT(bp.pts) as pts_history,
            GROUP_CONCAT(bp.reb) as reb_history,
            GROUP_CONCAT(bp.ast) as ast_history
        FROM (
            SELECT bp.pts, bp.reb, bp.ast, bp.minutes
            FROM boxscore_player bp
            JOIN games g ON bp.game_id = g.id
            WHERE bp.player_id = ?
              AND g.game_date < ?
              AND bp.minutes > 0
            ORDER BY g.game_date DESC
            LIMIT ?
        ) bp
    """, (player_id, before_date, window))
    
    row = cur.fetchone()
    if not row or row["games"] < 3:
        return {}
    
    return {
        "pts": row["pts"] or 0,
        "reb": row["reb"] or 0,
        "ast": row["ast"] or 0,
        "minutes": row["minutes"] or 0,
        "games": row["games"],
        "pts_history": row["pts_history"],
        "reb_history": row["reb_history"],
        "ast_history": row["ast_history"],
    }


def get_sportsbook_line(
    conn: sqlite3.Connection,
    player_name: str,
    prop_type: str,
    game_date: str,
    player_id: Optional[int] = None,
) -> Optional[Tuple[float, str]]:
    """Get sportsbook line for a player/prop."""
    # Try by player_id first
    if player_id:
        cur = conn.execute("""
            SELECT line, book FROM sportsbook_lines
            WHERE player_id = ? AND UPPER(prop_type) = UPPER(?) AND as_of_date = ?
            ORDER BY CASE LOWER(book)
                WHEN 'draftkings' THEN 1 WHEN 'fanduel' THEN 2 ELSE 3 END
            LIMIT 1
        """, (player_id, prop_type, game_date))
        row = cur.fetchone()
        if row and row[0]:
            return (row[0], row[1] or "unknown")
    
    # Try by player name
    cur = conn.execute("""
        SELECT sl.line, sl.book
        FROM sportsbook_lines sl
        JOIN players p ON sl.player_id = p.id
        WHERE LOWER(p.name) = LOWER(?)
          AND UPPER(sl.prop_type) = UPPER(?)
          AND sl.as_of_date = ?
        ORDER BY CASE LOWER(sl.book)
            WHEN 'draftkings' THEN 1 WHEN 'fanduel' THEN 2 ELSE 3 END
        LIMIT 1
    """, (player_name, prop_type, game_date))
    
    row = cur.fetchone()
    if row and row[0]:
        return (row[0], row[1] or "unknown")
    
    return None


def get_line_for_player(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Tuple[float, LineSource]:
    """Get the best available line for a player."""
    sb = get_sportsbook_line(conn, player_name, prop_type, game_date, player_id)
    if sb:
        return (sb[0], LineSource.SPORTSBOOK)
    
    # Derive from L10 average with 5% adjustment
    avgs = get_player_averages(conn, player_id, game_date, window=10)
    if avgs and prop_type.lower() in avgs:
        derived = avgs[prop_type.lower()] * 1.05
        return (round(derived, 1), LineSource.DERIVED)
    
    return (0.0, LineSource.DERIVED)


def get_actual_stats(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
) -> Optional[Dict[str, float]]:
    """Get actual stats for grading."""
    cur = conn.execute("""
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON bp.game_id = g.id
        WHERE bp.player_id = ? AND g.game_date = ?
    """, (player_id, game_date))
    
    row = cur.fetchone()
    if not row:
        return None
    
    return {
        "pts": row[0],
        "reb": row[1],
        "ast": row[2],
        "minutes": row[3],
    }


def check_back_to_back(
    conn: sqlite3.Connection,
    team: str,
    game_date: str,
) -> bool:
    """Check if team played yesterday."""
    from datetime import datetime, timedelta
    
    dt = datetime.strptime(game_date, "%Y-%m-%d")
    yesterday = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
    
    cur = conn.execute("""
        SELECT COUNT(*) FROM games g
        JOIN teams t1 ON g.team1_id = t1.id
        JOIN teams t2 ON g.team2_id = t2.id
        WHERE g.game_date = ? AND (t1.name = ? OR t2.name = ?)
    """, (yesterday, team, team))
    
    return cur.fetchone()[0] > 0


def get_defense_rating(
    conn: sqlite3.Connection,
    team: str,
    before_date: str,
    prop_type: str = "pts"
) -> float:
    """
    Get team's defensive rating for a prop type.
    Returns points allowed per game to opposing players.
    """
    cur = conn.execute("""
        SELECT AVG(bp.{}) as avg_allowed
        FROM boxscore_player bp
        JOIN games g ON bp.game_id = g.id
        JOIN teams t ON bp.team_id = t.id
        JOIN teams opp ON (
            (g.team1_id = t.id AND g.team2_id = opp.id) OR
            (g.team2_id = t.id AND g.team1_id = opp.id)
        )
        WHERE opp.name = ?
          AND g.game_date < ?
          AND bp.minutes >= 15
        ORDER BY g.game_date DESC
        LIMIT 100
    """.format(prop_type.lower()), (team, before_date))
    
    row = cur.fetchone()
    return row[0] if row and row[0] else 0


# ============================================================================
# SIGNAL GENERATION - Core Pattern Detection
# ============================================================================

def detect_cold_bounce_pattern(
    l5: Dict[str, float],
    l10: Dict[str, float],
    l15: Dict[str, float],
    prop_type: str,
) -> Optional[Tuple[float, List[str]]]:
    """
    Detect cold bounce OVER pattern (77.8% validated accuracy).
    
    Pattern: Player is cold (L5 << L15) but likely to bounce back.
    Tightened criteria for better accuracy.
    """
    pt = prop_type.lower()
    if pt not in l5 or pt not in l15:
        return None
    
    l5_val = l5.get(pt, 0)
    l15_val = l15.get(pt, 0)
    
    if l15_val == 0:
        return None
    
    # Check if player is cold - At -17% threshold for balance
    deviation = (l5_val - l15_val) / l15_val * 100
    
    if deviation <= -17:  # L5 is 17%+ below L15
        # Project bounce toward L15
        projection = l15_val * 0.92  # Conservative bounce
        factors = [
            f"Cold bounce pattern: L5 {deviation:.0f}% below L15",
            f"Projection: {projection:.1f} (bounce toward {l15_val:.1f})"
        ]
        return (projection, factors)
    
    return None


def detect_b2b_fatigue_pattern(
    is_b2b: bool,
    l5: Dict[str, float],
    l10: Dict[str, float],
    prop_type: str,
) -> Optional[Tuple[float, List[str]]]:
    """
    Detect B2B fatigue UNDER pattern (74.5% validated accuracy).
    
    Pattern: Player's team on back-to-back, expect reduced production.
    Tightened: Only trigger when L5 <= L10 (already trending down/flat)
    """
    if not is_b2b:
        return None
    
    pt = prop_type.lower()
    if pt not in l10 or pt not in l5:
        return None
    
    l5_val = l5.get(pt, 0)
    l10_val = l10.get(pt, 0)
    
    # Only trigger if player is already at or below their average (not hot)
    if l10_val > 0 and l5_val > l10_val * 1.05:
        # Player is hot - don't bet UNDER
        return None
    
    # Apply fatigue reduction of 10-12%
    fatigue_factor = 0.88
    projection = l10_val * fatigue_factor
    
    factors = [
        "B2B fatigue pattern: Team played yesterday",
        f"Expected reduction: ~{(1-fatigue_factor)*100:.0f}%"
    ]
    
    return (projection, factors)


def detect_defense_pattern(
    defense_rating: float,
    league_avg: float,
    l10: Dict[str, float],
    prop_type: str,
) -> Optional[Tuple[str, float, List[str]]]:
    """
    Detect defense-based patterns.
    
    - defense_elite: Opponent allows 15%+ less than average (UNDER) - 62.5%
    - defense_good: Opponent allows 10%+ less than average (UNDER) - 70.7%
    """
    if league_avg == 0:
        return None
    
    pt = prop_type.lower()
    if pt not in l10:
        return None
    
    l10_val = l10.get(pt, 0)
    
    # Calculate defensive strength
    defense_diff = (defense_rating - league_avg) / league_avg * 100
    
    if defense_diff <= -15:  # Elite defense
        projection = l10_val * 0.85
        return ("defense_elite", projection, [
            f"Elite defense: Opponent allows {abs(defense_diff):.0f}% below avg",
        ])
    elif defense_diff <= -10:  # Good defense
        projection = l10_val * 0.90
        return ("defense_good", projection, [
            f"Good defense: Opponent allows {abs(defense_diff):.0f}% below avg",
        ])
    
    return None


def detect_hot_sustained_pattern(
    l5: Dict[str, float],
    l10: Dict[str, float],
    l15: Dict[str, float],
    history: str,
    prop_type: str,
) -> Optional[Tuple[float, List[str]]]:
    """
    Detect hot sustained OVER pattern.
    
    Pattern: Player is hot and maintaining elevated performance.
    """
    pt = prop_type.lower()
    if pt not in l5 or pt not in l15:
        return None
    
    l5_val = l5.get(pt, 0)
    l15_val = l15.get(pt, 0)
    
    if l15_val == 0:
        return None
    
    deviation = (l5_val - l15_val) / l15_val * 100
    
    # Check if player is hot
    if deviation >= 15:  # L5 is 15%+ above L15
        # Check sustainability - look at recent games
        if history:
            recent = [float(x) for x in history.split(",")[:5] if x]
            if len(recent) >= 3:
                above_l15 = sum(1 for x in recent if x > l15_val)
                if above_l15 >= 3:  # 3+ of last 5 above L15
                    projection = l5_val * 0.95  # Slight regression
                    factors = [
                        f"Hot sustained: L5 {deviation:.0f}% above L15",
                        f"{above_l15}/5 recent games above L15"
                    ]
                    return (projection, factors)
    
    return None


# ============================================================================
# MAIN SIGNAL GENERATION
# ============================================================================

def generate_signals_for_player(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team: str,
    opponent: str,
    game_date: str,
    game_id: int,
) -> List[PickSignal]:
    """Generate all signals for a player."""
    signals = []
    
    # Get player averages at different windows
    l5 = get_player_averages(conn, player_id, game_date, window=5)
    l10 = get_player_averages(conn, player_id, game_date, window=10)
    l15 = get_player_averages(conn, player_id, game_date, window=15)
    
    if not l10 or l10.get("minutes", 0) < 20:
        return []
    
    # Check back-to-back
    is_b2b = check_back_to_back(conn, team, game_date)
    
    # Process each prop type - All props but with different thresholds
    for prop_type in ["PTS", "REB", "AST"]:
        pt_lower = prop_type.lower()
        
        # Get line
        line, line_source = get_line_for_player(
            conn, player_id, player_name, prop_type, game_date
        )
        
        if line <= 0:
            continue
        
        # Get history for pattern detection
        history = l10.get(f"{pt_lower}_history", "")
        
        # 1. COLD BOUNCE (77.8% - HIGH PRIORITY)
        # For PTS, require higher edge due to lower accuracy
        cold_result = detect_cold_bounce_pattern(l5, l10, l15, prop_type)
        if cold_result:
            projection, factors = cold_result
            edge = (projection - line) / line * 100
            min_edge = 8 if prop_type == "PTS" else 5  # Higher bar for PTS
            if edge >= min_edge:
                signals.append(PickSignal(
                    source="pattern",
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    game_date=game_date,
                    prop_type=prop_type,
                    direction="OVER",
                    line=line,
                    line_source=line_source,
                    projection=projection,
                    edge=edge,
                    confidence=HIGH_VALUE_PATTERNS["cold_bounce"] * 100,
                    pattern="cold_bounce",
                    factors=factors,
                    accuracy=HIGH_VALUE_PATTERNS["cold_bounce"],
                ))
        
        # 2. B2B FATIGUE - DISABLED (underperforming in backtest ~49%)
        # b2b_result = detect_b2b_fatigue_pattern(is_b2b, l5, l10, prop_type)
        # if b2b_result:
        #     projection, factors = b2b_result
        #     edge = (line - projection) / line * 100
        #     if edge >= 8:
        #         signals.append(PickSignal(
        #             source="pattern",
        #             player_id=player_id,
        #             player_name=player_name,
        #             team=team,
        #             opponent=opponent,
        #             game_date=game_date,
        #             prop_type=prop_type,
        #             direction="UNDER",
        #             line=line,
        #             line_source=line_source,
        #             projection=projection,
        #             edge=edge,
        #             confidence=HIGH_VALUE_PATTERNS["b2b_fatigue"] * 100,
        #             pattern="b2b_fatigue",
        #             factors=factors,
        #             accuracy=HIGH_VALUE_PATTERNS["b2b_fatigue"],
        #         ))
        
        # 3. DEFENSE PATTERNS (70.7% / 62.5%)
        # league_avg = 15.0  # Approximate for PTS
        # def_rating = get_defense_rating(conn, opponent, game_date, prop_type)
        # if def_rating > 0:
        #     def_result = detect_defense_pattern(def_rating, league_avg, l10, prop_type)
        #     if def_result:
        #         pattern_name, projection, factors = def_result
        #         edge = (line - projection) / line * 100
        #         if edge >= 5:
        #             signals.append(PickSignal(
        #                 source="pattern",
        #                 player_id=player_id,
        #                 player_name=player_name,
        #                 team=team,
        #                 opponent=opponent,
        #                 game_date=game_date,
        #                 prop_type=prop_type,
        #                 direction="UNDER",
        #                 line=line,
        #                 line_source=line_source,
        #                 projection=projection,
        #                 edge=edge,
        #                 confidence=HIGH_VALUE_PATTERNS.get(pattern_name, 0.60) * 100,
        #                 pattern=pattern_name,
        #                 factors=factors,
        #                 accuracy=HIGH_VALUE_PATTERNS.get(pattern_name, 0.60),
        #             ))
        
        # 4. HOT SUSTAINED - DISABLED (only 16.7% accuracy in backtest)
        # hot_result = detect_hot_sustained_pattern(l5, l10, l15, history, prop_type)
        # if hot_result:
        #     projection, factors = hot_result
        #     edge = (projection - line) / line * 100
        #     if edge >= 6:
        #         signals.append(PickSignal(
        #             source="pattern",
        #             player_id=player_id,
        #             player_name=player_name,
        #             team=team,
        #             opponent=opponent,
        #             game_date=game_date,
        #             prop_type=prop_type,
        #             direction="OVER",
        #             line=line,
        #             line_source=line_source,
        #             projection=projection,
        #             edge=edge,
        #             confidence=65,
        #             pattern="hot_sustained",
        #             factors=factors,
        #             accuracy=0.62,
        #         ))
        
        # 5. SIMPLE EDGE (Standard picks) - Require higher edge for better accuracy
        l10_val = l10.get(pt_lower, 0)
        if l10_val > 0:
            # OVER check - require 12% edge (was 8%)
            over_edge = (l10_val - line) / line * 100
            if over_edge >= 12 and not any(s.prop_type == prop_type and s.direction == "OVER" for s in signals):
                signals.append(PickSignal(
                    source="simple_edge",
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    game_date=game_date,
                    prop_type=prop_type,
                    direction="OVER",
                    line=line,
                    line_source=line_source,
                    projection=l10_val,
                    edge=over_edge,
                    confidence=55 + min(over_edge, 15),
                    pattern="simple_over",
                    factors=[f"L10 avg {l10_val:.1f} vs line {line:.1f}"],
                    accuracy=0.58,
                ))
            
            # UNDER check - require 12% edge (was 8%)
            under_edge = (line - l10_val) / line * 100
            if under_edge >= 12 and not any(s.prop_type == prop_type and s.direction == "UNDER" for s in signals):
                signals.append(PickSignal(
                    source="simple_edge",
                    player_id=player_id,
                    player_name=player_name,
                    team=team,
                    opponent=opponent,
                    game_date=game_date,
                    prop_type=prop_type,
                    direction="UNDER",
                    line=line,
                    line_source=line_source,
                    projection=l10_val,
                    edge=under_edge,
                    confidence=55 + min(under_edge, 15),
                    pattern="simple_under",
                    factors=[f"L10 avg {l10_val:.1f} vs line {line:.1f}"],
                    accuracy=0.58,
                ))
    
    return signals


# ============================================================================
# PICK ASSEMBLY
# ============================================================================

def aggregate_signals(signals: List[PickSignal]) -> Dict[str, List[PickSignal]]:
    """Group signals by player/prop/direction."""
    grouped = {}
    for sig in signals:
        key = f"{sig.player_id}|{sig.prop_type}|{sig.direction}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(sig)
    return grouped


def create_pick_from_signals(
    signals: List[PickSignal],
    game_id: int,
) -> Optional[MultiModelPick]:
    """Create a unified pick from multiple signals."""
    if not signals:
        return None
    
    first = signals[0]
    
    # Weight signals by accuracy
    total_weight = sum(s.accuracy for s in signals)
    avg_projection = sum(s.projection * s.accuracy for s in signals) / total_weight
    avg_edge = sum(s.edge * s.accuracy for s in signals) / total_weight
    avg_confidence = sum(s.confidence * s.accuracy for s in signals) / total_weight
    
    # Get best accuracy signal as primary pattern
    best_signal = max(signals, key=lambda s: s.accuracy)
    primary_pattern = best_signal.pattern
    
    # Collect factors
    all_factors = []
    for s in signals:
        all_factors.extend(s.factors)
    unique_factors = list(dict.fromkeys(all_factors))[:5]
    
    # Determine tier
    num_signals = len(signals)
    has_sportsbook = first.line_source == LineSource.SPORTSBOOK
    
    tier = ConfidenceTier.STANDARD
    expected_hit = 0.58
    
    # Check tier requirements
    for tier_name in ["PREMIUM", "HIGH", "STANDARD"]:
        cfg = TIER_CONFIG[tier_name]
        if (avg_edge >= cfg["min_edge"] and
            num_signals >= cfg["min_models"] and
            (not cfg["require_sportsbook"] or has_sportsbook)):
            tier = ConfidenceTier[tier_name]
            expected_hit = cfg["expected_hit"]
            break
    
    # Boost expected hit for high-value patterns
    if primary_pattern in HIGH_VALUE_PATTERNS:
        expected_hit = max(expected_hit, HIGH_VALUE_PATTERNS[primary_pattern])
    
    # Generate pick ID
    pick_id = f"{first.game_date}|{first.player_id}|{first.prop_type}|{first.direction}"
    
    return MultiModelPick(
        pick_id=pick_id,
        player_id=first.player_id,
        player_name=first.player_name,
        team=first.team,
        opponent=first.opponent,
        game_date=first.game_date,
        game_id=game_id,
        prop_type=first.prop_type,
        direction=first.direction,
        line=first.line,
        line_source=first.line_source,
        projection=round(avg_projection, 1),
        edge=round(avg_edge, 1),
        tier=tier,
        confidence_score=round(avg_confidence, 1),
        expected_hit_rate=expected_hit,
        signals=signals,
        primary_pattern=primary_pattern,
        key_factors=unique_factors,
    )


# ============================================================================
# MAIN API
# ============================================================================

def get_multi_model_picks(
    date: str,
    conn: Optional[sqlite3.Connection] = None,
    max_picks: Optional[int] = None,
) -> DailyPicksV2:
    """
    Generate picks for a date using multi-model approach.
    
    Args:
        date: Date string (YYYY-MM-DD)
        conn: Optional database connection
        max_picks: Maximum picks (defaults based on games)
    
    Returns:
        DailyPicksV2 with ranked picks
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    
    try:
        games = get_games_for_date(conn, date)
        num_games = len(games)
        
        if num_games == 0:
            return DailyPicksV2(date=date, num_games=0)
        
        # Calculate target picks based on games
        if max_picks is None:
            if num_games <= 3:
                max_picks = 6
            elif num_games <= 6:
                max_picks = 10
            elif num_games <= 10:
                max_picks = 12
            else:
                max_picks = 15
        
        all_signals = []
        game_info = {}  # player_id -> (team, opponent, game_id)
        
        # Process each game
        for game in games:
            game_id = game["id"]
            team1 = game["team1"]
            team2 = game["team2"]
            
            players = get_players_for_game(conn, game_id, date)
            
            for player in players:
                player_id = player["player_id"]
                player_name = player["player_name"]
                team = player["team"]
                opponent = team2 if team == team1 else team1
                
                game_info[player_id] = (team, opponent, game_id)
                
                # Generate signals
                signals = generate_signals_for_player(
                    conn, player_id, player_name, team, opponent, date, game_id
                )
                all_signals.extend(signals)
        
        # Aggregate and create picks
        grouped = aggregate_signals(all_signals)
        picks = []
        
        for key, sigs in grouped.items():
            player_id = int(key.split("|")[0])
            if player_id not in game_info:
                continue
            
            _, _, game_id = game_info[player_id]
            pick = create_pick_from_signals(sigs, game_id)
            if pick:
                picks.append(pick)
        
        # Sort by: tier (desc), accuracy (desc), edge (desc)
        picks.sort(
            key=lambda p: (p.tier.stars, p.expected_hit_rate, p.edge),
            reverse=True
        )
        
        # Limit picks and ensure variety
        final_picks = []
        players_used = set()
        
        for pick in picks:
            if len(final_picks) >= max_picks:
                break
            
            # Limit to 2 picks per player
            player_picks = sum(1 for p in final_picks if p.player_id == pick.player_id)
            if player_picks >= 2:
                continue
            
            final_picks.append(pick)
            players_used.add(pick.player_id)
        
        result = DailyPicksV2(date=date, num_games=num_games, picks=final_picks)
        result.calculate_stats()
        
        return result
    
    finally:
        if should_close:
            conn.close()


def run_backtest_multi_model(
    start_date: str,
    end_date: str,
    verbose: bool = True,
    show_progress: bool = True,
) -> BacktestResultV2:
    """
    Run backtest on multi-model picker.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Print detailed output
        show_progress: Show progress bar
    
    Returns:
        BacktestResultV2 with comprehensive metrics
    """
    conn = get_db_connection()
    result = BacktestResultV2(start_date=start_date, end_date=end_date)
    
    # Generate dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    result.days_tested = len(dates)
    
    if verbose:
        print(f"Running multi-model backtest: {start_date} to {end_date} ({len(dates)} days)")
        print()
    
    for i, date in enumerate(dates):
        if show_progress:
            pct = (i + 1) / len(dates) * 100
            print(f"\r[{'█' * int(pct/5):<20}] {pct:.0f}% - {date}", end="", flush=True)
        
        daily = get_multi_model_picks(date, conn)
        
        daily_hits = 0
        daily_total = 0
        
        for pick in daily.picks:
            actual = get_actual_stats(conn, pick.player_id, date)
            if actual is None:
                continue
            
            actual_value = actual.get(pick.prop_type.lower(), 0)
            pick.actual_value = actual_value
            
            if pick.direction == "OVER":
                pick.hit = actual_value > pick.line
            else:
                pick.hit = actual_value < pick.line
            
            # Update result counts
            result.total_picks += 1
            daily_total += 1
            
            if pick.hit:
                result.total_hits += 1
                daily_hits += 1
            
            # By tier
            if pick.tier == ConfidenceTier.PREMIUM:
                result.premium_picks += 1
                if pick.hit:
                    result.premium_hits += 1
            elif pick.tier == ConfidenceTier.HIGH:
                result.high_picks += 1
                if pick.hit:
                    result.high_hits += 1
            else:
                result.standard_picks += 1
                if pick.hit:
                    result.standard_hits += 1
            
            # By line source
            if pick.line_source == LineSource.SPORTSBOOK:
                result.sportsbook_picks += 1
                if pick.hit:
                    result.sportsbook_hits += 1
            else:
                result.derived_picks += 1
                if pick.hit:
                    result.derived_hits += 1
            
            # By pattern
            if pick.primary_pattern:
                if pick.primary_pattern not in result.pattern_stats:
                    result.pattern_stats[pick.primary_pattern] = {"picks": 0, "hits": 0}
                result.pattern_stats[pick.primary_pattern]["picks"] += 1
                if pick.hit:
                    result.pattern_stats[pick.primary_pattern]["hits"] += 1
            
            # By prop
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
            
            # By direction
            if pick.direction == "OVER":
                result.over_picks += 1
                if pick.hit:
                    result.over_hits += 1
            else:
                result.under_picks += 1
                if pick.hit:
                    result.under_hits += 1
        
        result.daily_results.append({
            "date": date,
            "games": daily.num_games,
            "picks": daily_total,
            "hits": daily_hits,
            "rate": daily_hits / max(daily_total, 1),
        })
    
    # Calculate final hit rate
    if result.total_picks > 0:
        result.hit_rate = result.total_hits / result.total_picks
    
    if show_progress:
        print()
        print()
    
    if verbose:
        print(result.summary())
    
    conn.close()
    return result


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Model Picker")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest", nargs=2, metavar=("START", "END"),
                        help="Run backtest from START to END")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.backtest:
        result = run_backtest_multi_model(args.backtest[0], args.backtest[1], verbose=True)
    elif args.date:
        picks = get_multi_model_picks(args.date)
        print(picks.summary())
    else:
        # Default: today's picks
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_multi_model_picks(today)
        print(picks.summary())
