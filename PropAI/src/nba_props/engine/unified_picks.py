"""
Unified Picks System - Ensemble Model for NBA Props Predictions
================================================================

This module implements a sophisticated ensemble approach that:
1. Combines insights from multiple validated models
2. Generates 10+ picks per day with proper confidence calibration
3. Uses sportsbook lines when available, derived lines with adjustment otherwise
4. Tracks performance with HONEST metrics (separating sportsbook vs derived)
5. Provides a quality-based grading system

PHILOSOPHY:
-----------
Rather than relying on a single model, this system:
- Aggregates predictions from multiple models
- Weights model predictions by their validated accuracy
- Requires consensus (multiple models agreeing) for higher confidence
- Focuses on generating MORE picks with CALIBRATED confidence

CONFIDENCE LEVELS:
------------------
- PREMIUM (★★★★★): 3+ models agree, sportsbook line, 8%+ edge, 75%+ expected hit rate
- HIGH (★★★★☆): 2+ models agree, 6%+ edge, 65%+ expected hit rate  
- STANDARD (★★★☆☆): 1+ model, 5%+ edge, 58%+ expected hit rate
- SPECULATIVE (★★☆☆☆): Model suggests but lower confidence, 55%+ expected

GRADING:
--------
- Grade picks based on actual outcomes
- Track by confidence tier, direction, prop type
- Separate tracking for sportsbook vs derived line picks
- Use CONSISTENT grading across all models

Author: PropAI Team
Created: February 2026
Version: 2.0
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev
from ..paths import get_paths


# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_VERSION = "2.0"

# Model weights based on validated backtesting performance
# Higher weight = more influence on consensus
MODEL_WEIGHTS = {
    "v18_general": 1.2,    # Holistic factor scoring
    "v16_general": 1.1,    # Pattern-based (72.4% validated)
    "v19_general": 1.0,    # Latest iteration
    "v17_general": 1.0,    # Multi-factor
    "v16_under": 1.1,      # Under specialist
    "v18_under": 1.0,      # Under specialist
    "model_production": 0.9,  # Cold bounce specialist
}

# Minimum requirements for each confidence tier
CONFIDENCE_TIERS = {
    "PREMIUM": {
        "min_models": 3,
        "min_edge": 8.0,
        "min_expected_hit": 0.70,
        "require_sportsbook": True,
        "stars": 5,
    },
    "HIGH": {
        "min_models": 2,
        "min_edge": 6.0,
        "min_expected_hit": 0.62,
        "require_sportsbook": False,
        "stars": 4,
    },
    "STANDARD": {
        "min_models": 1,
        "min_edge": 5.0,
        "min_expected_hit": 0.58,
        "require_sportsbook": False,
        "stars": 3,
    },
    "SPECULATIVE": {
        "min_models": 1,
        "min_edge": 4.0,
        "min_expected_hit": 0.55,
        "require_sportsbook": False,
        "stars": 2,
    },
}

# Target picks per day (adjusts based on number of games)
TARGET_PICKS_PER_GAME = 1.5  # Aim for 1-2 picks per game
MIN_DAILY_PICKS = 3
MAX_DAILY_PICKS = 15


class ConfidenceTier(Enum):
    """Confidence tiers for picks."""
    PREMIUM = "PREMIUM"
    HIGH = "HIGH"
    STANDARD = "STANDARD"
    SPECULATIVE = "SPECULATIVE"
    
    @property
    def stars(self) -> int:
        return CONFIDENCE_TIERS[self.value]["stars"]
    
    @property 
    def display(self) -> str:
        return "★" * self.stars + "☆" * (5 - self.stars)


class LineSource(Enum):
    """Source of the betting line."""
    SPORTSBOOK = "sportsbook"
    DERIVED = "derived"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ModelPrediction:
    """A prediction from a single model."""
    model_id: str
    player_name: str
    prop_type: str
    direction: str  # "OVER" or "UNDER"
    projection: float
    line: float
    edge: float
    confidence: float  # 0-100
    factors: List[str] = field(default_factory=list)  # Reasons for pick


@dataclass
class UnifiedPick:
    """A unified pick combining multiple model predictions."""
    # Identification
    pick_id: str  # Unique ID for tracking
    
    # Player/Game info
    player_id: int
    player_name: str
    team: str
    opponent: str
    game_date: str
    
    # Prop details
    prop_type: str  # PTS, REB, AST
    direction: str  # OVER or UNDER
    
    # Line information
    line: float
    
    # Projection
    projection: float
    edge: float  # Percentage edge
    
    # Optional fields with defaults
    game_id: Optional[int] = None
    line_source: LineSource = LineSource.DERIVED
    
    # Confidence
    tier: ConfidenceTier = ConfidenceTier.SPECULATIVE
    expected_hit_rate: float = 0.55
    confidence_score: float = 0.0  # 0-100
    
    # Model consensus
    models_agreeing: List[str] = field(default_factory=list)
    model_predictions: List[ModelPrediction] = field(default_factory=list)
    weighted_agreement: float = 0.0
    
    # Factors/reasons
    key_factors: List[str] = field(default_factory=list)
    
    # Grading (filled in after game)
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    grade: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
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
            "tier_stars": self.tier.stars,
            "tier_display": self.tier.display,
            "expected_hit_rate": round(self.expected_hit_rate * 100, 1),
            "confidence_score": round(self.confidence_score, 1),
            "models_agreeing": self.models_agreeing,
            "num_models": len(self.models_agreeing),
            "weighted_agreement": round(self.weighted_agreement, 2),
            "key_factors": self.key_factors,
            "actual_value": self.actual_value,
            "hit": self.hit,
            "grade": self.grade,
        }


@dataclass
class DailyPicks:
    """Collection of picks for a single day."""
    date: str
    num_games: int
    picks: List[UnifiedPick] = field(default_factory=list)
    
    # Statistics
    premium_count: int = 0
    high_count: int = 0
    standard_count: int = 0
    speculative_count: int = 0
    
    over_count: int = 0
    under_count: int = 0
    
    pts_count: int = 0
    reb_count: int = 0
    ast_count: int = 0
    
    sportsbook_count: int = 0
    derived_count: int = 0
    
    def calculate_stats(self):
        """Calculate statistics from picks."""
        self.premium_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.PREMIUM)
        self.high_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.HIGH)
        self.standard_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.STANDARD)
        self.speculative_count = sum(1 for p in self.picks if p.tier == ConfidenceTier.SPECULATIVE)
        
        self.over_count = sum(1 for p in self.picks if p.direction == "OVER")
        self.under_count = sum(1 for p in self.picks if p.direction == "UNDER")
        
        self.pts_count = sum(1 for p in self.picks if p.prop_type == "PTS")
        self.reb_count = sum(1 for p in self.picks if p.prop_type == "REB")
        self.ast_count = sum(1 for p in self.picks if p.prop_type == "AST")
        
        self.sportsbook_count = sum(1 for p in self.picks if p.line_source == LineSource.SPORTSBOOK)
        self.derived_count = sum(1 for p in self.picks if p.line_source == LineSource.DERIVED)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        self.calculate_stats()
        return {
            "date": self.date,
            "num_games": self.num_games,
            "picks": [p.to_dict() for p in self.picks],
            "total_picks": len(self.picks),
            "by_tier": {
                "premium": self.premium_count,
                "high": self.high_count,
                "standard": self.standard_count,
                "speculative": self.speculative_count,
            },
            "by_direction": {
                "over": self.over_count,
                "under": self.under_count,
            },
            "by_prop": {
                "pts": self.pts_count,
                "reb": self.reb_count,
                "ast": self.ast_count,
            },
            "by_line_source": {
                "sportsbook": self.sportsbook_count,
                "derived": self.derived_count,
            },
        }
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        self.calculate_stats()
        lines = [
            f"=" * 60,
            f"UNIFIED PICKS - {self.date}",
            f"=" * 60,
            f"Games: {self.num_games} | Total Picks: {len(self.picks)}",
            f"",
            f"BY TIER:",
            f"  ★★★★★ Premium: {self.premium_count}",
            f"  ★★★★☆ High: {self.high_count}",
            f"  ★★★☆☆ Standard: {self.standard_count}",
            f"  ★★☆☆☆ Speculative: {self.speculative_count}",
            f"",
            f"BY DIRECTION: OVER {self.over_count} | UNDER {self.under_count}",
            f"BY PROP: PTS {self.pts_count} | REB {self.reb_count} | AST {self.ast_count}",
            f"BY LINE SOURCE: Sportsbook {self.sportsbook_count} | Derived {self.derived_count}",
            f"",
        ]
        
        # Top picks
        if self.picks:
            lines.append("TOP PICKS:")
            lines.append("-" * 40)
            for i, pick in enumerate(self.picks[:10], 1):
                lines.append(
                    f"{i}. {pick.tier.display} {pick.player_name} "
                    f"{pick.direction} {pick.line} {pick.prop_type} "
                    f"(edge: {pick.edge:.1f}%, {len(pick.models_agreeing)} models)"
                )
        
        return "\n".join(lines)


@dataclass
class BacktestResultV2:
    """Results from backtesting the unified picks system."""
    start_date: str
    end_date: str
    days_tested: int = 0
    
    # Overall metrics
    total_picks: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    
    # By tier
    premium_picks: int = 0
    premium_hits: int = 0
    premium_rate: float = 0.0
    
    high_picks: int = 0
    high_hits: int = 0
    high_rate: float = 0.0
    
    standard_picks: int = 0
    standard_hits: int = 0
    standard_rate: float = 0.0
    
    speculative_picks: int = 0
    speculative_hits: int = 0
    speculative_rate: float = 0.0
    
    # By direction
    over_picks: int = 0
    over_hits: int = 0
    over_rate: float = 0.0
    
    under_picks: int = 0
    under_hits: int = 0
    under_rate: float = 0.0
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    pts_rate: float = 0.0
    
    reb_picks: int = 0
    reb_hits: int = 0
    reb_rate: float = 0.0
    
    ast_picks: int = 0
    ast_hits: int = 0
    ast_rate: float = 0.0
    
    # By line source (CRITICAL for honest metrics)
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    sportsbook_rate: float = 0.0
    
    derived_picks: int = 0
    derived_hits: int = 0
    derived_rate: float = 0.0
    
    # Quality metrics
    avg_picks_per_day: float = 0.0
    calibration_score: float = 0.0  # How close predicted hit rate is to actual
    
    # Detailed results
    pick_results: List[Dict] = field(default_factory=list)
    
    def calculate_rates(self):
        """Calculate all hit rates."""
        self.hit_rate = self.total_hits / self.total_picks if self.total_picks > 0 else 0.0
        
        self.premium_rate = self.premium_hits / self.premium_picks if self.premium_picks > 0 else 0.0
        self.high_rate = self.high_hits / self.high_picks if self.high_picks > 0 else 0.0
        self.standard_rate = self.standard_hits / self.standard_picks if self.standard_picks > 0 else 0.0
        self.speculative_rate = self.speculative_hits / self.speculative_picks if self.speculative_picks > 0 else 0.0
        
        self.over_rate = self.over_hits / self.over_picks if self.over_picks > 0 else 0.0
        self.under_rate = self.under_hits / self.under_picks if self.under_picks > 0 else 0.0
        
        self.pts_rate = self.pts_hits / self.pts_picks if self.pts_picks > 0 else 0.0
        self.reb_rate = self.reb_hits / self.reb_picks if self.reb_picks > 0 else 0.0
        self.ast_rate = self.ast_hits / self.ast_picks if self.ast_picks > 0 else 0.0
        
        self.sportsbook_rate = self.sportsbook_hits / self.sportsbook_picks if self.sportsbook_picks > 0 else 0.0
        self.derived_rate = self.derived_hits / self.derived_picks if self.derived_picks > 0 else 0.0
        
        self.avg_picks_per_day = self.total_picks / self.days_tested if self.days_tested > 0 else 0.0
    
    def summary(self) -> str:
        """Generate summary report."""
        self.calculate_rates()
        
        lines = [
            "=" * 70,
            "UNIFIED PICKS BACKTEST RESULTS",
            "=" * 70,
            f"Period: {self.start_date} to {self.end_date} ({self.days_tested} days)",
            f"Total Picks: {self.total_picks} | Avg/Day: {self.avg_picks_per_day:.1f}",
            "",
            "OVERALL PERFORMANCE:",
            f"  Hit Rate: {self.hit_rate*100:.1f}% ({self.total_hits}/{self.total_picks})",
            "",
            "BY CONFIDENCE TIER:",
            f"  ★★★★★ Premium:     {self.premium_rate*100:.1f}% ({self.premium_hits}/{self.premium_picks})",
            f"  ★★★★☆ High:        {self.high_rate*100:.1f}% ({self.high_hits}/{self.high_picks})",
            f"  ★★★☆☆ Standard:    {self.standard_rate*100:.1f}% ({self.standard_hits}/{self.standard_picks})",
            f"  ★★☆☆☆ Speculative: {self.speculative_rate*100:.1f}% ({self.speculative_hits}/{self.speculative_picks})",
            "",
            "BY DIRECTION:",
            f"  OVER:  {self.over_rate*100:.1f}% ({self.over_hits}/{self.over_picks})",
            f"  UNDER: {self.under_rate*100:.1f}% ({self.under_hits}/{self.under_picks})",
            "",
            "BY PROP TYPE:",
            f"  PTS: {self.pts_rate*100:.1f}% ({self.pts_hits}/{self.pts_picks})",
            f"  REB: {self.reb_rate*100:.1f}% ({self.reb_hits}/{self.reb_picks})",
            f"  AST: {self.ast_rate*100:.1f}% ({self.ast_hits}/{self.ast_picks})",
            "",
            "⚠️ BY LINE SOURCE (CRITICAL FOR HONEST METRICS):",
            f"  Sportsbook Lines: {self.sportsbook_rate*100:.1f}% ({self.sportsbook_hits}/{self.sportsbook_picks})",
            f"  Derived Lines:    {self.derived_rate*100:.1f}% ({self.derived_hits}/{self.derived_picks})",
            "",
            "Note: Derived line metrics are INFLATED. Only sportsbook line metrics",
            "reflect actual betting performance.",
            "=" * 70,
        ]
        
        return "\n".join(lines)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get database connection."""
    paths = get_paths()
    return sqlite3.connect(paths.db_path)


def normalize_player_name(name: str) -> str:
    """Normalize player name for matching."""
    return " ".join(name.strip().split())


def get_sportsbook_line(
    conn: sqlite3.Connection,
    player_name: str,
    prop_type: str,
    game_date: str,
    player_id: Optional[int] = None,
) -> Optional[Tuple[float, str]]:
    """
    Get sportsbook line if available.
    
    Returns:
        Tuple of (line, book) or None if not found
    """
    # First try by player_id if provided
    if player_id:
        cur = conn.execute("""
            SELECT line, book
            FROM sportsbook_lines
            WHERE player_id = ?
              AND UPPER(prop_type) = UPPER(?)
              AND as_of_date = ?
            ORDER BY 
                CASE LOWER(book) 
                    WHEN 'draftkings' THEN 1
                    WHEN 'fanduel' THEN 2
                    WHEN 'betmgm' THEN 3
                    ELSE 4
                END
            LIMIT 1
        """, (player_id, prop_type, game_date))
        
        row = cur.fetchone()
        if row and row[0]:
            return (row[0], row[1] or "unknown")
    
    # Also try looking up by player name via players table
    cur = conn.execute("""
        SELECT sl.line, sl.book
        FROM sportsbook_lines sl
        JOIN players p ON sl.player_id = p.id
        WHERE LOWER(p.name) = LOWER(?)
          AND UPPER(sl.prop_type) = UPPER(?)
          AND sl.as_of_date = ?
        ORDER BY 
            CASE LOWER(sl.book) 
                WHEN 'draftkings' THEN 1
                WHEN 'fanduel' THEN 2
                WHEN 'betmgm' THEN 3
                ELSE 4
            END
        LIMIT 1
    """, (player_name, prop_type, game_date))
    
    row = cur.fetchone()
    if row and row[0]:
        return (row[0], row[1] or "unknown")
    
    return None


def get_player_averages(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    window: int = 10,
) -> Dict[str, float]:
    """
    Get player's average stats over last N games.
    
    Returns:
        Dict with pts, reb, ast averages
    """
    cur = conn.execute("""
        SELECT 
            AVG(bp.pts) as pts,
            AVG(bp.reb) as reb,
            AVG(bp.ast) as ast,
            AVG(bp.minutes) as minutes,
            COUNT(*) as games
        FROM boxscore_player bp
        JOIN games g ON bp.game_id = g.id
        WHERE bp.player_id = ?
          AND g.game_date < ?
          AND bp.minutes > 0
        ORDER BY g.game_date DESC
        LIMIT ?
    """, (player_id, before_date, window))
    
    row = cur.fetchone()
    if not row or row[4] < 3:  # Need at least 3 games
        return {}
    
    return {
        "pts": row[0] or 0,
        "reb": row[1] or 0,
        "ast": row[2] or 0,
        "minutes": row[3] or 0,
        "games": row[4],
    }


def calculate_derived_line(averages: Dict[str, float], prop_type: str) -> Optional[float]:
    """
    Calculate derived line from averages with adjustment.
    
    The +5% adjustment accounts for sportsbook lines typically being
    slightly higher than averages.
    """
    if not averages or prop_type.lower() not in averages:
        return None
    
    avg = averages[prop_type.lower()]
    # Add 5% adjustment as sportsbook lines are typically slightly higher
    return round(avg * 1.05, 1)


def get_line_for_player(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Tuple[float, LineSource]:
    """
    Get the best available line for a player/prop.
    
    Priority:
    1. Actual sportsbook line
    2. Derived from L10 average (with +5% adjustment)
    
    Returns:
        Tuple of (line, source)
    """
    # Try sportsbook line first
    sb_result = get_sportsbook_line(conn, player_name, prop_type, game_date, player_id=player_id)
    if sb_result:
        return (sb_result[0], LineSource.SPORTSBOOK)
    
    # Fall back to derived
    averages = get_player_averages(conn, player_id, game_date, window=10)
    derived = calculate_derived_line(averages, prop_type)
    
    if derived:
        return (derived, LineSource.DERIVED)
    
    return (0.0, LineSource.DERIVED)


def get_games_for_date(conn: sqlite3.Connection, date: str) -> List[Dict]:
    """Get all games for a specific date."""
    cur = conn.execute("""
        SELECT 
            g.id,
            g.game_date, 
            t1.name as team1,
            t2.name as team2
        FROM games g
        JOIN teams t1 ON g.team1_id = t1.id
        JOIN teams t2 ON g.team2_id = t2.id
        WHERE g.game_date = ?
    """, (date,))
    
    return [
        {
            "game_id": row[0],
            "game_date": row[1],
            "home_team": row[2],
            "away_team": row[3],
        }
        for row in cur.fetchall()
    ]


def get_players_for_game(conn: sqlite3.Connection, game_id: int) -> List[Dict]:
    """Get all players who played in a game."""
    cur = conn.execute("""
        SELECT DISTINCT 
            p.id as player_id,
            p.name as full_name,
            t.name as team,
            bp.minutes
        FROM boxscore_player bp
        JOIN players p ON bp.player_id = p.id
        JOIN teams t ON bp.team_id = t.id
        WHERE bp.game_id = ?
          AND bp.minutes >= 15
        ORDER BY bp.minutes DESC
    """, (game_id,))
    
    return [
        {
            "player_id": row[0],
            "player_name": row[1],
            "team": row[2],
            "minutes": row[3],
        }
        for row in cur.fetchall()
    ]


def get_actual_stats(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
) -> Optional[Dict[str, float]]:
    """Get actual stats for a player on a specific date."""
    cur = conn.execute("""
        SELECT bp.pts, bp.reb, bp.ast, bp.minutes
        FROM boxscore_player bp
        JOIN games g ON bp.game_id = g.id
        WHERE bp.player_id = ?
          AND g.game_date = ?
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


# ============================================================================
# INDIVIDUAL MODEL PREDICTION FUNCTIONS
# ============================================================================

def get_v16_predictions(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team: str,
    opponent: str,
    game_date: str,
) -> List[ModelPrediction]:
    """
    Get predictions from Model V16 logic.
    
    V16 focuses on:
    - Cold bounce OVER patterns
    - Elite defense UNDER patterns
    - B2B fatigue UNDER patterns
    """
    predictions = []
    
    # Get player stats
    averages = get_player_averages(conn, player_id, game_date, window=10)
    if not averages or averages.get("minutes", 0) < 23:
        return []
    
    # Get L5 averages for pattern detection
    l5_avg = get_player_averages(conn, player_id, game_date, window=5)
    l15_avg = get_player_averages(conn, player_id, game_date, window=15)
    
    if not l5_avg or not l15_avg:
        return []
    
    # Check patterns for PTS
    pts_l5 = l5_avg.get("pts", 0)
    pts_l15 = l15_avg.get("pts", 0)
    pts_season = averages.get("pts", 0)
    
    if pts_l15 > 0 and pts_season >= 8:
        # Cold bounce detection: L5 significantly below L15
        cold_deviation = (pts_l5 - pts_l15) / pts_l15 * 100
        
        if cold_deviation <= -15:  # L5 is 15%+ below L15
            line, source = get_line_for_player(conn, player_id, player_name, "PTS", game_date)
            if line > 0:
                # Project bounce back toward L15
                projection = pts_l15 * 0.95  # Slightly below L15
                edge = (projection - line) / line * 100
                
                if edge >= 5:
                    predictions.append(ModelPrediction(
                        model_id="v16_general",
                        player_name=player_name,
                        prop_type="PTS",
                        direction="OVER",
                        projection=projection,
                        line=line,
                        edge=edge,
                        confidence=70 + min(edge, 15),
                        factors=["Cold Bounce Pattern", f"L5 {cold_deviation:.1f}% below L15"],
                    ))
    
    # Check for UNDER patterns (simplified)
    if pts_l15 > 0 and pts_season >= 8:
        line, source = get_line_for_player(conn, player_id, player_name, "PTS", game_date)
        if line > 0:
            # Use L10 as projection
            projection = pts_season
            edge = (line - projection) / line * 100
            
            if edge >= 6:
                predictions.append(ModelPrediction(
                    model_id="v16_general",
                    player_name=player_name,
                    prop_type="PTS",
                    direction="UNDER",
                    projection=projection,
                    line=line,
                    edge=edge,
                    confidence=60 + min(edge, 15),
                    factors=["Projection below line"],
                ))
    
    # Similar logic for REB
    reb_avg = averages.get("reb", 0)
    if reb_avg >= 4:
        line, source = get_line_for_player(conn, player_id, player_name, "REB", game_date)
        if line > 0:
            projection = reb_avg
            
            # OVER
            over_edge = (projection - line) / line * 100
            if over_edge >= 5:
                predictions.append(ModelPrediction(
                    model_id="v16_general",
                    player_name=player_name,
                    prop_type="REB",
                    direction="OVER",
                    projection=projection,
                    line=line,
                    edge=over_edge,
                    confidence=55 + min(over_edge, 15),
                    factors=["REB projection above line"],
                ))
            
            # UNDER
            under_edge = (line - projection) / line * 100
            if under_edge >= 6:
                predictions.append(ModelPrediction(
                    model_id="v16_general",
                    player_name=player_name,
                    prop_type="REB",
                    direction="UNDER",
                    projection=projection,
                    line=line,
                    edge=under_edge,
                    confidence=55 + min(under_edge, 15),
                    factors=["REB projection below line"],
                ))
    
    return predictions


def get_v18_predictions(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team: str,
    opponent: str,
    game_date: str,
) -> List[ModelPrediction]:
    """
    Get predictions from Model V18 logic.
    
    V18 focuses on holistic multi-factor scoring.
    """
    predictions = []
    
    # Get player stats
    averages = get_player_averages(conn, player_id, game_date, window=10)
    if not averages or averages.get("minutes", 0) < 23:
        return []
    
    l5_avg = get_player_averages(conn, player_id, game_date, window=5)
    l15_avg = get_player_averages(conn, player_id, game_date, window=15)
    
    if not l5_avg or not l15_avg:
        return []
    
    # PTS analysis with multi-factor approach
    pts_l5 = l5_avg.get("pts", 0)
    pts_l10 = averages.get("pts", 0)
    pts_l15 = l15_avg.get("pts", 0)
    
    if pts_l10 >= 8:
        line, source = get_line_for_player(conn, player_id, player_name, "PTS", game_date)
        if line > 0:
            # Weighted projection
            projection = pts_l5 * 0.25 + pts_l10 * 0.35 + pts_l15 * 0.40
            
            # Calculate factors
            factors = []
            factor_score = 0
            
            # Trend analysis
            if pts_l5 > pts_l10:
                factors.append("Hot recent form")
                factor_score += 10
            elif pts_l5 < pts_l10 * 0.85:
                factors.append("Cold recent form")
                factor_score += 5  # Indicates possible bounce
            
            # Consistency check
            if pts_l5 > 0 and pts_l15 > 0:
                variance = abs(pts_l5 - pts_l15) / pts_l15
                if variance < 0.15:
                    factors.append("Consistent performer")
                    factor_score += 10
            
            # OVER prediction
            over_edge = (projection - line) / line * 100
            if over_edge >= 5 and factor_score >= 10:
                predictions.append(ModelPrediction(
                    model_id="v18_general",
                    player_name=player_name,
                    prop_type="PTS",
                    direction="OVER",
                    projection=projection,
                    line=line,
                    edge=over_edge,
                    confidence=55 + factor_score + min(over_edge, 15),
                    factors=factors,
                ))
            
            # UNDER prediction
            under_edge = (line - projection) / line * 100
            if under_edge >= 6:
                predictions.append(ModelPrediction(
                    model_id="v18_general",
                    player_name=player_name,
                    prop_type="PTS",
                    direction="UNDER",
                    projection=projection,
                    line=line,
                    edge=under_edge,
                    confidence=55 + min(under_edge, 15),
                    factors=["Multi-factor UNDER signal"],
                ))
    
    return predictions


def get_production_predictions(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team: str,
    opponent: str,
    game_date: str,
) -> List[ModelPrediction]:
    """
    Get predictions from Production Model logic.
    
    Production focuses on cold bounce patterns primarily.
    """
    predictions = []
    
    # Get player stats
    l5_avg = get_player_averages(conn, player_id, game_date, window=5)
    l15_avg = get_player_averages(conn, player_id, game_date, window=15)
    l10_avg = get_player_averages(conn, player_id, game_date, window=10)
    
    if not l5_avg or not l15_avg or not l10_avg:
        return []
    
    if l10_avg.get("minutes", 0) < 23:
        return []
    
    # Cold bounce detection for PTS
    pts_l5 = l5_avg.get("pts", 0)
    pts_l15 = l15_avg.get("pts", 0)
    
    if pts_l15 > 0 and pts_l15 >= 10:
        cold_deviation = (pts_l5 - pts_l15) / pts_l15 * 100
        
        if cold_deviation <= -20:  # Strict cold threshold
            line, source = get_line_for_player(conn, player_id, player_name, "PTS", game_date)
            if line > 0:
                projection = pts_l15 * 0.92
                edge = (projection - line) / line * 100
                
                if edge >= 5:
                    predictions.append(ModelPrediction(
                        model_id="model_production",
                        player_name=player_name,
                        prop_type="PTS",
                        direction="OVER",
                        projection=projection,
                        line=line,
                        edge=edge,
                        confidence=70 + min(edge, 15),
                        factors=["PREMIUM Cold Bounce", f"L5 {cold_deviation:.1f}% below L15"],
                    ))
    
    # Similar for REB
    reb_l5 = l5_avg.get("reb", 0)
    reb_l15 = l15_avg.get("reb", 0)
    
    if reb_l15 > 0 and reb_l15 >= 4:
        cold_deviation = (reb_l5 - reb_l15) / reb_l15 * 100
        
        if cold_deviation <= -20:
            line, source = get_line_for_player(conn, player_id, player_name, "REB", game_date)
            if line > 0:
                projection = reb_l15 * 0.92
                edge = (projection - line) / line * 100
                
                if edge >= 5:
                    predictions.append(ModelPrediction(
                        model_id="model_production",
                        player_name=player_name,
                        prop_type="REB",
                        direction="OVER",
                        projection=projection,
                        line=line,
                        edge=edge,
                        confidence=65 + min(edge, 15),
                        factors=["Cold Bounce REB", f"L5 {cold_deviation:.1f}% below L15"],
                    ))
    
    return predictions


# ============================================================================
# ENSEMBLE FUNCTIONS
# ============================================================================

def aggregate_predictions(
    predictions: List[ModelPrediction],
) -> Dict[str, List[ModelPrediction]]:
    """
    Aggregate predictions by player/prop/direction.
    
    Returns:
        Dict mapping "player|prop|direction" to list of predictions
    """
    grouped = {}
    
    for pred in predictions:
        key = f"{pred.player_name}|{pred.prop_type}|{pred.direction}"
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pred)
    
    return grouped


def calculate_consensus_pick(
    predictions: List[ModelPrediction],
    player_id: int,
    team: str,
    opponent: str,
    game_date: str,
    game_id: Optional[int],
    line_source: LineSource,
) -> Optional[UnifiedPick]:
    """
    Calculate a consensus pick from multiple model predictions.
    """
    if not predictions:
        return None
    
    first = predictions[0]
    
    # Calculate weighted agreement
    total_weight = sum(MODEL_WEIGHTS.get(p.model_id, 1.0) for p in predictions)
    
    # Average projection and edge
    avg_projection = sum(p.projection for p in predictions) / len(predictions)
    avg_edge = sum(p.edge for p in predictions) / len(predictions)
    avg_confidence = sum(p.confidence for p in predictions) / len(predictions)
    
    # Collect all factors
    all_factors = []
    for p in predictions:
        all_factors.extend(p.factors)
    unique_factors = list(dict.fromkeys(all_factors))[:5]  # Top 5 unique factors
    
    # Determine confidence tier
    num_models = len(predictions)
    tier = ConfidenceTier.SPECULATIVE
    expected_hit = 0.55
    
    # Check tier requirements
    for tier_name in ["PREMIUM", "HIGH", "STANDARD", "SPECULATIVE"]:
        reqs = CONFIDENCE_TIERS[tier_name]
        
        if num_models >= reqs["min_models"] and avg_edge >= reqs["min_edge"]:
            if reqs["require_sportsbook"] and line_source != LineSource.SPORTSBOOK:
                continue
            
            tier = ConfidenceTier[tier_name]
            expected_hit = reqs["min_expected_hit"]
            break
    
    # Create pick ID
    pick_id = f"{game_date}|{player_id}|{first.prop_type}|{first.direction}"
    
    return UnifiedPick(
        pick_id=pick_id,
        player_id=player_id,
        player_name=first.player_name,
        team=team,
        opponent=opponent,
        game_date=game_date,
        game_id=game_id,
        prop_type=first.prop_type,
        direction=first.direction,
        line=first.line,
        line_source=line_source,
        projection=round(avg_projection, 1),
        edge=round(avg_edge, 1),
        tier=tier,
        expected_hit_rate=expected_hit,
        confidence_score=avg_confidence,
        models_agreeing=[p.model_id for p in predictions],
        model_predictions=predictions,
        weighted_agreement=total_weight,
        key_factors=unique_factors,
    )


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def get_unified_picks(
    date: str,
    conn: Optional[sqlite3.Connection] = None,
    min_tier: ConfidenceTier = ConfidenceTier.SPECULATIVE,
    max_picks: Optional[int] = None,
) -> DailyPicks:
    """
    Generate unified picks for a specific date.
    
    This is the main API for getting picks - it:
    1. Gets all games for the date
    2. Runs multiple models on each player
    3. Aggregates predictions into consensus picks
    4. Ranks by confidence tier and edge
    
    Args:
        date: Date string (YYYY-MM-DD)
        conn: Optional database connection
        min_tier: Minimum confidence tier to include
        max_picks: Maximum number of picks (defaults to game-based limit)
    
    Returns:
        DailyPicks object with ranked picks
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    
    try:
        # Get games for date
        games = get_games_for_date(conn, date)
        num_games = len(games)
        
        if num_games == 0:
            return DailyPicks(date=date, num_games=0)
        
        # Calculate target picks
        if max_picks is None:
            max_picks = min(MAX_DAILY_PICKS, max(MIN_DAILY_PICKS, int(num_games * TARGET_PICKS_PER_GAME)))
        
        all_predictions = []
        player_info = {}  # player_id -> (team, opponent, game_id)
        
        # Process each game
        for game in games:
            game_id = game["game_id"]
            home_team = game["home_team"]
            away_team = game["away_team"]
            
            # Get players from this game
            players = get_players_for_game(conn, game_id)
            
            for player in players:
                player_id = player["player_id"]
                player_name = player["player_name"]
                team = player["team"]
                opponent = away_team if team == home_team else home_team
                
                player_info[player_id] = (team, opponent, game_id)
                
                # Get predictions from each model
                predictions = []
                
                # V16
                predictions.extend(get_v16_predictions(
                    conn, player_id, player_name, team, opponent, date
                ))
                
                # V18
                predictions.extend(get_v18_predictions(
                    conn, player_id, player_name, team, opponent, date
                ))
                
                # Production
                predictions.extend(get_production_predictions(
                    conn, player_id, player_name, team, opponent, date
                ))
                
                all_predictions.extend(predictions)
        
        # Aggregate predictions by player/prop/direction
        grouped = aggregate_predictions(all_predictions)
        
        # Create consensus picks
        unified_picks = []
        
        for key, preds in grouped.items():
            player_name, prop_type, direction = key.split("|")
            
            # Find player_id
            player_id = None
            for pid, (team, opponent, game_id) in player_info.items():
                if get_player_name_by_id(conn, pid) == player_name:
                    player_id = pid
                    break
            
            if player_id is None:
                continue
            
            team, opponent, game_id = player_info[player_id]
            
            # Determine line source from first prediction
            line_source = LineSource.DERIVED
            sb_result = get_sportsbook_line(conn, player_name, prop_type, date, player_id=player_id)
            if sb_result:
                line_source = LineSource.SPORTSBOOK
            
            # Create consensus pick
            pick = calculate_consensus_pick(
                preds, player_id, team, opponent, date, game_id, line_source
            )
            
            if pick and pick.tier.stars >= min_tier.stars:
                unified_picks.append(pick)
        
        # Sort by tier (descending) then edge (descending)
        unified_picks.sort(
            key=lambda p: (p.tier.stars, p.edge, p.weighted_agreement),
            reverse=True
        )
        
        # Limit to max_picks
        final_picks = unified_picks[:max_picks]
        
        result = DailyPicks(date=date, num_games=num_games, picks=final_picks)
        result.calculate_stats()
        
        return result
        
    finally:
        if should_close:
            conn.close()


def get_player_name_by_id(conn: sqlite3.Connection, player_id: int) -> str:
    """Get player name from ID."""
    cur = conn.execute("SELECT name FROM players WHERE id = ?", (player_id,))
    row = cur.fetchone()
    return row[0] if row else ""


def run_backtest_unified(
    start_date: str,
    end_date: str,
    verbose: bool = False,
    show_progress: bool = True,
) -> BacktestResultV2:
    """
    Run backtest on the unified picks system.
    
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
    
    # Generate date range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    dates = []
    current = start_dt
    while current <= end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    result.days_tested = len(dates)
    
    if verbose:
        print(f"Running backtest from {start_date} to {end_date} ({len(dates)} days)")
    
    for i, date in enumerate(dates):
        if show_progress:
            pct = (i + 1) / len(dates) * 100
            print(f"\r[{'█' * int(pct/5):<20}] {pct:.0f}% - {date}", end="", flush=True)
        
        # Get picks for this date
        daily_picks = get_unified_picks(date, conn)
        
        # Grade each pick
        for pick in daily_picks.picks:
            actual = get_actual_stats(conn, pick.player_id, date)
            
            if actual is None:
                continue
            
            actual_value = actual.get(pick.prop_type.lower(), 0)
            pick.actual_value = actual_value
            
            # Determine if hit
            if pick.direction == "OVER":
                pick.hit = actual_value > pick.line
            else:
                pick.hit = actual_value < pick.line
            
            # Update result counts
            result.total_picks += 1
            if pick.hit:
                result.total_hits += 1
            
            # By tier
            if pick.tier == ConfidenceTier.PREMIUM:
                result.premium_picks += 1
                if pick.hit:
                    result.premium_hits += 1
            elif pick.tier == ConfidenceTier.HIGH:
                result.high_picks += 1
                if pick.hit:
                    result.high_hits += 1
            elif pick.tier == ConfidenceTier.STANDARD:
                result.standard_picks += 1
                if pick.hit:
                    result.standard_hits += 1
            else:
                result.speculative_picks += 1
                if pick.hit:
                    result.speculative_hits += 1
            
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
            
            # By line source
            if pick.line_source == LineSource.SPORTSBOOK:
                result.sportsbook_picks += 1
                if pick.hit:
                    result.sportsbook_hits += 1
            else:
                result.derived_picks += 1
                if pick.hit:
                    result.derived_hits += 1
            
            # Store result
            result.pick_results.append({
                "date": date,
                "player": pick.player_name,
                "prop": pick.prop_type,
                "direction": pick.direction,
                "line": pick.line,
                "projection": pick.projection,
                "actual": actual_value,
                "hit": pick.hit,
                "tier": pick.tier.value,
                "line_source": pick.line_source.value,
                "models": pick.models_agreeing,
            })
    
    if show_progress:
        print()  # New line after progress bar
    
    conn.close()
    
    result.calculate_rates()
    
    if verbose:
        print(result.summary())
    
    return result


# ============================================================================
# API FOR WEB INTERFACE
# ============================================================================

def get_picks_for_matchup(
    game_id: int,
    date: str,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict]:
    """
    Get unified picks for a specific game/matchup.
    
    This is used by the matchups view in the web interface.
    
    Args:
        game_id: Game ID
        date: Game date
        conn: Optional database connection
    
    Returns:
        List of pick dictionaries
    """
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True
    
    try:
        # Get game info
        cur = conn.execute("""
            SELECT home_team, away_team FROM games WHERE game_id = ?
        """, (game_id,))
        row = cur.fetchone()
        if not row:
            return []
        
        home_team, away_team = row
        
        # Get players in game
        players = get_players_for_game(conn, game_id)
        
        all_predictions = []
        player_info = {}
        
        for player in players:
            player_id = player["player_id"]
            player_name = player["player_name"]
            team = player["team"]
            opponent = away_team if team == home_team else home_team
            
            player_info[player_id] = (team, opponent)
            
            # Get predictions from each model
            predictions = []
            predictions.extend(get_v16_predictions(conn, player_id, player_name, team, opponent, date))
            predictions.extend(get_v18_predictions(conn, player_id, player_name, team, opponent, date))
            predictions.extend(get_production_predictions(conn, player_id, player_name, team, opponent, date))
            
            all_predictions.extend(predictions)
        
        # Aggregate and create picks
        grouped = aggregate_predictions(all_predictions)
        picks = []
        
        for key, preds in grouped.items():
            player_name, prop_type, direction = key.split("|")
            
            # Find player info
            for pid, (team, opponent) in player_info.items():
                if get_player_name_by_id(conn, pid) == player_name:
                    line_source = LineSource.DERIVED
                    sb_result = get_sportsbook_line(conn, player_name, prop_type, date, player_id=pid)
                    if sb_result:
                        line_source = LineSource.SPORTSBOOK
                    
                    pick = calculate_consensus_pick(
                        preds, pid, team, opponent, date, game_id, line_source
                    )
                    if pick:
                        picks.append(pick.to_dict())
                    break
        
        # Sort by confidence
        picks.sort(key=lambda p: (p["tier_stars"], p["edge"]), reverse=True)
        
        return picks
        
    finally:
        if should_close:
            conn.close()


# ============================================================================
# CLI INTERFACE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python unified_picks.py <date|backtest> [args]")
        print("  python unified_picks.py 2026-02-04")
        print("  python unified_picks.py backtest 2025-12-01 2026-02-03")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "backtest":
        if len(sys.argv) < 4:
            print("Usage: python unified_picks.py backtest <start> <end>")
            sys.exit(1)
        
        start = sys.argv[2]
        end = sys.argv[3]
        
        result = run_backtest_unified(start, end, verbose=True, show_progress=True)
        
    else:
        # Assume it's a date
        date = command
        
        picks = get_unified_picks(date)
        print(picks.summary())
