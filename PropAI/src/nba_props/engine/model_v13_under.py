"""
Model V13 Under - Specialized UNDER NBA Props Prediction Model
================================================================

This is the second half of a dual-model approach:
- Model V13 General: Focuses on OVER picks and general value (separate file)
- Model V13 Under: Focuses EXCLUSIVELY on UNDER opportunities

WHY A SEPARATE UNDER MODEL?
---------------------------
Analysis of previous models revealed:
1. PTS UNDER hits at 63.9% vs PTS OVER at 48.3% (huge difference!)
2. Negative factors compound more reliably than positive ones
3. Elite defense consistently limits production
4. Cold streaks tend to persist due to psychology and matchups

UNDER picks have fundamentally different drivers than OVER picks:
- OVER: Momentum, hot streaks, usage redistribution
- UNDER: Defense, fatigue, cold streaks, variance

THE KEY INSIGHT (from 1_MODEL_SUMMARY.md):
------------------------------------------
"UNDER picks are more predictable than OVER picks because negative factors 
compound more reliably than positive ones."

VALIDATED INSIGHTS INCORPORATED:
--------------------------------
1. DEFENSE VS POSITION (Primary Driver):
   - Elite defense (top 5) consistently limits production
   - Top 5 DVP: ~62% UNDER hit rate
   - Combine with cold streak for premium picks (~66%)

2. COLD STREAKS:
   - L5 below season avg = persistent cold
   - Players don't snap out immediately
   - Compound effect with defense

3. BACK-TO-BACK FATIGUE:
   - Measurable but smaller than expected
   - Use as supporting factor, not primary

4. SPORTSBOOK LINES:
   - Use when available (higher confidence)
   - Fall back to derived lines (L10 avg)
   - Track separately for honest reporting

MODEL V13 UNDER RULES:
----------------------
1. ONLY generate UNDER picks (leave OVER to General model)
2. Primary signal: Elite defense at position + cold streak
3. Secondary signals: Good defense + severe cold, B2B, injury rust
4. Use sportsbook lines when available
5. Track and report hit rates by line source
6. Focus on PTS and REB (AST excluded for low-avg players)

CONFIDENCE TIERS:
-----------------
- PREMIUM (5 stars): Elite defense + Cold streak + 85+ confidence
- HIGH (4 stars): Elite defense OR Good defense + Cold streak severe
- MEDIUM (3 stars): Multiple supporting factors
- Excluded: Single weak factor

USAGE:
------
    from src.nba_props.engine.model_v13_under import (
        get_daily_picks_v13_under,
        run_backtest_v13_under,
        ModelConfigV13Under,
    )
    
    # Get picks for today
    picks = get_daily_picks_v13_under("2026-02-03")
    
    # Run backtest
    result = run_backtest_v13_under("2025-12-01", "2026-02-02")

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
class ModelConfigV13Under:
    """
    Model V13 Under Configuration.
    
    Specialized for UNDER picks only with defense and negative factor focus.
    """
    # === VERSION INFO ===
    model_name: str = "Model V13 Under"
    model_version: str = "13.0"
    
    # === SPORTSBOOK LINE HANDLING ===
    require_sportsbook_line: bool = False
    sportsbook_line_confidence_boost: float = 8.0
    derived_line_adjustment: float = 1.05
    
    # === DATA REQUIREMENTS ===
    min_games_required: int = 10
    min_minutes_filter: int = 5
    min_avg_minutes: float = 20.0
    max_games_lookback: int = 20
    
    # === DEFENSE VS POSITION THRESHOLDS ===
    elite_defense_rank: int = 5       # Top 5 = elite
    good_defense_rank: int = 10       # Top 10 = good
    average_defense_rank: int = 15    # Top 15 = average
    
    # === COLD STREAK THRESHOLDS ===
    cold_streak_severe_pct: float = -20.0   # L5 is 20%+ below season
    cold_streak_mild_pct: float = -10.0     # L5 is 10%+ below season
    
    # === FACTOR WEIGHTS ===
    # These are validated from backtesting
    weights: Dict[str, float] = field(default_factory=lambda: {
        "defense_elite": 30,          # Primary factor
        "defense_good": 15,           # Secondary
        "defense_average": 5,         # Minor
        "cold_streak_severe": 22,     # Strong signal
        "cold_streak_mild": 12,       # Moderate signal
        "b2b_second": 8,              # Fatigue
        "b2b_third_in_four": 5,       # Mild fatigue
        "injury_first_back": 18,      # Rust factor
        "injury_second_back": 12,     # Moderate rust
        "high_variance": 6,           # Inconsistent player
        "historical_struggle": 10,    # Bad matchup history
        "elite_defender_matchup": 10, # Individual defender
        "away_disadvantage": 3,       # Minor
    })
    
    # === FACTOR ADJUSTMENTS (multipliers) ===
    adjustments: Dict[str, float] = field(default_factory=lambda: {
        "defense_elite": 0.88,        # -12%
        "defense_good": 0.94,         # -6%
        "defense_average": 0.98,      # -2%
        "cold_streak_severe": 0.86,   # -14%
        "cold_streak_mild": 0.93,     # -7%
        "b2b_second": 0.96,           # -4%
        "b2b_third_in_four": 0.98,    # -2%
        "injury_first_back": 0.80,    # -20%
        "injury_second_back": 0.90,   # -10%
        "high_variance": 0.97,        # -3%
        "historical_struggle": 0.94,  # -6%
        "elite_defender_matchup": 0.92, # -8%
        "away_disadvantage": 0.99,    # -1%
    })
    
    # === EDGE REQUIREMENTS ===
    min_edge_sportsbook: float = 5.0   # 5%+ edge vs actual line
    min_edge_derived: float = 8.0      # 8%+ edge vs derived line
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence: float = 88.0
    high_confidence: float = 75.0
    medium_confidence: float = 65.0
    min_confidence: float = 55.0
    
    # === PROP SELECTION ===
    prop_types: List[str] = field(default_factory=lambda: ['pts', 'reb', 'ast'])
    min_avg_for_prop: Dict[str, float] = field(default_factory=lambda: {
        'pts': 8.0,    # Min 8 PPG average
        'reb': 4.0,    # Min 4 RPG average
        'ast': 5.5,    # Min 5.5 APG average (higher bar for AST)
    })
    
    # === PICK LIMITS ===
    max_picks_per_game: int = 5
    max_picks_per_day: int = 25
    max_picks_per_player: int = 2
    
    # === QUALITY FILTERS ===
    # Require at least one of these combinations:
    # 1. Elite defense + cold streak (PREMIUM)
    # 2. Elite defense alone with high confidence
    # 3. Good defense + severe cold streak
    # 4. Injury factor + defense support
    require_defense_factor: bool = True  # At least good defense needed
    require_multiple_factors: int = 2    # Min 2 factors for non-elite defense
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PlayerStatsV13Under:
    """Player statistics for Under model."""
    player_id: int
    player_name: str
    team_abbrev: str
    position: str
    games_played: int
    avg_minutes: float
    
    # Stat averages
    season: Dict[str, float] = field(default_factory=dict)
    l5: Dict[str, float] = field(default_factory=dict)
    l10: Dict[str, float] = field(default_factory=dict)
    l20: Dict[str, float] = field(default_factory=dict)
    
    # Variance (standard deviation)
    stds: Dict[str, float] = field(default_factory=dict)
    
    # Deviations
    deviations: Dict[str, float] = field(default_factory=dict)  # L5 vs Season
    
    # Historical vs opponent
    vs_opponent: Dict[str, float] = field(default_factory=dict)
    vs_opponent_games: int = 0
    
    def get_cv(self, prop_type: str) -> float:
        """Get coefficient of variation."""
        mean = self.season.get(prop_type.lower(), 0)
        std = self.stds.get(prop_type.lower(), 0)
        if mean <= 0:
            return 1.0
        return std / mean


@dataclass
class DefenseProfileV13Under:
    """Defense vs position profile."""
    team_abbrev: str
    position: str
    data_available: bool = False
    
    pts_rank: int = 15
    pts_allowed: float = 0.0
    pts_rating: str = "average"
    
    reb_rank: int = 15
    reb_allowed: float = 0.0
    reb_rating: str = "average"
    
    ast_rank: int = 15
    ast_allowed: float = 0.0
    ast_rating: str = "average"
    
    def get_rating(self, prop_type: str) -> str:
        mapping = {'pts': self.pts_rating, 'reb': self.reb_rating, 'ast': self.ast_rating}
        return mapping.get(prop_type.lower(), "average")
    
    def get_rank(self, prop_type: str) -> int:
        mapping = {'pts': self.pts_rank, 'reb': self.reb_rank, 'ast': self.ast_rank}
        return mapping.get(prop_type.lower(), 15)


@dataclass
class LineInfoV13Under:
    """Betting line information."""
    value: float
    source: str
    book: str = "unknown"
    
    @property
    def is_sportsbook(self) -> bool:
        return self.source == "sportsbook"


@dataclass
class BackToBackInfo:
    """Back-to-back game information."""
    is_b2b: bool = False
    is_third_in_four: bool = False


@dataclass
class UnderFactorAnalysis:
    """Analysis of negative factors for a player/prop."""
    factors: Dict[str, float] = field(default_factory=dict)      # factor -> weight
    adjustments: Dict[str, float] = field(default_factory=dict)  # factor -> multiplier
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def total_weight(self) -> float:
        return sum(self.factors.values())
    
    @property
    def factor_count(self) -> int:
        return len(self.factors)
    
    @property
    def total_adjustment(self) -> float:
        adj = 1.0
        for a in self.adjustments.values():
            adj *= a
        return adj
    
    @property
    def has_elite_defense(self) -> bool:
        return "defense_elite" in self.factors
    
    @property
    def has_good_defense(self) -> bool:
        return "defense_good" in self.factors
    
    @property
    def has_cold_streak(self) -> bool:
        return "cold_streak_severe" in self.factors or "cold_streak_mild" in self.factors
    
    @property
    def has_severe_cold_streak(self) -> bool:
        return "cold_streak_severe" in self.factors


@dataclass
class PropPickV13Under:
    """A pick generated by Model V13 Under."""
    # Identity (required fields first - no defaults)
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    game_date: str
    position: str
    
    # Pick details
    prop_type: str
    
    # Line information
    line: float
    line_source: str
    book: str
    
    # Projections (required)
    season_avg: float
    l5_avg: float
    projected: float
    
    # Fields with defaults
    # Pick direction (always UNDER for this model)
    direction: str = "UNDER"
    
    # Factor analysis
    factors: Dict[str, float] = field(default_factory=dict)
    adjustment_factor: float = 1.0
    
    # Edge calculation
    edge_vs_line: float = 0.0
    
    # Confidence
    raw_score: float = 0.0
    confidence_score: float = 0.0
    confidence_tier: str = "MEDIUM"
    
    # Defense context
    defense_rank: int = 15
    defense_rating: str = "average"
    
    # Reasons
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Results (filled after game)
    actual_value: Optional[float] = None
    hit: Optional[bool] = None
    margin: Optional[float] = None
    
    @property
    def is_premium(self) -> bool:
        return self.confidence_tier == "PREMIUM"
    
    @property
    def is_high(self) -> bool:
        return self.confidence_tier == "HIGH"
    
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
            "season_avg": round(self.season_avg, 1),
            "l5_avg": round(self.l5_avg, 1),
            "projected": round(self.projected, 1),
            "edge": f"{self.edge_vs_line:.1f}%",
            "tier": self.confidence_tier,
            "confidence": round(self.confidence_score, 1),
            "defense_rank": self.defense_rank,
            "defense_rating": self.defense_rating,
            "factors": list(self.factors.keys()),
            "factor_count": len(self.factors),
            "reasons": self.reasons,
            "actual": self.actual_value,
            "hit": self.hit,
        }


@dataclass
class DailyPicksV13Under:
    """All picks for a day from Model V13 Under."""
    date: str
    games: int
    picks: List[PropPickV13Under] = field(default_factory=list)
    
    # Coverage
    players_with_sportsbook_lines: int = 0
    players_with_derived_lines: int = 0
    total_players_analyzed: int = 0
    
    @property
    def total_picks(self) -> int:
        return len(self.picks)
    
    @property
    def premium_picks(self) -> List[PropPickV13Under]:
        return [p for p in self.picks if p.confidence_tier == "PREMIUM"]
    
    @property
    def high_picks(self) -> List[PropPickV13Under]:
        return [p for p in self.picks if p.confidence_tier == "HIGH"]
    
    @property
    def sportsbook_picks(self) -> List[PropPickV13Under]:
        return [p for p in self.picks if p.line_source == "sportsbook"]
    
    @property
    def derived_picks(self) -> List[PropPickV13Under]:
        return [p for p in self.picks if p.line_source == "derived"]
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V13 UNDER PICKS - {self.date}",
            f"{'='*70}",
            f"Games: {self.games}",
            f"Total Picks: {self.total_picks}",
            f"  - With sportsbook lines: {len(self.sportsbook_picks)}",
            f"  - With derived lines: {len(self.derived_picks)}",
            f"  - PREMIUM: {len(self.premium_picks)}",
            f"  - HIGH: {len(self.high_picks)}",
            "",
        ]
        
        for tier in ["PREMIUM", "HIGH", "MEDIUM"]:
            tier_picks = [p for p in self.picks if p.confidence_tier == tier]
            if tier_picks:
                lines.append(f"--- {tier} ({len(tier_picks)}) ---")
                for p in tier_picks:
                    line_badge = "🎯" if p.line_source == "sportsbook" else "📊"
                    lines.append(
                        f"  📉{line_badge} {p.player_name} ({p.team_abbrev} vs {p.opponent_abbrev}): "
                        f"{p.prop_type} UNDER {p.line:.1f}"
                    )
                    lines.append(
                        f"      Proj: {p.projected:.1f} | Edge: {p.edge_vs_line:.1f}% | "
                        f"Def: {p.defense_rating} (#{p.defense_rank}) | Factors: {p.factor_count}"
                    )
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class BacktestResultV13Under:
    """Comprehensive backtest results for Model V13 Under."""
    start_date: str
    end_date: str
    config: ModelConfigV13Under
    
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
    medium_picks: int = 0
    medium_hits: int = 0
    
    # By prop type
    pts_picks: int = 0
    pts_hits: int = 0
    reb_picks: int = 0
    reb_hits: int = 0
    ast_picks: int = 0
    ast_hits: int = 0
    
    # By defense rating
    elite_defense_picks: int = 0
    elite_defense_hits: int = 0
    good_defense_picks: int = 0
    good_defense_hits: int = 0
    
    # By cold streak
    cold_streak_picks: int = 0
    cold_streak_hits: int = 0
    
    # Combined factors
    elite_plus_cold_picks: int = 0
    elite_plus_cold_hits: int = 0
    
    # Coverage
    days_tested: int = 0
    total_games: int = 0
    
    # Detailed
    all_picks: List[PropPickV13Under] = field(default_factory=list)
    daily_results: List[Dict] = field(default_factory=list)
    factor_effectiveness: Dict[str, Dict] = field(default_factory=dict)
    
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
    def elite_plus_cold_rate(self) -> float:
        return self.elite_plus_cold_hits / self.elite_plus_cold_picks * 100 if self.elite_plus_cold_picks > 0 else 0.0
    
    def summary(self) -> str:
        lines = [
            f"{'='*70}",
            f"MODEL V13 UNDER - BACKTEST RESULTS",
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
            f"  Medium:   {self.medium_hits}/{self.medium_picks} ({self.medium_hits/self.medium_picks*100:.1f}%)" if self.medium_picks else "  Medium: N/A",
            "",
            "BY PROP TYPE:",
            f"  PTS: {self.pts_hits}/{self.pts_picks} ({self.pts_hits/self.pts_picks*100:.1f}%)" if self.pts_picks else "  PTS: N/A",
            f"  REB: {self.reb_hits}/{self.reb_picks} ({self.reb_hits/self.reb_picks*100:.1f}%)" if self.reb_picks else "  REB: N/A",
            f"  AST: {self.ast_hits}/{self.ast_picks} ({self.ast_hits/self.ast_picks*100:.1f}%)" if self.ast_picks else "  AST: N/A",
            "",
            "BY DEFENSE RATING:",
            f"  Elite Defense: {self.elite_defense_hits}/{self.elite_defense_picks} ({self.elite_defense_hits/self.elite_defense_picks*100:.1f}%)" if self.elite_defense_picks else "  Elite: N/A",
            f"  Good Defense:  {self.good_defense_hits}/{self.good_defense_picks} ({self.good_defense_hits/self.good_defense_picks*100:.1f}%)" if self.good_defense_picks else "  Good: N/A",
            "",
            "PREMIUM COMBO (Elite Defense + Cold Streak):",
            f"  {self.elite_plus_cold_hits}/{self.elite_plus_cold_picks} ({self.elite_plus_cold_rate:.1f}%)" if self.elite_plus_cold_picks else "  N/A",
            "",
            "FACTOR EFFECTIVENESS:",
        ]
        
        # Sort factors by hit rate
        sorted_factors = sorted(
            self.factor_effectiveness.items(),
            key=lambda x: x[1].get('hit_rate', 0),
            reverse=True
        )
        for factor, data in sorted_factors[:10]:
            if data.get('picks', 0) >= 5:
                lines.append(
                    f"  {factor}: {data['hits']}/{data['picks']} ({data['hit_rate']*100:.1f}%)"
                )
        
        lines.append(f"{'='*70}")
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


def _map_position(pos: str) -> str:
    """Map position to standard DVP position."""
    if not pos:
        return "SF"
    pos = pos.upper().strip()
    if pos in ("PG", "SG", "SF", "PF", "C"):
        return pos
    mapping = {
        "G": "PG", "F": "SF", "G-F": "SG", "F-G": "SG",
        "F-C": "PF", "C-F": "PF", "GUARD": "PG",
        "FORWARD": "SF", "CENTER": "C",
    }
    return mapping.get(pos, "SF")


def _get_injured_players(conn: sqlite3.Connection, game_date: str) -> Set[int]:
    """Get set of player IDs who are OUT or DOUBTFUL."""
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


def _get_injured_player_names(conn: sqlite3.Connection, game_date: str) -> Set[str]:
    """Get set of normalized player names who are OUT/DOUBTFUL."""
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
    return {_normalize_name(row["player_name"]) for row in rows if row["player_name"]}


def _get_sportsbook_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
) -> Optional[Tuple[float, str]]:
    """Get sportsbook line if available."""
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
    
    # Fuzzy match by name
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


def _get_back_to_back_info(
    conn: sqlite3.Connection,
    team_abbrev: str,
    game_date: str,
) -> BackToBackInfo:
    """Check if team is on back-to-back."""
    info = BackToBackInfo()
    
    try:
        game_dt = datetime.strptime(game_date, "%Y-%m-%d")
        yesterday = (game_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        four_days_ago = (game_dt - timedelta(days=4)).strftime("%Y-%m-%d")
        
        # Check yesterday
        yesterday_game = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM games g
            JOIN teams t ON (t.id = g.team1_id OR t.id = g.team2_id)
            WHERE g.game_date = ? AND t.name LIKE ?
            """,
            (yesterday, f"%{team_abbrev}%"),
        ).fetchone()
        
        if yesterday_game and yesterday_game["cnt"] > 0:
            info.is_b2b = True
        
        # Check 3 in 4
        recent = conn.execute(
            """
            SELECT COUNT(*) as cnt
            FROM games g
            JOIN teams t ON (t.id = g.team1_id OR t.id = g.team2_id)
            WHERE g.game_date BETWEEN ? AND ? AND g.game_date < ? AND t.name LIKE ?
            """,
            (four_days_ago, game_date, game_date, f"%{team_abbrev}%"),
        ).fetchone()
        
        if recent and recent["cnt"] >= 2:
            info.is_third_in_four = True
    except:
        pass
    
    return info


def _get_defense_profile(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
    config: ModelConfigV13Under,
) -> DefenseProfileV13Under:
    """Get comprehensive defense profile."""
    dvp_position = _map_position(position)
    profile = DefenseProfileV13Under(team_abbrev=team_abbrev, position=dvp_position)
    
    row = conn.execute(
        """
        SELECT pts_rank, reb_rank, ast_rank, pts_allowed, reb_allowed, ast_allowed
        FROM team_defense_vs_position
        WHERE team_abbrev = ? AND position = ?
        ORDER BY updated_at DESC LIMIT 1
        """,
        (team_abbrev, dvp_position)
    ).fetchone()
    
    if row:
        profile.data_available = True
        profile.pts_rank = row["pts_rank"] or 15
        profile.reb_rank = row["reb_rank"] or 15
        profile.ast_rank = row["ast_rank"] or 15
        profile.pts_allowed = row["pts_allowed"] or 0
        profile.reb_allowed = row["reb_allowed"] or 0
        profile.ast_allowed = row["ast_allowed"] or 0
        
        for stat, rank_attr, rating_attr in [
            ('pts', 'pts_rank', 'pts_rating'),
            ('reb', 'reb_rank', 'reb_rating'),
            ('ast', 'ast_rank', 'ast_rating'),
        ]:
            rank = getattr(profile, rank_attr)
            if rank <= config.elite_defense_rank:
                rating = "elite"
            elif rank <= config.good_defense_rank:
                rating = "good"
            elif rank <= config.average_defense_rank:
                rating = "average"
            else:
                rating = "weak"
            setattr(profile, rating_attr, rating)
    
    return profile


def _check_elite_defender(
    conn: sqlite3.Connection,
    opponent_abbrev: str,
    position: str,
    game_date: str,
) -> Optional[str]:
    """Check if there's an elite defender on opponent at this position."""
    injured_names = _get_injured_player_names(conn, game_date)
    
    # Check elite_defenders table
    row = conn.execute(
        """
        SELECT ed.player_name
        FROM elite_defenders ed
        JOIN player_archetypes pa ON pa.player_name = ed.player_name
        WHERE ed.position = ? AND pa.team LIKE ? AND pa.season = '2025-26'
        LIMIT 1
        """,
        (position, f"%{opponent_abbrev}%"),
    ).fetchone()
    
    if row:
        if _normalize_name(row["player_name"]) not in injured_names:
            return row["player_name"]
    
    # Check archetypes
    arch = conn.execute(
        """
        SELECT player_name
        FROM player_archetypes
        WHERE team LIKE ? AND position = ? AND is_elite_defender = 1 AND season = '2025-26'
        LIMIT 1
        """,
        (f"%{opponent_abbrev}%", position),
    ).fetchone()
    
    if arch:
        if _normalize_name(arch["player_name"]) not in injured_names:
            return arch["player_name"]
    
    return None


# ============================================================================
# Core Model Functions
# ============================================================================

def _load_player_stats(
    conn: sqlite3.Connection,
    player_id: int,
    before_date: str,
    opponent_abbrev: str,
    config: ModelConfigV13Under,
) -> Optional[PlayerStatsV13Under]:
    """Load comprehensive player statistics."""
    player = conn.execute(
        "SELECT id, name FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    
    if not player:
        return None
    
    rows = conn.execute(
        """
        SELECT g.game_date, b.pts, b.reb, b.ast, b.minutes, b.pos, t.name as team_name
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        JOIN teams t ON t.id = b.team_id
        WHERE b.player_id = ? AND g.game_date < ? AND b.minutes > ?
        ORDER BY g.game_date DESC
        LIMIT ?
        """,
        (player_id, before_date, config.min_minutes_filter, config.max_games_lookback),
    ).fetchall()
    
    if len(rows) < config.min_games_required:
        return None
    
    games = [dict(r) for r in rows]
    n = len(games)
    
    avg_min = sum(g["minutes"] or 0 for g in games) / n
    if avg_min < config.min_avg_minutes:
        return None
    
    stats_data = {
        'pts': [g["pts"] or 0 for g in games],
        'reb': [g["reb"] or 0 for g in games],
        'ast': [g["ast"] or 0 for g in games],
    }
    
    def avg(vals, limit=None):
        subset = vals[:limit] if limit else vals
        return sum(subset) / len(subset) if subset else 0.0
    
    def safe_std(vals, limit=10):
        subset = vals[:limit]
        return statistics.stdev(subset) if len(subset) >= 2 else 0.0
    
    player_stats = PlayerStatsV13Under(
        player_id=player_id,
        player_name=player["name"],
        team_abbrev=abbrev_from_team_name(games[0]["team_name"]) or "",
        position=_map_position(games[0].get("pos") or "SF"),
        games_played=n,
        avg_minutes=avg_min,
    )
    
    for stat in ['pts', 'reb', 'ast']:
        vals = stats_data[stat]
        player_stats.season[stat] = avg(vals)
        player_stats.l5[stat] = avg(vals, 5)
        player_stats.l10[stat] = avg(vals, 10)
        player_stats.l20[stat] = avg(vals, 20) if n >= 20 else avg(vals)
        player_stats.stds[stat] = safe_std(vals)
        
        # Deviation (L5 vs Season)
        season = player_stats.season[stat]
        l5 = player_stats.l5[stat]
        player_stats.deviations[stat] = ((l5 - season) / season * 100) if season > 0 else 0.0
    
    # Historical vs opponent
    vs_opp = conn.execute(
        """
        SELECT bp.pts, bp.reb, bp.ast
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        JOIN teams t1 ON t1.id = g.team1_id
        JOIN teams t2 ON t2.id = g.team2_id
        WHERE bp.player_id = ? AND g.game_date < ?
          AND (t1.name LIKE ? OR t2.name LIKE ?)
          AND bp.minutes > 5
        ORDER BY g.game_date DESC LIMIT 10
        """,
        (player_id, before_date, f"%{opponent_abbrev}%", f"%{opponent_abbrev}%"),
    ).fetchall()
    
    if vs_opp and len(vs_opp) >= 2:
        player_stats.vs_opponent_games = len(vs_opp)
        player_stats.vs_opponent['pts'] = sum(g["pts"] or 0 for g in vs_opp) / len(vs_opp)
        player_stats.vs_opponent['reb'] = sum(g["reb"] or 0 for g in vs_opp) / len(vs_opp)
        player_stats.vs_opponent['ast'] = sum(g["ast"] or 0 for g in vs_opp) / len(vs_opp)
    
    return player_stats


def _analyze_under_factors(
    conn: sqlite3.Connection,
    stats: PlayerStatsV13Under,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    is_home: bool,
    config: ModelConfigV13Under,
) -> UnderFactorAnalysis:
    """Analyze all negative factors for an UNDER pick."""
    analysis = UnderFactorAnalysis()
    pt = prop_type.lower()
    
    season_avg = stats.season.get(pt, 0)
    l5_avg = stats.l5.get(pt, 0)
    deviation = stats.deviations.get(pt, 0)
    vs_opp = stats.vs_opponent.get(pt)
    
    # Get defense profile
    defense = _get_defense_profile(conn, opponent_abbrev, stats.position, config)
    
    # Get B2B info
    b2b = _get_back_to_back_info(conn, stats.team_abbrev, game_date)
    
    # 1. DEFENSE VS POSITION (Primary Factor)
    if defense.data_available:
        defense_rating = defense.get_rating(pt)
        defense_rank = defense.get_rank(pt)
        
        if defense_rating == "elite":
            analysis.factors["defense_elite"] = config.weights["defense_elite"]
            analysis.adjustments["defense_elite"] = config.adjustments["defense_elite"]
            analysis.reasons.append(
                f"🛡️ ELITE defense vs {stats.position}: {opponent_abbrev} ranks #{defense_rank}"
            )
        elif defense_rating == "good":
            analysis.factors["defense_good"] = config.weights["defense_good"]
            analysis.adjustments["defense_good"] = config.adjustments["defense_good"]
            analysis.reasons.append(
                f"🛡️ Good defense vs {stats.position}: {opponent_abbrev} ranks #{defense_rank}"
            )
        elif defense_rating == "average":
            analysis.factors["defense_average"] = config.weights["defense_average"]
            analysis.adjustments["defense_average"] = config.adjustments["defense_average"]
        elif defense_rating == "weak":
            analysis.warnings.append(
                f"⚠️ {opponent_abbrev} has WEAK defense vs {stats.position} (#{defense_rank})"
            )
    else:
        analysis.warnings.append("No defense vs position data")
    
    # 2. COLD STREAK
    if deviation <= config.cold_streak_severe_pct:
        analysis.factors["cold_streak_severe"] = config.weights["cold_streak_severe"]
        analysis.adjustments["cold_streak_severe"] = config.adjustments["cold_streak_severe"]
        analysis.reasons.append(
            f"❄️ SEVERE cold streak: L5 ({l5_avg:.1f}) is {deviation:.0f}% below season ({season_avg:.1f})"
        )
    elif deviation <= config.cold_streak_mild_pct:
        analysis.factors["cold_streak_mild"] = config.weights["cold_streak_mild"]
        analysis.adjustments["cold_streak_mild"] = config.adjustments["cold_streak_mild"]
        analysis.reasons.append(
            f"Cold: L5 ({l5_avg:.1f}) is {deviation:.0f}% below season ({season_avg:.1f})"
        )
    
    # 3. BACK-TO-BACK
    if b2b.is_b2b:
        analysis.factors["b2b_second"] = config.weights["b2b_second"]
        analysis.adjustments["b2b_second"] = config.adjustments["b2b_second"]
        analysis.reasons.append("🔋 Back-to-back fatigue")
    elif b2b.is_third_in_four:
        analysis.factors["b2b_third_in_four"] = config.weights["b2b_third_in_four"]
        analysis.adjustments["b2b_third_in_four"] = config.adjustments["b2b_third_in_four"]
        analysis.reasons.append("🔋 Third game in four nights")
    
    # 4. HIGH VARIANCE
    cv = stats.get_cv(pt)
    if cv > 0.35:
        analysis.factors["high_variance"] = config.weights["high_variance"]
        analysis.adjustments["high_variance"] = config.adjustments["high_variance"]
        analysis.reasons.append(f"📊 High variance (CV={cv:.2f})")
    
    # 5. HISTORICAL STRUGGLE
    if vs_opp is not None and stats.vs_opponent_games >= 2:
        if vs_opp < season_avg * 0.85:
            analysis.factors["historical_struggle"] = config.weights["historical_struggle"]
            analysis.adjustments["historical_struggle"] = config.adjustments["historical_struggle"]
            analysis.reasons.append(
                f"📉 Struggles vs {opponent_abbrev}: {vs_opp:.1f} avg ({stats.vs_opponent_games} games)"
            )
    
    # 6. ELITE DEFENDER MATCHUP
    elite_defender = _check_elite_defender(conn, opponent_abbrev, stats.position, game_date)
    if elite_defender:
        analysis.factors["elite_defender_matchup"] = config.weights["elite_defender_matchup"]
        analysis.adjustments["elite_defender_matchup"] = config.adjustments["elite_defender_matchup"]
        analysis.reasons.append(f"🔒 Facing elite defender: {elite_defender}")
    
    # 7. AWAY DISADVANTAGE
    if not is_home and defense.data_available:
        if defense.get_rating(pt) in ["elite", "good"]:
            analysis.factors["away_disadvantage"] = config.weights["away_disadvantage"]
            analysis.adjustments["away_disadvantage"] = config.adjustments["away_disadvantage"]
    
    return analysis


def _generate_pick(
    conn: sqlite3.Connection,
    stats: PlayerStatsV13Under,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    is_home: bool,
    config: ModelConfigV13Under,
) -> Optional[PropPickV13Under]:
    """Generate an UNDER pick for a player/prop."""
    pt = prop_type.lower()
    
    # Check minimum average for prop
    min_avg = config.min_avg_for_prop.get(pt, 5.0)
    season_avg = stats.season.get(pt, 0)
    if season_avg < min_avg:
        return None
    
    # Analyze factors
    analysis = _analyze_under_factors(
        conn, stats, pt, opponent_abbrev, game_date, is_home, config
    )
    
    # Must have at least one factor
    if analysis.factor_count == 0:
        return None
    
    # Quality filter: require defense factor or multiple factors
    if config.require_defense_factor:
        if not analysis.has_elite_defense and not analysis.has_good_defense:
            # No defense factor - require multiple other factors
            if analysis.factor_count < config.require_multiple_factors + 1:
                return None
    
    # Get line
    sportsbook_result = _get_sportsbook_line(conn, stats.player_id, stats.player_name, pt, game_date)
    
    if sportsbook_result:
        line = sportsbook_result[0]
        line_source = "sportsbook"
        book = sportsbook_result[1]
    else:
        line = stats.l10.get(pt, 0) * config.derived_line_adjustment
        line_source = "derived"
        book = "derived"
    
    if line < 3.0:
        return None
    
    # Calculate projection
    projected = season_avg * analysis.total_adjustment
    
    # Calculate edge
    edge = (line - projected) / line * 100 if line > 0 else 0
    
    # Check minimum edge
    min_edge = config.min_edge_sportsbook if line_source == "sportsbook" else config.min_edge_derived
    if edge < min_edge:
        return None
    
    # Calculate confidence
    raw_score = analysis.total_weight
    
    # Map raw score to confidence
    if raw_score >= 50:
        confidence = 88 + min(12, (raw_score - 50) * 0.25)
    elif raw_score >= 30:
        confidence = 70 + (raw_score - 30) * 0.9
    elif raw_score >= 20:
        confidence = 60 + (raw_score - 20) * 1.0
    else:
        confidence = 45 + raw_score
    
    # Sportsbook bonus
    if line_source == "sportsbook":
        confidence += config.sportsbook_line_confidence_boost
    
    # Edge bonus
    confidence += min(edge / 3, 8)
    
    confidence = min(100, confidence)
    
    # Check minimum confidence
    if confidence < config.min_confidence:
        return None
    
    # Determine tier
    if confidence >= config.premium_confidence and analysis.has_elite_defense and analysis.has_cold_streak:
        tier = "PREMIUM"
    elif confidence >= config.high_confidence:
        tier = "HIGH"
    elif confidence >= config.medium_confidence:
        tier = "MEDIUM"
    else:
        return None
    
    # Get defense context for reporting
    defense = _get_defense_profile(conn, opponent_abbrev, stats.position, config)
    
    return PropPickV13Under(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        game_date=game_date,
        position=stats.position,
        prop_type=prop_type.upper(),
        direction="UNDER",
        line=round(line, 1),
        line_source=line_source,
        book=book,
        season_avg=round(season_avg, 1),
        l5_avg=round(stats.l5.get(pt, 0), 1),
        projected=round(projected, 1),
        factors=analysis.factors,
        adjustment_factor=analysis.total_adjustment,
        edge_vs_line=round(edge, 1),
        raw_score=raw_score,
        confidence_score=round(confidence, 1),
        confidence_tier=tier,
        defense_rank=defense.get_rank(pt),
        defense_rating=defense.get_rating(pt),
        reasons=analysis.reasons,
        warnings=analysis.warnings,
    )


def _generate_game_picks(
    conn: sqlite3.Connection,
    game_date: str,
    team1_name: str,
    team2_name: str,
    config: ModelConfigV13Under,
    coverage_stats: Dict[str, int],
) -> List[PropPickV13Under]:
    """Generate UNDER picks for a game."""
    t1_abbrev = abbrev_from_team_name(team1_name) or ""
    t2_abbrev = abbrev_from_team_name(team2_name) or ""
    
    injured = _get_injured_players(conn, game_date)
    
    all_picks = []
    player_picks = {}
    
    for team_name, opp_abbrev, team_abbrev, is_home in [
        (team1_name, t2_abbrev, t1_abbrev, False),
        (team2_name, t1_abbrev, t2_abbrev, True),
    ]:
        team = conn.execute("SELECT id FROM teams WHERE name = ?", (team_name,)).fetchone()
        if not team:
            continue
        
        players = conn.execute(
            """
            SELECT b.player_id, AVG(b.minutes) as avg_min
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.team_id = ? AND g.game_date < ? AND b.minutes > ?
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
            
            stats = _load_player_stats(conn, player_id, game_date, opp_abbrev, config)
            if not stats:
                continue
            
            coverage_stats["analyzed"] += 1
            
            for pt in config.prop_types:
                if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                    break
                
                pick = _generate_pick(conn, stats, pt, opp_abbrev, game_date, is_home, config)
                
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

def get_daily_picks_v13_under(
    game_date: str,
    config: Optional[ModelConfigV13Under] = None,
    db_path: str = "data/db/nba_props.sqlite3",
) -> DailyPicksV13Under:
    """Generate UNDER picks for all games on a date."""
    if config is None:
        config = ModelConfigV13Under()
    
    db = Db(Path(db_path))
    daily = DailyPicksV13Under(date=game_date, games=0)
    
    all_picks = []
    coverage_stats = {"sportsbook": 0, "derived": 0, "analyzed": 0}
    
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
    
    daily.picks = all_picks[:config.max_picks_per_day]
    daily.players_with_sportsbook_lines = coverage_stats["sportsbook"]
    daily.players_with_derived_lines = coverage_stats["derived"]
    daily.total_players_analyzed = coverage_stats["analyzed"]
    
    return daily


def run_backtest_v13_under(
    start_date: str,
    end_date: str,
    config: Optional[ModelConfigV13Under] = None,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> BacktestResultV13Under:
    """Run comprehensive backtest for Model V13 Under."""
    if config is None:
        config = ModelConfigV13Under()
    
    db = Db(Path(db_path))
    result = BacktestResultV13Under(
        start_date=start_date,
        end_date=end_date,
        config=config,
    )
    
    factor_hits = {}
    factor_total = {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL V13 UNDER BACKTEST: {start_date} to {end_date}")
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
            coverage_stats = {"sportsbook": 0, "derived": 0, "analyzed": 0}
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
                    WHERE bp.player_id = ? AND g.game_date = ? AND bp.minutes > 0
                    """,
                    (pick.player_id, game_date),
                ).fetchone()
                
                if not actual:
                    continue
                
                actual_value = actual[pick.prop_type.lower()]
                if actual_value is None:
                    continue
                
                pick.actual_value = actual_value
                pick.margin = pick.line - actual_value  # Positive = UNDER hit
                pick.hit = actual_value < pick.line
                
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
                    result.medium_picks += 1
                    if pick.hit:
                        result.medium_hits += 1
                
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
                
                # By defense rating
                if pick.defense_rating == "elite":
                    result.elite_defense_picks += 1
                    if pick.hit:
                        result.elite_defense_hits += 1
                elif pick.defense_rating == "good":
                    result.good_defense_picks += 1
                    if pick.hit:
                        result.good_defense_hits += 1
                
                # Track combined factor
                if "defense_elite" in pick.factors:
                    if "cold_streak_severe" in pick.factors or "cold_streak_mild" in pick.factors:
                        result.elite_plus_cold_picks += 1
                        if pick.hit:
                            result.elite_plus_cold_hits += 1
                
                # Cold streak tracking
                if "cold_streak_severe" in pick.factors or "cold_streak_mild" in pick.factors:
                    result.cold_streak_picks += 1
                    if pick.hit:
                        result.cold_streak_hits += 1
                
                # Factor effectiveness
                for factor in pick.factors.keys():
                    if factor not in factor_total:
                        factor_total[factor] = 0
                        factor_hits[factor] = 0
                    factor_total[factor] += 1
                    if pick.hit:
                        factor_hits[factor] += 1
                
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
    
    # Calculate factor effectiveness
    for factor in factor_total:
        total = factor_total[factor]
        hits = factor_hits[factor]
        result.factor_effectiveness[factor] = {
            "picks": total,
            "hits": hits,
            "hit_rate": hits / total if total > 0 else 0.0,
        }
    
    if verbose:
        print()
        print(result.summary())
    
    return result


# ============================================================================
# CLI Integration
# ============================================================================

def main():
    """Command-line interface for Model V13 Under."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V13 Under - NBA Props")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_v13_under(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_v13_under(args.date)
        print(picks.summary())
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_v13_under(today)
        print(picks.summary())


if __name__ == "__main__":
    main()
