"""
Advanced Defense Analysis Engine

Provides sophisticated analysis of:
- Position-based defensive performance
- How teams defend different player archetypes
- Player performance vs specific teams
- Defensive strengths/weaknesses by category
- Matchup edge calculations
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta

from ..team_aliases import normalize_team_abbrev, abbrev_from_team_name
from ..standings import _team_ids_by_abbrev


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PositionDefenseProfile:
    """How a team defends a specific position."""
    team_abbrev: str
    position: str  # G, F, C
    sample_size: int
    
    # Stats allowed to this position
    pts_allowed_avg: float
    reb_allowed_avg: float
    ast_allowed_avg: float
    
    # League averages for comparison
    league_pts_avg: float
    league_reb_avg: float
    league_ast_avg: float
    
    # Factors (>1 = allows more, <1 = allows less)
    pts_factor: float
    reb_factor: float
    ast_factor: float
    
    # Rankings (1 = best defense at this position)
    pts_rank: int = 0
    reb_rank: int = 0
    ast_rank: int = 0
    
    # Rating: "elite", "good", "average", "poor", "weak"
    pts_rating: str = "average"
    reb_rating: str = "average"
    ast_rating: str = "average"


@dataclass
class ArchetypeDefenseProfile:
    """How a team defends specific player archetypes."""
    team_abbrev: str
    archetype: str
    sample_size: int
    
    # Stats allowed
    pts_allowed_avg: float
    reb_allowed_avg: float
    ast_allowed_avg: float
    
    # Factors vs league average for this archetype
    pts_factor: float
    reb_factor: float
    ast_factor: float


@dataclass
class PlayerVsTeamProfile:
    """Historical performance of a player against a specific team."""
    player_name: str
    opponent_abbrev: str
    games_played: int
    
    # Stats vs this team
    pts_avg: float
    reb_avg: float
    ast_avg: float
    min_avg: float
    
    # Overall averages for comparison
    overall_pts_avg: float
    overall_reb_avg: float
    overall_ast_avg: float
    
    # Differential (positive = performs better vs this team)
    pts_diff: float
    reb_diff: float
    ast_diff: float
    
    # Last 3 games vs this team
    recent_games: list[dict] = field(default_factory=list)
    
    # Has significant history (3+ games)
    has_history: bool = False


@dataclass
class PlayerTrend:
    """Recent performance trend for a player."""
    player_name: str
    player_id: int
    team_abbrev: str
    
    # Recent averages (last 5 games)
    recent_pts: float
    recent_reb: float
    recent_ast: float
    recent_min: float
    recent_games: int
    
    # Season averages
    season_pts: float
    season_reb: float
    season_ast: float
    season_games: int
    
    # Trend direction and magnitude
    pts_trend: str  # "hot", "cold", "stable"
    reb_trend: str
    ast_trend: str
    
    # Percent change (positive = trending up)
    pts_change_pct: float
    reb_change_pct: float
    ast_change_pct: float
    
    # Consistency (standard deviation)
    pts_consistency: float
    reb_consistency: float
    ast_consistency: float
    
    # Recent game log
    game_log: list[dict] = field(default_factory=list)


@dataclass
class MatchupEdge:
    """Calculated edge for a specific player matchup."""
    player_name: str
    player_id: int
    team_abbrev: str
    opponent_abbrev: str
    
    # Prop type and direction
    prop_type: str  # PTS, REB, AST
    direction: str  # OVER, UNDER
    
    # Values
    baseline_projection: float
    adjusted_projection: float
    adjustment_pct: float
    
    # Confidence factors
    confidence_score: float  # 0-100
    confidence_tier: str  # "HIGH", "MEDIUM", "LOW"
    
    # Individual factors
    factors: dict = field(default_factory=dict)
    
    # Reasoning
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    # Context
    is_close_game: bool = False
    spread: Optional[float] = None
    over_under: Optional[float] = None


@dataclass 
class ComprehensiveMatchupReport:
    """Full matchup analysis between two teams."""
    away_abbrev: str
    home_abbrev: str
    game_date: str
    
    # Team context
    away_b2b: bool
    home_b2b: bool
    away_rest_days: int
    home_rest_days: int
    spread: Optional[float]
    over_under: Optional[float]
    is_close_game: bool
    
    # Defense profiles
    away_defense_vs_guards: Optional[PositionDefenseProfile]
    away_defense_vs_forwards: Optional[PositionDefenseProfile]
    away_defense_vs_centers: Optional[PositionDefenseProfile]
    home_defense_vs_guards: Optional[PositionDefenseProfile]
    home_defense_vs_forwards: Optional[PositionDefenseProfile]
    home_defense_vs_centers: Optional[PositionDefenseProfile]
    
    # Player projections with adjustments
    away_player_projections: list[dict] = field(default_factory=list)
    home_player_projections: list[dict] = field(default_factory=list)
    
    # Top edges
    best_over_plays: list[MatchupEdge] = field(default_factory=list)
    best_under_plays: list[MatchupEdge] = field(default_factory=list)
    
    # Players to avoid
    avoid_players: list[dict] = field(default_factory=list)
    
    # Key storylines
    key_matchups: list[str] = field(default_factory=list)


# ============================================================================
# Position-Based Defense Analysis
# ============================================================================

def get_position_defense_profile(
    conn: sqlite3.Connection,
    team_abbrev: str,
    position: str,
    min_games: int = 5,
) -> Optional[PositionDefenseProfile]:
    """
    Analyze how a team defends a specific position.
    
    Args:
        conn: Database connection
        team_abbrev: Team abbreviation
        position: Position to analyze (G, F, C)
        min_games: Minimum games required
    
    Returns:
        PositionDefenseProfile or None if insufficient data
    """
    team_abbrev = normalize_team_abbrev(team_abbrev)
    team_ids_map = _team_ids_by_abbrev(conn)
    team_ids = team_ids_map.get(team_abbrev, [])
    
    if not team_ids:
        return None
    
    pos = position.upper()[:1] if position else ""
    if pos not in ("G", "F", "C"):
        return None
    
    placeholders = ",".join(["?"] * len(team_ids))
    
    # Get stats of players at this position AGAINST this team
    # (when this team was the opponent)
    rows = conn.execute(
        f"""
        SELECT 
            b.pts, b.reb, b.ast, b.minutes
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.pos = ?
          AND b.minutes IS NOT NULL
          AND b.minutes > 10
          AND b.team_id NOT IN ({placeholders})
          AND (g.team1_id IN ({placeholders}) OR g.team2_id IN ({placeholders}))
        """,
        (pos, *team_ids, *team_ids, *team_ids),
    ).fetchall()
    
    if len(rows) < min_games:
        return None
    
    # Calculate averages allowed
    pts_allowed = [r["pts"] or 0 for r in rows]
    reb_allowed = [r["reb"] or 0 for r in rows]
    ast_allowed = [r["ast"] or 0 for r in rows]
    
    pts_avg = sum(pts_allowed) / len(pts_allowed)
    reb_avg = sum(reb_allowed) / len(reb_allowed)
    ast_avg = sum(ast_allowed) / len(ast_allowed)
    
    # Get league averages for this position
    league_row = conn.execute(
        """
        SELECT 
            AVG(pts) as league_pts,
            AVG(reb) as league_reb,
            AVG(ast) as league_ast
        FROM boxscore_player
        WHERE pos = ?
          AND minutes IS NOT NULL
          AND minutes > 10
        """,
        (pos,),
    ).fetchone()
    
    league_pts = league_row["league_pts"] or pts_avg
    league_reb = league_row["league_reb"] or reb_avg
    league_ast = league_row["league_ast"] or ast_avg
    
    # Calculate factors
    pts_factor = pts_avg / league_pts if league_pts > 0 else 1.0
    reb_factor = reb_avg / league_reb if league_reb > 0 else 1.0
    ast_factor = ast_avg / league_ast if league_ast > 0 else 1.0
    
    # Determine ratings
    def get_rating(factor):
        if factor <= 0.92:
            return "elite"
        elif factor <= 0.97:
            return "good"
        elif factor <= 1.03:
            return "average"
        elif factor <= 1.08:
            return "poor"
        else:
            return "weak"
    
    return PositionDefenseProfile(
        team_abbrev=team_abbrev,
        position=pos,
        sample_size=len(rows),
        pts_allowed_avg=round(pts_avg, 1),
        reb_allowed_avg=round(reb_avg, 1),
        ast_allowed_avg=round(ast_avg, 1),
        league_pts_avg=round(league_pts, 1),
        league_reb_avg=round(league_reb, 1),
        league_ast_avg=round(league_ast, 1),
        pts_factor=round(pts_factor, 3),
        reb_factor=round(reb_factor, 3),
        ast_factor=round(ast_factor, 3),
        pts_rating=get_rating(pts_factor),
        reb_rating=get_rating(reb_factor),
        ast_rating=get_rating(ast_factor),
    )


def get_all_position_defense_profiles(
    conn: sqlite3.Connection,
    team_abbrev: str,
) -> dict[str, PositionDefenseProfile]:
    """Get defense profiles for all positions for a team."""
    profiles = {}
    for pos in ["G", "F", "C"]:
        profile = get_position_defense_profile(conn, team_abbrev, pos)
        if profile:
            profiles[pos] = profile
    return profiles


def rank_position_defense_profiles(
    conn: sqlite3.Connection,
    position: str,
) -> list[PositionDefenseProfile]:
    """
    Rank all teams by their defense against a specific position.
    
    Returns list sorted by pts_factor (best defense first).
    """
    from ..standings import ALL_ABBREVS
    
    profiles = []
    for abbrev in ALL_ABBREVS:
        profile = get_position_defense_profile(conn, abbrev, position)
        if profile:
            profiles.append(profile)
    
    # Sort by pts_factor (lower = better defense)
    profiles.sort(key=lambda p: p.pts_factor)
    
    # Assign ranks
    for i, p in enumerate(profiles, 1):
        p.pts_rank = i
    
    profiles_by_reb = sorted(profiles, key=lambda p: p.reb_factor)
    for i, p in enumerate(profiles_by_reb, 1):
        p.reb_rank = i
    
    profiles_by_ast = sorted(profiles, key=lambda p: p.ast_factor)
    for i, p in enumerate(profiles_by_ast, 1):
        p.ast_rank = i
    
    return profiles


# ============================================================================
# Player vs Team History
# ============================================================================

def get_player_vs_team_profile(
    conn: sqlite3.Connection,
    player_name: str,
    opponent_abbrev: str,
) -> Optional[PlayerVsTeamProfile]:
    """
    Get historical performance of a player against a specific team.
    """
    opponent_abbrev = normalize_team_abbrev(opponent_abbrev)
    team_ids_map = _team_ids_by_abbrev(conn)
    opponent_ids = team_ids_map.get(opponent_abbrev, [])
    
    if not opponent_ids:
        return None
    
    # Find player
    player_row = conn.execute(
        "SELECT id, name FROM players WHERE name LIKE ?",
        (f"%{player_name}%",),
    ).fetchone()
    
    if not player_row:
        return None
    
    player_id = player_row["id"]
    full_name = player_row["name"]
    
    placeholders = ",".join(["?"] * len(opponent_ids))
    
    # Get games against this opponent
    vs_rows = conn.execute(
        f"""
        SELECT 
            g.game_date, b.pts, b.reb, b.ast, b.minutes
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND b.minutes IS NOT NULL
          AND b.minutes > 5
          AND (g.team1_id IN ({placeholders}) OR g.team2_id IN ({placeholders}))
        ORDER BY g.game_date DESC
        """,
        (player_id, *opponent_ids, *opponent_ids),
    ).fetchall()
    
    if not vs_rows:
        return None
    
    # Get overall averages
    overall_row = conn.execute(
        """
        SELECT 
            AVG(pts) as avg_pts,
            AVG(reb) as avg_reb,
            AVG(ast) as avg_ast,
            AVG(minutes) as avg_min
        FROM boxscore_player
        WHERE player_id = ?
          AND minutes IS NOT NULL
          AND minutes > 5
        """,
        (player_id,),
    ).fetchone()
    
    # Calculate vs team averages
    pts_vs = sum(r["pts"] or 0 for r in vs_rows) / len(vs_rows)
    reb_vs = sum(r["reb"] or 0 for r in vs_rows) / len(vs_rows)
    ast_vs = sum(r["ast"] or 0 for r in vs_rows) / len(vs_rows)
    min_vs = sum(r["minutes"] or 0 for r in vs_rows) / len(vs_rows)
    
    overall_pts = overall_row["avg_pts"] or pts_vs
    overall_reb = overall_row["avg_reb"] or reb_vs
    overall_ast = overall_row["avg_ast"] or ast_vs
    
    return PlayerVsTeamProfile(
        player_name=full_name,
        opponent_abbrev=opponent_abbrev,
        games_played=len(vs_rows),
        pts_avg=round(pts_vs, 1),
        reb_avg=round(reb_vs, 1),
        ast_avg=round(ast_vs, 1),
        min_avg=round(min_vs, 1),
        overall_pts_avg=round(overall_pts, 1),
        overall_reb_avg=round(overall_reb, 1),
        overall_ast_avg=round(overall_ast, 1),
        pts_diff=round(pts_vs - overall_pts, 1),
        reb_diff=round(reb_vs - overall_reb, 1),
        ast_diff=round(ast_vs - overall_ast, 1),
        recent_games=[
            {
                "date": r["game_date"],
                "pts": r["pts"],
                "reb": r["reb"],
                "ast": r["ast"],
                "min": round(r["minutes"], 1) if r["minutes"] else 0,
            }
            for r in vs_rows[:5]
        ],
        has_history=len(vs_rows) >= 3,
    )


# ============================================================================
# Player Trend Analysis
# ============================================================================

def get_player_trend(
    conn: sqlite3.Connection,
    player_id: int,
    recent_games: int = 5,
) -> Optional[PlayerTrend]:
    """
    Analyze a player's recent performance trends.
    """
    # Get player info
    player_row = conn.execute(
        "SELECT id, name FROM players WHERE id = ?", (player_id,)
    ).fetchone()
    if not player_row:
        return None
    
    # Get all games
    rows = conn.execute(
        """
        SELECT 
            g.game_date, b.pts, b.reb, b.ast, b.minutes, t.name as team
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        JOIN teams t ON t.id = b.team_id
        WHERE b.player_id = ?
          AND b.minutes IS NOT NULL
          AND b.minutes > 5
        ORDER BY g.game_date DESC
        """,
        (player_id,),
    ).fetchall()
    
    if len(rows) < 3:
        return None
    
    # Get team abbrev from most recent game
    team_name = rows[0]["team"]
    team_abbrev = abbrev_from_team_name(team_name) or ""
    
    # Split into recent and season
    recent_rows = rows[:recent_games]
    all_rows = rows
    
    # Calculate recent averages
    recent_pts = sum(r["pts"] or 0 for r in recent_rows) / len(recent_rows)
    recent_reb = sum(r["reb"] or 0 for r in recent_rows) / len(recent_rows)
    recent_ast = sum(r["ast"] or 0 for r in recent_rows) / len(recent_rows)
    recent_min = sum(r["minutes"] or 0 for r in recent_rows) / len(recent_rows)
    
    # Calculate season averages
    season_pts = sum(r["pts"] or 0 for r in all_rows) / len(all_rows)
    season_reb = sum(r["reb"] or 0 for r in all_rows) / len(all_rows)
    season_ast = sum(r["ast"] or 0 for r in all_rows) / len(all_rows)
    
    # Calculate change percentages
    def pct_change(recent, season):
        if season == 0:
            return 0
        return ((recent - season) / season) * 100
    
    pts_change = pct_change(recent_pts, season_pts)
    reb_change = pct_change(recent_reb, season_reb)
    ast_change = pct_change(recent_ast, season_ast)
    
    # Determine trends
    def get_trend(change_pct):
        if change_pct >= 15:
            return "hot"
        elif change_pct <= -15:
            return "cold"
        else:
            return "stable"
    
    # Calculate consistency (std dev)
    import statistics
    pts_values = [r["pts"] or 0 for r in recent_rows]
    reb_values = [r["reb"] or 0 for r in recent_rows]
    ast_values = [r["ast"] or 0 for r in recent_rows]
    
    pts_std = statistics.stdev(pts_values) if len(pts_values) > 1 else 0
    reb_std = statistics.stdev(reb_values) if len(reb_values) > 1 else 0
    ast_std = statistics.stdev(ast_values) if len(ast_values) > 1 else 0
    
    return PlayerTrend(
        player_name=player_row["name"],
        player_id=player_id,
        team_abbrev=team_abbrev,
        recent_pts=round(recent_pts, 1),
        recent_reb=round(recent_reb, 1),
        recent_ast=round(recent_ast, 1),
        recent_min=round(recent_min, 1),
        recent_games=len(recent_rows),
        season_pts=round(season_pts, 1),
        season_reb=round(season_reb, 1),
        season_ast=round(season_ast, 1),
        season_games=len(all_rows),
        pts_trend=get_trend(pts_change),
        reb_trend=get_trend(reb_change),
        ast_trend=get_trend(ast_change),
        pts_change_pct=round(pts_change, 1),
        reb_change_pct=round(reb_change, 1),
        ast_change_pct=round(ast_change, 1),
        pts_consistency=round(pts_std, 1),
        reb_consistency=round(reb_std, 1),
        ast_consistency=round(ast_std, 1),
        game_log=[
            {
                "date": r["game_date"],
                "pts": r["pts"],
                "reb": r["reb"],
                "ast": r["ast"],
                "min": round(r["minutes"], 1) if r["minutes"] else 0,
            }
            for r in recent_rows
        ],
    )


# ============================================================================
# Comprehensive Matchup Edge Calculation
# ============================================================================

def calculate_matchup_edge(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    team_abbrev: str,
    opponent_abbrev: str,
    prop_type: str,
    baseline_value: float,
    is_b2b: bool = False,
    rest_days: int = 1,
    spread: Optional[float] = None,
    over_under: Optional[float] = None,
) -> MatchupEdge:
    """
    Calculate comprehensive matchup edge for a player prop.
    
    Considers:
    - Position-based defensive matchup
    - Historical performance vs this team
    - Recent trend (hot/cold)
    - Back-to-back/rest factors
    - Game context (spread, expected pace)
    - Elite defender presence
    """
    factors = {}
    reasons = []
    warnings = []
    adjustment = 1.0
    
    # Get player's position
    pos_row = conn.execute(
        """
        SELECT pos FROM boxscore_player 
        WHERE player_id = ? AND pos IS NOT NULL AND pos != ''
        ORDER BY game_id DESC LIMIT 1
        """,
        (player_id,),
    ).fetchone()
    position = pos_row["pos"] if pos_row else "G"
    
    # ============================
    # 1. Position Defense Factor
    # ============================
    pos_defense = get_position_defense_profile(conn, opponent_abbrev, position)
    if pos_defense:
        if prop_type == "PTS":
            pos_factor = pos_defense.pts_factor
            rating = pos_defense.pts_rating
        elif prop_type == "REB":
            pos_factor = pos_defense.reb_factor
            rating = pos_defense.reb_rating
        else:
            pos_factor = pos_defense.ast_factor
            rating = pos_defense.ast_rating
        
        # Clamp to reasonable range
        pos_factor = max(0.85, min(1.15, pos_factor))
        factors["position_defense"] = pos_factor
        adjustment *= pos_factor
        
        if rating in ("elite", "good"):
            warnings.append(f"{opponent_abbrev} has {rating} {position} defense")
        elif rating in ("poor", "weak"):
            reasons.append(f"{opponent_abbrev} allows extra {prop_type} to {position}s (+{int((pos_factor-1)*100)}%)")
    
    # ============================
    # 2. Player vs Team History
    # ============================
    vs_team = get_player_vs_team_profile(conn, player_name, opponent_abbrev)
    if vs_team and vs_team.has_history:
        if prop_type == "PTS":
            hist_diff = vs_team.pts_diff
            hist_avg = vs_team.pts_avg
        elif prop_type == "REB":
            hist_diff = vs_team.reb_diff
            hist_avg = vs_team.reb_avg
        else:
            hist_diff = vs_team.ast_diff
            hist_avg = vs_team.ast_avg
        
        # Historical performance adjustment (weighted)
        if abs(hist_diff) >= 2:
            hist_factor = 1 + (hist_diff / baseline_value * 0.5) if baseline_value > 0 else 1.0
            hist_factor = max(0.85, min(1.15, hist_factor))
            factors["historical"] = hist_factor
            adjustment *= hist_factor
            
            if hist_diff > 2:
                reasons.append(f"Averages {hist_avg} {prop_type} vs {opponent_abbrev} (+{hist_diff:.1f} vs avg)")
            elif hist_diff < -2:
                warnings.append(f"Only {hist_avg} {prop_type} vs {opponent_abbrev} ({hist_diff:.1f} vs avg)")
    
    # ============================
    # 3. Recent Trend
    # ============================
    trend = get_player_trend(conn, player_id)
    if trend:
        if prop_type == "PTS":
            trend_pct = trend.pts_change_pct
            trend_dir = trend.pts_trend
            consistency = trend.pts_consistency
        elif prop_type == "REB":
            trend_pct = trend.reb_change_pct
            trend_dir = trend.reb_trend
            consistency = trend.reb_consistency
        else:
            trend_pct = trend.ast_change_pct
            trend_dir = trend.ast_trend
            consistency = trend.ast_consistency
        
        # Trend adjustment (smaller weight)
        if abs(trend_pct) >= 10:
            trend_factor = 1 + (trend_pct / 100 * 0.3)
            trend_factor = max(0.90, min(1.10, trend_factor))
            factors["trend"] = trend_factor
            adjustment *= trend_factor
            
            if trend_dir == "hot":
                reasons.append(f"🔥 Hot streak: {trend_pct:+.0f}% {prop_type} over last {trend.recent_games} games")
            elif trend_dir == "cold":
                warnings.append(f"❄️ Cold streak: {trend_pct:.0f}% {prop_type} over last {trend.recent_games} games")
        
        # Consistency warning
        if consistency > baseline_value * 0.3:
            warnings.append(f"Inconsistent: ±{consistency:.1f} {prop_type} variance")
    
    # ============================
    # 4. Rest/B2B Factor
    # ============================
    if is_b2b:
        b2b_factor = 0.94
        factors["back_to_back"] = b2b_factor
        adjustment *= b2b_factor
        warnings.append("Playing back-to-back (-6%)")
    elif rest_days >= 3:
        rest_factor = 1.03
        factors["well_rested"] = rest_factor
        adjustment *= rest_factor
        reasons.append(f"Well rested ({rest_days} days off)")
    
    # ============================
    # 5. Game Context
    # ============================
    is_close_game = spread is not None and abs(spread) <= 6
    
    if is_close_game:
        # Close games = more playing time for starters
        close_factor = 1.03
        factors["close_game"] = close_factor
        adjustment *= close_factor
        reasons.append("Expected close game (+3% for starters)")
    elif spread is not None and abs(spread) > 10:
        # Blowout risk = potential rest
        blowout_factor = 0.95
        factors["blowout_risk"] = blowout_factor
        adjustment *= blowout_factor
        warnings.append(f"Blowout risk (spread {spread:+.1f})")
    
    # ============================
    # 6. Elite Defender Check
    # ============================
    from .roster import should_avoid_betting_over, get_roster_for_team, get_player_profile
    
    try:
        opponent_roster = [p.name for p in get_roster_for_team(opponent_abbrev)]
        avoid, defenders = should_avoid_betting_over(player_name, opponent_roster)
        if avoid and prop_type == "PTS":
            defender_factor = 0.94
            factors["elite_defender"] = defender_factor
            adjustment *= defender_factor
            warnings.append(f"⚠️ Elite defender: {', '.join(defenders[:2])}")
    except Exception:
        pass
    
    # ============================
    # Calculate Final Values
    # ============================
    adjusted_value = baseline_value * adjustment
    adjustment_pct = (adjustment - 1) * 100
    
    # Determine direction
    if adjusted_value > baseline_value:
        direction = "OVER"
    elif adjusted_value < baseline_value:
        direction = "UNDER"
    else:
        direction = "PASS"
    
    # Calculate confidence score
    confidence_score = 50  # Base
    
    # Add points for multiple positive factors
    if len(reasons) >= 2:
        confidence_score += 15
    if len(reasons) >= 3:
        confidence_score += 10
    
    # Subtract for warnings
    confidence_score -= len(warnings) * 8
    
    # Add for position defense rating
    if pos_defense:
        rating = pos_defense.pts_rating if prop_type == "PTS" else pos_defense.reb_rating
        if rating in ("elite", "good"):
            if direction == "UNDER":
                confidence_score += 15
            else:
                confidence_score -= 10
        elif rating in ("poor", "weak"):
            if direction == "OVER":
                confidence_score += 15
            else:
                confidence_score -= 10
    
    # Add for historical consistency
    if vs_team and vs_team.has_history:
        confidence_score += 10
    
    # Add for trend alignment
    if trend:
        trend_dir = trend.pts_trend if prop_type == "PTS" else trend.reb_trend
        if (trend_dir == "hot" and direction == "OVER") or (trend_dir == "cold" and direction == "UNDER"):
            confidence_score += 12
    
    # Clamp confidence
    confidence_score = max(0, min(100, confidence_score))
    
    # Determine tier
    if confidence_score >= 75:
        confidence_tier = "HIGH"
    elif confidence_score >= 55:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "LOW"
    
    return MatchupEdge(
        player_name=player_name,
        player_id=player_id,
        team_abbrev=team_abbrev,
        opponent_abbrev=opponent_abbrev,
        prop_type=prop_type,
        direction=direction,
        baseline_projection=round(baseline_value, 1),
        adjusted_projection=round(adjusted_value, 1),
        adjustment_pct=round(adjustment_pct, 1),
        confidence_score=confidence_score,
        confidence_tier=confidence_tier,
        factors=factors,
        reasons=reasons,
        warnings=warnings,
        is_close_game=is_close_game,
        spread=spread,
        over_under=over_under,
    )


# ============================================================================
# Team Defense Summary
# ============================================================================

def get_team_defense_summary(
    conn: sqlite3.Connection,
    team_abbrev: str,
) -> dict:
    """
    Get a comprehensive summary of a team's defensive performance.
    """
    team_abbrev = normalize_team_abbrev(team_abbrev)
    
    # Get position profiles
    guard_defense = get_position_defense_profile(conn, team_abbrev, "G")
    forward_defense = get_position_defense_profile(conn, team_abbrev, "F")
    center_defense = get_position_defense_profile(conn, team_abbrev, "C")
    
    # Overall rating
    all_factors = []
    if guard_defense:
        all_factors.append(guard_defense.pts_factor)
    if forward_defense:
        all_factors.append(forward_defense.pts_factor)
    if center_defense:
        all_factors.append(center_defense.pts_factor)
    
    overall_factor = sum(all_factors) / len(all_factors) if all_factors else 1.0
    
    def get_overall_rating(factor):
        if factor <= 0.94:
            return "Elite Defense"
        elif factor <= 0.98:
            return "Good Defense"
        elif factor <= 1.02:
            return "Average Defense"
        elif factor <= 1.06:
            return "Below Average Defense"
        else:
            return "Weak Defense"
    
    # Find strengths and weaknesses
    strengths = []
    weaknesses = []
    
    for name, profile in [("Guards", guard_defense), ("Forwards", forward_defense), ("Centers", center_defense)]:
        if not profile:
            continue
        
        if profile.pts_rating in ("elite", "good"):
            strengths.append(f"vs {name} ({profile.pts_rating})")
        elif profile.pts_rating in ("poor", "weak"):
            weaknesses.append(f"vs {name} ({profile.pts_rating})")
    
    return {
        "team_abbrev": team_abbrev,
        "overall_rating": get_overall_rating(overall_factor),
        "overall_factor": round(overall_factor, 3),
        "guard_defense": {
            "rating": guard_defense.pts_rating if guard_defense else "unknown",
            "pts_factor": guard_defense.pts_factor if guard_defense else 1.0,
            "reb_factor": guard_defense.reb_factor if guard_defense else 1.0,
            "ast_factor": guard_defense.ast_factor if guard_defense else 1.0,
        } if guard_defense else None,
        "forward_defense": {
            "rating": forward_defense.pts_rating if forward_defense else "unknown",
            "pts_factor": forward_defense.pts_factor if forward_defense else 1.0,
            "reb_factor": forward_defense.reb_factor if forward_defense else 1.0,
            "ast_factor": forward_defense.ast_factor if forward_defense else 1.0,
        } if forward_defense else None,
        "center_defense": {
            "rating": center_defense.pts_rating if center_defense else "unknown",
            "pts_factor": center_defense.pts_factor if center_defense else 1.0,
            "reb_factor": center_defense.reb_factor if center_defense else 1.0,
            "ast_factor": center_defense.ast_factor if center_defense else 1.0,
        } if center_defense else None,
        "strengths": strengths,
        "weaknesses": weaknesses,
    }

