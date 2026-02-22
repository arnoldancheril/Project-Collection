"""
Model V12 Under - Specialized UNDER Prediction Model
======================================================

This is the UNDER-specialized model of the V12 dual-model system. It focuses
exclusively on identifying high-probability UNDER opportunities through:

1. Defense vs Position (DVP) analysis - PRIMARY factor
2. Cold streak detection - Player trending down
3. Fatigue factors - Back-to-back games, heavy minutes
4. Injury rust - Players returning from injury
5. Historical struggles - Poor performance vs opponent

CORE PHILOSOPHY:
----------------
UNDER picks are MORE PREDICTABLE than OVER picks because:
- Negative factors compound more reliably than positive ones
- Elite defenses consistently limit player production
- Cold streaks tend to persist longer than hot streaks
- Fatigue effects are measurable and consistent
- High variance players are more likely to hit unders

The model uses a FACTOR-BASED SCORING SYSTEM where multiple negative
factors combine to increase confidence. The combination of factors is
more predictive than any single factor alone.

KEY FACTORS AND WEIGHTS:
------------------------
PRIMARY FACTORS (Highest predictive value):
- Elite Defense at Position (DVP rank 1-5): +30 weight
- Severe Cold Streak (L5 < 80% of season): +22 weight
- First Game Back from Injury: +20 weight

SECONDARY FACTORS:
- Good Defense (DVP rank 6-10): +15 weight
- Mild Cold Streak (L5 < 90% of season): +12 weight
- Second Game of Back-to-Back: +10 weight
- High Variance Player: +8 weight

TERTIARY FACTORS:
- Third Game in Four Nights: +6 weight
- Historical Struggle vs Opponent: +8 weight
- Low Expected Minutes: +8 weight

CONFIDENCE TIERS:
-----------------
- PREMIUM: Score >= 85 (Elite defense + cold streak + other factors)
- HIGH: Score >= 75 (Elite defense OR multiple factors)
- STANDARD: Score >= 65 (Single strong factor)

INTEGRATION WITH GENERAL MODEL:
-------------------------------
When combining models for daily picks:
1. V12_Under picks take priority for UNDER direction
2. V12_General handles OVER picks primarily
3. Deduplication by player/prop ensures no conflicts

USAGE:
------
    from src.nba_props.engine.model_v12_under import (
        get_daily_picks_under,
        run_backtest_under,
        UnderModelConfig,
    )
    
    # Get UNDER picks for today
    picks = get_daily_picks_under("2026-02-03")
    
    # Run backtest
    result = run_backtest_under("2025-12-01", "2026-02-02")

Author: PropAI Development Team
Created: February 2026
Version: 12.0
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev

from .model_v12_shared import (
    ModelV12Config,
    PlayerStatsV12,
    DefenseContextV12,
    LineInfo,
    PropPickV12,
    DailyPicksV12,
    BacktestResultV12,
    normalize_name_for_matching,
    get_injured_players,
    get_line_info,
    get_defense_context,
    load_player_stats,
    calculate_edge,
    apply_defense_adjustment,
    determine_confidence_tier,
    grade_pick,
)


# ============================================================================
# Constants - Factor Weights
# ============================================================================

# Factor weights based on historical analysis
FACTOR_WEIGHTS = {
    # PRIMARY FACTORS (Highest predictive value)
    "elite_defense": 30,        # DVP rank 1-5
    "cold_streak_severe": 22,   # L5 < 80% of season
    "injury_first_back": 20,    # First game back from injury
    
    # SECONDARY FACTORS
    "good_defense": 15,         # DVP rank 6-10
    "cold_streak_mild": 12,     # L5 < 90% of season
    "b2b_second_game": 10,      # Second of back-to-back
    "high_variance": 8,         # Inconsistent performer (CV > 0.35)
    "historical_struggle": 8,   # Poor history vs opponent
    "low_minutes_expected": 8,  # Projected lower minutes
    
    # TERTIARY FACTORS
    "third_in_four": 6,         # Third game in four nights
    "injury_second_back": 8,    # Second game back
    "average_defense": 5,       # DVP rank 11-15
    "away_vs_elite": 4,         # Away player vs top defense
}

# Factor adjustments (projection multipliers)
FACTOR_ADJUSTMENTS = {
    "elite_defense": 0.88,      # -12%
    "good_defense": 0.94,       # -6%
    "average_defense": 0.97,    # -3%
    "cold_streak_severe": 0.85, # -15%
    "cold_streak_mild": 0.92,   # -8%
    "b2b_second_game": 0.95,    # -5%
    "third_in_four": 0.97,      # -3%
    "injury_first_back": 0.80,  # -20%
    "injury_second_back": 0.90, # -10%
    "high_variance": 0.96,      # -4%
    "historical_struggle": 0.93, # -7%
    "low_minutes_expected": 0.90, # -10%
    "away_vs_elite": 0.98,      # -2%
}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UnderModelConfig(ModelV12Config):
    """
    Configuration specific to the Under Model (V12 Under).
    
    This model ONLY generates UNDER picks and uses a factor-based
    scoring system instead of pattern detection.
    """
    model_name: str = "V12_UNDER"
    
    # === DIRECTION ===
    # This model only generates UNDER picks
    direction: str = "UNDER"
    
    # === DEFENSE THRESHOLDS ===
    elite_defense_rank: int = 5
    good_defense_rank: int = 10
    average_defense_rank: int = 15
    
    # === COLD STREAK THRESHOLDS ===
    severe_cold_threshold: float = -20.0   # L5 is 20%+ below season
    mild_cold_threshold: float = -10.0     # L5 is 10%+ below season
    
    # === VARIANCE THRESHOLD ===
    high_variance_cv: float = 0.35         # Coefficient of variation > 35%
    
    # === BACK-TO-BACK DETECTION ===
    check_b2b: bool = True
    
    # === INJURY RUST ===
    check_injury_rust: bool = True
    games_back_threshold: int = 3          # First 3 games back from injury
    
    # === HISTORICAL MATCHUPS ===
    check_historical: bool = True
    min_historical_games: int = 2          # Need 2+ games vs opponent
    historical_poor_threshold: float = -15.0  # 15%+ worse than average
    
    # === CONFIDENCE THRESHOLDS ===
    premium_confidence: float = 85.0
    high_confidence: float = 75.0
    min_confidence: float = 65.0           # Minimum for any pick
    
    # === MINIMUM EDGE ===
    min_edge: float = 4.0                  # Lower than general (UNDER is better)
    
    # === FACTOR WEIGHTS (can be customized) ===
    factor_weights: Dict[str, int] = field(default_factory=lambda: FACTOR_WEIGHTS.copy())
    factor_adjustments: Dict[str, float] = field(default_factory=lambda: FACTOR_ADJUSTMENTS.copy())


# ============================================================================
# Factor Analysis
# ============================================================================

@dataclass
class UnderFactorAnalysis:
    """Detailed analysis of UNDER factors for a player/prop."""
    player_id: int
    player_name: str
    team_abbrev: str
    opponent_abbrev: str
    position: str
    prop_type: str
    
    # Line info
    line: float
    line_source: str
    
    # Base stats
    season_avg: float
    l5_avg: float
    l10_avg: float
    
    # Projection
    base_projection: float
    adjusted_projection: float
    total_adjustment: float
    
    # Factors detected
    active_factors: Dict[str, float] = field(default_factory=dict)  # factor -> weight
    factor_adjustments: Dict[str, float] = field(default_factory=dict)  # factor -> multiplier
    
    # Scoring
    raw_score: float = 0.0
    confidence_score: float = 0.0
    confidence_tier: str = "LOW"
    
    # Edge
    edge: float = 0.0
    
    # Reasons
    reasons: List[str] = field(default_factory=list)
    
    @property
    def factor_count(self) -> int:
        return len(self.active_factors)
    
    @property
    def total_weight(self) -> float:
        return sum(self.active_factors.values())


def check_back_to_back(
    conn: sqlite3.Connection,
    team_abbrev: str,
    game_date: str,
) -> Tuple[bool, bool]:
    """
    Check if team is on second game of back-to-back or third in four nights.
    
    Returns: (is_b2b, is_third_in_four)
    """
    from ..standings import _team_ids_by_abbrev
    
    team_ids = _team_ids_by_abbrev(conn).get(team_abbrev, [])
    if not team_ids:
        return False, False
    
    placeholders = ",".join(["?"] * len(team_ids))
    
    # Get recent games
    rows = conn.execute(
        f"""
        SELECT DISTINCT g.game_date
        FROM games g
        WHERE (g.team1_id IN ({placeholders}) OR g.team2_id IN ({placeholders}))
          AND g.game_date < ?
        ORDER BY g.game_date DESC
        LIMIT 3
        """,
        (*team_ids, *team_ids, game_date),
    ).fetchall()
    
    if not rows:
        return False, False
    
    # Parse dates
    target_date = datetime.strptime(game_date, "%Y-%m-%d")
    recent_dates = [datetime.strptime(r["game_date"], "%Y-%m-%d") for r in rows]
    
    # Check back-to-back (game yesterday)
    is_b2b = False
    if recent_dates:
        days_since_last = (target_date - recent_dates[0]).days
        is_b2b = days_since_last == 1
    
    # Check third in four nights
    is_third_in_four = False
    if len(recent_dates) >= 2:
        four_days_ago = target_date - timedelta(days=4)
        games_in_window = sum(1 for d in recent_dates if d >= four_days_ago)
        is_third_in_four = games_in_window >= 2
    
    return is_b2b, is_third_in_four


def check_historical_vs_opponent(
    conn: sqlite3.Connection,
    player_id: int,
    opponent_abbrev: str,
    prop_type: str,
    before_date: str,
    config: UnderModelConfig,
) -> Tuple[bool, float, int]:
    """
    Check player's historical performance vs this opponent.
    
    Returns: (struggles_vs_opponent, deviation_pct, games_count)
    """
    from ..standings import _team_ids_by_abbrev
    
    opp_ids = _team_ids_by_abbrev(conn).get(opponent_abbrev, [])
    if not opp_ids:
        return False, 0.0, 0
    
    placeholders = ",".join(["?"] * len(opp_ids))
    
    # Get historical games vs opponent
    rows = conn.execute(
        f"""
        SELECT b.{prop_type.lower()}, b.minutes
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND g.game_date < ?
          AND (g.team1_id IN ({placeholders}) OR g.team2_id IN ({placeholders}))
          AND b.minutes > 10
        ORDER BY g.game_date DESC
        LIMIT 10
        """,
        (player_id, before_date, *opp_ids, *opp_ids),
    ).fetchall()
    
    if len(rows) < config.min_historical_games:
        return False, 0.0, 0
    
    # Calculate average vs opponent
    values = [r[0] or 0 for r in rows]
    avg_vs_opp = sum(values) / len(values)
    
    # Get season average for comparison
    season_row = conn.execute(
        f"""
        SELECT AVG(b.{prop_type.lower()}) as avg_val
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND g.game_date < ?
          AND b.minutes > 10
        """,
        (player_id, before_date),
    ).fetchone()
    
    if not season_row or not season_row["avg_val"]:
        return False, 0.0, len(rows)
    
    season_avg = season_row["avg_val"]
    if season_avg <= 0:
        return False, 0.0, len(rows)
    
    deviation = (avg_vs_opp - season_avg) / season_avg * 100
    struggles = deviation <= config.historical_poor_threshold
    
    return struggles, deviation, len(rows)


def analyze_under_factors(
    conn: sqlite3.Connection,
    stats: PlayerStatsV12,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    line_info: LineInfo,
    config: UnderModelConfig,
) -> UnderFactorAnalysis:
    """
    Analyze all UNDER factors for a player/prop.
    
    This is the core analysis function that:
    1. Checks all factors
    2. Calculates weights and adjustments
    3. Builds comprehensive factor profile
    """
    pt = prop_type.lower()
    
    analysis = UnderFactorAnalysis(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        position=stats.position,
        prop_type=prop_type.upper(),
        line=line_info.line,
        line_source=line_info.source,
        season_avg=stats.season.get(pt, 0),
        l5_avg=stats.l5.get(pt, 0),
        l10_avg=stats.l10.get(pt, 0),
        base_projection=stats.get_projection(pt, config),
        adjusted_projection=stats.get_projection(pt, config),
        total_adjustment=1.0,
    )
    
    # Get defense context
    defense_context = get_defense_context(conn, opponent_abbrev, stats.position, config)
    defense_rank = defense_context.get_rank(pt)
    
    # FACTOR 1: Defense vs Position
    if defense_rank <= config.elite_defense_rank:
        analysis.active_factors["elite_defense"] = config.factor_weights["elite_defense"]
        analysis.factor_adjustments["elite_defense"] = config.factor_adjustments["elite_defense"]
        analysis.reasons.append(
            f"Elite defense: {opponent_abbrev} ranks #{defense_rank} vs {stats.position} for {pt.upper()}"
        )
    elif defense_rank <= config.good_defense_rank:
        analysis.active_factors["good_defense"] = config.factor_weights["good_defense"]
        analysis.factor_adjustments["good_defense"] = config.factor_adjustments["good_defense"]
        analysis.reasons.append(
            f"Good defense: {opponent_abbrev} ranks #{defense_rank} vs {stats.position} for {pt.upper()}"
        )
    elif defense_rank <= config.average_defense_rank:
        analysis.active_factors["average_defense"] = config.factor_weights["average_defense"]
        analysis.factor_adjustments["average_defense"] = config.factor_adjustments["average_defense"]
        analysis.reasons.append(
            f"Solid defense: {opponent_abbrev} ranks #{defense_rank} vs {stats.position}"
        )
    
    # FACTOR 2: Cold Streak
    deviation_season = stats.deviations_season.get(pt, 0)
    if deviation_season <= config.severe_cold_threshold:
        analysis.active_factors["cold_streak_severe"] = config.factor_weights["cold_streak_severe"]
        analysis.factor_adjustments["cold_streak_severe"] = config.factor_adjustments["cold_streak_severe"]
        analysis.reasons.append(
            f"Severe cold streak: L5 ({analysis.l5_avg:.1f}) is {deviation_season:.0f}% below season ({analysis.season_avg:.1f})"
        )
    elif deviation_season <= config.mild_cold_threshold:
        analysis.active_factors["cold_streak_mild"] = config.factor_weights["cold_streak_mild"]
        analysis.factor_adjustments["cold_streak_mild"] = config.factor_adjustments["cold_streak_mild"]
        analysis.reasons.append(
            f"Cold streak: L5 ({analysis.l5_avg:.1f}) is {deviation_season:.0f}% below season ({analysis.season_avg:.1f})"
        )
    
    # FACTOR 3: Back-to-Back / Third in Four
    if config.check_b2b:
        is_b2b, is_third_in_four = check_back_to_back(conn, stats.team_abbrev, game_date)
        if is_b2b:
            analysis.active_factors["b2b_second_game"] = config.factor_weights["b2b_second_game"]
            analysis.factor_adjustments["b2b_second_game"] = config.factor_adjustments["b2b_second_game"]
            analysis.reasons.append("Back-to-back: Second game in two nights")
        if is_third_in_four:
            analysis.active_factors["third_in_four"] = config.factor_weights["third_in_four"]
            analysis.factor_adjustments["third_in_four"] = config.factor_adjustments["third_in_four"]
            analysis.reasons.append("Fatigue: Third game in four nights")
    
    # FACTOR 4: High Variance
    cv = stats.get_cv(pt)
    if cv >= config.high_variance_cv:
        analysis.active_factors["high_variance"] = config.factor_weights["high_variance"]
        analysis.factor_adjustments["high_variance"] = config.factor_adjustments["high_variance"]
        analysis.reasons.append(f"High variance: CV={cv:.2f} (inconsistent performer)")
    
    # FACTOR 5: Historical vs Opponent
    if config.check_historical:
        struggles, deviation, games = check_historical_vs_opponent(
            conn, stats.player_id, opponent_abbrev, pt, game_date, config
        )
        if struggles:
            analysis.active_factors["historical_struggle"] = config.factor_weights["historical_struggle"]
            analysis.factor_adjustments["historical_struggle"] = config.factor_adjustments["historical_struggle"]
            analysis.reasons.append(
                f"Historical struggle: {deviation:.0f}% below avg vs {opponent_abbrev} ({games} games)"
            )
    
    # Calculate adjusted projection
    total_adjustment = 1.0
    for factor, adj in analysis.factor_adjustments.items():
        total_adjustment *= adj
    
    analysis.adjusted_projection = analysis.base_projection * total_adjustment
    analysis.total_adjustment = total_adjustment
    
    # Calculate raw score (sum of weights)
    analysis.raw_score = sum(analysis.active_factors.values())
    
    # Calculate confidence score
    # Mapping from raw score to confidence (calibrated)
    if analysis.raw_score >= 50:
        analysis.confidence_score = 85 + min(15, (analysis.raw_score - 50) * 0.3)
    elif analysis.raw_score >= 35:
        analysis.confidence_score = 75 + (analysis.raw_score - 35) * 0.67
    elif analysis.raw_score >= 25:
        analysis.confidence_score = 65 + (analysis.raw_score - 25) * 1.0
    else:
        analysis.confidence_score = 50 + analysis.raw_score * 0.6
    
    analysis.confidence_score = min(100, analysis.confidence_score)
    
    # Determine tier
    if analysis.confidence_score >= config.premium_confidence:
        analysis.confidence_tier = "PREMIUM"
    elif analysis.confidence_score >= config.high_confidence:
        analysis.confidence_tier = "HIGH"
    elif analysis.confidence_score >= config.min_confidence:
        analysis.confidence_tier = "STANDARD"
    else:
        analysis.confidence_tier = "LOW"
    
    # Calculate edge
    analysis.edge = calculate_edge(analysis.adjusted_projection, line_info.line, "UNDER")
    
    return analysis


# ============================================================================
# Pick Generation
# ============================================================================

def generate_under_pick(
    conn: sqlite3.Connection,
    stats: PlayerStatsV12,
    prop_type: str,
    opponent_abbrev: str,
    game_date: str,
    config: UnderModelConfig,
) -> Optional[PropPickV12]:
    """
    Generate an UNDER pick for a player/prop combination.
    """
    pt = prop_type.lower()
    
    # Get line info
    line_info = get_line_info(
        conn,
        stats.player_id,
        stats.player_name,
        pt,
        game_date,
        stats.l10.get(pt, 0),
        config,
    )
    
    if line_info.line <= 0:
        return None
    
    # Analyze UNDER factors
    analysis = analyze_under_factors(
        conn, stats, prop_type, opponent_abbrev, game_date, line_info, config
    )
    
    # Filter by minimum confidence
    if analysis.confidence_score < config.min_confidence:
        return None
    
    # Filter by minimum edge
    if analysis.edge < config.min_edge:
        return None
    
    # Require at least one factor
    if analysis.factor_count == 0:
        return None
    
    # Get defense context for pick
    defense_context = get_defense_context(conn, opponent_abbrev, stats.position, config)
    
    return PropPickV12(
        player_id=stats.player_id,
        player_name=stats.player_name,
        team_abbrev=stats.team_abbrev,
        opponent_abbrev=opponent_abbrev,
        game_date=game_date,
        prop_type=prop_type.upper(),
        direction="UNDER",
        line=round(line_info.line, 1),
        line_source=line_info.source,
        book=line_info.book,
        projection=round(analysis.adjusted_projection, 1),
        projection_std=stats.stds.get(pt, 0),
        edge=round(analysis.edge, 1),
        pattern=f"under_{analysis.factor_count}factors",
        confidence_tier=analysis.confidence_tier,
        confidence_score=round(analysis.confidence_score, 1),
        defense_rank=defense_context.get_rank(pt),
        defense_rating=defense_context.get_rating(pt),
        l3_avg=round(stats.l3.get(pt, 0), 1),
        l5_avg=round(stats.l5.get(pt, 0), 1),
        l10_avg=round(stats.l10.get(pt, 0), 1),
        l15_avg=round(stats.l15.get(pt, 0), 1),
        season_avg=round(stats.season.get(pt, 0), 1),
        reasons=analysis.reasons,
        model="V12_UNDER",
    )


def generate_game_under_picks(
    conn: sqlite3.Connection,
    game_date: str,
    team1_name: str,
    team2_name: str,
    config: UnderModelConfig,
    line_stats: Dict[str, int],
) -> List[PropPickV12]:
    """Generate UNDER picks for a single game."""
    t1_abbrev = abbrev_from_team_name(team1_name) or ""
    t2_abbrev = abbrev_from_team_name(team2_name) or ""
    
    injured = get_injured_players(conn, game_date)
    
    all_picks = []
    player_picks = {}
    
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
            
            if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                continue
            
            stats = load_player_stats(conn, player_id, game_date, config)
            if not stats:
                continue
            
            # Determine props to analyze
            props_to_check = []
            if config.include_pts:
                props_to_check.append('pts')
            if config.include_reb:
                props_to_check.append('reb')
            if config.include_ast and stats.season.get('ast', 0) >= config.ast_min_avg:
                props_to_check.append('ast')
            
            for pt in props_to_check:
                if player_picks.get(player_id, 0) >= config.max_picks_per_player:
                    break
                
                pick = generate_under_pick(conn, stats, pt, opp_abbrev, game_date, config)
                
                if pick:
                    all_picks.append(pick)
                    player_picks[player_id] = player_picks.get(player_id, 0) + 1
                    
                    if pick.line_source == "sportsbook":
                        line_stats["sportsbook"] = line_stats.get("sportsbook", 0) + 1
                    else:
                        line_stats["derived"] = line_stats.get("derived", 0) + 1
    
    return all_picks


# ============================================================================
# Public API
# ============================================================================

def get_daily_picks_under(
    game_date: str,
    config: Optional[UnderModelConfig] = None,
    db_path: str = "data/db/nba_props.sqlite3",
) -> DailyPicksV12:
    """
    Generate UNDER picks for all games on a date.
    
    Args:
        game_date: Date in YYYY-MM-DD format
        config: Model configuration (uses defaults if None)
        db_path: Path to database
        
    Returns:
        DailyPicksV12 with UNDER picks for the day
    """
    if config is None:
        config = UnderModelConfig()
    
    db = Db(db_path)
    daily = DailyPicksV12(date=game_date, games=0)
    
    all_picks = []
    line_stats = {"sportsbook": 0, "derived": 0}
    
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
                picks = generate_game_under_picks(
                    conn, game_date, game["team1"], game["team2"], config, line_stats
                )
                all_picks.extend(picks)
    
    # Sort by confidence
    all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
    
    # Limit picks
    daily.picks = all_picks[:config.max_picks_per_day]
    daily.players_with_sportsbook_lines = line_stats.get("sportsbook", 0)
    daily.players_with_derived_lines = line_stats.get("derived", 0)
    
    return daily


def run_backtest_under(
    start_date: str,
    end_date: str,
    config: Optional[UnderModelConfig] = None,
    db_path: str = "data/db/nba_props.sqlite3",
    verbose: bool = True,
) -> BacktestResultV12:
    """
    Run comprehensive backtest for the Under model.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        config: Model configuration
        db_path: Path to database
        verbose: Whether to print progress
        
    Returns:
        BacktestResultV12 with comprehensive results
    """
    if config is None:
        config = UnderModelConfig()
    
    db = Db(db_path)
    result = BacktestResultV12(
        start_date=start_date,
        end_date=end_date,
        model_name=config.model_name,
    )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"MODEL V12 UNDER BACKTEST: {start_date} to {end_date}")
        print(f"{'='*70}")
        print(f"Props: PTS={config.include_pts}, REB={config.include_reb}, AST={config.include_ast}")
        print(f"Min confidence: {config.min_confidence}")
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
            
            game_count = conn.execute(
                "SELECT COUNT(*) as cnt FROM games WHERE game_date = ?",
                (game_date,),
            ).fetchone()
            result.total_games += game_count["cnt"] if game_count else 0
            
            line_stats = {"sportsbook": 0, "derived": 0}
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
                picks = generate_game_under_picks(
                    conn, game_date, game["team1"], game["team2"], config, line_stats
                )
                all_picks.extend(picks)
            
            all_picks.sort(key=lambda p: p.confidence_score, reverse=True)
            all_picks = all_picks[:config.max_picks_per_day]
            
            daily_hits = 0
            daily_total = 0
            
            for pick in all_picks:
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
                
                pick = grade_pick(pick, actual_value)
                
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
                
                # All UNDER picks
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
                
                # By factor count pattern
                if pick.pattern not in result.pattern_stats:
                    result.pattern_stats[pick.pattern] = {"picks": 0, "hits": 0}
                result.pattern_stats[pick.pattern]["picks"] += 1
                if pick.hit:
                    result.pattern_stats[pick.pattern]["hits"] += 1
                
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
# CLI
# ============================================================================

def main():
    """Command-line interface for Model V12 Under."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Model V12 Under - UNDER Picks")
    parser.add_argument("--date", help="Date for picks (YYYY-MM-DD)")
    parser.add_argument("--backtest-start", help="Backtest start date")
    parser.add_argument("--backtest-end", help="Backtest end date")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.backtest_start and args.backtest_end:
        result = run_backtest_under(
            args.backtest_start,
            args.backtest_end,
            verbose=args.verbose,
        )
        print(result.summary())
    elif args.date:
        picks = get_daily_picks_under(args.date)
        print(picks.summary())
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        picks = get_daily_picks_under(today)
        print(picks.summary())


if __name__ == "__main__":
    main()
