"""
NBA Props Projection Engine
===========================

This module contains the core prediction and analysis logic for NBA player props.

Submodules:
-----------
- **projector**: Core player projection calculations (weighted averages, per-minute rates)
- **game_context**: Game context data (B2B status, team defense ratings)
- **edge_calculator**: Prop bet edge and probability calculations
- **matchup_advisor**: Advanced defense metrics and ADVISOR reports (MAIN OUTPUT)
- **under_picks_analyzer**: Separate UNDER picks model with defense factor integration
- **archetypes**: Player archetype definitions
- **archetype_db**: Database-backed archetype storage
- **roster**: Player roster and profiles
- **usage_redistribution**: Usage rate calculations when stars are out
- **alerts**: Edge alert scanning system
- **backtesting**: Historical accuracy testing

Main Entry Points:
-----------------
For basic projections:
    >>> from nba_props.engine import project_team_players, ProjectionConfig
    >>> projections = project_team_players(conn, "LAL", config)

For comprehensive matchup analysis (RECOMMENDED):
    >>> from nba_props.engine import generate_comprehensive_matchup_report
    >>> report = generate_comprehensive_matchup_report(conn, "LAL", "BOS", "2026-01-03")
    >>> print(report.best_over_plays)  # Get recommended OVER bets

NOTE: UNDER picks use a separate model (under_picks_analyzer.py).
The main model focuses on OVER picks for optimal accuracy.

For prop edge calculations:
    >>> from nba_props.engine import calculate_prop_edge
    >>> edge = calculate_prop_edge(projection, "PTS", line=24.5)
"""
from .projector import (
    PlayerProjection,
    ProjectionConfig,
    project_player_stats,
    project_team_players,
)
from .game_context import (
    BackToBackStatus,
    get_back_to_back_status,
    get_team_defense_rating,
    get_all_team_defense_ratings,
    apply_matchup_adjustments,
    MatchupRecommendation,
    get_position_defense_rating,
    get_player_vs_team_history,
    generate_matchup_recommendations,
)
from .edge_calculator import (
    PropEdge,
    calculate_prop_edge,
    rank_prop_opportunities,
    generate_prop_report,
)
from .matchup_advisor import (
    # Data classes for defense analysis
    PositionDefenseProfile,
    ArchetypeDefenseProfile,
    PlayerVsTeamProfile,
    PlayerTrend,
    MatchupEdge,
    ComprehensiveMatchupReport,
    # Position-based defense functions
    get_position_defense_profile,
    get_all_position_defense_profiles,
    rank_position_defense_profiles,
    # Player analysis functions
    get_player_vs_team_profile,
    get_player_trend,
    # Edge calculation
    calculate_matchup_edge,
    # Team defense summary
    get_team_defense_summary,
    # MAIN ADVISOR FUNCTION
    generate_comprehensive_matchup_report,
)
from .usage_redistribution import (
    PlayerUsageProfile,
    UsageRedistributionResult,
    get_team_usage_profiles,
    calculate_usage_redistribution,
    get_historical_impact,
)

__all__ = [
    # Projector
    "PlayerProjection",
    "ProjectionConfig",
    "project_player_stats",
    "project_team_players",
    # Game Context (B2B, defense ratings)
    "BackToBackStatus",
    "get_back_to_back_status",
    "get_team_defense_rating",
    "get_all_team_defense_ratings",
    "apply_matchup_adjustments",
    "MatchupRecommendation",
    "get_position_defense_rating",
    "get_player_vs_team_history",
    "generate_matchup_recommendations",
    # Edge Calculator
    "PropEdge",
    "calculate_prop_edge",
    "rank_prop_opportunities",
    "generate_prop_report",
    # Matchup Advisor (MAIN OUTPUT)
    "PositionDefenseProfile",
    "ArchetypeDefenseProfile",
    "PlayerVsTeamProfile",
    "PlayerTrend",
    "MatchupEdge",
    "ComprehensiveMatchupReport",
    "get_position_defense_profile",
    "get_all_position_defense_profiles",
    "rank_position_defense_profiles",
    "get_player_vs_team_profile",
    "get_player_trend",
    "calculate_matchup_edge",
    "get_team_defense_summary",
    "generate_comprehensive_matchup_report",
    # Usage Redistribution
    "PlayerUsageProfile",
    "UsageRedistributionResult",
    "get_team_usage_profiles",
    "calculate_usage_redistribution",
    "get_historical_impact",
]
