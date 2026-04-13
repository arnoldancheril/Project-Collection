"""Projection engine module for NBA Props."""
from .projector import (
    PlayerProjection,
    ProjectionConfig,
    project_player_stats,
    project_team_players,
)
from .matchups import (
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
from .props import (
    PropEdge,
    calculate_prop_edge,
    rank_prop_opportunities,
)
from .usage_redistribution import (
    PlayerUsageProfile,
    UsageRedistributionResult,
    get_team_usage_profiles,
    calculate_usage_redistribution,
    get_historical_impact,
)

__all__ = [
    "PlayerProjection",
    "ProjectionConfig",
    "project_player_stats",
    "project_team_players",
    "BackToBackStatus",
    "get_back_to_back_status",
    "get_team_defense_rating",
    "get_all_team_defense_ratings",
    "apply_matchup_adjustments",
    "MatchupRecommendation",
    "get_position_defense_rating",
    "get_player_vs_team_history",
    "generate_matchup_recommendations",
    "PropEdge",
    "calculate_prop_edge",
    "rank_prop_opportunities",
    "PlayerUsageProfile",
    "UsageRedistributionResult",
    "get_team_usage_profiles",
    "calculate_usage_redistribution",
    "get_historical_impact",
]

