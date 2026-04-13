"""Prop edge calculation and ranking."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Optional

from .projector import PlayerProjection


@dataclass
class PropEdge:
    """Calculated edge for a player prop bet."""
    player_id: int
    player_name: str
    team_abbrev: str
    
    # Prop details
    prop_type: str  # PTS, REB, AST
    line: float
    odds_american: Optional[int]
    book: Optional[str]
    
    # Projection
    projected_value: float
    projected_std: float
    
    # Edge calculations
    over_probability: float  # Probability of going over the line
    under_probability: float
    edge_over: float  # Expected edge if betting over
    edge_under: float
    
    # Recommendation
    recommendation: str  # "OVER", "UNDER", or "PASS"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    edge_pct: float  # Best edge as percentage
    
    # Context
    games_sample: int
    is_top_7: bool


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """
    Calculate cumulative distribution function for normal distribution.
    P(X <= x) for X ~ N(mean, std^2)
    """
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    
    z = (x - mean) / std
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds >= 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def _american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds >= 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def calculate_prop_edge(
    projection: PlayerProjection,
    prop_type: str,
    line: float,
    odds_american: Optional[int] = None,
    book: Optional[str] = None,
) -> PropEdge:
    """
    Calculate the edge for a player prop bet.
    
    Args:
        projection: Player projection with stats and uncertainty
        prop_type: Type of prop (PTS, REB, AST)
        line: Sportsbook line
        odds_american: American odds (e.g., -110, +100)
        book: Sportsbook name
    
    Returns:
        PropEdge with calculated probabilities and recommendation
    """
    # Get projected value and std based on prop type
    if prop_type == "PTS":
        proj_value = projection.proj_pts
        proj_std = projection.pts_std
    elif prop_type == "REB":
        proj_value = projection.proj_reb
        proj_std = projection.reb_std
    elif prop_type == "AST":
        proj_value = projection.proj_ast
        proj_std = projection.ast_std
    else:
        raise ValueError(f"Unknown prop type: {prop_type}")
    
    # Ensure minimum std (floor at 10% of value or 1.0)
    proj_std = max(proj_std, proj_value * 0.1, 1.0)
    
    # Calculate over/under probabilities using normal distribution
    # P(over) = P(X > line) = 1 - P(X <= line)
    under_prob = _normal_cdf(line, proj_value, proj_std)
    over_prob = 1 - under_prob
    
    # Calculate expected value / edge
    # Default to -110 odds (fair odds for 50/50) if not provided
    implied_prob = _american_to_implied_prob(odds_american if odds_american else -110)
    
    # Edge = Our probability - Implied probability
    edge_over = over_prob - implied_prob
    edge_under = under_prob - implied_prob
    
    # Determine recommendation
    threshold_high = 0.10  # 10% edge
    threshold_medium = 0.05  # 5% edge
    
    if edge_over > edge_under:
        best_edge = edge_over
        recommendation = "OVER" if best_edge > 0.02 else "PASS"
    else:
        best_edge = edge_under
        recommendation = "UNDER" if best_edge > 0.02 else "PASS"
    
    if recommendation != "PASS":
        if best_edge >= threshold_high:
            confidence = "HIGH"
        elif best_edge >= threshold_medium:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    else:
        confidence = "LOW"
    
    return PropEdge(
        player_id=projection.player_id,
        player_name=projection.player_name,
        team_abbrev=projection.team_abbrev,
        prop_type=prop_type,
        line=line,
        odds_american=odds_american,
        book=book,
        projected_value=proj_value,
        projected_std=proj_std,
        over_probability=round(over_prob, 3),
        under_probability=round(under_prob, 3),
        edge_over=round(edge_over, 3),
        edge_under=round(edge_under, 3),
        recommendation=recommendation,
        confidence=confidence,
        edge_pct=round(best_edge * 100, 1),
        games_sample=projection.games_played,
        is_top_7=projection.is_top_7,
    )


def rank_prop_opportunities(
    conn: sqlite3.Connection,
    projections: list[PlayerProjection],
    as_of_date: str,
    min_edge: float = 0.03,
    top_7_only: bool = False,  # Deprecated, use top_10_only
    top_10_only: bool = True,
) -> list[PropEdge]:
    """
    Rank prop opportunities by edge.
    
    Args:
        conn: Database connection
        projections: List of player projections
        as_of_date: Date to look up lines for
        min_edge: Minimum edge to include (default 3%)
        top_7_only: Only include top 7 players per team
    
    Returns:
        List of PropEdge objects sorted by edge (best first)
    """
    # Get lines for this date
    lines = conn.execute(
        """
        SELECT sl.player_id, p.name as player_name, sl.prop_type, sl.line, sl.odds_american, sl.book
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = ?
        """,
        (as_of_date,),
    ).fetchall()
    
    # Create lookup by player_id and prop_type
    lines_lookup = {}
    for line in lines:
        key = (line["player_id"], line["prop_type"])
        lines_lookup[key] = {
            "line": line["line"],
            "odds_american": line["odds_american"],
            "book": line["book"],
        }
    
    # Calculate edges for all projections with matching lines
    edges = []
    for proj in projections:
        # Check top 10 filter (with legacy top 7 support)
        if top_10_only and not getattr(proj, 'is_top_10', proj.is_top_7):
            continue
        
        for prop_type in ["PTS", "REB", "AST"]:
            key = (proj.player_id, prop_type)
            if key not in lines_lookup:
                continue
            
            line_data = lines_lookup[key]
            
            edge = calculate_prop_edge(
                projection=proj,
                prop_type=prop_type,
                line=line_data["line"],
                odds_american=line_data["odds_american"],
                book=line_data["book"],
            )
            
            # Filter by minimum edge
            if abs(edge.edge_over) >= min_edge or abs(edge.edge_under) >= min_edge:
                edges.append(edge)
    
    # Sort by absolute edge (best opportunities first)
    edges.sort(key=lambda e: -e.edge_pct)
    
    return edges


def _projection_to_dict(p: PlayerProjection) -> dict:
    """Convert a projection to a dict, including archetype info."""
    from .archetypes import get_player_archetype
    
    arch = get_player_archetype(p.player_name)
    archetype_info = None
    if arch:
        archetype_info = {
            "tier": arch.tier,
            "primary": arch.primary_offensive,
            "secondary": arch.secondary_offensive,
            "defensive": arch.defensive_role,
        }
    
    return {
        "player": p.player_name,
        "minutes": p.proj_minutes,
        "pts": p.proj_pts,
        "reb": p.proj_reb,
        "ast": p.proj_ast,
        "games": p.games_played,
        "is_top_7": p.is_top_7,
        "archetype": archetype_info,
    }


def generate_prop_report(
    conn: sqlite3.Connection,
    away_abbrev: str,
    home_abbrev: str,
    game_date: str,
    lines_date: Optional[str] = None,
) -> dict:
    """
    Generate a comprehensive prop report for a matchup.
    
    Args:
        conn: Database connection
        away_abbrev: Away team abbreviation
        home_abbrev: Home team abbreviation
        game_date: Game date
        lines_date: Date for sportsbook lines (defaults to game_date)
    
    Returns:
        Dictionary with projections and recommendations
    """
    from .projector import project_team_players, ProjectionConfig
    from .matchups import get_back_to_back_status, get_team_defense_rating, apply_matchup_adjustments
    from .archetypes import get_player_archetype
    
    config = ProjectionConfig()
    lines_date = lines_date or game_date
    
    # Get back-to-back status
    away_b2b = get_back_to_back_status(conn, away_abbrev, game_date)
    home_b2b = get_back_to_back_status(conn, home_abbrev, game_date)
    
    # Get defense ratings
    away_defense = get_team_defense_rating(conn, away_abbrev)
    home_defense = get_team_defense_rating(conn, home_abbrev)
    
    # Project players (away team plays against home defense)
    away_projections = project_team_players(
        conn=conn,
        team_abbrev=away_abbrev,
        config=config,
        opponent_abbrev=home_abbrev,
        is_back_to_back=away_b2b.is_back_to_back,
        rest_days=away_b2b.rest_days,
    )
    
    # Apply opponent adjustments
    for proj in away_projections:
        adj_pts, adj_reb, adj_ast, adj_info = apply_matchup_adjustments(
            proj.proj_pts, proj.proj_reb, proj.proj_ast, home_defense
        )
        proj.proj_pts = adj_pts
        proj.proj_reb = adj_reb
        proj.proj_ast = adj_ast
        proj.adjustments.update(adj_info)
    
    # Project home team
    home_projections = project_team_players(
        conn=conn,
        team_abbrev=home_abbrev,
        config=config,
        opponent_abbrev=away_abbrev,
        is_back_to_back=home_b2b.is_back_to_back,
        rest_days=home_b2b.rest_days,
    )
    
    # Apply opponent adjustments
    for proj in home_projections:
        adj_pts, adj_reb, adj_ast, adj_info = apply_matchup_adjustments(
            proj.proj_pts, proj.proj_reb, proj.proj_ast, away_defense
        )
        proj.proj_pts = adj_pts
        proj.proj_reb = adj_reb
        proj.proj_ast = adj_ast
        proj.adjustments.update(adj_info)
    
    # Get prop edges
    all_projections = away_projections + home_projections
    edges = rank_prop_opportunities(
        conn=conn,
        projections=all_projections,
        as_of_date=lines_date,
        min_edge=0.02,
        top_10_only=True,
    )
    
    return {
        "matchup": {
            "away": away_abbrev,
            "home": home_abbrev,
            "date": game_date,
        },
        "context": {
            "away_b2b": away_b2b.is_back_to_back,
            "away_rest_days": away_b2b.rest_days,
            "home_b2b": home_b2b.is_back_to_back,
            "home_rest_days": home_b2b.rest_days,
        },
        "defense_ratings": {
            "away": {
                "pts_factor": away_defense.pts_factor if away_defense else 1.0,
                "reb_factor": away_defense.reb_factor if away_defense else 1.0,
                "ast_factor": away_defense.ast_factor if away_defense else 1.0,
            } if away_defense else None,
            "home": {
                "pts_factor": home_defense.pts_factor if home_defense else 1.0,
                "reb_factor": home_defense.reb_factor if home_defense else 1.0,
                "ast_factor": home_defense.ast_factor if home_defense else 1.0,
            } if home_defense else None,
        },
        "away_projections": [
            _projection_to_dict(p)
            for p in away_projections
        ],
        "home_projections": [
            _projection_to_dict(p)
            for p in home_projections
        ],
        "recommendations": [
            {
                "player": e.player_name,
                "team": e.team_abbrev,
                "prop": e.prop_type,
                "line": e.line,
                "projected": e.projected_value,
                "recommendation": e.recommendation,
                "confidence": e.confidence,
                "edge_pct": e.edge_pct,
                "over_prob": e.over_probability,
                "under_prob": e.under_probability,
                "book": e.book,
            }
            for e in edges
        ],
    }


def generate_comprehensive_matchup_report(
    conn: sqlite3.Connection,
    away_abbrev: str,
    home_abbrev: str,
    game_date: str,
    spread: Optional[float] = None,
    over_under: Optional[float] = None,
) -> dict:
    """
    Generate a comprehensive matchup report with advanced analytics.
    
    Includes:
    - Position-based defensive analysis
    - Player trends (hot/cold)
    - Historical player vs team performance
    - Elite defender matchups
    - Comprehensive edge calculations
    """
    from .projector import project_team_players, ProjectionConfig
    from .matchups import get_back_to_back_status, get_team_defense_rating
    from .defense_analysis import (
        get_team_defense_summary,
        get_all_position_defense_profiles,
        get_player_trend,
        get_player_vs_team_profile,
        calculate_matchup_edge,
    )
    from .roster import get_roster_for_team, should_avoid_betting_over, PLAYER_DATABASE
    
    config = ProjectionConfig()
    
    # Get back-to-back status
    away_b2b = get_back_to_back_status(conn, away_abbrev, game_date)
    home_b2b = get_back_to_back_status(conn, home_abbrev, game_date)
    
    # Get defense summaries
    away_defense_summary = get_team_defense_summary(conn, away_abbrev)
    home_defense_summary = get_team_defense_summary(conn, home_abbrev)
    
    # Get position-specific defense profiles
    away_position_defense = get_all_position_defense_profiles(conn, away_abbrev)
    home_position_defense = get_all_position_defense_profiles(conn, home_abbrev)
    
    # Project players
    away_projections = project_team_players(
        conn=conn,
        team_abbrev=away_abbrev,
        config=config,
        opponent_abbrev=home_abbrev,
        is_back_to_back=away_b2b.is_back_to_back,
        rest_days=away_b2b.rest_days,
    )
    
    home_projections = project_team_players(
        conn=conn,
        team_abbrev=home_abbrev,
        config=config,
        opponent_abbrev=away_abbrev,
        is_back_to_back=home_b2b.is_back_to_back,
        rest_days=home_b2b.rest_days,
    )
    
    # Calculate matchup edges for each player
    is_close_game = spread is not None and abs(spread) <= 6
    all_edges = []
    
    def process_player_projections(projections, opponent_abbrev, is_b2b, rest_days):
        enhanced_projections = []
        player_edges = []
        
        for proj in projections:
            if not getattr(proj, 'is_top_10', proj.is_top_7):
                continue
            
            # Get trend
            trend = get_player_trend(conn, proj.player_id)
            
            # Get history vs opponent
            vs_history = get_player_vs_team_profile(conn, proj.player_name, opponent_abbrev)
            
            # Calculate edges for each prop type
            edges = {}
            for prop_type in ["PTS", "REB", "AST"]:
                baseline = getattr(proj, f"proj_{prop_type.lower()}")
                edge = calculate_matchup_edge(
                    conn=conn,
                    player_id=proj.player_id,
                    player_name=proj.player_name,
                    team_abbrev=proj.team_abbrev,
                    opponent_abbrev=opponent_abbrev,
                    prop_type=prop_type,
                    baseline_value=baseline,
                    is_b2b=is_b2b,
                    rest_days=rest_days,
                    spread=spread,
                    over_under=over_under,
                )
                edges[prop_type] = edge
                
                # Add to all edges if significant
                if abs(edge.adjustment_pct) >= 3 and edge.confidence_score >= 50:
                    player_edges.append(edge)
            
            # Build enhanced projection
            enhanced = {
                "player_id": proj.player_id,
                "player": proj.player_name,
                "team": proj.team_abbrev,
                "position": proj.position,
                "minutes": proj.proj_minutes,
                "pts": edges["PTS"].adjusted_projection,
                "reb": edges["REB"].adjusted_projection,
                "ast": edges["AST"].adjusted_projection,
                "pts_baseline": proj.proj_pts,
                "reb_baseline": proj.proj_reb,
                "ast_baseline": proj.proj_ast,
                "games": proj.games_played,
                "is_top_7": proj.is_top_7,
                "is_top_10": getattr(proj, 'is_top_10', proj.is_top_7),
                
                # Trend info
                "trend": {
                    "pts": trend.pts_trend if trend else "stable",
                    "reb": trend.reb_trend if trend else "stable",
                    "ast": trend.ast_trend if trend else "stable",
                    "pts_change": trend.pts_change_pct if trend else 0,
                    "reb_change": trend.reb_change_pct if trend else 0,
                    "ast_change": trend.ast_change_pct if trend else 0,
                } if trend else None,
                
                # Game log
                "game_log": trend.game_log if trend else [],
                
                # History vs opponent
                "vs_opponent": {
                    "games": vs_history.games_played,
                    "pts_avg": vs_history.pts_avg,
                    "reb_avg": vs_history.reb_avg,
                    "ast_avg": vs_history.ast_avg,
                    "pts_diff": vs_history.pts_diff,
                    "reb_diff": vs_history.reb_diff,
                    "ast_diff": vs_history.ast_diff,
                } if vs_history and vs_history.has_history else None,
                
                # Edges
                "edges": {
                    "PTS": {
                        "direction": edges["PTS"].direction,
                        "adjusted": edges["PTS"].adjusted_projection,
                        "adjustment_pct": edges["PTS"].adjustment_pct,
                        "confidence": edges["PTS"].confidence_tier,
                        "confidence_score": edges["PTS"].confidence_score,
                        "reasons": edges["PTS"].reasons,
                        "warnings": edges["PTS"].warnings,
                    },
                    "REB": {
                        "direction": edges["REB"].direction,
                        "adjusted": edges["REB"].adjusted_projection,
                        "adjustment_pct": edges["REB"].adjustment_pct,
                        "confidence": edges["REB"].confidence_tier,
                        "confidence_score": edges["REB"].confidence_score,
                        "reasons": edges["REB"].reasons,
                        "warnings": edges["REB"].warnings,
                    },
                    "AST": {
                        "direction": edges["AST"].direction,
                        "adjusted": edges["AST"].adjusted_projection,
                        "adjustment_pct": edges["AST"].adjustment_pct,
                        "confidence": edges["AST"].confidence_tier,
                        "confidence_score": edges["AST"].confidence_score,
                        "reasons": edges["AST"].reasons,
                        "warnings": edges["AST"].warnings,
                    },
                },
                
                # Archetype info
                "archetype": proj.adjustments.get("archetype"),
            }
            
            enhanced_projections.append(enhanced)
        
        return enhanced_projections, player_edges
    
    # Process both teams
    away_enhanced, away_edges = process_player_projections(
        away_projections, home_abbrev, away_b2b.is_back_to_back, away_b2b.rest_days
    )
    home_enhanced, home_edges = process_player_projections(
        home_projections, away_abbrev, home_b2b.is_back_to_back, home_b2b.rest_days
    )
    
    all_edges = away_edges + home_edges
    
    # Sort edges by confidence score
    all_edges.sort(key=lambda e: -e.confidence_score)
    
    # Separate over and under plays
    over_plays = [e for e in all_edges if e.direction == "OVER" and e.confidence_tier in ("HIGH", "MEDIUM")]
    under_plays = [e for e in all_edges if e.direction == "UNDER" and e.confidence_tier in ("HIGH", "MEDIUM")]
    
    # Find players to avoid
    avoid_players = []
    for proj in away_enhanced + home_enhanced:
        warnings = []
        for prop_type in ["PTS", "REB", "AST"]:
            warnings.extend(proj["edges"][prop_type].get("warnings", []))
        
        if len(warnings) >= 2:
            avoid_players.append({
                "player": proj["player"],
                "team": proj["team"],
                "warnings": list(set(warnings))[:3],
            })
    
    # Build key storylines
    key_matchups = []
    
    if away_b2b.is_back_to_back:
        key_matchups.append(f"⚠️ {away_abbrev} playing on back-to-back")
    if home_b2b.is_back_to_back:
        key_matchups.append(f"⚠️ {home_abbrev} playing on back-to-back")
    
    if away_b2b.rest_days >= 3:
        key_matchups.append(f"✨ {away_abbrev} well-rested ({away_b2b.rest_days} days)")
    if home_b2b.rest_days >= 3:
        key_matchups.append(f"✨ {home_abbrev} well-rested ({home_b2b.rest_days} days)")
    
    if is_close_game:
        key_matchups.append(f"🎯 Expected close game (spread: {spread:+.1f})")
    
    # Defense matchup notes
    for name, defense, opponent in [
        (away_abbrev, home_defense_summary, away_enhanced),
        (home_abbrev, away_defense_summary, home_enhanced),
    ]:
        if defense and defense.get("weaknesses"):
            for weakness in defense["weaknesses"][:2]:
                key_matchups.append(f"📈 {name} advantage: {defense.get('team_abbrev')} {weakness}")
    
    return {
        "matchup": {
            "away": away_abbrev,
            "home": home_abbrev,
            "date": game_date,
            "spread": spread,
            "over_under": over_under,
            "is_close_game": is_close_game,
        },
        "context": {
            "away_b2b": away_b2b.is_back_to_back,
            "away_rest_days": away_b2b.rest_days,
            "home_b2b": home_b2b.is_back_to_back,
            "home_rest_days": home_b2b.rest_days,
        },
        "defense": {
            "away": away_defense_summary,
            "home": home_defense_summary,
        },
        "position_defense": {
            "away": {
                pos: {
                    "pts_factor": p.pts_factor,
                    "reb_factor": p.reb_factor,
                    "ast_factor": p.ast_factor,
                    "pts_rating": p.pts_rating,
                    "reb_rating": p.reb_rating,
                    "ast_rating": p.ast_rating,
                }
                for pos, p in away_position_defense.items()
            } if away_position_defense else {},
            "home": {
                pos: {
                    "pts_factor": p.pts_factor,
                    "reb_factor": p.reb_factor,
                    "ast_factor": p.ast_factor,
                    "pts_rating": p.pts_rating,
                    "reb_rating": p.reb_rating,
                    "ast_rating": p.ast_rating,
                }
                for pos, p in home_position_defense.items()
            } if home_position_defense else {},
        },
        "away_projections": away_enhanced,
        "home_projections": home_enhanced,
        "best_over_plays": [
            {
                "player": e.player_name,
                "team": e.team_abbrev,
                "opponent": e.opponent_abbrev,
                "prop": e.prop_type,
                "baseline": e.baseline_projection,
                "adjusted": e.adjusted_projection,
                "adjustment_pct": e.adjustment_pct,
                "confidence": e.confidence_tier,
                "confidence_score": e.confidence_score,
                "reasons": e.reasons,
                "warnings": e.warnings,
            }
            for e in over_plays[:8]
        ],
        "best_under_plays": [
            {
                "player": e.player_name,
                "team": e.team_abbrev,
                "opponent": e.opponent_abbrev,
                "prop": e.prop_type,
                "baseline": e.baseline_projection,
                "adjusted": e.adjusted_projection,
                "adjustment_pct": e.adjustment_pct,
                "confidence": e.confidence_tier,
                "confidence_score": e.confidence_score,
                "reasons": e.reasons,
                "warnings": e.warnings,
            }
            for e in under_plays[:8]
        ],
        "avoid_players": avoid_players[:5],
        "key_matchups": key_matchups,
    }

