"""
Web scraper for NBA player prop lines from free sources.

This module provides functions to scrape player prop betting lines from publicly
accessible websites without requiring login.

Note: Many betting/DFS sites (PrizePicks, Underdog, etc.) now use aggressive
bot protection (PerimeterX, Cloudflare) that blocks automated requests.
For reliable data, use The Odds API client instead (odds_api_client.py).

This module is maintained for:
1. Future implementation when reliable sources become available
2. Manual data entry parsing (copy-paste from websites)
3. Fallback/comparison with API data

Requires: requests, beautifulsoup4
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from collections import defaultdict

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING_DEPS = True
except ImportError:
    HAS_SCRAPING_DEPS = False


# ============================================================================
# Configuration
# ============================================================================

# User agent for requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# PrizePicks API endpoint
PRIZEPICKS_PROJECTIONS_URL = "https://api.prizepicks.com/projections"

# NBA league ID in PrizePicks
PRIZEPICKS_NBA_LEAGUE_ID = "7"

# Stat type mappings
STAT_TYPE_MAP = {
    "Points": "PTS",
    "Rebounds": "REB",
    "Assists": "AST",
    "Pts+Rebs+Asts": "PRA",  # Combined stat - skip for our purposes
    "Pts+Rebs": "PR",  # Combined stat - skip
    "Pts+Asts": "PA",  # Combined stat - skip
    "Rebs+Asts": "RA",  # Combined stat - skip
    "Steals": "STL",
    "Blocks": "BLK",
    "Turnovers": "TOV",
    "3-PT Made": "3PM",
    "Fantasy Score": "FAN",
}

# Stat types we want (only single-stat props)
WANTED_STAT_TYPES = {"Points", "Rebounds", "Assists"}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ScrapedPlayerProp:
    """A player prop line scraped from a website."""
    source: str  # prizepicks, etc.
    player_name: str
    team: str
    prop_type: str  # PTS, REB, AST
    line: float
    opponent: Optional[str] = None
    game_time: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    # PrizePicks-specific fields
    player_id: Optional[str] = None
    projection_id: Optional[str] = None
    is_promo: bool = False


@dataclass
class ScrapeResult:
    """Result of a scraping operation."""
    props: list[ScrapedPlayerProp]
    success: bool
    source: str
    fetched_at: str
    error_message: Optional[str] = None
    player_count: int = 0
    prop_count: int = 0


# ============================================================================
# Utility Functions
# ============================================================================

def _check_dependencies():
    """Check that scraping dependencies are available."""
    if not HAS_SCRAPING_DEPS:
        raise ImportError(
            "requests and beautifulsoup4 are required for web scraping. "
            "Install them with: pip install requests beautifulsoup4"
        )


def _normalize_player_name(name: str) -> str:
    """Normalize player name for consistent matching."""
    if not name:
        return ""
    return " ".join(name.strip().split())


def _get_headers() -> dict:
    """Get standard request headers."""
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/html,application/xhtml+xml,*/*",
        "Accept-Language": "en-US,en;q=0.5",
    }


# ============================================================================
# PrizePicks Scraper
# ============================================================================

def fetch_prizepicks_projections(
    stat_types: list[str] = None,
    verbose: bool = False,
) -> ScrapeResult:
    """
    Fetch NBA player projections from PrizePicks public API.
    
    This is a free, public API that doesn't require authentication.
    It provides player props with lines that can be used as reference
    for sportsbook lines.
    
    Args:
        stat_types: List of stat types to fetch (default: PTS, REB, AST)
        verbose: Print progress information
        
    Returns:
        ScrapeResult with list of ScrapedPlayerProp objects
    """
    _check_dependencies()
    
    if stat_types is None:
        stat_types = ["PTS", "REB", "AST"]
    
    # Convert our stat types to PrizePicks stat names
    wanted_stats = set()
    for st in stat_types:
        for pp_name, our_name in STAT_TYPE_MAP.items():
            if our_name == st:
                wanted_stats.add(pp_name)
    
    # Default to basic stats if none mapped
    if not wanted_stats:
        wanted_stats = WANTED_STAT_TYPES
    
    fetched_at = datetime.utcnow().isoformat() + "Z"
    
    if verbose:
        print(f"Fetching PrizePicks projections...")
        print(f"  Stat types: {wanted_stats}")
    
    try:
        response = requests.get(
            PRIZEPICKS_PROJECTIONS_URL,
            headers=_get_headers(),
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return ScrapeResult(
            props=[],
            success=False,
            source="prizepicks",
            fetched_at=fetched_at,
            error_message=f"Request failed: {str(e)}",
        )
    
    try:
        data = response.json()
    except ValueError as e:
        return ScrapeResult(
            props=[],
            success=False,
            source="prizepicks",
            fetched_at=fetched_at,
            error_message=f"Invalid JSON response: {str(e)}",
        )
    
    # Parse the response
    props = _parse_prizepicks_response(data, wanted_stats, verbose)
    
    # Count unique players
    unique_players = set(p.player_name for p in props)
    
    if verbose:
        print(f"  Found {len(props)} props for {len(unique_players)} players")
    
    return ScrapeResult(
        props=props,
        success=True,
        source="prizepicks",
        fetched_at=fetched_at,
        player_count=len(unique_players),
        prop_count=len(props),
    )


def _parse_prizepicks_response(
    data: dict,
    wanted_stats: set[str],
    verbose: bool = False,
) -> list[ScrapedPlayerProp]:
    """Parse PrizePicks API response into ScrapedPlayerProp objects."""
    props = []
    
    # Extract lookup tables from "included" array
    included = data.get("included", [])
    
    players = {}
    stat_types = {}
    teams = {}
    games = {}
    
    for item in included:
        item_type = item.get("type")
        item_id = item.get("id")
        attrs = item.get("attributes", {})
        
        if item_type == "new_player":
            players[item_id] = {
                "name": attrs.get("name", ""),
                "team": attrs.get("team", ""),
                "team_name": attrs.get("team_name", ""),
                "position": attrs.get("position", ""),
                "league_id": attrs.get("league_id"),
            }
        elif item_type == "stat_type":
            stat_types[item_id] = attrs.get("name", "")
        elif item_type == "team":
            teams[item_id] = attrs.get("name", "")
        elif item_type == "game":
            games[item_id] = {
                "home_team": attrs.get("home_team", ""),
                "away_team": attrs.get("away_team", ""),
                "start_time": attrs.get("start_time", ""),
            }
    
    # Process projections
    projections = data.get("data", [])
    
    # Track which player/stat combos we've seen to get primary line only
    seen = set()
    
    for proj in projections:
        attrs = proj.get("attributes", {})
        rel = proj.get("relationships", {})
        
        # Check if NBA
        league_data = rel.get("league", {}).get("data", {})
        if league_data.get("id") != PRIZEPICKS_NBA_LEAGUE_ID:
            continue
        
        # Get stat type
        stat_type_data = rel.get("stat_type", {}).get("data", {})
        stat_type_id = stat_type_data.get("id")
        stat_name = stat_types.get(stat_type_id, "")
        
        # Filter to wanted stat types
        if stat_name not in wanted_stats:
            continue
        
        # Map to our stat type
        prop_type = STAT_TYPE_MAP.get(stat_name)
        if not prop_type:
            continue
        
        # Get player info
        player_data = rel.get("new_player", {}).get("data", {})
        player_id = player_data.get("id")
        player_info = players.get(player_id, {})
        
        # Only include NBA players
        if player_info.get("league_id") != 7:
            continue
        
        player_name = _normalize_player_name(player_info.get("name", ""))
        if not player_name:
            continue
        
        # Get the line
        line = attrs.get("line_score")
        if line is None:
            continue
        
        # Create unique key for deduplication (take first/primary line only)
        key = (player_name.lower(), prop_type)
        if key in seen:
            continue
        seen.add(key)
        
        # Get game info for opponent
        game_data = rel.get("game", {}).get("data", {})
        game_id = game_data.get("id")
        game_info = games.get(game_id, {})
        
        team = player_info.get("team", "")
        opponent = None
        if game_info:
            home = game_info.get("home_team", "")
            away = game_info.get("away_team", "")
            # Determine opponent based on player's team
            if team and home and away:
                opponent = away if team.upper() in home.upper() else home
        
        props.append(ScrapedPlayerProp(
            source="prizepicks",
            player_name=player_name,
            team=team,
            prop_type=prop_type,
            line=float(line),
            opponent=opponent,
            game_time=attrs.get("start_time"),
            player_id=player_id,
            projection_id=proj.get("id"),
            is_promo=attrs.get("is_promo", False),
        ))
    
    # Sort by player name and prop type
    props.sort(key=lambda p: (p.player_name.lower(), p.prop_type))
    
    return props


# ============================================================================
# Additional Scrapers (Fallbacks)
# ============================================================================

def fetch_covers_odds(verbose: bool = False) -> ScrapeResult:
    """
    Fetch NBA player props from Covers.com.
    
    Note: This is a placeholder for potential future implementation.
    Covers.com requires JavaScript rendering, making it harder to scrape.
    """
    return ScrapeResult(
        props=[],
        success=False,
        source="covers",
        fetched_at=datetime.utcnow().isoformat() + "Z",
        error_message="Covers.com scraping not yet implemented",
    )


# ============================================================================
# Combined Scraper
# ============================================================================

def fetch_all_scraped_lines(
    sources: list[str] = None,
    stat_types: list[str] = None,
    verbose: bool = False,
) -> list[ScrapedPlayerProp]:
    """
    Fetch player props from all available scraped sources.
    
    Args:
        sources: List of sources to use (default: ["prizepicks"])
        stat_types: List of stat types (default: ["PTS", "REB", "AST"])
        verbose: Print progress information
        
    Returns:
        List of ScrapedPlayerProp from all sources
    """
    if sources is None:
        sources = ["prizepicks"]
    
    if stat_types is None:
        stat_types = ["PTS", "REB"]  # Default to PTS and REB per model recommendation
    
    all_props = []
    
    for source in sources:
        if verbose:
            print(f"\n=== Fetching from {source} ===")
        
        if source == "prizepicks":
            result = fetch_prizepicks_projections(stat_types=stat_types, verbose=verbose)
        elif source == "covers":
            result = fetch_covers_odds(verbose=verbose)
        else:
            if verbose:
                print(f"Unknown source: {source}")
            continue
        
        if result.success:
            all_props.extend(result.props)
            if verbose:
                print(f"  ✓ Got {result.prop_count} props from {result.player_count} players")
        else:
            if verbose:
                print(f"  ✗ Failed: {result.error_message}")
    
    return all_props


# ============================================================================
# Database Storage Functions
# ============================================================================

def store_scraped_props(
    conn,
    props: list[ScrapedPlayerProp],
    as_of_date: str,
    verbose: bool = False,
) -> int:
    """
    Store scraped player props in the sportsbook_lines table.
    
    Args:
        conn: SQLite connection
        props: List of props from fetch_all_scraped_lines()
        as_of_date: Date string (YYYY-MM-DD) for the lines
        verbose: Print progress
        
    Returns:
        Number of props stored
    """
    # Import here to avoid circular imports
    from .odds_api_client import find_player_id_by_name
    
    stored_count = 0
    
    for prop in props:
        # Find or create player
        player_id = find_player_id_by_name(conn, prop.player_name)
        
        if not player_id:
            # Create new player entry
            cur = conn.execute(
                "INSERT INTO players(name) VALUES (?)",
                (prop.player_name,)
            )
            player_id = cur.lastrowid
            if verbose:
                print(f"  Created new player: {prop.player_name} (id={player_id})")
        
        # Check if line already exists
        existing = conn.execute(
            """
            SELECT id FROM sportsbook_lines 
            WHERE as_of_date = ? AND player_id = ? AND prop_type = ? AND book = ?
            """,
            (as_of_date, player_id, prop.prop_type, prop.source)
        ).fetchone()
        
        if existing:
            # Update existing line
            conn.execute(
                """
                UPDATE sportsbook_lines 
                SET line = ?
                WHERE id = ?
                """,
                (prop.line, existing["id"])
            )
            if verbose:
                print(f"  Updated {prop.player_name} {prop.prop_type} = {prop.line}")
        else:
            # Insert new line
            conn.execute(
                """
                INSERT INTO sportsbook_lines(as_of_date, game_id, team_id, player_id, prop_type, line, odds_american, book)
                VALUES (?, NULL, NULL, ?, ?, ?, NULL, ?)
                """,
                (as_of_date, player_id, prop.prop_type, prop.line, prop.source)
            )
            if verbose:
                print(f"  Inserted {prop.player_name} {prop.prop_type} = {prop.line}")
        
        stored_count += 1
    
    return stored_count


# ============================================================================
# Comparison Functions
# ============================================================================

def compare_lines_sources(
    scraped_props: list[ScrapedPlayerProp],
    api_props: list,  # OddsAPIPlayerProp from odds_api_client
    verbose: bool = False,
) -> dict:
    """
    Compare player props from scraped sources vs API sources.
    
    This is useful for validating that scraped lines are reasonable
    and for understanding line discrepancies.
    
    Returns:
        Dictionary with comparison statistics
    """
    # Index scraped props by (player_name, prop_type)
    scraped_by_key = {}
    for prop in scraped_props:
        key = (prop.player_name.lower(), prop.prop_type)
        scraped_by_key[key] = prop
    
    # Index API props
    api_by_key = {}
    for prop in api_props:
        key = (prop.player_name.lower(), prop.prop_type)
        api_by_key[key] = prop
    
    # Find overlapping players
    scraped_keys = set(scraped_by_key.keys())
    api_keys = set(api_by_key.keys())
    
    common_keys = scraped_keys & api_keys
    scraped_only = scraped_keys - api_keys
    api_only = api_keys - scraped_keys
    
    # Calculate line differences for common players
    differences = []
    for key in common_keys:
        scraped_line = scraped_by_key[key].line
        api_line = api_by_key[key].line
        diff = scraped_line - api_line
        pct_diff = (diff / api_line * 100) if api_line else 0
        differences.append({
            "player": scraped_by_key[key].player_name,
            "prop_type": key[1],
            "scraped_line": scraped_line,
            "api_line": api_line,
            "diff": diff,
            "pct_diff": pct_diff,
        })
    
    # Sort by absolute difference
    differences.sort(key=lambda x: abs(x["diff"]), reverse=True)
    
    # Calculate statistics
    if differences:
        avg_diff = sum(d["diff"] for d in differences) / len(differences)
        avg_abs_diff = sum(abs(d["diff"]) for d in differences) / len(differences)
        max_diff = max(abs(d["diff"]) for d in differences)
    else:
        avg_diff = avg_abs_diff = max_diff = 0
    
    result = {
        "common_count": len(common_keys),
        "scraped_only_count": len(scraped_only),
        "api_only_count": len(api_only),
        "avg_diff": avg_diff,
        "avg_abs_diff": avg_abs_diff,
        "max_diff": max_diff,
        "differences": differences[:20],  # Top 20 differences
    }
    
    if verbose:
        print(f"\n=== Line Comparison ===")
        print(f"Common players/props: {result['common_count']}")
        print(f"Scraped only: {result['scraped_only_count']}")
        print(f"API only: {result['api_only_count']}")
        print(f"Average difference: {avg_diff:.2f}")
        print(f"Average absolute difference: {avg_abs_diff:.2f}")
        print(f"Max difference: {max_diff:.2f}")
        
        if differences:
            print(f"\nTop line differences:")
            for d in differences[:10]:
                print(f"  {d['player']:<25} {d['prop_type']}: "
                      f"Scraped={d['scraped_line']:.1f} API={d['api_line']:.1f} "
                      f"Diff={d['diff']:+.1f} ({d['pct_diff']:+.1f}%)")
    
    return result


# ============================================================================
# CLI Helper Functions
# ============================================================================

def print_scraped_summary(props: list[ScrapedPlayerProp]):
    """Print a summary of scraped props."""
    if not props:
        print("No props found.")
        return
    
    # Group by prop type
    by_type = defaultdict(list)
    for prop in props:
        by_type[prop.prop_type].append(prop)
    
    print(f"Scraped {len(props)} player props:")
    for prop_type, type_props in sorted(by_type.items()):
        print(f"\n  {prop_type} ({len(type_props)} players):")
        # Sort by line descending
        type_props.sort(key=lambda p: p.line, reverse=True)
        # Show top 10
        for prop in type_props[:10]:
            team_str = f"({prop.team})" if prop.team else ""
            print(f"    {prop.player_name:<25} {team_str:<6} {prop.line:>5.1f}")
        if len(type_props) > 10:
            print(f"    ... and {len(type_props) - 10} more")
