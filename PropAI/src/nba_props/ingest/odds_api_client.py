"""
The Odds API client for fetching NBA player prop lines.

This module provides functions to fetch player prop betting lines from The Odds API:
- NBA game events
- Player points props (over/under)
- Player rebounds props (over/under)
- Player assists props (over/under)

API Documentation: https://the-odds-api.com/liveapi/guides/v4/

Requires: requests
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Any
from urllib.parse import urlencode

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ============================================================================
# Configuration
# ============================================================================

# Default API key - can be overridden via environment variable or function parameter
DEFAULT_API_KEY = "00ecbed197d3a7be09df31d336d2afaa"

# API base URL
API_BASE_URL = "https://api.the-odds-api.com"

# Sport key for NBA
SPORT_KEY = "basketball_nba"

# Player prop markets we care about
PLAYER_PROP_MARKETS = [
    "player_points",
    "player_rebounds", 
    "player_assists",
]

# Regions to fetch (us = American odds from US books)
REGIONS = ["us"]

# Preferred bookmakers in order of preference
PREFERRED_BOOKS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pointsbet",
    "bovada",
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class OddsAPIPlayerProp:
    """A single player prop line from The Odds API."""
    event_id: str
    event_date: str  # ISO format
    home_team: str
    away_team: str
    bookmaker: str
    market: str  # player_points, player_rebounds, player_assists
    player_name: str
    prop_type: str  # PTS, REB, AST
    line: float
    over_price: float  # decimal odds
    under_price: float  # decimal odds
    over_odds_american: int
    under_odds_american: int
    fetched_at: str  # ISO timestamp


@dataclass
class OddsAPIEvent:
    """An NBA game event from The Odds API."""
    id: str
    sport_key: str
    sport_title: str
    commence_time: str  # ISO format
    home_team: str
    away_team: str


@dataclass
class OddsAPIUsage:
    """API usage information returned in response headers."""
    requests_used: int
    requests_remaining: int


@dataclass
class OddsAPIResponse:
    """Response from The Odds API with data and usage info."""
    data: Any
    usage: OddsAPIUsage
    success: bool
    error_message: Optional[str] = None


# ============================================================================
# Utility Functions
# ============================================================================

def _decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


def _market_to_prop_type(market: str) -> str:
    """Convert API market name to our prop type."""
    mapping = {
        "player_points": "PTS",
        "player_rebounds": "REB",
        "player_assists": "AST",
        "player_points_alternate": "PTS",
        "player_rebounds_alternate": "REB",
        "player_assists_alternate": "AST",
    }
    return mapping.get(market, market.upper())


def _normalize_player_name(name: str) -> str:
    """Normalize player name for consistent matching."""
    # Handle special characters and normalize whitespace
    name = name.strip()
    # Remove Jr., Sr., III, II suffixes for matching but keep canonical form
    return " ".join(name.split())


def _get_api_key() -> str:
    """Get API key from environment or use default."""
    return os.environ.get("ODDS_API_KEY", DEFAULT_API_KEY)


# ============================================================================
# API Request Functions
# ============================================================================

def _make_api_request(
    endpoint: str, 
    params: dict = None,
    api_key: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> OddsAPIResponse:
    """Make a request to The Odds API with retry logic."""
    if not HAS_REQUESTS:
        raise ImportError("requests is required for The Odds API. Install with: pip install requests")
    
    key = api_key or _get_api_key()
    
    # Build URL with API key
    if params is None:
        params = {}
    params["apiKey"] = key
    
    url = f"{API_BASE_URL}{endpoint}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Extract usage info from headers
            usage = OddsAPIUsage(
                requests_used=int(response.headers.get("x-requests-used", 0)),
                requests_remaining=int(response.headers.get("x-requests-remaining", 0)),
            )
            
            if response.status_code == 200:
                return OddsAPIResponse(
                    data=response.json(),
                    usage=usage,
                    success=True,
                )
            elif response.status_code == 401:
                return OddsAPIResponse(
                    data=None,
                    usage=usage,
                    success=False,
                    error_message="Invalid API key",
                )
            elif response.status_code == 422:
                return OddsAPIResponse(
                    data=None,
                    usage=usage,
                    success=False,
                    error_message=f"Invalid parameters: {response.text}",
                )
            elif response.status_code == 429:
                # Rate limited
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                return OddsAPIResponse(
                    data=None,
                    usage=usage,
                    success=False,
                    error_message="Rate limit exceeded",
                )
            else:
                return OddsAPIResponse(
                    data=None,
                    usage=usage,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}",
                )
                
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            return OddsAPIResponse(
                data=None,
                usage=OddsAPIUsage(0, 0),
                success=False,
                error_message=f"Request error: {str(e)}",
            )
    
    return OddsAPIResponse(
        data=None,
        usage=OddsAPIUsage(0, 0),
        success=False,
        error_message="Max retries exceeded",
    )


# ============================================================================
# Public API Functions
# ============================================================================

def get_api_usage(api_key: Optional[str] = None) -> OddsAPIResponse:
    """
    Check API usage/quota status.
    
    This is a free endpoint that returns usage info in headers.
    """
    response = _make_api_request(
        f"/v4/sports",
        api_key=api_key,
    )
    return response


def fetch_nba_events(
    api_key: Optional[str] = None,
    date_filter: Optional[str] = None,
) -> OddsAPIResponse:
    """
    Fetch all NBA events (games).
    
    This endpoint is FREE - does not count against quota.
    
    Args:
        api_key: Optional API key override
        date_filter: Optional date filter (YYYY-MM-DD) - only return games on/after this date
        
    Returns:
        OddsAPIResponse with list of OddsAPIEvent objects
    """
    params = {}
    
    # Filter to only upcoming events
    # IMPORTANT: NBA games typically start 7 PM - 10:30 PM local US time
    # This means evening games on Feb 4 local might be Feb 5 in UTC
    # To handle this, we extend the window to include both the requested date
    # and the next day in UTC time
    if date_filter:
        # Parse the date and create a window that spans local evening to next day morning in UTC
        # This ensures we catch games that appear on the "next" UTC day
        from datetime import datetime as dt, timedelta
        try:
            local_date = dt.strptime(date_filter, "%Y-%m-%d")
            # Start from 10 AM UTC (5 AM EST / 3 AM MST) of requested date to catch any daytime games
            # End at 11:59 PM UTC of NEXT day to catch evening games that cross midnight UTC
            next_date = local_date + timedelta(days=1)
            params["commenceTimeFrom"] = f"{date_filter}T10:00:00Z"
            params["commenceTimeTo"] = f"{next_date.strftime('%Y-%m-%d')}T11:59:59Z"
        except ValueError:
            # Fallback to simple date filtering if parsing fails
            params["commenceTimeFrom"] = f"{date_filter}T00:00:00Z"
            params["commenceTimeTo"] = f"{date_filter}T23:59:59Z"
    
    response = _make_api_request(
        f"/v4/sports/{SPORT_KEY}/events",
        params=params,
        api_key=api_key,
    )
    
    if response.success and response.data:
        events = []
        for event_data in response.data:
            events.append(OddsAPIEvent(
                id=event_data["id"],
                sport_key=event_data["sport_key"],
                sport_title=event_data["sport_title"],
                commence_time=event_data["commence_time"],
                home_team=event_data["home_team"],
                away_team=event_data["away_team"],
            ))
        response.data = events
    
    return response


def fetch_player_props_for_event(
    event_id: str,
    markets: list[str] = None,
    regions: list[str] = None,
    bookmakers: list[str] = None,
    api_key: Optional[str] = None,
) -> OddsAPIResponse:
    """
    Fetch player props for a specific NBA event.
    
    COSTS QUOTA: Each market costs 1 request per region.
    Example: player_points + player_rebounds in US region = 2 requests
    
    Args:
        event_id: The event ID from fetch_nba_events()
        markets: List of markets (default: player_points, player_rebounds, player_assists)
        regions: List of regions (default: us)
        bookmakers: Optional list of specific bookmakers to filter
        api_key: Optional API key override
        
    Returns:
        OddsAPIResponse with list of OddsAPIPlayerProp objects
    """
    if markets is None:
        markets = PLAYER_PROP_MARKETS
    if regions is None:
        regions = REGIONS
    
    params = {
        "regions": ",".join(regions),
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
    }
    
    if bookmakers:
        params["bookmakers"] = ",".join(bookmakers)
    
    response = _make_api_request(
        f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds",
        params=params,
        api_key=api_key,
    )
    
    if response.success and response.data:
        props = _parse_player_props_response(response.data)
        response.data = props
    
    return response


def _parse_player_props_response(data: dict) -> list[OddsAPIPlayerProp]:
    """Parse the API response into OddsAPIPlayerProp objects."""
    props = []
    
    event_id = data.get("id", "")
    commence_time = data.get("commence_time", "")
    home_team = data.get("home_team", "")
    away_team = data.get("away_team", "")
    fetched_at = datetime.utcnow().isoformat() + "Z"
    
    bookmakers = data.get("bookmakers", [])
    
    for bookmaker in bookmakers:
        book_key = bookmaker.get("key", "")
        book_markets = bookmaker.get("markets", [])
        
        for market in book_markets:
            market_key = market.get("key", "")
            prop_type = _market_to_prop_type(market_key)
            outcomes = market.get("outcomes", [])
            
            # Group outcomes by player (Over and Under for same player)
            player_outcomes = {}
            for outcome in outcomes:
                player_name = outcome.get("description", "")
                if not player_name:
                    continue
                    
                if player_name not in player_outcomes:
                    player_outcomes[player_name] = {}
                
                outcome_name = outcome.get("name", "").lower()
                player_outcomes[player_name][outcome_name] = {
                    "price": outcome.get("price", 0),
                    "point": outcome.get("point", 0),
                }
            
            # Create prop entries
            for player_name, outcomes_dict in player_outcomes.items():
                over_data = outcomes_dict.get("over", {})
                under_data = outcomes_dict.get("under", {})
                
                if not over_data or not under_data:
                    continue
                
                # Line should be the same for over and under
                line = over_data.get("point", under_data.get("point", 0))
                over_price = over_data.get("price", 1.91)
                under_price = under_data.get("price", 1.91)
                
                props.append(OddsAPIPlayerProp(
                    event_id=event_id,
                    event_date=commence_time,
                    home_team=home_team,
                    away_team=away_team,
                    bookmaker=book_key,
                    market=market_key,
                    player_name=_normalize_player_name(player_name),
                    prop_type=prop_type,
                    line=line,
                    over_price=over_price,
                    under_price=under_price,
                    over_odds_american=_decimal_to_american(over_price),
                    under_odds_american=_decimal_to_american(under_price),
                    fetched_at=fetched_at,
                ))
    
    return props


def fetch_all_player_props_for_date(
    date: str,
    markets: list[str] = None,
    preferred_book: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> tuple[list[OddsAPIPlayerProp], OddsAPIUsage]:
    """
    Fetch all player props for all NBA games on a specific date.
    
    This is a convenience function that:
    1. Fetches all NBA events for the date (FREE)
    2. Fetches player props for each event (COSTS QUOTA)
    
    Args:
        date: Date string in YYYY-MM-DD format
        markets: List of markets to fetch (default: PTS and REB only to save quota)
        preferred_book: If specified, only return lines from this bookmaker
        api_key: Optional API key override
        verbose: Print progress information
        
    Returns:
        Tuple of (list of props, final usage info)
    """
    if markets is None:
        # Default to just PTS and REB (assists are less reliable per model analysis)
        markets = ["player_points", "player_rebounds"]
    
    all_props = []
    final_usage = OddsAPIUsage(0, 0)
    
    # Step 1: Get events for the date (FREE)
    if verbose:
        print(f"Fetching NBA events for {date}...")
    
    events_response = fetch_nba_events(api_key=api_key, date_filter=date)
    
    if not events_response.success:
        print(f"Error fetching events: {events_response.error_message}")
        return [], final_usage
    
    events = events_response.data
    if not events:
        if verbose:
            print(f"No NBA events found for {date}")
        return [], events_response.usage
    
    if verbose:
        print(f"Found {len(events)} NBA events for {date}")
    
    # Step 2: Fetch props for each event (COSTS QUOTA)
    for i, event in enumerate(events):
        if verbose:
            print(f"  [{i+1}/{len(events)}] Fetching props for {event.away_team} @ {event.home_team}...")
        
        props_response = fetch_player_props_for_event(
            event_id=event.id,
            markets=markets,
            api_key=api_key,
        )
        
        final_usage = props_response.usage
        
        if not props_response.success:
            if verbose:
                print(f"    Warning: {props_response.error_message}")
            continue
        
        props = props_response.data
        
        # Filter by preferred book if specified
        if preferred_book:
            props = [p for p in props if p.bookmaker == preferred_book]
        
        all_props.extend(props)
        
        if verbose:
            print(f"    Got {len(props)} props (remaining quota: {final_usage.requests_remaining})")
        
        # Small delay between requests to be nice to the API
        if i < len(events) - 1:
            time.sleep(0.5)
    
    # Deduplicate: keep best line from preferred books
    if not preferred_book:
        all_props = _deduplicate_props(all_props)
    
    if verbose:
        print(f"Total props fetched: {len(all_props)}")
        print(f"API quota remaining: {final_usage.requests_remaining}")
    
    return all_props, final_usage


def _deduplicate_props(props: list[OddsAPIPlayerProp]) -> list[OddsAPIPlayerProp]:
    """
    Deduplicate props by keeping one line per player per prop type.
    
    Priority: Use the line from the first preferred bookmaker found.
    """
    # Create lookup by (player, prop_type) -> list of props
    by_player_prop = {}
    for prop in props:
        key = (prop.player_name.lower(), prop.prop_type)
        if key not in by_player_prop:
            by_player_prop[key] = []
        by_player_prop[key].append(prop)
    
    # For each player/prop, pick the best line
    deduplicated = []
    for key, prop_list in by_player_prop.items():
        # Sort by bookmaker preference
        def book_priority(p: OddsAPIPlayerProp) -> int:
            try:
                return PREFERRED_BOOKS.index(p.bookmaker)
            except ValueError:
                return len(PREFERRED_BOOKS)
        
        prop_list.sort(key=book_priority)
        deduplicated.append(prop_list[0])
    
    return deduplicated


# ============================================================================
# Player Name Matching Utilities
# ============================================================================

def normalize_name_for_matching(name: str) -> str:
    """
    Normalize player name for fuzzy matching with database.
    
    Handles:
    - Different capitalization
    - Jr./Sr./III/II suffixes
    - Accent marks
    - Nicknames vs full names
    """
    import unicodedata
    
    name = name.strip().lower()
    
    # Remove accent marks
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))
    
    # Remove suffixes
    suffixes = [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Normalize whitespace
    name = ' '.join(name.split())
    
    return name


def find_player_id_by_name(
    conn,
    api_player_name: str,
    team_hint: Optional[str] = None,
) -> Optional[int]:
    """
    Find a player ID in the database matching the API player name.
    
    Uses fuzzy matching to handle name variations.
    
    Args:
        conn: SQLite connection
        api_player_name: Player name from the API
        team_hint: Optional team name to narrow search
        
    Returns:
        Player ID if found, None otherwise
    """
    normalized_api_name = normalize_name_for_matching(api_player_name)
    
    # First try exact match (case-insensitive)
    row = conn.execute(
        "SELECT id, name FROM players WHERE LOWER(name) = ?",
        (api_player_name.lower(),)
    ).fetchone()
    
    if row:
        return row["id"]
    
    # Try normalized match
    all_players = conn.execute("SELECT id, name FROM players").fetchall()
    
    for player in all_players:
        if normalize_name_for_matching(player["name"]) == normalized_api_name:
            return player["id"]
    
    # Try partial match (last name)
    api_parts = normalized_api_name.split()
    if len(api_parts) >= 2:
        api_last = api_parts[-1]
        api_first = api_parts[0]
        
        for player in all_players:
            db_normalized = normalize_name_for_matching(player["name"])
            db_parts = db_normalized.split()
            if len(db_parts) >= 2:
                db_last = db_parts[-1]
                db_first = db_parts[0]
                
                # Match on last name + first initial
                if db_last == api_last and db_first[0] == api_first[0]:
                    return player["id"]
    
    return None


def create_player_if_not_exists(conn, player_name: str) -> int:
    """
    Get or create a player in the database.
    
    Returns the player ID.
    """
    # First try to find existing
    player_id = find_player_id_by_name(conn, player_name)
    if player_id:
        return player_id
    
    # Create new player
    cur = conn.execute(
        "INSERT INTO players(name) VALUES (?)",
        (player_name,)
    )
    return cur.lastrowid


# ============================================================================
# Database Storage Functions
# ============================================================================

def store_player_props(
    conn,
    props: list[OddsAPIPlayerProp],
    as_of_date: str,
    verbose: bool = False,
) -> int:
    """
    Store fetched player props in the sportsbook_lines table.
    
    Args:
        conn: SQLite connection
        props: List of props from fetch_all_player_props_for_date()
        as_of_date: Date string (YYYY-MM-DD) for the lines
        verbose: Print progress
        
    Returns:
        Number of props stored
    """
    stored_count = 0
    skipped_count = 0
    
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
            (as_of_date, player_id, prop.prop_type, prop.bookmaker)
        ).fetchone()
        
        if existing:
            # Update existing line
            conn.execute(
                """
                UPDATE sportsbook_lines 
                SET line = ?, odds_american = ?
                WHERE id = ?
                """,
                (prop.line, prop.over_odds_american, existing["id"])
            )
            if verbose:
                print(f"  Updated {prop.player_name} {prop.prop_type} = {prop.line}")
        else:
            # Insert new line
            conn.execute(
                """
                INSERT INTO sportsbook_lines(as_of_date, game_id, team_id, player_id, prop_type, line, odds_american, book)
                VALUES (?, NULL, NULL, ?, ?, ?, ?, ?)
                """,
                (as_of_date, player_id, prop.prop_type, prop.line, prop.over_odds_american, prop.bookmaker)
            )
            if verbose:
                print(f"  Inserted {prop.player_name} {prop.prop_type} = {prop.line}")
        
        stored_count += 1
    
    return stored_count


# ============================================================================
# CLI Helper Functions
# ============================================================================

def print_api_status(api_key: Optional[str] = None):
    """Print current API usage status."""
    response = get_api_usage(api_key=api_key)
    
    if response.success:
        print(f"The Odds API Status:")
        print(f"  Requests used: {response.usage.requests_used}")
        print(f"  Requests remaining: {response.usage.requests_remaining}")
    else:
        print(f"Error checking API status: {response.error_message}")


def print_events_summary(events: list[OddsAPIEvent]):
    """Print a summary of NBA events."""
    if not events:
        print("No events found.")
        return
    
    print(f"Found {len(events)} NBA events:")
    for event in events:
        # Parse and format the commence time
        try:
            dt = datetime.fromisoformat(event.commence_time.replace('Z', '+00:00'))
            time_str = dt.strftime("%I:%M %p ET")
        except:
            time_str = event.commence_time
        
        print(f"  {event.away_team} @ {event.home_team} - {time_str}")


def print_props_summary(props: list[OddsAPIPlayerProp]):
    """Print a summary of fetched props."""
    if not props:
        print("No props found.")
        return
    
    # Group by prop type
    by_type = {}
    for prop in props:
        if prop.prop_type not in by_type:
            by_type[prop.prop_type] = []
        by_type[prop.prop_type].append(prop)
    
    print(f"Fetched {len(props)} player props:")
    for prop_type, type_props in sorted(by_type.items()):
        print(f"\n  {prop_type} ({len(type_props)} players):")
        # Sort by line descending
        type_props.sort(key=lambda p: p.line, reverse=True)
        # Show top 10
        for prop in type_props[:10]:
            odds_str = f"{prop.over_odds_american:+d}" if prop.over_odds_american else ""
            print(f"    {prop.player_name:<25} {prop.line:>5.1f} {odds_str:>5} ({prop.bookmaker})")
        if len(type_props) > 10:
            print(f"    ... and {len(type_props) - 10} more")
