"""Parser for NBA matchup/schedule data with betting lines.

Parses copy-pasted schedule data like:
    Saturday, January 3, 2026
    ...
    Minnesota
      @  
    
    Miami
    3:00 PM	
    NBA TV
    Tickets as low as $28	
    Line: MIN -2.5
    O/U: 238.5
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..team_aliases import abbrev_from_team_name


# Team city/name mapping for partial matches
_CITY_TO_TEAM: dict[str, str] = {
    "atlanta": "Atlanta Hawks",
    "boston": "Boston Celtics",
    "brooklyn": "Brooklyn Nets",
    "charlotte": "Charlotte Hornets",
    "chicago": "Chicago Bulls",
    "cleveland": "Cleveland Cavaliers",
    "dallas": "Dallas Mavericks",
    "denver": "Denver Nuggets",
    "detroit": "Detroit Pistons",
    "golden state": "Golden State Warriors",
    "houston": "Houston Rockets",
    "indiana": "Indiana Pacers",
    "la clippers": "Los Angeles Clippers",
    "la lakers": "Los Angeles Lakers",
    "los angeles clippers": "Los Angeles Clippers",
    "los angeles lakers": "Los Angeles Lakers",
    "memphis": "Memphis Grizzlies",
    "miami": "Miami Heat",
    "milwaukee": "Milwaukee Bucks",
    "minnesota": "Minnesota Timberwolves",
    "new orleans": "New Orleans Pelicans",
    "new york": "New York Knicks",
    "oklahoma city": "Oklahoma City Thunder",
    "orlando": "Orlando Magic",
    "philadelphia": "Philadelphia 76ers",
    "phoenix": "Phoenix Suns",
    "portland": "Portland Trail Blazers",
    "sacramento": "Sacramento Kings",
    "san antonio": "San Antonio Spurs",
    "toronto": "Toronto Raptors",
    "utah": "Utah Jazz",
    "washington": "Washington Wizards",
    # Short forms
    "la": "Los Angeles Clippers",  # Default LA to Clippers (context-dependent)
}

# Abbreviation mapping for lines
_ABBREV_MAP: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "GS": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NO": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "NY": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "SA": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}


@dataclass
class ParsedMatchup:
    """Parsed matchup data from schedule paste."""
    game_date: str  # YYYY-MM-DD format
    away_team: str  # Full team name
    home_team: str  # Full team name
    away_abbrev: str
    home_abbrev: str
    game_time: Optional[str] = None  # e.g., "3:00 PM"
    spread: Optional[float] = None  # Positive = home favored, negative = away favored
    favorite_abbrev: Optional[str] = None  # Which team is favored
    over_under: Optional[float] = None
    tv_channel: Optional[str] = None


def _resolve_team_name(text: str) -> Optional[str]:
    """Resolve a city/team name to full team name."""
    text_lower = text.strip().lower()

    # Guard against matching team names inside non-team lines like:
    # "Line: LAL -4.5" or "Tickets as low as $65"
    if (":" in text_lower or any(ch.isdigit() for ch in text_lower)) and text_lower not in _CITY_TO_TEAM:
        return None
    
    # Empty or too short strings can't be team names
    if len(text_lower) < 2:
        return None
    
    # Check exact matches first
    if text_lower in _CITY_TO_TEAM:
        return _CITY_TO_TEAM[text_lower]
    
    # Check partial matches (avoid very short keys like "la" to prevent false positives, e.g. "LAL")
    if len(text_lower) >= 3:
        for city, team in _CITY_TO_TEAM.items():
            if len(city) < 3:
                continue
            if city in text_lower or text_lower in city:
                return team
    
    # Check if it's a full team name
    for team in _CITY_TO_TEAM.values():
        if text_lower in team.lower():
            return team
    
    return None


def _resolve_abbrev_to_team(abbrev: str) -> Optional[str]:
    """Resolve an abbreviation to full team name."""
    return _ABBREV_MAP.get(abbrev.upper())


def _parse_date(text: str) -> Optional[str]:
    """Parse a date string to YYYY-MM-DD format."""
    # Try common formats
    formats = [
        "%A, %B %d, %Y",  # "Saturday, January 3, 2026"
        "%B %d, %Y",      # "January 3, 2026"
        "%m/%d/%Y",       # "01/03/2026"
        "%m-%d-%Y",       # "01-03-2026"
        "%Y-%m-%d",       # "2026-01-03"
    ]
    
    text = text.strip()
    
    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return None


def _parse_spread_line(line: str) -> tuple[Optional[str], Optional[float]]:
    """
    Parse a spread line like 'Line: MIN -2.5' or 'Line: NY -3.5'.
    Returns (team_abbrev, spread_value) where positive = team is favored.
    """
    # Pattern: Line: ABBREV -X.X or Line: ABBREV +X.X
    m = re.search(r"Line:\s*([A-Z]{2,3})\s*([+-]?\d+(?:\.\d+)?)", line, re.IGNORECASE)
    if m:
        abbrev = m.group(1).upper()
        spread_val = float(m.group(2))
        return abbrev, spread_val
    return None, None


def _parse_over_under(line: str) -> Optional[float]:
    """Parse over/under from line like 'O/U: 238.5'."""
    m = re.search(r"O/U:\s*(\d+(?:\.\d+)?)", line, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_time(line: str) -> Optional[str]:
    """Parse game time from line like '3:00 PM'."""
    m = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", line, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def parse_matchups_text(text: str) -> list[ParsedMatchup]:
    """
    Parse pasted matchup schedule text into structured matchup data.
    
    The expected format is messy copy-paste from ESPN or similar:
    
        Saturday, January 3, 2026
        MATCHUP
        TIME
        TV
        tickets
        Odds by
        draft kings
        
        Minnesota
          @  
        
        Miami
        3:00 PM	
        NBA TV
        Tickets as low as $28	
        Line: MIN -2.5
        O/U: 238.5
        
        Philadelphia
          @  
        
        New York
        ...
    
    Returns:
        List of ParsedMatchup objects
    """
    matchups: list[ParsedMatchup] = []
    lines = text.splitlines()
    
    # First, try to find the date
    game_date: Optional[str] = None
    for line in lines[:10]:  # Check first 10 lines for date
        parsed = _parse_date(line.strip())
        if parsed:
            game_date = parsed
            break
    
    # If no date found, use today
    if not game_date:
        game_date = datetime.now().strftime("%Y-%m-%d")
    
    # Find matchup blocks
    # Pattern: Team1 (away) -> "@" -> Team2 (home) -> time -> channel -> line -> O/U
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip header/noise lines
        if not line or line.upper() in ("MATCHUP", "TIME", "TV", "TICKETS", "ODDS BY", "DRAFT KINGS"):
            i += 1
            continue
        
        # Check if this line looks like a team city/name
        away_team = _resolve_team_name(line)
        if not away_team:
            i += 1
            continue
        
        # Look for "@" indicator and home team in next lines
        home_team = None
        game_time = None
        tv_channel = None
        spread = None
        favorite_abbrev = None
        over_under = None
        
        # Scan ahead for the rest of the matchup info
        j = i + 1
        found_at = False
        while j < len(lines) and j < i + 15:  # Look ahead up to 15 lines
            next_line = lines[j].strip()
            
            # Found the @ separator
            if "@" in next_line and len(next_line) < 5:
                found_at = True
                j += 1
                continue
            
            # After @, look for home team
            if found_at and not home_team:
                potential_home = _resolve_team_name(next_line)
                if potential_home:
                    home_team = potential_home
                    j += 1
                    continue
            
            # Look for game time
            if not game_time:
                time_match = _parse_time(next_line)
                if time_match:
                    game_time = time_match
            
            # Look for TV channel
            if "NBA TV" in next_line.upper() or "ESPN" in next_line.upper() or "TNT" in next_line.upper():
                tv_channel = next_line.strip()
            
            # Look for spread line
            if "line:" in next_line.lower():
                abbrev, val = _parse_spread_line(next_line)
                if abbrev and val is not None:
                    favorite_abbrev = abbrev
                    spread = val
            
            # Look for O/U
            if "o/u:" in next_line.lower():
                over_under = _parse_over_under(next_line)
            
            # Stop if we hit another likely team line (start of next matchup)
            if home_team and _resolve_team_name(next_line) and _parse_time(next_line) is None and ":" not in next_line:
                break
            
            j += 1
        
        # If we found a valid matchup, add it
        if away_team and home_team:
            away_abbrev = abbrev_from_team_name(away_team) or ""
            home_abbrev = abbrev_from_team_name(home_team) or ""

            # Disambiguate "Los Angeles" when the pasted matchup doesn't specify Lakers vs Clippers.
            # If the spread favorite indicates LAL/LAC but our resolved team abbrevs don't match,
            # prefer the favorite abbrev for the LA team.
            if favorite_abbrev in ("LAL", "LAC") and favorite_abbrev not in (away_abbrev, home_abbrev):
                if away_team.startswith("Los Angeles"):
                    away_team = "Los Angeles Lakers" if favorite_abbrev == "LAL" else "Los Angeles Clippers"
                    away_abbrev = favorite_abbrev
                elif home_team.startswith("Los Angeles"):
                    home_team = "Los Angeles Lakers" if favorite_abbrev == "LAL" else "Los Angeles Clippers"
                    home_abbrev = favorite_abbrev
            
            # Convert spread to home team perspective
            # If favorite is away team, spread is negative for home
            if spread is not None and favorite_abbrev:
                if favorite_abbrev == away_abbrev:
                    # Away team favored, home team spread is positive (underdog)
                    spread = abs(spread)
                elif favorite_abbrev == home_abbrev:
                    # Home team favored, spread is negative
                    spread = -abs(spread)
            
            matchups.append(ParsedMatchup(
                game_date=game_date,
                away_team=away_team,
                home_team=home_team,
                away_abbrev=away_abbrev,
                home_abbrev=home_abbrev,
                game_time=game_time,
                spread=spread,
                favorite_abbrev=favorite_abbrev,
                over_under=over_under,
                tv_channel=tv_channel,
            ))
            
            # Move past this matchup
            i = j
        else:
            i += 1
    
    return matchups


def parse_simple_matchup(away: str, home: str, date: str, spread: Optional[float] = None, over_under: Optional[float] = None) -> Optional[ParsedMatchup]:
    """Create a matchup from simple inputs (for manual entry)."""
    away_team = _resolve_team_name(away) or _resolve_abbrev_to_team(away)
    home_team = _resolve_team_name(home) or _resolve_abbrev_to_team(home)
    
    if not away_team or not home_team:
        return None
    
    away_abbrev = abbrev_from_team_name(away_team) or ""
    home_abbrev = abbrev_from_team_name(home_team) or ""
    
    return ParsedMatchup(
        game_date=date,
        away_team=away_team,
        home_team=home_team,
        away_abbrev=away_abbrev,
        home_abbrev=home_abbrev,
        spread=spread,
        over_under=over_under,
    )

