"""Parser for NBA injury report data.

Parses copy-pasted injury reports from the official NBA injury report format like:
    Injury Report: 01/03/26 02:45 PM
    Page 1 of 6
    Game Date Game Time Matchup Team Player Name Current Status Reason
    01/03/2026 05:00 (ET) MIN@MIA Minnesota Timberwolves Beringer, Joan Out G League - On Assignment
    ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from ..team_aliases import abbrev_from_team_name, team_name_from_abbrev, normalize_team_abbrev


@dataclass
class InjuryEntry:
    """A single injury report entry."""
    game_date: str  # YYYY-MM-DD format
    team_name: str
    team_abbrev: str
    player_name: str
    status: str  # OUT, QUESTIONABLE, PROBABLE, DOUBTFUL, AVAILABLE
    reason: str
    game_time: Optional[str] = None
    matchup: Optional[str] = None  # e.g., "MIN@MIA"
    is_g_league: bool = False  # G League assignment or Two-Way
    

@dataclass
class ParsedInjuryReport:
    """Complete parsed injury report."""
    report_date: str  # Date/time the report was generated
    entries: list[InjuryEntry] = field(default_factory=list)
    teams_not_submitted: list[str] = field(default_factory=list)


# Status mappings
_STATUS_MAP = {
    "OUT": "OUT",
    "O": "OUT",
    "QUESTIONABLE": "QUESTIONABLE",
    "Q": "QUESTIONABLE",
    "PROBABLE": "PROBABLE",
    "P": "PROBABLE",
    "DOUBTFUL": "DOUBTFUL",
    "D": "DOUBTFUL",
    "AVAILABLE": "AVAILABLE",
    "GTD": "QUESTIONABLE",  # Game Time Decision
}

# Team name mappings (full names)
_TEAM_NAMES = {
    "atlanta hawks": "Atlanta Hawks",
    "boston celtics": "Boston Celtics",
    "brooklyn nets": "Brooklyn Nets",
    "charlotte hornets": "Charlotte Hornets",
    "chicago bulls": "Chicago Bulls",
    "cleveland cavaliers": "Cleveland Cavaliers",
    "dallas mavericks": "Dallas Mavericks",
    "denver nuggets": "Denver Nuggets",
    "detroit pistons": "Detroit Pistons",
    "golden state warriors": "Golden State Warriors",
    "houston rockets": "Houston Rockets",
    "indiana pacers": "Indiana Pacers",
    "la clippers": "Los Angeles Clippers",
    "los angeles clippers": "Los Angeles Clippers",
    "la lakers": "Los Angeles Lakers",
    "los angeles lakers": "Los Angeles Lakers",
    "memphis grizzlies": "Memphis Grizzlies",
    "miami heat": "Miami Heat",
    "milwaukee bucks": "Milwaukee Bucks",
    "minnesota timberwolves": "Minnesota Timberwolves",
    "new orleans pelicans": "New Orleans Pelicans",
    "new york knicks": "New York Knicks",
    "oklahoma city thunder": "Oklahoma City Thunder",
    "orlando magic": "Orlando Magic",
    "philadelphia 76ers": "Philadelphia 76ers",
    "phoenix suns": "Phoenix Suns",
    "portland trail blazers": "Portland Trail Blazers",
    "sacramento kings": "Sacramento Kings",
    "san antonio spurs": "San Antonio Spurs",
    "toronto raptors": "Toronto Raptors",
    "utah jazz": "Utah Jazz",
    "washington wizards": "Washington Wizards",
}


def _normalize_team_name(name: str) -> Optional[str]:
    """Normalize a team name to its standard form."""
    name_lower = name.strip().lower()
    return _TEAM_NAMES.get(name_lower)


# Map team names to abbreviations
_TEAM_ABBREVS = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def _get_team_abbrev(team_name: str) -> str:
    """Get team abbreviation from full team name."""
    return _TEAM_ABBREVS.get(team_name, team_name[:3].upper())


def _parse_player_name(name: str) -> str:
    """Convert 'LastName, FirstName' to 'FirstName LastName'."""
    name = name.strip()
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    return name


def _normalize_status(status: str) -> str:
    """Normalize injury status to standard form."""
    status = status.strip().upper()
    return _STATUS_MAP.get(status, status)


def _is_g_league_reason(reason: str) -> bool:
    """Check if the reason indicates G League assignment."""
    reason_lower = reason.lower()
    return any(term in reason_lower for term in [
        "g league",
        "two-way",
        "on assignment",
        "g-league",
    ])


def _parse_date(date_str: str) -> Optional[str]:
    """Parse date string to YYYY-MM-DD format."""
    formats = [
        "%m/%d/%Y",  # 01/03/2026
        "%m/%d/%y",  # 01/03/26
        "%Y-%m-%d",  # 2026-01-03
    ]
    
    date_str = date_str.strip()
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    return None


def _parse_report_header(text: str) -> Optional[str]:
    """Extract the report date/time from header."""
    # Pattern: "Injury Report: 01/03/26 02:45 PM"
    m = re.search(r"Injury Report:\s*(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}\s*[AP]M)", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return None


def parse_injury_report_text(text: str) -> ParsedInjuryReport:
    """
    Parse the raw injury report text into structured data.
    
    The injury report format is typically:
    - Header with date/time
    - Table with columns: Game Date, Game Time, Matchup, Team, Player Name, Current Status, Reason
    - Some entries span multiple lines when reasons are long
    
    Returns:
        ParsedInjuryReport with all parsed entries
    """
    result = ParsedInjuryReport(
        report_date=_parse_report_header(text) or datetime.now().strftime("%Y-%m-%d %H:%M")
    )
    
    lines = text.splitlines()
    
    # Current parsing state
    current_game_date: Optional[str] = None
    current_game_time: Optional[str] = None
    current_matchup: Optional[str] = None
    current_team: Optional[str] = None
    
    # Pattern to match date at start of line: 01/03/2026
    date_pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4})\s+")
    # Pattern for game time: 05:00 (ET) or 07:30 (ET)
    time_pattern = re.compile(r"(\d{1,2}:\d{2})\s*\(ET\)")
    # Pattern for matchup: MIN@MIA or ATL@TOR
    matchup_pattern = re.compile(r"\b([A-Z]{2,3})@([A-Z]{2,3})\b")
    
    # Build a pattern to match any known team name (for searching anywhere in line)
    team_names_sorted = sorted(_TEAM_NAMES.values(), key=len, reverse=True)  # Longer names first
    team_pattern_anywhere = re.compile(
        r"(" + "|".join(re.escape(name) for name in team_names_sorted) + r")\s+",
        re.IGNORECASE
    )
    # Pattern to match team name at start of line
    team_pattern_start = re.compile(
        r"^(" + "|".join(re.escape(name) for name in team_names_sorted) + r")\s+",
        re.IGNORECASE
    )
    
    # Status pattern
    status_words = ["Out", "Questionable", "Probable", "Doubtful", "Available", "GTD"]
    status_pattern = re.compile(r"\b(" + "|".join(status_words) + r")\b", re.IGNORECASE)
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and headers
        if not line or line.startswith("Page ") or "Game Date" in line:
            i += 1
            continue
        
        # Skip the header
        if line.startswith("Injury Report:"):
            i += 1
            continue
        
        # Check for NOT YET SUBMITTED
        if "NOT YET SUBMITTED" in line.upper():
            # Try to extract team name
            for team_name in _TEAM_NAMES.values():
                if team_name.lower() in line.lower():
                    result.teams_not_submitted.append(team_name)
                    break
            i += 1
            continue
        
        # Check if line starts with a date (new game block or full entry)
        date_match = date_pattern.match(line)
        if date_match:
            parsed_date = _parse_date(date_match.group(1))
            if parsed_date:
                current_game_date = parsed_date
            
            # Try to extract time from same line
            time_match_local = time_pattern.search(line)
            if time_match_local:
                current_game_time = time_match_local.group(1)
            
            # Try to extract matchup from same line
            matchup_match = matchup_pattern.search(line)
            if matchup_match:
                current_matchup = f"{matchup_match.group(1)}@{matchup_match.group(2)}"
            
            # If the line also contains player info (full entry), parse it
            if current_game_date:
                team_match = team_pattern_anywhere.search(line)
                if team_match:
                    team_name = _normalize_team_name(team_match.group(1)) or team_match.group(1)
                    remaining = line[team_match.end():].strip()
                    entry = _parse_player_from_remaining(remaining, team_name, current_game_date, current_game_time, current_matchup)
                    if entry:
                        result.entries.append(entry)
                        current_team = team_name
            
            i += 1
            continue
        
        # Check if line starts with a known team name (continuation entry)
        team_match = team_pattern_start.match(line)
        if team_match:
            team_name = _normalize_team_name(team_match.group(1)) or team_match.group(1)
            current_team = team_name
            remaining = line[team_match.end():].strip()
            
            if current_game_date and remaining:
                entry = _parse_player_from_remaining(remaining, team_name, current_game_date, current_game_time, current_matchup)
                if entry:
                    result.entries.append(entry)
            
            i += 1
            continue
        
        # Try to parse as player entry with current team context
        if current_team and current_game_date:
            entry = _parse_player_from_remaining(line, current_team, current_game_date, current_game_time, current_matchup)
            if entry:
                result.entries.append(entry)
                i += 1
                continue
        
        i += 1
    
    return result


def _parse_player_from_remaining(
    remaining: str,
    team_name: str,
    game_date: str,
    game_time: Optional[str],
    matchup: Optional[str],
) -> Optional[InjuryEntry]:
    """Parse player info from the remaining part of a line after team name."""
    
    status_words = ["Out", "Questionable", "Probable", "Doubtful", "Available", "GTD"]
    status_pattern = re.compile(r"\b(" + "|".join(status_words) + r")\b", re.IGNORECASE)
    
    status_match = status_pattern.search(remaining)
    if not status_match:
        return None
    
    status = _normalize_status(status_match.group(1))
    
    # Everything before status is player name
    player_part = remaining[:status_match.start()].strip()
    # Everything after status is reason
    reason = remaining[status_match.end():].strip()
    
    # Parse player name (might be "LastName, FirstName" format)
    player_name = _parse_player_name(player_part)
    
    if not player_name:
        return None
    
    team_abbrev = _get_team_abbrev(team_name)
    is_g_league = _is_g_league_reason(reason)
    
    return InjuryEntry(
        game_date=game_date,
        team_name=team_name,
        team_abbrev=team_abbrev,
        player_name=player_name,
        status=status,
        reason=reason,
        game_time=game_time,
        matchup=matchup,
        is_g_league=is_g_league,
    )


def filter_meaningful_injuries(report: ParsedInjuryReport) -> list[InjuryEntry]:
    """
    Filter injury report to only include meaningful entries for betting analysis.
    Excludes G-League assignments and two-way players.
    """
    return [
        entry for entry in report.entries
        if not entry.is_g_league and entry.status in ("OUT", "QUESTIONABLE", "PROBABLE", "DOUBTFUL")
    ]


def get_injuries_by_team(report: ParsedInjuryReport, team: str) -> list[InjuryEntry]:
    """Get all injuries for a specific team."""
    team_upper = team.upper()
    return [
        entry for entry in report.entries
        if entry.team_abbrev.upper() == team_upper or entry.team_name.lower() == team.lower()
    ]


def get_injuries_for_date(report: ParsedInjuryReport, date: str) -> list[InjuryEntry]:
    """Get all injuries for a specific date."""
    return [entry for entry in report.entries if entry.game_date == date]


def summarize_injury_report(report: ParsedInjuryReport) -> dict:
    """Generate a summary of the injury report."""
    meaningful = filter_meaningful_injuries(report)
    
    by_status = {}
    by_team = {}
    
    for entry in meaningful:
        # Count by status
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
        
        # Group by team
        if entry.team_abbrev not in by_team:
            by_team[entry.team_abbrev] = []
        by_team[entry.team_abbrev].append({
            "player": entry.player_name,
            "status": entry.status,
            "reason": entry.reason,
        })
    
    return {
        "report_date": report.report_date,
        "total_entries": len(report.entries),
        "meaningful_entries": len(meaningful),
        "by_status": by_status,
        "by_team": by_team,
        "teams_not_submitted": report.teams_not_submitted,
    }
