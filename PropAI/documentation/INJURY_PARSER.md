# NBA Injury Report System - Technical Documentation

## Overview

The injury system supports **two data sources**:

1. **ESPN Web Scraper (Primary)** - Automatic scraping from ESPN's injury page
2. **Manual Parser (Secondary)** - Copy-paste from official NBA injury reports

Both sources store data in the same `injury_report` database table, allowing consistent querying throughout the application.

---

## Data Source 1: ESPN Web Scraper (Primary)

### How It Works

The ESPN scraper (`src/nba_props/ingest/web_scraper.py`) automatically fetches injury data from ESPN's NBA injuries page at `https://www.espn.com/nba/injuries`.

### Web Interface Usage

1. Navigate to **Data > Web Scraping** in the web interface
2. Click **"🏥 Fetch Injuries"** to scrape current injury data from ESPN
3. Review the scraped injuries displayed on screen
4. Click **"💾 Save to Database"** to store the injuries

### ESPN Status Mapping

ESPN uses different status terms than the standard NBA format. The system automatically maps them:

| ESPN Status | Mapped Status | Betting Implication |
|-------------|---------------|---------------------|
| Out | OUT | Exclude from bets |
| Suspension | OUT | Exclude from bets |
| Injured Reserve | OUT | Exclude from bets |
| Day-To-Day | QUESTIONABLE | Monitor closely |
| Questionable | QUESTIONABLE | 50/50 to play |
| Doubtful | DOUBTFUL | Likely out (25% chance) |
| Probable | PROBABLE | Should play (75% chance) |
| Available/Active | *(skipped)* | No action needed |

### API Endpoints for ESPN Scraping

**Scrape injuries from ESPN:**
```
POST /api/scrape/injuries
Content-Type: application/json

{}
```

**Response:**
```json
{
    "success": true,
    "scrape_time": "2026-02-03T14:30:00",
    "count": 45,
    "injuries": [
        {
            "team_name": "Boston Celtics",
            "player_name": "Jaylen Brown",
            "position": "SF",
            "status": "Day-To-Day",
            "est_return_date": "Feb 5",
            "comment": "Right shoulder soreness"
        }
    ],
    "text": "... formatted text output ..."
}
```

**Save scraped injuries to database:**
```
POST /api/scrape/injuries/save
Content-Type: application/json

{
    "injuries": [...],  // from scrape response
    "date": "2026-02-03"  // optional, defaults to today
}
```

**Response:**
```json
{
    "success": true,
    "saved": 30,
    "updated": 10,
    "total": 40,
    "date": "2026-02-03"
}
```

---

## Data Source 2: Manual Parser (Secondary)

The manual parser (`src/nba_props/ingest/injury_parser.py`) is for parsing copy-pasted NBA injury reports from the official NBA format when ESPN data is unavailable or for historical backfilling.

## Manual Input Format

The NBA releases injury reports in a specific format, typically copied from PDF documents. Here's an example:

```
Injury Report: 01/14/26 11:45 AM
Page 1 of 6
Game Date Game Time Matchup Team Player Name Current Status Reason
01/14/2026 07:00 (ET) CLE@PHI Cleveland Cavaliers Livingston, Chris Out G League - Two-Way
Strus, Max Out
Injury/Illness - Left Foot; Surgery -
Jones Fracture
Philadelphia 76ers Embiid, Joel Probable
Injury/Illness - Left Knee; Injury
Management
TOR@IND Toronto Raptors Barrett, RJ Out Injury/Illness - Left Ankle; Sprain
Injury Report: 01/14/26 11:45 AM
Page 2 of 6
Quickley, Immanuel Questionable Injury/Illness - Back; Spasms
```

### Format Characteristics

1. **Header**: Report timestamp at the top
2. **Page breaks**: "Page X of Y" appears throughout, breaking the flow
3. **Game blocks**: Each game has:
   - Date (e.g., `01/14/2026`)
   - Time (e.g., `07:00 (ET)`)
   - Matchup (e.g., `CLE@PHI`)
4. **Team blocks**: Teams within a game (away team first, then home team)
5. **Player entries**: Player name in "LastName, FirstName" format
6. **Status**: OUT, QUESTIONABLE, PROBABLE, DOUBTFUL, AVAILABLE
7. **Reason**: Injury details (may span multiple lines)
8. **NOT YET SUBMITTED**: Some teams haven't submitted their report yet

---

## Key Parsing Challenges

### Challenge 1: New Matchup Detection

**Problem**: Matchup blocks don't always start with a date. Sometimes a new matchup appears on a line like:
```
TOR@IND Toronto Raptors Barrett, RJ Out Injury/Illness - Left Ankle; Sprain
```

Without proper detection, players from Toronto would be incorrectly attributed to Philadelphia (the previous team).

**Solution**: Check for matchup pattern (`XXX@YYY`) as the **FIRST** check in the parsing loop, before checking for dates or other patterns. This ensures the matchup context is always updated when a new game block starts.

### Challenge 2: Page Breaks Mid-Team

**Problem**: Page breaks like "Page 2 of 6" can appear in the middle of a team's player list:
```
Poeltl, Jakob Out Injury/Illness - Lower Back; Strain
Injury Report: 01/14/26 11:45 AM
Page 2 of 6
Quickley, Immanuel Questionable Injury/Illness - Back; Spasms
```

If not handled correctly, the team context would be lost and subsequent players would be orphaned.

**Solution**: Page breaks and report headers are identified and skipped WITHOUT resetting the team context. The parser continues attributing players to the current team.

### Challenge 3: NOT YET SUBMITTED with Matchup

**Problem**: Lines can contain both a matchup AND "NOT YET SUBMITTED":
```
09:30 (ET) DEN@DAL Denver Nuggets NOT YET SUBMITTED
Dallas Mavericks Christie, Max Doubtful ...
```

The original parser checked for "NOT YET SUBMITTED" first, which skipped the line without updating the matchup context. This caused Dallas players to be attributed to the previous matchup (UTA@CHI).

**Solution**: 
1. Check for matchup pattern FIRST and update `current_matchup`
2. THEN check if the line contains "NOT YET SUBMITTED"
3. Add the team to `teams_not_submitted` list
4. Continue to next line with updated matchup context

### Challenge 4: Multi-line Reasons

**Problem**: Injury reasons can span multiple lines:
```
Strus, Max Out
Injury/Illness - Left Foot; Surgery -
Jones Fracture
```

**Solution**: The parser treats continuation lines (without a status word) as part of the previous entry. This is handled by only parsing player entries when a status word is found.

### Challenge 5: Player Name Normalization

**Problem**: 
- Injury reports use "LastName, FirstName" format
- Names may lack accents (e.g., "Doncic" vs "Dončić")
- Suffixes need proper handling (e.g., "Oubre Jr., Kelly" → "Kelly Oubre Jr.")

**Solution**: 
1. `_parse_player_name()` converts "LastName, FirstName" to "FirstName LastName"
2. `normalize_player_name_for_db_match()` removes accents using Unicode normalization
3. The API uses fuzzy matching when linking to database players

---

## Parser Architecture

### Data Structures

```python
@dataclass
class InjuryEntry:
    game_date: str       # YYYY-MM-DD format
    team_name: str       # Full team name
    team_abbrev: str     # 3-letter abbreviation
    player_name: str     # "FirstName LastName" format
    status: str          # OUT, QUESTIONABLE, PROBABLE, DOUBTFUL, AVAILABLE
    reason: str          # Injury details
    game_time: str       # Optional game time
    matchup: str         # e.g., "CLE@PHI"
    is_g_league: bool    # True if G-League assignment

@dataclass
class ParsedInjuryReport:
    report_date: str                    # Report timestamp
    entries: list[InjuryEntry]          # All parsed entries
    teams_not_submitted: list[str]      # Teams with "NOT YET SUBMITTED"
    warnings: list[str]                 # Any parsing issues
```

### Parsing Flow

```
1. EXTRACT REPORT HEADER
   └── Get report date/time from "Injury Report: MM/DD/YY HH:MM AM/PM"

2. FOR EACH LINE:
   ├── Skip empty lines
   ├── Skip page breaks (preserve team context!)
   │
   ├── CHECK FOR MATCHUP (ABC@DEF)  ← MOST IMPORTANT CHECK
   │   ├── Update current_matchup
   │   ├── Update matchup_away_team, matchup_home_team
   │   ├── Extract date/time if present
   │   ├── Find team name after matchup
   │   ├── If NOT YET SUBMITTED → add to list, continue
   │   └── Parse player entry if present
   │
   ├── CHECK FOR NOT YET SUBMITTED (without matchup)
   │   └── Extract team name, add to list
   │
   ├── CHECK FOR DATE LINE
   │   └── Update current_game_date
   │
   ├── CHECK FOR TEAM NAME AT START
   │   ├── Update current_team
   │   ├── Validate against current matchup (emit warning if mismatch)
   │   └── Parse player entry from remainder
   │
   └── PARSE AS PLAYER ENTRY (with current team context)
       └── Extract player name, status, reason
```

### Key Detection Patterns

```python
# Matchup pattern: "CLE@PHI", "TOR@IND", etc.
matchup_pattern = re.compile(r'\b([A-Z]{2,3})@([A-Z]{2,3})\b')

# Date pattern at start of line
date_pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4})\s+')

# Time pattern
time_pattern = re.compile(r'(\d{1,2}:\d{2})\s*\(ET\)')

# Status words
status_pattern = re.compile(r'\b(Out|Questionable|Probable|Doubtful|Available|GTD)\b', re.IGNORECASE)

# Team names (sorted by length, longest first)
team_pattern = re.compile(r'(Philadelphia 76ers|Golden State Warriors|...)', re.IGNORECASE)
```

---

## Status Meanings

| Status | Meaning | Betting Implication |
|--------|---------|---------------------|
| **OUT** | Player will not play | Exclude from prop bets, consider usage redistribution |
| **DOUBTFUL** | Unlikely to play (25% chance) | Treat as likely OUT |
| **QUESTIONABLE** | 50/50 to play | Monitor closely, may play limited minutes |
| **PROBABLE** | Likely to play (75% chance) | Should play, possibly limited |
| **AVAILABLE** | Cleared to play | Normal betting analysis |
| **GTD** | Game-Time Decision | Same as QUESTIONABLE |

---

## G-League Filtering

The parser identifies G-League assignments and two-way players:

```python
def _is_g_league_reason(reason: str) -> bool:
    return any(term in reason.lower() for term in [
        "g league",
        "two-way",
        "on assignment",
        "g-league",
    ])
```

These entries are:
- Still parsed and stored (for completeness)
- Marked with `is_g_league=True`
- Filtered out by `filter_meaningful_injuries()` for betting analysis

---

## Database Integration

### Storage Schema

```sql
CREATE TABLE injury_report (
    id INTEGER PRIMARY KEY,
    game_date TEXT,           -- YYYY-MM-DD
    team_id INTEGER,          -- FK to teams table
    player_id INTEGER,        -- FK to players table (nullable)
    player_name TEXT,         -- Original name from report
    status TEXT,              -- OUT, QUESTIONABLE, etc.
    minutes_limit INTEGER,    -- Optional minutes restriction
    notes TEXT                -- Injury reason/details
);
```

### Player Matching

When ingesting, the API attempts to match players to the database:

```python
def normalize_name_for_match(name: str) -> str:
    """Remove accents and lowercase for matching."""
    nfkd = unicodedata.normalize('NFKD', name)
    ascii_name = ''.join(c for c in nfkd if not unicodedata.combining(c))
    return ascii_name.lower().strip()

# Match by exact name
player_id = player_lookup.get(normalized_entry_name)

# Fallback: partial matching
if not player_id:
    for db_name, pid in player_lookup.items():
        if entry_name in db_name or db_name in entry_name:
            player_id = pid
            break
```

---

## API Endpoints

### Ingest Injury Report

```
POST /api/ingest/injury-report
Content-Type: application/json

{
    "text": "Injury Report: 01/14/26 11:45 AM\n...",
    "date": "2026-01-14",        // Optional: filter to specific date
    "include_g_league": false    // Optional: include G-League assignments
}
```

**Response:**
```json
{
    "success": true,
    "report_date": "01/14/26 11:45 AM",
    "total_entries": 47,
    "ingested": 30,
    "skipped": 17,
    "teams_not_submitted": ["New Orleans Pelicans", "Chicago Bulls"],
    "warnings": [],
    "summary": {
        "by_team": {
            "PHI": [{"player": "Joel Embiid", "status": "PROBABLE"}],
            "TOR": [{"player": "RJ Barrett", "status": "OUT"}]
        }
    }
}
```

### Get Injuries

```
GET /api/injuries?date=2026-01-14&team=PHI
```

---

## Best Practices for Data Entry

### 1. Copy the Full Report
Always copy the entire injury report, including headers and page breaks. The parser handles them correctly.

### 2. Include All Pages
Page breaks mid-team are handled correctly. Don't try to "clean up" the data before pasting.

### 3. Multiple Reports Per Day
Injury reports are released every 15 minutes. When pasting a new report:
- Existing entries are **updated** (not duplicated)
- New entries are added
- The `as_of` timestamp reflects the latest report

### 4. Check the Response
Always review the API response for:
- `warnings`: Any parsing issues
- `teams_not_submitted`: Teams that haven't reported yet
- `ingested` vs `skipped`: Ensure expected counts

### 5. G-League Players
By default, G-League assignments are skipped for betting analysis. Use `include_g_league: true` if you need them.

---

## Error Handling

### Warnings Generated

The parser emits warnings for:
1. **Team mismatch**: A team appears that doesn't match the current matchup
2. **Unknown team**: A team name that can't be normalized
3. **Parse failure**: A line that looks like it should contain player data but couldn't be parsed

### Validation

The parser validates:
- Player names are at least 2 characters
- Player names aren't all digits
- Status is a recognized value
- Team names are valid NBA teams

---

## Example Usage

### Python

```python
from nba_props.ingest.injury_parser import (
    parse_injury_report_text,
    filter_meaningful_injuries,
    summarize_injury_report,
)

# Parse raw text
report = parse_injury_report_text(raw_text)

# Get only meaningful injuries (no G-League)
meaningful = filter_meaningful_injuries(report)

# Get injuries for a specific team
phi_injuries = [e for e in report.entries if e.team_abbrev == "PHI"]

# Generate summary
summary = summarize_injury_report(report)
print(f"Teams not yet submitted: {summary['teams_not_submitted']}")
```

### API via curl

```bash
curl -X POST http://localhost:5000/api/ingest/injury-report \
  -H "Content-Type: application/json" \
  -d '{"text": "Injury Report: 01/14/26 11:45 AM\n..."}'
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Initial | Basic parsing |
| 2.0 | Jan 2026 | Fixed matchup detection priority |
| 2.1 | Jan 2026 | Fixed NOT YET SUBMITTED with matchup |
| 2.2 | Jan 2026 | Added warnings, improved name normalization |
| **3.0** | **Feb 2026** | **Added ESPN web scraper as primary data source** |

---

## Troubleshooting

### ESPN Scraper Issues

**Symptom**: "Web scraping dependencies not installed" error

**Solution**: Install required packages:
```bash
pip install requests beautifulsoup4
```

**Symptom**: Rate limited by ESPN (429 error)

**Solution**: The scraper has automatic retry logic with exponential backoff. If issues persist, wait a few minutes before retrying.

**Symptom**: No injuries scraped or missing teams

**Cause**: ESPN page structure may have changed

**Solution**: Check the ESPN injuries page manually. If structure changed significantly, the scraper may need updating.

### Manual Parser Issues

### Players Attributed to Wrong Team

**Symptom**: Toronto players showing up as Philadelphia players

**Cause**: The parser wasn't detecting new matchup blocks properly

**Solution**: Fixed in v2.0 - matchup detection now happens FIRST in the parsing loop

### Missing Players After Page Break

**Symptom**: Players after "Page 2 of 6" are missing

**Cause**: Page breaks were resetting team context

**Solution**: Fixed in v2.0 - page breaks are skipped without resetting context

### NOT YET SUBMITTED Breaking Context

**Symptom**: Players from next team have wrong matchup

**Cause**: NOT YET SUBMITTED lines with matchup weren't updating context

**Solution**: Fixed in v2.1 - matchup is extracted and updated before checking NOT YET SUBMITTED

---

## Querying Injury Data

Both data sources store injuries in the same `injury_report` table. Use the unified API to query:

**Get all injuries for a date:**
```
GET /api/injuries?date=2026-02-03
```

**Get injuries for a specific team:**
```
GET /api/injuries?date=2026-02-03&team=Celtics
```

**Get injuries for a matchup (for projections):**
```
GET /api/injuries?away=LAL&home=BOS&date=2026-02-03
```

**Response (matchup mode):**
```json
{
    "away_injuries": {
        "Anthony Davis": "QUESTIONABLE"
    },
    "home_injuries": {
        "Jaylen Brown": "QUESTIONABLE"
    },
    "date": "2026-02-03"
}
```

**Clear injuries for a date:**
```
POST /api/injuries/clear
Content-Type: application/json

{"date": "2026-02-03"}  // or {} to clear all
```

---

## Integration with Models

The injury data is automatically used by the prediction models:

1. **Matchup Advisor** (`matchup_advisor.py`): Filters out OUT/DOUBTFUL players from recommendations
2. **Under Model V2** (`under_model_v2.py`): Uses injury data to detect:
   - Elite defenders who are injured (affects UNDER picks)
   - Players returning from injury (injury rust factor)
3. **Usage Redistribution**: Calculates increased opportunity when star players are OUT

### How Models Access Injury Data

```python
from nba_props.engine.matchup_advisor import _get_team_injury_report

# Get injuries as dict: player_name -> status
injuries = _get_team_injury_report(conn, "BOS", "2026-02-03")
# Returns: {"Jaylen Brown": "QUESTIONABLE", ...}
```

---

*Last Updated: February 3, 2026*
