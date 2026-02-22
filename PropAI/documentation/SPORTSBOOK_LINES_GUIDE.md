# Sportsbook Lines Integration Guide

> **Last Updated:** February 3, 2026  
> **Author:** PropAI Development Team  
> **Purpose:** Complete documentation for fetching and storing real sportsbook betting lines

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Feature Matters](#why-this-feature-matters)
3. [The Odds API Integration](#the-odds-api-integration)
4. [CLI Commands](#cli-commands)
5. [Database Schema](#database-schema)
6. [Player Name Matching](#player-name-matching)
7. [Model Integration](#model-integration)
8. [Web Interface](#web-interface)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Sportsbook Lines Integration feature enables PropAI to fetch **real betting lines** from licensed sportsbooks via The Odds API. This addresses the critical flaw identified in earlier models where derived lines (calculated from player averages) were inflating hit rates by 5-15%.

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| **Odds API Client** | `src/nba_props/ingest/odds_api_client.py` | Fetches lines from The Odds API |
| **Lines Scraper** | `src/nba_props/ingest/lines_scraper.py` | Framework for web scraping (fallback) |
| **CLI Commands** | `src/nba_props/cli.py` | Terminal commands for fetching/viewing lines |
| **Database Table** | `sportsbook_lines` | SQLite table storing all lines |
| **Web Interface** | `src/nba_props/web/` | GUI for viewing lines |

---

## Why This Feature Matters

### The Derived Lines Problem

Previous models calculated betting lines as player averages:
```
"Line" = Average of last 10 games (L10)
```

**Example of the problem:**
- Peyton Watson L10 average: 4.9 points
- Model "line": 4.9
- Actual DraftKings line: 6.5
- Model showed OVER edge, reality showed UNDER opportunity

### The Solution

By fetching actual sportsbook lines:
- Performance metrics become realistic
- Edge calculations are accurate
- Model V9's 68.6% hit rate is validated against real betting conditions

---

## The Odds API Integration

### API Details

| Setting | Value |
|---------|-------|
| **Base URL** | `https://api.the-odds-api.com` |
| **Sport Key** | `basketball_nba` |
| **API Key** | Stored in code (can be overridden via `ODDS_API_KEY` env var) |
| **Monthly Quota** | 500 requests (free tier) |

### Available Markets

| Market | Prop Type | Cost per Request |
|--------|-----------|------------------|
| `player_points` | PTS | 1 credit |
| `player_rebounds` | REB | 1 credit |
| `player_assists` | AST | 1 credit |

### Supported Bookmakers (in order of preference)

1. **DraftKings** (`draftkings`)
2. **FanDuel** (`fanduel`)
3. **BetMGM** (`betmgm`)
4. **Caesars** (`caesars`)
5. **PointsBet** (`pointsbet`)
6. **Bovada** (`bovada`)

### API Response Structure

```json
{
  "id": "event_id_here",
  "sport_key": "basketball_nba",
  "home_team": "Detroit Pistons",
  "away_team": "Denver Nuggets",
  "bookmakers": [
    {
      "key": "draftkings",
      "markets": [
        {
          "key": "player_points",
          "outcomes": [
            {
              "name": "Over",
              "description": "Nikola Jokic",
              "price": 1.91,
              "point": 24.5
            },
            {
              "name": "Under",
              "description": "Nikola Jokic",
              "price": 1.91,
              "point": 24.5
            }
          ]
        }
      ]
    }
  ]
}
```

---

## CLI Commands

### `fetch-lines-api`

Fetch player prop lines from The Odds API and store in database.

```bash
# Basic usage (fetches PTS and REB for today)
python3 run_cli.py fetch-lines-api

# Specific date
python3 run_cli.py fetch-lines-api --date 2026-02-03

# Specific bookmaker
python3 run_cli.py fetch-lines-api --book draftkings

# All prop types
python3 run_cli.py fetch-lines-api --pts --reb --ast

# Dry run (preview without storing)
python3 run_cli.py fetch-lines-api --dry-run --verbose

# Full verbose mode
python3 run_cli.py fetch-lines-api --verbose
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--date` | As-of date (YYYY-MM-DD) | Today |
| `--book` | Specific bookmaker | All (deduplicated) |
| `--pts` | Fetch points props | Default ON |
| `--reb` | Fetch rebounds props | Default ON |
| `--ast` | Fetch assists props | Default OFF |
| `--dry-run` | Preview without storing | OFF |
| `--verbose` / `-v` | Detailed output | OFF |

### `api-status`

Check The Odds API quota usage.

```bash
python3 run_cli.py api-status
```

**Example output:**
```
The Odds API Status:
  Requests used: 66
  Requests remaining: 434
  Usage: 13.2%
```

### `list-lines`

View stored sportsbook lines.

```bash
# All recent lines
python3 run_cli.py list-lines

# Filter by date
python3 run_cli.py list-lines --date 2026-02-03
```

**Example output:**
```
2026-02-03  Shai Gilgeous-Alexander  PTS 32.5  -120 (draftkings)
2026-02-03  Luka Dončić              PTS 30.5  -114 (draftkings)
2026-02-03  Nikola Jokić             REB 10.5  -110 (draftkings)
```

### `compare-lines`

Compare lines from different bookmakers for the same players.

```bash
python3 run_cli.py compare-lines --date 2026-02-03
```

**Example output:**
```
Line Comparison for 2026-02-03
======================================================================
Players with multiple sources: 249

Aaron Wiggins             PTS:
    draftkings      13.5
    fanduel         12.5
    → Spread: 1.0
```

---

## Database Schema

### `sportsbook_lines` Table

```sql
CREATE TABLE IF NOT EXISTS sportsbook_lines (
  id INTEGER PRIMARY KEY,
  as_of_date TEXT NOT NULL,      -- Date the line is for (YYYY-MM-DD)
  game_id INTEGER,               -- Optional FK to games table
  team_id INTEGER,               -- Optional FK to teams table
  player_id INTEGER,             -- FK to players table
  prop_type TEXT NOT NULL,       -- PTS, REB, AST
  line REAL NOT NULL,            -- The betting line (e.g., 24.5)
  odds_american INTEGER,         -- American odds (e.g., -110)
  book TEXT,                     -- Sportsbook name
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (game_id) REFERENCES games(id),
  FOREIGN KEY (team_id) REFERENCES teams(id),
  FOREIGN KEY (player_id) REFERENCES players(id)
);
```

### Example Data

| as_of_date | player_id | prop_type | line | odds_american | book |
|------------|-----------|-----------|------|---------------|------|
| 2026-02-03 | 362 | PTS | 32.5 | -120 | draftkings |
| 2026-02-03 | 362 | REB | 4.5 | -125 | draftkings |
| 2026-02-03 | 389 | PTS | 30.5 | -114 | draftkings |

---

## Player Name Matching

### The Challenge

API player names may differ from database names:
- API: "Nikola Jokic"
- Database: "Nikola Jokić" (with diacritics)

### Matching Algorithm

1. **Exact match** by player_id (if known)
2. **Case-insensitive** exact match
3. **Normalized match** (removes diacritics, suffixes)
4. **Partial match** (last name + first initial)

### Normalization Function

```python
def normalize_name_for_matching(name: str) -> str:
    """
    Handles:
    - Different capitalization
    - Jr./Sr./III/II suffixes
    - Accent marks (diacritics)
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
    
    return ' '.join(name.split())
```

---

## Model Integration

### Model V16 Integration (Current Production)

Model V16 represents the definitive solution to the "Derived Line Fallacy" problem through
its **hybrid line handling** approach:

```python
# In model_v16_shared.py

def get_line_info(conn, player_id: int, player_name: str, prop_type: str, 
                  game_date: str, l10_avg: float) -> LineInfo:
    """
    Hybrid line strategy:
    1. Try to fetch actual sportsbook line (high confidence)
    2. Fall back to derived line (L10 * 1.05) with stricter thresholds
    """
    sportsbook_line = _get_sportsbook_line(conn, player_id, player_name, prop_type, game_date)
    
    if sportsbook_line is not None:
        return LineInfo(
            line=sportsbook_line,
            source=LineSource.SPORTSBOOK,
            confidence=1.0
        )
    else:
        # Derived line with conservative 5% adjustment
        derived_line = l10_avg * 1.05
        return LineInfo(
            line=derived_line,
            source=LineSource.DERIVED,
            confidence=0.7  # Lower confidence requires higher edge
        )
```

### V16 Edge Thresholds (Key Innovation)

The breakthrough in V16 was applying **different edge requirements** based on line source:

| Line Source | Min Edge | Rationale |
|-------------|----------|-----------|
| Sportsbook | 6.0% | High confidence in line accuracy |
| Derived | 10.0% | Lower confidence, need more buffer |
| Premium patterns | 15.0% | Only the best opportunities |

### V16 Configuration

```python
@dataclass
class ModelConfigV16General:
    # Line handling
    use_sportsbook_lines: bool = True
    derived_line_adjustment: float = 1.05  # Inflate L10 by 5%
    
    # Edge thresholds (the key to solving derived line fallacy)
    min_edge_sportsbook: float = 6.0   # Lower threshold for real lines
    min_edge_derived: float = 10.0     # Higher threshold for derived
    premium_edge_required: float = 15.0  # For "premium" confidence
```

### V16 Backtest Results with Hybrid Lines

```
OVERALL: 72.4% (92/127 picks)
- Sportsbook lines: ~75% of picks, 74% hit rate
- Derived lines: ~25% of picks, 68% hit rate (still profitable due to higher edge req)
```

---

### Legacy: Model V9 Integration

Model V9 was the first model designed to use sportsbook lines:

```python
# In model_v9.py

def _get_sportsbook_line(conn, player_id, player_name, prop_type, game_date):
    """
    Fetch actual sportsbook line for a player/prop/date.
    
    Priority:
    1. Match by player_id
    2. Fall back to fuzzy name matching
    """
    # Try by player_id first
    if player_id:
        row = conn.execute(
            """
            SELECT line FROM sportsbook_lines
            WHERE player_id = ? AND prop_type = ? AND as_of_date = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (player_id, prop_type.upper(), game_date)
        ).fetchone()
        
        if row:
            return row["line"]
    
    # Fuzzy match fallback...
```

### Configuration

```python
@dataclass
class ModelConfigV9:
    use_sportsbook_lines: bool = True  # Prefer actual betting lines
    line_adjustment_factor: float = 1.05  # Derived lines typically 5% below actual
    min_edge_vs_actual_line: float = 5.0  # Need 5%+ edge vs ACTUAL line
```

### Line Source Tracking

Model V9 tracks where each line comes from:

```python
class PickV9:
    line_source: str  # "sportsbook" or "derived"
    line: float
    sportsbook_line: Optional[float]  # Actual if available
```

---

## Web Interface

### Lines Page

The web interface includes a dedicated **Lines** page that displays:

1. **All stored sportsbook lines** organized by date
2. **Filters** for prop type (PTS/REB/AST)
3. **Comparison view** when multiple books available
4. **Quick stats** showing coverage

### Accessing the Lines Page

1. Start the web GUI:
   ```bash
   python3 run_cli.py gui
   ```

2. Navigate to `http://localhost:5050/lines`

### Features

- **Date picker** to select which date's lines to view
- **Book filter** to show lines from specific sportsbook
- **Prop type tabs** (Points, Rebounds, Assists)
- **Search** to find specific players
- **Export** functionality for analysis

---

## Best Practices

### API Quota Management

| Action | Quota Cost |
|--------|------------|
| Fetch events (games list) | FREE |
| Fetch 1 market for 1 game | 1 credit |
| Fetch PTS+REB for 10 games | 20 credits |

**Recommendations:**
- Fetch once daily (morning before games)
- Default to PTS and REB only (AST less reliable)
- Use `--dry-run` to preview before fetching
- Check quota with `api-status` before large fetches

### Data Freshness

- Lines change throughout the day
- Fetch lines close to game time for accuracy
- Consider storing timestamp for staleness checks

### Deduplication Strategy

When fetching from all books:
1. Group by (player, prop_type)
2. Prefer lines from books in order: DK → FD → BetMGM → ...
3. Store only one line per player/prop

---

## Troubleshooting

### Common Issues

#### "No NBA events found for date"

The Odds API only returns upcoming games. Past dates won't have events.

```bash
# This works (upcoming)
python3 run_cli.py fetch-lines-api

# This may fail (past date)
python3 run_cli.py fetch-lines-api --date 2025-01-01
```

#### "Rate limit exceeded"

You've hit the API quota. Wait until next month or upgrade plan.

```bash
# Check remaining quota
python3 run_cli.py api-status
```

#### "Player not found in database"

The API player name didn't match any database player. The system will auto-create new players.

#### Lines not showing in Model V9

Ensure:
1. Lines exist for the correct date
2. `config.use_sportsbook_lines = True`
3. Player names match (check with `list-lines`)

### Debug Mode

For detailed debugging:

```python
# In odds_api_client.py
response = fetch_player_props_for_event(event_id, markets, api_key)
print(f"Response: {response}")
print(f"Props: {response.data}")
```

---

## Appendix: Code Examples

### Fetching Lines Programmatically

```python
from src.nba_props.ingest.odds_api_client import (
    fetch_nba_events,
    fetch_player_props_for_event,
    store_player_props,
)
from src.nba_props.db import Db, init_db
from src.nba_props.paths import get_paths

# Initialize database
paths = get_paths()
init_db(paths.db_path)
db = Db(path=paths.db_path)

# Fetch events
events_response = fetch_nba_events()
events = events_response.data

# Fetch props for first game
props_response = fetch_player_props_for_event(
    event_id=events[0].id,
    markets=["player_points", "player_rebounds"],
)
props = props_response.data

# Store to database
with db.connect() as conn:
    stored = store_player_props(conn, props, as_of_date="2026-02-03")
    conn.commit()
    print(f"Stored {stored} props")
```

### Querying Lines

```python
from src.nba_props.db import Db
from src.nba_props.paths import get_paths

db = Db(path=get_paths().db_path)

with db.connect() as conn:
    # Get all PTS lines for a date
    rows = conn.execute("""
        SELECT p.name, sl.line, sl.book
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = '2026-02-03'
          AND sl.prop_type = 'PTS'
        ORDER BY sl.line DESC
    """).fetchall()
    
    for row in rows:
        print(f"{row['name']}: {row['line']} ({row['book']})")
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | 2026-02-03 | Added Model V16 integration (hybrid line handling) |
| 1.0 | 2026-02-03 | Initial implementation with The Odds API |

---

## Related Documentation

- **MODEL_V16.md** - Complete V16 model documentation (hybrid line solution)
- **MODEL_VERSION_TRACKING.md** - All model versions and their line handling approaches
- **1_MODEL_SUMMARY.md** - Quick reference for all models

---

*This document should be updated when new features are added or API changes occur.*
