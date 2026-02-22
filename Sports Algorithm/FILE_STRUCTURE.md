# NBA Props Predictor - File Structure Documentation

## Overview

This document provides a comprehensive overview of the NBA Props Predictor codebase structure, explaining the purpose of each module and how they interact.

---

## Directory Structure

```
Sports Algorithm/
├── run_cli.py                  # CLI entry point for running the application
├── pyproject.toml              # Python project configuration and dependencies
├── README.md                   # Project overview and setup instructions
├── REPORT.md                   # Analysis reports and findings
├── Idea.txt                    # Project ideas and notes
├── FILE_STRUCTURE.md           # This file - codebase documentation
│
├── data/                       # Data storage directory
│   ├── db/                     # SQLite database files
│   │   └── nba_props.sqlite3   # Main database with games, players, stats
│   ├── exports/                # Exported reports and data
│   └── raw/                    # Raw data files
│       ├── boxscores/          # Game box score files by date
│       │   └── 2025-26/        # Season folder with date subfolders
│       ├── lines/              # Sportsbook lines data
│       │   └── 2025-26/
│       └── metadata/           # Game metadata
│           └── 2025-26/
│
├── Sample Data/                # Example data files for testing
│   ├── *.txt                   # Sample box scores and stats
│
└── src/                        # Main source code
    └── nba_props/              # Core package
        ├── __init__.py         # Package initialization
        ├── __main__.py         # Module entry point (python -m nba_props)
        ├── cli.py              # Command-line interface commands
        ├── db.py               # Database connection and management
        ├── paths.py            # File path configuration
        ├── standings.py        # Conference standings calculations
        ├── team_aliases.py     # Team name/abbreviation mappings
        ├── util.py             # General utility functions
        ├── validation.py       # Data validation helpers
        │
        ├── engine/             # Projection and analysis engine
        │   ├── __init__.py     # Engine module exports
        │   ├── projector.py    # Core player projection logic
        │   ├── game_context.py # Game context (B2B status, team defense ratings)
        │   ├── edge_calculator.py  # Prop bet edge and probability calculations
        │   ├── matchup_advisor.py  # Advanced defense metrics & ADVISOR reports (MAIN OUTPUT)
        │   ├── under_picks_analyzer.py # Dedicated UNDER picks model with defense factors
        │   ├── archetypes.py   # Player archetype definitions
        │   ├── archetype_db.py # Database-backed archetype storage
        │   ├── roster.py       # Player roster and profiles
        │   ├── usage_redistribution.py  # Usage rate calculations
        │   ├── alerts.py       # Edge alert scanning system
        │   └── backtesting.py  # Historical accuracy testing
        │
        ├── ingest/             # Data ingestion modules
        │   ├── __init__.py
        │   ├── boxscore_ingest.py   # Import box scores to database
        │   ├── boxscore_parser.py   # Parse box score text/files
        │   ├── lines_parser.py      # Parse sportsbook lines
        │   ├── matchups_parser.py   # Parse matchup information
        │   ├── paste.py             # Handle pasted text input
        │   ├── salary_parser.py     # Parse player salary data
        │   ├── team_stats_ingest.py # Import team statistics
        │   └── team_stats_parser.py # Parse team stats files
        │
        └── web/                # Web application
            ├── __init__.py
            ├── app.py          # Flask application and API routes
            ├── static/         # Static assets (CSS, JS, images)
            └── templates/      # Jinja2 HTML templates
```

---

## Module Details

### Core Package (`src/nba_props/`)

#### `__init__.py`
Package initialization. Exposes key modules and version information.

#### `__main__.py`
Entry point for running the package as a module:
```bash
python -m nba_props gui  # Start web interface
python -m nba_props cli  # Use command line
```

#### `cli.py`
Command-line interface implementation using Click. Provides commands for:
- Ingesting box scores
- Running projections
- Managing the database
- Starting the web server

#### `db.py`
Database management:
- `Db` class - Database connection wrapper
- `init_db()` - Initialize database schema
- Schema includes: `games`, `teams`, `players`, `boxscore_player`, `boxscore_team_totals`, `sportsbook_lines`, `player_archetypes`

#### `paths.py`
Centralized path configuration:
- `Paths` dataclass with all file/folder paths
- `get_paths()` - Get configured paths instance

#### `standings.py`
Conference standings and team statistics:
- `compute_conference_standings()` - Calculate current standings
- `compute_player_averages_for_team()` - Get player stats by team
- Conference assignments (East/West)

#### `team_aliases.py`
Team name normalization:
- `normalize_team_abbrev()` - Standardize abbreviations
- `abbrev_from_team_name()` - Full name to abbreviation
- `team_name_from_abbrev()` - Abbreviation to full name
- Handles all team name variations

#### `util.py`
General utilities:
- Date parsing and formatting
- Statistical calculations
- Helper functions

#### `validation.py`
Data validation:
- Schema validation
- Input sanitization
- Error checking helpers

---

### Engine Module (`src/nba_props/engine/`)

The engine contains the core prediction and analysis logic.

#### `projector.py`
**Core projection calculations:**

Classes:
- `PlayerProjection` - Projected stats for a player
- `ProjectionConfig` - Configuration for projections

Functions:
- `project_player_stats()` - Generate individual player projection
- `project_team_players()` - Project all players on a team

The projector uses:
- Historical averages (weighted toward recent games)
- Minutes projections
- Team context (pace, style)
- Injury/absence adjustments

#### `game_context.py`
**Game Context & Matchup Adjustments:**

Provides contextual information about games affecting player performance.

Classes:
- `BackToBackStatus` - Track rest days and B2B games
- `MatchupRecommendation` - Betting recommendations
- `TeamDefenseRating` - Defensive efficiency metrics

Functions:
- `get_back_to_back_status()` - Check team rest situation
- `get_team_defense_rating()` - Overall defensive metrics
- `apply_matchup_adjustments()` - Adjust projections based on opponent
- `get_position_defense_rating()` - Defense vs specific positions
- `get_player_vs_team_history()` - Historical performance vs opponent

#### `matchup_advisor.py`
**Advanced Defense Analysis and Matchup Reporting (ADVISOR LAYER)**

This is the most sophisticated analysis module, providing actionable betting advice.

Data Classes:
- `PositionDefenseProfile` - How a team defends each position (G/F/C)
- `ArchetypeDefenseProfile` - Defense vs player archetypes
- `PlayerVsTeamProfile` - Individual player history vs opponent
- `PlayerTrend` - Hot/cold streak tracking
- `MatchupEdge` - Calculated edge for a specific prop
- `ComprehensiveMatchupReport` - **Full matchup analysis object**

Key Functions:
- `get_position_defense_profile()` - Analyze defense by position
- `get_all_position_defense_profiles()` - All position profiles for a team
- `rank_position_defense_profiles()` - Rank teams by positional defense
- `get_player_vs_team_profile()` - Player historical performance
- `get_player_trend()` - Recent performance trend analysis
- `calculate_matchup_edge()` - Calculate edge with all factors
- `get_team_defense_summary()` - Overview of team defense
- `generate_comprehensive_matchup_report()` - **MAIN ADVISOR FUNCTION**

The `generate_comprehensive_matchup_report()` function provides:
- Best OVER plays (sorted by confidence)
- Best UNDER plays (sorted by confidence) - *Currently delegated to specialized logic*
- Players to AVOID betting on
- Key matchup storylines
- Defense-by-position analysis
- Player trends and historical context

#### `edge_calculator.py`
**Prop Bet Edge & Probability Calculations:**

Uses statistical probability models to calculate betting edges.

Classes:
- `PropEdge` - Calculated edge for a prop bet with win probability

Functions:
- `calculate_prop_edge()` - Edge calculation using normal distribution CDF
- `rank_prop_opportunities()` - Sort props by calculated edge
- `generate_prop_report()` - Basic matchup projection report

The math: Uses scipy.stats.norm to calculate probability of player exceeding line
based on projected value and historical variance (standard deviation).

#### `under_picks_analyzer.py`
**Dedicated Under Picks Analyzer:**

Specialized model focused exclusively on identifying high-confidence UNDER plays.
Separated from the main projection engine to handle defense-specific factors.

Classes:
- `UnderCandidate` - Candidate for an UNDER bet with confidence scoring

Factors Analyzed:
- Elite defense at position (from Hashtag Basketball data)
- Back-to-back fatigue impact
- Cold streaks / recent performance decline
- Injury rust (returning from absence)
- Role reduction indicators

Key Functions:
- Analysis logic to identify players likely to underperform their line

#### `archetypes.py`
**Player Archetype Definitions:**

Contains:
- Offensive archetypes (SCORING, PLAYMAKING, SHOOTING, etc.)
- Defensive roles (ELITE_WING, RIM_PROTECTOR, etc.)
- `KNOWN_ARCHETYPES` - Dictionary of known player archetypes
- `get_player_archetype()` - Retrieve archetype for player
- `classify_player_by_stats()` - Auto-classify based on stats

#### `archetype_db.py`
**Database-Backed Archetype Storage:**

Functions for persisting archetypes:
- `get_player_archetype_db()` - Fetch from database
- `get_all_archetypes_db()` - Get all stored archetypes
- `update_player_archetype()` - Save archetype to DB
- `delete_player_archetype()` - Remove archetype
- `seed_archetypes_from_defaults()` - Initialize from defaults
- `get_elite_defenders_db()` - Query elite defenders
- `get_similar_players_db()` - Find similar player profiles
- `should_avoid_betting_over_db()` - Check for elite defender matchup

#### `roster.py`
**Player Roster and Profiles:**

Classes:
- `PlayerProfile` - Complete player profile with archetype
- `PlayerTier` - Tier classification (MVP, Star, Starter, etc.)
- `OffensiveArchetype` - Offensive style enum
- `DefensiveRole` - Defensive role enum

Data:
- `PLAYER_DATABASE` - Static database of known player profiles

Functions:
- `get_roster_for_team()` - Get all players for a team
- `get_player_profile()` - Get individual player profile
- `should_avoid_betting_over()` - Elite defender warning
- `find_similar_players()` - Find comparable players

#### `usage_redistribution.py`
**Usage Rate Calculations:**

Classes:
- `PlayerUsageProfile` - Usage share and tendencies
- `UsageRedistributionResult` - Impact of player absence

Functions:
- `get_team_usage_profiles()` - Usage for all team players
- `calculate_usage_redistribution()` - Impact when player out
- `get_historical_impact()` - Historical data on absences

#### `alerts.py`
**Alert and Notification System:**

Functions for generating alerts on:
- Line movement
- Injury news
- Edge opportunities
- Matchup advantages

#### `backtesting.py`
**Historical Accuracy Testing and Model Validation:**

Classes:
- `PropResult` - Result of a single prop bet evaluation
- `BacktestResult` - Aggregate results from a backtest run

Functions:
- `get_player_actual_stats()` - Get actual stats for a player on a specific date
- `calculate_profit_from_odds()` - Calculate profit/loss from bet result
- `run_backtest()` - Run backtest comparing lines to actual outcomes
- `compare_projection_accuracy()` - Compare projection accuracy for a player
- `analyze_projection_bias()` - Analyze systematic biases in projections

The backtesting system tracks:
- Hit rates by prop type (PTS, REB, AST)
- Hit rates by confidence level (HIGH, MEDIUM, LOW)
- Hit rates by direction (OVER, UNDER)
- Calibration bins (predicted probability vs actual outcome)
- Theoretical ROI calculations

---

### Ingest Module (`src/nba_props/ingest/`)

Handles importing data from various sources.

#### `boxscore_ingest.py`
Import parsed box scores into the database.
- `ingest_boxscore_file()` - Process a single box score file

#### `boxscore_parser.py`
Parse box score text from various formats:
- ESPN format
- NBA.com format
- Custom paste format
- Handles different column layouts

#### `lines_parser.py`
Parse sportsbook prop lines:
- Extract player, prop type, line value
- Handle various odds formats (American, decimal)

#### `matchups_parser.py`
Parse matchup information:
- `parse_matchups_text()` - Parse full matchup slate
- `parse_simple_matchup()` - Parse single game matchup

#### `paste.py`
Handle pasted text input:
- `save_pasted_boxscore_text()` - Save pasted content to file
- Auto-detect team names
- Format standardization

#### `salary_parser.py`
Parse player salary information:
- DFS salaries
- Contract values

#### `defense_position_parser.py`
**Defense vs Position Data Parser (NEW):**

Parses raw data from Hashtag Basketball's "NBA Defense vs Position" page.

Classes:
- `DefenseVsPositionRow` - Single row of defense data for team/position combo
- `DefenseVsPositionParseResult` - Complete parse result with metadata

Key Functions:
- `parse_defense_vs_position_text()` - Parse raw pasted text into structured data
- `save_defense_vs_position_to_db()` - Store parsed data in database
- `get_defense_vs_position()` - Retrieve defense data for team/position
- `get_all_defense_vs_position_for_team()` - All 5 positions for a team
- `calculate_defense_factor()` - **Calculate how team's defense affects player at position**
- `get_best_defenses_at_position()` - Teams with strongest defense vs position
- `get_worst_defenses_at_position()` - Teams with weakest defense vs position

Defense Factor Calculation:
```python
factor = stat_allowed_by_opponent / league_average_for_position
# factor < 1.0 = strong defense (good for UNDERs)
# factor > 1.0 = weak defense (good for OVERs)
```

Rating Classifications:
- **Elite** (Rank 1-5): Strongest defense at position
- **Good** (Rank 6-10): Above average defense
- **Average** (Rank 11-20): League average defense
- **Poor** (Rank 21-25): Below average defense
- **Terrible** (Rank 26-30): Weakest defense at position

Team Abbreviation Normalization:
- Handles Hashtag Basketball abbreviations (GS, NY, PHO, SA, NO)
- Converts to standard NBA abbreviations (GSW, NYK, PHX, SAS, NOP)

#### `player_drtg_parser.py`
**Player Defensive Rating (DRTG) Parser (NEW):**

Parses raw data from StatMuse's player defensive rating pages.

Data Class:
- `PlayerDRTGRow` - Single player's defensive rating data including:
  - `player_name` - Full player name
  - `team_abbrev` - Team abbreviation
  - `drtg` - Defensive Rating (lower is better)
  - `games_played` - Games played
  - `minutes_per_game` - Average minutes
  - `ppg`, `rpg`, `apg`, `spg`, `bpg` - Per-game stats
  - `plus_minus` - Plus/minus rating

Key Functions:
- `parse_player_drtg_text()` - Parse raw pasted text from StatMuse
- `save_player_drtg_to_db()` - Store parsed data in player_drtg table
- `get_team_drtg_rankings()` - Get all players' DRTG for a team
- `get_league_drtg_rankings()` - Get league-wide DRTG rankings
- `get_player_drtg()` - Get specific player's DRTG
- `get_teams_needing_drtg_update()` - Teams with stale/missing DRTG data

DRTG Rating Classifications:
- **Elite** (DRTG < 100): Elite defender
- **Good** (100 ≤ DRTG < 105): Above average defender
- **Average** (105 ≤ DRTG < 115): League average
- **Poor** (DRTG ≥ 115): Below average defender

Use Cases:
- Identify elite individual defenders
- Adjust projections based on defensive impact
- Supplement team-level defense data with player granularity
- Track defensive improvement/decline over season

#### `team_stats_ingest.py` / `team_stats_parser.py`
Import and parse team-level statistics.

---

### Web Module (`src/nba_props/web/`)

Flask-based web application.

#### `app.py`
Main Flask application with:

**Pages (Routes):**
- `/` - Main dashboard
- `/games` - Games list
- `/paste` - Paste box score
- `/projections` - Projections page
- `/teams` - Teams overview
- `/team/<abbrev>` - Team detail
- `/players` - Players and matchups
- `/matchups` - Today's matchups
- `/data` - Data management

**API Endpoints:**
- `GET /api/stats` - Database statistics
- `GET /api/games` - List games
- `GET /api/game/<id>` - Game detail
- `GET /api/standings` - Conference standings
- `GET /api/team/<abbrev>` - Team info
- `GET /api/team/<abbrev>/dashboard` - Team dashboard
- `GET /api/team/<abbrev>/defense-profile` - Defense analysis
- `GET /api/player/<id>/trend` - Player trend
- `GET /api/player/<name>/vs-team/<opp>` - Player vs team history
- `POST /api/projections` - Generate projections
- `POST /api/matchup-analysis` - **Comprehensive matchup report**
- `POST /api/ingest/boxscore` - Import box score
- `POST /api/ingest/lines` - Import lines

**Player DRTG API Endpoints (NEW):**
- `POST /api/ingest/player-drtg` - Import player DRTG data from StatMuse
- `GET /api/player-drtg/<team>` - Get DRTG rankings for a team
- `GET /api/player-drtg/league` - Get league-wide DRTG rankings
- `GET /api/player-drtg/status` - Get data freshness for all teams
- `GET /api/player/<name>/drtg` - Get DRTG for specific player

**Backtesting/Model Testing API Endpoints:**
- `POST /api/backtesting/generate-picks` - Generate or load cached picks for a date
  - Returns picks with results if already graded
  - Generates new picks from scheduled matchups if not cached
  - Use `force: true` to regenerate picks
- `POST /api/backtesting/compare-results` - Grade picks against actual box scores
  - Compares predictions to actual outcomes
  - Stores results in `model_pick_results` table
  - Updates daily performance statistics
- `GET /api/backtesting/performance` - Get overall model performance statistics
  - Aggregate hit rates across all tracked days
  - Daily breakdown of performance metrics
- `GET /api/backtesting/picks-history` - Get historical picks with results
  - Supports date range filtering
  - Returns statistics summary

#### `templates/`
Jinja2 HTML templates for each page.

#### `static/`
Static assets:
- CSS stylesheets
- JavaScript files
- Images and icons

---

## Data Flow

```
1. DATA INGESTION
   Raw Files → Parsers → Database
   - Box scores → boxscore_parser → boxscore_player table
   - Lines → lines_parser → sportsbook_lines table
   - Defense data → defense_position_parser → team_defense_vs_position table
   - Player DRTG → player_drtg_parser → player_drtg table (NEW)
   
2. PROJECTION GENERATION
   Database → Projector → Base Projections
   - Weighted averages (L5: 35%, L20: 40%, Season: 25%)
   - Per-minute production rates
   
3. MATCHUP ADJUSTMENTS
   Base Projections → Game Context Module → Adjusted Projections
   - Archetype-based adjustments
   - Elite defender warnings
   - Defense vs Position factors (for OVERs: boost if factor > 1.02)
   - Player DRTG analysis (individual defensive impact)
   
4. UNDER ANALYSIS (Separate Model)
   Database → Under Picks Analyzer → UNDER Recommendations
   - Defense vs Position factors (factor < 1.0 = strong defense = good for UNDERs)
   - B2B fatigue, cold streaks, injury rust
   - Historical vs opponent performance
   - Player DRTG data (identify elite defenders)
   
5. EDGE CALCULATION
   Adjusted Projections → Edge Calculator → PropEdge probabilities
   
6. ADVISOR REPORT GENERATION
   All Analyses → Matchup Advisor → ComprehensiveMatchupReport → UI/API
```

---

## Key Algorithms

### Projection Formula
```
Projected Stats = Weighted Average × Opponent Adjustment × Defense vs Position × Rest Adjustment × Trend Adjustment

Where:
- Weighted Average: L5 (35%) + L20 (40%) + Season (25%) - Recent games weighted higher
- Opponent Adjustment: Based on archetype matchups and elite defenders
- Defense vs Position: Position-specific factor from Hashtag Basketball data
  * For OVERs: factor > 1.02 triggers boost (capped at 15%)
  * For UNDERs: factor < 1.0 is favorable (strong defense)
- Rest Adjustment: B2B penalty (~6%) or rest bonus (~3%)
- Trend Adjustment: Hot/cold streak factor
```

### Edge Calculation
```
Confidence Score = Base(50) 
    + Position Defense Match(±15)
    + Historical Performance(+10)
    + Trend Alignment(+12)
    - Warnings Count(×8)
```

### Defense vs Position Factor
```
Factor = Stat Allowed by Opponent / League Average for Position

Example (PG Points):
- Boston allows 21.3 PTS to PGs, League Avg = 24.3
- Factor = 21.3 / 24.3 = 0.878 (Strong defense)

Rating Classifications (Position-Specific Rank 1-30):
- Elite (Rank 1-5): factor ≤ 0.92 — Best targets for UNDERs
- Good (Rank 6-10): factor ≤ 0.97
- Average (Rank 11-20): factor ≈ 1.00
- Poor (Rank 21-25): factor ≥ 1.03
- Terrible (Rank 26-30): factor ≥ 1.08 — Best targets for OVERs
```

---

## Usage Examples

### Generate Matchup Report (CLI)
```bash
python run_cli.py projections --away LAL --home BOS --date 2026-01-03
```

### Generate Matchup Report (API)
```python
from nba_props.engine.matchup_advisor import generate_comprehensive_matchup_report
from nba_props.db import Db

db = Db()
with db.connect() as conn:
    report = generate_comprehensive_matchup_report(
        conn=conn,
        away_abbrev="LAL",
        home_abbrev="BOS",
        game_date="2026-01-03",
        spread=-3.5,
        over_under=220.5
    )
    
    # Access best plays
    for play in report.best_over_plays[:5]:
        print(f"OVER: {play.player_name} {play.prop_type} - Confidence: {play.confidence_tier}")
```

### Start Web Interface
```bash
python run_cli.py gui
# Opens at http://localhost:5000
```

---

## Architecture Principles

1. **Separation of Concerns**
   - Data ingestion separate from analysis
   - Projection logic separate from presentation
   - Database access abstracted

2. **Layered Analysis**
   - Raw stats → Projections → Adjustments → Recommendations
   - Each layer adds context and refinement

3. **Configurable Behavior**
   - `ProjectionConfig` controls projection parameters
   - Database vs static data sources
   - Adjustable thresholds

4. **Advisor Pattern**
   - Don't just show numbers
   - Provide actionable recommendations
   - Explain reasoning (reasons/warnings)

---

## Database Schema (Key Tables)

```sql
-- Core tables
games(id, game_date, season, team1_id, team2_id)
teams(id, name)
players(id, name)

-- Box score data
boxscore_player(game_id, team_id, player_id, pos, minutes, pts, reb, ast, ...)
boxscore_team_totals(game_id, team_id, pts, reb, ast, ...)

-- Betting data
sportsbook_lines(id, as_of_date, game_id, player_id, prop_type, line, odds_american, book)

-- Scheduled/upcoming games
scheduled_games(id, game_date, away_team_id, home_team_id, spread, over_under, status)

-- Defense vs Position (NEW)
team_defense_vs_position(
    id, season, position,        -- Position: PG, SG, SF, PF, C
    team_abbrev, overall_rank,   -- Overall rank 1-150 across all positions
    pts_allowed, pts_rank,       -- Stats allowed + cross-position rank
    reb_allowed, reb_rank,
    ast_allowed, ast_rank,
    fg_pct_allowed, fg_pct_rank,
    ft_pct_allowed, ft_pct_rank,
    tpm_allowed, tpm_rank,       -- 3-pointers made
    stl_allowed, stl_rank,
    blk_allowed, blk_rank,
    to_allowed, to_rank,
    as_of_date                   -- Data freshness tracking
)

-- Player Defensive Rating (NEW)
player_drtg(
    id, season,
    player_name, team_abbrev,
    drtg,                        -- Defensive Rating (lower is better)
    games_played, minutes_per_game,
    ppg, rpg, apg, spg, bpg,     -- Per-game averages
    plus_minus,                  -- Plus/minus rating
    as_of_date                   -- Data freshness tracking
)

-- Model Testing/Backtesting
model_picks(id, pick_date, player_id, player_name, team_abbrev, opponent_abbrev,
            prop_type, direction, projection, confidence, confidence_score, reasons, rank)
model_pick_results(id, pick_id, actual_value, hit, margin, graded_at)
model_performance_daily(id, performance_date, total_picks, hits, misses, hit_rate, grade, ...)

-- Archetypes
player_archetypes(id, player_name, position, tier, primary_offensive, ...)
```

---

## Contributing

When adding new features:

1. **Engine logic** → Add to appropriate engine module
2. **Data parsing** → Add to ingest module
3. **API endpoints** → Add to `web/app.py`
4. **UI components** → Add to templates and static files

Always update this documentation when adding new modules or significant features.

---

*Last Updated: January 7, 2026*