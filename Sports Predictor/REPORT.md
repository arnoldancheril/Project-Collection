# NBA Player Props Finder — Project Report (Local)

## 1) Objective (What we are building)

You will run this locally and **manually provide data** (no scraping). The system should:

- Ingest full-game **box scores** you provide (your `.txt` files).
- Maintain an updatable local dataset of **games, players, teams, and derived stats**.
- Ingest **sportsbook lines** (points / rebounds / assists over-under, plus spread).
- Produce a ranked list of the **best prop opportunities** (edge vs line), focusing only on the **top 7 players per team**.
- Optionally output a **game winner / spread lean**.

Key challenges we must model:

- Player roles/archetypes within positions (e.g., “stretch 5” vs “rim runner”).
- Injuries / players out / minutes limitations.
- Back-to-backs (fatigue) and schedule effects.
- Opponent matchups: how specific player types perform vs specific defenses/styles.
- Avoid being misled by “position average” alone (stars differ from role players).

---

## 2) Inputs You Provide (Source-of-truth data)

### 2.1 Box score files (per game)

We support **two formats** found in your `Sample Data/`:

#### Format A — Markdown tables + explicit CSV section (preferred)

Your `01-01-26 Rockets vs Nets.txt` contains a `CSV Version of File:` section we can parse reliably.

Properties:

- Two teams with player rows
- Player row statuses like `Played`, `DNP - Coach's Decision`, `DND - Injury/Illness`
- A `TOTALS` row per team (team totals)
- An `Inactive Players` section at the end

#### Format B — Tabbed “PLAYER MIN FGM ...” lines (fallback)

Your `12-31-25 Warriors vs Hornets.txt` contains:

- Team name header
- A `PLAYER    MIN    FGM ... +/-` tabbed header
- Repeating blocks like:
  - `undefined Headshot` (noise)
  - player name
  - position line OR a `DNP - ...` line
  - stats line (tab-separated)
- A `TOTALS` row
- `Inactive Players` lines for both teams

**Important ingestion rule:** if `CSV Version of File:` exists, we will ingest from that section and ignore the markdown tables above it.

---

### 2.2 Sportsbook lines (daily)

You will provide (manually entered, or pasted into a CSV):

- Player props: **PTS**, **REB**, **AST** (and later maybe PRA, 3PM, etc.)
- Game spread (we will focus on spread ≤ 6 for “closer games”)
- Optional moneyline and totals
- Book name and timestamp (optional but helpful)

---

### 2.3 Injuries / availability (daily)

You will mark:

- Out / Questionable / Doubtful / Probable
- Minutes limits (if known)
- Late scratches (if known)

This is critical because it changes:

- Player minutes projections
- Usage distribution (who absorbs shots/assists/rebounds)
- Team quality (win probability)

---

### 2.4 Team defense / standings / roster metadata (periodic)

From your samples:

- `teamdefense25-26.txt` (team defense rating table)
- `Conference Rankings.txt` (standings)
- `NBA Salaries.txt` (salary ranks; proxy for “star-ness”)
- Team stats files (example: `Phoenix Suns Stats 2025-26.txt`)

We will store these as:

- **raw source files** (exactly what you provide, archived)
- **normalized tables** (cleaned values used by the model)

---

## 3) Data Storage Strategy (82 games x 30 teams, scalable)

### 3.1 Guiding principles

- Keep **raw inputs immutable** (audit trail, easy debugging).
- Parse into a **single local database** for fast queries (SQLite).
- Derive “features” and projections from DB tables, not from raw text files.

### 3.2 Directory layout (recommended)

```
Sports Algorithm/
  REPORT.md
  README.md
  src/
    nba_props/
      ...
  data/
    raw/
      boxscores/
        2025-26/
          2026-01-01/
            HOU_vs_BKN__source.txt
          2025-12-31/
            GSW_vs_CHA__source.txt
      lines/
        2025-26/
          2026-01-02__lines.csv
      metadata/
        2025-26/
          salaries__2026-01-01.txt
          team_defense__2026-01-01.txt
          standings__2026-01-01.txt
    db/
      nba_props.sqlite3
    exports/
      picks__2026-01-02.csv
```

### 3.3 Database tables (initial)

Minimum viable schema (we can expand later):

- **`teams`**: team_id, name
- **`players`**: player_id, name (later: canonical IDs)
- **`games`**: game_id, season, game_date, team1_id, team2_id, source_file
- **`boxscore_player`**: game_id, team_id, player_id, status, pos, minutes, pts, reb, ast, plus raw shooting/TO/etc.
- **`boxscore_team_totals`**: game_id, team_id, pts, reb, ast, etc.
- **`inactive_players`**: game_id, team_id, player_name, reason
- **`sportsbook_lines`**: as_of_date, game_id, player_id, prop_type, line, odds, book
- **`player_role_labels`**: player_id, season, base_pos, archetype, confidence, method
- **`injury_report`**: game_date, team_id, player_id (or name), status, minutes_limit, notes

Why SQLite:

- Zero setup
- Great for local analytics
- Easy to export to CSV

---

## 4) Player Roles / Archetypes (How we classify “center types”, etc.)

We will store two layers:

### 4.1 Base position (coarse)

From your box score: `G`, `F`, `C` (sometimes blank).

### 4.2 Archetype (fine)

Examples (extendable):

- **Centers**: rim runner, stretch 5, point center, two-way, post scorer
- **Guards**: primary creator, secondary creator, 3&D, off-ball shooter, slasher
- **Forwards**: wing stopper, stretch 4, point forward, rim pressure finisher

### 4.3 How we assign archetypes (practical + incremental)

Phase 1 (MVP): **rules + season-to-date rates** derived from box scores:

- 3PA per minute, AST per minute, REB per minute, BLK per minute, etc.
- Salary rank as a weak “star prior”
- Minutes share (top 7 proxy)

Phase 2: clustering within position groups (k-means / GMM), once we have enough games.

We will always keep a way for you to **override** a player label (manual truth beats automation).

---

## 5) Core Prop Model (How we project PTS/REB/AST)

We will build projections in layers, so each layer is testable:

### 5.1 Minutes projection (most important)

Minutes drives everything. We’ll estimate:

- Baseline minutes from last N games (e.g., 10)
- Adjustment for player status (out/questionable/returning)
- Adjustment for blowout risk (based on spread)
- Adjustment for back-to-back / rest days

### 5.2 Per-minute production projection

For each stat (PTS/REB/AST):

- Player baseline per-minute
- Role/archetype adjustments
- Opponent adjustments (team defense / style)
- Context adjustments (pace proxy, team strength, injuries)

### 5.3 Variance / uncertainty

We need uncertainty to rank props by **edge probability**:

- Use rolling standard deviation (last N) as a first approximation
- Later: add matchup-conditional variance

### 5.4 Prop edge scoring

For each candidate prop:

- Predicted mean \( \mu \)
- Predicted std \( \sigma \)
- Book line \( L \)
- Compute \( P(\text{Over}) \) and \( P(\text{Under}) \) (normal approx for MVP)
- Rank by expected value / probability edge

### 5.5 “Top 7 players only” rule

Define “top 7” by a stable metric:

- Primary: season average minutes (or last-10 average minutes)
- Tie-breakers: salary rank, usage proxy (FGA+FTA+AST), coach DNP frequency

We’ll enforce this rule in the recommendation layer.

---

## 6) Team Winner Prediction (Optional)

MVP approach:

- Start from team strength proxy (standings + net rating proxies if available)
- Adjust for injuries (missing top minutes/production)
- Adjust for rest/back-to-back
- Apply spread filter (we care most when spread ≤ 6)

Later we can model win probability directly once you provide enough historical results + spreads.

---

## 7) How You Will Run & Test This Locally

### 7.1 Daily workflow (intended)

1. Drop new box score `.txt` files into `data/raw/boxscores/<season>/<date>/`
2. Run ingestion to update SQLite
3. Add tomorrow’s lines to `data/raw/lines/...`
4. Enter injuries for tomorrow (file or GUI form)
5. Run “recommend props” for the slate
6. Export picks to CSV

### 7.2 Testing strategy (important for trust)

- **Parser tests**: each sample file becomes a regression test (Format A + B)
- **DB consistency checks**: totals rows, minutes parsing, duplicates
- **Projection sanity checks**: mean projections close to recent averages on neutral matchups
- **Backtesting** (once you provide historical lines):
  - Compare predicted over probability vs actual outcomes
  - Track calibration (are 60% picks hitting ~60%?)
  - Track ROI by threshold

---

## 8) GUI Plan (Local, easy data entry)

We’ll start with a simple local GUI that:

- Lets you choose a box score file and ingest it
- Shows parsed games/teams
- Lets you enter injuries and lines (simple table form)
- Runs projections and shows ranked props

Implementation options:

- **Tkinter (stdlib)**: no installs, runs anywhere, simplest MVP.
- **Streamlit**: much nicer UI, but requires installing dependencies.

We will scaffold Tkinter first so it works immediately; we can upgrade later.

---

## 9) What Other Data Would Be Useful (If you can provide it)

High value:

- **Game location** (home/away), start time, travel (optional)
- **Starting lineup** (or “who started”)
- **Team pace / possessions** (even approximate)
- **Opponent positional stats** (allowed points/reb/ast by position/archetype)
- **Sportsbook odds** (American odds) in addition to the line (for EV)
- **Closing line vs open** (optional)
- **Rest days** / schedule (we can compute if we have all game dates)

Nice-to-have:

- Player usage rate, touches, potential assists (advanced tracking) — only if you have it.

---

## 10) Development Phases (How we will build it)

### Phase 0 — Setup (now)

- Project structure
- SQLite schema
- Box score ingestion (Format A + B)
- Store raw files + parsed rows
- Basic CLI + basic GUI stub

### Phase 1 — Core analytics

- Season-to-date player and team aggregates
- “Top 7 players” selection logic
- Basic minutes model
- Basic projections for PTS/REB/AST

### Phase 2 — Matchup intelligence

- Archetype labeling (rules → clustering)
- Team defense/style adjustments
- Injury redistribution logic (who benefits when a star is out)

### Phase 3 — Backtesting & iteration

- Lines ingestion
- Prop edge scoring
- Backtest reports and calibration


---

## 11) Implementation Status (What’s been built so far)

This section documents the **current working state** of the repo as of now.

### 11.1 Repository structure created

- **`src/nba_props/`**: Python package (stdlib-only so far)
- **`data/`**:
  - **`data/raw/boxscores/<season>/<YYYY-MM-DD>/...`**: raw game inputs (files you import or paste)
  - **`data/raw/metadata/<season>/...`**: standings/defense/team stats raw files
  - **`data/raw/lines/<season>/...`**: (reserved; current paste-lines writes to DB directly)
  - **`data/db/nba_props.sqlite3`**: SQLite database (created on demand)
  - **`data/exports/`**: (reserved for later picks exports)

We also copied your sample inputs into the new `data/raw/...` structure as examples.

---

### 11.2 Database implemented (SQLite)

SQLite schema is created in `src/nba_props/db.py`. Tables currently used:

- **`teams`**: unique team names
- **`players`**: unique player names
- **`games`**: one row per ingested game:
  - stores **game_date** (YYYY-MM-DD), teams, source file path, and source hash
  - de-duplication now happens by **(game_date + matchup)** (order-insensitive)
- **`boxscore_player`**: one row per (game, team, player) with minutes + stats (PTS/REB/AST, etc.)
- **`boxscore_team_totals`**: team totals per game (when present)
- **`inactive_players`**: inactive lists per game/team
- **`sportsbook_lines`**: pasted/ingested lines per as-of date (PTS/REB/AST)
- **`injury_report`**: schema exists (UI + ingestion not built yet)

---

### 11.3 Box score ingestion implemented (3 input formats)

The ingestion pipeline is:

- raw file (or pasted text) → `parse_boxscore_text(...)` → normalized `ParsedGame`
- insert/update DB rows in `games`, `boxscore_player`, `boxscore_team_totals`, `inactive_players`

Supported formats:

1) **CSV-section format** (preferred): files containing `CSV Version of File:`  
   - We parse the CSV section for robustness.

2) **Tabbed/space-aligned “PLAYER MIN FGM ...” format** (your “undefined Headshot” style)  
   - Handles:
     - optional separate position line (`G/F/C`)
     - or missing pos line (stats line begins with `MM:SS`)
     - DNP/DND entries
     - totals rows

3) **Markdown-table format** (no CSV section)  
   - Parses the `## <Team> — Box Score` tables and the `## Inactive Players` section.

Inactive player mapping improvements:

- Inactive sections like `PHI: ...` or `* **MIA:** ...` are mapped to full team names using `src/nba_props/team_aliases.py`.

Date inference:

- If filename contains `MM-DD-YY`, we infer date from it.
- Otherwise, we infer date from parent folder named `YYYY-MM-DD` (supports canonical filenames like `HOU_vs_BKN__source.txt`).

---

### 11.4 Paste-first workflow implemented (so you don’t have to format anything)

You can now paste raw unformatted text directly and the app will:

- save the pasted text as a raw file under `data/raw/boxscores/<season>/<YYYY-MM-DD>/...`
- ingest it into SQLite

This exists in both CLI and GUI.

---

### 11.5 Sportsbook lines ingestion implemented (paste format)

We implemented a parser for your lines format:

- Section headers like:
  - `Points line:`
  - `Player Rebounds:`
  - `Player Assists:`
- Rows like:
  - `CJ McCollum: 18.5 -125`

The parsed lines are stored in `sportsbook_lines` with:

- `as_of_date`
- `player_id`
- `prop_type` in `{PTS, REB, AST}`
- `line`
- `odds_american` (optional)
- `book` (optional)

Current limitation (intentional for MVP):

- Lines are not yet linked to a specific **game_id/team_id** (we’ll add matchup binding next).

---

### 11.6 CLI commands added (for visibility + overlap checking)

Run commands from the repo root using:

- **`python3 run_cli.py <command>`**

Implemented commands:

- **`init-db`**: initialize SQLite
- **`ingest-boxscore <file>`**: ingest a `.txt` file
- **`ingest-boxscore-stdin --date YYYY-MM-DD --label LABEL`**: paste boxscore into stdin and ingest
- **`list-games --limit N`**: list ingested games (shows dates)
- **`show-game <game_id>`**: show parsed player lines for a game
- **`summary`**: counts of teams/players/games/rows/lines
- **`audit-duplicates`**: overlap check (duplicate games by date+matchup)
- **`ingest-lines-stdin --date YYYY-MM-DD --book NAME`**: paste sportsbook lines into stdin and ingest
- **`list-lines [--date YYYY-MM-DD]`**: view ingested lines
- **`gui`**: run the GUI app

**New Analysis Commands:**

- **`validate [--fix] [--verbose]`**: run data validation checks, optionally fix issues
- **`cleanup [--dry-run]`**: remove orphaned data (teams with no games, etc.)
- **`project --team TEAM [--opponent OPP] [--date DATE]`**: generate projections for a team
- **`usage-impact --team TEAM --out PLAYER [--historical]`**: show usage redistribution when a player is out
- **`matchup --away TEAM --home TEAM [--date DATE]`**: generate matchup-specific prop recommendations
- **`backtest [--start DATE] [--end DATE] [--min-edge PCT]`**: run backtest on historical lines
- **`accuracy --player NAME [--stat PTS|REB|AST]`**: analyze projection accuracy for a player
- **`bias-analysis [--min-games N]`**: analyze systematic projection biases
- **`alerts [--date DATE] [--min-edge PCT] [--team TEAM]`**: find edge alerts where projection differs from line

---

### 11.7 GUI improvements implemented

GUI is Tkinter-based (stdlib) and launched with:

- `python3 run_cli.py gui`

Tabs currently available:

- **Games**
  - shows game list (with **date**)
  - button: “Use selected game’s date in Paste tab”
  - import `.txt` file button
- **Paste Box Score**
  - paste raw text + set date + label + ingest
  - **clears the paste box after successful ingest**
  - “Recent dates” dropdown to speed up multi-game entry on same date
  - “Ingest pasted text” button moved to its own bar to avoid disappearing on smaller windows
- **Standings**
  - displays the latest `data/raw/metadata/<season>/standings__*.txt`
- **Team Defense**
  - displays the latest `data/raw/metadata/<season>/team_defense__*.txt`
- **Team Stats**
  - dropdown to open a `team_stats__*__*.txt` file

Bottom status bar:

- Shows: `Games | Players | Lines | Duplicate games`

---

### 11.8 Database-Backed Player Archetypes (NEW)

The player archetype system has been completely redesigned:

**Problem (before):**
- Player archetypes were stored in giant hard-coded Python dictionaries (`roster.py`, `archetypes.py`)
- ~2000+ lines of static data that would go stale with trades, roster changes
- No way to edit without changing code

**Solution (now):**
- New `player_archetypes` table in SQLite database
- New `player_similarity_groups` and `elite_defenders` tables
- Database-backed functions that:
  1. First check the database for player data
  2. Fall back to built-in defaults if not found
  3. Allow manual overrides without code changes

**New CLI Commands:**
- `seed-archetypes` - Populate database from built-in defaults (~200 players)
- `list-archetypes` - List archetypes with filtering
- `show-archetype <player>` - Show detailed player info

**New API Endpoints:**
- `GET /api/archetypes-db` - List all archetypes (DB + defaults)
- `GET /api/archetypes-db/player/<name>` - Get player archetype
- `PUT /api/archetypes-db/player/<name>` - Update player archetype
- `DELETE /api/archetypes-db/player/<name>` - Delete (reverts to defaults)
- `POST /api/archetypes-db/seed` - Seed database from defaults
- `GET /api/archetypes-db/stats` - Get archetype statistics

**Benefits:**
- User can update player teams after trades
- User can add new players not in defaults
- User can adjust archetypes based on game observations
- All without touching code

---

### 11.9 Flask Made Optional (NEW)

The application now runs stdlib-only for core functionality:

**Before:**
- Flask was a required dependency
- Couldn't run any CLI commands without Flask installed

**After:**
- Flask is an optional dependency (`pip install -e ".[web]"`)
- All CLI commands work without Flask:
  - `init-db`, `ingest-boxscore`, `list-games`, `show-game`
  - `summary`, `audit-duplicates`, `list-lines`
  - `seed-archetypes`, `list-archetypes`, `show-archetype`
- Web GUI requires Flask and shows helpful error message if not installed

---

---

## 12) Idea.txt Requirements Checklist

This section tracks all requirements from `Idea.txt` and their implementation status.

### ✅ Fully Implemented

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Run locally, no hosting | ✅ | Runs at `http://127.0.0.1:5050` |
| Store full box scores | ✅ | SQLite `boxscore_player` table |
| Parse raw ESPN format | ✅ | `boxscore_parser.py` handles "undefined Headshot" format |
| Parse formatted tables | ✅ | Markdown table parser |
| Target PTS/REB/AST props | ✅ | Projector generates these three stats |
| Good GUI for data entry | ✅ | Flask web interface |
| Copy-paste box scores | ✅ | Paste page in web UI |
| Player archetypes (point centers, stretch 5s, etc.) | ✅ | DB-backed `player_archetypes` table |
| Handle different player types | ✅ | 20+ offensive/defensive archetype classifications |
| Track injuries (DND/DNP) | ✅ | Captured from box scores, stored in DB |
| Top 7 players only | ✅ | `is_top_7` flag in projector |
| Back-to-back detection | ✅ | `matchups.py` → `get_back_to_back_status()` |
| Close game targeting (spread ≤ 6) | ✅ | `game_lines` table, `close_only` filter in API |
| Player salaries storage | ✅ | `player_salaries` table |
| Similar player groupings | ✅ | `player_similarity_groups` table |
| Elite defender tracking | ✅ | `elite_defenders` table, `is_elite_defender` flag |
| Report documentation | ✅ | This REPORT.md file |
| Database storage | ✅ | SQLite at `data/db/nba_props.sqlite3` |
| Team defense ratings | ✅ | `team_defense_ratings` table |
| Inactive player tracking | ✅ | `inactive_players` table |
| Avoid hard-coded data going stale | ✅ | DB-backed archetypes (can edit without code changes) |

### ⚠️ Partially Implemented

| Requirement | Status | Notes |
|-------------|--------|-------|
| Sportsbook lines comparison | ✅ | Full edge calculation with `alerts` command and API |
| Team win prediction | ⚠️ | Basic spread/line storage; needs win probability model |
| Projections vs lines | ✅ | `matchup` and `alerts` CLI commands, full API support |

### 📋 Recently Implemented (New)

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Usage redistribution when star out | ✅ | `usage-impact` CLI command, calculates stat redistribution |
| Matchup-specific prop recommendations | ✅ | `matchup` CLI command, position/player vs team analysis |
| Backtesting with historical lines | ✅ | `backtest`, `accuracy`, `bias-analysis` CLI commands |
| Automated edge alerts | ✅ | `alerts` CLI command, scans lines vs projections |
| Data validation & safety checks | ✅ | `validate` and `cleanup` CLI commands |

### 📋 Not Yet Implemented

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Team playing style comparisons | Low | "Teams with similar styles" |
| Full 82-game workflow guide | Low | Documentation for season-long use |
| Team win probability model | Low | More sophisticated spread analysis |

---

## 13) Running the Application Locally

### How It Works

This application runs **entirely on your computer**. When you run:

```bash
python3 run_cli.py gui
```

You'll see:

```
🏀 NBA Props Predictor
   Running at: http://127.0.0.1:5050
   Press Ctrl+C to stop

 * Serving Flask app 'nba_props.web.app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
```

**This is completely normal!** Here's what it means:

- `127.0.0.1` = "localhost" = your computer only
- `5050` = the port number
- The "WARNING" is just Flask being cautious - for personal use, this is perfect
- `200` responses mean everything is working

### To Access the App

1. Keep the terminal running
2. Open any web browser
3. Go to: **http://127.0.0.1:5050**
4. That's it! The app is running locally.

### No Internet Required

- All data is stored on your computer
- No cloud hosting needed
- No subscription fees
- Your data never leaves your machine

---

## 14) Next Steps (Recommended Order)

### Immediate: Add More Game Data

The projection system improves with more data. Priority:

1. **Add recent games** - Paste box scores from the last week
2. **Seed archetypes** - Run `python3 run_cli.py seed-archetypes`
3. **Add today's lines** - Use the Data page to enter sportsbook lines

### Short-term: Use the Projections

1. Go to Projections page
2. Select a matchup (e.g., LAL @ BOS)
3. Review projections for each player
4. Compare to your sportsbook's lines
5. Look for edges (projection significantly above/below line)

### Medium-term: Improve Accuracy

1. Track which predictions hit/miss
2. Adjust player archetypes if needed
3. Note which defenders actually limit players
4. Update the DB with your observations

### Long-term: Build History

- Enter games daily throughout the season
- Build up enough data for backtesting
- Calibrate the projection model
- Export winning patterns

---

## 15) Data Entry Workflow (Daily)

### Morning Routine (After Last Night's Games)

```bash
# 1. Start the web app
python3 run_cli.py gui

# 2. For each game from last night:
#    - Go to ESPN box score
#    - Select all (Ctrl+A) and copy (Ctrl+C)
#    - Go to Paste page in the app
#    - Set the correct date
#    - Paste and click "Ingest Box Score"

# 3. Check your data
python3 run_cli.py summary
```

### Before Tonight's Games

```bash
# 1. Add injury info if any key players are out
#    (Use Data page → Injuries section)

# 2. Add sportsbook lines for tonight's games
#    (Use Data page → Lines section)

# 3. Go to Projections page
#    - Select tonight's matchups
#    - Compare projections to lines
#    - Look for edges
```

### Weekly Maintenance

```bash
# Update archetypes if there were trades
python3 run_cli.py seed-archetypes --overwrite

# Check data quality
python3 run_cli.py audit-duplicates
python3 run_cli.py summary
```

