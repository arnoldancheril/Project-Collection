# Trade Deadline Handling System (V19.1)

## Overview

The trade deadline (February 6, 2026) causes significant roster upheaval that directly impacts player prop predictions. This system addresses:

1. **Traded players** — Historical stats become unreliable on a new team
2. **Tanking teams** — Star minutes get restricted, DNPs increase
3. **Roster chemistry shifts** — New teammates change usage patterns
4. **Role changes** — A bench player on one team may start on another

## Architecture

Three new engine modules work together:

```
engine/
├── trade_tracker.py          # Database schema + CRUD for trade records
├── tank_detector.py          # Statistical detection of tanking teams
└── post_trade_adjustments.py # Central engine combining both systems
```

Both `model_v19_general.py` and `model_v19_under.py` integrate via `post_trade_adjustments.py`.

---

## Module Details

### 1. `trade_tracker.py` — Trade Database

**Database tables created:**

| Table | Purpose |
|-------|---------|
| `player_trades` | Records each player transaction (trade, waiver, signing, buyout) |
| `team_roster_status` | Tracks team-level status (tanking flag, players lost/gained) |
| `post_trade_performance` | Stores post-trade game performance for learning |

**Key data classes:**

- `TradeInfo` — Represents a single trade with computed properties:
  - `days_since_trade` — How many days since the player moved
  - `confidence_discount` — How much to reduce prediction confidence (0.30 at day 0 → 0.05 after 30+ games)
  - `projection_weight_new_team` — How much to weight new-team data vs old (0.0 initially → 1.0 after 30 games)
  
- `TeamRosterStatus` — Team-level flags for tanking, players traded, star players lost/gained

- `TradeAdjustedStats` — Container for adjusted projections post-trade

**Key functions:**

```python
record_trade(conn, player_name, from_team, to_team, trade_date, ...)
record_team_status(conn, team_abbrev, is_tanking, ...)
get_player_trade_info(conn, player_name) -> Optional[TradeInfo]
get_all_traded_players(conn) -> List[TradeInfo]
get_trades_affecting_team(conn, team_abbrev) -> dict
get_team_trade_summary(conn, team_abbrev) -> dict
```

### 2. `tank_detector.py` — Automatic Tank Detection

Analyzes 5 statistical signals to determine if a team is tanking:

| Signal | Weight | Description |
|--------|--------|-------------|
| `record_poor` | High | Win percentage below .350 (or .300 = very poor) |
| `post_deadline_collapse` | High | Win% dropped 15%+ after deadline |
| `minutes_reduction` | Medium | Star players getting fewer minutes post-deadline |
| `dnp_increase` | Medium | More healthy DNPs for rotation players |
| `trade_selling` | Medium | Team traded away more players than acquired |

**Key functions:**

```python
detect_tanking(conn, team_abbrev, as_of_date, deadline_date) -> TankDetectionResult
detect_all_tanking_teams(conn, as_of_date, deadline_date) -> List[TankDetectionResult]
get_tank_adjusted_minutes(conn, team_abbrev, player_id, as_of_date) -> float
```

**Output example:**
```
WAS: is_tanking=True, confidence=0.70
  Signal: record_poor (strength=0.70): Very poor record suggests tanking
```

### 3. `post_trade_adjustments.py` — Central Engine

Combines trade tracking and tank detection into actionable adjustments for the prediction models.

**Key data classes:**

- `TradeContext` — Complete context for a player on game day:
  - `is_traded` — Was this player traded?
  - `is_on_tanking_team` — Is their current team tanking?
  - `minutes_factor` — Multiplier for expected minutes (e.g., 0.85 on tanking team)
  - `projection_factor` — Multiplier for stat projections
  - `confidence_factor` — Multiplier for prediction confidence
  - `risk_level` — LOW / MEDIUM / HIGH / VERY_HIGH
  - `has_any_impact` — Quick check if any adjustments needed

- `AdjustedProjection` — Adjusted pts/reb/ast projections with reasoning

**Key functions:**

```python
get_trade_context(conn, player_id, player_name, team_abbrev, game_date) -> TradeContext
apply_trade_adjustments(conn, player_id, ..., base_projection, ...) -> AdjustedProjection
should_skip_player(conn, player_id, player_name, team_abbrev, game_date) -> (bool, str)
get_trade_factor_for_under(trade_ctx, prop_type) -> (score, count, reasons)
get_trade_factor_for_over(trade_ctx, prop_type) -> (score, count, reasons)
generate_trade_deadline_report(conn, as_of_date) -> str
```

---

## Model Integration

### How it works in `evaluate_player_for_prop` / `evaluate_player_for_under`

```
1. Load player stats (existing)
2. Get defense/B2B/game context (existing)
3. NEW: get_trade_context() → TradeContext
4. Calculate factors + score (existing)
5. NEW: Add trade factors to under/over score via get_trade_factor_for_under/over()
6. Calculate projection (existing)  
7. NEW: apply_trade_adjustments() → adjusted projection
8. Calculate edge using adjusted projection
9. Generate pick (existing)
```

### Skip logic in `get_daily_picks`

Before loading stats for each player:
```python
skip, skip_reason = should_skip_player(conn, player_id, "", team_abbrev, game_date)
if skip:
    continue  # Player was recently traded, not enough new-team data
```

### Confidence ramp-up curve

For traded players, confidence starts low and ramps up:

| Games on new team | Confidence discount | New-team weight |
|-------------------|--------------------:|----------------:|
| 0 games | 30% reduction | 0% |
| 5 games | 25% reduction | 17% |
| 10 games | 18% reduction | 33% |
| 15 games | 12% reduction | 50% |
| 20 games | 8% reduction | 67% |
| 25 games | 5% reduction | 83% |
| 30+ games | No reduction | 100% |

---

## CLI Commands

### Record a trade
```bash
python run_cli.py record-trade \
  --player "Jimmy Butler" \
  --from-team MIA \
  --to-team PHX \
  --date 2026-02-05 \
  --type trade \
  --role star \
  --new-role star \
  --notes "Part of 3-team deal"
```

### Record team status
```bash
python run_cli.py record-team-status \
  --team WAS \
  --tanking \
  --tank-confidence 0.8 \
  --players-traded 3 \
  --players-acquired 5 \
  --minutes-factor 0.85 \
  --notes "Full rebuild mode"
```

### Generate trade deadline report
```bash
python run_cli.py trade-report
python run_cli.py trade-report --date 2026-02-15
```

### Check trade impact for a player
```bash
python run_cli.py trade-impact --player "Jimmy Butler" --date 2026-02-15
```

### Scan for tanking teams
```bash
python run_cli.py tank-scan
python run_cli.py tank-scan --date 2026-02-15 --deadline 2026-02-06
```

### List all recorded trades
```bash
python run_cli.py list-trades
```

---

## Workflow: Post-Trade-Deadline Setup

After the trade deadline passes, follow these steps:

### Step 1: Record major trades
```bash
# Example real trades
python run_cli.py record-trade --player "Jimmy Butler" --from-team MIA --to-team PHX --date 2026-02-05 --role star --new-role star
python run_cli.py record-trade --player "De'Aaron Fox" --from-team SAC --to-team SAS --date 2026-02-05 --role star --new-role star
# ... repeat for all significant trades
```

### Step 2: Record team statuses
```bash
python run_cli.py record-team-status --team WAS --tanking --tank-confidence 0.8 --players-traded 3 --minutes-factor 0.85
python run_cli.py record-team-status --team UTA --tanking --tank-confidence 0.6 --players-traded 2 --minutes-factor 0.90
```

### Step 3: Verify with reports
```bash
python run_cli.py trade-report
python run_cli.py tank-scan
```

### Step 4: Run daily picks as normal
The model now automatically:
- Skips recently-traded players with no new-team data
- Adjusts projections for traded players based on games played on new team
- Reduces confidence for tanking-team players
- Adds trade-related factors to under/over scoring
- Adjusts minutes expectations for tanking teams

---

## Constants and Tuning

Key constants in `post_trade_adjustments.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `TRADE_DEADLINE_DATE` | `"2026-02-06"` | The NBA trade deadline date |
| `MIN_GAMES_ON_NEW_TEAM` | 3 | Games before any predictions for traded player |
| `FULL_CONFIDENCE_GAMES` | 30 | Games until confidence is fully restored |
| `TANK_MINUTES_REDUCTION_FACTOR` | 0.85 | Minutes multiplier for tanking team stars |
| `TANK_CONFIDENCE_REDUCTION` | 0.10 | Confidence reduction for tanking team players |

---

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS player_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    from_team TEXT NOT NULL,
    to_team TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    trade_type TEXT DEFAULT 'trade',
    was_starter INTEGER DEFAULT 0,
    old_team_role TEXT DEFAULT 'rotation',
    expected_new_role TEXT DEFAULT 'rotation',
    notes TEXT DEFAULT '',
    games_on_new_team INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_name, trade_date)
);

CREATE TABLE IF NOT EXISTS team_roster_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_abbrev TEXT NOT NULL UNIQUE,
    is_tanking INTEGER DEFAULT 0,
    tank_confidence REAL DEFAULT 0.0,
    players_traded_away INTEGER DEFAULT 0,
    players_acquired INTEGER DEFAULT 0,
    star_players_lost TEXT DEFAULT '',
    star_players_gained TEXT DEFAULT '',
    expected_minutes_factor REAL DEFAULT 1.0,
    record_at_deadline TEXT DEFAULT '',
    playoff_probability REAL DEFAULT 0.0,
    notes TEXT DEFAULT '',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS post_trade_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    team_abbrev TEXT NOT NULL,
    game_date TEXT NOT NULL,
    minutes REAL,
    pts REAL, reb REAL, ast REAL,
    is_post_trade INTEGER DEFAULT 1,
    game_number_on_new_team INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Files Modified

| File | Changes |
|------|---------|
| `engine/trade_tracker.py` | **NEW** — Trade database schema + CRUD |
| `engine/tank_detector.py` | **NEW** — Statistical tank detection |
| `engine/post_trade_adjustments.py` | **NEW** — Central adjustment engine |
| `engine/model_v19_general.py` | Added trade context, factor scoring, projection adjustment, skip logic |
| `engine/model_v19_under.py` | Added trade context, factor scoring, projection adjustment, skip logic |
| `db.py` | Added `_init_trade_tables()` to `init_db()` |
| `cli.py` | Added 6 new commands: `record-trade`, `record-team-status`, `trade-report`, `trade-impact`, `tank-scan`, `list-trades` |


---

## V19.4 - Post-Trade-Deadline Fixes (February 2026)

After running backtests against actual post-deadline games (Feb 6-12, 2026), several bugs were found and fixed.

### Critical Bug Fixes

1. **trade_uncertainty_active never triggered** - Was checking stats.was_traded instead of trade_ctx.player_was_traded (authoritative from player_trades table). Trade confidence discounts were never applied.
2. **trade_confidence_discount never applied** - hasattr(stats, 'trade_confidence_discount') was always False. Now explicitly set before pick creation.
3. **get_projection() trade-aware weights bypassed** - Always passed config.get_weights() explicitly, bypassing the built-in trade-aware logic inside get_projection() (only active when weights=None). Now uses dynamic weight selection.
4. **Tank detection false positives** - OKC (40-13) and BOS (34-18) wrongly flagged as tanking. Fixed with win% gate and sustained-average minutes cliff.

### Dynamic Lookback Weights (V19.4)

Post-deadline, historical averages (L15, season) may reflect a completely different team context. Three-tier weight adjustment:

- Traded player (< 10 new-team games): pass weights=None to get_projection() -> built-in trade-aware schedule
- Disrupted team stability <= 0.35 (e.g. CHI 6in/6out): L3=35%, L5=35%, L10=20%, L15=5%, Season=5%
- Disrupted team stability <= 0.50 (e.g. MEM 4out/3in): L3=25%, L5=30%, L10=25%, L15=10%, Season=10%
- Disrupted team stability 0.50-0.59 (e.g. UTA): L3=15%, L5=25%, L10=30%, L15=15%, Season=15%
- All others: Standard weights L3=10%, L5=20%, L10=30%, L15=20%, Season=20%

Teams affected in 2026: CHI (0.30), MEM (0.44), UTA (0.52)

### Tank Detection Win% Gate

| Win Rate | Max Tanking Confidence |
|----------|----------------------|
| >= 55% | 0.20 (monitoring only) |
| >= 50% | 0.25 |
| >= 42% | 0.40 |
| >= 38% | 0.55 |
| < 38% or known tanker | no cap |

### auto_update_team_roster_status Fix

Previously, auto_update_team_roster_status() always wrote is_tanking=False, silently overwriting manually-set flags for UTA and WAS. Now reads and preserves existing tanking flags before updating roster counts.

### Setup Script: run_trade_setup.py

New script to fully populate trade and tank data:

    python3 run_trade_setup.py

- Auto-detects 44 trades from boxscore history
- Updates games_with_new_team counts
- Sets team roster status for 26 teams
- Marks UTA (85%) and WAS (80%) as confirmed tankers
- Runs live tank detection for all 30 teams

### V19.4 Post-Deadline Backtest Results

- Pre-deadline: 56.3% hit rate (1,405 picks)
- Post-deadline: 54.8% hit rate (126 picks), ROI +4.55%
- Sportsbook lines post-deadline: 84.6% (11/13) -- prioritize when available
- Derived lines post-deadline: 51.3% (needs improvement as sample grows)
