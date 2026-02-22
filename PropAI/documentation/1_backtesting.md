# Comprehensive Backtesting Guide

## Overview

This guide covers everything you need to know about data sources, backtesting procedures, and model validation for PropAI. It combines data requirements, honest backtesting practices, and the comprehensive model comparison system.

**Last Updated:** February 4, 2026

> **NEW:** For the definitive walk-forward backtesting methodology that eliminates look-ahead bias, see [WALK_FORWARD_BACKTESTING.md](WALK_FORWARD_BACKTESTING.md).

---

## Table of Contents

1. [Why Backtesting Matters](#1-why-backtesting-matters)
2. [Data Requirements](#2-data-requirements)
3. [How to Run Honest Backtests](#3-how-to-run-honest-backtests)
4. [Detecting Inflated Accuracy](#4-detecting-inflated-accuracy)
5. [Grading Methodology](#5-grading-methodology)
6. [Comprehensive Backtesting System](#6-comprehensive-backtesting-system)
7. [Line Projection Methods](#7-line-projection-methods)
8. [Reporting Standards](#8-reporting-standards)
9. [Web Interface Usage](#9-web-interface-usage)
10. [Data Sources Reference](#10-data-sources-reference)
11. [Database Schema](#11-database-schema)
12. [Best Practices](#12-best-practices)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Why Backtesting Matters

### The Problem with Previous Models

LLM-generated models sometimes have inflated accuracy because:
1. Using future data that wouldn't be available at prediction time
2. Comparing against "easy" derived lines instead of real sportsbook lines
3. Cherry-picking favorable date ranges
4. Not accounting for line source differences
5. Bugs that count ties as wins

### The "Derived Line Fallacy"

**The Core Issue:**
Previous models (V2-V15) tested picks against derived lines (player averages) rather than actual sportsbook lines. This inflated hit rates by 5-15%.

**Example:**
```
Player's L10 average: 22.5 PTS
Our derived line: 22.5 PTS
Our projection: 24.0 PTS → OVER with 6.7% edge
Actual sportsbook line: 24.5 PTS
Actual result: 23 PTS

Old models count: HIT (23 > 22.5)
Reality: MISS (23 < 24.5)
```

**Solution:** This guide ensures HONEST, REPRODUCIBLE backtesting with separate tracking for sportsbook vs derived lines.

> **CRITICAL:** Use walk-forward backtesting (see [WALK_FORWARD_BACKTESTING.md](WALK_FORWARD_BACKTESTING.md)) to ensure no future data leakage.

---

## 2. Data Requirements

### A. Sportsbook Lines (Preferred - Most Honest)

**Location:** `data/db/nba_props.sqlite3` → `sportsbook_lines` table

**Check available dates:**
```sql
SELECT DISTINCT as_of_date FROM sportsbook_lines ORDER BY as_of_date;
```

**Why they matter:**
- Sportsbook lines are the TRUE measure of model performance
- They include the house edge and market efficiency
- Testing against them reflects actual betting outcomes

### B. Derived Lines (Fallback)

Calculated as: `L10 average × 1.05`, rounded to 0.5

**Important:** Derived lines are "easier" to beat than sportsbook lines. Models typically show 5-15% higher accuracy against derived lines.

### C. Required Data for Each Backtest

| Data Type | Source | Purpose |
|-----------|--------|---------|
| Box Scores | ESPN, NBA.com | Player stats, actual results |
| Matchups | ESPN | Game schedule, opponents |
| Injuries | ESPN scraper | Player availability |
| Defense vs Position | Hashtag Basketball | UNDER targeting |
| Sportsbook Lines | The Odds API | Honest edge calculation |

---

## 3. How to Run Honest Backtests

### Step 1: Choose Date Range

Use dates that have actual game results in the database.

**Find valid dates:**
```python
import sqlite3
conn = sqlite3.connect('data/db/nba_props.sqlite3')
dates = conn.execute('''
    SELECT DISTINCT DATE(game_date) as gd 
    FROM games 
    ORDER BY game_date DESC 
    LIMIT 30
''').fetchall()
print([d[0] for d in dates])
```

### Step 2: Run the Backtest

Each model has a backtest function:

```python
# Model V16 (Recommended)
from src.nba_props.engine.model_v16_general import run_backtest_v16_general
result = run_backtest_v16_general("2025-12-01", "2026-02-03")

# Model V17
from src.nba_props.engine.model_v17_general import run_backtest_v17_general
result = run_backtest_v17_general("2025-12-01", "2026-02-03")

# Model V18
from src.nba_props.engine.model_v18_general import run_backtest_v18_general
result = run_backtest_v18_general("2025-12-01", "2026-02-03")

# Model V19
from src.nba_props.engine.model_v19_general import run_backtest_v19_general
result = run_backtest_v19_general("2025-12-01", "2026-02-03")
```

### Step 3: Extract Honest Metrics

Key metrics to record:

```python
# Total performance
total_picks = result.total_picks
total_hits = result.hits
hit_rate = total_hits / total_picks

# Line source breakdown (CRITICAL for honesty)
sportsbook_picks = result.sportsbook_picks
sportsbook_hits = result.sportsbook_hits
sportsbook_rate = sportsbook_hits / sportsbook_picks  # TRUE accuracy

derived_picks = result.derived_picks
derived_hits = result.derived_hits
derived_rate = derived_hits / derived_picks  # Inflated accuracy

# Prop type breakdown
pts_rate = result.pts_hits / result.pts_picks
reb_rate = result.reb_hits / result.reb_picks
```

---

## 4. Detecting Inflated Accuracy

### Red Flags

1. **No Line Source Separation**
   - If model doesn't track sportsbook vs derived separately, be suspicious
   - FIX: Add tracking to the model's backtest function

2. **Derived Rate >> Sportsbook Rate**
   - If derived accuracy is 10%+ higher than sportsbook, the model may be "overfitting" to easy-to-beat derived lines
   - The TRUE accuracy is closer to the sportsbook rate

3. **Inconsistent Attribute Names**
   - Some models use different names: `total_picks` vs `picks` vs `num_picks`
   - Use getattr with fallbacks:
   ```python
   picks = getattr(r, 'total_picks', 0) or getattr(r, 'picks', 0)
   hits = getattr(r, 'hits', 0) or getattr(r, 'total_hits', 0)
   ```

4. **Too-Perfect Results**
   - If any model shows >80% accuracy consistently, verify:
     - Is it using future data accidentally?
     - Is it only counting "confident" picks and ignoring misses?
     - Is the sample size too small?

---

## 5. Grading Methodology

### Step A: Calculate Line

1. If sportsbook line available → Use it (most accurate)
2. If not available → Use derived line:
   ```python
   derived_line = round(L10_average * 1.05 * 2) / 2  # Round to 0.5
   ```

### Step B: Get Model's Prediction

Model outputs:
- Projected value (what model thinks player will score)
- Direction (OVER or UNDER)
- Confidence (PREMIUM, HIGH, STANDARD)

### Step C: Get Actual Result

```sql
SELECT pts, reb, ast FROM boxscore_player 
WHERE player_id = ? AND game_id = ?;
```

### Step D: Grade the Pick

```python
if direction == "OVER":
    hit = actual > line
elif direction == "UNDER":
    hit = actual < line
else:  # Exact match (push)
    hit = None  # Don't count pushes
```

**Important:** Pushes (actual == line) should NOT be counted as wins.

### Step E: Aggregate Results

```python
hit_rate = total_hits / (total_picks - pushes)
```

---

## 6. Comprehensive Backtesting System

The system provides unified backtesting across 23+ prediction models.

### Terminal Usage (CLI)

```bash
# Basic comprehensive backtest
python run_cli.py comprehensive-backtest --weeks 8

# Compare latest models (V16-V19)
python run_cli.py comprehensive-backtest --latest --start 2025-12-01 --end 2026-01-15

# Compare UNDER-specialized models
python run_cli.py comprehensive-backtest --under --start 2025-12-01 --end 2026-01-15

# Filter by category
python run_cli.py comprehensive-backtest --category multi --weeks 4

# Save results to JSON
python run_cli.py comprehensive-backtest --latest --output results.json
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--weeks N` | Number of weeks to backtest (default: 8) |
| `--start DATE` | Start date (YYYY-MM-DD), overrides --weeks |
| `--end DATE` | End date (YYYY-MM-DD) |
| `--category CAT` | Filter: multi, single, specialized |
| `--latest` | Compare only V16-V19 models |
| `--under` | Compare only UNDER-specialized models |
| `--verbose` | Show detailed output |
| `--output FILE` | Save results to JSON file |

### Quality Score Formula

Balances accuracy with pick volume:

```python
quality_score = (hit_rate * 100) * math.log10(max(10, total_picks))
```

**Example:**
| Model | Hit Rate | Picks | Quality Score |
|-------|----------|-------|---------------|
| Model A | 60% | 100 | 120.0 |
| Model B | 80% | 10 | 80.0 |
| Model C | 55% | 500 | 148.6 |

Model C ranks highest despite lower hit rate because of volume.

### Model Categories

| Category | Models | Description |
|----------|--------|-------------|
| Multi-File | V12-V19 | Separate general/under/shared files |
| Single-File | V2-V10 | Self-contained legacy models |
| Specialized | production, final, under_v2 | Purpose-built models |

---

## 7. Line Projection Methods

### Three Methods Compared

| Method | Source | Accuracy | Use Case |
|--------|--------|----------|----------|
| **Sportsbook** | API | Best | When available |
| **Projected** | Season avg × round | MAE: 1.14 | Primary fallback |
| **Derived** | L10 × 1.05 | MAE: 1.26 | Legacy/comparison |

### Comparison Results

```
PROJECTED LINES:
  Mean Absolute Error: 1.14
  Within 0.5 points: 51.9%

DERIVED LINES (Legacy):
  Mean Absolute Error: 1.26
  Within 0.5 points: 47.6%

IMPROVEMENT: Projected lines are 9.4% more accurate
```

---

## 8. Reporting Standards

### Template

```
Model: [NAME]
Period: [START] to [END]
Total: [HITS]/[PICKS] ([RATE]%)

Line Source Performance:
  Sportsbook: [SB_HITS]/[SB_PICKS] ([SB_RATE]%) ← TRUE ACCURACY
  Derived: [DER_HITS]/[DER_PICKS] ([DER_RATE]%)

By Prop:
  PTS: [PTS_RATE]%
  REB: [REB_RATE]%
  
By Direction:
  OVER: [OVER_RATE]%
  UNDER: [UNDER_RATE]%

By Tier:
  PREMIUM: [PREM_RATE]%
  HIGH: [HIGH_RATE]%
  STANDARD: [STD_RATE]%
```

### Performance Thresholds

| Performance Level | Hit Rate |
|-------------------|----------|
| Profitable (at -110 odds) | ≥52.4% |
| Good | 55-60% |
| Excellent | 60%+ |
| Elite | 65%+ |

---

## 9. Web Interface Usage

### Model Lab (`/modellab`)

**Tabs:**
1. **🔬 Comprehensive Backtest** - Run analysis across all models
2. **📋 All Models** - Browse 23+ registered models
3. **📊 Quick Compare** - One-click comparisons
4. **🎯 Best Picks** - Best model by category
5. **⚙️ Legacy Lab** - Original functionality

### Matchups Tab (`/matchups`)

Shows:
- Today's games with predictions
- Prop recommendations with confidence
- Defense matchup breakdowns
- Injury impact analysis

### Data Quality Indicators

- 🟢 Fresh (< 1 day old)
- 🟡 Stale (1-3 days old)
- 🔴 Outdated (> 3 days old)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/modellab/models` | GET | List all models |
| `/api/modellab/comprehensive-backtest` | POST | Run full backtest |
| `/api/modellab/single-backtest` | POST | Single model test |
| `/api/modellab/compare-latest` | POST | Compare V16-V19 |
| `/api/backtesting/generate-picks` | POST | Generate picks |
| `/api/backtesting/compare-results` | POST | Grade picks |
| `/api/backtesting/performance` | GET | Performance stats |

---

## 10. Data Sources Reference

### Box Scores (Foundation)

**Source:** ESPN, Basketball Reference, manual paste
**Parser:** `src/nba_props/ingest/boxscore_parser.py`
**Table:** `boxscore_player`, `boxscore_team_totals`

**How to Add:**
1. Web Interface: `/paste` → paste raw box score
2. CLI: `python -m nba_props ingest-boxscore path/to/file.txt`
3. File Drop: `data/raw/boxscores/2025-26/YYYY-MM-DD/`

### Defense vs Position (Critical for UNDER)

**Source:** [Hashtag Basketball](https://hashtagbasketball.com/nba-defense-vs-position)
**Parser:** `src/nba_props/ingest/defense_position_parser.py`
**Table:** `team_defense_vs_position`

**Rating Thresholds:**
- **Elite** (Rank 1-5): Best targets for UNDER
- **Good** (Rank 6-10): Above average defense
- **Average** (Rank 11-20): League average
- **Poor** (Rank 21-25): Below average
- **Terrible** (Rank 26-30): Best targets for OVER

### Injury Reports

**Primary:** ESPN Web Scraper (`scrape_espn_injuries()`)
**Secondary:** Manual paste from NBA official reports
**Parser:** `src/nba_props/ingest/injury_parser.py`, `web_scraper.py`
**Table:** `injury_report`

**Status Mapping:**
| Status | Model Treatment |
|--------|-----------------|
| OUT | Removed from projections |
| DOUBTFUL | Treated as OUT (~25% play) |
| QUESTIONABLE | Flagged, adjusted |
| PROBABLE | Normal with caution |

### Sportsbook Lines

**Source:** DraftKings, FanDuel, BetMGM via The Odds API
**Parser:** `src/nba_props/ingest/lines_parser.py`
**Table:** `sportsbook_lines`

---

## 11. Database Schema

### Core Tables

```sql
-- Games
CREATE TABLE games (
    id INTEGER PRIMARY KEY,
    date TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT
);

-- Box Score (Player Stats)
CREATE TABLE boxscore_player (
    id INTEGER PRIMARY KEY,
    game_id INTEGER,
    player_id INTEGER,
    team_id INTEGER,
    minutes TEXT,
    pts INTEGER, reb INTEGER, ast INTEGER,
    fgm INTEGER, fga INTEGER,
    tpm INTEGER, tpa INTEGER,
    ftm INTEGER, fta INTEGER,
    oreb INTEGER, dreb INTEGER,
    stl INTEGER, blk INTEGER,
    tov INTEGER, pf INTEGER,
    plus_minus INTEGER
);

-- Sportsbook Lines
CREATE TABLE sportsbook_lines (
    id INTEGER PRIMARY KEY,
    player_id INTEGER,
    game_id INTEGER,
    prop_type TEXT,
    line_value REAL,
    over_odds INTEGER,
    under_odds INTEGER,
    source TEXT,
    captured_at TEXT
);

-- Team Defense vs Position
CREATE TABLE team_defense_vs_position (
    id INTEGER PRIMARY KEY,
    team_abbrev TEXT,
    position TEXT,
    pts_allowed REAL, pts_rank INTEGER,
    reb_allowed REAL, reb_rank INTEGER,
    ast_allowed REAL, ast_rank INTEGER,
    tpm_allowed REAL, tpm_rank INTEGER,
    updated_at TEXT
);
```

### Model Version Tables

```sql
-- Model Versions
CREATE TABLE model_versions (
    id INTEGER PRIMARY KEY,
    version_name TEXT UNIQUE,
    description TEXT,
    config_json TEXT,
    created_at TEXT
);

-- Model Picks
CREATE TABLE model_version_picks (
    id INTEGER PRIMARY KEY,
    version_id INTEGER,
    game_id INTEGER,
    game_date TEXT,
    player_name TEXT,
    prop_type TEXT,
    direction TEXT,
    projected REAL,
    line REAL,
    line_source TEXT,
    edge_pct REAL,
    confidence_score REAL,
    confidence_tier TEXT,
    actual_value REAL,
    hit INTEGER
);
```

---

## 12. Best Practices

### Daily Workflow

1. **Morning:** Update defense vs position from Hashtag Basketball
2. **Pre-Game:** Add injury reports (scrape from ESPN)
3. **Post-Game:** Ingest box scores
4. **Analysis:** Review model performance, run backtests

### Backtest Checklist

✅ At least 30 days of box scores
✅ Defense vs Position data updated
✅ Injury reports for historical dates
✅ Test multiple date ranges
✅ Check multiple prop types (PTS, REB)
✅ Verify confidence tier accuracy
✅ Analyze OVER vs UNDER separately
✅ Track line source (sportsbook vs derived)

### Model Validation Metrics

- Overall hit rate
- Hit rate by confidence tier
- Hit rate by prop type
- Hit rate by direction
- Model calibration (predicted vs actual)
- ROI simulation

---

## 13. Troubleshooting

### Common Issues

**"Database is locked"**
- Don't run multiple backtests in parallel
- Run sequentially

**Model shows 0 picks**
- Check if model requires specific configuration
- Verify data exists for date range

**Different attribute names**
```python
picks = getattr(r, 'total_picks', 0) or getattr(r, 'picks', 0)
hits = getattr(r, 'hits', 0) or getattr(r, 'total_hits', 0)
```

**Missing box scores**
```sql
SELECT g.id, g.date, ht.abbrev || ' vs ' || at.abbrev as matchup
FROM games g
JOIN teams ht ON g.home_team_id = ht.id
JOIN teams at ON g.away_team_id = at.id
LEFT JOIN boxscore_player bp ON bp.game_id = g.id
WHERE bp.id IS NULL
ORDER BY g.date DESC;
```

**Defense data not loading**
1. Check table: `SELECT COUNT(*) FROM team_defense_vs_position;`
2. Verify positions: `SELECT DISTINCT position FROM team_defense_vs_position;`
3. Re-paste from Hashtag Basketball

### Quick Reference

| Metric | Threshold |
|--------|-----------|
| Profitable at -110 | ≥52.4% |
| Good performance | 55-60% |
| Excellent | 60%+ |
| Elite | 65%+ |

**Note:** Sportsbook rate is the TRUE accuracy measure. Derived rate is likely inflated by 5-15%.

---

## Summary

| Data Source | Primary Use | Update Frequency |
|-------------|-------------|------------------|
| Box Scores | Player averages, trends | Daily (post-game) |
| Matchups | Schedule, game lines | Daily (pre-game) |
| Injuries | Usage redistribution | Daily (pre-game) |
| Defense vs Position | UNDER targeting | Weekly |
| Sportsbook Lines | Honest edge calculation | Daily (pre-game) |
| Player DRTG | Individual matchups | Monthly |

By maintaining fresh data and running regular backtests, you can achieve **65-75%+ hit rates** on high-confidence picks.

---

*Last Updated: February 4, 2026*
*See also: 1_FILE_STRUCTURE.md, 1_model_comparison.txt*
