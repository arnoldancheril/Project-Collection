# Walk-Forward Backtesting Methodology

## Overview

This document describes the **correct** methodology for backtesting PropAI models. Previous backtests may have been conducted incorrectly by inadvertently using future data or not properly simulating real-world conditions.

**Last Updated:** February 4, 2026

---

## Table of Contents

1. [The Problem with Traditional Backtesting](#1-the-problem-with-traditional-backtesting)
2. [Walk-Forward Methodology](#2-walk-forward-methodology)
3. [Implementation Details](#3-implementation-details)
4. [Line Projection Model Integration](#4-line-projection-model-integration)
5. [Running Walk-Forward Backtests](#5-running-walk-forward-backtests)
6. [Interpreting Results](#6-interpreting-results)
7. [Common Pitfalls to Avoid](#7-common-pitfalls-to-avoid)

---

## 1. The Problem with Traditional Backtesting

### What Can Go Wrong

Traditional backtesting can produce **inflated results** through several mechanisms:

#### 1.1 Look-Ahead Bias (Data Leakage)

**Problem:** Using future data that wouldn't have been available at prediction time.

**Example:**
```
Testing for December 15th:
❌ WRONG: Using season averages that include December 15th results
✓ CORRECT: Using only data from before December 15th
```

#### 1.2 Static Data Window

**Problem:** Using the same data snapshot for all test dates.

**Example:**
```
❌ WRONG: 
  - Testing Dec 1-31 using full season data through Dec 31
  - All predictions have access to 82+ games of data

✓ CORRECT:
  - Testing Dec 1: Use only data through Nov 30
  - Testing Dec 15: Use only data through Dec 14
  - Testing Dec 31: Use only data through Dec 30
```

#### 1.3 Line Source Confusion

**Problem:** Comparing predictions against "easy" derived lines instead of actual sportsbook lines.

**Example:**
```
Player's L10 average: 22.5 PTS
Derived line: 22.5 × 1.05 = 23.6 → 23.5 PTS
Our projection: 24.0 PTS → OVER with 2.1% edge
Actual sportsbook line: 25.5 PTS
Actual result: 24 PTS

❌ WRONG (vs derived): HIT (24 > 23.5)
✓ CORRECT (vs sportsbook): MISS (24 < 25.5)
```

---

## 2. Walk-Forward Methodology

### 2.1 Core Concept

Walk-forward backtesting simulates real-world conditions by:

1. **Rolling Data Window:** Only data available BEFORE the test date is used
2. **Sequential Processing:** Each day is processed in chronological order
3. **Cumulative Learning:** New data is incorporated after each day

### 2.2 Step-by-Step Process

```
START DATE: December 1, 2025
END DATE: February 3, 2026

Step 1: December 1, 2025
────────────────────────
Available Data: October 21, 2025 - November 30, 2025
Actions:
  1. Generate projected lines using data through Nov 30
  2. Generate picks for Dec 1 games
  3. Wait for actual results
  4. Grade picks against actual outcomes
  5. Store: picks, lines, results

Step 2: December 2, 2025
────────────────────────
Available Data: October 21, 2025 - December 1, 2025
                ↑ NOW INCLUDES Dec 1 data
Actions:
  1. Incorporate Dec 1 box scores into averages
  2. Generate projected lines using data through Dec 1
  3. Generate picks for Dec 2 games
  4. Grade picks against actual outcomes
  5. Store: picks, lines, results

Step 3: December 3, 2025
────────────────────────
Available Data: October 21, 2025 - December 2, 2025
                ↑ NOW INCLUDES Dec 2 data
... and so on

Final Step: February 3, 2026
────────────────────────────
Available Data: October 21, 2025 - February 2, 2026
Actions:
  1. Generate picks using all data through Feb 2
  2. Grade picks
  3. Aggregate all results
```

### 2.3 Mathematical Representation

For each test date $D_i$ in range $[D_{start}, D_{end}]$:

$$
\text{TrainingData}(D_i) = \{ \text{data} : \text{date} < D_i \}
$$

$$
\text{Predictions}(D_i) = f(\text{TrainingData}(D_i))
$$

$$
\text{Grade}(D_i) = \text{compare}(\text{Predictions}(D_i), \text{Actuals}(D_i))
$$

---

## 3. Implementation Details

### 3.1 Database Queries Must Use Date Filters

All data retrieval functions MUST filter by date:

```python
# CORRECT: Filter games before test date
def get_player_season_averages(conn, player_id, before_date):
    return conn.execute("""
        SELECT AVG(pts), AVG(reb), AVG(ast)
        FROM boxscore_player bp
        JOIN games g ON g.id = bp.game_id
        WHERE bp.player_id = ?
        AND g.game_date < ?  -- CRITICAL: Only games BEFORE test date
    """, (player_id, before_date)).fetchone()
```

### 3.2 Line Projection Must Respect Date Boundaries

The Line Projection Model uses season averages. These MUST be calculated using only pre-date data:

```python
# From line_projector.py
def project_player_line(conn, player_id, prop_type, for_date, config):
    # Gets season averages BEFORE for_date
    season_stats = get_player_season_averages(
        conn, player_id, for_date,  # <-- for_date is the cutoff
        config.min_minutes_threshold
    )
    # ... projection logic
```

### 3.3 Model Stats Loading

All model stat loading functions (e.g., `load_player_stats`) include a `before_date` parameter:

```python
# From model_v16_shared.py
def load_player_stats(conn, player_id, before_date, ...):
    rows = conn.execute("""
        SELECT g.game_date, b.pts, b.reb, b.ast, b.minutes
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        WHERE b.player_id = ?
          AND g.game_date < ?  -- Only historical data
        ORDER BY g.game_date DESC
        LIMIT ?
    """, (player_id, before_date, max_games)).fetchall()
```

### 3.4 Defense Rankings Must Be Versioned

Defense vs Position rankings change over time. Ideally:

```python
# Future enhancement: Time-versioned defense data
def get_defense_context(conn, team_abbrev, position, as_of_date):
    return conn.execute("""
        SELECT pts_rank, reb_rank, ast_rank
        FROM team_defense_vs_position
        WHERE team_abbrev = ?
        AND position = ?
        AND updated_at < ?
        ORDER BY updated_at DESC
        LIMIT 1
    """, (team_abbrev, position, as_of_date)).fetchone()
```

Note: Current implementation may not have historical DVP data. This is a known limitation.

---

## 4. Line Projection Model Integration

### 4.1 How Line Projection Works

The Line Projection Model (documented in `1_LINE_PROJECTION_MODEL.md`) generates betting lines without API calls:

**Algorithm:**
1. Calculate player's season average (only using games < test_date)
2. Round to nearest 0.5 (sportsbook standard)
3. Assign confidence based on sample size

**Accuracy:**
| Stat | MAE vs Sportsbook | Within 0.5 pts |
|------|-------------------|----------------|
| PTS  | 1.81              | 28.4%          |
| REB  | 0.62              | 68.2%          |
| AST  | 0.89              | ~68%           |

### 4.2 Line Source Priority

For each prediction, lines are obtained in this order:

1. **Sportsbook Line** (if available) - Most accurate for edge calculation
2. **Projected Line** - From Line Projection Model
3. **Derived Line** - Legacy fallback (L10 avg × 1.05)

### 4.3 Walk-Forward Line Generation

Each day, lines are projected using only available data:

```python
# Walk-forward line projection
for test_date in date_range(start, end):
    # Project lines using only pre-date data
    projected_lines = project_all_lines_for_date(
        conn, 
        test_date,  # Lines will only use data < test_date
        prop_types=['PTS', 'REB'],
        min_games=10
    )
    
    # Generate model picks using projected lines
    picks = model.get_daily_picks(test_date, lines=projected_lines)
    
    # Grade against actual outcomes
    graded = grade_picks(picks, actuals[test_date])
```

---

## 5. Running Walk-Forward Backtests

### 5.1 CLI Command

```bash
# Run walk-forward backtest for a model
python run_cli.py walk-forward-backtest \
    --model v16_general \
    --start 2025-12-01 \
    --end 2026-02-03 \
    --verbose

# Compare all models with walk-forward methodology
python run_cli.py walk-forward-comparison \
    --start 2025-12-01 \
    --end 2026-02-03 \
    --output results/walk_forward_results.json
```

### 5.2 Python API

```python
from src.nba_props.engine.walk_forward_backtester import (
    run_walk_forward_backtest,
    WalkForwardConfig
)

config = WalkForwardConfig(
    start_date="2025-12-01",
    end_date="2026-02-03",
    use_projected_lines=True,
    track_line_sources=True,
)

# Run for single model
result = run_walk_forward_backtest(
    model="v16_general",
    config=config,
    verbose=True
)

print(result.summary())
```

### 5.3 Comprehensive Multi-Model Backtest

```python
from src.nba_props.engine.walk_forward_backtester import (
    run_comprehensive_walk_forward,
    compare_models_walk_forward
)

# Test all models
results = run_comprehensive_walk_forward(
    start_date="2025-12-01",
    end_date="2026-02-03",
    models=['v12', 'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19'],
)

# Generate comparison report
report = compare_models_walk_forward(results)
print(report)
```

---

## 6. Interpreting Results

### 6.1 Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Hit Rate | % of picks that hit | >52.4% (break-even at -110) |
| SB Hit Rate | Hit rate vs sportsbook lines only | True performance measure |
| Derived Hit Rate | Hit rate vs derived lines | May be inflated |
| MAE (Line) | Mean absolute error of projected lines | <2.0 for PTS |

### 6.2 Red Flags for Invalid Results

⚠️ **Warning Signs:**

1. **Derived >> Sportsbook Rate** - If derived hit rate is 10%+ higher, model may be exploiting easy targets
2. **Perfect Early Performance** - Unusually high rates early in season (insufficient data)
3. **No Date Filtering** - Check SQL queries for proper date constraints
4. **Static Line Sources** - All lines from same snapshot instead of rolling

### 6.3 Report Template

```
================================================================================
WALK-FORWARD BACKTEST RESULTS
Model: V16 General
Period: 2025-12-01 to 2026-02-03 (65 days)
================================================================================

OVERALL PERFORMANCE
-------------------
Total Picks: 450
Hits: 288
Hit Rate: 64.0%

BY LINE SOURCE (CRITICAL)
-------------------------
Sportsbook Lines: 45 picks, 30 hits (66.7%)  ← TRUE ACCURACY
Projected Lines: 350 picks, 218 hits (62.3%)
Derived Lines: 55 picks, 40 hits (72.7%)  ← May be inflated

BY PROP TYPE
------------
PTS: 300 picks, 186 hits (62.0%)
REB: 150 picks, 102 hits (68.0%)

BY DIRECTION
------------
OVER: 250 picks, 155 hits (62.0%)
UNDER: 200 picks, 133 hits (66.5%)

LINE PROJECTION ACCURACY
------------------------
PTS MAE: 1.85 (vs actual sportsbook)
REB MAE: 0.68 (vs actual sportsbook)
Within 0.5: 48.2% (PTS), 67.5% (REB)

DAILY BREAKDOWN
---------------
Date       Games  Picks  Hits   Rate
2025-12-01    8     12     8   66.7%
2025-12-02    6      9     5   55.6%
...
================================================================================
```

---

## 7. Common Pitfalls to Avoid

### 7.1 Pitfall: Not Using Date Parameters

```python
# ❌ WRONG: No date filter
stats = get_player_averages(conn, player_id)

# ✓ CORRECT: Filter by date
stats = get_player_averages(conn, player_id, before_date=game_date)
```

### 7.2 Pitfall: Static Sportsbook Lines

```python
# ❌ WRONG: Using today's lines for historical test
lines = fetch_current_sportsbook_lines()

# ✓ CORRECT: Use historical lines or project them
lines = get_historical_lines(game_date) or project_lines(game_date)
```

### 7.3 Pitfall: Batch Processing Without Date Isolation

```python
# ❌ WRONG: Process all dates with same data
all_picks = []
for date in test_dates:
    picks = model.get_picks(date, global_stats)  # Same stats for all
    all_picks.extend(picks)

# ✓ CORRECT: Isolate data per date
all_picks = []
for date in sorted(test_dates):
    # Stats only include data before this date
    stats = load_stats_before_date(date)
    picks = model.get_picks(date, stats)
    all_picks.extend(picks)
    # Results from this date now available for next iteration
```

### 7.4 Pitfall: Not Tracking Line Sources

```python
# ❌ WRONG: Aggregate all results together
total_hits = sum(p.hit for p in picks)

# ✓ CORRECT: Track by line source
sb_hits = sum(p.hit for p in picks if p.line_source == "sportsbook")
projected_hits = sum(p.hit for p in picks if p.line_source == "projected")
derived_hits = sum(p.hit for p in picks if p.line_source == "derived")
```

---

## Summary

Walk-forward backtesting is **essential** for honest model evaluation:

1. **Use only pre-date data** for all predictions
2. **Process dates sequentially** to simulate real-world conditions
3. **Track line sources separately** to identify inflated results
4. **Integrate Line Projection Model** properly with date boundaries
5. **Report both sportsbook and derived rates** for transparency

Following this methodology ensures that backtest results reflect actual expected performance when deployed in production.

---

*Document Version: 1.0*
*Created: February 4, 2026*
*Author: PropAI Team*
