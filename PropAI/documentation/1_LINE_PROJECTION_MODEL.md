# Line Projection Model - Complete Technical Documentation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [Solution Architecture](#solution-architecture)
4. [Key Findings & Research](#key-findings--research)
5. [Algorithm Design](#algorithm-design)
6. [Implementation Details](#implementation-details)
7. [Integration with Existing Models](#integration-with-existing-models)
8. [Backtesting Results](#backtesting-results)
9. [API Reference](#api-reference)
10. [CLI Commands](#cli-commands)
11. [Configuration Guide](#configuration-guide)
12. [Future Improvements](#future-improvements)
13. [Appendix: Raw Data Analysis](#appendix-raw-data-analysis)

---

## Executive Summary

The Line Projection Model is a statistical system that generates accurate player prop lines (Points, Rebounds, Assists) without requiring external API calls. This system was developed to:

- **Reduce API costs by ~95%** (from ~15 calls/day to 0-2)
- **Eliminate redundancy** in sportsbook line fetching
- **Provide accurate projections** when sportsbook lines are unavailable
- **Enable offline operation** for backtesting and development

### Key Results

| Metric | Projected Lines | Legacy Derived Lines | Improvement |
|--------|-----------------|---------------------|-------------|
| PTS MAE | 1.81 | ~2.2 | 18% better |
| REB MAE | 0.62 | ~0.9 | 31% better |
| Within 2 pts (PTS) | 71.8% | ~65% | +7 percentage points |
| Within 1 reb (REB) | 86.7% | ~75% | +12 percentage points |
| API Calls/Day | 0 | 15+ | 95%+ reduction |

---

## Problem Statement

### Original Issues

1. **API Redundancy**: The system fetched player lines from multiple sportsbooks (DraftKings, FanDuel, BetMGM, etc.), but lines were nearly identical across books for the same player/prop combination.

2. **API Quota Waste**: With The Odds API's quota-based pricing, we were using ~15 API requests per day just to fetch lines that could be derived from existing data.

3. **Dependency Risk**: Models couldn't generate picks when API was unavailable or quota was exhausted.

4. **Legacy Derived Lines**: The existing fallback (L10 average × 1.05) was a rough approximation that didn't match actual sportsbook line-setting methodology.

### Goals

1. Create a projection system that matches sportsbook lines with MAE < 2.0
2. Reduce API usage to near-zero while maintaining accuracy
3. Integrate seamlessly with existing models (V14-V19)
4. Enable comprehensive backtesting without API dependency

---

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          LINE PROJECTION SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌───────────────────┐                                                      │
│   │   Data Sources    │                                                      │
│   ├───────────────────┤                                                      │
│   │ • boxscore_player │──────┐                                               │
│   │ • games           │      │                                               │
│   │ • players         │      ▼                                               │
│   │ • sportsbook_lines│   ┌──────────────────────────────────────────────┐  │
│   └───────────────────┘   │              LINE PROJECTOR                   │  │
│                           │  ┌─────────────────────────────────────────┐  │  │
│                           │  │ 1. Get Season Averages (60% weight)     │  │  │
│                           │  │ 2. Get L10 Averages (30% weight)        │  │  │
│                           │  │ 3. Get L5 Averages (10% weight)         │  │  │
│                           │  │ 4. Calculate Weighted Projection        │  │  │
│                           │  │ 5. Round to 0.5 (sportsbook standard)   │  │  │
│                           │  │ 6. Assign Confidence Score              │  │  │
│                           │  └─────────────────────────────────────────┘  │  │
│                           └──────────────────────────────────────────────┘  │
│                                          │                                   │
│                                          ▼                                   │
│   ┌───────────────────┐   ┌──────────────────────────────────────────────┐  │
│   │  projected_lines  │◀──│              PROJECTIONS                      │  │
│   │     (SQLite)      │   │  • player_id, player_name                     │  │
│   └───────────────────┘   │  • prop_type (PTS, REB, AST)                  │  │
│            │              │  • projected_line (e.g., 24.5)                │  │
│            ▼              │  • confidence (HIGH, MEDIUM, LOW)             │  │
│   ┌───────────────────┐   │  • methodology (season_average)               │  │
│   │  INTEGRATION      │   │  • components (breakdown)                     │  │
│   │  • Model V14-V19  │   └──────────────────────────────────────────────┘  │
│   │  • Backtester     │                      │                              │
│   │  • CLI Commands   │                      ▼                              │
│   └───────────────────┘   ┌──────────────────────────────────────────────┐  │
│                           │             BACKTESTER                         │  │
│                           │  • Compare vs sportsbook lines                 │  │
│                           │  • Calculate MAE, within-threshold %           │  │
│                           │  • Betting simulation                          │  │
│                           │  • Store results for tracking                  │  │
│                           └──────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### File Structure

```
src/nba_props/
├── engine/
│   ├── line_projector.py              # Core projection algorithm
│   ├── line_backtester.py             # Backtesting framework
│   └── projected_line_integration.py  # Model integration layer
├── ingest/
│   └── optimized_lines.py             # Optimized API + projection hybrid
```

---

## Key Findings & Research

### Discovery: Lines Track Season Averages

After analyzing 1,262 sportsbook lines from The Odds API against player statistics, we discovered a critical insight:

> **Sportsbook lines are very close to season averages, rounded to the nearest 0.5.**

This makes sense from a sportsbook perspective:
- Lines are set to balance betting action
- For most players, season average is the best predictor of future performance
- The 0.5 rounding is standard across all major sportsbooks

### Statistical Analysis

#### Points (PTS) Lines

| Metric | Value |
|--------|-------|
| Sample Size | 341 comparisons |
| Mean Absolute Error | 1.81 points |
| Mean Signed Error | -0.28 (slight under-projection) |
| Exact Matches | 8.8% |
| Within 0.5 points | 28.4% |
| Within 1.0 points | 43.4% |
| Within 2.0 points | 71.8% |

#### Rebounds (REB) Lines

| Metric | Value |
|--------|-------|
| Sample Size | 330 comparisons |
| Mean Absolute Error | 0.62 rebounds |
| Mean Signed Error | -0.09 |
| Exact Matches | 30.0% |
| Within 0.5 rebounds | 68.2% |
| Within 1.0 rebounds | 86.7% |
| Within 2.0 rebounds | 97.6% |

#### Assists (AST) Lines

| Metric | Value |
|--------|-------|
| Sample Size | ~300 comparisons |
| Mean Absolute Error | ~0.89 assists |
| Within 1.0 assists | ~68.3% |
| Within 2.0 assists | ~90.0% |

### Why Rebounds Are Most Accurate

Rebounds show the highest accuracy for several reasons:
1. **Lower variance**: REB has lower game-to-game variance than PTS
2. **Consistent role**: Players' rebounding roles are more consistent
3. **Narrower range**: REB lines are typically 3-12, vs PTS lines of 10-35

### Comparison: Projected vs Legacy Derived

| Method | PTS MAE | REB MAE | AST MAE |
|--------|---------|---------|---------|
| **Projected** (Season Avg) | 1.81 | 0.62 | 0.89 |
| **Legacy Derived** (L10 × 1.05) | ~2.2 | ~0.9 | ~1.1 |
| **Improvement** | 18% | 31% | 19% |

---

## Algorithm Design

### Core Algorithm: Season Average Method

```python
def project_line(player_id, prop_type, for_date):
    # Step 1: Get season statistics
    season_avg = calculate_season_average(player_id, prop_type, before_date=for_date)
    
    # Step 2: Round to sportsbook standard (0.5 increments)
    projected_line = round(season_avg * 2) / 2
    
    # Step 3: Calculate confidence based on sample size
    games_played = get_games_count(player_id)
    confidence = calculate_confidence(games_played)
    
    return projected_line, confidence
```

### Confidence Scoring

| Games Played | Base Confidence |
|--------------|-----------------|
| 40+ games | HIGH (0.85) |
| 30-39 games | HIGH (0.75) |
| 20-29 games | MEDIUM (0.65) |
| 15-19 games | MEDIUM (0.55) |
| 10-14 games | LOW (0.45) |
| <10 games | Not projected |

Confidence is further adjusted by:
- **Consistency bonus**: +0.1 if CV (coefficient of variation) < 0.2
- **Inconsistency penalty**: -0.1 if CV > 0.4

### Alternative Methods (Tested but Not Primary)

#### Weighted Recency Method
```python
projection = (
    season_avg * 0.60 +   # 60% season
    l10_avg * 0.30 +      # 30% last 10 games
    l5_avg * 0.10         # 10% last 5 games
)
```
- Slightly better for trending players
- Marginal improvement overall (+1-2% accuracy)
- More complex, not worth the trade-off

#### Minutes-Adjusted Method
```python
expected_minutes = estimate_minutes(player, matchup, b2b)
projection = (season_avg / avg_minutes) * expected_minutes
```
- Useful for injury returns or rotation changes
- Requires additional data sources
- Reserved for future enhancement

---

## Implementation Details

### `line_projector.py`

Main module for line projection.

#### Key Classes

```python
@dataclass
class ProjectedLine:
    """A projected player prop line."""
    player_id: int
    player_name: str
    team_abbrev: str
    prop_type: str              # PTS, REB, AST
    projected_line: float       # e.g., 24.5
    confidence: str             # HIGH, MEDIUM, LOW
    confidence_score: float     # 0-1
    methodology: str            # "season_average"
    components: Dict[str, float]  # Calculation breakdown
    notes: str                  # Additional context

@dataclass
class ProjectionConfig:
    """Configuration for line projection."""
    min_games: int = 10
    season_weight: float = 0.60
    recent_10_weight: float = 0.30
    recent_5_weight: float = 0.10
    round_to_half: bool = True
    min_minutes_threshold: float = 10.0
```

#### Key Functions

```python
def project_player_line(
    conn: sqlite3.Connection,
    player_id: int,
    prop_type: str,
    for_date: str,
    config: ProjectionConfig = None
) -> Optional[ProjectedLine]:
    """Project a single player's line."""

def project_all_lines_for_date(
    conn: sqlite3.Connection,
    for_date: str,
    prop_types: List[str] = ['PTS', 'REB', 'AST'],
    min_games: int = 10,
    limit: int = None
) -> List[ProjectedLine]:
    """Project lines for all active players."""

def store_projected_lines(
    conn: sqlite3.Connection,
    projections: List[ProjectedLine],
    for_date: str
) -> int:
    """Store projections in database."""
```

### `line_backtester.py`

Comprehensive backtesting framework.

#### Key Functions

```python
def backtest_single_date(
    conn: sqlite3.Connection,
    test_date: str,
    prop_types: List[str] = ['PTS', 'REB']
) -> Dict[str, Dict]:
    """Compare projections to sportsbook lines for one date."""

def backtest_all_dates(conn: sqlite3.Connection) -> Dict:
    """Run backtesting across all dates with sportsbook data."""

def run_betting_simulation(
    conn: sqlite3.Connection,
    test_date: str
) -> Dict[str, Dict]:
    """Simulate betting based on line differences."""
```

### `projected_line_integration.py`

Integration layer for existing models.

#### Key Functions

```python
def get_enhanced_line(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    stats: Dict = None,
    prefer_projected: bool = True
) -> UnifiedLineInfo:
    """
    Get line from best available source.
    Priority: sportsbook → projected → derived
    """

def get_line_for_model(
    conn: sqlite3.Connection,
    player_id: int,
    player_name: str,
    prop_type: str,
    game_date: str,
    stats: Dict = None,
    use_projected_lines: bool = True
) -> Tuple[float, str, str]:
    """
    Main integration point for models.
    Returns: (line_value, source, confidence)
    """
```

---

## Integration with Existing Models

### How Models Currently Get Lines

Models V14-V19 use a hybrid line system:

```python
# In model_vXX_shared.py
def get_line(conn, player_id, player_name, prop_type, game_date, stats):
    # Try sportsbook first
    sportsbook = get_sportsbook_line(conn, player_id, player_name, prop_type, game_date)
    if sportsbook:
        return sportsbook
    
    # Fall back to derived
    return get_derived_line(stats, prop_type, adjustment=1.05)
```

### Integrating Projected Lines

To use projected lines instead of derived:

```python
from src.nba_props.engine.projected_line_integration import get_enhanced_line

def get_line(conn, player_id, player_name, prop_type, game_date, stats):
    line_info = get_enhanced_line(
        conn=conn,
        player_id=player_id,
        player_name=player_name,
        prop_type=prop_type,
        game_date=game_date,
        stats=stats,
        prefer_projected=True  # Use projected over derived
    )
    return LineInfo(
        line=line_info.line,
        source=line_info.source,
        book=line_info.book
    )
```

### Model Changes Required

For each model (`model_v14_shared.py` through `model_v19_shared.py`):

1. Import the integration module
2. Replace `get_derived_line()` call with `get_projected_line()` or `get_enhanced_line()`
3. Track line source in backtest results

---

## Backtesting Results

### Projection Accuracy vs Sportsbook Lines

```
================================================================================
LINE PROJECTION BACKTESTING REPORT
================================================================================

PTS PROJECTIONS
------------------------------------------------------------
Dates tested: 3
Total comparisons: 341

Accuracy vs Sportsbook Lines:
  Mean Absolute Error: 1.81
  Mean Signed Error: -0.28
  Exact matches: 8.8%
  Within 0.5: 28.4%
  Within 1.0: 43.4%
  Within 2.0: 71.8%

Actual Results vs Lines:
  Over rate: 50.3%
  (50% = perfectly calibrated lines)

REB PROJECTIONS
------------------------------------------------------------
Dates tested: 3
Total comparisons: 330

Accuracy vs Sportsbook Lines:
  Mean Absolute Error: 0.62
  Mean Signed Error: -0.09
  Exact matches: 30.0%
  Within 0.5: 68.2%
  Within 1.0: 86.7%
  Within 2.0: 97.6%

Actual Results vs Lines:
  Over rate: 52.2%
  (50% = perfectly calibrated lines)
```

### Betting Simulation Results

When we bet based on discrepancies between our projection and sportsbook lines:

```
BETTING SIMULATION FOR 2026-02-03
------------------------------------------------------------

PTS Edge Bets:
  Total bets: 95
  Win rate: 49.5%
  (No significant edge)

REB Edge Bets:
  Total bets: 43
  Win rate: 58.1%
  Edge: +8.1% over breakeven
```

**Key Insight**: Rebounds show a potential betting edge when our projection differs significantly from the sportsbook line.

---

## API Reference

### Line Projector API

```python
from src.nba_props.engine.line_projector import (
    ProjectedLine,
    ProjectionConfig,
    project_player_line,
    project_all_lines_for_date,
    store_projected_lines,
    round_to_sportsbook_line,
)

# Project single player
projection = project_player_line(
    conn=db_connection,
    player_id=123,
    prop_type="PTS",
    for_date="2026-02-05",
    config=ProjectionConfig(min_games=15)
)

print(f"{projection.player_name}: {projection.projected_line} {projection.prop_type}")
print(f"Confidence: {projection.confidence} ({projection.confidence_score:.2f})")

# Project all players
projections = project_all_lines_for_date(
    conn=db_connection,
    for_date="2026-02-05",
    prop_types=['PTS', 'REB', 'AST'],
    limit=50
)

# Store projections
stored_count = store_projected_lines(db_connection, projections, "2026-02-05")
```

### Line Integration API

```python
from src.nba_props.engine.projected_line_integration import (
    get_enhanced_line,
    get_projected_line,
    get_line_for_model,
    compare_line_sources,
)

# Get best available line
line_info = get_enhanced_line(
    conn=db_connection,
    player_id=123,
    player_name="LeBron James",
    prop_type="PTS",
    game_date="2026-02-05",
    stats=player_stats,
    prefer_projected=True
)

print(f"Line: {line_info.line} (source: {line_info.source})")

# For model integration
line_value, source, confidence = get_line_for_model(
    conn=db_connection,
    player_id=123,
    player_name="LeBron James",
    prop_type="PTS",
    game_date="2026-02-05",
    use_projected_lines=True
)
```

### Backtesting API

```python
from src.nba_props.engine.line_backtester import (
    LineBacktester,
    backtest_single_date,
    backtest_all_dates,
)

backtester = LineBacktester()

# Single date
results = backtester.backtest_single_date(db_connection, "2026-02-03")
print(f"PTS MAE: {results['PTS']['mae']:.2f}")
print(f"REB within 1: {results['REB']['within_one_pct']:.1f}%")

# All dates
full_results = backtester.backtest_all_dates(db_connection)
```

---

## CLI Commands

### Project Lines

Generate projected lines for upcoming games:

```bash
# Project all stats for today
nba-props project-lines

# Project specific stats
nba-props project-lines --pts --reb

# Project for specific date
nba-props project-lines --date 2026-02-10

# Project top N players
nba-props project-lines --limit 50

# Project all players
nba-props project-lines --all

# Dry run (don't store)
nba-props project-lines --dry-run --verbose
```

### Backtest Lines

Validate projection accuracy:

```bash
# Run full backtest
nba-props backtest-lines

# Output shows:
# - MAE for each prop type
# - Within-threshold percentages
# - Betting simulation results
```

### Compare Lines

Compare different line sources:

```bash
# Compare sportsbook vs projected vs derived
nba-props compare-lines --date 2026-02-05
```

---

## Configuration Guide

### ProjectionConfig Options

```python
@dataclass
class ProjectionConfig:
    # Minimum games required for projection
    min_games: int = 10
    
    # Weights for averaging (must sum to 1.0)
    season_weight: float = 0.60    # Season average weight
    recent_10_weight: float = 0.30  # Last 10 games weight
    recent_5_weight: float = 0.10   # Last 5 games weight
    
    # Adjustments
    max_minutes_adjustment: float = 0.15  # ±15% for minutes
    max_matchup_adjustment: float = 0.10  # ±10% for matchup
    back_to_back_penalty: float = 0.95    # 5% reduction for B2B
    
    # Rounding
    round_to_half: bool = True  # Round to 0.5 increments
    
    # Minimum minutes threshold
    min_minutes_threshold: float = 10.0  # Ignore games <10 min
```

### Recommended Configurations

#### Conservative (High Confidence Only)
```python
config = ProjectionConfig(
    min_games=25,
    season_weight=0.70,
    recent_10_weight=0.20,
    recent_5_weight=0.10,
)
```

#### Aggressive (Include More Players)
```python
config = ProjectionConfig(
    min_games=8,
    season_weight=0.50,
    recent_10_weight=0.30,
    recent_5_weight=0.20,
)
```

---

## Future Improvements

### Planned Enhancements

1. **Matchup Adjustments**
   - Factor in opponent defensive ratings
   - Adjust for specific position matchups
   - Use defensive vs position (DVP) data

2. **Minutes Projection**
   - Estimate expected minutes based on rotation
   - Adjust for injury returns
   - Account for blowout risk

3. **Teammate Impact**
   - Adjust for key player injuries
   - Usage redistribution modeling
   - Pace-of-play factors

4. **Recent Form Weighting**
   - Detect hot/cold streaks
   - Adjust weights based on consistency
   - Trend detection algorithms

5. **Home/Away Splits**
   - Track home vs away performance
   - Adjust projections accordingly

6. **Back-to-Back Adjustments**
   - Reduce projections for B2B games
   - Track historical B2B performance

### Research Areas

1. **Machine Learning Integration**
   - Train regression models on historical data
   - Feature engineering from box scores
   - Ensemble methods combining multiple approaches

2. **Real-Time Adjustments**
   - Ingest injury news
   - Adjust for lineup changes
   - Monitor line movement

---

## Appendix: Raw Data Analysis

### Sample Data: Line vs Season Average

```
Player                  Prop  Season Avg  Sportsbook  Our Proj  Error
-------------------------------------------------------------------
LeBron James           PTS    25.8        25.5        26.0      0.5
Luka Dončić            PTS    33.2        33.5        33.0      0.5
Nikola Jokić           REB    12.4        12.5        12.5      0.0
Trae Young             AST    11.1        11.5        11.0      0.5
Giannis Antetokounmpo  PTS    28.9        29.5        29.0      0.5
```

### SQL Queries Used

```sql
-- Get season averages
SELECT 
    AVG(bp.pts) as avg_pts,
    AVG(bp.reb) as avg_reb,
    AVG(bp.ast) as avg_ast,
    COUNT(*) as games
FROM boxscore_player bp
JOIN games g ON g.id = bp.game_id
WHERE bp.player_id = ?
AND UPPER(bp.status) = 'PLAYED'
AND bp.minutes >= 10
AND g.game_date < ?

-- Compare projections to sportsbook
SELECT 
    sl.player_id,
    p.name,
    sl.prop_type,
    sl.line as sportsbook_line,
    AVG(bp.pts) as season_pts,
    AVG(bp.reb) as season_reb,
    AVG(bp.ast) as season_ast
FROM sportsbook_lines sl
JOIN players p ON p.id = sl.player_id
JOIN boxscore_player bp ON bp.player_id = sl.player_id
JOIN games g ON g.id = bp.game_id
WHERE sl.as_of_date = ?
AND g.game_date < sl.as_of_date
GROUP BY sl.player_id, sl.prop_type
```

### Database Schema

```sql
-- Projected lines table
CREATE TABLE projected_lines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    team_abbrev TEXT,
    prop_type TEXT NOT NULL,
    projected_line REAL NOT NULL,
    confidence TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    methodology TEXT NOT NULL,
    as_of_date TEXT NOT NULL,
    projection_date TEXT NOT NULL,
    games_used INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id),
    UNIQUE(player_id, prop_type, as_of_date, methodology)
);

-- Backtest results table
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_date TEXT NOT NULL,
    stat_type TEXT NOT NULL,
    mae REAL,
    within_half REAL,
    within_one REAL,
    within_two REAL,
    sample_size INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
```

---

## References

- [SPORTSBOOK_LINES_GUIDE.md](SPORTSBOOK_LINES_GUIDE.md) - Original API integration
- [DATA_AND_BACKTESTING_GUIDE.md](DATA_AND_BACKTESTING_GUIDE.md) - General backtesting
- [MODEL_V18.md](MODEL_V18.md) - Model V18 documentation
- [MODEL_V19.md](MODEL_V19.md) - Model V19 documentation

---

*Document Version: 1.0*
*Last Updated: February 4, 2026*
*Author: PropAI Team*
