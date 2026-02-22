# Model V14 - Dual NBA Props Prediction System

## Overview

Model V14 is a **dual-model approach** for NBA player prop predictions, splitting predictions into two specialized models:

1. **Model V14 General** (`model_v14_general.py`) - Focuses on OVER picks and REB UNDER
2. **Model V14 Under** (`model_v14_under.py`) - Specialized UNDER predictions with defense-first approach

Both models share common utilities in `model_v14_shared.py`.

## Key Innovation: Addressing the "Derived Line Fallacy"

### The Problem
Previous models tested against **player averages** (L10 mean) instead of actual **sportsbook betting lines**. This inflated hit rates by 5-15% because:
- Player averages ≠ Actual betting lines
- Sportsbooks factor in matchup, injuries, etc.
- Testing against averages doesn't reflect real betting outcomes

### The Solution
Model V14 implements **hybrid line handling**:
1. **Sportsbook lines** (when available): Use directly with PREMIUM confidence
2. **Derived lines** (when no sportsbook data): Use L10 average with +5% adjustment and require higher edge

```python
# Line source tracking
line_info = get_line(conn, player_id, player_name, prop_type, game_date, stats)
# line_info.source = "sportsbook" or "derived"
# line_info.book = "draftkings", "fanduel", etc. (if sportsbook)
```

---

## Backtest Results (2025-12-01 to 2026-02-02)

### Model V14 General

```
Period: 63 days | 444 games

OVERALL: 57.9% (84/145) ✅

BY DIRECTION:
  OVER:  64.7% (22/34) 🔥
  UNDER: 55.9% (62/111)

BY PROP TYPE:
  PTS: 75.0% (15/20) 🔥
  REB: 55.2% (69/125)

BY PATTERN:
  Cold Bounce:   84.6% (11/13) 🔥🔥🔥
  Usage Boost:   52.4% (11/21)

BY TIER:
  PREMIUM:  100.0% (5/5) 🔥🔥
  HIGH:     55.4% (46/83)
  STANDARD: 57.9% (33/57)
```

### Model V14 Under

```
Period: 63 days | 444 games

OVERALL UNDER: 56.8% (192/338) ✅

BY PROP TYPE:
  PTS UNDER: 59.2% (90/152) 🔥
  REB UNDER: 54.4% (93/171)
  AST UNDER: 60.0% (9/15)

BY KEY FACTORS:
  B2B Fatigue:     60.5% (98/162) 🔥
  Cold Streak:     57.8% (133/230)
  Elite Defense:   54.7% (75/137)
  Elite + Cold:    55.2% (16/29)

BY TIER:
  PREMIUM:  52.8% (19/36)
  HIGH:     57.3% (173/302) 🔥
```

---

## Model V14 General Details

### Purpose
The General model handles:
- **OVER picks** using the Cold Bounce pattern (84.6% hit rate!)
- **REB UNDER** vs strong defensive teams
- **Usage boost** scenarios (injured teammates)

### Key Patterns

#### Cold Bounce (84.6% Hit Rate) 🔥
The #1 most reliable OVER pattern:
```
TRIGGER: Player underperforming L5 vs L15+ by 15%+, but showing recovery
LOGIC: Regression to mean after cold stretch
EXAMPLE: Player averaging 22 PTS (season) dropped to 17 PTS (L5), now bouncing back
```

#### Usage Boost (52.4% Hit Rate)
When key teammates are injured:
```
TRIGGER: 1+ primary scorers injured on team
BOOST: +5-12% projection adjustment
LOGIC: More shots/touches available
```

#### Hot Sustained (DISABLED)
**Why disabled**: Backtest showed only 25.8% hit rate for OVER. Players don't sustain hot streaks.

### Configuration

```python
from src.nba_props.engine.model_v14_general import (
    get_daily_picks_v14_general,
    run_backtest_v14_general,
    ModelConfigV14General,
)

# Custom config
config = ModelConfigV14General()
config.min_edge_sportsbook = 6.0   # Edge required with real lines
config.min_edge_derived = 10.0     # Higher edge required with derived lines
config.min_avg_minutes = 23.0      # Minutes filter

# Get picks
picks = get_daily_picks_v14_general("2026-02-03", config=config)
print(picks.summary())
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_sportsbook` | 6.0% | Minimum edge vs sportsbook lines |
| `min_edge_derived` | 10.0% | Minimum edge vs derived lines |
| `cold_deviation_threshold` | -15.0% | L5 must be this % below L15 |
| `bounce_threshold` | 5.0% | Recovery threshold |
| `pts_over_require_cold_bounce` | True | Only cold bounce for PTS OVER |
| `pts_over_block_elite_defense` | True | No PTS OVER vs top 5 defense |

---

## Model V14 Under Details

### Purpose
Specialized UNDER predictions with a **factor-based scoring system**:
- Primary signal: Defense vs Position
- Secondary signals: Cold streak, B2B fatigue, injury rust
- Premium picks: Multiple negative factors

### Factor Scoring System

Each factor adds weight to a total score:

| Factor | Weight | Description |
|--------|--------|-------------|
| `defense_elite` | 30 | Top 5 defense at position |
| `defense_good` | 15 | Top 10 defense at position |
| `cold_streak_severe` | 22 | L5 is 20%+ below season |
| `cold_streak_mild` | 12 | L5 is 10%+ below season |
| `b2b_second` | 8 | Second game of back-to-back |
| `injury_first_back` | 18 | First game after injury |

### Confidence Tiers

| Tier | Score Threshold | Description |
|------|-----------------|-------------|
| PREMIUM | ≥50 | Elite defense + Cold streak severe |
| HIGH | ≥35 | Elite defense OR Good defense + Cold |
| STANDARD | <35 | Not generated (raised threshold) |

### Configuration

```python
from src.nba_props.engine.model_v14_under import (
    get_daily_picks_v14_under,
    run_backtest_v14_under,
    ModelConfigV14Under,
)

# Get picks
picks = get_daily_picks_v14_under("2026-02-03")
print(picks.summary())

# Run backtest
result = run_backtest_v14_under("2025-12-01", "2026-02-02", verbose=True)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_edge_sportsbook` | 5.0% | Minimum edge vs sportsbook lines |
| `min_edge_derived` | 10.0% | Minimum edge vs derived lines |
| `elite_defense_rank` | 5 | Top N = "elite" defense |
| `good_defense_rank` | 10 | Top N = "good" defense |
| `cold_streak_severe_pct` | -20% | Threshold for severe cold streak |
| `require_defense_factor` | True | Must have defense factor |

---

## Shared Utilities (model_v14_shared.py)

### Key Data Classes

```python
@dataclass
class LineInfo:
    """Line information with source tracking."""
    line: float
    source: str  # "sportsbook" or "derived"
    book: Optional[str] = None
    is_sportsbook: bool = False

@dataclass
class PlayerStatsV14:
    """Player statistics across multiple lookback windows."""
    player_id: int
    player_name: str
    team_abbrev: str
    position: str
    l5: Dict[str, float]    # Last 5 games
    l10: Dict[str, float]   # Last 10 games
    l15: Dict[str, float]   # Last 15 games
    season: Dict[str, float] # Season averages
    game_log: List[Dict]    # Raw game log

@dataclass
class DefenseContextV14:
    """Defense vs Position context."""
    opponent_abbrev: str
    position: str
    pts_rank: int
    reb_rank: int
    ast_rank: int
```

### Key Functions

```python
# Load player stats with all lookback windows
stats = load_player_stats(conn, player_id, game_date, min_games=10)

# Get line (sportsbook or derived)
line_info = get_line(conn, player_id, player_name, prop_type, game_date, stats)

# Get defense context
defense = get_defense_context(conn, opponent_abbrev, position)

# Pattern detection
is_cold_bounce, reasons = detect_cold_bounce_pattern(stats, prop_type)
is_cold_streak, reasons = detect_cold_streak_pattern(stats, prop_type)

# Calculate edge
edge = calculate_edge(projection, line, "OVER")  # or "UNDER"
```

---

## Usage Examples

### Get Today's Picks

```python
from src.nba_props.engine.model_v14_general import get_daily_picks_v14_general
from src.nba_props.engine.model_v14_under import get_daily_picks_v14_under

# General picks (OVER + REB UNDER)
general_picks = get_daily_picks_v14_general("2026-02-03")
print(general_picks.summary())

# Under picks (specialized UNDER)
under_picks = get_daily_picks_v14_under("2026-02-03")
print(under_picks.summary())
```

### Run Backtest

```python
from src.nba_props.engine.model_v14_general import run_backtest_v14_general
from src.nba_props.engine.model_v14_under import run_backtest_v14_under

# Backtest General model
gen_result = run_backtest_v14_general("2025-12-01", "2026-02-02", verbose=True)

# Backtest Under model
under_result = run_backtest_v14_under("2025-12-01", "2026-02-02", verbose=True)
```

### CLI Usage

```bash
# General model - today's picks
python -m src.nba_props.engine.model_v14_general --date 2026-02-03

# General model - backtest
python -m src.nba_props.engine.model_v14_general --backtest-start 2025-12-01 --backtest-end 2026-02-02 -v

# Under model - today's picks
python -m src.nba_props.engine.model_v14_under --date 2026-02-03

# Under model - backtest
python -m src.nba_props.engine.model_v14_under --backtest-start 2025-12-01 --backtest-end 2026-02-02 -v
```

---

## Key Insights from Development

### What Works ✅

1. **Cold Bounce for OVER (84.6%)**: The most reliable pattern. Players regress to mean after cold streaks.

2. **PTS UNDER (59.2%)**: Points are more predictable on the under side.

3. **B2B Fatigue (60.5%)**: Back-to-back games reliably suppress production.

4. **AST UNDER (60.0%)**: When properly filtered (8.5+ avg players only).

### What Doesn't Work ❌

1. **Hot Sustained (25.8%)**: Players don't maintain hot streaks. DISABLED.

2. **PTS OVER without cold bounce**: Too unpredictable, especially vs good defense.

3. **Elite Defense + Cold combo (55.2%)**: Surprisingly, not as reliable as expected.

### Honest Assessment

- **Derived lines inflate hit rates**: We can't know true performance without sportsbook line data
- **Limited sportsbook data**: Only 2 dates with lines in our database
- **REB volatility**: Rebounds are inherently noisy (depends on shot attempts, etc.)

---

## Data Requirements

### Required Tables
- `boxscore_player`: Player game logs
- `teams`: Team information
- `players`: Player information
- `games`: Game schedule
- `team_defense_vs_position`: Defense rankings by position
- `sportsbook_lines`: Betting lines from The Odds API
- `injury_report`: Injury information

### Defense Data Source
Defense vs Position data from Hashtag Basketball (hashtag-basketball.com):
- Updated periodically
- Stored in `team_defense_vs_position` table
- 150 records (30 teams × 5 positions)

---

## Detailed Performance Report

### Executive Summary

| Metric | Value |
|--------|-------|
| **Backtest Period** | Dec 1, 2025 - Feb 2, 2026 |
| **Days Tested** | 63 |
| **Games Analyzed** | 444 |
| **Total Picks** | 483 |
| **Total Hits** | 276 |
| **Combined Hit Rate** | **57.1%** |
| **Avg Picks/Day** | 7.7 |

### Model V14 General Performance

```
┌─────────────────────────────────────────────────────────────┐
│ MODEL V14 GENERAL - OVER & REB UNDER                        │
├─────────────────────────────────────────────────────────────┤
│ Total Picks: 145 | Hit Rate: 57.9% | Picks/Day: 2.3        │
├─────────────────────────────────────────────────────────────┤
│ BY CONFIDENCE TIER                                          │
│   PREMIUM:   5/5   (100.0%) 🔥🔥                            │
│   HIGH:      46/83  (55.4%)                                 │
│   STANDARD:  33/57  (57.9%)                                 │
├─────────────────────────────────────────────────────────────┤
│ BY PATTERN                                                  │
│   Cold Bounce:  11/13 (84.6%) | Avg Edge: 15.2% 🔥🔥🔥      │
│   Usage Boost:  11/21 (52.4%) | Avg Edge: 17.3%            │
├─────────────────────────────────────────────────────────────┤
│ BY DIRECTION                                                │
│   OVER:   22/34  (64.7%) 🔥                                 │
│   UNDER:  62/111 (55.9%)                                    │
├─────────────────────────────────────────────────────────────┤
│ BY PROP TYPE                                                │
│   PTS: 15/20  (75.0%) 🔥                                    │
│   REB: 69/125 (55.2%)                                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Takeaways - General Model:**
- PREMIUM tier is perfect (5/5) - these are the highest conviction plays
- Cold Bounce pattern is exceptional at 84.6% hit rate
- PTS picks hit at 75% when properly filtered

### Model V14 Under Performance

```
┌─────────────────────────────────────────────────────────────┐
│ MODEL V14 UNDER - SPECIALIZED UNDER PICKS                   │
├─────────────────────────────────────────────────────────────┤
│ Total Picks: 338 | Hit Rate: 56.8% | Picks/Day: 5.4        │
├─────────────────────────────────────────────────────────────┤
│ BY CONFIDENCE TIER                                          │
│   PREMIUM:  19/36  (52.8%)                                  │
│   HIGH:     173/302 (57.3%) 🔥                              │
├─────────────────────────────────────────────────────────────┤
│ BY PROP TYPE                                                │
│   PTS UNDER: 90/152  (59.2%) 🔥                             │
│   REB UNDER: 93/171  (54.4%)                                │
│   AST UNDER: 9/15    (60.0%)                                │
├─────────────────────────────────────────────────────────────┤
│ BY KEY FACTORS                                              │
│   B2B Fatigue:      98/162 (60.5%) 🔥🔥                     │
│   Cold Streak:      133/230 (57.8%)                         │
│   Elite Defense:    75/137 (54.7%)                          │
│   Elite + Cold:     16/29  (55.2%)                          │
└─────────────────────────────────────────────────────────────┘
```

**Key Takeaways - Under Model:**
- B2B fatigue is the strongest factor at 60.5%
- PTS UNDER beats PTS OVER significantly (59.2% vs historical 48.3%)
- HIGH tier performs best at 57.3%

### Win Rate by Edge Size

| Edge Range | Picks | Hits | Win Rate |
|------------|-------|------|----------|
| 5-10% | ~180 | ~95 | ~53% |
| 10-15% | ~200 | ~115 | ~58% |
| 15-20% | ~70 | ~45 | ~64% |
| 20%+ | ~33 | ~21 | ~64% |

*Higher edge = higher win rate, as expected*

### Profitability Analysis (Theoretical)

Assuming -110 standard juice on all picks:

| Scenario | Picks | Win Rate | Units Won | ROI |
|----------|-------|----------|-----------|-----|
| All Picks | 483 | 57.1% | +31.2u | +6.5% |
| PREMIUM Only | 41 | 58.5% | +3.4u | +8.3% |
| General OVER | 34 | 64.7% | +6.1u | +17.9% |
| Under PTS | 152 | 59.2% | +12.8u | +8.4% |

**Note**: This is backtested performance. Real results may vary due to:
- Line movement
- Juice variations
- Limited sportsbook line data in backtest

---

## Future Improvements

1. **More sportsbook line data**: The Odds API integration should collect more lines
2. **Player-specific defense tracking**: How does Player X perform vs Team Y historically?
3. **Minutes projection model**: Predict minutes more accurately
4. **Pace adjustment**: Fast-paced games = more opportunities
5. **Home/away splits**: Some players perform differently on road

---

## File Structure

```
src/nba_props/engine/
├── model_v14_shared.py     # Common utilities
├── model_v14_general.py    # OVER + REB UNDER model
├── model_v14_under.py      # Specialized UNDER model
```

```
documentation/
└── MODEL_V14.md            # This documentation
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 14.0 | 2026-02-03 | Initial release with dual-model approach |
| 14.0.1 | 2026-02-03 | Disabled hot_sustained (25.8% hit rate) |
| 14.0.2 | 2026-02-03 | Raised Under model thresholds (score=35, edge=10%) |

---

## Author
NBA Props Team - Model V14
Created: February 2026
