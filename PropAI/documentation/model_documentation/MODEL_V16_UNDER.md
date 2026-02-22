# Model V16.5 Under - NBA Props UNDER Prediction Model

## Overview

Model V16.5 Under is a **specialized UNDER-only model** designed as part of the dual-model approach:
- **Model V16 General**: Handles OVER picks (cold bounce) + UNDER via B2B fatigue
- **Model V16.5 Under** (this model): Dedicated UNDER prediction with factor-based scoring

## GUI Integration

The V16 Under model is fully integrated into the web interface:
- **Model Performance Tab**: Runs alongside V16 General, picks labeled with `[V16 Under]` source
- **Matchups Tab**: V16 Under picks appear with `factor_based` pattern label
- **Pick Details**: Shows all contributing factors (defense, cold streak, B2B, injury rust)

## Backtest Results (2025-10-22 to 2026-02-03)

```
======================================================================
MODEL V16.5 UNDER - BACKTEST RESULTS
======================================================================
Period: 2025-10-22 to 2026-02-03
Days: 103 | Games: 752
Avg picks/day: 1.7

OVERALL: 60.6% (103/170)

BY LINE SOURCE (Honest Reporting):
  Sportsbook lines: 0.0% (0/1)
  Derived lines:    60.9% (103/169)

BY CONFIDENCE TIER:
  PREMIUM (score ≥50): 84.6% (11/13)   ← BEST TIER
  HIGH (score 35-49):  58.8% (47/80)
  STANDARD (score 25-34): 58.4% (45/77)

BY PROP TYPE:
  PTS UNDER: 60.6% (103/170)
  REB UNDER: N/A (disabled by default)

BY PRIMARY FACTOR:
  Elite Defense (rank 1-5):  68.9% (42/61)   ← PRIMARY SIGNAL
  Good Defense (rank 6-10):  65.9% (27/41)
  Cold Streak Severe (-20%): 51.7% (30/58)   ← Weak alone!
  Cold Streak Mild (-10%):   60.0% (21/35)
  B2B Fatigue:               63.6% (28/44)
  Injury Rust:               66.7% (6/9)
  Combined Elite+Cold:       83.3% (10/12)   ← BEST COMBO

BY FACTOR SCORE BUCKET:
  Score ≥50 (Premium):  84.6% (11/13)   ← FOCUS HERE
  Score 35-49 (High):   58.8% (47/80)
  Score 25-34 (Std):    58.4% (45/77)
======================================================================
```

## Key Insights from Backtesting

### 1. Defense is the PRIMARY Factor
- Elite Defense (rank 1-5): **68.9%** hit rate
- Good Defense (rank 6-10): **65.9%** hit rate
- Defense alone is a stronger signal than cold streaks alone

### 2. Cold Streaks Need Defense Pairing
- Cold Streak Severe **alone**: Only **51.7%** (barely better than coin flip)
- Cold Streak Severe **+ Elite Defense**: **83.3%** (exceptional!)
- **Always pair cold streak signals with defense signals**

### 3. PREMIUM Tier is the Sweet Spot
- PREMIUM (score ≥50): **84.6%** hit rate with 13 picks
- These picks have **both** elite defense AND cold streak
- High confidence, smaller sample, but very reliable

### 4. B2B Fatigue is Reliable
- **63.6%** hit rate on B2B fatigue picks
- Consistent secondary factor
- Works well paired with defense

## Architecture

### Factor-Based Scoring System

The model calculates a **factor score** for each potential UNDER pick:

```python
FACTOR_WEIGHTS = {
    "defense_elite": 30,        # Rank 1-5 at position
    "defense_good": 15,         # Rank 6-10 at position
    "defense_average": 5,       # Rank 11-15
    "cold_streak_severe": 22,   # L5 < 80% of season
    "cold_streak_mild": 12,     # L5 < 90% of season
    "b2b_fatigue": 8,           # Second game of back-to-back
    "b2b_third_in_four": 5,     # Third game in 4 nights
    "injury_rust_first": 18,    # First game back
    "injury_rust_second": 12,   # Second game back
    "injury_rust_third": 6,     # Third game back
    "high_variance": 6,         # CV > 0.35
    "historical_struggle": 10,  # Poor vs opponent
    "blowout_risk": 5,          # Large spread
}
```

### Confidence Tiers

| Tier | Score | Example Combination | Hit Rate |
|------|-------|---------------------|----------|
| PREMIUM | ≥50 | Elite Defense (30) + Cold Severe (22) | 84.6% |
| HIGH | 35-49 | Elite Defense (30) + B2B (8) | 58.8% |
| STANDARD | 25-34 | Good Defense (15) + Cold Mild (12) | 58.4% |

### Projection Adjustments

Each factor also applies a **reduction** to the player's projection:

```python
FACTOR_ADJUSTMENTS = {
    "defense_elite": 0.88,      # 12% reduction
    "defense_good": 0.94,       # 6% reduction
    "cold_streak_severe": 0.86, # 14% reduction
    "cold_streak_mild": 0.93,   # 7% reduction
    "b2b_fatigue": 0.95,        # 5% reduction
    "injury_rust_first": 0.80,  # 20% reduction
    # ...
}
```

Example: Player with Elite Defense + Cold Streak Severe
- Base projection: 22.0 points
- Adjusted: 22.0 × 0.88 × 0.86 = **16.6 points**
- If line is 20.5: Edge = (20.5 - 16.6) / 16.6 = **23.5%**

## Hybrid Line Handling

### The Derived Line Fallacy

Previous models tested against **derived lines** (projections), not actual sportsbook lines. This inflated hit rates by 5-15% because:
1. Derived lines are optimistically biased
2. They don't account for market efficiency
3. True edge is smaller than calculated

### Model V16.5 Solution

1. **Always check for sportsbook lines first**
2. **Still generate picks without lines** (use derived with +5% adjustment)
3. **Track line source** for honest reporting
4. **Different edge requirements:**
   - Sportsbook lines: 5% minimum edge
   - Derived lines: 8% minimum edge

```python
config = ModelConfigV16Under(
    require_sportsbook_line=False,  # Never require
    derived_line_adjustment=1.05,   # +5% for derived
    min_edge_sportsbook=5.0,        # Lower bar for real lines
    min_edge_derived=8.0,           # Higher bar for derived
)
```

## Usage

### Generate Daily Picks

```python
from src.nba_props.engine.model_v16_under import (
    get_daily_picks_v16_under,
    ModelConfigV16Under,
)

# Default configuration
picks = get_daily_picks_v16_under("2026-02-03")
print(picks.summary())

# Focus only on PREMIUM picks
for pick in picks.premium_picks:
    print(pick.summary_line())
```

### Run Backtest

```python
from src.nba_props.engine.model_v16_under import run_backtest_v16_under

result = run_backtest_v16_under(
    start_date="2025-10-22",
    end_date="2026-02-03",
    verbose=True,
)
print(result.summary())
```

### CLI Usage

```bash
# Today's picks
python -m src.nba_props.engine.model_v16_under

# Specific date
python -m src.nba_props.engine.model_v16_under --date 2026-02-03

# Backtest
python -m src.nba_props.engine.model_v16_under --backtest --start 2025-10-22 --end 2026-02-03 -v
```

## Configuration Options

```python
@dataclass
class ModelConfigV16Under:
    # Prop selection
    include_pts_under: bool = True   # PTS UNDER (primary)
    include_reb_under: bool = False  # REB UNDER (disabled - too volatile)
    include_ast_under: bool = False  # AST UNDER (disabled - coin flip)
    
    # Defense requirements
    require_defense_data: bool = True
    max_defense_rank_for_under: int = 20  # No UNDER vs weak defense
    
    # Factor thresholds
    cold_streak_mild_threshold: float = -10.0   # L5 is 10%+ below season
    cold_streak_severe_threshold: float = -20.0 # L5 is 20%+ below season
    high_variance_threshold: float = 0.35       # CV > 0.35
    
    # Score thresholds
    premium_score_threshold: float = 50.0  # PREMIUM tier
    high_score_threshold: float = 35.0     # HIGH tier
    min_score_threshold: float = 25.0      # Minimum for any pick
    
    # Edge requirements
    min_edge_sportsbook: float = 5.0   # vs sportsbook line
    min_edge_derived: float = 8.0      # vs derived line
    
    # Limits
    max_picks_per_day: int = 20
    max_picks_per_player: int = 1
```

## What This Model Excludes

1. **REB UNDER** - Too volatile (~52-54% hit rate)
2. **AST UNDER** - Coin flip for most players
3. **UNDER vs Weak Defense** - Rank 21-30 excluded
4. **Low Factor Score** - Below 25 excluded
5. **Low Edge** - Below 5% (sportsbook) / 8% (derived)
6. **Garbage Time Players** - <23 min avg excluded
7. **Small Sample** - <10 games excluded

## Integration with V16 General

The dual-model approach:

1. **V16 General** runs first for all picks (OVER focused)
2. **V16.5 Under** runs for dedicated UNDER picks
3. Combine results, deduplicate by player
4. Prioritize PREMIUM tier picks from V16.5 Under

```python
from src.nba_props.engine.model_v16_general import get_daily_picks_v16_general
from src.nba_props.engine.model_v16_under import get_daily_picks_v16_under

# Get both
general = get_daily_picks_v16_general("2026-02-03")
under = get_daily_picks_v16_under("2026-02-03")

# Combine (V16.5 Under PREMIUM picks take priority)
all_picks = under.premium_picks + general.picks
```

## Recommendations

Based on backtesting results:

### Best Use Cases (PREMIUM Tier)
1. Elite defense (rank 1-5) + Cold streak severe
2. Elite defense + Injury rust (first game back)
3. Elite defense + B2B fatigue + Cold streak mild

### Good Use Cases (HIGH Tier)
1. Elite defense alone (68.9%)
2. Good defense + Cold streak severe
3. Elite defense + B2B fatigue

### Caution (STANDARD Tier)
1. Cold streak severe without defense (51.7%)
2. Good defense alone
3. Multiple weak factors

### Avoid
1. Cold streak without any defense signal
2. UNDER vs weak defense (rank 20+)
3. REB UNDER (disabled for a reason)

## Future Improvements

1. **More sportsbook line data** - Currently 99%+ derived lines
2. **Position-specific thresholds** - Guards vs Centers
3. **Pace adjustments** - Fast teams vs slow teams
4. **Spread integration** - Blowout risk factor refinement
5. **Opponent recent form** - Defense trending up/down

---

*Model V16.5 Under - February 2026*
*Overall: 60.6% | PREMIUM: 84.6%*
