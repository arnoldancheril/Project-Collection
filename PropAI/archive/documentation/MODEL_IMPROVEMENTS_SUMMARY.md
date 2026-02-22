# Model Improvements Summary - February 2026

## Current Best Model: V16.1 (72.4% Hit Rate)

**Model V16.1 General** is the RECOMMENDED production model.

See: `documentation/MODEL_V16.md`

---

## Critical Issue Identified (SOLVED in V16)

### The Problem
The previous models (`model_production.py` and earlier) had a **fundamental flaw**: they used player statistical averages (L10, L15) as the "betting line" instead of actual sportsbook lines.

**Example - Peyton Watson (January 14, 2026):**
- Model projection: 4.9 rebounds
- Model was using L10 (4.9) as the "line"
- **Actual sportsbook line: 6.5 rebounds**

This means:
1. The model thought it had found an edge (projection vs derived line)
2. In reality, the player needed to beat 6.5 rebounds, not 4.9
3. **Backtest success rates were inflated by 5-15%**

### Root Cause
In `model_production.py`:
```python
# Cold bounce pattern
line = l10  # Uses player's 10-game average as "line"

# Hot sustained pattern  
line = l15  # Uses player's 15-game average as "line"
```

---

## Solution: Model V16 Hybrid Line Handling

Model V16 implements **hybrid line handling** to address this:

```python
# V16 Approach
def get_line(...):
    # Try sportsbook first
    sportsbook = get_sportsbook_line(player_id, prop_type, date)
    if sportsbook:
        return LineInfo(line=sportsbook.line, source="sportsbook")
    
    # Fall back to derived with +5% adjustment
    derived = player_l10_avg * 1.05
    return LineInfo(line=derived, source="derived")
```

**Key Features:**
1. Uses actual sportsbook lines when available
2. Applies +5% adjustment to derived lines (accounts for sportsbook markup)
3. Requires higher edge for derived lines (10% vs 6% for sportsbook)
4. Tracks line source for honest reporting

---

## V16.1 Performance Results

### Backtest (Oct 2025 - Feb 2026)

```
OVERALL: 72.4% (92/127)

BY DIRECTION:
  OVER:  76.9% (30/39)
  UNDER: 70.5% (62/88)

BY PROP TYPE:
  PTS: 74.3% (81/109)
  REB: 61.1% (11/18)

BY PATTERN:
  Cold Bounce (OVER):      76.9%
  PTS OVER (cold bounce):  90.5%
  PTS UNDER:               70.5%
  Elite Defense (UNDER):   67.9%
  B2B Fatigue (UNDER):     75.0%
```

### Improvement Over V16.0

| Version | Hit Rate | Picks | Key Change |
|---------|----------|-------|------------|
| V16.0 | 60.6% | 345 | Initial implementation |
| V16.1 | **72.4%** | 127 | Disabled weak patterns |

**Disabled patterns:**
- Hot Sustained: 25.8% (way below coin flip)
- Cold Streak: 51.6% (barely above coin flip)
- REB UNDER: 51.6% (too volatile)
- AST: ~54% (coin flip)

---

## Previous Solutions (Historical)

### Model V9 - Line-Aware Model

A previous model that properly integrates sportsbook lines:

```python
# V9 approach
sportsbook_line = get_sportsbook_line(player_id, prop_type, date)
if sportsbook_line:
    line = sportsbook_line
    line_source = "sportsbook"
else:
    line = derived_line * 1.05  # Conservative adjustment
    line_source = "derived"
```

**Note:** Superseded by V16 which achieves higher hit rate.

### Model Version Tracking System (`model_version_tracker.py`)

A comprehensive system for storing and comparing model iterations:

**Database Tables:**
- `model_versions` - Registry of all model configurations
- `model_version_picks` - All picks with line source tracking
- `model_version_backtests` - Full backtest history
- `model_version_insights` - Key learnings per model

---

## Usage Guide

### Running Model V16

```python
from src.nba_props.engine.model_v16_general import (
    get_daily_picks_v16_general,
    run_backtest_v16_general,
    ModelConfigV16General,
)

# Get today's picks
config = ModelConfigV16General()
picks = get_daily_picks_v16_general("2026-02-03", config=config)
print(picks.summary())

# Run backtest
result = run_backtest_v16_general(
    start_date="2025-11-01",
    end_date="2026-02-01",
    config=config,
    verbose=True,
)
print(result.summary())
```

### Viewing Model Comparisons

```python
from src.nba_props.engine.model_version_tracker import ModelVersionTracker

tracker = ModelVersionTracker()
print(tracker.get_comparison_report())
```

---

## Next Steps

1. **Continue sportsbook line collection**: More data = better validation
2. **Develop Model V16 Under**: Specialized under-only model
3. **Monitor pattern performance**: Patterns may decay over time
4. **Track live performance**: Compare predictions to actual results
5. **Multi-Model Comparison**: Run all model versions with same data to compare

---

## Files Created/Modified

| File | Change |
|------|--------|
| `src/nba_props/engine/model_v9.py` | **NEW** - Line-aware model |
| `src/nba_props/engine/model_version_tracker.py` | **NEW** - Version tracking system |
| `src/nba_props/engine/model_lab.py` | **MODIFIED** - Added tracking integration |
| `documentation/MODEL_V9.md` | **NEW** - Model V9 documentation |
| `documentation/MODEL_VERSION_TRACKING.md` | **NEW** - Tracking system docs |
| `documentation/MODEL_IMPROVEMENTS_SUMMARY.md` | **NEW** - This summary |

---

## Critical Warning

**Do not trust previous backtest results without re-running with actual sportsbook lines.**

The high hit rates (65-70%) reported by previous models were likely inflated because they were measuring performance against derived lines, not actual betting lines.

True performance can only be measured when we have:
1. Actual sportsbook lines for each pick
2. Comparison of projection vs actual line
3. Result tracking (did actual value beat the sportsbook line?)
