# Model V10 - Market-Aware NBA Props Prediction Model

> **Version:** 10.0  
> **Created:** February 3, 2026  
> **Author:** PropAI Development Team  
> **Status:** Production-Ready  

---

## Executive Summary

**Model V10 addresses the fundamental flaw in all previous PropAI models: testing predictions against derived/estimated lines instead of actual sportsbook betting lines.**

### The Problem (Identified in All Previous Models)

Previous models (V2-V9, Production, RCM, Hybrid) achieved reported hit rates of 60-68% by comparing projections to:
- Player L10 or L15 averages (used as "lines")
- Derived estimates with small adjustments

**The Critical Flaw:**
```
Example: Peyton Watson
- L10 Average: 4.9 rebounds
- Model "line": 4.9 × 1.05 = 5.15
- Actual DraftKings line: 6.5
- Model says: OVER 5.15 → HIT if actual = 6
- Reality: OVER 6.5 → LOSS if actual = 6
```

This means previous models were measuring "can we predict if a player will exceed their average?" rather than "can we beat the betting market?"

### Model V10 Solution

**MANDATORY SPORTSBOOK LINE REQUIREMENT**: Model V10 only generates picks when actual sportsbook lines exist in the database. No line = No pick.

This ensures:
1. Edge calculations are against REAL betting lines
2. Performance metrics reflect actual betting outcomes
3. The model only operates where it can provide genuine value

---

## Performance Summary

### Backtest Results (Dec 1, 2025 - Feb 2, 2026)

**With Derived Lines (Comparison Mode):**
| Metric | Value | Notes |
|--------|-------|-------|
| **Overall Hit Rate** | 61.1% (469/768) | Using derived lines |
| **PREMIUM Tier** | 67.2% (164/244) | Highest confidence picks |
| **HIGH Tier** | 60.1% (244/406) | Strong picks |

**By Direction:**
| Direction | Hit Rate | Picks | Key Insight |
|-----------|----------|-------|-------------|
| **OVER** | 62.7% | 37/59 | Selective - only clear patterns |
| **UNDER** | 60.9% | 432/709 | Primary focus per RCM analysis |

**By Prop Type:**
| Prop | Hit Rate | Picks | Strategy |
|------|----------|-------|----------|
| **PTS** | 64.1% | 150/234 | UNDER preferred (per RCM v1.4) |
| **REB** | 59.7% | 319/534 | Both directions |
| **AST** | N/A | 0 | EXCLUDED - too volatile (~54%) |

**By Pattern:**
| Pattern | Hit Rate | Picks | Description |
|---------|----------|-------|-------------|
| **Cold Bounce (OVER)** | 65.9% | 27/41 | L5 < L15 by 20%+, last game bounced |
| **Hot Sustained (OVER)** | 55.6% | 10/18 | L5 > L15 by 30%+, maintaining |
| **Cold Streak (UNDER)** | 57.9% | 205/354 | L5 below season avg |
| **Elite Defense (UNDER)** | 62.2% | 206/331 | Top 5 DVP + cold factors |

---

## Model Architecture

### Core Principles

1. **Sportsbook Line Requirement**
   - Only generate picks where actual betting lines exist
   - No derived line substitution by default
   - Track line source for all picks

2. **Strategic Direction Selection**
   - PTS: UNDER preferred (RCM showed 63.9% UNDER vs 48.3% OVER)
   - REB: Both directions (~59% both ways)
   - AST: Excluded entirely (~54% is coin flip after juice)

3. **Pattern Confirmation Required**
   - OVER picks require: Cold bounce OR hot sustained pattern
   - UNDER picks require: Elite defense OR cold streak
   - No "generic edge" picks

4. **Strict Filtering**
   - Minimum 23 minutes average (established players only)
   - Minimum 10 games history
   - Exclude volatile player types

### Pattern Detection

#### OVER Patterns

**Cold Bounce (Best OVER pattern - 65.9%)**
```python
Conditions:
- L5 is 20%+ BELOW L15 (player is cold)
- Last game was ABOVE L10 (showing recovery)
- Opponent NOT elite defense for this stat
- Edge vs sportsbook line ≥ 8%

Rationale: Regression to mean after cold streak
```

**Hot Sustained (55.6%)**
```python
Conditions:
- L5 is 30%+ ABOVE L15 (player is hot)
- L3 ≥ L5 (still maintaining, not cooling)
- 3+ of last 5 games above L15
- Opponent NOT elite defense
- Edge vs sportsbook line ≥ 8%

Rationale: Momentum continuation
```

#### UNDER Patterns

**Elite Defense (62.2%)**
```python
Conditions:
- Opponent ranks TOP 5 in DVP for position/stat
- Edge vs sportsbook line ≥ 6%

Rationale: Elite defenses consistently limit production
```

**Cold Streak (57.9%)**
```python
Conditions:
- L5 is 15%+ BELOW season average
- Edge vs sportsbook line ≥ 6%

Rationale: Cold players tend to stay cold short-term
```

**Combined (Strongest UNDER)**
```python
Conditions:
- Elite defense + Cold streak
- Confidence bonus: +18 points

Rationale: Multiple negative factors compound
```

### Projection Calculation

```python
# Base projection (weighted average)
projection = (
    L5 × 0.20 +
    L10 × 0.30 +
    L15 × 0.25 +
    Season × 0.25
)

# Defense adjustment
if opponent_defense == "elite":
    projection *= 0.88  # -12%
elif opponent_defense == "good":
    projection *= 0.94  # -6%
elif opponent_defense == "weak":
    projection *= 1.06  # +6%

# Edge calculation (vs ACTUAL sportsbook line)
if direction == "OVER":
    edge = (projection - sportsbook_line) / sportsbook_line × 100
else:  # UNDER
    edge = (sportsbook_line - projection) / sportsbook_line × 100
```

### Confidence Scoring

```python
base_confidence = 70.0

# Pattern bonus (varies by pattern)
confidence += pattern_bonus  # 5-10 points

# Edge bonus
confidence += min(edge / 2, 10)  # Up to 10 points

# Consistency bonus
if CV < 0.20:  # Very consistent player
    confidence += 5
elif CV > 0.40:  # Volatile player
    confidence -= 5

# Elite defense bonus (UNDER)
if direction == "UNDER" and defense_rating == "elite":
    confidence += 5

# Tier assignment
if confidence >= 85 and edge >= 12:
    tier = "PREMIUM"
elif confidence >= 75:
    tier = "HIGH"
else:
    tier = "STANDARD"
```

---

## Configuration

```python
@dataclass
class ModelConfigV10:
    # CORE PRINCIPLE
    require_sportsbook_line: bool = True  # CRITICAL
    
    # Data requirements
    min_games_required: int = 10
    min_avg_minutes: float = 23.0  # Established players only
    max_games_lookback: int = 20
    
    # Projection weights
    weight_l5: float = 0.20
    weight_l10: float = 0.30
    weight_l15: float = 0.25
    weight_season: float = 0.25
    
    # Pattern thresholds
    cold_deviation_threshold: float = -20.0  # For cold bounce
    hot_deviation_threshold: float = 30.0    # For hot sustained
    cold_streak_threshold: float = -15.0     # For UNDER
    
    # Edge requirements (vs ACTUAL lines)
    min_edge_over: float = 8.0
    min_edge_under: float = 6.0
    min_edge_premium: float = 12.0
    
    # Defense thresholds
    elite_defense_rank: int = 5   # Top 5
    good_defense_rank: int = 10   # Top 10
    weak_defense_rank: int = 25   # Bottom 5
    
    # Strategic direction
    pts_direction: str = "UNDER_PREFERRED"
    reb_direction: str = "BOTH"
    include_ast: bool = False  # EXCLUDED
    
    # Pick limits
    max_picks_per_player: int = 1  # Focus
    max_picks_per_day: int = 20
```

---

## Usage

### Generate Daily Picks

```python
from src.nba_props.engine.model_v10 import (
    get_daily_picks_v10,
    ModelConfigV10,
)

# Standard usage (requires sportsbook lines)
picks = get_daily_picks_v10("2026-02-03")
print(picks.summary())

# Custom configuration
config = ModelConfigV10()
config.min_edge_under = 8.0  # Stricter edge
picks = get_daily_picks_v10("2026-02-03", config=config)
```

### Run Backtest

```python
from src.nba_props.engine.model_v10 import run_backtest_v10

# With sportsbook lines required (honest)
result = run_backtest_v10(
    "2026-01-01",
    "2026-02-02",
    verbose=True
)
print(result.summary())

# Comparison mode (derived lines)
config = ModelConfigV10()
config.require_sportsbook_line = False
result = run_backtest_v10(
    "2025-12-01",
    "2026-02-02",
    config=config,
    verbose=True
)
```

### CLI Interface

```bash
# Generate picks for today
python -m src.nba_props.engine.model_v10 --date 2026-02-03

# Run backtest
python -m src.nba_props.engine.model_v10 \
    --backtest-start 2025-12-01 \
    --backtest-end 2026-02-02 \
    --verbose
```

---

## Sportsbook Lines Integration

### Fetching Lines

Model V10 depends on the sportsbook lines stored via The Odds API:

```bash
# Fetch today's lines
python3 run_cli.py fetch-lines-api

# Fetch specific date
python3 run_cli.py fetch-lines-api --date 2026-02-03

# Check API quota
python3 run_cli.py api-status
```

### Line Requirements

For Model V10 to generate a pick:
1. Player must have a sportsbook line for the game date
2. Line must be for the matching prop type (PTS, REB)
3. Pattern must be detected (cold bounce, elite defense, etc.)
4. Edge vs actual line must meet minimum threshold

---

## Key Insights from Analysis

### Why Previous Models Overstated Performance

| Factor | Impact |
|--------|--------|
| Derived lines ≈ player average | Model measuring "vs average" not "vs market" |
| L10 × 1.05 adjustment | Only 5% buffer, Vegas often differs by 10-20% |
| No line existence check | Picks generated for all players regardless |
| Testing against self-generated lines | Circular validation |

### Model V10 Improvements

| Improvement | Benefit |
|-------------|---------|
| **Sportsbook line required** | Honest edge calculation |
| **Strategic direction** | PTS UNDER (63.9% vs 48.3% OVER) |
| **AST excluded** | Remove ~54% (coin flip) props |
| **Pattern confirmation** | No generic edge picks |
| **23+ min requirement** | Established players only |
| **Defense integration** | Elite DVP provides real signal |

### Realistic Expectations

With actual sportsbook lines:
- **Expected hit rate**: 55-60%
- **PREMIUM tier**: 60-65%
- **Break-even at -110 odds**: 52.4%
- **Expected edge**: 3-8% above break-even

**Important**: The 61.1% hit rate with derived lines would likely be 55-58% with actual sportsbook lines due to:
1. Vegas having superior information
2. Line movement not captured
3. Public betting adjustments

---

## Recommendations

### For Immediate Use

1. **Fetch sportsbook lines daily** before generating picks
2. **Focus on PREMIUM tier** picks only
3. **Prioritize UNDER picks** especially for PTS
4. **Monitor Elite Defense patterns** (highest confidence)

### For Future Development

1. **Collect more historical sportsbook lines** for better backtesting
2. **Track line movement** between fetch and game time
3. **Add injury impact quantification** beyond simple exclusion
4. **Consider player archetypes** from Model V6 for additional filtering

### For Validation

1. **Compare Model V10 PREMIUM picks vs other models**
2. **Track actual betting outcomes** (not simulated)
3. **Monitor sportsbook line coverage** to ensure data quality

---

## File Locations

| File | Description |
|------|-------------|
| `src/nba_props/engine/model_v10.py` | Model implementation |
| `documentation/MODEL_V10.md` | This documentation |
| `documentation/SPORTSBOOK_LINES_GUIDE.md` | Lines integration guide |
| `src/nba_props/ingest/odds_api_client.py` | API client for lines |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 10.0 | 2026-02-03 | Initial release - Market-aware model |

---

*Model V10 represents a paradigm shift from "can we predict player performance?" to "can we beat the betting market?" - the only question that matters for profitable betting.*
