# Comprehensive Model Backtesting System

This document describes the comprehensive backtesting system for all NBA Props prediction models.

## Overview

The system provides unified backtesting across 23+ prediction models with:
- **Quality scoring** that balances accuracy with pick volume
- **Terminal progress bars** for real-time progress tracking
- **GUI progress bars** for web interface monitoring
- **Strengths/weaknesses analysis** for each model
- **Best model recommendations** by category

---

## 1. Terminal Usage (CLI)

### Basic Comprehensive Backtest
Run a full backtest across all active models:
```bash
python run_cli.py comprehensive-backtest --weeks 8
```

### Compare Latest Models (v16-v19)
Quick comparison of the most recent model versions:
```bash
python run_cli.py comprehensive-backtest --latest --start 2025-12-01 --end 2026-01-15
```

### Compare UNDER-Specialized Models
Compare models optimized for UNDER picks:
```bash
python run_cli.py comprehensive-backtest --under --start 2025-12-01 --end 2026-01-15
```

### Filter by Category
Test only specific model categories:
```bash
# Test multi-file models (v12-v19)
python run_cli.py comprehensive-backtest --category multi --weeks 4

# Test single-file legacy models (v2-v10)
python run_cli.py comprehensive-backtest --category single --weeks 4

# Test specialized models (production, final, under_v2)
python run_cli.py comprehensive-backtest --category specialized --weeks 4
```

### Save Results to JSON
Export results for further analysis:
```bash
python run_cli.py comprehensive-backtest --latest --output results.json
```

### CLI Options
| Option | Description |
|--------|-------------|
| `--weeks N` | Number of weeks to backtest (default: 8) |
| `--start DATE` | Start date (YYYY-MM-DD), overrides --weeks |
| `--end DATE` | End date (YYYY-MM-DD) |
| `--category CAT` | Filter: multi, single, specialized |
| `--latest` | Compare only v16-v19 models |
| `--under` | Compare only UNDER-specialized models |
| `--verbose` | Show detailed output for each model |
| `--output FILE` | Save results to JSON file |

---

## 2. GUI Usage (Model Lab)

Access the comprehensive Model Lab at: **http://localhost:5050/modellab**

### Tabs

1. **🔬 Comprehensive Backtest**
   - Run analysis across all models
   - View progress bars during testing
   - See rankings by quality score
   - Get best model recommendations

2. **📋 All Models**
   - Browse all 23+ registered models
   - View by category (multi-file, single-file, specialized)
   - Click any model to run individual backtest

3. **📊 Quick Compare**
   - One-click comparison of model groups
   - "Compare Latest" for v16-v19
   - "Compare UNDER" for UNDER-specialized models
   - "Compare Legacy" for v2-v10 models

4. **🎯 Best Picks**
   - Best model for UNDER picks
   - Best model for OVER picks
   - Best model for PTS props
   - Best model for REB props
   - Models to avoid

5. **⚙️ Legacy Lab**
   - Original model lab functionality
   - Backward compatibility

---

## 3. Quality Score Explained

The quality score balances accuracy with pick volume, addressing the problem where models with very few picks might show high accuracy but are unreliable.

**Formula:**
```
quality_score = (hit_rate * 100) * math.log10(max(10, total_picks))
```

**Example Rankings:**
| Model | Hit Rate | Picks | Quality Score |
|-------|----------|-------|---------------|
| Model A | 60% | 100 | 120.0 |
| Model B | 80% | 10 | 80.0 |
| Model C | 55% | 500 | 148.6 |

Model C ranks highest despite lower hit rate because of its volume.

---

## 4. Model Categories

### Multi-File Models (v12-v19)
Latest architecture with separate files for different concerns:
- `model_v{N}_general.py` - General OVER/UNDER picks
- `model_v{N}_under.py` - Specialized UNDER picks  
- `model_v{N}_shared.py` - Shared utilities

### Single-File Models (v2-v10)
Legacy models with self-contained logic:
- `model_v2.py` through `model_v10.py`

### Specialized Models
- `model_production.py` - Production-ready model
- `model_final.py` - Final optimized model
- `under_model_v2.py` - UNDER specialist

---

## 5. API Endpoints

### List All Models
```
GET /api/modellab/models
```

### Run Comprehensive Backtest
```
POST /api/modellab/comprehensive-backtest
Body: { "start_date": "2025-12-01", "end_date": "2026-01-15" }
```

### Run Single Model Backtest  
```
POST /api/modellab/single-backtest
Body: { "model_id": "v18_general", "start_date": "2025-12-01", "end_date": "2026-01-15" }
```

### Compare Latest Models
```
POST /api/modellab/compare-latest
Body: { "start_date": "2025-12-01", "end_date": "2026-01-15" }
```

### Compare UNDER Models
```
POST /api/modellab/compare-under
Body: { "start_date": "2025-12-01", "end_date": "2026-01-15" }
```

---

## 6. Files Created

| File | Purpose |
|------|---------|
| `src/nba_props/engine/model_registry.py` | Central catalog of all models |
| `src/nba_props/engine/comprehensive_backtester.py` | Unified backtest runner |
| `src/nba_props/web/templates/modellab_comprehensive.html` | New GUI template |

---

## 7. Example Output

Terminal progress bar:
```
Testing Models |█████████████████████████---------------| 62.5% v17_general
```

Summary report:
```
╔══════════════════════════════════════════════════════════════════════════╗
║              COMPREHENSIVE MODEL BACKTEST RESULTS                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Test Period: 2025-12-01 to 2025-12-15                                  ║
║  Models Tested: 8 | Succeeded: 8 | Failed: 0                           ║
║  Total Runtime: 12.0s                                                    ║
╚══════════════════════════════════════════════════════════════════════════╝

  📊 TOP MODELS BY QUALITY SCORE
  ─────────────────────────────────────────────────────────────────────────
  🥇 # 1 Model V16 Under           | Hit:  84.0% | Quality: 154.8 | Picks:    25
  🥈 # 2 Model V16 General         | Hit:  82.4% | Quality: 145.6 | Picks:    17
  🥉 # 3 Model V18 Under           | Hit:  77.4% | Quality: 135.3 | Picks:    31
```
