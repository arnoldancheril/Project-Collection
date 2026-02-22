# NBA Props Predictor - File Structure Documentation

## Overview

This document provides a comprehensive overview of the NBA Props Predictor codebase structure, explaining the purpose of each module and how they interact.

---

## Directory Structure

```
PropAI/
├── archive/                    # Archived legacy code and documentation
│   ├── documentation/          # Old reports and docs (REPORT.md, etc.)
│   └── models/                 # Legacy models (v2-v8)
│
├── bin/
│   └── normalizer              # Binary utility
│
├── data/                       # Data storage directory
│   ├── db/                     # SQLite database files
│   │   └── nba_props.sqlite3   # Main database with games, players, stats
│   ├── exports/                # Exported reports and CSV data
│   ├── raw/                    # Raw data files from pastes/scrapes
│   │   ├── boxscores/          # Game box score files (organized by date)
│   │   ├── lines/              # Sportsbook lines data
│   │   └── metadata/           # Salaries, standings, team stats
│   └── data_entry.txt          # Data entry scratchpad
│
├── documentation/              # Project documentation
│   ├── model_documentation/    # Specific documentation for Model Versions
│   │   ├── MODEL_V9.md         # Line-aware model
│   │   ├── MODEL_V10.md        # Market-aware model
│   │   ├── MODEL_V14.md
│   │   ├── MODEL_V15.md
│   │   ├── MODEL_V16.md        # (Key Model) V16 Documentation
│   │   ├── MODEL_V16_UNDER.md
│   │   ├── MODEL_V17.md
│   │   ├── MODEL_V18.md
│   │   ├── MODEL_V19.md
│   │   ├── REGRESSION_CONTRIBUTION_MODEL.md
│   │   └── UNDER_MODEL_V2_GUIDE.md
│   │
│   ├── 0_FILE_STRUCTURE.md     # This file
│   ├── 1_MODEL_SUMMARY.md      # High-level model comparison and summary
│   ├── COMPREHENSIVE_BACKTESTING.md
│   ├── DATA_AND_BACKTESTING_GUIDE.md
│   ├── GAME_MANAGEMENT.md
│   ├── INJURY_PARSER.md
│   ├── MODEL_VERSION_TRACKING.md
│   ├── SPORTSBOOK_LINES_GUIDE.md
│   └── ... (various guides and notes)
│
├── Sample Data/                # Example input files for testing parsers
│
├── scripts/                    # Standalone utility scripts
│   └── analyze_lines.py
│
├── src/
│   └── nba_props/              # Main Python package
│       ├── __init__.py
│       ├── __main__.py         # Entry point (python -m nba_props)
│       ├── cli.py              # CLI command handling
│       ├── db.py               # Database connection and queries
│       ├── paths.py            # Path configuration
│       ├── standings.py        # Conference standings logic
│       ├── team_aliases.py     # Team name normalization
│       ├── util.py             # Shared utilities
│       ├── validation.py       # Data validation
│       │
│       ├── engine/             # Core Modeling & Projection Engine
│       │   ├── __init__.py
│       │   ├── accuracy_tracker.py
│       │   ├── alerts.py               # Edge alerts and notifications
│       │   ├── archetype_db.py         # Player archetype database
│       │   ├── archetypes.py           # Archetype definitions
│       │   ├── backtesting.py          # Backtesting engine
│       │   ├── comprehensive_backtester.py # Multi-model backtesting
│       │   ├── edge_calculator.py      # Edge analysis logic
│       │   ├── edge_engine.py
│       │   ├── enhanced_model.py
│       │   ├── game_context.py         # Context (B2B, Home/Away)
│       │   ├── hybrid_model.py
│       │   ├── line_backtester.py
│       │   ├── line_projector.py
│       │   ├── matchup_advisor.py      # Matchup analysis logic
│       │   ├── minutes_projection.py
│       │   ├── model_final.py          # Final production model selector
│       │   ├── model_lab.py            # Experimental model framework
│       │   ├── model_production.py
│       │   ├── model_registry.py       # Registry of available models
│       │   ├── model_version_tracker.py
│       │   ├── multi_model_picker.py
│       │   ├── optimization.py
│       │   ├── projected_line_integration.py
│       │   ├── projector.py            # Base projection logic
│       │   ├── regression_contribution_model.py
│       │   ├── roster.py               # Player rosters
│       │   ├── under_model_v2.py       # Specialized Under model
│       │   ├── under_picks_analyzer.py
│       │   ├── unified_picks.py
│       │   ├── usage_redistribution.py
│       │   │
│       │   │ #--- Model Versions ---
│       │   ├── model_v9.py             # Baseline Line-Aware
│       │   ├── model_v10.py            # Market-Aware
│       │   ├── model_v12_*.py          # V12 Family (general, under, shared)
│       │   ├── model_v13_*.py          # V13 Family
│       │   ├── model_v14_*.py          # V14 Family
│       │   ├── model_v15_*.py          # V15 Family
│       │   ├── model_v16_*.py          # V16 Family (Current Standard)
│       │   ├── model_v17_*.py          # V17 Family
│       │   ├── model_v18_*.py          # V18 Family
│       │   └── model_v19_*.py          # V19 Family
│       │
│       ├── ingest/             # Data Ingestion & Parsing
│       │   ├── __init__.py
│       │   ├── boxscore_ingest.py      # Database insertion for boxscores
│       │   ├── boxscore_parser.py      # Text parser for boxscores
│       │   ├── defense_position_parser.py
│       │   ├── injury_parser.py
│       │   ├── lines_parser.py
│       │   ├── lines_scraper.py
│       │   ├── matchups_parser.py
│       │   ├── odds_api_client.py      # External Odds API integration
│       │   ├── optimized_lines.py
│       │   ├── paste.py                # Paste handling utility
│       │   ├── player_drtg_parser.py
│       │   ├── salary_parser.py
│       │   ├── team_stats_ingest.py
│       │   ├── team_stats_parser.py
│       │   └── web_scraper.py
│       │
│       └── web/                # Web Interface (Flask)
│           ├── __init__.py
│           ├── app.py          # Flask App Entry
│           ├── static/
│           │   ├── css/
│           │   └── js/
│           └── templates/      # HTML Templates
│               ├── backtesting.html
│               ├── index.html
│               ├── modellab.html
│               ├── projections.html
│               └── ... (other templates)
│
├── analysis_output.txt         # Output buffer for analysis tools
├── cleanup_script.py           # Database maintenance
├── data_entry.txt              # Root data entry file
├── pyproject.toml              # Project configuration
├── README.md                   # Main README
├── requirements.txt            # Dependency list
│
├── run_cli.py                  # Main CLI entry point
├── run_comprehensive_model_analysis.py  # Analysis runner
├── run_model_backtest_with_projected_lines.py
└── run_v16_backtest.py         # Specific backtest runner
```

## Loose Dependencies

The root directory also contains installed package folders (e.g., `requests/`, `urllib3/`, `idna/`, `certifi/`). These are dependencies and not part of the core project source code.

## Key Modules

### `src/nba_props/engine`
This is the core of the application. It contains the logic for:
- **Projector**: Base class for projecting player stats.
- **Models (v9-v19)**: Specific implementations of projection logic.
  - Generational models (e.g., `model_v16_general.py`, `model_v16_under.py`) allow for versioning and comparison.
- **Matchup Advisor**: Analyzes defensive matchups.
- **Backtesting**: Framework for testing models against historical data.

### `src/nba_props/ingest`
Handles all data input, whether from scraping, API calls, or parsing raw text pastes:
- `boxscore_parser.py`: Critical for converting raw game text into structured data.
- `lines_parser.py`: Ingests sportsbook lines.

### `src/nba_props/web`
A Flask-based web interface to visualize:
- Projections (`projections.html`)
- Model Lab experiments (`modellab.html`)
- Backtesting results (`backtesting.html`)

---

*Last Updated: February 4, 2026*
