from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Db:
    path: Path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS teams (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS players (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS games (
  id INTEGER PRIMARY KEY,
  season TEXT,
  game_date TEXT NOT NULL,
  team1_id INTEGER NOT NULL,
  team2_id INTEGER NOT NULL,
  source_file TEXT NOT NULL,
  source_hash TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (team1_id) REFERENCES teams(id),
  FOREIGN KEY (team2_id) REFERENCES teams(id)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);

CREATE TABLE IF NOT EXISTS boxscore_player (
  id INTEGER PRIMARY KEY,
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  pos TEXT,
  minutes REAL,
  pts INTEGER,
  reb INTEGER,
  ast INTEGER,
  oreb INTEGER,
  dreb INTEGER,
  stl INTEGER,
  blk INTEGER,
  tov INTEGER,
  pf INTEGER,
  fgm INTEGER,
  fga INTEGER,
  fg_pct REAL,
  tpm INTEGER,
  tpa INTEGER,
  tp_pct REAL,
  ftm INTEGER,
  fta INTEGER,
  ft_pct REAL,
  plus_minus INTEGER,
  raw_line TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(game_id, team_id, player_id),
  FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
  FOREIGN KEY (team_id) REFERENCES teams(id),
  FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS boxscore_team_totals (
  id INTEGER PRIMARY KEY,
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  pts INTEGER,
  reb INTEGER,
  ast INTEGER,
  oreb INTEGER,
  dreb INTEGER,
  stl INTEGER,
  blk INTEGER,
  tov INTEGER,
  pf INTEGER,
  fgm INTEGER,
  fga INTEGER,
  tpm INTEGER,
  tpa INTEGER,
  ftm INTEGER,
  fta INTEGER,
  plus_minus INTEGER,
  raw_line TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(game_id, team_id),
  FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
  FOREIGN KEY (team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS inactive_players (
  id INTEGER PRIMARY KEY,
  game_id INTEGER NOT NULL,
  team_id INTEGER NOT NULL,
  player_name TEXT NOT NULL,
  reason TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE,
  FOREIGN KEY (team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS sportsbook_lines (
  id INTEGER PRIMARY KEY,
  as_of_date TEXT NOT NULL,
  game_id INTEGER,
  team_id INTEGER,
  player_id INTEGER,
  prop_type TEXT NOT NULL, -- PTS, REB, AST
  line REAL NOT NULL,
  odds_american INTEGER,
  book TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (game_id) REFERENCES games(id),
  FOREIGN KEY (team_id) REFERENCES teams(id),
  FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS injury_report (
  id INTEGER PRIMARY KEY,
  game_date TEXT NOT NULL,
  team_id INTEGER NOT NULL,
  player_id INTEGER,
  player_name TEXT,
  status TEXT NOT NULL, -- OUT, Q, D, P, ACTIVE
  minutes_limit REAL,
  notes TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (team_id) REFERENCES teams(id),
  FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Team stats snapshots (season-to-date “team stats” tables you provide, per team).
CREATE TABLE IF NOT EXISTS team_stats_snapshot (
  id INTEGER PRIMARY KEY,
  season TEXT,
  as_of_date TEXT,
  team_id INTEGER NOT NULL,
  source_file TEXT NOT NULL,
  source_hash TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(team_id, season, as_of_date),
  FOREIGN KEY (team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS team_stats_player (
  id INTEGER PRIMARY KEY,
  snapshot_id INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pos TEXT,
  gp INTEGER,
  gs INTEGER,
  min REAL,
  pts REAL,
  oreb REAL,
  dreb REAL,
  reb REAL,
  ast REAL,
  stl REAL,
  blk REAL,
  tov REAL,
  pf REAL,
  ast_to REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(snapshot_id, player_id),
  FOREIGN KEY (snapshot_id) REFERENCES team_stats_snapshot(id) ON DELETE CASCADE,
  FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE TABLE IF NOT EXISTS team_stats_shooting (
  id INTEGER PRIMARY KEY,
  snapshot_id INTEGER NOT NULL,
  player_id INTEGER NOT NULL,
  pos TEXT,
  fgm REAL,
  fga REAL,
  fg_pct REAL,
  tpm REAL,
  tpa REAL,
  tp_pct REAL,
  ftm REAL,
  fta REAL,
  ft_pct REAL,
  twopm REAL,
  twopa REAL,
  twop_pct REAL,
  sc_eff REAL,
  sh_eff REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(snapshot_id, player_id),
  FOREIGN KEY (snapshot_id) REFERENCES team_stats_snapshot(id) ON DELETE CASCADE,
  FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Game betting lines (spread, over/under)
CREATE TABLE IF NOT EXISTS game_lines (
  id INTEGER PRIMARY KEY,
  game_date TEXT NOT NULL,
  away_team_id INTEGER NOT NULL,
  home_team_id INTEGER NOT NULL,
  spread REAL,  -- Home team spread (negative = home favored)
  over_under REAL,
  moneyline_away INTEGER,
  moneyline_home INTEGER,
  book TEXT DEFAULT 'consensus',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(game_date, away_team_id, home_team_id, book),
  FOREIGN KEY (away_team_id) REFERENCES teams(id),
  FOREIGN KEY (home_team_id) REFERENCES teams(id)
);

-- Player salary data
CREATE TABLE IF NOT EXISTS player_salaries (
  id INTEGER PRIMARY KEY,
  player_name TEXT NOT NULL,
  position TEXT,
  team TEXT NOT NULL,
  salary INTEGER NOT NULL,
  salary_rank INTEGER,
  season TEXT DEFAULT '2025-26',
  UNIQUE(player_name, season)
);

-- Team defensive ratings (for matchup analysis)
CREATE TABLE IF NOT EXISTS team_defense_ratings (
  id INTEGER PRIMARY KEY,
  team_id INTEGER NOT NULL,
  season TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  def_rating REAL,  -- Points allowed per 100 possessions
  pts_allowed_pg REAL,
  reb_allowed_pg REAL,
  ast_allowed_pg REAL,
  opp_fg_pct REAL,
  opp_3p_pct REAL,
  pace REAL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(team_id, season, as_of_date),
  FOREIGN KEY (team_id) REFERENCES teams(id)
);

-- Player archetypes stored in database (editable, not hard-coded)
CREATE TABLE IF NOT EXISTS player_archetypes (
  id INTEGER PRIMARY KEY,
  player_id INTEGER,
  player_name TEXT NOT NULL,  -- Allow entries even before player exists in players table
  team TEXT,
  season TEXT DEFAULT '2025-26',
  
  -- Position info
  position TEXT,  -- PG, SG, SF, PF, C
  height TEXT,  -- e.g., "6'6"
  
  -- Archetypes
  primary_offensive TEXT NOT NULL,
  secondary_offensive TEXT,
  defensive_role TEXT NOT NULL,
  
  -- Tier (1-6, 1=MVP candidate)
  tier INTEGER NOT NULL DEFAULT 5,
  
  -- Metadata
  is_elite_defender INTEGER DEFAULT 0,  -- Boolean as 0/1
  strengths TEXT,  -- JSON array
  weaknesses TEXT,  -- JSON array
  notes TEXT,
  guards_positions TEXT,  -- JSON array of positions this player guards
  avoid_betting_against TEXT,  -- JSON array of defender names
  
  -- Source tracking
  source TEXT DEFAULT 'manual',  -- 'manual', 'auto', 'seed'
  confidence REAL DEFAULT 1.0,  -- How confident in this classification
  
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  
  UNIQUE(player_name, season),
  FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE INDEX IF NOT EXISTS idx_player_archetypes_player ON player_archetypes(player_name);
CREATE INDEX IF NOT EXISTS idx_player_archetypes_team ON player_archetypes(team);
CREATE INDEX IF NOT EXISTS idx_player_archetypes_tier ON player_archetypes(tier);

-- Player similarity groups (editable)
CREATE TABLE IF NOT EXISTS player_similarity_groups (
  id INTEGER PRIMARY KEY,
  group_name TEXT NOT NULL,
  player_name TEXT NOT NULL,
  season TEXT DEFAULT '2025-26',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(group_name, player_name, season)
);

-- Elite defenders by position (editable)
CREATE TABLE IF NOT EXISTS elite_defenders (
  id INTEGER PRIMARY KEY,
  player_name TEXT NOT NULL,
  position TEXT NOT NULL,  -- PG, SG, SF, PF, C
  season TEXT DEFAULT '2025-26',
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(player_name, position, season)
);

-- Scheduled/upcoming games for prediction
CREATE TABLE IF NOT EXISTS scheduled_games (
  id INTEGER PRIMARY KEY,
  game_date TEXT NOT NULL,
  game_time TEXT,
  away_team_id INTEGER NOT NULL,
  home_team_id INTEGER NOT NULL,
  spread REAL,  -- Positive = away underdog, negative = home favored
  over_under REAL,
  tv_channel TEXT,
  status TEXT DEFAULT 'scheduled',  -- scheduled, in_progress, final
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(game_date, away_team_id, home_team_id),
  FOREIGN KEY (away_team_id) REFERENCES teams(id),
  FOREIGN KEY (home_team_id) REFERENCES teams(id)
);

CREATE INDEX IF NOT EXISTS idx_scheduled_games_date ON scheduled_games(game_date);

-- Data freshness tracking (to avoid stale data issues)
CREATE TABLE IF NOT EXISTS data_freshness (
  id INTEGER PRIMARY KEY,
  data_type TEXT NOT NULL UNIQUE,  -- e.g., 'boxscores', 'lines', 'injuries', 'matchups'
  last_updated TEXT NOT NULL DEFAULT (datetime('now')),
  records_count INTEGER DEFAULT 0,
  notes TEXT
);

-- Player trends cache (recent performance metrics)
CREATE TABLE IF NOT EXISTS player_trends (
  id INTEGER PRIMARY KEY,
  player_id INTEGER NOT NULL,
  as_of_date TEXT NOT NULL,
  games_sample INTEGER DEFAULT 5,
  avg_pts REAL,
  avg_reb REAL,
  avg_ast REAL,
  avg_min REAL,
  pts_trend TEXT,  -- 'up', 'down', 'stable'
  reb_trend TEXT,
  ast_trend TEXT,
  hot_streak INTEGER DEFAULT 0,  -- consecutive games over average
  cold_streak INTEGER DEFAULT 0,
  last_3_pts TEXT,  -- JSON array of last 3 games pts
  last_3_reb TEXT,
  last_3_ast TEXT,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(player_id, as_of_date),
  FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE INDEX IF NOT EXISTS idx_player_trends_player ON player_trends(player_id);
CREATE INDEX IF NOT EXISTS idx_player_trends_date ON player_trends(as_of_date);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def get_or_create_team(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM teams WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute("INSERT INTO teams(name) VALUES (?)", (name,))
    return int(cur.lastrowid)


def get_or_create_player(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if row:
        return int(row["id"])
    cur = conn.execute("INSERT INTO players(name) VALUES (?)", (name,))
    return int(cur.lastrowid)


def find_game_id(
    conn: sqlite3.Connection,
    game_date: str,
    team1_id: int,
    team2_id: int,
) -> Optional[int]:
    row = conn.execute(
        """
        SELECT id FROM games
        WHERE game_date = ?
          AND (
            (team1_id = ? AND team2_id = ?)
            OR
            (team1_id = ? AND team2_id = ?)
          )
        ORDER BY id ASC
        LIMIT 1
        """,
        (game_date, team1_id, team2_id, team2_id, team1_id),
    ).fetchone()
    return int(row["id"]) if row else None


