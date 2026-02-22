"""
Trade Tracker Module
====================

Tracks NBA player trades, roster changes, and team status changes.

KEY PROBLEMS SOLVED:
1. Players traded to new teams → historical stats are less relevant
2. New teammates → chemistry unknown → higher variance
3. Role changes → usage rates shift unpredictably
4. Teams tanking → star minutes reduced deliberately
5. Sportsbook lines may not fully adjust post-trade

APPROACH:
- Maintain a trade log with dates, old/new teams
- Flag recently-traded players and discount pre-trade stats
- Track "new team games" count for confidence adjustment
- Detect role/minute changes post-trade for teammates

Author: PropAI Team
Created: February 2026
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any
from pathlib import Path

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev


# ============================================================================
# Database Schema for Trade Tracking
# ============================================================================

TRADE_SCHEMA_SQL = """
-- Player trade history: tracks every trade/waiver/signing
CREATE TABLE IF NOT EXISTS player_trades (
    id INTEGER PRIMARY KEY,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    
    -- Old team info
    from_team TEXT NOT NULL,           -- Team abbreviation (e.g., "MIA")
    
    -- New team info
    to_team TEXT NOT NULL,             -- Team abbreviation (e.g., "GSW")
    
    -- Trade details
    trade_date TEXT NOT NULL,          -- YYYY-MM-DD
    trade_type TEXT DEFAULT 'trade',   -- 'trade', 'waiver', 'signing', 'buyout'
    
    -- Context
    was_starter INTEGER DEFAULT 0,    -- Was starter on old team
    old_team_role TEXT,               -- 'star', 'starter', 'rotation', 'bench'
    expected_new_role TEXT,           -- Expected role on new team
    notes TEXT,                       -- Free-form notes about the trade
    
    -- Tracking
    games_with_new_team INTEGER DEFAULT 0,  -- Updated as games are played
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    
    UNIQUE(player_name, trade_date, to_team),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

CREATE INDEX IF NOT EXISTS idx_player_trades_player ON player_trades(player_name);
CREATE INDEX IF NOT EXISTS idx_player_trades_date ON player_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_player_trades_teams ON player_trades(from_team, to_team);

-- Team roster status: tracks team-level changes post-trade-deadline
CREATE TABLE IF NOT EXISTS team_roster_status (
    id INTEGER PRIMARY KEY,
    team_abbrev TEXT NOT NULL,
    season TEXT DEFAULT '2025-26',
    
    -- Team direction
    is_tanking INTEGER DEFAULT 0,          -- 0=competing, 1=tanking
    tank_confidence REAL DEFAULT 0.0,      -- 0.0-1.0 confidence in tank assessment
    
    -- Roster stability
    players_traded_away INTEGER DEFAULT 0, -- Count of players traded away
    players_acquired INTEGER DEFAULT 0,    -- Count of players acquired
    roster_stability_score REAL DEFAULT 1.0, -- 1.0=stable, 0.0=completely changed
    
    -- Expected impact
    star_players_lost TEXT,                -- JSON list of star players traded away
    star_players_gained TEXT,              -- JSON list of star players acquired
    
    -- Minutes adjustment
    expected_minutes_factor REAL DEFAULT 1.0,  -- Multiplier for star minutes (tanking teams < 1.0)
    
    -- Context
    record_at_deadline TEXT,               -- e.g., "20-32" 
    playoff_probability REAL,              -- Pre-deadline playoff %
    notes TEXT,
    
    -- Tracking
    last_updated TEXT NOT NULL DEFAULT (datetime('now')),
    
    UNIQUE(team_abbrev, season)
);

CREATE INDEX IF NOT EXISTS idx_team_roster_status_team ON team_roster_status(team_abbrev);

-- Post-trade player game log: tracks performance on new team specifically
CREATE TABLE IF NOT EXISTS post_trade_performance (
    id INTEGER PRIMARY KEY,
    player_id INTEGER,
    player_name TEXT NOT NULL,
    new_team TEXT NOT NULL,
    trade_date TEXT NOT NULL,
    
    -- Game info
    game_date TEXT NOT NULL,
    game_id INTEGER,
    
    -- Stats
    minutes REAL,
    pts INTEGER,
    reb INTEGER,
    ast INTEGER,
    fgm INTEGER,
    fga INTEGER,
    plus_minus INTEGER,
    
    -- Context
    game_number_with_team INTEGER,  -- 1st, 2nd, 3rd game etc.
    
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    
    UNIQUE(player_name, game_date),
    FOREIGN KEY (player_id) REFERENCES players(id),
    FOREIGN KEY (game_id) REFERENCES games(id)
);

CREATE INDEX IF NOT EXISTS idx_post_trade_perf_player ON post_trade_performance(player_name);
"""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TradeInfo:
    """Information about a single player trade."""
    player_id: Optional[int]
    player_name: str
    from_team: str
    to_team: str
    trade_date: str
    trade_type: str = "trade"
    was_starter: bool = False
    old_team_role: str = "rotation"
    expected_new_role: str = "rotation"
    games_with_new_team: int = 0
    notes: str = ""
    
    @property
    def days_since_trade(self) -> int:
        """Days since the trade occurred."""
        try:
            td = datetime.strptime(self.trade_date, "%Y-%m-%d")
            return (datetime.now() - td).days
        except:
            return 0
    
    @property
    def is_recent_trade(self) -> bool:
        """Trade within last 14 days — stats unreliable."""
        return self.days_since_trade <= 14
    
    @property
    def is_settling_in(self) -> bool:
        """Player still adjusting to new team (14-30 days or <10 games)."""
        return self.days_since_trade <= 30 or self.games_with_new_team < 10
    
    @property
    def has_enough_new_team_data(self) -> bool:
        """Player has played enough games with new team for reliable stats."""
        return self.games_with_new_team >= 5
    
    @property
    def confidence_discount(self) -> float:
        """
        Discount factor for prediction confidence (0.0-1.0).
        
        Recently-traded players have HIGH uncertainty:
        - 0 games: 0.3 (70% discount - almost no confidence)
        - 1-2 games: 0.5 (50% discount)
        - 3-5 games: 0.65 (35% discount)
        - 6-10 games: 0.8 (20% discount)
        - 10+ games: 0.9 (10% discount)
        - 20+ games: 1.0 (fully confident)
        """
        g = self.games_with_new_team
        if g == 0:
            return 0.30
        elif g <= 2:
            return 0.50
        elif g <= 5:
            return 0.65
        elif g <= 10:
            return 0.80
        elif g <= 20:
            return 0.90
        return 1.0
    
    @property 
    def projection_weight_new_team(self) -> float:
        """
        How much to weight new-team data vs old-team data.
        
        - 0 games: 0.0 (all old team, but heavily discounted)
        - 1-2 games: 0.3 (30% new, 70% old)
        - 3-5 games: 0.5 (50/50 blend)
        - 6-10 games: 0.7 (70% new, 30% old)
        - 10-15 games: 0.85 
        - 15+ games: 0.95 (almost all new team)
        """
        g = self.games_with_new_team
        if g == 0:
            return 0.0
        elif g <= 2:
            return 0.3
        elif g <= 5:
            return 0.5
        elif g <= 10:
            return 0.7
        elif g <= 15:
            return 0.85
        return 0.95


@dataclass
class TeamRosterStatus:
    """Team-level roster status post-trade-deadline."""
    team_abbrev: str
    is_tanking: bool = False
    tank_confidence: float = 0.0
    players_traded_away: int = 0
    players_acquired: int = 0
    roster_stability_score: float = 1.0
    star_players_lost: List[str] = field(default_factory=list)
    star_players_gained: List[str] = field(default_factory=list)
    expected_minutes_factor: float = 1.0
    record_at_deadline: str = ""
    playoff_probability: float = 0.0
    notes: str = ""
    
    @property
    def had_significant_changes(self) -> bool:
        """Team had significant roster changes."""
        return (self.players_traded_away + self.players_acquired >= 3 or 
                len(self.star_players_lost) > 0 or 
                len(self.star_players_gained) > 0)
    
    @property
    def confidence_impact(self) -> float:
        """
        How much team changes should reduce prediction confidence.
        
        Returns multiplier 0.5-1.0:
        - 1.0 = no impact (stable roster)
        - 0.5 = massive changes (unreliable predictions)
        """
        if not self.had_significant_changes:
            return 1.0
        
        # Base penalty from player movement
        movement = self.players_traded_away + self.players_acquired
        base = max(0.6, 1.0 - (movement * 0.05))
        
        # Extra penalty for losing stars
        star_penalty = len(self.star_players_lost) * 0.1
        
        # Tanking teams get additional penalty
        tank_penalty = self.tank_confidence * 0.15
        
        return max(0.5, base - star_penalty - tank_penalty)


@dataclass
class TradeAdjustedStats:
    """
    Adjusted player statistics accounting for trade effects.
    
    Contains both old-team and new-team weighted projections.
    """
    player_name: str
    was_traded: bool = False
    trade_info: Optional[TradeInfo] = None
    team_status: Optional[TeamRosterStatus] = None
    
    # New-team performance (if available)
    new_team_games: int = 0
    new_team_avg_pts: float = 0.0
    new_team_avg_reb: float = 0.0
    new_team_avg_ast: float = 0.0
    new_team_avg_minutes: float = 0.0
    
    # Adjusted projections (blended old + new)
    adjusted_pts: float = 0.0
    adjusted_reb: float = 0.0
    adjusted_ast: float = 0.0
    adjusted_minutes: float = 0.0
    
    # Confidence metrics
    confidence_factor: float = 1.0  # Applied to overall confidence
    data_quality: str = "good"  # "good", "limited", "poor", "no_data"
    
    # Warnings for the user
    warnings: List[str] = field(default_factory=list)
    
    @property
    def should_skip(self) -> bool:
        """Whether to skip this player entirely (too risky)."""
        return self.data_quality == "no_data" and self.was_traded


# ============================================================================
# Database Operations
# ============================================================================

def init_trade_tables(conn: sqlite3.Connection) -> None:
    """Create trade tracking tables if they don't exist."""
    conn.executescript(TRADE_SCHEMA_SQL)
    conn.commit()


def record_trade(
    conn: sqlite3.Connection,
    player_name: str,
    from_team: str,
    to_team: str,
    trade_date: str,
    trade_type: str = "trade",
    was_starter: bool = False,
    old_team_role: str = "rotation",
    expected_new_role: str = "rotation",
    notes: str = "",
) -> int:
    """
    Record a player trade in the database.
    
    Returns the trade record ID.
    """
    # Normalize team abbreviations
    from_team = normalize_team_abbrev(from_team) or from_team.upper()
    to_team = normalize_team_abbrev(to_team) or to_team.upper()
    
    # Try to find player ID
    player_row = conn.execute(
        "SELECT id FROM players WHERE LOWER(name) = LOWER(?)", (player_name,)
    ).fetchone()
    player_id = player_row["id"] if player_row else None
    
    cur = conn.execute(
        """
        INSERT OR REPLACE INTO player_trades 
        (player_id, player_name, from_team, to_team, trade_date, trade_type,
         was_starter, old_team_role, expected_new_role, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (player_id, player_name, from_team, to_team, trade_date, trade_type,
         1 if was_starter else 0, old_team_role, expected_new_role, notes),
    )
    conn.commit()
    
    # Update the player's archetype team if it exists
    if player_id:
        try:
            conn.execute(
                "UPDATE player_archetypes SET team = ?, updated_at = datetime('now') WHERE player_id = ?",
                (to_team, player_id),
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # player_archetypes table may not exist
    
    print(f"  ✅ Recorded: {player_name} traded from {from_team} → {to_team} on {trade_date}")
    return cur.lastrowid


def record_team_status(
    conn: sqlite3.Connection,
    team_abbrev: str,
    is_tanking: bool = False,
    tank_confidence: float = 0.0,
    players_traded_away: int = 0,
    players_acquired: int = 0,
    star_players_lost: Optional[List[str]] = None,
    star_players_gained: Optional[List[str]] = None,
    expected_minutes_factor: float = 1.0,
    record_at_deadline: str = "",
    playoff_probability: float = 0.0,
    notes: str = "",
) -> None:
    """Record team roster status post-trade deadline."""
    import json
    
    team_abbrev = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
    
    # Calculate roster stability score
    total_moves = players_traded_away + players_acquired
    stability = max(0.3, 1.0 - (total_moves * 0.08))
    
    conn.execute(
        """
        INSERT OR REPLACE INTO team_roster_status
        (team_abbrev, is_tanking, tank_confidence, players_traded_away,
         players_acquired, roster_stability_score, star_players_lost,
         star_players_gained, expected_minutes_factor, record_at_deadline,
         playoff_probability, notes, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (team_abbrev, 1 if is_tanking else 0, tank_confidence,
         players_traded_away, players_acquired, stability,
         json.dumps(star_players_lost or []),
         json.dumps(star_players_gained or []),
         expected_minutes_factor, record_at_deadline,
         playoff_probability, notes),
    )
    conn.commit()
    
    status = "🏳️ TANKING" if is_tanking else "🏀 Competing"
    print(f"  {status} {team_abbrev}: {players_traded_away} out / {players_acquired} in "
          f"(stability: {stability:.2f})")


def get_player_trade_info(
    conn: sqlite3.Connection,
    player_name: str,
    as_of_date: Optional[str] = None,
) -> Optional[TradeInfo]:
    """
    Get trade info for a player, if they were traded.
    
    Returns None if the player was NOT traded.
    """
    query = """
        SELECT * FROM player_trades 
        WHERE LOWER(player_name) = LOWER(?)
    """
    params: list = [player_name]
    
    if as_of_date:
        query += " AND trade_date <= ?"
        params.append(as_of_date)
    
    query += " ORDER BY trade_date DESC LIMIT 1"
    
    row = conn.execute(query, params).fetchone()
    if not row:
        return None
    
    return TradeInfo(
        player_id=row["player_id"],
        player_name=row["player_name"],
        from_team=row["from_team"],
        to_team=row["to_team"],
        trade_date=row["trade_date"],
        trade_type=row["trade_type"],
        was_starter=bool(row["was_starter"]),
        old_team_role=row["old_team_role"] or "rotation",
        expected_new_role=row["expected_new_role"] or "rotation",
        games_with_new_team=row["games_with_new_team"] or 0,
        notes=row["notes"] or "",
    )


def get_player_trade_info_by_id(
    conn: sqlite3.Connection,
    player_id: int,
    as_of_date: Optional[str] = None,
) -> Optional[TradeInfo]:
    """Get trade info for a player by player_id."""
    query = """
        SELECT * FROM player_trades 
        WHERE player_id = ?
    """
    params: list = [player_id]
    
    if as_of_date:
        query += " AND trade_date <= ?"
        params.append(as_of_date)
    
    query += " ORDER BY trade_date DESC LIMIT 1"
    
    row = conn.execute(query, params).fetchone()
    if not row:
        return None
    
    return TradeInfo(
        player_id=row["player_id"],
        player_name=row["player_name"],
        from_team=row["from_team"],
        to_team=row["to_team"],
        trade_date=row["trade_date"],
        trade_type=row["trade_type"],
        was_starter=bool(row["was_starter"]),
        old_team_role=row["old_team_role"] or "rotation",
        expected_new_role=row["expected_new_role"] or "rotation",
        games_with_new_team=row["games_with_new_team"] or 0,
        notes=row["notes"] or "",
    )


def get_team_roster_status(
    conn: sqlite3.Connection,
    team_abbrev: str,
) -> Optional[TeamRosterStatus]:
    """Get team roster status post-trade deadline."""
    import json
    
    team_abbrev = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
    
    row = conn.execute(
        "SELECT * FROM team_roster_status WHERE team_abbrev = ?",
        (team_abbrev,),
    ).fetchone()
    
    if not row:
        return None
    
    return TeamRosterStatus(
        team_abbrev=row["team_abbrev"],
        is_tanking=bool(row["is_tanking"]),
        tank_confidence=row["tank_confidence"] or 0.0,
        players_traded_away=row["players_traded_away"] or 0,
        players_acquired=row["players_acquired"] or 0,
        roster_stability_score=row["roster_stability_score"] or 1.0,
        star_players_lost=json.loads(row["star_players_lost"]) if row["star_players_lost"] else [],
        star_players_gained=json.loads(row["star_players_gained"]) if row["star_players_gained"] else [],
        expected_minutes_factor=row["expected_minutes_factor"] or 1.0,
        record_at_deadline=row["record_at_deadline"] or "",
        playoff_probability=row["playoff_probability"] or 0.0,
        notes=row["notes"] or "",
    )


def get_all_traded_players(
    conn: sqlite3.Connection,
    since_date: Optional[str] = None,
) -> List[TradeInfo]:
    """Get all traded players, optionally since a specific date."""
    query = "SELECT * FROM player_trades"
    params: list = []
    
    if since_date:
        query += " WHERE trade_date >= ?"
        params.append(since_date)
    
    query += " ORDER BY trade_date DESC"
    
    rows = conn.execute(query, params).fetchall()
    
    return [
        TradeInfo(
            player_id=r["player_id"],
            player_name=r["player_name"],
            from_team=r["from_team"],
            to_team=r["to_team"],
            trade_date=r["trade_date"],
            trade_type=r["trade_type"],
            was_starter=bool(r["was_starter"]),
            old_team_role=r["old_team_role"] or "rotation",
            expected_new_role=r["expected_new_role"] or "rotation",
            games_with_new_team=r["games_with_new_team"] or 0,
            notes=r["notes"] or "",
        )
        for r in rows
    ]


def get_trades_affecting_team(
    conn: sqlite3.Connection,
    team_abbrev: str,
    since_date: Optional[str] = None,
) -> Dict[str, List[TradeInfo]]:
    """
    Get all trades affecting a team (both departures and arrivals).
    
    Returns dict with keys: 'departed', 'arrived'
    """
    team_abbrev = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
    
    result = {"departed": [], "arrived": []}
    
    # Players who left
    query_from = "SELECT * FROM player_trades WHERE from_team = ?"
    params: list = [team_abbrev]
    if since_date:
        query_from += " AND trade_date >= ?"
        params.append(since_date)
    
    for r in conn.execute(query_from, params).fetchall():
        result["departed"].append(TradeInfo(
            player_id=r["player_id"],
            player_name=r["player_name"],
            from_team=r["from_team"],
            to_team=r["to_team"],
            trade_date=r["trade_date"],
            trade_type=r["trade_type"],
            was_starter=bool(r["was_starter"]),
            old_team_role=r["old_team_role"] or "rotation",
            expected_new_role=r["expected_new_role"] or "rotation",
            games_with_new_team=r["games_with_new_team"] or 0,
            notes=r["notes"] or "",
        ))
    
    # Players who arrived
    query_to = "SELECT * FROM player_trades WHERE to_team = ?"
    params2: list = [team_abbrev]
    if since_date:
        query_to += " AND trade_date >= ?"
        params2.append(since_date)
    
    for r in conn.execute(query_to, params2).fetchall():
        result["arrived"].append(TradeInfo(
            player_id=r["player_id"],
            player_name=r["player_name"],
            from_team=r["from_team"],
            to_team=r["to_team"],
            trade_date=r["trade_date"],
            trade_type=r["trade_type"],
            was_starter=bool(r["was_starter"]),
            old_team_role=r["old_team_role"] or "rotation",
            expected_new_role=r["expected_new_role"] or "rotation",
            games_with_new_team=r["games_with_new_team"] or 0,
            notes=r["notes"] or "",
        ))
    
    return result


def update_post_trade_game_counts(conn: sqlite3.Connection) -> int:
    """
    Scan boxscores to count how many games each traded player has played
    with their new team since the trade. Updates the games_with_new_team count.
    
    Returns number of trades updated.
    """
    trades = conn.execute("SELECT * FROM player_trades").fetchall()
    updated = 0
    
    for trade in trades:
        player_id = trade["player_id"]
        if not player_id:
            # Try to find player by name
            p = conn.execute(
                "SELECT id FROM players WHERE LOWER(name) = LOWER(?)",
                (trade["player_name"],)
            ).fetchone()
            if p:
                player_id = p["id"]
                conn.execute(
                    "UPDATE player_trades SET player_id = ? WHERE id = ?",
                    (player_id, trade["id"])
                )
            else:
                continue
        
        # Count games AFTER trade date with new team
        to_team = trade["to_team"]
        # Get team_id for the new team
        team_row = conn.execute(
            """SELECT t.id FROM teams t 
               WHERE t.name LIKE '%' || ? || '%'
               OR t.id IN (
                   SELECT id FROM teams WHERE name IN (
                       SELECT name FROM teams 
                       WHERE name LIKE '%' || ? || '%'
                   )
               )
               LIMIT 1""",
            (to_team, to_team)
        ).fetchone()
        
        if not team_row:
            # Try to match via abbrev
            from ..team_aliases import team_name_from_abbrev
            full_name = team_name_from_abbrev(to_team)
            if full_name:
                team_row = conn.execute(
                    "SELECT id FROM teams WHERE name = ?", (full_name,)
                ).fetchone()
        
        if not team_row:
            continue
            
        count = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.player_id = ?
              AND b.team_id = ?
              AND g.game_date > ?
              AND b.minutes > 0
            """,
            (player_id, team_row["id"], trade["trade_date"]),
        ).fetchone()["cnt"]
        
        if count != trade["games_with_new_team"]:
            conn.execute(
                "UPDATE player_trades SET games_with_new_team = ? WHERE id = ?",
                (count, trade["id"])
            )
            updated += 1
    
    conn.commit()
    return updated


def auto_detect_trades_from_boxscores(
    conn: sqlite3.Connection,
    since_date: Optional[str] = None,
    deadline_date: str = "2026-02-06",
    verbose: bool = True,
) -> List[TradeInfo]:
    """
    Auto-detect player trades by scanning boxscore data for team changes.

    For each player, we look at consecutive games and check if team_id changed.
    If a player appeared for Team A in one game and Team B in the next game,
    and the transition happened around the trade deadline, we record it as a trade.

    This eliminates the need for manual CLI entry of every single trade.

    Args:
        conn: Database connection
        since_date: Only scan games since this date (default: 30 days before deadline)
        deadline_date: The trade deadline date
        verbose: Print discovered trades

    Returns:
        List of newly detected TradeInfo objects
    """
    init_trade_tables(conn)

    if since_date is None:
        # Scan from 7 days before the deadline to catch pre-deadline trades too
        dl = datetime.strptime(deadline_date, "%Y-%m-%d")
        since_date = (dl - timedelta(days=7)).strftime("%Y-%m-%d")

    if verbose:
        print(f"\n🔍 Scanning boxscores for player team changes since {since_date}...")

    # Get all players who have boxscore entries since the scan date
    # Order by game_date to see chronological team changes
    rows = conn.execute(
        """
        SELECT
            b.player_id,
            p.name AS player_name,
            b.team_id,
            t.name AS team_name,
            g.game_date,
            b.minutes,
            b.pts,
            b.status
        FROM boxscore_player b
        JOIN players p ON p.id = b.player_id
        JOIN teams t ON t.id = b.team_id
        JOIN games g ON g.id = b.game_id
        WHERE g.game_date >= ?
          AND b.minutes IS NOT NULL
          AND b.minutes > 0
        ORDER BY b.player_id, g.game_date ASC
        """,
        (since_date,),
    ).fetchall()

    if not rows:
        if verbose:
            print("  No boxscore data found in the scan range.")
        return []

    # Group by player
    from collections import defaultdict
    player_games: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        player_games[r["player_id"]].append(dict(r))

    detected_trades: List[TradeInfo] = []
    already_recorded: Set[str] = set()

    # Check which trades are already recorded
    try:
        existing = conn.execute(
            "SELECT player_name, trade_date, to_team FROM player_trades"
        ).fetchall()
        for e in existing:
            key = f"{e['player_name'].lower()}|{e['trade_date']}|{e['to_team']}"
            already_recorded.add(key)
    except sqlite3.OperationalError:
        pass

    for player_id, games in player_games.items():
        if len(games) < 2:
            continue

        # Look for team_id transitions
        for i in range(1, len(games)):
            prev_game = games[i - 1]
            curr_game = games[i]

            if prev_game["team_id"] != curr_game["team_id"]:
                player_name = curr_game["player_name"]
                from_team_name = prev_game["team_name"]
                to_team_name = curr_game["team_name"]

                from_abbrev = abbrev_from_team_name(from_team_name) or from_team_name[:3].upper()
                to_abbrev = abbrev_from_team_name(to_team_name) or to_team_name[:3].upper()

                # Skip false positives: same team with different DB entries
                # (e.g., "LA Clippers" id=13 vs "Los Angeles Clippers" id=32)
                if from_abbrev == to_abbrev:
                    continue

                # Trade date is between the two games; use the day before the new-team game
                trade_date_dt = datetime.strptime(curr_game["game_date"], "%Y-%m-%d") - timedelta(days=1)
                trade_date = trade_date_dt.strftime("%Y-%m-%d")

                # Check if already recorded
                key = f"{player_name.lower()}|{trade_date}|{to_abbrev}"
                # Also check with the actual game date as trade date (±1 day tolerance)
                alt_key = f"{player_name.lower()}|{curr_game['game_date']}|{to_abbrev}"
                prev_day_key = f"{player_name.lower()}|{prev_game['game_date']}|{to_abbrev}"

                if key in already_recorded or alt_key in already_recorded or prev_day_key in already_recorded:
                    continue

                # Determine role from previous team stats
                was_starter = prev_game.get("status", "").lower() == "starter"
                avg_min = statistics.mean(
                    g["minutes"] for g in games[:i] if g["team_id"] == prev_game["team_id"] and g["minutes"]
                ) if any(g["team_id"] == prev_game["team_id"] for g in games[:i]) else prev_game.get("minutes", 0)
                avg_pts = statistics.mean(
                    g["pts"] for g in games[:i] if g["team_id"] == prev_game["team_id"] and g["pts"] is not None
                ) if any(g["team_id"] == prev_game["team_id"] and g.get("pts") is not None for g in games[:i]) else 0

                if avg_min >= 30 and avg_pts >= 20:
                    old_role = "star"
                elif avg_min >= 28 and avg_pts >= 15:
                    old_role = "starter"
                elif avg_min >= 20:
                    old_role = "rotation"
                else:
                    old_role = "bench"

                # Count games with new team (from boxscore data)
                new_team_games = sum(
                    1 for g in games[i:] if g["team_id"] == curr_game["team_id"]
                )

                # Record the trade
                try:
                    trade_id = record_trade(
                        conn,
                        player_name=player_name,
                        from_team=from_abbrev,
                        to_team=to_abbrev,
                        trade_date=trade_date,
                        trade_type="trade",
                        was_starter=was_starter,
                        old_team_role=old_role,
                        expected_new_role=old_role,  # Assume similar role initially
                        notes=f"Auto-detected from boxscore data. First game with {to_abbrev}: {curr_game['game_date']}",
                    )

                    # Update games count
                    conn.execute(
                        "UPDATE player_trades SET games_with_new_team = ? WHERE id = ?",
                        (new_team_games, trade_id),
                    )
                    conn.commit()

                    trade_info = TradeInfo(
                        player_id=player_id,
                        player_name=player_name,
                        from_team=from_abbrev,
                        to_team=to_abbrev,
                        trade_date=trade_date,
                        trade_type="trade",
                        was_starter=was_starter,
                        old_team_role=old_role,
                        expected_new_role=old_role,
                        games_with_new_team=new_team_games,
                        notes=f"Auto-detected",
                    )
                    detected_trades.append(trade_info)
                    already_recorded.add(key)

                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ Error recording trade for {player_name}: {e}")

    if verbose:
        if detected_trades:
            print(f"\n✅ Auto-detected {len(detected_trades)} trades:")
            for t in detected_trades:
                print(f"   {t.player_name}: {t.from_team} → {t.to_team} "
                      f"(~{t.trade_date}) [{t.old_team_role}] "
                      f"({t.games_with_new_team} new-team games)")
        else:
            print("  No new trades detected.")

    return detected_trades


def auto_update_team_roster_status(
    conn: sqlite3.Connection,
    deadline_date: str = "2026-02-06",
    verbose: bool = True,
) -> Dict[str, TeamRosterStatus]:
    """
    Auto-generate team roster status from detected trades.

    For each team that had trades, calculate:
    - Number of players traded away / acquired
    - Star players lost / gained
    - Roster stability score

    This automates the 'record-team-status' CLI command.
    """
    init_trade_tables(conn)

    if verbose:
        print(f"\n📊 Auto-updating team roster status from trade data...")

    # Get all trades around the deadline
    try:
        trades = conn.execute(
            "SELECT * FROM player_trades WHERE trade_date >= date(?, '-7 days')",
            (deadline_date,),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    if not trades:
        if verbose:
            print("  No trades recorded. Run auto-detect or record trades first.")
        return {}

    # Group by team
    from collections import defaultdict
    team_departed: Dict[str, List[dict]] = defaultdict(list)
    team_arrived: Dict[str, List[dict]] = defaultdict(list)

    for t in trades:
        team_departed[t["from_team"]].append(dict(t))
        team_arrived[t["to_team"]].append(dict(t))

    all_teams = set(team_departed.keys()) | set(team_arrived.keys())
    results: Dict[str, TeamRosterStatus] = {}

    for team in all_teams:
        departed = team_departed.get(team, [])
        arrived = team_arrived.get(team, [])

        stars_lost = [t["player_name"] for t in departed
                      if t.get("old_team_role") in ("star", "starter") and t.get("was_starter")]
        stars_gained = [t["player_name"] for t in arrived
                        if t.get("expected_new_role") in ("star", "starter")]

        # V19.4: Preserve existing manually-set tanking flags so that
        # auto_update doesn't silently overwrite is_tanking=True set by
        # run_trade_setup or the CLI.
        existing_row = conn.execute(
            "SELECT is_tanking, tank_confidence FROM team_roster_status "
            "WHERE team_abbrev = ?",
            (team,),
        ).fetchone()
        preserve_is_tanking = bool(existing_row["is_tanking"]) if existing_row else False
        preserve_tank_conf = (existing_row["tank_confidence"] or 0.0) if existing_row else 0.0

        record_team_status(
            conn,
            team_abbrev=team,
            is_tanking=preserve_is_tanking,
            tank_confidence=preserve_tank_conf,
            players_traded_away=len(departed),
            players_acquired=len(arrived),
            star_players_lost=stars_lost,
            star_players_gained=stars_gained,
            notes=f"Auto-generated from {len(departed)} departures, {len(arrived)} arrivals",
        )

        results[team] = TeamRosterStatus(
            team_abbrev=team,
            players_traded_away=len(departed),
            players_acquired=len(arrived),
            star_players_lost=stars_lost,
            star_players_gained=stars_gained,
        )

    if verbose:
        print(f"  Updated roster status for {len(results)} teams.")

    return results


def get_team_trade_summary(
    conn: sqlite3.Connection,
    team_abbrev: str,
) -> str:
    """Generate a human-readable trade summary for a team."""
    team_abbrev = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
    
    trades = get_trades_affecting_team(conn, team_abbrev)
    status = get_team_roster_status(conn, team_abbrev)
    
    lines = [f"\n{'='*60}", f"Trade Summary: {team_abbrev}", f"{'='*60}"]
    
    if status:
        if status.is_tanking:
            lines.append(f"⚠️  TEAM STATUS: TANKING (confidence: {status.tank_confidence:.0%})")
            lines.append(f"   Expected minutes factor: {status.expected_minutes_factor:.2f}")
        else:
            lines.append(f"✅ TEAM STATUS: Competing")
        lines.append(f"   Roster stability: {status.roster_stability_score:.2f}")
        if status.record_at_deadline:
            lines.append(f"   Record at deadline: {status.record_at_deadline}")
    
    if trades["departed"]:
        lines.append(f"\n📤 Players Departed ({len(trades['departed'])}):")
        for t in trades["departed"]:
            role = f" [{t.old_team_role}]" if t.old_team_role else ""
            lines.append(f"   - {t.player_name}{role} → {t.to_team} ({t.trade_date})")
    
    if trades["arrived"]:
        lines.append(f"\n📥 Players Arrived ({len(trades['arrived'])}):")
        for t in trades["arrived"]:
            games_str = f" ({t.games_with_new_team} games)" if t.games_with_new_team else " (no games yet)"
            lines.append(f"   - {t.player_name} from {t.from_team}{games_str}")
    
    if not trades["departed"] and not trades["arrived"]:
        lines.append("\nNo trades recorded for this team.")
    
    return "\n".join(lines)
