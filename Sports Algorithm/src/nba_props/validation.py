"""Data validation and integrity checks for NBA props database."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .team_aliases import abbrev_from_team_name, team_name_from_abbrev
from .standings import ALL_ABBREVS


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    details: list[dict] = field(default_factory=list)
    severity: str = "WARNING"  # INFO, WARNING, ERROR


@dataclass
class ValidationReport:
    """Complete validation report."""
    results: list[ValidationResult] = field(default_factory=list)
    total_checks: int = 0
    passed_checks: int = 0
    warnings: int = 0
    errors: int = 0
    
    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)
        self.total_checks += 1
        if result.passed:
            self.passed_checks += 1
        elif result.severity == "WARNING":
            self.warnings += 1
        elif result.severity == "ERROR":
            self.errors += 1
    
    def is_valid(self) -> bool:
        """Returns True if no errors (warnings are OK)."""
        return self.errors == 0


def check_duplicate_games(conn: sqlite3.Connection) -> ValidationResult:
    """Check for duplicate games (same teams on same date)."""
    rows = conn.execute(
        """
        WITH normalized AS (
          SELECT
            id,
            game_date,
            CASE WHEN team1_id < team2_id THEN team1_id ELSE team2_id END AS a_id,
            CASE WHEN team1_id < team2_id THEN team2_id ELSE team1_id END AS b_id
          FROM games
        )
        SELECT
          n.game_date,
          ta.name AS team_a,
          tb.name AS team_b,
          COUNT(*) AS cnt,
          GROUP_CONCAT(n.id) AS game_ids
        FROM normalized n
        JOIN teams ta ON ta.id = n.a_id
        JOIN teams tb ON tb.id = n.b_id
        GROUP BY n.game_date, n.a_id, n.b_id
        HAVING COUNT(*) > 1
        ORDER BY n.game_date DESC, cnt DESC
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="duplicate_games",
            passed=True,
            message="No duplicate games found",
        )
    
    return ValidationResult(
        check_name="duplicate_games",
        passed=False,
        message=f"Found {len(rows)} duplicate game(s) by date + matchup",
        details=[
            {
                "date": r["game_date"],
                "team_a": r["team_a"],
                "team_b": r["team_b"],
                "count": r["cnt"],
                "game_ids": r["game_ids"],
            }
            for r in rows
        ],
        severity="ERROR",
    )


def check_team_played_twice_same_day(conn: sqlite3.Connection) -> ValidationResult:
    """Check for teams that appear in multiple games on the same day."""
    rows = conn.execute(
        """
        WITH team_dates AS (
            SELECT game_date, team1_id as team_id FROM games
            UNION ALL
            SELECT game_date, team2_id as team_id FROM games
        )
        SELECT td.game_date, t.name as team, COUNT(*) as cnt
        FROM team_dates td
        JOIN teams t ON t.id = td.team_id
        GROUP BY td.game_date, td.team_id
        HAVING COUNT(*) > 1
        ORDER BY td.game_date DESC, cnt DESC
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="team_played_twice_same_day",
            passed=True,
            message="No teams played multiple games on the same day",
        )
    
    return ValidationResult(
        check_name="team_played_twice_same_day",
        passed=False,
        message=f"Found {len(rows)} instance(s) of teams playing multiple games on same day",
        details=[
            {
                "date": r["game_date"],
                "team": r["team"],
                "games": r["cnt"],
            }
            for r in rows
        ],
        severity="ERROR",
    )


def check_invalid_team_names(conn: sqlite3.Connection) -> ValidationResult:
    """Check for team names that don't match known NBA teams."""
    rows = conn.execute("SELECT id, name FROM teams ORDER BY name").fetchall()
    
    # Build list of valid team names from abbreviations
    valid_team_names = set()
    for abbrev in ALL_ABBREVS:
        name = team_name_from_abbrev(abbrev)
        if name:
            valid_team_names.add(name)
    
    invalid_teams = []
    for r in rows:
        name = r["name"]
        # Check if it's a valid team name
        if name not in valid_team_names and abbrev_from_team_name(name) is None:
            invalid_teams.append({"id": r["id"], "name": name})
    
    if not invalid_teams:
        return ValidationResult(
            check_name="invalid_team_names",
            passed=True,
            message="All team names are valid NBA teams",
        )
    
    return ValidationResult(
        check_name="invalid_team_names",
        passed=False,
        message=f"Found {len(invalid_teams)} invalid team name(s)",
        details=invalid_teams,
        severity="WARNING",
    )


def check_orphaned_teams(conn: sqlite3.Connection) -> ValidationResult:
    """Check for teams with no associated games."""
    rows = conn.execute(
        """
        SELECT t.id, t.name
        FROM teams t
        LEFT JOIN games g ON g.team1_id = t.id OR g.team2_id = t.id
        WHERE g.id IS NULL
        ORDER BY t.name
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="orphaned_teams",
            passed=True,
            message="No orphaned teams found",
        )
    
    return ValidationResult(
        check_name="orphaned_teams",
        passed=False,
        message=f"Found {len(rows)} team(s) with no games",
        details=[{"id": r["id"], "name": r["name"]} for r in rows],
        severity="INFO",
    )


def check_extreme_player_stats(conn: sqlite3.Connection) -> ValidationResult:
    """Check for player stats that seem unreasonable."""
    rows = conn.execute(
        """
        SELECT p.name, g.game_date, t.name as team, 
               b.pts, b.reb, b.ast, b.minutes
        FROM boxscore_player b
        JOIN players p ON p.id = b.player_id
        JOIN games g ON g.id = b.game_id
        JOIN teams t ON t.id = b.team_id
        WHERE b.pts > 70 
           OR b.reb > 35 
           OR b.ast > 25 
           OR b.minutes > 60
           OR (b.pts < 0 AND b.pts IS NOT NULL)
           OR (b.reb < 0 AND b.reb IS NOT NULL)
           OR (b.ast < 0 AND b.ast IS NOT NULL)
        ORDER BY g.game_date DESC
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="extreme_player_stats",
            passed=True,
            message="No extreme player stats found",
        )
    
    return ValidationResult(
        check_name="extreme_player_stats",
        passed=False,
        message=f"Found {len(rows)} player performance(s) with extreme stats",
        details=[
            {
                "player": r["name"],
                "date": r["game_date"],
                "team": r["team"],
                "pts": r["pts"],
                "reb": r["reb"],
                "ast": r["ast"],
                "minutes": r["minutes"],
            }
            for r in rows
        ],
        severity="WARNING",
    )


def check_game_date_range(conn: sqlite3.Connection) -> ValidationResult:
    """Check that game dates are within reasonable range for current season."""
    rows = conn.execute(
        """
        SELECT game_date, COUNT(*) as cnt
        FROM games
        WHERE game_date < '2025-10-01' OR game_date > '2026-06-30'
        GROUP BY game_date
        ORDER BY game_date
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="game_date_range",
            passed=True,
            message="All game dates are within 2025-26 season range",
        )
    
    return ValidationResult(
        check_name="game_date_range",
        passed=False,
        message=f"Found {len(rows)} game date(s) outside expected season range",
        details=[{"date": r["game_date"], "count": r["cnt"]} for r in rows],
        severity="WARNING",
    )


def check_team_totals_consistency(conn: sqlite3.Connection) -> ValidationResult:
    """Check that team totals match sum of player stats (when totals exist)."""
    rows = conn.execute(
        """
        SELECT g.game_date, t.name as team,
               tt.pts as team_pts,
               (SELECT SUM(b.pts) FROM boxscore_player b 
                WHERE b.game_id = g.id AND b.team_id = t.id AND b.pts IS NOT NULL) as sum_pts,
               tt.reb as team_reb,
               (SELECT SUM(b.reb) FROM boxscore_player b 
                WHERE b.game_id = g.id AND b.team_id = t.id AND b.reb IS NOT NULL) as sum_reb,
               tt.ast as team_ast,
               (SELECT SUM(b.ast) FROM boxscore_player b 
                WHERE b.game_id = g.id AND b.team_id = t.id AND b.ast IS NOT NULL) as sum_ast
        FROM boxscore_team_totals tt
        JOIN games g ON g.id = tt.game_id
        JOIN teams t ON t.id = tt.team_id
        WHERE tt.pts IS NOT NULL
        """
    ).fetchall()
    
    mismatches = []
    for r in rows:
        if r["team_pts"] != r["sum_pts"]:
            mismatches.append({
                "date": r["game_date"],
                "team": r["team"],
                "stat": "PTS",
                "team_total": r["team_pts"],
                "sum_players": r["sum_pts"],
            })
        if r["team_reb"] and r["sum_reb"] and r["team_reb"] != r["sum_reb"]:
            mismatches.append({
                "date": r["game_date"],
                "team": r["team"],
                "stat": "REB",
                "team_total": r["team_reb"],
                "sum_players": r["sum_reb"],
            })
    
    if not mismatches:
        return ValidationResult(
            check_name="team_totals_consistency",
            passed=True,
            message="Team totals match sum of player stats",
        )
    
    return ValidationResult(
        check_name="team_totals_consistency",
        passed=False,
        message=f"Found {len(mismatches)} mismatch(es) between team totals and player sums",
        details=mismatches[:10],  # Limit to first 10
        severity="WARNING",
    )


def check_player_name_duplicates(conn: sqlite3.Connection) -> ValidationResult:
    """Check for potential duplicate player entries (similar names)."""
    # This is a simple check - could be improved with fuzzy matching
    rows = conn.execute(
        """
        SELECT p1.id as id1, p1.name as name1, p2.id as id2, p2.name as name2
        FROM players p1
        JOIN players p2 ON p1.id < p2.id
        WHERE (
            LOWER(REPLACE(p1.name, ' ', '')) = LOWER(REPLACE(p2.name, ' ', ''))
            OR LOWER(REPLACE(REPLACE(p1.name, '.', ''), ' ', '')) = LOWER(REPLACE(REPLACE(p2.name, '.', ''), ' ', ''))
        )
        ORDER BY p1.name
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="player_name_duplicates",
            passed=True,
            message="No potential duplicate player names found",
        )
    
    return ValidationResult(
        check_name="player_name_duplicates",
        passed=False,
        message=f"Found {len(rows)} potential duplicate player name(s)",
        details=[
            {
                "player1": {"id": r["id1"], "name": r["name1"]},
                "player2": {"id": r["id2"], "name": r["name2"]},
            }
            for r in rows
        ],
        severity="WARNING",
    )


def check_games_with_too_few_players(conn: sqlite3.Connection) -> ValidationResult:
    """Check for games where a team has fewer than 5 players with stats."""
    rows = conn.execute(
        """
        SELECT g.id, g.game_date, t.name as team, COUNT(*) as player_count
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        JOIN teams t ON t.id = b.team_id
        WHERE b.minutes IS NOT NULL AND b.minutes > 0
        GROUP BY g.id, b.team_id
        HAVING COUNT(*) < 5
        ORDER BY g.game_date DESC
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="games_with_too_few_players",
            passed=True,
            message="All games have at least 5 players per team with stats",
        )
    
    return ValidationResult(
        check_name="games_with_too_few_players",
        passed=False,
        message=f"Found {len(rows)} team-game(s) with fewer than 5 players",
        details=[
            {
                "game_id": r["id"],
                "date": r["game_date"],
                "team": r["team"],
                "players_with_stats": r["player_count"],
            }
            for r in rows
        ],
        severity="WARNING",
    )


def check_missing_team_totals(conn: sqlite3.Connection) -> ValidationResult:
    """Check for games missing team totals."""
    rows = conn.execute(
        """
        SELECT g.id, g.game_date, t1.name as team1, t2.name as team2
        FROM games g
        JOIN teams t1 ON t1.id = g.team1_id
        JOIN teams t2 ON t2.id = g.team2_id
        LEFT JOIN boxscore_team_totals tt1 ON tt1.game_id = g.id AND tt1.team_id = g.team1_id
        LEFT JOIN boxscore_team_totals tt2 ON tt2.game_id = g.id AND tt2.team_id = g.team2_id
        WHERE tt1.id IS NULL OR tt2.id IS NULL
        ORDER BY g.game_date DESC
        """
    ).fetchall()
    
    if not rows:
        return ValidationResult(
            check_name="missing_team_totals",
            passed=True,
            message="All games have team totals for both teams",
        )
    
    return ValidationResult(
        check_name="missing_team_totals",
        passed=False,
        message=f"Found {len(rows)} game(s) missing team totals",
        details=[
            {
                "game_id": r["id"],
                "date": r["game_date"],
                "team1": r["team1"],
                "team2": r["team2"],
            }
            for r in rows[:10]
        ],
        severity="INFO",
    )


def run_all_validations(conn: sqlite3.Connection) -> ValidationReport:
    """Run all validation checks and return a comprehensive report."""
    report = ValidationReport()
    
    # Critical checks
    report.add_result(check_duplicate_games(conn))
    report.add_result(check_team_played_twice_same_day(conn))
    
    # Data quality checks
    report.add_result(check_invalid_team_names(conn))
    report.add_result(check_extreme_player_stats(conn))
    report.add_result(check_game_date_range(conn))
    report.add_result(check_team_totals_consistency(conn))
    report.add_result(check_player_name_duplicates(conn))
    report.add_result(check_games_with_too_few_players(conn))
    
    # Informational checks
    report.add_result(check_orphaned_teams(conn))
    report.add_result(check_missing_team_totals(conn))
    
    return report


def cleanup_orphaned_teams(conn: sqlite3.Connection) -> int:
    """Remove teams that have no games. Returns count of removed teams."""
    cur = conn.execute(
        """
        DELETE FROM teams
        WHERE id NOT IN (
            SELECT team1_id FROM games
            UNION
            SELECT team2_id FROM games
        )
        """
    )
    conn.commit()
    return cur.rowcount


def merge_duplicate_players(conn: sqlite3.Connection, keep_id: int, remove_id: int) -> int:
    """Merge a duplicate player into another player record."""
    # Update all references from remove_id to keep_id
    updates = 0
    
    # Update boxscore_player
    cur = conn.execute(
        """
        UPDATE boxscore_player SET player_id = ?
        WHERE player_id = ? AND game_id NOT IN (
            SELECT game_id FROM boxscore_player WHERE player_id = ?
        )
        """,
        (keep_id, remove_id, keep_id),
    )
    updates += cur.rowcount
    
    # Delete duplicate entries (if same player in same game)
    conn.execute(
        "DELETE FROM boxscore_player WHERE player_id = ?",
        (remove_id,),
    )
    
    # Update sportsbook_lines
    cur = conn.execute(
        "UPDATE sportsbook_lines SET player_id = ? WHERE player_id = ?",
        (keep_id, remove_id),
    )
    updates += cur.rowcount
    
    # Update injury_report
    cur = conn.execute(
        "UPDATE injury_report SET player_id = ? WHERE player_id = ?",
        (keep_id, remove_id),
    )
    updates += cur.rowcount
    
    # Delete the duplicate player record
    conn.execute("DELETE FROM players WHERE id = ?", (remove_id,))
    
    conn.commit()
    return updates

