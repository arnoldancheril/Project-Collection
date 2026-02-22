#!/usr/bin/env python3
"""
Trade Setup & Post-Deadline Data Initialization
================================================

Runs all the post-trade-deadline setup:
1. Auto-detect trades from boxscores
2. Update games_with_new_team counts
3. Auto-update team roster status
4. Manually mark known tanking teams in the DB
5. Run tank detection on all teams
6. Print summary

Usage:
    python3 run_trade_setup.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nba_props.db import Db
from nba_props.paths import get_paths
from nba_props.engine.trade_tracker import (
    auto_detect_trades_from_boxscores,
    auto_update_team_roster_status,
    update_post_trade_game_counts,
    init_trade_tables,
    record_team_status,
)
from nba_props.engine.tank_detector import (
    detect_all_tanking_teams,
    clear_tank_detection_cache,
    KNOWN_TANKING_TEAMS,
)


# ============================================================================
# Known tanking teams for 2025-26 season (manually curated)
# Match these with the KNOWN_TANKING_TEAMS watchlist
# ============================================================================
MANUAL_TANKING_STATUS = {
    "UTA": {
        "is_tanking": True,
        "tank_confidence": 0.85,
        "record_at_deadline": "16-37",
        "notes": "Jazz benching healthy stars (Markkanen, Nurkic) in close 4th quarters to lose games. Sold Hendricks/Clayton to MEM.",
    },
    "WAS": {
        "is_tanking": True,
        "tank_confidence": 0.80,
        "record_at_deadline": "14-37",
        "notes": "Wizards ruling out healthy players, using G-League callups. Sold AJ Johnson, Marvin Bagley, Khris Middleton.",
    },
    "NOP": {
        "is_tanking": False,
        "tank_confidence": 0.0,
        "record_at_deadline": "14-40",
        "notes": "Pelicans rebuilding due to injuries; not intentional tanking but depleted roster.",
    },
    "IND": {
        "is_tanking": False,
        "tank_confidence": 0.0,
        "record_at_deadline": "13-39",
        "notes": "Pacers in a tough spot but not intentionally tanking.",
    },
    "BKN": {
        "is_tanking": False,
        "tank_confidence": 0.0,
        "record_at_deadline": "14-37",
        "notes": "Nets rebuilding. Traded Cam Thomas to MIL. Poor record but roster young/rebuilding.",
    },
    "SAC": {
        "is_tanking": False,
        "tank_confidence": 0.0,
        "record_at_deadline": "12-41",
        "notes": "Kings have terrible record but not intentionally tanking — playing stars full minutes.",
    },
}


def main():
    paths = get_paths()
    db = Db(paths.db_path)

    with db.connect() as conn:
        init_trade_tables(conn)

        print("=" * 60)
        print("  TRADE & TANK SETUP — Post-Trade-Deadline 2026")
        print("=" * 60)
        print()

        # Step 1: Auto-detect trades from boxscores
        print("Step 1: Auto-detecting trades from boxscores (since 2026-01-20)")
        print("-" * 60)
        new_trades = auto_detect_trades_from_boxscores(
            conn, since_date="2026-01-20", verbose=True
        )
        print()

        # Step 2: Update games_with_new_team counts
        print("Step 2: Updating games_with_new_team counts")
        print("-" * 60)
        updated = update_post_trade_game_counts(conn)
        print(f"  Updated {updated} trade records with new games count")
        print()

        # Step 3: Auto-update team roster status
        print("Step 3: Auto-updating team roster status from trades")
        print("-" * 60)
        team_statuses = auto_update_team_roster_status(conn, verbose=True)
        print()

        # Step 4: Manually mark known tanking teams
        print("Step 4: Applying manual tanking flags to known teams")
        print("-" * 60)
        for team_abbrev, info in MANUAL_TANKING_STATUS.items():
            record_team_status(
                conn,
                team_abbrev=team_abbrev,
                is_tanking=info["is_tanking"],
                tank_confidence=info["tank_confidence"],
                record_at_deadline=info["record_at_deadline"],
                notes=info["notes"],
            )
        print()

        # Step 5: Tank detection on all teams (with fixed algorithm)
        print("Step 5: Running tank detection on all teams (V19.4 fixed)")
        print("-" * 60)
        print(f"  Known tanking teams watchlist: {list(KNOWN_TANKING_TEAMS.keys())}")
        clear_tank_detection_cache()
        tank_results = detect_all_tanking_teams(conn, as_of_date="2026-02-12")
        tanking = [r for r in tank_results if r.is_tanking]
        monitoring = [r for r in tank_results if not r.is_tanking and r.confidence > 0.15]

        print(f"\n  Confirmed tanking teams ({len(tanking)}):")
        for r in sorted(tanking, key=lambda x: x.confidence, reverse=True):
            print(f"    {r.team_abbrev}: confidence={r.confidence:.0%}, "
                  f"record={r.team_record}, minutes_factor={r.star_minutes_factor:.2f}")
            for s in sorted(r.signals, key=lambda x: x.strength, reverse=True)[:2]:
                print(f"      [{s.signal_type}] {s.description}")

        print(f"\n  Monitoring (not confirmed tanking) ({len(monitoring)}):")
        for r in sorted(monitoring, key=lambda x: x.confidence, reverse=True)[:8]:
            print(f"    {r.team_abbrev}: confidence={r.confidence:.0%}, record={r.team_record}")
        print()

        # Full trade summary
        print("=" * 60)
        print("  COMPLETE TRADE DATABASE SUMMARY")
        print("=" * 60)
        rows = conn.execute(
            """SELECT player_name, from_team, to_team, trade_date, 
                      old_team_role, games_with_new_team 
               FROM player_trades 
               ORDER BY trade_date, player_name"""
        ).fetchall()
        for r in rows:
            print(f"  {r['player_name']}: {r['from_team']} -> {r['to_team']} "
                  f"on {r['trade_date']} [{r['old_team_role']}] "
                  f"({r['games_with_new_team']} games)")
        print(f"\n  Total: {len(rows)} trade records")
        print()

        # Team roster status summary
        print("=" * 60)
        print("  TEAM ROSTER STATUS SUMMARY")
        print("=" * 60)
        rows2 = conn.execute(
            """SELECT team_abbrev, is_tanking, tank_confidence, 
                      players_traded_away, players_acquired, 
                      roster_stability_score
               FROM team_roster_status 
               ORDER BY is_tanking DESC, tank_confidence DESC"""
        ).fetchall()
        for r in rows2:
            tank = "TANKING" if r["is_tanking"] else "Competing"
            conf = f" ({r['tank_confidence']:.0%} conf)" if r["is_tanking"] else ""
            print(f"  {r['team_abbrev']:5s} [{tank:10s}]{conf} "
                  f"{r['players_traded_away']} out / {r['players_acquired']} in "
                  f"(stability: {r['roster_stability_score']:.2f})")
        print(f"\n  Total: {len(rows2)} teams tracked")


if __name__ == "__main__":
    main()



def main():
    paths = get_paths()
    db = Db(paths.db_path)

    with db.connect() as conn:
        init_trade_tables(conn)

        print("=" * 60)
        print("  TRADE & TANK SETUP — Post-Trade-Deadline 2026")
        print("=" * 60)
        print()

        # Step 1: Auto-detect trades from boxscores
        print("Step 1: Auto-detecting trades from boxscores (since 2026-01-20)")
        print("-" * 60)
        new_trades = auto_detect_trades_from_boxscores(
            conn, since_date="2026-01-20", verbose=True
        )
        print()

        # Step 2: Update games_with_new_team counts
        print("Step 2: Updating games_with_new_team counts")
        print("-" * 60)
        updated = update_post_trade_game_counts(conn)
        print(f"  Updated {updated} trade records with new games count")
        print()

        # Step 3: Auto-update team roster status
        print("Step 3: Auto-updating team roster status")
        print("-" * 60)
        team_statuses = auto_update_team_roster_status(conn, verbose=True)
        print()

        # Step 4: Tank detection on all teams
        print("Step 4: Running tank detection on all teams")
        print("-" * 60)
        print(f"  Known tanking teams watchlist: {list(KNOWN_TANKING_TEAMS.keys())}")
        clear_tank_detection_cache()
        tank_results = detect_all_tanking_teams(conn, as_of_date="2026-02-12")
        tanking = [r for r in tank_results if r.is_tanking]
        print(f"  Tanking teams detected ({len(tanking)}):")
        for r in sorted(tanking, key=lambda x: x.confidence, reverse=True):
            print(f"    {r.team_abbrev}: confidence={r.confidence:.0%}, "
                  f"record={r.team_record}, minutes_factor={r.star_minutes_factor:.2f}")
            for s in sorted(r.signals, key=lambda x: x.strength, reverse=True)[:3]:
                print(f"      [{s.signal_type}] {s.description}")
        print()

        # Full trade summary
        print("=" * 60)
        print("  COMPLETE TRADE DATABASE SUMMARY")
        print("=" * 60)
        rows = conn.execute(
            """SELECT player_name, from_team, to_team, trade_date, 
                      old_team_role, games_with_new_team 
               FROM player_trades 
               ORDER BY trade_date, player_name"""
        ).fetchall()
        for r in rows:
            print(f"  {r['player_name']}: {r['from_team']} -> {r['to_team']} "
                  f"on {r['trade_date']} [{r['old_team_role']}] "
                  f"({r['games_with_new_team']} games with new team)")
        print(f"\n  Total: {len(rows)} trade records")
        print()

        # Team roster status summary
        print("=" * 60)
        print("  TEAM ROSTER STATUS SUMMARY")
        print("=" * 60)
        rows2 = conn.execute(
            """SELECT team_abbrev, is_tanking, tank_confidence, 
                      players_traded_away, players_acquired, 
                      roster_stability_score, star_players_lost
               FROM team_roster_status 
               ORDER BY team_abbrev"""
        ).fetchall()
        for r in rows2:
            tank = "TANKING" if r["is_tanking"] else "Competing"
            print(f"  {r['team_abbrev']:5s} [{tank:10s}] "
                  f"{r['players_traded_away']} out / {r['players_acquired']} in "
                  f"(stability: {r['roster_stability_score']:.2f})")
        print(f"\n  Total: {len(rows2)} teams tracked")


if __name__ == "__main__":
    main()
