#!/usr/bin/env python3
"""
Bulk Box Score Scraper & Ingester
=================================

Scrapes NBA box scores from NBA.com JSON API for a date range and ingests
them into the SQLite database.

Usage:
    python3 scripts/bulk_scrape_and_ingest.py --start 2026-01-26 --end 2026-02-19
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.nba_props.db import Db, init_db
from src.nba_props.paths import get_paths
from src.nba_props.ingest.web_scraper import (
    scrape_box_scores_for_date,
    ScrapedBoxScore,
    ScrapedPlayerStats,
)
from src.nba_props.ingest.boxscore_ingest import ingest_boxscore_file

ALL_STAR_BREAK_DATES = {
    # 2026 All-Star Weekend & Break:
    # Feb 13 = Rising Stars Challenge (G-League / rookies, not regular-season)
    # Feb 14 = All-Star Friday (Skills, Dunk, 3-Point Contest)
    # Feb 15 = All-Star Saturday
    # Feb 16 = All-Star Game (Sunday)
    # Feb 17 = Presidents Day / break day (no regular games)
    "2026-02-13", "2026-02-14", "2026-02-15", "2026-02-16", "2026-02-17",
}


def scraped_boxscore_to_text(bs: ScrapedBoxScore) -> str:
    """Convert ScrapedBoxScore to the tab-separated format the parser expects."""
    lines = []
    by_team = {}
    for ps in bs.player_stats:
        if ps.team not in by_team:
            by_team[ps.team] = []
        by_team[ps.team].append(ps)

    for team_name, players in by_team.items():
        # Use plain team name (not ## markdown) so parser uses _parse_tabbed_boxscore
        lines.append(team_name)
        lines.append(
            "PLAYER\tMIN\tFGM\tFGA\tFG%\t3PM\t3PA\t3P%\tFTM\tFTA\tFT%\t"
            "OREB\tDREB\tREB\tAST\tSTL\tBLK\tTO\tPF\tPTS\t+/-"
        )
        for p in players:
            def parse_ma(val):
                if val and "-" in str(val):
                    parts = str(val).split("-")
                    return parts[0], parts[1]
                return "0", "0"
            fgm, fga = parse_ma(p.fg)
            tpm, tpa = parse_ma(p.tp)
            ftm, fta = parse_ma(p.ft)
            mins = p.minutes or "0:00"
            fg_pct = p.fg_pct or "0.0"
            tp_pct = p.tp_pct or "0.0"
            ft_pct = p.ft_pct or "0.0"
            oreb = p.oreb if p.oreb is not None else 0
            dreb = p.dreb if p.dreb is not None else 0
            reb = p.reb if p.reb is not None else 0
            ast = p.ast if p.ast is not None else 0
            stl = p.stl if p.stl is not None else 0
            blk = p.blk if p.blk is not None else 0
            tov = p.tov if p.tov is not None else 0
            pf = p.pf if p.pf is not None else 0
            pts = p.pts if p.pts is not None else 0
            pm = p.plus_minus if p.plus_minus is not None else 0
            pm_str = f"+{pm}" if pm > 0 else str(pm)
            lines.append(
                f"{p.player_name}\t{mins}\t{fgm}\t{fga}\t{fg_pct}\t"
                f"{tpm}\t{tpa}\t{tp_pct}\t{ftm}\t{fta}\t{ft_pct}\t"
                f"{oreb}\t{dreb}\t{reb}\t{ast}\t{stl}\t{blk}\t{tov}\t{pf}\t{pts}\t{pm_str}"
            )
        lines.append("")
    return "\n".join(lines)


def _infer_season(game_date):
    yyyy = int(game_date[0:4])
    mm = int(game_date[5:7])
    start_year = yyyy if mm >= 10 else yyyy - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def _get_team_abbrev_safe(team_name):
    from src.nba_props.team_aliases import abbrev_from_team_name
    try:
        return abbrev_from_team_name(team_name)
    except Exception:
        words = team_name.split()
        return words[-1][:3].upper() if words else "UNK"


def scrape_and_save_date(date_str, paths, verbose=True):
    saved_files = []
    if date_str in ALL_STAR_BREAK_DATES:
        if verbose:
            print(f"  Skipping All-Star Break: {date_str}")
        return saved_files

    season = _infer_season(date_str)
    date_dir = paths.raw_dir / "boxscores" / season / date_str
    if date_dir.exists() and any(date_dir.iterdir()):
        existing = list(date_dir.glob("*.txt"))
        if verbose:
            print(f"  Already have {len(existing)} files for {date_str}")
        return existing

    try:
        box_scores = scrape_box_scores_for_date(date_str)
    except Exception as e:
        print(f"  Error scraping {date_str}: {e}")
        return saved_files

    if not box_scores:
        if verbose:
            print(f"  No completed games for {date_str}")
        return saved_files

    if verbose:
        print(f"  Found {len(box_scores)} completed games")

    for bs in box_scores:
        if not bs.player_stats:
            continue
        text_content = scraped_boxscore_to_text(bs)
        date_dir.mkdir(parents=True, exist_ok=True)
        away_abbrev = _get_team_abbrev_safe(bs.away_team)
        home_abbrev = _get_team_abbrev_safe(bs.home_team)
        filename = f"{away_abbrev}_vs_{home_abbrev}.txt"
        file_path = date_dir / filename
        file_path.write_text(text_content, encoding="utf-8")
        saved_files.append(file_path)
        score_str = ""
        if bs.away_score and bs.home_score:
            score_str = f" ({bs.away_score}-{bs.home_score})"
        if verbose:
            print(f"  Saved: {filename}{score_str}")
    return saved_files


def ingest_date_files(date_str, paths, db_path, verbose=True):
    season = _infer_season(date_str)
    date_dir = paths.raw_dir / "boxscores" / season / date_str
    if not date_dir.exists():
        return 0
    txt_files = sorted(date_dir.glob("*.txt"))
    if not txt_files:
        return 0
    db = Db(path=db_path)
    ingested = 0
    for txt_file in txt_files:
        try:
            with db.connect() as conn:
                game_id = ingest_boxscore_file(conn, source_file=txt_file)
                conn.commit()
                ingested += 1
                if verbose:
                    print(f"  Ingested: {txt_file.name} -> game_id={game_id}")
        except Exception as e:
            err_msg = str(e)
            if "Expected 2 teams" in err_msg:
                if verbose:
                    print(f"  Skip {txt_file.name}: {err_msg}")
            else:
                if verbose:
                    print(f"  Error {txt_file.name}: {err_msg}")
    return ingested


def generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def main():
    parser = argparse.ArgumentParser(description="Bulk scrape and ingest NBA box scores")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--scrape-only", action="store_true")
    parser.add_argument("--ingest-only", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    paths = get_paths()
    db_path = paths.db_path
    init_db(db_path)

    dates = generate_date_range(args.start, args.end)
    print(f"Date range: {args.start} -> {args.end} ({len(dates)} dates)")
    print(f"All-Star excluded: {sorted(ALL_STAR_BREAK_DATES)}")

    total_scraped = 0
    total_ingested = 0

    for i, date_str in enumerate(dates):
        if date_str in ALL_STAR_BREAK_DATES:
            print(f"[{i+1}/{len(dates)}] {date_str} - All-Star Break, skipping")
            continue
        print(f"\n[{i+1}/{len(dates)}] {date_str}")

        if not args.ingest_only:
            saved = scrape_and_save_date(date_str, paths)
            total_scraped += len(saved)
            if i < len(dates) - 1:
                time.sleep(args.delay)

        if not args.scrape_only:
            count = ingest_date_files(date_str, paths, db_path)
            total_ingested += count

    print(f"\n{'='*50}")
    print(f"DONE: Scraped {total_scraped} files, Ingested {total_ingested} games")
    print(f"{'='*50}")


if __name__ == "__main__":
    sys.exit(main())
