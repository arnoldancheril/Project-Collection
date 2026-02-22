#!/usr/bin/env python3
"""
Analyze sportsbook lines vs actual player stats to understand how lines are calculated.
"""
import sqlite3
from collections import defaultdict
from pathlib import Path


def analyze_player_averages_vs_lines():
    """Compare player season averages with sportsbook lines."""
    db_path = Path(__file__).parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get top players by games played with their averages
    print("=" * 80)
    print("TOP SCORERS - SEASON AVERAGES (min 10 games, 20+ mins)")
    print("=" * 80)
    
    players = conn.execute("""
        SELECT 
            p.id,
            p.name,
            ROUND(AVG(bp.pts), 1) as avg_pts,
            ROUND(AVG(bp.reb), 1) as avg_reb,
            ROUND(AVG(bp.ast), 1) as avg_ast,
            ROUND(AVG(bp.minutes), 1) as avg_min,
            COUNT(*) as games
        FROM boxscore_player bp
        JOIN players p ON p.id = bp.player_id
        WHERE bp.status = 'PLAYED' AND bp.minutes > 20
        GROUP BY p.id
        HAVING COUNT(*) >= 10
        ORDER BY AVG(bp.pts) DESC
        LIMIT 50
    """).fetchall()
    
    print(f"{'Player':<28} {'PTS':>6} {'REB':>6} {'AST':>6} {'MIN':>6} {'GP':>4}")
    print("-" * 58)
    for p in players:
        print(f"{p['name']:<28} {p['avg_pts']:>6} {p['avg_reb']:>6} {p['avg_ast']:>6} {p['avg_min']:>6} {p['games']:>4}")
    
    return players


def analyze_lines_by_date():
    """Analyze what lines we have for each date."""
    db_path = Path(__file__).parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print("\n" + "=" * 80)
    print("SPORTSBOOK LINES BY DATE")
    print("=" * 80)
    
    dates = conn.execute("""
        SELECT as_of_date, COUNT(*) as count, COUNT(DISTINCT book) as books
        FROM sportsbook_lines
        GROUP BY as_of_date
        ORDER BY as_of_date DESC
    """).fetchall()
    
    for d in dates:
        print(f"{d['as_of_date']}: {d['count']} lines from {d['books']} books")
    
    return dates


def analyze_redundancy():
    """Analyze how many duplicate lines we have per player/prop."""
    db_path = Path(__file__).parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print("\n" + "=" * 80)
    print("LINE REDUNDANCY ANALYSIS (duplicate lines per player/prop)")
    print("=" * 80)
    
    # Check recent date
    recent_date = conn.execute("""
        SELECT as_of_date FROM sportsbook_lines 
        ORDER BY as_of_date DESC LIMIT 1
    """).fetchone()['as_of_date']
    
    duplicates = conn.execute("""
        SELECT 
            p.name,
            sl.prop_type,
            COUNT(*) as num_books,
            MIN(sl.line) as min_line,
            MAX(sl.line) as max_line,
            GROUP_CONCAT(sl.book || ':' || sl.line) as books_lines
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        WHERE sl.as_of_date = ?
        GROUP BY sl.player_id, sl.prop_type
        HAVING COUNT(*) > 1
        ORDER BY COUNT(*) DESC
        LIMIT 30
    """, (recent_date,)).fetchall()
    
    print(f"\nDate: {recent_date}")
    print(f"{'Player':<25} {'Type':>4} {'Books':>6} {'Min':>6} {'Max':>6} {'Spread':>7}")
    print("-" * 60)
    
    total_redundant = 0
    for d in duplicates:
        spread = d['max_line'] - d['min_line']
        print(f"{d['name']:<25} {d['prop_type']:>4} {d['num_books']:>6} {d['min_line']:>6.1f} {d['max_line']:>6.1f} {spread:>7.1f}")
        total_redundant += d['num_books'] - 1
    
    print(f"\nTotal redundant entries (could be removed): {total_redundant}")
    
    return duplicates


def compare_lines_to_actuals():
    """Compare sportsbook lines to what players actually scored."""
    db_path = Path(__file__).parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print("\n" + "=" * 80)
    print("LINES VS ACTUAL RESULTS (where we have both)")
    print("=" * 80)
    
    # Find dates where we have both lines and game results
    comparisons = conn.execute("""
        SELECT 
            sl.as_of_date,
            p.name,
            sl.prop_type,
            AVG(sl.line) as line,
            bp.pts,
            bp.reb,
            bp.ast
        FROM sportsbook_lines sl
        JOIN players p ON p.id = sl.player_id
        JOIN games g ON g.game_date = sl.as_of_date
        JOIN boxscore_player bp ON bp.player_id = p.id AND bp.game_id = g.id
        WHERE bp.status = 'PLAYED' AND bp.minutes > 0
        GROUP BY sl.as_of_date, sl.player_id, sl.prop_type
        ORDER BY sl.as_of_date DESC, p.name
        LIMIT 100
    """).fetchall()
    
    if not comparisons:
        print("No matching data found between lines and game results.")
        return []
    
    pts_diffs = []
    reb_diffs = []
    ast_diffs = []
    
    print(f"{'Date':<12} {'Player':<25} {'Type':>4} {'Line':>6} {'Actual':>7} {'Diff':>7} {'Result':>8}")
    print("-" * 80)
    
    for c in comparisons:
        if c['prop_type'] == 'PTS':
            actual = c['pts']
            pts_diffs.append(actual - c['line'])
        elif c['prop_type'] == 'REB':
            actual = c['reb']
            reb_diffs.append(actual - c['line'])
        elif c['prop_type'] == 'AST':
            actual = c['ast']
            ast_diffs.append(actual - c['line'])
        else:
            continue
            
        diff = actual - c['line']
        result = "OVER" if diff > 0 else "UNDER" if diff < 0 else "PUSH"
        print(f"{c['as_of_date']:<12} {c['name']:<25} {c['prop_type']:>4} {c['line']:>6.1f} {actual:>7} {diff:>+7.1f} {result:>8}")
    
    # Summary stats
    print("\n" + "-" * 40)
    print("SUMMARY STATISTICS")
    print("-" * 40)
    
    if pts_diffs:
        avg_diff = sum(pts_diffs) / len(pts_diffs)
        overs = sum(1 for d in pts_diffs if d > 0)
        unders = sum(1 for d in pts_diffs if d < 0)
        print(f"PTS: Avg diff: {avg_diff:+.2f}, Overs: {overs}, Unders: {unders}, Total: {len(pts_diffs)}")
    
    if reb_diffs:
        avg_diff = sum(reb_diffs) / len(reb_diffs)
        overs = sum(1 for d in reb_diffs if d > 0)
        unders = sum(1 for d in reb_diffs if d < 0)
        print(f"REB: Avg diff: {avg_diff:+.2f}, Overs: {overs}, Unders: {unders}, Total: {len(reb_diffs)}")
    
    if ast_diffs:
        avg_diff = sum(ast_diffs) / len(ast_diffs)
        overs = sum(1 for d in ast_diffs if d > 0)
        unders = sum(1 for d in ast_diffs if d < 0)
        print(f"AST: Avg diff: {avg_diff:+.2f}, Overs: {overs}, Unders: {unders}, Total: {len(ast_diffs)}")
    
    return comparisons


def analyze_line_setting_patterns():
    """Analyze how lines relate to season averages."""
    db_path = Path(__file__).parent.parent / "data" / "db" / "nba_props.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    print("\n" + "=" * 80)
    print("LINE VS SEASON AVERAGE COMPARISON")
    print("=" * 80)
    
    # Get most recent lines date
    recent_date = conn.execute("""
        SELECT as_of_date FROM sportsbook_lines 
        ORDER BY as_of_date DESC LIMIT 1
    """).fetchone()['as_of_date']
    
    # Compare lines to season averages
    comparison = conn.execute("""
        WITH player_avgs AS (
            SELECT 
                p.id,
                p.name,
                AVG(bp.pts) as avg_pts,
                AVG(bp.reb) as avg_reb,
                AVG(bp.ast) as avg_ast,
                AVG(bp.minutes) as avg_min,
                COUNT(*) as games
            FROM boxscore_player bp
            JOIN players p ON p.id = bp.player_id
            WHERE bp.status = 'PLAYED' AND bp.minutes > 10
            GROUP BY p.id
            HAVING COUNT(*) >= 5
        ),
        lines_agg AS (
            SELECT 
                player_id,
                prop_type,
                AVG(line) as line
            FROM sportsbook_lines
            WHERE as_of_date = ?
            GROUP BY player_id, prop_type
        )
        SELECT 
            pa.name,
            pa.avg_pts,
            pa.avg_reb,
            pa.avg_ast,
            pa.avg_min,
            pa.games,
            l_pts.line as pts_line,
            l_reb.line as reb_line,
            l_ast.line as ast_line
        FROM player_avgs pa
        LEFT JOIN lines_agg l_pts ON l_pts.player_id = pa.id AND l_pts.prop_type = 'PTS'
        LEFT JOIN lines_agg l_reb ON l_reb.player_id = pa.id AND l_reb.prop_type = 'REB'
        LEFT JOIN lines_agg l_ast ON l_ast.player_id = pa.id AND l_ast.prop_type = 'AST'
        WHERE l_pts.line IS NOT NULL OR l_reb.line IS NOT NULL
        ORDER BY pa.avg_pts DESC
        LIMIT 40
    """, (recent_date,)).fetchall()
    
    print(f"\nDate: {recent_date}")
    print(f"{'Player':<25} {'Avg PTS':>8} {'Line':>6} {'Diff':>6} | {'Avg REB':>8} {'Line':>6} {'Diff':>6} | {'GP':>3}")
    print("-" * 90)
    
    pts_diffs = []
    reb_diffs = []
    
    for c in comparison:
        pts_diff = ""
        if c['pts_line']:
            diff = c['pts_line'] - c['avg_pts']
            pts_diff = f"{diff:+.1f}"
            pts_diffs.append(diff)
        
        reb_diff = ""
        if c['reb_line']:
            diff = c['reb_line'] - c['avg_reb']
            reb_diff = f"{diff:+.1f}"
            reb_diffs.append(diff)
        
        pts_line = f"{c['pts_line']:.1f}" if c['pts_line'] else "-"
        reb_line = f"{c['reb_line']:.1f}" if c['reb_line'] else "-"
        
        print(f"{c['name']:<25} {c['avg_pts']:>8.1f} {pts_line:>6} {pts_diff:>6} | {c['avg_reb']:>8.1f} {reb_line:>6} {reb_diff:>6} | {c['games']:>3}")
    
    print("\n" + "-" * 40)
    print("LINE SETTING PATTERNS")
    print("-" * 40)
    
    if pts_diffs:
        avg_adj = sum(pts_diffs) / len(pts_diffs)
        print(f"PTS: Avg line adjustment from season avg: {avg_adj:+.2f}")
        print(f"     Lines are typically {abs(avg_adj):.1f} pts {'above' if avg_adj > 0 else 'below'} season avg")
    
    if reb_diffs:
        avg_adj = sum(reb_diffs) / len(reb_diffs)
        print(f"REB: Avg line adjustment from season avg: {avg_adj:+.2f}")
        print(f"     Lines are typically {abs(avg_adj):.1f} reb {'above' if avg_adj > 0 else 'below'} season avg")
    
    return comparison


if __name__ == "__main__":
    analyze_player_averages_vs_lines()
    analyze_lines_by_date()
    analyze_redundancy()
    compare_lines_to_actuals()
    analyze_line_setting_patterns()
