"""Flask web application for NBA Props Predictor."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from ..db import Db, init_db
from ..ingest import ingest_boxscore_file
from ..ingest.paste import save_pasted_boxscore_text
from ..ingest.lines_parser import parse_lines_text
from ..ingest.matchups_parser import parse_matchups_text, parse_simple_matchup
from ..paths import get_paths
from ..standings import compute_conference_standings, compute_player_averages_for_team
from ..team_aliases import abbrev_from_team_name, team_name_from_abbrev
from ..engine.projector import project_team_players, ProjectionConfig
from ..engine.game_context import get_back_to_back_status, get_team_defense_rating, apply_matchup_adjustments
from ..engine.edge_calculator import generate_prop_report, calculate_prop_edge
from ..engine.archetypes import get_player_archetype, classify_player_by_stats, KNOWN_ARCHETYPES
from ..engine.archetype_db import (
    get_player_archetype_db,
    get_all_archetypes_db,
    update_player_archetype,
    delete_player_archetype,
    seed_archetypes_from_defaults,
    get_similarity_groups_db,
    get_elite_defenders_db,
    get_similar_players_db,
    should_avoid_betting_over_db,
    get_roster_for_team_db,
    get_archetype_count_db,
    toggle_star_status,
    get_star_players_for_team,
    get_all_star_players,
    is_star_player,
    set_bet_status,
    get_bet_status,
    BET_STATUS_AVOID,
    BET_STATUS_NEUTRAL,
    BET_STATUS_STAR,
)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    paths = get_paths()
    
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["SECRET_KEY"] = "nba-props-local-dev"
    
    # Initialize database
    init_db(paths.db_path)
    
    def get_db() -> Db:
        return Db(path=paths.db_path)
    
    # -------------------------------------------------------------------------
    # Pages
    # -------------------------------------------------------------------------
    
    @app.route("/")
    def index():
        """Main dashboard page."""
        return render_template("index.html")
    
    @app.route("/games")
    def games_page():
        """Games list page."""
        return render_template("games.html")
    
    @app.route("/paste")
    def paste_page():
        """Paste box score page."""
        return render_template("paste.html")
    
    @app.route("/projections")
    def projections_page():
        """Projections and props page."""
        return render_template("projections.html")
    
    @app.route("/teams")
    def teams_page():
        """Teams overview page."""
        return render_template("teams.html")
    
    @app.route("/team/<abbrev>")
    def team_detail_page(abbrev: str):
        """Team detail page."""
        return render_template("team_detail.html", team_abbrev=abbrev.upper())
    
    @app.route("/data")
    def data_page():
        """Data management page."""
        return render_template("data.html")
    
    @app.route("/players")
    def players_page():
        """Unified players page with roster, archetypes, and matchup analysis."""
        return render_template("players.html")
    
    @app.route("/matchups")
    def matchups_page():
        """Today's matchups and predictions page."""
        return render_template("matchups.html")
    
    # Legacy routes - redirect to unified players page
    @app.route("/archetypes")
    def archetypes_page():
        """Player archetypes page - redirects to players."""
        return render_template("players.html")
    
    @app.route("/roster")
    def roster_page():
        """Player roster with detailed archetypes - redirects to players."""
        return render_template("players.html")
    
    @app.route("/backtesting")
    def backtesting_page():
        """Model testing and backtesting dashboard."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Find a good default date: last date with picks or scheduled games
        db = get_db()
        default_date = today
        with db.connect() as conn:
            # Try to get the most recent date with picks first
            last_pick = conn.execute(
                "SELECT pick_date FROM model_picks ORDER BY pick_date DESC LIMIT 1"
            ).fetchone()
            if last_pick:
                default_date = last_pick["pick_date"]
            else:
                # Fall back to most recent scheduled game date
                last_game = conn.execute(
                    "SELECT game_date FROM scheduled_games ORDER BY game_date DESC LIMIT 1"
                ).fetchone()
                if last_game:
                    default_date = last_game["game_date"]
        
        return render_template("backtesting.html", today=today, default_date=default_date)
    
    @app.route("/matchup")
    def matchup_page():
        """Matchup analysis with defender tracking - redirects to players."""
        return render_template("players.html")
    
    # -------------------------------------------------------------------------
    # API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/stats")
    def api_stats():
        """Get database statistics."""
        db = get_db()
        with db.connect() as conn:
            games = conn.execute("SELECT COUNT(*) AS n FROM games").fetchone()["n"]
            players = conn.execute("SELECT COUNT(*) AS n FROM players").fetchone()["n"]
            teams = conn.execute("SELECT COUNT(*) AS n FROM teams").fetchone()["n"]
            lines = conn.execute("SELECT COUNT(*) AS n FROM sportsbook_lines").fetchone()["n"]
            
            # Get archetype count from DB
            archetypes_db = get_archetype_count_db(conn)
            
            # Get latest game date
            latest = conn.execute(
                "SELECT game_date FROM games ORDER BY game_date DESC LIMIT 1"
            ).fetchone()
            latest_date = latest["game_date"] if latest else None
            
        # Get default archetype count
        from ..engine.roster import PLAYER_DATABASE
        archetypes_default = len(PLAYER_DATABASE)
            
        return jsonify({
            "games": games,
            "players": players,
            "teams": teams,
            "lines": lines,
            "archetypes_db": archetypes_db,
            "archetypes_default": archetypes_default,
            "latest_game_date": latest_date,
        })
    
    @app.route("/api/games")
    def api_games():
        """Get list of games."""
        limit = request.args.get("limit", 50, type=int)
        db = get_db()
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT g.id, g.game_date, g.season,
                       t1.name AS team1, t2.name AS team2,
                       tt1.pts AS team1_pts, tt2.pts AS team2_pts
                FROM games g
                JOIN teams t1 ON t1.id = g.team1_id
                JOIN teams t2 ON t2.id = g.team2_id
                LEFT JOIN boxscore_team_totals tt1 ON tt1.game_id = g.id AND tt1.team_id = g.team1_id
                LEFT JOIN boxscore_team_totals tt2 ON tt2.game_id = g.id AND tt2.team_id = g.team2_id
                ORDER BY g.game_date DESC, g.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        
        games = []
        for r in rows:
            games.append({
                "id": r["id"],
                "date": r["game_date"],
                "season": r["season"],
                "team1": r["team1"],
                "team2": r["team2"],
                "team1_abbrev": abbrev_from_team_name(r["team1"]) or "",
                "team2_abbrev": abbrev_from_team_name(r["team2"]) or "",
                "team1_pts": r["team1_pts"],
                "team2_pts": r["team2_pts"],
            })
        return jsonify({"games": games})
    
    @app.route("/api/game/<int:game_id>")
    def api_game_detail(game_id: int):
        """Get detailed game information."""
        db = get_db()
        with db.connect() as conn:
            game = conn.execute(
                """
                SELECT g.id, g.game_date, g.season,
                       t1.name AS team1, t2.name AS team2
                FROM games g
                JOIN teams t1 ON t1.id = g.team1_id
                JOIN teams t2 ON t2.id = g.team2_id
                WHERE g.id = ?
                """,
                (game_id,),
            ).fetchone()
            
            if not game:
                return jsonify({"error": "Game not found"}), 404
            
            players = conn.execute(
                """
                SELECT t.name AS team, p.name AS player, 
                       b.pos, b.status, b.minutes, b.pts, b.reb, b.ast,
                       b.fgm, b.fga, b.tpm, b.tpa, b.ftm, b.fta,
                       b.stl, b.blk, b.tov, b.plus_minus
                FROM boxscore_player b
                JOIN teams t ON t.id = b.team_id
                JOIN players p ON p.id = b.player_id
                WHERE b.game_id = ?
                ORDER BY t.name, (b.minutes IS NULL) ASC, b.minutes DESC
                """,
                (game_id,),
            ).fetchall()
            
            totals = conn.execute(
                """
                SELECT t.name AS team, tt.pts, tt.reb, tt.ast
                FROM boxscore_team_totals tt
                JOIN teams t ON t.id = tt.team_id
                WHERE tt.game_id = ?
                """,
                (game_id,),
            ).fetchall()
        
        return jsonify({
            "game": {
                "id": game["id"],
                "date": game["game_date"],
                "season": game["season"],
                "team1": game["team1"],
                "team2": game["team2"],
            },
            "players": [dict(p) for p in players],
            "totals": {t["team"]: {"pts": t["pts"], "reb": t["reb"], "ast": t["ast"]} for t in totals},
        })
    
    @app.route("/api/standings")
    def api_standings():
        """Get conference standings."""
        db = get_db()
        with db.connect() as conn:
            standings = compute_conference_standings(conn)
        
        result = {"East": [], "West": []}
        for conf in ["East", "West"]:
            for row in standings.get(conf, []):
                result[conf].append({
                    "seed": row.seed,
                    "abbrev": row.abbr,
                    "team": row.team_name,
                    "wins": row.wins,
                    "losses": row.losses,
                    "win_pct": round(row.win_pct, 3) if row.win_pct else 0,
                })
        return jsonify(result)
    
    @app.route("/api/team/<abbrev>")
    def api_team_detail(abbrev: str):
        """Get team details and player averages."""
        abbrev = abbrev.upper()
        team_name = team_name_from_abbrev(abbrev)
        if not team_name:
            return jsonify({"error": "Unknown team"}), 404
        
        db = get_db()
        with db.connect() as conn:
            standings = compute_conference_standings(conn)
            players = compute_player_averages_for_team(conn, abbrev)
            
            # Find team record
            team_record = None
            for conf in ["East", "West"]:
                for row in standings.get(conf, []):
                    if row.abbr == abbrev:
                        team_record = {
                            "conference": conf,
                            "seed": row.seed,
                            "wins": row.wins,
                            "losses": row.losses,
                            "win_pct": round(row.win_pct, 3) if row.win_pct else 0,
                        }
                        break
        
        return jsonify({
            "abbrev": abbrev,
            "name": team_name,
            "record": team_record,
            "players": players,
        })
    
    @app.route("/api/team/<abbrev>/dashboard")
    def api_team_dashboard(abbrev: str):
        """Get comprehensive team dashboard data including roster archetypes."""
        from ..engine.roster import get_roster_for_team, PLAYER_DATABASE, PlayerTier
        from ..engine.game_context import get_team_defense_rating
        
        # Check if we should use DB-backed archetypes
        use_db_archetypes = request.args.get("use_db", "true").lower() == "true"
        
        abbrev = abbrev.upper()
        team_name = team_name_from_abbrev(abbrev)
        if not team_name:
            return jsonify({"error": "Unknown team"}), 404
        
        db = get_db()
        with db.connect() as conn:
            standings = compute_conference_standings(conn)
            player_stats = compute_player_averages_for_team(conn, abbrev)
            
            # Find team record
            team_record = None
            for conf in ["East", "West"]:
                for row in standings.get(conf, []):
                    if row.abbr == abbrev:
                        team_record = {
                            "conference": conf,
                            "seed": row.seed,
                            "wins": row.wins,
                            "losses": row.losses,
                            "win_pct": round(row.win_pct, 3) if row.win_pct else 0,
                        }
                        break
            
            # Get recent games for this team
            recent_games = conn.execute(
                """
                SELECT g.id, g.game_date, 
                       t1.name AS team1, t2.name AS team2,
                       tt1.pts AS team1_pts, tt2.pts AS team2_pts
                FROM games g
                JOIN teams t1 ON t1.id = g.team1_id
                JOIN teams t2 ON t2.id = g.team2_id
                LEFT JOIN boxscore_team_totals tt1 ON tt1.game_id = g.id AND tt1.team_id = g.team1_id
                LEFT JOIN boxscore_team_totals tt2 ON tt2.game_id = g.id AND tt2.team_id = g.team2_id
                WHERE t1.name = ? OR t2.name = ?
                ORDER BY g.game_date DESC
                LIMIT 10
                """,
                (team_name, team_name),
            ).fetchall()
            
            # Get recent player performances (last 5 games)
            recent_performances = conn.execute(
                """
                SELECT p.name, g.game_date,
                       b.pts, b.reb, b.ast, b.minutes
                FROM boxscore_player b
                JOIN players p ON p.id = b.player_id
                JOIN games g ON g.id = b.game_id
                JOIN teams t ON t.id = b.team_id
                WHERE t.name = ?
                ORDER BY g.game_date DESC, b.pts DESC
                LIMIT 50
                """,
                (team_name,),
            ).fetchall()
            
            # Get team defense rating
            try:
                defense_rating = get_team_defense_rating(conn, abbrev)
            except Exception:
                defense_rating = None
        
        # Get roster archetypes - prefer DB, fallback to static PLAYER_DATABASE
        roster_profiles = []
        elite_defenders = []
        star_players = []
        
        if use_db_archetypes:
            # Try to get from database first
            db_roster = get_roster_for_team_db(conn, team_name)
            if db_roster:
                for profile in db_roster:
                    player_data = {
                        "name": profile.player_name,
                        "position": profile.position,
                        "height": profile.height,
                        "primary_offensive": profile.primary_offensive,
                        "secondary_offensive": profile.secondary_offensive,
                        "defensive_role": profile.defensive_role,
                        "tier": f"TIER_{profile.tier}",
                        "tier_value": profile.tier,
                        "is_elite_defender": profile.is_elite_defender,
                        "is_star": profile.is_star,
                        "bet_status": profile.bet_status,
                        "strengths": profile.strengths,
                        "weaknesses": profile.weaknesses,
                        "notes": profile.notes,
                        "guards_positions": profile.guards_positions,
                        "source": profile.source,
                    }
                    roster_profiles.append(player_data)
        
        # Fallback to static PLAYER_DATABASE if no DB data
        if not roster_profiles:
            team_roster = get_roster_for_team(team_name)
            for profile in team_roster:
                player_data = {
                    "name": profile.name,
                    "position": profile.position,
                    "height": profile.height,
                    "primary_offensive": profile.primary_offensive.value,
                    "secondary_offensive": profile.secondary_offensive.value if profile.secondary_offensive else None,
                    "defensive_role": profile.defensive_role.value,
                    "tier": profile.tier.name,
                    "tier_value": profile.tier.value,
                    "is_elite_defender": profile.is_elite_defender,
                    "is_star": False,  # Static players default to not star
                    "bet_status": BET_STATUS_NEUTRAL,  # Default to neutral
                    "strengths": profile.strengths,
                    "weaknesses": profile.weaknesses,
                    "notes": profile.notes,
                    "guards_positions": profile.guards_positions,
                    "source": "default",
                }
                roster_profiles.append(player_data)
        
        # Build elite defenders and star players from roster_profiles
        for player_data in roster_profiles:
            if player_data.get("is_elite_defender"):
                elite_defenders.append(player_data)
            
            # Use bet_status to determine star status
            bet_status = player_data.get("bet_status", BET_STATUS_NEUTRAL)
            if bet_status == BET_STATUS_STAR:
                star_players.append(player_data)
            elif bet_status == BET_STATUS_NEUTRAL and player_data.get("tier_value", 6) <= 2:
                # Fallback: consider tier 1-2 (MVP, Two-Way Star) as star if neutral
                star_players.append(player_data)
        
        # Sort roster by tier, then by bet_status (stars first)
        roster_profiles.sort(key=lambda x: (-(x.get("bet_status", 1)), x["tier_value"], x["name"]))
        
        # Merge player stats with roster profiles
        stats_by_name = {p["player"]: p for p in player_stats}
        for profile in roster_profiles:
            if profile["name"] in stats_by_name:
                profile["stats"] = stats_by_name[profile["name"]]
        
        # Calculate hot players (best recent performers)
        hot_players = []
        player_recent_games = {}
        for perf in recent_performances:
            name = perf["name"]
            if name not in player_recent_games:
                player_recent_games[name] = []
            if len(player_recent_games[name]) < 5:
                player_recent_games[name].append({
                    "date": perf["game_date"],
                    "pts": perf["pts"],
                    "reb": perf["reb"],
                    "ast": perf["ast"],
                    "minutes": perf["minutes"],
                })
        
        for name, games in player_recent_games.items():
            if len(games) >= 2:
                avg_pts = sum(g["pts"] or 0 for g in games) / len(games)
                avg_reb = sum(g["reb"] or 0 for g in games) / len(games)
                avg_ast = sum(g["ast"] or 0 for g in games) / len(games)
                hot_players.append({
                    "name": name,
                    "games": len(games),
                    "avg_pts": round(avg_pts, 1),
                    "avg_reb": round(avg_reb, 1),
                    "avg_ast": round(avg_ast, 1),
                    "recent_games": games[:3],
                })
        
        # Sort by average points
        hot_players.sort(key=lambda x: x["avg_pts"], reverse=True)
        
        # Format recent games
        formatted_games = []
        for game in recent_games:
            is_team1 = game["team1"] == team_name
            opponent = game["team2"] if is_team1 else game["team1"]
            team_pts = game["team1_pts"] if is_team1 else game["team2_pts"]
            opp_pts = game["team2_pts"] if is_team1 else game["team1_pts"]
            
            if team_pts and opp_pts:
                result = "W" if team_pts > opp_pts else "L"
            else:
                result = None
            
            formatted_games.append({
                "date": game["game_date"],
                "opponent": opponent,
                "opponent_abbrev": abbrev_from_team_name(opponent),
                "team_pts": team_pts,
                "opp_pts": opp_pts,
                "result": result,
                "home": not is_team1,
            })
        
        # Team descriptions (static data for now)
        team_descriptions = {
            "BOS": "The defending champions feature elite two-way players and a deep roster. Known for ball movement and versatile defenders.",
            "MIL": "Built around Giannis Antetokounmpo's dominance at the rim. High-powered offense with improving perimeter defense.",
            "PHI": "Joel Embiid anchors both ends. Physical, half-court oriented team with strong interior defense.",
            "NYK": "Defensive-minded team under Thibodeau. Physical play style with strong rebounding.",
            "CLE": "Young, athletic core with elite guard play. Aggressive defense and transition offense.",
            "MIA": "The Heat Culture emphasizes toughness and defense. Zone defense specialists with 3-point shooting.",
            "ATL": "Trae Young's playmaking drives the offense. High-volume shooting team building toward contention.",
            "CHI": "Dynamic backcourt with solid veterans. Balanced scoring with room for growth.",
            "IND": "Tyrese Haliburton runs a fast-paced offense. Elite pace and transition scoring.",
            "ORL": "Young defensive core led by Paolo Banchero. Long, athletic team still developing.",
            "DET": "Rebuilding around Cade Cunningham. Young roster gaining experience.",
            "TOR": "Versatile, switchable defenders. Unconventional roster construction with length.",
            "BKN": "Retooling roster with young talent. High-upside players developing together.",
            "CHA": "LaMelo Ball's creativity leads the offense. Athletic team working toward consistency.",
            "WAS": "Young roster in development phase. Focus on player growth and future assets.",
            "DEN": "Jokic orchestrates elite offense from the post. Back-to-back champions with deep playoff experience.",
            "OKC": "Young, athletic core with elite defense. Shai Gilgeous-Alexander leads a rising contender.",
            "MIN": "Elite defensive team with Anthony Edwards leading. Physical, playoff-tested roster.",
            "LAC": "Kawhi Leonard and Paul George lead a championship-caliber roster when healthy. Elite wing defenders.",
            "PHX": "Kevin Durant and Devin Booker form elite scoring duo. Superstar-driven offense.",
            "LAL": "LeBron James and Anthony Davis anchor a veteran contender. Size and experience.",
            "SAC": "De'Aaron Fox's speed drives fast-paced offense. Light the Beam! Exciting, up-tempo style.",
            "GSW": "Dynasty core with Steph Curry's gravity. Elite shooting and playoff pedigree.",
            "DAL": "Luka Doncic's brilliance powers everything. Elite offensive rating with improving defense.",
            "NOP": "Zion Williamson's power and versatile roster. Injury-plagued but talented when healthy.",
            "MEM": "Ja Morant's athleticism leads young core. Physical, defensive-minded team.",
            "HOU": "Young rebuilding roster with high picks developing. Fast-paced with room to grow.",
            "SAS": "Victor Wembanyama anchors the rebuild. Historic franchise developing next generation.",
            "UTA": "Rebuilding with young talent. Focus on development and draft assets.",
            "POR": "Building around young guards. Transitioning to next competitive window.",
        }
        
        # Defensive weaknesses analysis
        defensive_analysis = {
            "rating": None,
            "pts_allowed_pg": None,
            "weaknesses": [],
            "strengths": [],
        }
        
        if defense_rating:
            defensive_analysis["rating"] = "elite" if defense_rating.pts_factor < 0.95 else "good" if defense_rating.pts_factor < 1.0 else "average" if defense_rating.pts_factor < 1.05 else "poor"
            defensive_analysis["pts_allowed_pg"] = defense_rating.pts_allowed_pg
            defensive_analysis["reb_allowed_pg"] = defense_rating.reb_allowed_pg
            defensive_analysis["ast_allowed_pg"] = defense_rating.ast_allowed_pg
            
            # Determine strengths/weaknesses
            if defense_rating.pts_factor < 0.98:
                defensive_analysis["strengths"].append("Limiting opponent scoring")
            if defense_rating.reb_factor < 0.98:
                defensive_analysis["strengths"].append("Defensive rebounding")
            if defense_rating.ast_factor < 0.98:
                defensive_analysis["strengths"].append("Disrupting ball movement")
                
            if defense_rating.pts_factor > 1.02:
                defensive_analysis["weaknesses"].append("Allowing easy baskets")
            if defense_rating.reb_factor > 1.02:
                defensive_analysis["weaknesses"].append("Giving up offensive rebounds")
            if defense_rating.ast_factor > 1.02:
                defensive_analysis["weaknesses"].append("Susceptible to ball movement")
        
        return jsonify({
            "abbrev": abbrev,
            "name": team_name,
            "description": team_descriptions.get(abbrev, ""),
            "record": team_record,
            "roster": roster_profiles,
            "star_players": star_players,
            "elite_defenders": elite_defenders,
            "player_stats": player_stats,
            "hot_players": hot_players[:5],
            "recent_games": formatted_games,
            "defensive_analysis": defensive_analysis,
        })
    
    @app.route("/api/recent-dates")
    def api_recent_dates():
        """Get recent game dates for quick selection."""
        db = get_db()
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT game_date FROM games ORDER BY game_date DESC LIMIT 30"
            ).fetchall()
        return jsonify({"dates": [r["game_date"] for r in rows]})
    
    @app.route("/api/ingest/boxscore", methods=["POST"])
    def api_ingest_boxscore():
        """Ingest a pasted box score."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        game_date = data.get("date", "").strip()
        label = data.get("label", "PASTE").strip()
        
        if not text:
            return jsonify({"error": "No box score text provided"}), 400
        if not game_date:
            return jsonify({"error": "No game date provided"}), 400
        
        try:
            saved = save_pasted_boxscore_text(
                text=text,
                game_date=game_date,
                paths=paths,
                label=label,
            )
            
            db = get_db()
            with db.connect() as conn:
                game_id = ingest_boxscore_file(conn, source_file=saved.path)
                conn.commit()
            
            return jsonify({
                "success": True,
                "game_id": game_id,
                "saved_path": str(saved.path),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/ingest/lines", methods=["POST"])
    def api_ingest_lines():
        """Ingest sportsbook lines."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        as_of_date = data.get("date", "").strip()
        book = data.get("book", "").strip() or None
        
        if not text:
            return jsonify({"error": "No lines text provided"}), 400
        if not as_of_date:
            return jsonify({"error": "No as-of date provided"}), 400
        
        try:
            items = parse_lines_text(text)
            if not items:
                return jsonify({"error": "No lines parsed from text"}), 400
            
            db = get_db()
            with db.connect() as conn:
                for item in items:
                    player_row = conn.execute(
                        "SELECT id FROM players WHERE name = ?", (item.player,)
                    ).fetchone()
                    if player_row:
                        pid = int(player_row["id"])
                    else:
                        cur = conn.execute(
                            "INSERT INTO players(name) VALUES (?)", (item.player,)
                        )
                        pid = int(cur.lastrowid)
                    
                    conn.execute(
                        """
                        INSERT INTO sportsbook_lines(as_of_date, game_id, team_id, player_id, prop_type, line, odds_american, book)
                        VALUES (?, NULL, NULL, ?, ?, ?, ?, ?)
                        """,
                        (as_of_date, pid, item.prop_type, item.line, item.odds_american, book),
                    )
                conn.commit()
            
            return jsonify({
                "success": True,
                "count": len(items),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/detect-teams", methods=["POST"])
    def api_detect_teams():
        """Detect team names from pasted text."""
        data = request.get_json()
        text = data.get("text", "") if data else ""
        
        # Look for team names in the text
        detected = []
        for team_name in [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "Los Angeles Clippers", "LA Clippers", "Los Angeles Lakers", "LA Lakers",
            "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves",
            "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic",
            "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
            "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards",
        ]:
            if team_name in text:
                abbrev = abbrev_from_team_name(team_name)
                if abbrev and abbrev not in [d.get("abbrev") for d in detected]:
                    detected.append({"name": team_name, "abbrev": abbrev})
        
        return jsonify({"teams": detected[:2]})  # Return at most 2 teams
    
    @app.route("/api/suggest-date", methods=["POST"])
    def api_suggest_date():
        """Suggest a game date based on detected teams and recent games."""
        data = request.get_json()
        teams = data.get("teams", []) if data else []
        
        if len(teams) < 2:
            # Return today's date as default
            return jsonify({"date": datetime.now().strftime("%Y-%m-%d"), "source": "default"})
        
        # Look for recent games between these teams
        db = get_db()
        with db.connect() as conn:
            # Get the most recent game date in the database
            latest = conn.execute(
                "SELECT game_date FROM games ORDER BY game_date DESC LIMIT 1"
            ).fetchone()
            
            if latest:
                # Suggest the day after the latest game (common pattern for adding new games)
                from datetime import timedelta
                latest_date = datetime.strptime(latest["game_date"], "%Y-%m-%d")
                next_date = latest_date + timedelta(days=1)
                return jsonify({
                    "date": next_date.strftime("%Y-%m-%d"),
                    "source": "next_day",
                    "latest_game": latest["game_date"]
                })
        
        # Default to today
        return jsonify({"date": datetime.now().strftime("%Y-%m-%d"), "source": "default"})
    
    @app.route("/api/projections", methods=["POST"])
    def api_projections():
        """Generate projections for a matchup."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        away = data.get("away", "").strip().upper()
        home = data.get("home", "").strip().upper()
        game_date = data.get("date", "").strip() or datetime.now().strftime("%Y-%m-%d")
        lines_date = data.get("lines_date", "").strip() or game_date
        
        if not away or not home:
            return jsonify({"error": "Please provide both away and home teams"}), 400
        
        if away == home:
            return jsonify({"error": "Teams must be different"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                report = generate_prop_report(
                    conn=conn,
                    away_abbrev=away,
                    home_abbrev=home,
                    game_date=game_date,
                    lines_date=lines_date,
                )
            return jsonify(report)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/matchup-analysis", methods=["POST"])
    def api_matchup_analysis():
        """
        Generate comprehensive matchup analysis with all advanced metrics.
        
        This endpoint uses the Advisor layer from matchup_advisor module
        to provide actionable betting recommendations.
        
        Request Body (JSON):
            - away: Away team abbreviation (e.g., "LAL")
            - home: Home team abbreviation (e.g., "BOS")
            - date: Game date (YYYY-MM-DD)
            - spread: Optional point spread
            - over_under: Optional total points line
        
        Returns:
            ComprehensiveMatchupReport as JSON with:
            - best_over_plays: List of recommended OVER bets
            - best_under_plays: List of recommended UNDER bets
            - avoid_players: Players to avoid betting on
            - key_matchups: Important storylines
            - Defense profiles for both teams
        """
        from ..engine.matchup_advisor import generate_comprehensive_matchup_report
        from dataclasses import asdict
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        away = data.get("away", "").strip().upper()
        home = data.get("home", "").strip().upper()
        game_date = data.get("date", "").strip() or datetime.now().strftime("%Y-%m-%d")
        spread = data.get("spread")
        over_under = data.get("over_under")
        
        if not away or not home:
            return jsonify({"error": "Please provide both away and home teams"}), 400
        
        if away == home:
            return jsonify({"error": "Teams must be different"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                report = generate_comprehensive_matchup_report(
                    conn=conn,
                    away_abbrev=away,
                    home_abbrev=home,
                    game_date=game_date,
                    spread=float(spread) if spread is not None else None,
                    over_under=float(over_under) if over_under is not None else None,
                )
            
            # Transform the dataclass to the format expected by the frontend
            def transform_defense_profile(profile):
                if profile is None:
                    return None
                return {
                    "pts_factor": profile.pts_factor,
                    "reb_factor": profile.reb_factor,
                    "ast_factor": profile.ast_factor,
                    "rating": profile.pts_rating,
                    "pts_rating": profile.pts_rating,
                    "reb_rating": profile.reb_rating,
                    "ast_rating": profile.ast_rating,
                }
            
            def transform_player_projection(proj_dict):
                return {
                    "player_id": proj_dict.get("player_id"),
                    "player": proj_dict.get("player_name"),
                    "team": proj_dict.get("team_abbrev"),
                    "position": proj_dict.get("position", ""),
                    "minutes": proj_dict.get("proj_minutes", 0),
                    "pts": proj_dict.get("proj_pts", 0),
                    "reb": proj_dict.get("proj_reb", 0),
                    "ast": proj_dict.get("proj_ast", 0),
                    "games": proj_dict.get("games_played", 0),
                    "is_top_7": proj_dict.get("is_top_7", True),
                    "is_top_10": proj_dict.get("is_top_10", True),
                    "trend": {
                        "pts": proj_dict.get("pts_trend", "stable"),
                        "reb": proj_dict.get("reb_trend", "stable"),
                        "ast": proj_dict.get("ast_trend", "stable"),
                        "pts_change": proj_dict.get("pts_trend_pct", 0),
                        "reb_change": proj_dict.get("reb_trend_pct", 0),
                        "ast_change": proj_dict.get("ast_trend_pct", 0),
                    },
                    "archetype": proj_dict.get("adjustments", {}).get("archetype"),
                    "edges": proj_dict.get("matchup_edges", {}),
                    "game_log": [],
                }
            
            def transform_matchup_edge(edge):
                return {
                    "player": edge.player_name,
                    "team": edge.team_abbrev,
                    "opponent": edge.opponent_abbrev,
                    "prop": edge.prop_type,
                    "baseline": edge.baseline_projection,
                    "adjusted": edge.adjusted_projection,
                    "adjustment_pct": edge.adjustment_pct,
                    "confidence": edge.confidence_tier,
                    "confidence_score": edge.confidence_score,
                    "reasons": edge.reasons,
                    "warnings": edge.warnings,
                }
            
            result = {
                "matchup": {
                    "away": report.away_abbrev,
                    "home": report.home_abbrev,
                    "date": report.game_date,
                    "spread": report.spread,
                    "over_under": report.over_under,
                    "is_close_game": report.is_close_game,
                },
                "context": {
                    "away_b2b": report.away_b2b,
                    "away_rest_days": report.away_rest_days,
                    "home_b2b": report.home_b2b,
                    "home_rest_days": report.home_rest_days,
                },
                "defense": {
                    "away": {
                        "overall_rating": "Average Defense",
                        "guard_defense": transform_defense_profile(report.away_defense_vs_guards),
                        "forward_defense": transform_defense_profile(report.away_defense_vs_forwards),
                        "center_defense": transform_defense_profile(report.away_defense_vs_centers),
                    },
                    "home": {
                        "overall_rating": "Average Defense",
                        "guard_defense": transform_defense_profile(report.home_defense_vs_guards),
                        "forward_defense": transform_defense_profile(report.home_defense_vs_forwards),
                        "center_defense": transform_defense_profile(report.home_defense_vs_centers),
                    },
                },
                "away_projections": [
                    transform_player_projection(p) for p in report.away_player_projections
                ],
                "home_projections": [
                    transform_player_projection(p) for p in report.home_player_projections
                ],
                "best_over_plays": [
                    transform_matchup_edge(e) for e in report.best_over_plays
                ],
                "best_under_plays": [
                    transform_matchup_edge(e) for e in report.best_under_plays
                ],
                "avoid_players": report.avoid_players,
                "key_matchups": report.key_matchups,
            }
            
            return jsonify(result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/team/<abbrev>/defense-profile")
    def api_team_defense_profile(abbrev: str):
        """Get detailed defensive profile for a team by position."""
        from ..engine.matchup_advisor import (
            get_team_defense_summary,
            get_all_position_defense_profiles,
        )
        
        abbrev = abbrev.upper()
        
        db = get_db()
        try:
            with db.connect() as conn:
                summary = get_team_defense_summary(conn, abbrev)
                position_profiles = get_all_position_defense_profiles(conn, abbrev)
                
                profiles_dict = {}
                for pos, profile in position_profiles.items():
                    profiles_dict[pos] = {
                        "pts_allowed_avg": profile.pts_allowed_avg,
                        "reb_allowed_avg": profile.reb_allowed_avg,
                        "ast_allowed_avg": profile.ast_allowed_avg,
                        "pts_factor": profile.pts_factor,
                        "reb_factor": profile.reb_factor,
                        "ast_factor": profile.ast_factor,
                        "pts_rating": profile.pts_rating,
                        "reb_rating": profile.reb_rating,
                        "ast_rating": profile.ast_rating,
                        "sample_size": profile.sample_size,
                    }
                
                return jsonify({
                    "team": abbrev,
                    "summary": summary,
                    "position_profiles": profiles_dict,
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/player/<int:player_id>/trend")
    def api_player_trend(player_id: int):
        """Get trend analysis for a player."""
        from ..engine.matchup_advisor import get_player_trend
        
        db = get_db()
        try:
            with db.connect() as conn:
                trend = get_player_trend(conn, player_id)
                
                if not trend:
                    return jsonify({"error": "Insufficient data for trend analysis"}), 404
                
                return jsonify({
                    "player_name": trend.player_name,
                    "player_id": trend.player_id,
                    "team": trend.team_abbrev,
                    "recent": {
                        "pts": trend.recent_pts,
                        "reb": trend.recent_reb,
                        "ast": trend.recent_ast,
                        "min": trend.recent_min,
                        "games": trend.recent_games,
                    },
                    "season": {
                        "pts": trend.season_pts,
                        "reb": trend.season_reb,
                        "ast": trend.season_ast,
                        "games": trend.season_games,
                    },
                    "trends": {
                        "pts": trend.pts_trend,
                        "reb": trend.reb_trend,
                        "ast": trend.ast_trend,
                    },
                    "changes": {
                        "pts": trend.pts_change_pct,
                        "reb": trend.reb_change_pct,
                        "ast": trend.ast_change_pct,
                    },
                    "consistency": {
                        "pts": trend.pts_consistency,
                        "reb": trend.reb_consistency,
                        "ast": trend.ast_consistency,
                    },
                    "game_log": trend.game_log,
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/player/<player_name>/vs-team/<opponent>")
    def api_player_vs_team(player_name: str, opponent: str):
        """Get player's historical performance against a specific team."""
        from ..engine.matchup_advisor import get_player_vs_team_profile
        
        db = get_db()
        try:
            with db.connect() as conn:
                profile = get_player_vs_team_profile(conn, player_name, opponent.upper())
                
                if not profile:
                    return jsonify({"error": "No data found"}), 404
                
                return jsonify({
                    "player": profile.player_name,
                    "opponent": profile.opponent_abbrev,
                    "games_played": profile.games_played,
                    "has_history": profile.has_history,
                    "vs_opponent": {
                        "pts": profile.pts_avg,
                        "reb": profile.reb_avg,
                        "ast": profile.ast_avg,
                        "min": profile.min_avg,
                    },
                    "overall": {
                        "pts": profile.overall_pts_avg,
                        "reb": profile.overall_reb_avg,
                        "ast": profile.overall_ast_avg,
                    },
                    "differential": {
                        "pts": profile.pts_diff,
                        "reb": profile.reb_diff,
                        "ast": profile.ast_diff,
                    },
                    "recent_games": profile.recent_games,
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/position-defense-rankings/<position>")
    def api_position_defense_rankings(position: str):
        """Get defense rankings for all teams against a specific position."""
        from ..engine.matchup_advisor import rank_position_defense_profiles
        
        position = position.upper()[:1]
        if position not in ("G", "F", "C"):
            return jsonify({"error": "Position must be G, F, or C"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                rankings = rank_position_defense_profiles(conn, position)
                
                return jsonify({
                    "position": position,
                    "rankings": [
                        {
                            "rank": i + 1,
                            "team": r.team_abbrev,
                            "pts_allowed_avg": r.pts_allowed_avg,
                            "pts_factor": r.pts_factor,
                            "pts_rating": r.pts_rating,
                            "reb_factor": r.reb_factor,
                            "reb_rating": r.reb_rating,
                            "ast_factor": r.ast_factor,
                            "ast_rating": r.ast_rating,
                            "sample_size": r.sample_size,
                        }
                        for i, r in enumerate(rankings)
                    ]
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/team/<abbrev>/projections")
    def api_team_projections(abbrev: str):
        """Get projections for a team's players."""
        abbrev = abbrev.upper()
        opponent = request.args.get("opponent", "").upper()
        game_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        db = get_db()
        try:
            with db.connect() as conn:
                # Get back-to-back status
                b2b = get_back_to_back_status(conn, abbrev, game_date)
                
                # Get opponent defense if specified
                opp_defense = None
                if opponent:
                    opp_defense = get_team_defense_rating(conn, opponent)
                
                # Generate projections
                projections = project_team_players(
                    conn=conn,
                    team_abbrev=abbrev,
                    opponent_abbrev=opponent or None,
                    is_back_to_back=b2b.is_back_to_back,
                    rest_days=b2b.rest_days,
                )
                
                # Apply opponent adjustments
                results = []
                for proj in projections:
                    adj_pts, adj_reb, adj_ast, adj_info = apply_matchup_adjustments(
                        proj.proj_pts, proj.proj_reb, proj.proj_ast, opp_defense
                    )
                    
                    results.append({
                        "player_id": proj.player_id,
                        "player": proj.player_name,
                        "position": proj.position,
                        "minutes": proj.proj_minutes,
                        "pts": adj_pts,
                        "reb": adj_reb,
                        "ast": adj_ast,
                        "pts_std": proj.pts_std,
                        "reb_std": proj.reb_std,
                        "ast_std": proj.ast_std,
                        "games": proj.games_played,
                        "is_top_7": proj.is_top_7,
                        "adjustments": {**proj.adjustments, **adj_info},
                    })
                
                return jsonify({
                    "team": abbrev,
                    "game_date": game_date,
                    "opponent": opponent or None,
                    "is_back_to_back": b2b.is_back_to_back,
                    "rest_days": b2b.rest_days,
                    "projections": results,
                })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/defense-ratings")
    def api_defense_ratings():
        """Get defense ratings for all teams."""
        from ..engine.game_context import get_all_team_defense_ratings
        
        db = get_db()
        try:
            with db.connect() as conn:
                ratings = get_all_team_defense_ratings(conn)
            
            return jsonify({
                "ratings": [
                    {
                        "team": r.team_abbrev,
                        "games": r.games_played,
                        "pts_allowed": r.pts_allowed_pg,
                        "reb_allowed": r.reb_allowed_pg,
                        "ast_allowed": r.ast_allowed_pg,
                        "pts_factor": r.pts_factor,
                        "reb_factor": r.reb_factor,
                        "ast_factor": r.ast_factor,
                    }
                    for r in ratings.values()
                ]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/lines")
    def api_lines():
        """Get sportsbook lines."""
        date = request.args.get("date", "")
        limit = request.args.get("limit", 100, type=int)
        
        db = get_db()
        with db.connect() as conn:
            if date:
                rows = conn.execute(
                    """
                    SELECT sl.id, sl.as_of_date, p.name AS player, sl.prop_type, 
                           sl.line, sl.odds_american, sl.book
                    FROM sportsbook_lines sl
                    JOIN players p ON p.id = sl.player_id
                    WHERE sl.as_of_date = ?
                    ORDER BY p.name, sl.prop_type
                    """,
                    (date,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT sl.id, sl.as_of_date, p.name AS player, sl.prop_type, 
                           sl.line, sl.odds_american, sl.book
                    FROM sportsbook_lines sl
                    JOIN players p ON p.id = sl.player_id
                    ORDER BY sl.as_of_date DESC, p.name, sl.prop_type
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        
        return jsonify({
            "lines": [dict(r) for r in rows]
        })
    
    @app.route("/api/player/<player_name>/archetype")
    def api_player_archetype(player_name: str):
        """Get archetype classification for a player."""
        archetype = get_player_archetype(player_name)
        
        if not archetype:
            # Try to classify from database stats
            db = get_db()
            with db.connect() as conn:
                player_row = conn.execute(
                    "SELECT id FROM players WHERE name LIKE ?", (f"%{player_name}%",)
                ).fetchone()
                if player_row:
                    archetype = classify_player_by_stats(conn, player_row["id"])
        
        if not archetype:
            return jsonify({"error": "Player not found or unclassified"}), 404
        
        return jsonify({
            "player": archetype.player_name,
            "tier": archetype.tier,
            "primary_offensive": archetype.primary_offensive,
            "secondary_offensive": archetype.secondary_offensive,
            "defensive_role": archetype.defensive_role,
            "notes": archetype.notes,
        })
    
    @app.route("/api/archetypes")
    def api_archetypes():
        """Get all known player archetypes."""
        archetypes = []
        for name, data in KNOWN_ARCHETYPES.items():
            primary, secondary, defensive, tier, notes = data
            archetypes.append({
                "player": name,
                "tier": tier,
                "primary_offensive": primary,
                "secondary_offensive": secondary,
                "defensive_role": defensive,
                "notes": notes,
            })
        
        # Sort by tier, then by player name
        archetypes.sort(key=lambda x: (x["tier"], x["player"]))
        return jsonify({"archetypes": archetypes})
    
    # -------------------------------------------------------------------------
    # Roster System API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/roster")
    def api_roster():
        """Get complete player roster with archetypes."""
        from ..engine.roster import PLAYER_DATABASE, PlayerTier
        
        tier = request.args.get("tier", "")
        team = request.args.get("team", "")
        elite_defenders_only = request.args.get("elite_defenders", "false").lower() == "true"
        
        players = []
        for name, profile in PLAYER_DATABASE.items():
            # Apply filters
            if tier and profile.tier.name.lower() != tier.lower():
                continue
            if team and profile.team.lower() != team.lower():
                continue
            if elite_defenders_only and not profile.is_elite_defender:
                continue
            
            players.append({
                "name": name,
                "team": profile.team,
                "position": profile.position,
                "height": profile.height,
                "primary_offensive": profile.primary_offensive.value,
                "secondary_offensive": profile.secondary_offensive.value if profile.secondary_offensive else None,
                "defensive_role": profile.defensive_role.value,
                "tier": profile.tier.name,
                "tier_value": profile.tier.value,
                "is_elite_defender": profile.is_elite_defender,
                "strengths": profile.strengths,
                "weaknesses": profile.weaknesses,
                "notes": profile.notes,
                "guards_positions": profile.guards_positions,
                "avoid_betting_against": profile.avoid_betting_against,
            })
        
        # Sort by tier, then name
        players.sort(key=lambda x: (x["tier_value"], x["name"]))
        
        return jsonify({
            "players": players,
            "count": len(players),
        })
    
    @app.route("/api/roster/player/<player_name>")
    def api_roster_player(player_name: str):
        """Get detailed profile for a specific player."""
        from ..engine.roster import get_player_profile, get_similar_players
        
        profile = get_player_profile(player_name)
        if not profile:
            return jsonify({"error": "Player not found in roster"}), 404
        
        similar = get_similar_players(player_name)
        
        return jsonify({
            "name": profile.name,
            "team": profile.team,
            "position": profile.position,
            "height": profile.height,
            "primary_offensive": profile.primary_offensive.value,
            "secondary_offensive": profile.secondary_offensive.value if profile.secondary_offensive else None,
            "defensive_role": profile.defensive_role.value,
            "tier": profile.tier.name,
            "is_elite_defender": profile.is_elite_defender,
            "strengths": profile.strengths,
            "weaknesses": profile.weaknesses,
            "notes": profile.notes,
            "guards_positions": profile.guards_positions,
            "avoid_betting_against": profile.avoid_betting_against,
            "similar_players": similar,
        })
    
    @app.route("/api/roster/similarity-groups")
    def api_similarity_groups():
        """Get all player similarity groups."""
        from ..engine.roster import PLAYER_SIMILARITY_GROUPS
        
        return jsonify({
            "groups": {
                name: players for name, players in PLAYER_SIMILARITY_GROUPS.items()
            }
        })
    
    @app.route("/api/roster/elite-defenders")
    def api_elite_defenders():
        """Get all elite defenders grouped by position."""
        from ..engine.roster import ELITE_DEFENDERS_BY_POSITION, get_player_profile
        
        result = {}
        for position, defenders in ELITE_DEFENDERS_BY_POSITION.items():
            result[position] = []
            for name in defenders:
                profile = get_player_profile(name)
                if profile:
                    result[position].append({
                        "name": name,
                        "team": profile.team,
                        "defensive_role": profile.defensive_role.value,
                    })
                else:
                    result[position].append({
                        "name": name,
                        "team": "Unknown",
                        "defensive_role": "Unknown",
                    })
        
        return jsonify({"defenders_by_position": result})
    
    @app.route("/api/roster/matchup-check", methods=["POST"])
    def api_matchup_check():
        """Check if we should avoid betting on a player based on opponent's defenders."""
        from ..engine.roster import should_avoid_betting_over, get_player_profile, get_roster_for_team
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        player_name = data.get("player", "").strip()
        opponent_team = data.get("opponent", "").strip()
        
        if not player_name or not opponent_team:
            return jsonify({"error": "Player and opponent team required"}), 400
        
        profile = get_player_profile(player_name)
        if not profile:
            return jsonify({
                "player": player_name,
                "opponent": opponent_team,
                "avoid": False,
                "reason": "Player not in database",
                "defenders": [],
            })
        
        # Get opponent roster
        opponent_roster = [p.name for p in get_roster_for_team(opponent_team)]
        
        avoid, defenders = should_avoid_betting_over(player_name, opponent_roster)
        
        return jsonify({
            "player": player_name,
            "player_position": profile.position,
            "player_archetype": profile.primary_offensive.value,
            "opponent": opponent_team,
            "avoid": avoid,
            "elite_defenders": defenders,
            "recommendation": "Consider UNDER or avoid" if avoid else "Standard projection",
        })
    
    @app.route("/api/roster/tiers")
    def api_roster_tiers():
        """Get players grouped by tier."""
        from ..engine.roster import PLAYER_DATABASE, get_players_by_tier, PlayerTier
        
        result = {}
        for tier in PlayerTier:
            players = get_players_by_tier(tier)
            result[tier.name] = {
                "tier_value": tier.value,
                "description": {
                    PlayerTier.MVP_CANDIDATE: "Heliocentric stars, high usage",
                    PlayerTier.TWO_WAY_STAR: "Elite two-way players",
                    PlayerTier.ELITE_BIG: "Top tier big men",
                    PlayerTier.ELITE_ROLE: "Championship-level role players",
                    PlayerTier.SPECIALIST: "Scoring and other specialists",
                    PlayerTier.ROTATION: "Key rotation pieces",
                }.get(tier, ""),
                "count": len(players),
                "players": players,
            }
        
        return jsonify({"tiers": result})
    
    # -------------------------------------------------------------------------
    # Database-Backed Archetype API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/archetypes-db")
    def api_archetypes_db():
        """Get all player archetypes from database (with fallback to defaults)."""
        season = request.args.get("season", "2025-26")
        tier = request.args.get("tier", type=int)
        team = request.args.get("team", "")
        elite_only = request.args.get("elite_defenders", "false").lower() == "true"
        stars_only = request.args.get("stars_only", "false").lower() == "true"
        
        db = get_db()
        with db.connect() as conn:
            archetypes = get_all_archetypes_db(
                conn, 
                season=season,
                tier=tier,
                team=team if team else None,
                elite_defenders_only=elite_only,
                stars_only=stars_only,
            )
            
            # Get count
            count = get_archetype_count_db(conn, season)
        
        return jsonify({
            "archetypes": [
                {
                    "id": a.id,
                    "player": a.player_name,
                    "team": a.team,
                    "position": a.position,
                    "height": a.height,
                    "primary_offensive": a.primary_offensive,
                    "secondary_offensive": a.secondary_offensive,
                    "defensive_role": a.defensive_role,
                    "tier": a.tier,
                    "is_elite_defender": a.is_elite_defender,
                    "is_star": a.is_star,
                    "strengths": a.strengths,
                    "weaknesses": a.weaknesses,
                    "notes": a.notes,
                    "guards_positions": a.guards_positions,
                    "avoid_betting_against": a.avoid_betting_against,
                    "source": a.source,
                    "confidence": a.confidence,
                }
                for a in archetypes
            ],
            "count": len(archetypes),
            "db_count": count,
            "season": season,
        })
    
    @app.route("/api/archetypes-db/player/<player_name>")
    def api_archetype_db_player(player_name: str):
        """Get archetype for a specific player from database."""
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        with db.connect() as conn:
            archetype = get_player_archetype_db(conn, player_name, season)
            similar = get_similar_players_db(conn, player_name, season) if archetype else []
        
        if not archetype:
            return jsonify({"error": "Player not found"}), 404
        
        return jsonify({
            "player": archetype.player_name,
            "team": archetype.team,
            "position": archetype.position,
            "height": archetype.height,
            "primary_offensive": archetype.primary_offensive,
            "secondary_offensive": archetype.secondary_offensive,
            "defensive_role": archetype.defensive_role,
            "tier": archetype.tier,
            "is_elite_defender": archetype.is_elite_defender,
            "is_star": archetype.is_star,
            "strengths": archetype.strengths,
            "weaknesses": archetype.weaknesses,
            "notes": archetype.notes,
            "guards_positions": archetype.guards_positions,
            "avoid_betting_against": archetype.avoid_betting_against,
            "source": archetype.source,
            "confidence": archetype.confidence,
            "similar_players": similar,
        })
    
    @app.route("/api/archetypes-db/player/<player_name>", methods=["PUT", "POST"])
    def api_archetype_db_update(player_name: str):
        """Update or create an archetype for a player."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        season = data.get("season", "2025-26")
        
        # Extract updatable fields
        update_fields = {}
        allowed_fields = [
            "team", "position", "height", "primary_offensive", "secondary_offensive",
            "defensive_role", "tier", "is_elite_defender", "is_star", "strengths", "weaknesses",
            "notes", "guards_positions", "avoid_betting_against"
        ]
        
        for field in allowed_fields:
            if field in data:
                update_fields[field] = data[field]
        
        if not update_fields:
            return jsonify({"error": "No fields to update"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                success = update_player_archetype(conn, player_name, season, **update_fields)
            
            return jsonify({
                "success": success,
                "player": player_name,
                "updated_fields": list(update_fields.keys()),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/archetypes-db/player/<player_name>", methods=["DELETE"])
    def api_archetype_db_delete(player_name: str):
        """Delete a player's archetype from database (will fall back to defaults)."""
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        try:
            with db.connect() as conn:
                success = delete_player_archetype(conn, player_name, season)
            
            return jsonify({
                "success": success,
                "player": player_name,
                "note": "Player will now use default archetype if available",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/archetypes-db/seed", methods=["POST"])
    def api_archetypes_seed():
        """Seed database with default archetypes from PLAYER_DATABASE."""
        data = request.get_json() or {}
        season = data.get("season", "2025-26")
        overwrite = data.get("overwrite", False)
        
        db = get_db()
        try:
            with db.connect() as conn:
                count = seed_archetypes_from_defaults(conn, season, overwrite)
            
            return jsonify({
                "success": True,
                "seeded_count": count,
                "season": season,
                "overwrite": overwrite,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/archetypes-db/similarity-groups")
    def api_similarity_groups_db():
        """Get player similarity groups from database."""
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        with db.connect() as conn:
            groups = get_similarity_groups_db(conn, season)
        
        return jsonify({"groups": groups, "season": season})
    
    @app.route("/api/archetypes-db/elite-defenders")
    def api_elite_defenders_db():
        """Get elite defenders by position from database."""
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        with db.connect() as conn:
            defenders = get_elite_defenders_db(conn, season)
        
        return jsonify({"defenders_by_position": defenders, "season": season})
    
    @app.route("/api/archetypes-db/matchup-check", methods=["POST"])
    def api_matchup_check_db():
        """Check if we should avoid betting on a player using DB-backed data."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        player_name = data.get("player", "").strip()
        opponent_team = data.get("opponent", "").strip()
        season = data.get("season", "2025-26")
        
        if not player_name or not opponent_team:
            return jsonify({"error": "Player and opponent team required"}), 400
        
        db = get_db()
        with db.connect() as conn:
            archetype = get_player_archetype_db(conn, player_name, season)
            if not archetype:
                return jsonify({
                    "player": player_name,
                    "opponent": opponent_team,
                    "avoid": False,
                    "reason": "Player not in database or defaults",
                    "defenders": [],
                })
            
            avoid, defenders = should_avoid_betting_over_db(conn, player_name, opponent_team, season)
        
        return jsonify({
            "player": player_name,
            "player_position": archetype.position,
            "player_archetype": archetype.primary_offensive,
            "opponent": opponent_team,
            "avoid": avoid,
            "elite_defenders": defenders,
            "recommendation": "Consider UNDER or avoid" if avoid else "Standard projection",
            "source": archetype.source,
        })
    
    @app.route("/api/archetypes-db/team/<team_name>")
    def api_team_roster_db(team_name: str):
        """Get roster archetypes for a team from database."""
        season = request.args.get("season", "2025-26")
        
        # Convert abbreviation to full name if needed
        full_name = team_name_from_abbrev(team_name.upper()) or team_name
        
        db = get_db()
        with db.connect() as conn:
            roster = get_roster_for_team_db(conn, full_name, season)
        
        return jsonify({
            "team": full_name,
            "abbrev": abbrev_from_team_name(full_name),
            "roster": [
                {
                    "player": p.player_name,
                    "position": p.position,
                    "height": p.height,
                    "primary_offensive": p.primary_offensive,
                    "secondary_offensive": p.secondary_offensive,
                    "defensive_role": p.defensive_role,
                    "tier": p.tier,
                    "is_elite_defender": p.is_elite_defender,
                    "is_star": p.is_star,
                    "source": p.source,
                }
                for p in roster
            ],
            "count": len(roster),
            "season": season,
        })
    
    @app.route("/api/archetypes-db/stats")
    def api_archetypes_stats():
        """Get statistics about archetypes in database."""
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        with db.connect() as conn:
            total = get_archetype_count_db(conn, season)
            
            # Get counts by tier
            tier_counts = conn.execute(
                """
                SELECT tier, COUNT(*) as count
                FROM player_archetypes
                WHERE season = ?
                GROUP BY tier
                ORDER BY tier
                """,
                (season,),
            ).fetchall()
            
            # Get counts by source
            source_counts = conn.execute(
                """
                SELECT source, COUNT(*) as count
                FROM player_archetypes
                WHERE season = ?
                GROUP BY source
                """,
                (season,),
            ).fetchall()
            
            # Get count of elite defenders
            elite_count = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM player_archetypes
                WHERE season = ? AND is_elite_defender = 1
                """,
                (season,),
            ).fetchone()
        
        # Count from defaults (hard-coded)
        from ..engine.roster import PLAYER_DATABASE
        defaults_count = len(PLAYER_DATABASE)
        
        return jsonify({
            "season": season,
            "total_in_db": total,
            "defaults_available": defaults_count,
            "by_tier": {str(r["tier"]): r["count"] for r in tier_counts},
            "by_source": {r["source"]: r["count"] for r in source_counts},
            "elite_defenders": elite_count["count"] if elite_count else 0,
        })
    
    # -------------------------------------------------------------------------
    # Star Player Management API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/archetypes-db/player/<player_name>/star", methods=["POST", "PUT"])
    def api_toggle_star_status(player_name: str):
        """Toggle or set star status for a player."""
        data = request.get_json() or {}
        is_star = data.get("is_star", True)
        season = data.get("season", "2025-26")
        
        db = get_db()
        try:
            with db.connect() as conn:
                success = toggle_star_status(conn, player_name, is_star, season)
            
            return jsonify({
                "success": success,
                "player": player_name,
                "is_star": is_star,
                "season": season,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/archetypes-db/stars")
    def api_all_star_players():
        """Get all star players across all teams."""
        season = request.args.get("season", "2025-26")
        team = request.args.get("team", "")
        
        db = get_db()
        with db.connect() as conn:
            if team:
                full_name = team_name_from_abbrev(team.upper()) or team
                stars = get_star_players_for_team(conn, full_name, season)
            else:
                stars = get_all_star_players(conn, season)
        
        return jsonify({
            "stars": [
                {
                    "player": s.player_name,
                    "team": s.team,
                    "position": s.position,
                    "tier": s.tier,
                    "primary_offensive": s.primary_offensive,
                    "is_elite_defender": s.is_elite_defender,
                }
                for s in stars
            ],
            "count": len(stars),
            "season": season,
        })
    
    @app.route("/api/archetypes-db/team/<team_name>/stars")
    def api_team_stars(team_name: str):
        """Get star players for a specific team."""
        season = request.args.get("season", "2025-26")
        full_name = team_name_from_abbrev(team_name.upper()) or team_name
        
        db = get_db()
        with db.connect() as conn:
            stars = get_star_players_for_team(conn, full_name, season)
        
        return jsonify({
            "team": full_name,
            "abbrev": abbrev_from_team_name(full_name),
            "stars": [
                {
                    "player": s.player_name,
                    "position": s.position,
                    "tier": s.tier,
                    "primary_offensive": s.primary_offensive,
                    "is_elite_defender": s.is_elite_defender,
                }
                for s in stars
            ],
            "count": len(stars),
            "season": season,
        })

    @app.route("/api/star-players/toggle", methods=["POST"])
    def api_star_players_toggle():
        """Toggle star status for a player (used by team detail page)."""
        data = request.get_json() or {}
        player_name = data.get("player_name")
        season = data.get("season", "2025-26")
        
        if not player_name:
            return jsonify({"success": False, "error": "player_name is required"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                # First check current star status
                current_status = is_star_player(conn, player_name, season)
                new_status = not current_status
                
                # Toggle the status
                success = toggle_star_status(conn, player_name, new_status, season)
                
                return jsonify({
                    "success": success,
                    "player_name": player_name,
                    "is_star": new_status,
                    "season": season,
                })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/player/bet-status", methods=["POST"])
    def api_player_bet_status():
        """Set bet status for a player (0=avoid, 1=neutral, 2=star)."""
        data = request.get_json() or {}
        player_name = data.get("player_name")
        new_status = data.get("bet_status")
        team = data.get("team")  # Team is important for new entries
        season = data.get("season", "2025-26")
        
        if not player_name:
            return jsonify({"success": False, "error": "player_name is required"}), 400
        
        if new_status is None:
            return jsonify({"success": False, "error": "bet_status is required"}), 400
        
        if new_status not in [0, 1, 2]:
            return jsonify({"success": False, "error": "bet_status must be 0, 1, or 2"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                # Set the new bet status
                success = set_bet_status(conn, player_name, new_status, team=team, season=season)
                
                return jsonify({
                    "success": success,
                    "player_name": player_name,
                    "bet_status": new_status,
                    "is_star": new_status == BET_STATUS_STAR,
                    "team": team,
                    "season": season,
                })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/player/bet-status/<player_name>")
    def api_get_player_bet_status(player_name: str):
        """Get bet status for a player."""
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        try:
            with db.connect() as conn:
                status = get_bet_status(conn, player_name, season)
                
                return jsonify({
                    "success": True,
                    "player_name": player_name,
                    "bet_status": status,
                    "is_star": status == BET_STATUS_STAR,
                    "season": season,
                })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400

    @app.route("/api/salaries")
    def api_salaries():
        """Get player salaries."""
        limit = request.args.get("limit", 100, type=int)
        team = request.args.get("team", "")
        
        db = get_db()
        with db.connect() as conn:
            # Check if salary table exists and has data
            try:
                if team:
                    rows = conn.execute(
                        """
                        SELECT salary_rank, player_name, position, team, salary
                        FROM player_salaries
                        WHERE team LIKE ?
                        ORDER BY salary DESC
                        LIMIT ?
                        """,
                        (f"%{team}%", limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT salary_rank, player_name, position, team, salary
                        FROM player_salaries
                        ORDER BY salary DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                
                return jsonify({
                    "salaries": [
                        {
                            "rank": r["salary_rank"],
                            "player": r["player_name"],
                            "position": r["position"],
                            "team": r["team"],
                            "salary": r["salary"],
                            "salary_formatted": f"${r['salary']:,}",
                        }
                        for r in rows
                    ]
                })
            except Exception:
                return jsonify({"salaries": [], "note": "Salary data not yet imported"})
    
    @app.route("/api/ingest/salaries", methods=["POST"])
    def api_ingest_salaries():
        """Ingest salary data from pasted text."""
        from ..ingest.salary_parser import parse_salary_text, ingest_salaries
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No salary text provided"}), 400
        
        try:
            salaries = parse_salary_text(text)
            if not salaries:
                return jsonify({"error": "No salaries parsed from text"}), 400
            
            db = get_db()
            with db.connect() as conn:
                count = ingest_salaries(conn, salaries)
            
            return jsonify({
                "success": True,
                "count": count,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/ingest/game-line", methods=["POST"])
    def api_ingest_game_line():
        """Ingest a game spread/line."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        game_date = data.get("date", "").strip()
        away_team = data.get("away", "").strip().upper()
        home_team = data.get("home", "").strip().upper()
        spread = data.get("spread")  # Home team spread (negative = home favored)
        over_under = data.get("over_under")
        book = data.get("book", "consensus").strip()
        
        if not game_date or not away_team or not home_team:
            return jsonify({"error": "Date, away team, and home team are required"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                from ..db import get_or_create_team
                away_team_name = team_name_from_abbrev(away_team) or away_team
                home_team_name = team_name_from_abbrev(home_team) or home_team
                
                away_id = get_or_create_team(conn, away_team_name)
                home_id = get_or_create_team(conn, home_team_name)
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO game_lines 
                    (game_date, away_team_id, home_team_id, spread, over_under, book)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (game_date, away_id, home_id, spread, over_under, book),
                )
                conn.commit()
            
            return jsonify({
                "success": True,
                "game_date": game_date,
                "matchup": f"{away_team} @ {home_team}",
                "spread": spread,
                "over_under": over_under,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/game-lines")
    def api_game_lines():
        """Get game lines/spreads."""
        date = request.args.get("date", "")
        close_only = request.args.get("close_only", "false").lower() == "true"
        
        db = get_db()
        with db.connect() as conn:
            try:
                if date:
                    sql = """
                        SELECT gl.game_date, t1.name AS away_team, t2.name AS home_team,
                               gl.spread, gl.over_under, gl.book
                        FROM game_lines gl
                        JOIN teams t1 ON t1.id = gl.away_team_id
                        JOIN teams t2 ON t2.id = gl.home_team_id
                        WHERE gl.game_date = ?
                    """
                    params = [date]
                else:
                    sql = """
                        SELECT gl.game_date, t1.name AS away_team, t2.name AS home_team,
                               gl.spread, gl.over_under, gl.book
                        FROM game_lines gl
                        JOIN teams t1 ON t1.id = gl.away_team_id
                        JOIN teams t2 ON t2.id = gl.home_team_id
                        ORDER BY gl.game_date DESC
                        LIMIT 50
                    """
                    params = []
                
                if close_only:
                    # Filter to games with spread <= 6 points
                    if date:
                        sql = sql.replace("WHERE", "WHERE ABS(gl.spread) <= 6 AND")
                    else:
                        sql = sql.replace("ORDER BY", "WHERE ABS(gl.spread) <= 6 ORDER BY")
                
                rows = conn.execute(sql, params).fetchall()
                
                return jsonify({
                    "game_lines": [
                        {
                            "date": r["game_date"],
                            "away": r["away_team"],
                            "away_abbrev": abbrev_from_team_name(r["away_team"]),
                            "home": r["home_team"],
                            "home_abbrev": abbrev_from_team_name(r["home_team"]),
                            "spread": r["spread"],
                            "over_under": r["over_under"],
                            "book": r["book"],
                            "is_close": abs(r["spread"] or 0) <= 6,
                        }
                        for r in rows
                    ]
                })
            except Exception:
                return jsonify({"game_lines": [], "note": "No game lines data yet"})
    
    @app.route("/api/injuries")
    def api_injuries():
        """Get injury report."""
        date = request.args.get("date", "")
        team = request.args.get("team", "")
        
        db = get_db()
        with db.connect() as conn:
            try:
                sql = """
                    SELECT ir.game_date, t.name AS team, 
                           COALESCE(p.name, ir.player_name) AS player,
                           ir.status, ir.minutes_limit, ir.notes
                    FROM injury_report ir
                    JOIN teams t ON t.id = ir.team_id
                    LEFT JOIN players p ON p.id = ir.player_id
                """
                params = []
                conditions = []
                
                if date:
                    conditions.append("ir.game_date = ?")
                    params.append(date)
                if team:
                    conditions.append("t.name LIKE ?")
                    params.append(f"%{team}%")
                
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY ir.game_date DESC, t.name, ir.status"
                
                rows = conn.execute(sql, params).fetchall()
                
                return jsonify({
                    "injuries": [
                        {
                            "date": r["game_date"],
                            "team": r["team"],
                            "team_abbrev": abbrev_from_team_name(r["team"]),
                            "player": r["player"],
                            "status": r["status"],
                            "minutes_limit": r["minutes_limit"],
                            "notes": r["notes"],
                        }
                        for r in rows
                    ]
                })
            except Exception:
                return jsonify({"injuries": []})
    
    @app.route("/api/ingest/injury", methods=["POST"])
    def api_ingest_injury():
        """Add an injury report entry."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        game_date = data.get("date", "").strip()
        team = data.get("team", "").strip().upper()
        player = data.get("player", "").strip()
        status = data.get("status", "OUT").strip().upper()
        minutes_limit = data.get("minutes_limit")
        notes = data.get("notes", "").strip() or None
        
        if not game_date or not team or not player:
            return jsonify({"error": "Date, team, and player are required"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                from ..db import get_or_create_team, get_or_create_player
                
                team_name = team_name_from_abbrev(team) or team
                team_id = get_or_create_team(conn, team_name)
                
                # Try to find existing player
                player_row = conn.execute(
                    "SELECT id FROM players WHERE name LIKE ?", (f"%{player}%",)
                ).fetchone()
                player_id = player_row["id"] if player_row else None
                
                conn.execute(
                    """
                    INSERT INTO injury_report 
                    (game_date, team_id, player_id, player_name, status, minutes_limit, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (game_date, team_id, player_id, player, status, minutes_limit, notes),
                )
                conn.commit()
            
            return jsonify({
                "success": True,
                "player": player,
                "team": team,
                "status": status,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/ingest/injury-report", methods=["POST"])
    def api_ingest_injury_report():
        """Parse and ingest a full injury report from raw text."""
        from ..ingest.injury_parser import parse_injury_report_text, summarize_injury_report
        from ..db import get_or_create_team
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No injury report text provided"}), 400
        
        # Optional: filter to specific date
        filter_date = data.get("date", "").strip() or None
        # Whether to include G-League assignments
        include_g_league = data.get("include_g_league", False)
        
        try:
            report = parse_injury_report_text(text)
            
            db = get_db()
            ingested_count = 0
            skipped_count = 0
            
            with db.connect() as conn:
                for entry in report.entries:
                    # Filter by date if specified
                    if filter_date and entry.game_date != filter_date:
                        skipped_count += 1
                        continue
                    
                    # Skip G-League unless requested
                    if entry.is_g_league and not include_g_league:
                        skipped_count += 1
                        continue
                    
                    team_id = get_or_create_team(conn, entry.team_name)
                    
                    # Try to find existing player
                    player_row = conn.execute(
                        "SELECT id FROM players WHERE name LIKE ?", 
                        (f"%{entry.player_name}%",)
                    ).fetchone()
                    player_id = player_row["id"] if player_row else None
                    
                    # Check if entry already exists
                    existing = conn.execute(
                        """
                        SELECT id FROM injury_report 
                        WHERE game_date = ? AND team_id = ? AND player_name = ?
                        """,
                        (entry.game_date, team_id, entry.player_name),
                    ).fetchone()
                    
                    if existing:
                        # Update existing entry
                        conn.execute(
                            """
                            UPDATE injury_report 
                            SET status = ?, notes = ?, player_id = ?
                            WHERE id = ?
                            """,
                            (entry.status, entry.reason, player_id, existing["id"]),
                        )
                    else:
                        # Insert new entry
                        conn.execute(
                            """
                            INSERT INTO injury_report 
                            (game_date, team_id, player_id, player_name, status, notes)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (entry.game_date, team_id, player_id, entry.player_name, 
                             entry.status, entry.reason),
                        )
                    
                    ingested_count += 1
                
                conn.commit()
            
            summary = summarize_injury_report(report)
            
            return jsonify({
                "success": True,
                "report_date": report.report_date,
                "total_entries": len(report.entries),
                "ingested": ingested_count,
                "skipped": skipped_count,
                "teams_not_submitted": report.teams_not_submitted,
                "summary": summary,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/injuries/clear", methods=["POST"])
    def api_clear_injuries():
        """Clear injury report entries for a specific date or all."""
        data = request.get_json() or {}
        date = data.get("date", "").strip()
        
        db = get_db()
        try:
            with db.connect() as conn:
                if date:
                    conn.execute("DELETE FROM injury_report WHERE game_date = ?", (date,))
                else:
                    conn.execute("DELETE FROM injury_report")
                conn.commit()
            
            return jsonify({
                "success": True,
                "cleared_date": date or "all",
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    # -------------------------------------------------------------------------
    # Defense vs Position Data API
    # -------------------------------------------------------------------------
    
    @app.route("/api/ingest/defense-vs-position", methods=["POST"])
    def api_ingest_defense_vs_position():
        """
        Parse and ingest defense vs position data from Hashtag Basketball.
        
        Expected JSON: { "text": "raw page content", "position": "PG" }
        Position can be: PG, SG, SF, PF, C
        """
        from ..ingest.defense_position_parser import (
            parse_defense_vs_position_text,
            save_defense_vs_position_to_db,
        )
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        position = data.get("position", "").strip().upper()
        
        if not text:
            return jsonify({"error": "No defense data text provided"}), 400
        
        if position and position not in ["PG", "SG", "SF", "PF", "C"]:
            return jsonify({"error": f"Invalid position: {position}. Must be PG, SG, SF, PF, or C"}), 400
        
        try:
            # Parse the text
            result = parse_defense_vs_position_text(text, expected_position=position if position else None)
            
            if not result.rows:
                error_msg = "No valid data rows parsed. "
                if result.errors:
                    error_msg += ". ".join(result.errors)
                return jsonify({"error": error_msg}), 400
            
            # Save to database
            db = get_db()
            with db.connect() as conn:
                save_result = save_defense_vs_position_to_db(conn, result)
            
            return jsonify({
                "success": True,
                "position": save_result.get("position", result.position),
                "as_of_date": save_result.get("as_of_date"),
                "inserted": save_result.get("inserted", 0),
                "updated": save_result.get("updated", 0),
                "total": save_result.get("total", len(result.rows)),
                "source_date": result.last_updated,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/defense-vs-position/status")
    def api_defense_vs_position_status():
        """
        Get the last updated status for all defense vs position data.
        Returns when each position's data was last imported.
        """
        from ..ingest.defense_position_parser import get_defense_vs_position_last_updated
        
        db = get_db()
        with db.connect() as conn:
            try:
                status = get_defense_vs_position_last_updated(conn)
                return jsonify({
                    "positions": status,
                })
            except Exception as e:
                return jsonify({"positions": {}, "error": str(e)})
    
    @app.route("/api/defense-vs-position/<team_abbrev>")
    def api_defense_vs_position_team(team_abbrev: str):
        """
        Get defense vs position data for a specific team.
        Returns how the team defends each position.
        """
        from ..ingest.defense_position_parser import get_all_defense_vs_position_for_team
        
        team_abbrev = team_abbrev.upper()
        
        db = get_db()
        with db.connect() as conn:
            try:
                rows = get_all_defense_vs_position_for_team(conn, team_abbrev)
                
                if not rows:
                    return jsonify({
                        "team": team_abbrev,
                        "positions": [],
                        "note": "No defense vs position data available for this team",
                    })
                
                return jsonify({
                    "team": team_abbrev,
                    "positions": [
                        {
                            "position": row.position,
                            "overall_rank": row.overall_rank,
                            "pts_allowed": row.pts_allowed,
                            "pts_rank": row.pts_rank,
                            "reb_allowed": row.reb_allowed,
                            "reb_rank": row.reb_rank,
                            "ast_allowed": row.ast_allowed,
                            "ast_rank": row.ast_rank,
                            "fg_pct_allowed": row.fg_pct_allowed,
                            "fg_pct_rank": row.fg_pct_rank,
                            "tpm_allowed": row.tpm_allowed,
                            "tpm_rank": row.tpm_rank,
                        }
                        for row in rows
                    ]
                })
            except Exception as e:
                return jsonify({"team": team_abbrev, "positions": [], "error": str(e)})
    
    @app.route("/api/defense-vs-position/analysis/<player_position>/<opponent_abbrev>")
    def api_defense_analysis_for_matchup(player_position: str, opponent_abbrev: str):
        """
        Get matchup analysis for a player position against a specific opponent.
        Returns factors for PTS, REB, AST predictions.
        """
        from ..ingest.defense_position_parser import calculate_defense_factor
        
        player_position = player_position.upper()
        opponent_abbrev = opponent_abbrev.upper()
        
        if player_position not in ["PG", "SG", "SF", "PF", "C"]:
            return jsonify({"error": f"Invalid position: {player_position}"}), 400
        
        db = get_db()
        with db.connect() as conn:
            try:
                pts_factor = calculate_defense_factor(conn, opponent_abbrev, player_position, "pts")
                reb_factor = calculate_defense_factor(conn, opponent_abbrev, player_position, "reb")
                ast_factor = calculate_defense_factor(conn, opponent_abbrev, player_position, "ast")
                
                if not pts_factor:
                    return jsonify({
                        "position": player_position,
                        "opponent": opponent_abbrev,
                        "note": "No defense vs position data available",
                        "factors": None,
                    })
                
                return jsonify({
                    "position": player_position,
                    "opponent": opponent_abbrev,
                    "factors": {
                        "pts": pts_factor,
                        "reb": reb_factor,
                        "ast": ast_factor,
                    },
                    "recommendation": {
                        "pts": "OVER" if pts_factor["factor"] > 1.05 else ("UNDER" if pts_factor["factor"] < 0.95 else "NEUTRAL"),
                        "reb": "OVER" if reb_factor["factor"] > 1.05 else ("UNDER" if reb_factor["factor"] < 0.95 else "NEUTRAL"),
                        "ast": "OVER" if ast_factor["factor"] > 1.05 else ("UNDER" if ast_factor["factor"] < 0.95 else "NEUTRAL"),
                    }
                })
            except Exception as e:
                return jsonify({
                    "position": player_position,
                    "opponent": opponent_abbrev,
                    "error": str(e),
                })

    # -------------------------------------------------------------------------
    # Player DRTG (Defensive Rating) API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/ingest/player-drtg", methods=["POST"])
    def api_ingest_player_drtg():
        """
        Parse and ingest player DRTG data from StatMuse or similar sources.
        
        Expected JSON: { "text": "raw page content", "team": "PHX", "season": "2025-26" }
        Team is optional - will be auto-detected from data.
        """
        from ..ingest.player_drtg_parser import parse_player_drtg_text, save_player_drtg_to_db
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        team = data.get("team", "").strip().upper() or None
        season = data.get("season", "2025-26").strip()
        
        if not text:
            return jsonify({"error": "No DRTG data text provided"}), 400
        
        try:
            # Parse the text (team will be auto-detected if not provided)
            result = parse_player_drtg_text(text, expected_team=team, expected_season=season)
            
            if not result.rows:
                error_msg = "No valid DRTG data rows parsed. "
                if result.errors:
                    error_msg += ". ".join(result.errors)
                return jsonify({"error": error_msg}), 400
            
            # Save to database
            db = get_db()
            with db.connect() as conn:
                save_result = save_player_drtg_to_db(conn, result)
            
            detected_team = save_result.get("team", result.team_abbrev)
            was_auto_detected = team is None and detected_team is not None
            
            return jsonify({
                "success": True,
                "team": detected_team,
                "team_auto_detected": was_auto_detected,
                "season": season,
                "inserted": save_result.get("inserted", 0),
                "updated": save_result.get("updated", 0),
                "total": save_result.get("total", len(result.rows)),
                "players": [
                    {"name": r.name, "drtg": r.drtg, "rank": r.rank}
                    for r in result.rows[:10]
                ],
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/player-drtg/<team_abbrev>")
    def api_player_drtg_team(team_abbrev: str):
        """Get player DRTG data for a specific team."""
        from ..ingest.player_drtg_parser import get_team_drtg_rankings, get_drtg_data_freshness
        
        team_abbrev = team_abbrev.upper()
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        with db.connect() as conn:
            rankings = get_team_drtg_rankings(conn, team_abbrev, season)
            freshness = get_drtg_data_freshness(conn, team_abbrev)
        
        if not rankings:
            return jsonify({
                "team": team_abbrev,
                "players": [],
                "note": "No DRTG data available for this team",
            })
        
        return jsonify({
            "team": team_abbrev,
            "season": season,
            "last_updated": freshness.get("last_updated"),
            "players": [
                {
                    "rank": r.rank,
                    "name": r.name,
                    "drtg": r.drtg,
                    "games_played": r.games_played,
                    "minutes_per_game": r.minutes_per_game,
                    "ppg": r.ppg,
                    "rpg": r.rpg,
                    "apg": r.apg,
                    "spg": r.spg,
                    "bpg": r.bpg,
                    "plus_minus": r.plus_minus,
                }
                for r in rankings
            ],
        })
    
    @app.route("/api/player-drtg/league")
    def api_player_drtg_league():
        """Get league-wide DRTG rankings (best defenders)."""
        from ..ingest.player_drtg_parser import get_league_drtg_rankings
        
        season = request.args.get("season", "2025-26")
        limit = request.args.get("limit", 50, type=int)
        min_minutes = request.args.get("min_minutes", 15.0, type=float)
        
        db = get_db()
        with db.connect() as conn:
            rankings = get_league_drtg_rankings(conn, season, limit, min_minutes)
        
        return jsonify({
            "season": season,
            "min_minutes": min_minutes,
            "players": [
                {
                    "rank": r.rank,
                    "name": r.name,
                    "team": r.team_abbrev,
                    "drtg": r.drtg,
                    "games_played": r.games_played,
                    "minutes_per_game": r.minutes_per_game,
                    "ppg": r.ppg,
                    "rpg": r.rpg,
                    "spg": r.spg,
                    "bpg": r.bpg,
                }
                for r in rankings
            ],
        })
    
    @app.route("/api/player-drtg/status")
    def api_player_drtg_status():
        """Get DRTG data freshness status for all teams."""
        from ..ingest.player_drtg_parser import (
            get_drtg_data_freshness,
            get_teams_needing_drtg_update,
        )
        
        max_age = request.args.get("max_age", 14, type=int)
        
        db = get_db()
        with db.connect() as conn:
            freshness = get_drtg_data_freshness(conn)
            needs_update = get_teams_needing_drtg_update(conn, max_age)
        
        return jsonify({
            "teams_with_data": freshness,
            "teams_count": len(freshness),
            "needs_update": needs_update,
            "needs_update_count": len(needs_update),
            "max_age_days": max_age,
        })
    
    @app.route("/api/player/<player_name>/drtg")
    def api_player_drtg_individual(player_name: str):
        """Get DRTG data for an individual player."""
        from ..ingest.player_drtg_parser import get_player_drtg
        
        season = request.args.get("season", "2025-26")
        
        db = get_db()
        with db.connect() as conn:
            drtg = get_player_drtg(conn, player_name, season)
        
        if not drtg:
            return jsonify({"error": "Player DRTG data not found"}), 404
        
        return jsonify({
            "player": drtg.name,
            "team": drtg.team_abbrev,
            "season": drtg.season,
            "drtg": drtg.drtg,
            "rank": drtg.rank,
            "games_played": drtg.games_played,
            "minutes_per_game": drtg.minutes_per_game,
            "ppg": drtg.ppg,
            "rpg": drtg.rpg,
            "apg": drtg.apg,
            "spg": drtg.spg,
            "bpg": drtg.bpg,
            "plus_minus": drtg.plus_minus,
        })

    @app.route("/api/ingest/matchups", methods=["POST"])
    def api_ingest_matchups():
        """Parse matchup text and store for today's games."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No matchup text provided"}), 400
        
        try:
            matchups = parse_matchups_text(text)
            
            if not matchups:
                return jsonify({"error": "No matchups could be parsed from text"}), 400
            
            # Determine if date was auto-extracted (check if all matchups have same date)
            detected_date = matchups[0].game_date if matchups else None
            
            # Store matchups in session/memory for analysis
            # We'll return them for client-side usage
            result_matchups = []
            for m in matchups:
                result_matchups.append({
                    "game_date": m.game_date,
                    "away_team": m.away_team,
                    "home_team": m.home_team,
                    "away_abbrev": m.away_abbrev,
                    "home_abbrev": m.home_abbrev,
                    "game_time": m.game_time,
                    "spread": m.spread,
                    "favorite_abbrev": m.favorite_abbrev,
                    "over_under": m.over_under,
                    "tv_channel": m.tv_channel,
                })
            
            return jsonify({
                "success": True,
                "count": len(result_matchups),
                "detected_date": detected_date,
                "matchups": result_matchups,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/recalculate", methods=["POST"])
    def api_recalculate():
        """
        Trigger recalculation of all projections and model updates.
        This should be called after adding all box scores for a day.
        """
        from ..engine.matchup_advisor import generate_comprehensive_matchup_report
        
        data = request.get_json() or {}
        target_date = data.get("date", "").strip() or datetime.now().strftime("%Y-%m-%d")
        
        db = get_db()
        try:
            with db.connect() as conn:
                # 1. Get count of games added recently
                games_today = conn.execute(
                    "SELECT COUNT(*) as n FROM games WHERE game_date = ?",
                    (target_date,),
                ).fetchone()["n"]
                
                # 2. Get count of all games
                total_games = conn.execute(
                    "SELECT COUNT(*) as n FROM games"
                ).fetchone()["n"]
                
                # 3. Get latest game date
                latest = conn.execute(
                    "SELECT game_date FROM games ORDER BY game_date DESC LIMIT 1"
                ).fetchone()
                latest_date = latest["game_date"] if latest else None
                
                # 4. Calculate updated team defense ratings
                from ..engine.game_context import get_all_team_defense_ratings
                defense_ratings = get_all_team_defense_ratings(conn)
                
                # 5. Get summary stats
                players = conn.execute("SELECT COUNT(*) as n FROM players").fetchone()["n"]
                
            return jsonify({
                "success": True,
                "message": "Model recalculated with latest data",
                "stats": {
                    "games_for_date": games_today,
                    "total_games": total_games,
                    "total_players": players,
                    "latest_game_date": latest_date,
                    "teams_with_defense_data": len(defense_ratings),
                },
                "date": target_date,
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    # -------------------------------------------------------------------------
    # Usage Redistribution API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/usage/<abbrev>")
    def api_team_usage(abbrev: str):
        """Get usage profiles for a team."""
        from ..engine.usage_redistribution import get_team_usage_profiles
        
        abbrev = abbrev.upper()
        db = get_db()
        with db.connect() as conn:
            profiles = get_team_usage_profiles(conn, abbrev)
        
        
        if not profiles:
            return jsonify({"error": "No data for team"}), 404
        
        return jsonify({
            "team": abbrev,
            "profiles": [
                {
                    "player_id": p.player_id,
                    "player": p.player_name,
                    "position": p.position,
                    "avg_minutes": p.avg_minutes,
                    "avg_pts": p.avg_pts,
                    "avg_reb": p.avg_reb,
                    "avg_ast": p.avg_ast,
                    "games": p.games_played,
                    "usage_rate": p.usage_rate,
                    "minutes_share": p.minutes_share,
                    "assist_rate": p.assist_rate,
                    "is_primary_scorer": p.is_primary_scorer,
                    "is_primary_playmaker": p.is_primary_playmaker,
                    "tier": p.tier,
                }
                for p in profiles
            ],
        })
    
    @app.route("/api/usage/<abbrev>/impact")
    def api_usage_impact(abbrev: str):
        """Calculate usage redistribution when a player is out."""
        from ..engine.usage_redistribution import (
            calculate_usage_redistribution,
            get_historical_impact,
        )
        
        abbrev = abbrev.upper()
        absent_player = request.args.get("out", "").strip()
        include_historical = request.args.get("historical", "false").lower() == "true"
        
        if not absent_player:
            return jsonify({"error": "Missing 'out' parameter (player who is absent)"}), 400
        
        db = get_db()
        with db.connect() as conn:
            result = calculate_usage_redistribution(conn, abbrev, absent_player)
            
            if not result:
                return jsonify({"error": f"Player not found: {absent_player}"}), 404
            
            response = {
                "team": abbrev,
                "absent_player": result.absent_player,
                "absent_stats": result.absent_stats,
                "redistributions": result.redistributions,
                "total_redistributed": {
                    "pts": result.total_pts_redistributed,
                    "reb": result.total_reb_redistributed,
                    "ast": result.total_ast_redistributed,
                },
            }
            
            if include_historical:
                historical = get_historical_impact(conn, abbrev, absent_player)
                if historical:
                    response["historical"] = historical
            
            return jsonify(response)
    
    @app.route("/api/validation")
    def api_validation():
        """Run data validation checks."""
        from ..validation import run_all_validations
        
        db = get_db()
        with db.connect() as conn:
            report = run_all_validations(conn)
        
        return jsonify({
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "errors": report.errors,
            "warnings": report.warnings,
            "is_valid": report.is_valid(),
            "results": [
                {
                    "check": r.check_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "message": r.message,
                    "details": r.details[:5] if not r.passed else [],
                }
                for r in report.results
            ],
        })
    
    @app.route("/api/matchup/recommendations")
    def api_matchup_recommendations():
        """Get matchup-specific prop recommendations."""
        from ..engine.game_context import generate_matchup_recommendations
        
        away = request.args.get("away", "").strip().upper()
        home = request.args.get("home", "").strip().upper()
        game_date = request.args.get("date", "").strip() or datetime.now().strftime("%Y-%m-%d")
        min_edge = request.args.get("min_edge", 5.0, type=float)
        
        if not away or not home:
            return jsonify({"error": "Missing away or home team"}), 400
        
        db = get_db()
        with db.connect() as conn:
            recommendations = generate_matchup_recommendations(
                conn, away, home, game_date, min_edge
            )
        
        return jsonify({
            "matchup": f"{away} @ {home}",
            "date": game_date,
            "recommendations": [
                {
                    "player": r.player_name,
                    "team": r.team_abbrev,
                    "opponent": r.opponent_abbrev,
                    "prop": r.prop_type,
                    "direction": r.direction,
                    "baseline": r.baseline_value,
                    "adjusted": r.adjusted_value,
                    "line": r.line,
                    "defense_rating": r.defense_rating,
                    "back_to_back": r.back_to_back,
                    "rest_advantage": r.rest_advantage,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                }
                for r in recommendations
            ],
        })
    
    @app.route("/api/matchup/player-history")
    def api_player_history():
        """Get player's historical performance against a team."""
        from ..engine.game_context import get_player_vs_team_history
        
        player = request.args.get("player", "").strip()
        opponent = request.args.get("opponent", "").strip().upper()
        
        if not player or not opponent:
            return jsonify({"error": "Missing player or opponent"}), 400
        
        db = get_db()
        with db.connect() as conn:
            history = get_player_vs_team_history(conn, player, opponent)
        
        if not history:
            return jsonify({"error": "No history found"}), 404
        
        return jsonify(history)
    
    @app.route("/api/matchup/position-defense")
    def api_position_defense():
        """Get team's defensive rating against a position."""
        from ..engine.game_context import get_position_defense_rating
        
        team = request.args.get("team", "").strip().upper()
        position = request.args.get("position", "").strip().upper()
        
        if not team or not position:
            return jsonify({"error": "Missing team or position"}), 400
        
        db = get_db()
        with db.connect() as conn:
            rating = get_position_defense_rating(conn, team, position)
        
        if not rating:
            return jsonify({"error": "Insufficient data"}), 404
        
        return jsonify(rating)
    
    # -------------------------------------------------------------------------
    # Edge Alerts API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/alerts")
    def api_alerts():
        """Get edge alerts for a date."""
        from ..engine.alerts import scan_for_edge_alerts, daily_edge_report
        
        scan_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        min_edge = request.args.get("min_edge", 5.0, type=float)
        full_report = request.args.get("full_report", "false").lower() == "true"
        
        db = get_db()
        with db.connect() as conn:
            if full_report:
                report = daily_edge_report(conn, scan_date, min_edge)
                return jsonify(report)
            else:
                result = scan_for_edge_alerts(conn, scan_date, min_edge)
                return jsonify({
                    "date": result.scan_date,
                    "lines_scanned": result.lines_scanned,
                    "alerts_found": result.alerts_found,
                    "alerts": [
                        {
                            "player": a.player_name,
                            "team": a.team_abbrev,
                            "prop": a.prop_type,
                            "direction": a.direction,
                            "line": a.line,
                            "projection": a.projected_value,
                            "edge_pct": a.edge_pct,
                            "confidence": a.confidence,
                            "reasons": a.reasons,
                            "over_prob": a.over_probability,
                            "under_prob": a.under_probability,
                        }
                        for a in result.all_alerts
                    ],
                })
    
    @app.route("/api/alerts/team/<abbrev>")
    def api_alerts_team(abbrev: str):
        """Get edge alerts for a specific team."""
        from ..engine.alerts import find_value_plays_by_team
        
        abbrev = abbrev.upper()
        scan_date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        min_edge = request.args.get("min_edge", 3.0, type=float)
        
        db = get_db()
        with db.connect() as conn:
            alerts = find_value_plays_by_team(conn, abbrev, scan_date, min_edge)
        
        return jsonify({
            "team": abbrev,
            "date": scan_date,
            "alerts": [
                {
                    "player": a.player_name,
                    "prop": a.prop_type,
                    "direction": a.direction,
                    "line": a.line,
                    "projection": a.projected_value,
                    "edge_pct": a.edge_pct,
                    "confidence": a.confidence,
                }
                for a in alerts
            ],
        })
    
    @app.route("/api/backtest")
    def api_backtest():
        """Run a backtest over a date range."""
        from ..engine.backtesting import run_backtest
        
        start_date = request.args.get("start", "")
        end_date = request.args.get("end", "")
        min_edge = request.args.get("min_edge", 3.0, type=float)
        
        if not start_date or not end_date:
            return jsonify({"error": "Missing start or end date"}), 400
        
        db = get_db()
        with db.connect() as conn:
            result = run_backtest(conn, start_date, end_date, min_edge)
        
        return jsonify({
            "start_date": result.start_date,
            "end_date": result.end_date,
            "total_props": result.total_props,
            "hits": result.hits,
            "misses": result.misses,
            "hit_rate": round(result.hit_rate, 3),
            "by_type": {
                "pts": {"hits": result.pts_hits, "total": result.pts_total},
                "reb": {"hits": result.reb_hits, "total": result.reb_total},
                "ast": {"hits": result.ast_hits, "total": result.ast_total},
            },
            "by_confidence": {
                "high": {"hits": result.high_conf_hits, "total": result.high_conf_total},
                "medium": {"hits": result.med_conf_hits, "total": result.med_conf_total},
                "low": {"hits": result.low_conf_hits, "total": result.low_conf_total},
            },
            "theoretical": {
                "profit": round(result.theoretical_profit, 2),
                "wagers": result.theoretical_wagers,
                "roi": round(result.theoretical_roi, 2),
            },
            "calibration": result.calibration_bins,
        })
    
    # -------------------------------------------------------------------------
    # Scheduled Games / Matchups API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/scheduled-games")
    def api_scheduled_games():
        """Get scheduled games for a date."""
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        db = get_db()
        with db.connect() as conn:
            rows = conn.execute(
                """
                SELECT sg.id, sg.game_date, sg.game_time, 
                       t1.name AS away_team, t2.name AS home_team,
                       sg.spread, sg.over_under, sg.tv_channel, sg.status
                FROM scheduled_games sg
                JOIN teams t1 ON t1.id = sg.away_team_id
                JOIN teams t2 ON t2.id = sg.home_team_id
                WHERE sg.game_date = ?
                ORDER BY sg.game_time
                """,
                (date,),
            ).fetchall()
            
            # Get team records from standings
            standings = compute_conference_standings(conn)
            team_records = {}
            for conf in ["East", "West"]:
                for row in standings.get(conf, []):
                    team_records[row.abbr] = f"{row.wins}-{row.losses}"
        
        games = []
        for r in rows:
            away_abbrev = abbrev_from_team_name(r["away_team"]) or ""
            home_abbrev = abbrev_from_team_name(r["home_team"]) or ""
            
            games.append({
                "id": r["id"],
                "date": r["game_date"],
                "game_time": r["game_time"],
                "away_team": r["away_team"],
                "home_team": r["home_team"],
                "away_abbrev": away_abbrev,
                "home_abbrev": home_abbrev,
                "away_record": team_records.get(away_abbrev),
                "home_record": team_records.get(home_abbrev),
                "spread": r["spread"],
                "over_under": r["over_under"],
                "tv_channel": r["tv_channel"],
                "status": r["status"],
            })
        
        return jsonify({"games": games, "date": date})
    
    @app.route("/api/scheduled-games", methods=["POST"])
    def api_add_scheduled_game():
        """Add a scheduled game manually."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        away = data.get("away", "").strip().upper()
        home = data.get("home", "").strip().upper()
        date = data.get("date", "").strip()
        time = data.get("time", "").strip() or None
        spread = data.get("spread")
        over_under = data.get("over_under")
        
        if not away or not home or not date:
            return jsonify({"error": "Away team, home team, and date are required"}), 400
        
        away_team = team_name_from_abbrev(away)
        home_team = team_name_from_abbrev(home)
        
        if not away_team or not home_team:
            return jsonify({"error": "Invalid team abbreviation"}), 400
        
        db = get_db()
        try:
            with db.connect() as conn:
                from ..db import get_or_create_team
                
                away_id = get_or_create_team(conn, away_team)
                home_id = get_or_create_team(conn, home_team)
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO scheduled_games 
                    (game_date, game_time, away_team_id, home_team_id, spread, over_under)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (date, time, away_id, home_id, spread, over_under),
                )
                
                # Update data freshness
                conn.execute(
                    """
                    INSERT OR REPLACE INTO data_freshness (data_type, last_updated, records_count)
                    VALUES ('matchups', datetime('now'), 
                            (SELECT COUNT(*) FROM scheduled_games WHERE game_date = ?))
                    """,
                    (date,),
                )
                
                conn.commit()
            
            return jsonify({
                "success": True,
                "matchup": f"{away} @ {home}",
                "date": date,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/scheduled-games/parse", methods=["POST"])
    def api_parse_scheduled_games():
        """Parse and add scheduled games from pasted text."""
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        try:
            matchups = parse_matchups_text(text)
            if not matchups:
                return jsonify({"error": "No matchups could be parsed from text"}), 400
            
            db = get_db()
            count = 0
            game_date = matchups[0].game_date if matchups else None
            
            with db.connect() as conn:
                from ..db import get_or_create_team
                
                for m in matchups:
                    away_id = get_or_create_team(conn, m.away_team)
                    home_id = get_or_create_team(conn, m.home_team)
                    
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO scheduled_games 
                        (game_date, game_time, away_team_id, home_team_id, 
                         spread, over_under, tv_channel)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (m.game_date, m.game_time, away_id, home_id, 
                         m.spread, m.over_under, m.tv_channel),
                    )
                    count += 1
                
                # Update data freshness
                conn.execute(
                    """
                    INSERT OR REPLACE INTO data_freshness (data_type, last_updated, records_count)
                    VALUES ('matchups', datetime('now'), ?)
                    """,
                    (count,),
                )
                
                conn.commit()
            
            return jsonify({
                "success": True,
                "count": count,
                "date": game_date,
                "matchups": [
                    {"away": m.away_abbrev, "home": m.home_abbrev, "spread": m.spread, "ou": m.over_under}
                    for m in matchups
                ],
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/scheduled-games/<int:game_id>", methods=["DELETE"])
    def api_delete_scheduled_game(game_id: int):
        """Delete a scheduled game."""
        db = get_db()
        try:
            with db.connect() as conn:
                conn.execute("DELETE FROM scheduled_games WHERE id = ?", (game_id,))
                conn.commit()
            return jsonify({"success": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    # -------------------------------------------------------------------------
    # Data Freshness API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/data-freshness")
    def api_data_freshness():
        """Get data freshness status to avoid stale data issues."""
        db = get_db()
        with db.connect() as conn:
            # Get freshness data
            rows = conn.execute(
                "SELECT data_type, last_updated, records_count FROM data_freshness"
            ).fetchall()
            
            # Get latest game date
            latest_game = conn.execute(
                "SELECT game_date FROM games ORDER BY game_date DESC LIMIT 1"
            ).fetchone()
            
            # Get scheduled games count for today
            today = datetime.now().strftime("%Y-%m-%d")
            scheduled_today = conn.execute(
                "SELECT COUNT(*) as cnt FROM scheduled_games WHERE game_date = ?",
                (today,),
            ).fetchone()
        
        freshness = {r["data_type"]: {
            "last_updated": r["last_updated"],
            "records_count": r["records_count"],
        } for r in rows}
        
        # Calculate staleness
        latest_game_date = latest_game["game_date"] if latest_game else None
        hours_since_game = None
        is_stale = False
        
        if latest_game_date:
            try:
                latest_dt = datetime.strptime(latest_game_date, "%Y-%m-%d")
                hours_since_game = (datetime.now() - latest_dt).total_seconds() / 3600
                # Consider stale if no game data for more than 48 hours
                is_stale = hours_since_game > 48
            except ValueError:
                pass
        
        return jsonify({
            "freshness": freshness,
            "latest_game_date": latest_game_date,
            "hours_since_update": round(hours_since_game, 1) if hours_since_game else None,
            "is_stale": is_stale,
            "last_update": latest_game_date,
            "scheduled_games_today": scheduled_today["cnt"] if scheduled_today else 0,
        })
    
    @app.route("/api/data-freshness/update", methods=["POST"])
    def api_update_freshness():
        """Manually update data freshness timestamps."""
        data = request.get_json() or {}
        data_type = data.get("type", "general")
        
        db = get_db()
        try:
            with db.connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO data_freshness (data_type, last_updated)
                    VALUES (?, datetime('now'))
                    """,
                    (data_type,),
                )
                conn.commit()
            return jsonify({"success": True, "type": data_type})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    # -------------------------------------------------------------------------
    # Top Picks API Endpoint
    # -------------------------------------------------------------------------
    
    @app.route("/api/top-picks")
    def api_top_picks():
        """Get top prop picks across all scheduled games for a date."""
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        min_edge = request.args.get("min_edge", 5.0, type=float)
        limit = request.args.get("limit", 10, type=int)
        
        db = get_db()
        all_picks = []
        
        try:
            with db.connect() as conn:
                # Get all scheduled games for the date
                games = conn.execute(
                    """
                    SELECT sg.id, t1.name AS away_team, t2.name AS home_team,
                           sg.spread, sg.over_under
                    FROM scheduled_games sg
                    JOIN teams t1 ON t1.id = sg.away_team_id
                    JOIN teams t2 ON t2.id = sg.home_team_id
                    WHERE sg.game_date = ?
                    """,
                    (date,),
                ).fetchall()
                
                for game in games:
                    away_abbrev = abbrev_from_team_name(game["away_team"]) or ""
                    home_abbrev = abbrev_from_team_name(game["home_team"]) or ""
                    
                    if not away_abbrev or not home_abbrev:
                        continue
                    
                    # Check if this is a close game (better for props)
                    is_close = game["spread"] is not None and abs(game["spread"]) <= 6
                    
                    try:
                        report = generate_prop_report(
                            conn=conn,
                            away_abbrev=away_abbrev,
                            home_abbrev=home_abbrev,
                            game_date=date,
                            lines_date=date,
                        )
                        
                        for rec in report.get("recommendations", []):
                            if rec.get("edge_pct", 0) >= min_edge:
                                pick = {
                                    "player": rec["player"],
                                    "team": rec["team"],
                                    "opponent": home_abbrev if rec["team"] == away_abbrev else away_abbrev,
                                    "prop": rec["prop"],
                                    "line": rec["line"],
                                    "projection": rec["projected"],
                                    "direction": rec["recommendation"],
                                    "edge_pct": rec["edge_pct"],
                                    "confidence": rec["confidence"],
                                    "over_prob": rec.get("over_prob"),
                                    "under_prob": rec.get("under_prob"),
                                    "is_close_game": is_close,
                                    "reasons": [],
                                }
                                
                                # Add reasoning
                                if is_close:
                                    pick["reasons"].append("Close game (spread ≤6)")
                                if rec["edge_pct"] >= 10:
                                    pick["reasons"].append(f"Strong edge: {rec['edge_pct']:.1f}%")
                                if rec["confidence"] == "HIGH":
                                    pick["reasons"].append("High confidence projection")
                                
                                all_picks.append(pick)
                    except Exception:
                        continue
            
            # Sort by edge and limit
            all_picks.sort(key=lambda x: -x["edge_pct"])
            all_picks = all_picks[:limit]
            
            return jsonify({
                "date": date,
                "picks": all_picks,
                "total_games": len(games) if games else 0,
            })
        except Exception as e:
            return jsonify({"error": str(e), "picks": []}), 400
    
    # -------------------------------------------------------------------------
    # Player Trends API Endpoint
    # -------------------------------------------------------------------------
    
    @app.route("/api/player/<player_name>/trends")
    def api_player_trends(player_name: str):
        """Get recent performance trends for a player."""
        games_sample = request.args.get("games", 5, type=int)
        
        db = get_db()
        with db.connect() as conn:
            # Find player
            player_row = conn.execute(
                "SELECT id FROM players WHERE name LIKE ?", (f"%{player_name}%",)
            ).fetchone()
            
            if not player_row:
                return jsonify({"error": "Player not found"}), 404
            
            player_id = player_row["id"]
            
            # Get recent games
            recent_games = conn.execute(
                """
                SELECT g.game_date, b.pts, b.reb, b.ast, b.minutes,
                       t.name as team, t2.name as opponent
                FROM boxscore_player b
                JOIN games g ON g.id = b.game_id
                JOIN teams t ON t.id = b.team_id
                LEFT JOIN teams t2 ON (t2.id = g.team1_id OR t2.id = g.team2_id) AND t2.id != t.id
                WHERE b.player_id = ? AND b.minutes IS NOT NULL AND b.minutes > 0
                ORDER BY g.game_date DESC
                LIMIT ?
                """,
                (player_id, games_sample),
            ).fetchall()
            
            if not recent_games:
                return jsonify({"error": "No recent games"}), 404
            
            # Calculate trends
            pts_values = [g["pts"] or 0 for g in recent_games]
            reb_values = [g["reb"] or 0 for g in recent_games]
            ast_values = [g["ast"] or 0 for g in recent_games]
            min_values = [g["minutes"] or 0 for g in recent_games]
            
            avg_pts = sum(pts_values) / len(pts_values)
            avg_reb = sum(reb_values) / len(reb_values)
            avg_ast = sum(ast_values) / len(ast_values)
            avg_min = sum(min_values) / len(min_values)
            
            # Determine trend direction (compare first half to second half)
            mid = len(pts_values) // 2
            if mid > 0:
                recent_pts = sum(pts_values[:mid]) / mid
                older_pts = sum(pts_values[mid:]) / (len(pts_values) - mid)
                pts_trend = "up" if recent_pts > older_pts * 1.05 else "down" if recent_pts < older_pts * 0.95 else "stable"
                
                recent_reb = sum(reb_values[:mid]) / mid
                older_reb = sum(reb_values[mid:]) / (len(reb_values) - mid)
                reb_trend = "up" if recent_reb > older_reb * 1.05 else "down" if recent_reb < older_reb * 0.95 else "stable"
                
                recent_ast = sum(ast_values[:mid]) / mid
                older_ast = sum(ast_values[mid:]) / (len(ast_values) - mid)
                ast_trend = "up" if recent_ast > older_ast * 1.05 else "down" if recent_ast < older_ast * 0.95 else "stable"
            else:
                pts_trend = reb_trend = ast_trend = "stable"
            
            # Hot/cold streak
            hot_streak = 0
            cold_streak = 0
            for pts in pts_values:
                if pts > avg_pts:
                    hot_streak += 1
                else:
                    break
            for pts in pts_values:
                if pts < avg_pts:
                    cold_streak += 1
                else:
                    break
        
        return jsonify({
            "player": player_name,
            "games_sample": len(recent_games),
            "averages": {
                "pts": round(avg_pts, 1),
                "reb": round(avg_reb, 1),
                "ast": round(avg_ast, 1),
                "min": round(avg_min, 1),
            },
            "trends": {
                "pts": pts_trend,
                "reb": reb_trend,
                "ast": ast_trend,
            },
            "streaks": {
                "hot": hot_streak,
                "cold": cold_streak,
            },
            "recent_games": [
                {
                    "date": g["game_date"],
                    "pts": g["pts"],
                    "reb": g["reb"],
                    "ast": g["ast"],
                    "min": g["minutes"],
                    "opponent": abbrev_from_team_name(g["opponent"]) if g["opponent"] else None,
                }
                for g in recent_games
            ],
        })
    
    # -------------------------------------------------------------------------
    # Backtesting API Endpoints
    # -------------------------------------------------------------------------
    
    @app.route("/api/backtesting/generate-picks", methods=["POST"])
    def api_backtesting_generate_picks():
        """
        Generate top 40 picks for a given date based on scheduled matchups.
        This analyzes all matchups and returns the best betting opportunities.
        Stores picks in the database for historical tracking.
        """
        from ..engine.matchup_advisor import generate_comprehensive_matchup_report
        import json as json_lib
        
        data = request.get_json() or {}
        target_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
        force = bool(data.get("force", False))
        
        db = get_db()
        all_picks = []
        
        try:
            with db.connect() as conn:
                # If picks already exist for this date, return them instead of regenerating.
                # This prevents accidental wiping of graded results and avoids perceived "duplicate" history.
                if not force:
                    existing = conn.execute(
                        """
                        SELECT mp.*, mpr.actual_value, mpr.hit, mpr.margin, mpr.graded_at
                        FROM model_picks mp
                        LEFT JOIN model_pick_results mpr ON mpr.pick_id = mp.id
                        WHERE mp.pick_date = ?
                        ORDER BY mp.rank
                        """,
                        (target_date,),
                    ).fetchall()

                    if existing:
                        picks = []
                        for p in existing:
                            reasons = []
                            if p["reasons"]:
                                try:
                                    reasons = json_lib.loads(p["reasons"])
                                except:
                                    reasons = []
                            picks.append({
                                "player": p["player_name"],
                                "player_id": p["player_id"],
                                "team": p["team_abbrev"],
                                "opponent": p["opponent_abbrev"],
                                "prop": p["prop_type"],
                                "projection": p["projection"],
                                "line": p["line"],  # The calculated line (avg of last 10/7/5 games)
                                "direction": p["direction"],
                                "confidence": p["confidence"],
                                "score": p["confidence_score"],
                                "reasons": reasons,
                                "actual": p["actual_value"],
                                "hit": p["hit"],
                                "result": True if p["hit"] in (0, 1) else None,
                            })

                        return jsonify({
                            "date": target_date,
                            "picks": picks,
                            "cached": True,
                            "total_analyzed": None,
                            "matchups_count": None,
                            "insights": None,
                        })

                # Get scheduled games for this date
                scheduled = conn.execute(
                    """
                    SELECT sg.id, t1.name AS away_team, t2.name AS home_team,
                           sg.spread, sg.over_under
                    FROM scheduled_games sg
                    JOIN teams t1 ON t1.id = sg.away_team_id
                    JOIN teams t2 ON t2.id = sg.home_team_id
                    WHERE sg.game_date = ?
                    """,
                    (target_date,),
                ).fetchall()
                
                # Return error if no scheduled games exist
                if not scheduled:
                    return jsonify({
                        "error": f"No scheduled games found for {target_date}. Please add matchups first using the Matchups page.",
                        "date": target_date,
                        "picks": [],
                        "matchups_count": 0,
                    }), 400
                
                for game in scheduled:
                    away_abbrev = abbrev_from_team_name(game["away_team"])
                    home_abbrev = abbrev_from_team_name(game["home_team"])
                    
                    if not away_abbrev or not home_abbrev:
                        continue
                    
                    try:
                        report = generate_comprehensive_matchup_report(
                            conn=conn,
                            away_abbrev=away_abbrev,
                            home_abbrev=home_abbrev,
                            game_date=target_date,
                            spread=float(game["spread"]) if game["spread"] else None,
                            over_under=float(game["over_under"]) if game["over_under"] else None,
                            stars_only=True,  # Only suggest props for designated star players
                        )
                        
                        # FIXED: Include BOTH OVER and UNDER picks from the model
                        # Per Idea.txt: suggest the best picks in order, do not discriminate
                        # Each pick now includes the calculated "line" (average of last 10/7/5 games)
                        
                        # Process OVER plays
                        for edge in report.best_over_plays:
                            all_picks.append({
                                "player": edge.player_name,
                                "player_id": edge.player_id,
                                "team": edge.team_abbrev,
                                "opponent": edge.opponent_abbrev,
                                "prop": edge.prop_type,
                                "projection": edge.adjusted_projection,
                                "line": edge.line,  # Line calculated as avg of last 10/7/5 games
                                "games_for_line": edge.games_for_line,
                                "direction": "OVER",
                                "confidence": edge.confidence_tier,
                                "score": edge.confidence_score + abs(edge.adjustment_pct),
                                "reasons": edge.reasons,
                            })
                        
                        # Process UNDER plays - now included in main model
                        for edge in report.best_under_plays:
                            all_picks.append({
                                "player": edge.player_name,
                                "player_id": edge.player_id,
                                "team": edge.team_abbrev,
                                "opponent": edge.opponent_abbrev,
                                "prop": edge.prop_type,
                                "projection": edge.adjusted_projection,
                                "line": edge.line,  # Line calculated as avg of last 10/7/5 games
                                "games_for_line": edge.games_for_line,
                                "direction": "UNDER",
                                "confidence": edge.confidence_tier,
                                "score": edge.confidence_score + abs(edge.adjustment_pct),
                                "reasons": edge.reasons,
                            })
                            
                    except Exception as e:
                        print(f"Error processing {away_abbrev} vs {home_abbrev}: {e}")
                        continue
                
                # FIXED: Rank ALL picks together (OVER + UNDER) by score
                # Do not discriminate by direction - suggest best picks regardless
                all_picks.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # Take top 40 picks (mix of OVER and UNDER based on quality)
                top_picks = all_picks[:40]
                
                # Replace existing picks for this date and store new ones (forced regeneration)
                conn.execute("DELETE FROM model_picks WHERE pick_date = ?", (target_date,))
                
                for i, pick in enumerate(top_picks):
                    reasons_json = json_lib.dumps(pick.get("reasons", [])) if pick.get("reasons") else None
                    conn.execute(
                        """
                        INSERT INTO model_picks (
                            pick_date, player_id, player_name, team_abbrev, opponent_abbrev,
                            prop_type, direction, projection, line, confidence, confidence_score, reasons, rank
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            target_date,
                            pick.get("player_id"),
                            pick.get("player"),
                            pick.get("team"),
                            pick.get("opponent"),
                            pick.get("prop"),
                            pick.get("direction"),
                            pick.get("projection"),
                            pick.get("line"),  # Line calculated as avg of last 10/7/5 games
                            pick.get("confidence"),
                            pick.get("score"),
                            reasons_json,
                            i + 1,
                        ),
                    )
                conn.commit()
                
                return jsonify({
                    "date": target_date,
                    "picks": top_picks,
                    "total_analyzed": len(all_picks),
                    "matchups_count": len(scheduled),
                    "insights": None,  # Will be populated by frontend
                })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/backtesting/compare-results", methods=["POST"])
    def api_backtesting_compare_results():
        """
        Compare predictions against actual results from box scores.
        Loads picks from database and compares with actual game stats.
        """
        data = request.get_json() or {}
        target_date = data.get("date", datetime.now().strftime("%Y-%m-%d"))
        
        db = get_db()
        results = []
        
        try:
            with db.connect() as conn:
                # Load picks from database for this date
                # FIXED: Now includes 'line' column for proper grading
                picks = conn.execute(
                    """
                    SELECT id, player_name, team_abbrev, opponent_abbrev, prop_type, 
                           direction, projection, line, confidence, confidence_score, rank
                    FROM model_picks
                    WHERE pick_date = ?
                    ORDER BY rank
                    """,
                    (target_date,),
                ).fetchall()
                
                if not picks:
                    return jsonify({
                        "error": f"No picks found for {target_date}. Please generate picks first.",
                        "date": target_date,
                        "results": [],
                    }), 400
                
                # Check if we have box scores for this date
                box_score_count = conn.execute(
                    """
                    SELECT COUNT(*) as cnt FROM boxscore_player bp
                    JOIN games g ON g.id = bp.game_id
                    WHERE g.game_date = ?
                    """,
                    (target_date,),
                ).fetchone()["cnt"]
                
                if box_score_count == 0:
                    return jsonify({
                        "error": f"No box scores found for {target_date}. Please add box scores using the 'Add Game' page or paste box score data.",
                        "date": target_date,
                        "results": [],
                        "no_boxscores": True,
                    }), 400
                
                for pick in picks:
                    pick_id = pick["id"]
                    player_name = pick["player_name"]
                    prop_type = pick["prop_type"].upper()
                    projection = pick["projection"]
                    line = pick["line"]  # The calculated line (avg of last 10/7/5 games)
                    direction = pick["direction"]
                    
                    # Look up actual stats from box scores
                    actual_row = conn.execute(
                        """
                        SELECT b.pts, b.reb, b.ast, b.minutes, p.name
                        FROM boxscore_player b
                        JOIN players p ON p.id = b.player_id
                        JOIN games g ON g.id = b.game_id
                        WHERE p.name LIKE ?
                          AND g.game_date = ?
                        """,
                        (f"%{player_name}%", target_date),
                    ).fetchone()
                    
                    result = {
                        "pick_id": pick_id,
                        "player": player_name,
                        "team": pick["team_abbrev"],
                        "opponent": pick["opponent_abbrev"],
                        "prop": prop_type,
                        "direction": direction,
                        "projection": projection,
                        "line": line,  # Include line in result for transparency
                        "confidence": pick["confidence"],
                        "rank": pick["rank"],
                    }
                    
                    if actual_row:
                        if prop_type == "PTS":
                            actual = actual_row["pts"] or 0
                        elif prop_type == "REB":
                            actual = actual_row["reb"] or 0
                        elif prop_type == "AST":
                            actual = actual_row["ast"] or 0
                        else:
                            actual = 0
                        
                        # FIXED: Grade against LINE if available, else fall back to projection
                        # Per Idea.txt: Line = average of player's last 10/7/5 games
                        grade_target = line if line is not None else projection
                        
                        result["actual"] = actual
                        result["result"] = True
                        result["margin"] = actual - grade_target
                        result["graded_vs"] = "line" if line is not None else "projection"
                        
                        # Determine if hit or miss based on line (not projection)
                        if direction == "OVER":
                            result["hit"] = actual > grade_target
                        else:
                            result["hit"] = actual < grade_target
                        
                        # Update the pick result in database
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO model_pick_results (pick_id, actual_value, hit, margin, graded_at)
                            VALUES (?, ?, ?, ?, datetime('now'))
                            """,
                            (pick_id, actual, 1 if result["hit"] else 0, result["margin"]),
                        )
                    else:
                        result["actual"] = None
                        result["result"] = None
                        result["hit"] = None
                        result["margin"] = None
                    
                    results.append(result)
                
                conn.commit()
                
                # FIXED: Include BOTH OVER and UNDER picks in grading
                # Per Idea.txt: suggest the best picks in order, do not discriminate
                over_results = [r for r in results if r.get("direction") == "OVER"]
                under_results = [r for r in results if r.get("direction") == "UNDER"]
                
                # ALL picks are graded (both OVER and UNDER)
                graded_picks = results  # All picks participate in grading
                
                # Calculate grade based on ALL picks (OVER + UNDER)
                graded_completed = [r for r in graded_picks if r.get("result") is not None]
                graded_hits = [r for r in graded_completed if r.get("hit")]
                
                grade = None
                
                if graded_completed:
                    hit_rate = len(graded_hits) / len(graded_completed) * 100
                    
                    # By prop type for all graded picks
                    pts_picks = [r for r in graded_completed if r.get("prop") == "PTS"]
                    pts_hits = len([r for r in pts_picks if r.get("hit")])
                    pts_accuracy = (pts_hits / len(pts_picks) * 100) if pts_picks else 0
                    
                    reb_picks = [r for r in graded_completed if r.get("prop") == "REB"]
                    reb_hits = len([r for r in reb_picks if r.get("hit")])
                    reb_accuracy = (reb_hits / len(reb_picks) * 100) if reb_picks else 0
                    
                    ast_picks = [r for r in graded_completed if r.get("prop") == "AST"]
                    ast_hits = len([r for r in ast_picks if r.get("hit")])
                    ast_accuracy = (ast_hits / len(ast_picks) * 100) if ast_picks else 0
                    
                    high_conf_picks = [r for r in graded_completed if r.get("confidence") == "HIGH"]
                    high_conf_hits = len([r for r in high_conf_picks if r.get("hit")])
                    high_conf_accuracy = (high_conf_hits / len(high_conf_picks) * 100) if high_conf_picks else 0
                    
                    # Calculate OVER and UNDER hit rates separately for transparency
                    over_graded = [r for r in graded_completed if r.get("direction") == "OVER"]
                    over_graded_hits = len([r for r in over_graded if r.get("hit")])
                    over_hit_rate = (over_graded_hits / len(over_graded) * 100) if over_graded else 0
                    
                    under_graded = [r for r in graded_completed if r.get("direction") == "UNDER"]
                    under_graded_hits = len([r for r in under_graded if r.get("hit")])
                    under_hit_rate = (under_graded_hits / len(under_graded) * 100) if under_graded else 0
                    
                    # Calculate letter grade based on ALL graded picks
                    if hit_rate >= 65:
                        letter = "A"
                    elif hit_rate >= 58:
                        letter = "B"
                    elif hit_rate >= 52:
                        letter = "C"
                    elif hit_rate >= 45:
                        letter = "D"
                    else:
                        letter = "F"
                    
                    grade = {
                        "letter": letter,
                        "hit_rate": hit_rate,
                        "pts_accuracy": pts_accuracy,
                        "reb_accuracy": reb_accuracy,
                        "ast_accuracy": ast_accuracy,
                        "high_conf_accuracy": high_conf_accuracy,
                        "graded_picks_count": len(graded_completed),
                        "over_hit_rate": over_hit_rate,
                        "under_hit_rate": under_hit_rate,
                    }
                    
                    # Update daily performance table with ALL picks stats (OVER + UNDER)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO model_performance_daily (
                            performance_date, total_picks, hits, misses, pending, hit_rate,
                            pts_picks, pts_hits, reb_picks, reb_hits, ast_picks, ast_hits,
                            high_conf_picks, high_conf_hits, over_picks, over_hits, under_picks, under_hits,
                            grade, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                        """,
                        (
                            target_date,
                            len(graded_picks),
                            len(graded_hits),
                            len(graded_completed) - len(graded_hits),
                            len(graded_picks) - len(graded_completed),
                            hit_rate,
                            len(pts_picks), pts_hits,
                            len(reb_picks), reb_hits,
                            len(ast_picks), ast_hits,
                            len(high_conf_picks), high_conf_hits,
                            len(over_graded), over_graded_hits,
                            len(under_graded), under_graded_hits,
                            letter,
                        ),
                    )
                    conn.commit()
                
                return jsonify({
                    "date": target_date,
                    "results": results,
                    "grade": grade,
                    "summary": {
                        "total": len(graded_picks),
                        "completed": len(graded_completed),
                        "hits": len(graded_hits),
                        "misses": len(graded_completed) - len(graded_hits),
                        "pending": len(graded_picks) - len(graded_completed),
                    },
                    "grading_info": {
                        "over_count": len(over_results),
                        "under_count": len(under_results),
                        "note": "Grade is based on ALL picks (OVER + UNDER) graded against line"
                    },
                })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/backtesting/performance")
    def api_backtesting_performance():
        """
        Get overall model performance statistics across all dates.
        """
        db = get_db()
        
        try:
            with db.connect() as conn:
                # Get daily performance records
                daily_stats = conn.execute(
                    """
                    SELECT * FROM model_performance_daily
                    ORDER BY performance_date DESC
                    """
                ).fetchall()
                
                # Calculate aggregate stats
                total_picks = 0
                total_hits = 0
                total_completed = 0
                
                dates = []
                for row in daily_stats:
                    total_picks += row["total_picks"]
                    total_hits += row["hits"]
                    total_completed += row["hits"] + row["misses"]
                    
                    dates.append({
                        "date": row["performance_date"],
                        "total_picks": row["total_picks"],
                        "hits": row["hits"],
                        "misses": row["misses"],
                        "pending": row["pending"],
                        "hit_rate": row["hit_rate"],
                        "grade": row["grade"],
                        "pts_accuracy": (row["pts_hits"] / row["pts_picks"] * 100) if row["pts_picks"] else None,
                        "reb_accuracy": (row["reb_hits"] / row["reb_picks"] * 100) if row["reb_picks"] else None,
                        "ast_accuracy": (row["ast_hits"] / row["ast_picks"] * 100) if row["ast_picks"] else None,
                    })
                
                overall_hit_rate = (total_hits / total_completed * 100) if total_completed > 0 else 0
                
                # Calculate streaks
                current_streak = 0
                best_streak = 0
                worst_streak = 0
                
                return jsonify({
                    "overall": {
                        "total_picks": total_picks,
                        "total_hits": total_hits,
                        "total_misses": total_completed - total_hits,
                        "total_completed": total_completed,
                        "overall_hit_rate": overall_hit_rate,
                        "days_tracked": len(dates),
                    },
                    "daily": dates,
                })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    @app.route("/api/backtesting/picks-history")
    def api_backtesting_picks_history():
        """
        Get historical picks with their results.
        """
        db = get_db()
        limit = request.args.get("limit", 200, type=int)
        date = request.args.get("date")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        
        try:
            with db.connect() as conn:
                where = ""
                params: list[Any] = []

                if date:
                    where = "WHERE mp.pick_date = ?"
                    params.append(date)
                else:
                    if start_date and end_date:
                        where = "WHERE mp.pick_date BETWEEN ? AND ?"
                        params.extend([start_date, end_date])
                    elif start_date:
                        where = "WHERE mp.pick_date >= ?"
                        params.append(start_date)
                    elif end_date:
                        where = "WHERE mp.pick_date <= ?"
                        params.append(end_date)

                picks = conn.execute(
                    f"""
                    SELECT mp.*, mpr.actual_value, mpr.hit, mpr.margin, mpr.graded_at
                    FROM model_picks mp
                    LEFT JOIN model_pick_results mpr ON mpr.pick_id = mp.id
                    {where}
                    ORDER BY mp.pick_date DESC, mp.rank
                    LIMIT ?
                    """,
                    (*params, limit),
                ).fetchall()

                stats_row = conn.execute(
                    f"""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN mpr.hit = 1 THEN 1 ELSE 0 END) AS hits,
                        SUM(CASE WHEN mpr.hit = 0 THEN 1 ELSE 0 END) AS misses,
                        SUM(CASE WHEN mpr.hit IS NULL THEN 1 ELSE 0 END) AS pending,
                        SUM(CASE WHEN mp.prop_type = 'PTS' THEN 1 ELSE 0 END) AS pts,
                        SUM(CASE WHEN mp.prop_type = 'REB' THEN 1 ELSE 0 END) AS reb,
                        SUM(CASE WHEN mp.prop_type = 'AST' THEN 1 ELSE 0 END) AS ast,
                        SUM(CASE WHEN mp.direction = 'OVER' THEN 1 ELSE 0 END) AS over_picks,
                        SUM(CASE WHEN mp.direction = 'UNDER' THEN 1 ELSE 0 END) AS under_picks,
                        SUM(CASE WHEN mp.confidence = 'HIGH' THEN 1 ELSE 0 END) AS high_conf,
                        SUM(CASE WHEN mp.confidence = 'MEDIUM' THEN 1 ELSE 0 END) AS med_conf,
                        SUM(CASE WHEN mp.confidence = 'LOW' THEN 1 ELSE 0 END) AS low_conf
                    FROM model_picks mp
                    LEFT JOIN model_pick_results mpr ON mpr.pick_id = mp.id
                    {where}
                    """,
                    params,
                ).fetchone()
                
                result = []
                for p in picks:
                    reasons = []
                    if p["reasons"]:
                        try:
                            import json as json_lib
                            reasons = json_lib.loads(p["reasons"])
                        except:
                            reasons = []
                    result.append({
                        "id": p["id"],
                        "date": p["pick_date"],
                        "player": p["player_name"],
                        "team": p["team_abbrev"],
                        "opponent": p["opponent_abbrev"],
                        "prop": p["prop_type"],
                        "direction": p["direction"],
                        "projection": p["projection"],
                        "confidence": p["confidence"],
                        "score": p["confidence_score"],
                        "reasons": reasons,
                        "rank": p["rank"],
                        "actual": p["actual_value"],
                        "hit": p["hit"],
                        "result": True if p["hit"] in (0, 1) else None,
                        "margin": p["margin"],
                        "graded_at": p["graded_at"],
                    })

                hits = int(stats_row["hits"] or 0) if stats_row else 0
                misses = int(stats_row["misses"] or 0) if stats_row else 0
                completed = hits + misses
                hit_rate = (hits / completed * 100.0) if completed else None

                # Calculate top 20 + top 10 under stats (graded picks)
                over_picks_list = [p for p in result if p["direction"] == "OVER"]
                under_picks_list = [p for p in result if p["direction"] == "UNDER"]
                
                # Top 20 over + top 10 under based on rank
                top_20_over = [p for p in over_picks_list if p["rank"] and p["rank"] <= 40][:20]
                top_10_under = [p for p in under_picks_list if p["rank"] and p["rank"] > 40][:10]
                graded_picks_list = top_20_over + top_10_under
                
                graded_hits = len([p for p in graded_picks_list if p["hit"] == 1])
                graded_misses = len([p for p in graded_picks_list if p["hit"] == 0])
                graded_completed = graded_hits + graded_misses
                graded_hit_rate = (graded_hits / graded_completed * 100.0) if graded_completed else None

                stats = {
                    "total": int(stats_row["total"] or 0) if stats_row else 0,
                    "hits": hits,
                    "misses": misses,
                    "pending": int(stats_row["pending"] or 0) if stats_row else 0,
                    "hit_rate": hit_rate,
                    "by_prop": {
                        "PTS": int(stats_row["pts"] or 0) if stats_row else 0,
                        "REB": int(stats_row["reb"] or 0) if stats_row else 0,
                        "AST": int(stats_row["ast"] or 0) if stats_row else 0,
                    },
                    "by_direction": {
                        "OVER": int(stats_row["over_picks"] or 0) if stats_row else 0,
                        "UNDER": int(stats_row["under_picks"] or 0) if stats_row else 0,
                    },
                    "by_confidence": {
                        "HIGH": int(stats_row["high_conf"] or 0) if stats_row else 0,
                        "MEDIUM": int(stats_row["med_conf"] or 0) if stats_row else 0,
                        "LOW": int(stats_row["low_conf"] or 0) if stats_row else 0,
                    },
                    # Top 20 + top 10 under stats (what the model is graded on)
                    "graded": {
                        "total": len(graded_picks_list),
                        "hits": graded_hits,
                        "misses": graded_misses,
                        "pending": len(graded_picks_list) - graded_completed,
                        "hit_rate": graded_hit_rate,
                        "top_over_count": len(top_20_over),
                        "top_under_count": len(top_10_under),
                    },
                }

                return jsonify({
                    "picks": result,
                    "stats": stats,
                    "filters": {
                        "date": date,
                        "start_date": start_date,
                        "end_date": end_date,
                        "limit": limit,
                    },
                })
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 400
    
    return app


def run_web_app(host: str = "127.0.0.1", port: int = 5050, debug: bool = True) -> None:
    """Run the Flask web application."""
    app = create_app()
    print(f"\n🏀 NBA Props Predictor")
    print(f"   Running at: http://{host}:{port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)

