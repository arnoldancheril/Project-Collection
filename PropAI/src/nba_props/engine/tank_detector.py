"""
Tank Detection Module
=====================

Detects NBA teams that are "tanking" — intentionally losing games
to improve their draft position.

KEY SIGNALS OF TANKING:
1. Star players getting reduced minutes (minutes restriction)
2. Star players being rested/DNP for "injury management" 
3. Record significantly below .500 (e.g., 20-32)
4. Multiple star players traded away at deadline
5. Playing young/developmental players more minutes
6. Post-deadline lineup changes (starters → bench players starting)

IMPACT ON PREDICTIONS:
- Star players on tanking teams should have REDUCED minute projections
- Higher variance in player output (less predictable lineups)
- UNDER bets on stars become more attractive (limited minutes)
- Tanking teams also affect opponent projections (garbage time, blowouts)

IMPORTANT NUANCE:
- Tanking is not always explicit — teams may deny it
- Some stars genuinely have minor injuries being rested
- We use STATISTICAL SIGNALS, not just team record
- This module auto-detects from boxscore data patterns

Author: PropAI Team
Created: February 2026
"""
from __future__ import annotations

import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Set, Any

from ..db import Db
from ..team_aliases import abbrev_from_team_name, normalize_team_abbrev


# ============================================================================
# Constants
# ============================================================================

# Thresholds for tank detection
TANK_WIN_PCT_THRESHOLD = 0.380      # Below .380 → strong tank signal
TANK_MODERATE_WIN_PCT = 0.430       # Below .430 → moderate signal

# Minutes reduction signals
MINUTES_DROP_THRESHOLD = -12.0      # Star minutes drop > 12% = suspicious
MINUTES_DROP_SEVERE = -20.0         # > 20% drop = very likely tanking

# Game count thresholds
MIN_PRE_DEADLINE_GAMES = 10         # Need at least 10 games before deadline
MIN_POST_DEADLINE_GAMES = 1         # V19.1: Reduced from 3 to 1 for early detection

# DNP frequency thresholds
DNP_SPIKE_THRESHOLD = 2.0           # 2x increase in star DNPs = suspicious

# Tank confidence thresholds
HIGH_TANK_CONFIDENCE = 0.75
MODERATE_TANK_CONFIDENCE = 0.50
LOW_TANK_CONFIDENCE = 0.30

# V19.3: Stealth-tank detection — benching stars in close games
CLOSE_GAME_MARGIN = 10              # Final margin ≤10 = game was close
STEALTH_MINUTES_DROP_PCT = -12.0    # >12% drop in close games = suspicious
MIN_CLOSE_GAMES_REQUIRED = 1        # Minimum close games to evaluate

# V19.3: Known tanking teams watchlist — manually curated for immediate detection
# These teams have been publicly identified as tanking in the 2025-26 season.
# The watchlist supplements automated detection by providing immediate flags
# before enough post-deadline data accumulates for statistical detection.
KNOWN_TANKING_TEAMS: Dict[str, str] = {
    "UTA": "Utah Jazz — benching healthy stars in close 4th quarters",
    "WAS": "Washington Wizards — ruling out healthy players, G-League call-ups",
}

# Cache for tanking detection results (per date)
_tank_detection_cache: Dict[str, Dict[str, TankDetectionResult]] = {}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TankSignal:
    """Individual signal contributing to tank detection."""
    signal_type: str    # e.g., "minutes_reduction", "dnp_increase", "record"
    strength: float     # 0.0 - 1.0
    description: str
    evidence: str       # Specific data supporting this signal


@dataclass
class PlayerMinutesAnalysis:
    """Analysis of a player's minutes before/after a cutoff date."""
    player_name: str
    player_id: int
    pre_avg_minutes: float = 0.0
    post_avg_minutes: float = 0.0
    pre_games: int = 0
    post_games: int = 0
    minutes_change_pct: float = 0.0
    pre_avg_pts: float = 0.0
    post_avg_pts: float = 0.0
    is_star: bool = False
    
    @property
    def has_significant_drop(self) -> bool:
        return self.minutes_change_pct < MINUTES_DROP_THRESHOLD
    
    @property
    def has_severe_drop(self) -> bool:
        return self.minutes_change_pct < MINUTES_DROP_SEVERE


@dataclass 
class TankDetectionResult:
    """Complete tank detection analysis for a team."""
    team_abbrev: str
    is_tanking: bool = False
    confidence: float = 0.0
    signals: List[TankSignal] = field(default_factory=list)
    player_analyses: List[PlayerMinutesAnalysis] = field(default_factory=list)
    
    # Adjustment factors
    star_minutes_factor: float = 1.0     # Multiplier for star player minutes
    overall_confidence_impact: float = 1.0  # Multiplier for prediction confidence
    
    # Context
    team_record: str = ""
    win_pct: float = 0.0
    post_deadline_record: str = ""
    
    def summary(self) -> str:
        """Generate human-readable tank detection summary."""
        lines = []
        emoji = "🏳️" if self.is_tanking else "✅"
        status = "TANKING DETECTED" if self.is_tanking else "COMPETING"
        
        lines.append(f"{emoji} {self.team_abbrev}: {status} (confidence: {self.confidence:.0%})")
        lines.append(f"   Record: {self.team_record} ({self.win_pct:.3f})")
        
        if self.signals:
            lines.append(f"   Signals ({len(self.signals)}):")
            for s in sorted(self.signals, key=lambda x: x.strength, reverse=True):
                strength_bar = "█" * int(s.strength * 5) + "░" * (5 - int(s.strength * 5))
                lines.append(f"     [{strength_bar}] {s.description}")
                lines.append(f"            Evidence: {s.evidence}")
        
        if self.player_analyses:
            affected = [p for p in self.player_analyses if p.has_significant_drop]
            if affected:
                lines.append(f"   Star Players Affected ({len(affected)}):")
                for p in affected:
                    lines.append(
                        f"     - {p.player_name}: {p.pre_avg_minutes:.1f} → "
                        f"{p.post_avg_minutes:.1f} min ({p.minutes_change_pct:+.1f}%)"
                    )
        
        lines.append(f"   → Minutes adjustment factor: {self.star_minutes_factor:.2f}")
        lines.append(f"   → Confidence impact: {self.overall_confidence_impact:.2f}")
        
        return "\n".join(lines)


# ============================================================================
# Core Detection Functions
# ============================================================================

def detect_tanking(
    conn: sqlite3.Connection,
    team_abbrev: str,
    deadline_date: str = "2026-02-06",
    as_of_date: Optional[str] = None,
) -> TankDetectionResult:
    """
    Analyze whether a team is tanking based on statistical signals.
    
    Uses multiple signals:
    1. Overall win percentage
    2. Post-deadline win percentage vs pre-deadline
    3. Star player minutes changes
    4. DNP frequency changes
    5. Trade deadline activity (if trade data exists)
    
    Args:
        conn: Database connection
        team_abbrev: Team abbreviation
        deadline_date: Trade deadline date
        as_of_date: Date to evaluate from (default: today)
    
    Returns:
        TankDetectionResult with full analysis
    """
    team_abbrev = normalize_team_abbrev(team_abbrev) or team_abbrev.upper()
    
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    
    result = TankDetectionResult(team_abbrev=team_abbrev)
    
    # =========================================================================
    # Signal 1: Overall Record
    # =========================================================================
    record = _get_team_record(conn, team_abbrev, as_of_date)
    if record:
        wins, losses = record
        total = wins + losses
        if total > 0:
            win_pct = wins / total
            result.team_record = f"{wins}-{losses}"
            result.win_pct = win_pct
            
            if win_pct < TANK_WIN_PCT_THRESHOLD:
                result.signals.append(TankSignal(
                    signal_type="record_poor",
                    strength=0.7,
                    description="Very poor record suggests tanking",
                    evidence=f"{wins}-{losses} ({win_pct:.3f} win%)"
                ))
            elif win_pct < TANK_MODERATE_WIN_PCT:
                result.signals.append(TankSignal(
                    signal_type="record_below_average",
                    strength=0.3,
                    description="Below average record — possible tanking",
                    evidence=f"{wins}-{losses} ({win_pct:.3f} win%)"
                ))
    
    # =========================================================================
    # Signal 2: Post-Deadline Record Decline
    # =========================================================================
    post_record = _get_team_record(conn, team_abbrev, as_of_date, since_date=deadline_date)
    pre_record = _get_team_record(conn, team_abbrev, deadline_date)
    
    if post_record and pre_record:
        post_wins, post_losses = post_record
        pre_wins, pre_losses = pre_record
        
        post_total = post_wins + post_losses
        pre_total = pre_wins + pre_losses
        
        # V19.4 FIX: Require at least 5 post-deadline games to avoid
        # sampling noise from a 1-2 game losing streak.
        MIN_POST_DEADLINE_GAMES_COLLAPSE = 5
        if post_total >= MIN_POST_DEADLINE_GAMES_COLLAPSE and pre_total >= MIN_PRE_DEADLINE_GAMES:
            post_pct = post_wins / post_total if post_total > 0 else 0
            pre_pct = pre_wins / pre_total if pre_total > 0 else 0
            
            result.post_deadline_record = f"{post_wins}-{post_losses}"
            
            pct_drop = pre_pct - post_pct
            if pct_drop > 0.200:  # 20%+ win rate drop
                result.signals.append(TankSignal(
                    signal_type="post_deadline_collapse",
                    strength=0.8,
                    description="Significant post-deadline performance drop",
                    evidence=f"Pre: {pre_pct:.3f} → Post: {post_pct:.3f} ({pct_drop:+.3f})"
                ))
            elif pct_drop > 0.100:  # 10%+ drop
                result.signals.append(TankSignal(
                    signal_type="post_deadline_decline",
                    strength=0.4,
                    description="Moderate post-deadline performance decline",
                    evidence=f"Pre: {pre_pct:.3f} → Post: {post_pct:.3f} ({pct_drop:+.3f})"
                ))
    
    # =========================================================================
    # Signal 3: Star Player Minutes Changes
    # =========================================================================
    minutes_analyses = _analyze_star_minutes(conn, team_abbrev, deadline_date, as_of_date)
    result.player_analyses = minutes_analyses
    
    stars_with_drop = [p for p in minutes_analyses if p.is_star and p.has_significant_drop]
    stars_with_severe_drop = [p for p in minutes_analyses if p.is_star and p.has_severe_drop]
    
    if stars_with_severe_drop:
        names = ", ".join(p.player_name for p in stars_with_severe_drop)
        avg_drop = statistics.mean(p.minutes_change_pct for p in stars_with_severe_drop)
        result.signals.append(TankSignal(
            signal_type="severe_minutes_reduction",
            strength=0.9,
            description=f"Star players severely limited: {names}",
            evidence=f"Average minutes drop: {avg_drop:.1f}%"
        ))
    elif stars_with_drop:
        names = ", ".join(p.player_name for p in stars_with_drop)
        avg_drop = statistics.mean(p.minutes_change_pct for p in stars_with_drop)
        result.signals.append(TankSignal(
            signal_type="minutes_reduction",
            strength=0.5,
            description=f"Star players getting fewer minutes: {names}",
            evidence=f"Average minutes drop: {avg_drop:.1f}%"
        ))
    
    # =========================================================================
    # Signal 4: Increased DNPs for Starters
    # =========================================================================
    dnp_signal = _analyze_dnp_patterns(conn, team_abbrev, deadline_date, as_of_date)
    if dnp_signal:
        result.signals.append(dnp_signal)
    
    # =========================================================================
    # Signal 5: Trade Deadline Activity
    # =========================================================================
    trade_signal = _analyze_trade_activity(conn, team_abbrev)
    if trade_signal:
        result.signals.append(trade_signal)
    
    # =========================================================================
    # Signal 6: Seller Score (V19.1) — Net seller of talent
    # =========================================================================
    seller_signal = _analyze_seller_score(conn, team_abbrev)
    if seller_signal:
        result.signals.append(seller_signal)
    
    # =========================================================================
    # Signal 7: Standing-Based Risk (V19.1) — Bottom teams auto-flagged
    # =========================================================================
    if record:
        wins, losses = record
        total = wins + losses
        if total >= 30:  # At least 30 games played
            win_pct = wins / total
            # Bottom-5 record teams get auto-flagged for monitoring
            if win_pct < 0.300:  # Below .300 — strong tank territory
                result.signals.append(TankSignal(
                    signal_type="bottom_standing",
                    strength=0.6,
                    description="Bottom-tier record — strong tank territory",
                    evidence=f"{wins}-{losses} ({win_pct:.3f}) — among worst records in NBA"
                ))
            elif win_pct < 0.350:  # Below .350 — monitoring zone
                result.signals.append(TankSignal(
                    signal_type="poor_standing",
                    strength=0.3,
                    description="Poor record — potential tank candidate",
                    evidence=f"{wins}-{losses} ({win_pct:.3f})"
                ))
    
    # =========================================================================
    # Signal 8: Minutes Cliff (V19.1) — Sudden star minutes drop game-over-game
    # =========================================================================
    cliff_signal = _detect_minutes_cliff(conn, team_abbrev, deadline_date, as_of_date)
    if cliff_signal:
        result.signals.append(cliff_signal)
    
    # =========================================================================
    # Signal 9: Stealth Tank (V19.3) — Benching stars in close games
    # Catches Jazz/Wizards pattern: healthy stars benched late even in
    # competitive games. Different from minutes_cliff (total minutes drop)
    # because this focuses on close-game context.
    # =========================================================================
    stealth_signal = _detect_stealth_tank(conn, team_abbrev, deadline_date, as_of_date)
    if stealth_signal:
        result.signals.append(stealth_signal)
    
    # =========================================================================
    # Signal 10: Known Tanking Team Watchlist (V19.3) — Manual override
    # Immediate flag for teams publicly identified as tanking before
    # enough statistical data accumulates for automated detection.
    # =========================================================================
    if team_abbrev.upper() in KNOWN_TANKING_TEAMS:
        # Only add watchlist signal if no strong automated signals already exist
        existing_strong = sum(1 for s in result.signals if s.strength >= 0.6)
        if existing_strong < 2:
            reason = KNOWN_TANKING_TEAMS[team_abbrev.upper()]
            result.signals.append(TankSignal(
                signal_type="known_tanking_watchlist",
                strength=0.65,
                description=f"Known tanking team (manual watchlist)",
                evidence=reason,
            ))
    
    # =========================================================================
    # Calculate Final Tank Score
    # =========================================================================
    if result.signals:
        # Weighted average of all signals
        total_weight = sum(s.strength for s in result.signals)
        max_possible = len(result.signals)  # Each signal can have max 1.0
        
        # Normalize to 0-1 range
        raw_confidence = total_weight / max(max_possible, 1)
        
        # Boost if multiple strong signals align
        strong_signals = sum(1 for s in result.signals if s.strength >= 0.6)
        if strong_signals >= 2:
            raw_confidence = min(1.0, raw_confidence * 1.3)
        
        # V19.4: WIN% GATE — Teams above .500 cannot be flagged as high-confidence
        # tankers unless they are on the manual watchlist. This prevents flagging
        # load management or natural blowout rest as "tanking".
        # OKC, BOS, DEN, etc. with 35+ wins are NOT tanking — they're managing load.
        is_known_tanker = team_abbrev.upper() in KNOWN_TANKING_TEAMS
        
        if not is_known_tanker and result.win_pct >= 0.550:
            # Good teams: max confidence 0.20 — monitoring only, never "is_tanking"
            raw_confidence = min(raw_confidence, 0.20)
        elif not is_known_tanker and result.win_pct >= 0.500:
            # Above .500 teams: cap at 0.25 — monitoring only (below is_tanking threshold)
            raw_confidence = min(raw_confidence, 0.25)
        elif not is_known_tanker and result.win_pct >= 0.420:
            # Below .500 but not terrible: cap at 0.40 (soft tank monitoring)
            raw_confidence = min(raw_confidence, 0.40)
        elif not is_known_tanker and result.win_pct >= 0.380:
            # Below average: cap at 0.55 (possible tanking)
            raw_confidence = min(raw_confidence, 0.55)
        # Below 0.380 win% or known tankers → no cap, full detection
        
        result.confidence = min(1.0, raw_confidence)
        result.is_tanking = result.confidence >= LOW_TANK_CONFIDENCE
        
        # Calculate adjustment factors
        if result.is_tanking:
            # Star minutes should be reduced
            # High confidence tank → bigger reduction
            result.star_minutes_factor = max(0.75, 1.0 - (result.confidence * 0.25))
            
            # Overall confidence in predictions is lower
            result.overall_confidence_impact = max(0.6, 1.0 - (result.confidence * 0.3))
    
    return result


def detect_all_tanking_teams(
    conn: sqlite3.Connection,
    deadline_date: str = "2026-02-06",
    as_of_date: Optional[str] = None,
) -> List[TankDetectionResult]:
    """
    Analyze all teams for tanking behavior.
    
    Returns list of results sorted by tank confidence (highest first).
    """
    # Get all teams from the database
    teams = conn.execute("SELECT DISTINCT name FROM teams").fetchall()
    
    results = []
    seen_abbrevs: Set[str] = set()  # Avoid duplicate processing (e.g., "LA Clippers" vs "Los Angeles Clippers")
    
    for team in teams:
        abbrev = abbrev_from_team_name(team["name"])
        if abbrev and abbrev not in seen_abbrevs:
            seen_abbrevs.add(abbrev)
            result = detect_tanking(conn, abbrev, deadline_date, as_of_date)
            if result.signals:  # Only include teams with signals
                results.append(result)
    
    return sorted(results, key=lambda r: r.confidence, reverse=True)


# ============================================================================
# Internal Analysis Functions
# ============================================================================

def _get_team_record(
    conn: sqlite3.Connection,
    team_abbrev: str,
    before_date: str,
    since_date: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    """
    Get team win-loss record from boxscore data.
    
    Returns (wins, losses) tuple.
    """
    team_abbrev_upper = team_abbrev.upper()
    
    # Find team IDs that match this abbreviation
    from ..team_aliases import team_name_from_abbrev
    full_name = team_name_from_abbrev(team_abbrev_upper)
    if not full_name:
        return None
    
    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    
    if not team_row:
        return None
    
    team_id = team_row["id"]
    
    # Count wins and losses from team totals
    query = """
        SELECT 
            g.game_date,
            bt.pts as team_pts,
            obt.pts as opp_pts
        FROM boxscore_team_totals bt
        JOIN games g ON g.id = bt.game_id
        JOIN boxscore_team_totals obt ON obt.game_id = g.id AND obt.team_id != bt.team_id
        WHERE bt.team_id = ?
          AND g.game_date < ?
    """
    params: list = [team_id, before_date]
    
    if since_date:
        query += " AND g.game_date >= ?"
        params.append(since_date)
    
    rows = conn.execute(query, params).fetchall()
    
    wins = sum(1 for r in rows if r["team_pts"] and r["opp_pts"] and r["team_pts"] > r["opp_pts"])
    losses = sum(1 for r in rows if r["team_pts"] and r["opp_pts"] and r["team_pts"] < r["opp_pts"])
    
    return (wins, losses)


def _analyze_star_minutes(
    conn: sqlite3.Connection,
    team_abbrev: str,
    deadline_date: str,
    as_of_date: str,
    min_pre_minutes: float = 25.0,
) -> List[PlayerMinutesAnalysis]:
    """
    Compare star player minutes before and after deadline.
    
    Identifies players who were getting 25+ minutes pre-deadline
    and checks if their minutes have changed post-deadline.
    """
    from ..team_aliases import team_name_from_abbrev
    full_name = team_name_from_abbrev(team_abbrev.upper())
    if not full_name:
        return []
    
    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    if not team_row:
        return []
    
    team_id = team_row["id"]
    
    # Get players who averaged 25+ minutes in the 15 games before deadline
    pre_stats = conn.execute(
        """
        SELECT 
            b.player_id,
            p.name as player_name,
            AVG(b.minutes) as avg_min,
            AVG(b.pts) as avg_pts,
            COUNT(*) as games
        FROM boxscore_player b
        JOIN games g ON g.id = b.game_id
        JOIN players p ON p.id = b.player_id
        WHERE b.team_id = ?
          AND g.game_date < ?
          AND g.game_date >= date(?, '-45 days')
          AND b.minutes > 5
          AND b.status = 'starter'
        GROUP BY b.player_id
        HAVING AVG(b.minutes) >= ? AND COUNT(*) >= 5
        """,
        (team_id, deadline_date, deadline_date, min_pre_minutes),
    ).fetchall()
    
    results = []
    
    for ps in pre_stats:
        # Get post-deadline minutes
        post_stats = conn.execute(
            """
            SELECT 
                AVG(b.minutes) as avg_min,
                AVG(b.pts) as avg_pts,
                COUNT(*) as games
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.player_id = ?
              AND b.team_id = ?
              AND g.game_date >= ?
              AND g.game_date <= ?
              AND b.minutes > 0
            """,
            (ps["player_id"], team_id, deadline_date, as_of_date),
        ).fetchone()
        
        analysis = PlayerMinutesAnalysis(
            player_name=ps["player_name"],
            player_id=ps["player_id"],
            pre_avg_minutes=ps["avg_min"],
            pre_games=ps["games"],
            pre_avg_pts=ps["avg_pts"],
            is_star=ps["avg_min"] >= 28,  # 28+ min = star
        )
        
        if post_stats and post_stats["games"] and post_stats["games"] > 0:
            analysis.post_avg_minutes = post_stats["avg_min"]
            analysis.post_games = post_stats["games"]
            analysis.post_avg_pts = post_stats["avg_pts"]
            
            if analysis.pre_avg_minutes > 0:
                analysis.minutes_change_pct = (
                    (analysis.post_avg_minutes - analysis.pre_avg_minutes) 
                    / analysis.pre_avg_minutes * 100
                )
        
        results.append(analysis)
    
    return results


def _analyze_dnp_patterns(
    conn: sqlite3.Connection,
    team_abbrev: str,
    deadline_date: str,
    as_of_date: str,
) -> Optional[TankSignal]:
    """
    Check if starters are getting more DNPs (Did Not Play) after deadline.
    
    Tanking teams will rest healthy players more often.
    """
    from ..team_aliases import team_name_from_abbrev
    full_name = team_name_from_abbrev(team_abbrev.upper())
    if not full_name:
        return None
    
    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    if not team_row:
        return None
    
    team_id = team_row["id"]
    
    # Count inactive players per game before deadline (last 15 games)
    pre_inactives = conn.execute(
        """
        SELECT COUNT(*) as cnt, g.game_date
        FROM inactive_players ip
        JOIN games g ON g.id = ip.game_id
        WHERE ip.team_id = ?
          AND g.game_date < ?
          AND g.game_date >= date(?, '-30 days')
        GROUP BY g.game_date
        """,
        (team_id, deadline_date, deadline_date),
    ).fetchall()
    
    # Count after deadline
    post_inactives = conn.execute(
        """
        SELECT COUNT(*) as cnt, g.game_date
        FROM inactive_players ip
        JOIN games g ON g.id = ip.game_id
        WHERE ip.team_id = ?
          AND g.game_date >= ?
          AND g.game_date <= ?
        GROUP BY g.game_date
        """,
        (team_id, deadline_date, as_of_date),
    ).fetchall()
    
    if not pre_inactives or not post_inactives:
        return None
    
    pre_avg = statistics.mean(r["cnt"] for r in pre_inactives)
    post_avg = statistics.mean(r["cnt"] for r in post_inactives)
    
    if pre_avg > 0:
        increase_ratio = post_avg / pre_avg
        if increase_ratio >= DNP_SPIKE_THRESHOLD:
            return TankSignal(
                signal_type="dnp_increase",
                strength=min(0.7, (increase_ratio - 1) * 0.35),
                description="Significant increase in DNPs for starters",
                evidence=f"Pre-deadline avg: {pre_avg:.1f}/game → Post: {post_avg:.1f}/game ({increase_ratio:.1f}x)"
            )
    
    return None


def _analyze_trade_activity(
    conn: sqlite3.Connection,
    team_abbrev: str,
) -> Optional[TankSignal]:
    """
    Check if team traded away key players at the deadline.
    
    Selling at the deadline = strong tank signal.
    """
    # Check if trade tables exist
    try:
        trades_out = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM player_trades 
            WHERE from_team = ? AND old_team_role IN ('star', 'starter')
            """,
            (team_abbrev.upper(),),
        ).fetchone()
        
        trades_in = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM player_trades 
            WHERE to_team = ? AND expected_new_role IN ('star', 'starter')
            """,
            (team_abbrev.upper(),),
        ).fetchone()
        
        out_count = trades_out["cnt"] if trades_out else 0
        in_count = trades_in["cnt"] if trades_in else 0
        
        # Net sellers of star/starter talent = tanking signal
        if out_count > in_count and out_count >= 2:
            net = out_count - in_count
            return TankSignal(
                signal_type="trade_selling",
                strength=min(0.8, net * 0.25),
                description=f"Net seller of talent at deadline",
                evidence=f"Traded away {out_count} starters/stars, acquired {in_count}"
            )
    except sqlite3.OperationalError:
        pass  # Trade tables may not exist yet
    
    return None


def _analyze_seller_score(
    conn: sqlite3.Connection,
    team_abbrev: str,
) -> Optional[TankSignal]:
    """
    V19.1: Compute a "seller score" based on total outgoing trades.
    
    Different from _analyze_trade_activity which only looks at star/starter.
    This counts ALL trades — a team moving 4+ players is clearly selling.
    """
    try:
        all_out = conn.execute(
            "SELECT COUNT(*) as cnt FROM player_trades WHERE from_team = ?",
            (team_abbrev.upper(),),
        ).fetchone()
        
        all_in = conn.execute(
            "SELECT COUNT(*) as cnt FROM player_trades WHERE to_team = ?",
            (team_abbrev.upper(),),
        ).fetchone()
        
        out_count = all_out["cnt"] if all_out else 0
        in_count = all_in["cnt"] if all_in else 0
        
        net_out = out_count - in_count
        
        if net_out >= 3:
            return TankSignal(
                signal_type="heavy_seller",
                strength=min(0.85, 0.3 + net_out * 0.1),
                description=f"Heavy seller at deadline — {net_out} net departures",
                evidence=f"Total out: {out_count}, in: {in_count} (net: -{net_out})"
            )
        elif net_out >= 1 and out_count >= 3:
            return TankSignal(
                signal_type="moderate_seller",
                strength=0.35,
                description=f"Moderate roster churn — {out_count} total moves out",
                evidence=f"Total out: {out_count}, in: {in_count}"
            )
    except sqlite3.OperationalError:
        pass
    
    return None


def _detect_minutes_cliff(
    conn: sqlite3.Connection,
    team_abbrev: str,
    deadline_date: str,
    as_of_date: str,
) -> Optional[TankSignal]:
    """
    V19.1: Detect sudden "minutes cliff" — a star's minutes dropping >15%
    game-over-game right after the deadline.
    
    This catches the most blatant tanking signal: a team suddenly benching
    a star who was playing 32+ minutes.
    """
    from ..team_aliases import team_name_from_abbrev
    full_name = team_name_from_abbrev(team_abbrev.upper())
    if not full_name:
        return None
    
    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    if not team_row:
        return None
    
    team_id = team_row["id"]
    
    # Find players who had 28+ min pre-deadline (starters/stars)
    stars = conn.execute(
        """
        SELECT 
            b.player_id, p.name,
            AVG(b.minutes) as pre_avg_min
        FROM boxscore_player b
        JOIN players p ON p.id = b.player_id
        JOIN games g ON g.id = b.game_id
        WHERE b.team_id = ?
          AND g.game_date < ?
          AND g.game_date >= date(?, '-30 days')
          AND b.minutes > 5
        GROUP BY b.player_id
        HAVING AVG(b.minutes) >= 28 AND COUNT(*) >= 5
        """,
        (team_id, deadline_date, deadline_date),
    ).fetchall()
    
    cliff_players = []
    for star in stars:
        # V19.4 FIX: Require a SUSTAINED pattern across ALL post-deadline games
        # (not just a single game drop which could be blowout/foul trouble/load mgmt).
        # We need at least 2 post-deadline games and the AVERAGE minutes must be
        # >15% below the pre-deadline average.
        recent = conn.execute(
            """
            SELECT b.minutes, g.game_date
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            WHERE b.player_id = ?
              AND b.team_id = ?
              AND g.game_date >= ?
              AND g.game_date <= ?
              AND b.minutes > 0
            ORDER BY g.game_date DESC
            """,
            (star["player_id"], team_id, deadline_date, as_of_date),
        ).fetchall()
        
        # Need at least 2 post-deadline games to confirm sustained pattern
        if len(recent) < 2:
            continue
        
        if star["pre_avg_min"] > 0:
            avg_post_min = statistics.mean(g["minutes"] for g in recent)
            avg_drop_pct = (avg_post_min - star["pre_avg_min"]) / star["pre_avg_min"] * 100
            
            # Sustained average drop >15% (not a single-game blip)
            if avg_drop_pct < -15:
                cliff_players.append({
                    "name": star["name"],
                    "pre_avg": star["pre_avg_min"],
                    "post_avg": avg_post_min,
                    "drop_pct": avg_drop_pct,
                    "games": len(recent),
                })
    
    if cliff_players:
        names = ", ".join(p["name"] for p in cliff_players)
        worst_drop = min(p["drop_pct"] for p in cliff_players)
        total_games = max(p["games"] for p in cliff_players)
        return TankSignal(
            signal_type="minutes_cliff",
            strength=min(0.85, 0.4 + len(cliff_players) * 0.15),
            description=f"Sustained minutes cliff for star(s): {names}",
            evidence=f"Worst avg drop vs pre-deadline: {worst_drop:.1f}% (over {total_games} games)"
        )
    
    return None


def _detect_stealth_tank(
    conn: sqlite3.Connection,
    team_abbrev: str,
    deadline_date: str,
    as_of_date: str,
) -> Optional[TankSignal]:
    """
    V19.3: Detect "stealth tanking" — stars getting reduced minutes specifically
    in close games after the trade deadline.

    This differs from minutes_cliff (Signal 8) which looks at total minutes
    drops game-over-game. Stealth tanking is more subtle:
    - Overall minutes may look normal (coaches maintain averages)
    - But in close games (final margin ≤10), star minutes drop sharply
    - Jazz pattern: play stars normally through 3 quarters, bench in 4th even when close
    - Wizards pattern: rule out healthy players for "rest" or vague injuries

    We compare star minutes in close games post-deadline vs pre-deadline.
    A significant drop (>12%) in close games while non-close game minutes
    remain similar is a strong stealth-tank signal.
    """
    from ..team_aliases import team_name_from_abbrev
    full_name = team_name_from_abbrev(team_abbrev.upper())
    if not full_name:
        return None

    team_row = conn.execute(
        "SELECT id FROM teams WHERE name = ?", (full_name,)
    ).fetchone()
    if not team_row:
        return None

    team_id = team_row["id"]

    # Find star players (28+ avg min pre-deadline)
    stars = conn.execute(
        """
        SELECT 
            b.player_id, p.name,
            AVG(b.minutes) as pre_avg_min
        FROM boxscore_player b
        JOIN players p ON p.id = b.player_id
        JOIN games g ON g.id = b.game_id
        WHERE b.team_id = ?
          AND g.game_date < ?
          AND g.game_date >= date(?, '-30 days')
          AND b.minutes > 5
        GROUP BY b.player_id
        HAVING AVG(b.minutes) >= 28 AND COUNT(*) >= 5
        """,
        (team_id, deadline_date, deadline_date),
    ).fetchall()

    if not stars:
        return None

    stealth_players = []
    for star in stars:
        # Get post-deadline games that were CLOSE (margin ≤ CLOSE_GAME_MARGIN)
        # We determine closeness by the final score difference (team pts vs opponent pts)
        close_games = conn.execute(
            """
            SELECT 
                b.minutes,
                g.game_date,
                ABS(bt.pts - obt.pts) as margin
            FROM boxscore_player b
            JOIN games g ON g.id = b.game_id
            JOIN boxscore_team_totals bt ON bt.game_id = g.id AND bt.team_id = ?
            JOIN boxscore_team_totals obt ON obt.game_id = g.id AND obt.team_id != ? AND obt.team_id = (
                SELECT bt2.team_id FROM boxscore_team_totals bt2 
                WHERE bt2.game_id = g.id AND bt2.team_id != ?
                LIMIT 1
            )
            WHERE b.player_id = ?
              AND b.team_id = ?
              AND g.game_date >= ?
              AND g.game_date <= ?
              AND b.minutes > 0
              AND ABS(bt.pts - obt.pts) <= ?
            ORDER BY g.game_date DESC
            """,
            (team_id, team_id, team_id, star["player_id"], team_id,
             deadline_date, as_of_date, CLOSE_GAME_MARGIN),
        ).fetchall()

        if len(close_games) < MIN_CLOSE_GAMES_REQUIRED:
            continue

        avg_close_game_mins = statistics.mean(g["minutes"] for g in close_games)

        # Compare to pre-deadline average
        if star["pre_avg_min"] > 0:
            close_drop_pct = (avg_close_game_mins - star["pre_avg_min"]) / star["pre_avg_min"] * 100
            if close_drop_pct < STEALTH_MINUTES_DROP_PCT:
                stealth_players.append({
                    "name": star["name"],
                    "pre_avg": star["pre_avg_min"],
                    "close_game_avg": avg_close_game_mins,
                    "drop_pct": close_drop_pct,
                    "close_games": len(close_games),
                })

    if stealth_players:
        names = ", ".join(p["name"] for p in stealth_players)
        avg_drop = statistics.mean(p["drop_pct"] for p in stealth_players)
        return TankSignal(
            signal_type="stealth_tank_close_game",
            strength=min(0.85, 0.45 + len(stealth_players) * 0.15),
            description=f"Stealth tanking: stars benched in close games: {names}",
            evidence=(
                f"Post-deadline close-game minutes avg drop: {avg_drop:.1f}% "
                f"vs pre-deadline baseline ({len(stealth_players)} star(s) affected)"
            ),
        )

    return None


def detect_tanking_cached(
    conn: sqlite3.Connection,
    team_abbrev: str,
    deadline_date: str = "2026-02-06",
    as_of_date: Optional[str] = None,
) -> TankDetectionResult:
    """
    V19.3: Cached version of detect_tanking.
    
    Caches results per (as_of_date, team) to avoid repeated DB queries
    when the projection pipeline checks tanking status for every player
    on the same team on the same date.
    """
    team_abbrev = (normalize_team_abbrev(team_abbrev) or team_abbrev).upper()
    
    if as_of_date is None:
        as_of_date = datetime.now().strftime("%Y-%m-%d")
    
    cache_key = as_of_date
    if cache_key in _tank_detection_cache:
        if team_abbrev in _tank_detection_cache[cache_key]:
            return _tank_detection_cache[cache_key][team_abbrev]
    
    # Cache miss — run full detection
    result = detect_tanking(conn, team_abbrev, deadline_date, as_of_date)
    
    if cache_key not in _tank_detection_cache:
        _tank_detection_cache[cache_key] = {}
    _tank_detection_cache[cache_key][team_abbrev] = result
    
    return result


def clear_tank_detection_cache() -> None:
    """Clear the tank detection cache (e.g., between backtest dates)."""
    _tank_detection_cache.clear()


# ============================================================================
# Minutes Projection Adjustments for Tanking
# ============================================================================

def get_tank_adjusted_minutes(
    conn: sqlite3.Connection,
    player_name: str,
    team_abbrev: str,
    base_minutes: float,
    deadline_date: str = "2026-02-06",
) -> Tuple[float, List[str]]:
    """
    Adjust projected minutes for tanking effects.
    
    Returns (adjusted_minutes, [warning_messages])
    """
    warnings = []
    
    # Check if team is tanking
    result = detect_tanking(conn, team_abbrev, deadline_date)
    
    if not result.is_tanking:
        return base_minutes, warnings
    
    # Check if this specific player has minutes reduction
    player_analysis = None
    for pa in result.player_analyses:
        if pa.player_name.lower() == player_name.lower():
            player_analysis = pa
            break
    
    adjusted = base_minutes
    
    if player_analysis and player_analysis.has_significant_drop:
        # Use actual observed post-deadline minutes as the new baseline
        if player_analysis.post_games >= 3:
            adjusted = player_analysis.post_avg_minutes
            warnings.append(
                f"⚠️ TANK ALERT: {player_name} minutes dropped from "
                f"{player_analysis.pre_avg_minutes:.1f} → {player_analysis.post_avg_minutes:.1f} "
                f"({player_analysis.minutes_change_pct:+.1f}%)"
            )
        else:
            # Not enough post-deadline data, apply estimated reduction
            adjusted = base_minutes * result.star_minutes_factor
            warnings.append(
                f"⚠️ TANK WARNING: {team_abbrev} may be tanking "
                f"(confidence: {result.confidence:.0%}). Minutes reduced."
            )
    elif result.confidence >= MODERATE_TANK_CONFIDENCE:
        # Team is tanking but this player hasn't shown reduction yet
        adjusted = base_minutes * result.star_minutes_factor
        warnings.append(
            f"⚠️ TANK RISK: {team_abbrev} showing tanking signals "
            f"({result.confidence:.0%} confidence). Star minutes may decrease."
        )
    
    return adjusted, warnings
