#!/usr/bin/env python3
"""
Comprehensive Model Analysis & Backtest Suite
==============================================

This script performs thorough backtesting on all 19+ NBA Props prediction models,
providing detailed analysis of:
- Overall accuracy and hit rates
- Picks-to-accuracy ratio (quality scoring)
- Strengths and weaknesses of each model
- Direction (OVER/UNDER) performance
- Prop type (PTS/REB/AST) performance
- Confidence tier performance

Features:
- Progress bars for visual feedback
- Handles single-file, multi-file, and specialized models
- Comprehensive quality scoring system
- Detailed strength/weakness identification
- Model ranking by multiple metrics

Usage:
    python run_comprehensive_model_analysis.py
    python run_comprehensive_model_analysis.py --start 2025-12-01 --end 2026-02-03

Author: PropAI Analysis Suite
Created: February 2026
"""
from __future__ import annotations

import sys
import time
import traceback
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from pathlib import Path
import importlib
import statistics

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# CONSOLE UTILITIES
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str, char: str = "═") -> None:
    """Print a styled header."""
    width = 80
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * width}{Colors.ENDC}")
    print()


def print_subheader(text: str) -> None:
    """Print a styled subheader."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}  ▶ {text}{Colors.ENDC}")
    print(f"  {'-' * 60}")


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 40,
    fill: str = "█",
    empty: str = "░"
) -> None:
    """Print a progress bar to the terminal."""
    if total <= 0:
        return
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    
    # Color based on progress
    if iteration == total:
        color = Colors.GREEN
    elif iteration > total * 0.5:
        color = Colors.CYAN
    else:
        color = Colors.YELLOW
    
    sys.stdout.write(f'\r  {prefix} {color}|{bar}|{Colors.ENDC} {percent}% {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        print()


def print_model_progress(current: int, total: int, model_name: str) -> None:
    """Print progress for model testing."""
    bar_length = 30
    filled = int(bar_length * current / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = (current / total) * 100
    
    sys.stdout.write(f'\r  Overall Progress: |{bar}| {percent:.1f}% - Testing: {model_name:<30}')
    sys.stdout.flush()


# ============================================================================
# MODEL INFORMATION
# ============================================================================

@dataclass
class ModelDefinition:
    """Definition of a model for testing."""
    model_id: str
    display_name: str
    version: str
    module_path: str
    backtest_func: str
    config_class: Optional[str] = None
    is_active: bool = True
    description: str = ""
    expected_runtime: float = 5.0  # minutes


# Complete list of all models to test
ALL_MODELS: List[ModelDefinition] = [
    # Multi-file models (v12-v19) - newest first
    ModelDefinition("v19_general", "Model V19 General", "19.0", 
                   "src.nba_props.engine.model_v19_general", 
                   "run_backtest_v19_general", "ModelConfigV19General",
                   description="Multi-factor with strict alignment requirements"),
    
    ModelDefinition("v19_under", "Model V19 Under", "19.0",
                   "src.nba_props.engine.model_v19_under",
                   "run_backtest_v19_under", "ModelConfigV19Under",
                   description="Specialized UNDER with multi-factor requirements"),
    
    ModelDefinition("v18_general", "Model V18 General", "18.0",
                   "src.nba_props.engine.model_v18_general",
                   "run_backtest_v18_general", "ModelConfigV18General",
                   description="Holistic multi-factor with box score analysis"),
    
    ModelDefinition("v18_under", "Model V18 Under", "18.5",
                   "src.nba_props.engine.model_v18_under",
                   "run_backtest_v18_under", "ModelConfigV18Under",
                   description="Specialized UNDER with validated factors"),
    
    ModelDefinition("v17_general", "Model V17 General", "17.0",
                   "src.nba_props.engine.model_v17_general",
                   "run_backtest_v17_general", "ModelConfigV17General",
                   description="Holistic multi-factor with strategic direction"),
    
    ModelDefinition("v17_under", "Model V17 Under", "17.0",
                   "src.nba_props.engine.model_v17_under",
                   "run_backtest_v17_under", "ModelConfigV17Under",
                   description="V17 specialized UNDER model"),
    
    ModelDefinition("v16_general", "Model V16 General", "16.0",
                   "src.nba_props.engine.model_v16_general",
                   "run_backtest_v16_general", "ModelConfigV16General",
                   description="Pattern-based with hybrid line approach"),
    
    ModelDefinition("v16_under", "Model V16 Under", "16.5",
                   "src.nba_props.engine.model_v16_under",
                   "run_backtest_v16_under", "ModelConfigV16Under",
                   description="Specialized UNDER with defense integration"),
    
    ModelDefinition("v15_general", "Model V15 General", "15.0",
                   "src.nba_props.engine.model_v15_general",
                   "run_backtest_v15_general", "ModelConfigV15General",
                   description="Derived line fallacy fix"),
    
    ModelDefinition("v14_general", "Model V14 General", "14.0",
                   "src.nba_props.engine.model_v14_general",
                   "run_backtest_v14_general", "ModelConfigV14General",
                   description="Market-aware with hybrid line handling"),
    
    ModelDefinition("v13_general", "Model V13 General", "13.0",
                   "src.nba_props.engine.model_v13_general",
                   "run_backtest_v13_general", "ModelConfigV13General",
                   description="Direction preferences model"),
    
    ModelDefinition("v12_general", "Model V12 General", "12.0",
                   "src.nba_props.engine.model_v12_general",
                   "run_backtest_general", "GeneralModelConfig",
                   description="Pattern-based with sportsbook integration"),
    
    ModelDefinition("v12_combined", "Model V12 Combined", "12.0",
                   "src.nba_props.engine.model_v12_combined",
                   "run_backtest_v12", None,
                   description="Combined general + under model"),
    
    # Single-file models (v2-v10)
    ModelDefinition("v10", "Model V10", "10.0",
                   "src.nba_props.engine.model_v10",
                   "run_backtest_v10", "ModelConfigV10",
                   description="Market-aware requiring sportsbook lines"),
    
    ModelDefinition("v9", "Model V9", "9.0",
                   "src.nba_props.engine.model_v9",
                   "run_backtest_v9", "ModelConfigV9",
                   description="Line-aware predictions"),
    
    ModelDefinition("v8", "Model V8", "8.0",
                   "src.nba_props.engine.model_v8",
                   "run_backtest", "ModelV8Config",
                   description="Pattern detection with learning"),
    
    ModelDefinition("v5", "Model V5", "5.0",
                   "src.nba_props.engine.model_v5",
                   "run_full_backtest", "ModelV5Config",
                   description="Comprehensive data utilization"),
    
    ModelDefinition("v4", "Model V4", "4.0",
                   "src.nba_props.engine.model_v4",
                   "run_full_backtest", "ModelV4Config",
                   description="Balanced prop type distribution"),
    
    ModelDefinition("v3", "Model V3", "3.0",
                   "src.nba_props.engine.model_v3",
                   "run_backtest_v3", "ModelV3Config",
                   description="Stat-type specific weights"),
    
    ModelDefinition("v2", "Model V2", "2.0",
                   "src.nba_props.engine.model_v2",
                   "run_backtest", "ModelV2Config",
                   description="First generation OVER-focused"),
    
    # Specialized models
    ModelDefinition("production", "Model Production", "1.0",
                   "src.nba_props.engine.model_production",
                   "run_backtest", "ModelConfig",
                   description="Production deployment model"),
    
    ModelDefinition("final", "Model Final", "1.0",
                   "src.nba_props.engine.model_final",
                   "run_full_backtest", "FinalModelConfig",
                   description="Final consolidated model"),
    
    ModelDefinition("under_v2", "Under Model V2", "2.0",
                   "src.nba_props.engine.under_model_v2",
                   "backtest_under_model_v2", None,
                   description="Specialized UNDER predictions"),
]


# ============================================================================
# RESULT STORAGE
# ============================================================================

@dataclass
class ModelResult:
    """Results from testing a single model."""
    model_id: str
    model_name: str
    version: str
    
    # Test metadata
    start_date: str = ""
    end_date: str = ""
    days_tested: int = 0
    runtime_seconds: float = 0.0
    success: bool = False
    error_message: str = ""
    
    # Core metrics
    total_picks: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    
    # Direction breakdown
    over_picks: int = 0
    over_hits: int = 0
    over_rate: float = 0.0
    
    under_picks: int = 0
    under_hits: int = 0
    under_rate: float = 0.0
    
    # Prop type breakdown
    pts_picks: int = 0
    pts_hits: int = 0
    pts_rate: float = 0.0
    
    reb_picks: int = 0
    reb_hits: int = 0
    reb_rate: float = 0.0
    
    ast_picks: int = 0
    ast_hits: int = 0
    ast_rate: float = 0.0
    
    # Confidence tiers (if available)
    premium_picks: int = 0
    premium_hits: int = 0
    premium_rate: float = 0.0
    
    high_picks: int = 0
    high_hits: int = 0
    high_rate: float = 0.0
    
    # Line source (if tracked)
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    sportsbook_rate: float = 0.0
    
    derived_picks: int = 0
    derived_hits: int = 0
    derived_rate: float = 0.0
    
    # Quality metrics
    picks_per_day: float = 0.0
    quality_score: float = 0.0
    volume_score: float = 0.0
    consistency_score: float = 0.0
    
    # Analysis
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # Raw result object
    raw_result: Any = None
    
    def calculate_metrics(self) -> None:
        """Calculate all derived metrics."""
        # Hit rate
        self.hit_rate = self.total_hits / self.total_picks if self.total_picks > 0 else 0.0
        
        # Direction rates
        self.over_rate = self.over_hits / self.over_picks if self.over_picks > 0 else 0.0
        self.under_rate = self.under_hits / self.under_picks if self.under_picks > 0 else 0.0
        
        # Prop type rates
        self.pts_rate = self.pts_hits / self.pts_picks if self.pts_picks > 0 else 0.0
        self.reb_rate = self.reb_hits / self.reb_picks if self.reb_picks > 0 else 0.0
        self.ast_rate = self.ast_hits / self.ast_picks if self.ast_picks > 0 else 0.0
        
        # Confidence rates
        self.premium_rate = self.premium_hits / self.premium_picks if self.premium_picks > 0 else 0.0
        self.high_rate = self.high_hits / self.high_picks if self.high_picks > 0 else 0.0
        
        # Line source rates
        self.sportsbook_rate = self.sportsbook_hits / self.sportsbook_picks if self.sportsbook_picks > 0 else 0.0
        self.derived_rate = self.derived_hits / self.derived_picks if self.derived_picks > 0 else 0.0
        
        # Picks per day
        self.picks_per_day = self.total_picks / self.days_tested if self.days_tested > 0 else 0.0
        
        # Quality score
        self._calculate_quality_score()
        
        # Identify strengths/weaknesses
        self._identify_strengths_weaknesses()
    
    def _calculate_quality_score(self) -> None:
        """Calculate comprehensive quality score."""
        if self.total_picks == 0:
            self.quality_score = 0.0
            return
        
        # Accuracy score (50% = 0, 65% = 100)
        accuracy_score = max(0, (self.hit_rate - 0.50) / 0.15 * 100)
        
        # Volume score (optimal 5-15 picks/day)
        if self.picks_per_day < 2:
            self.volume_score = max(0, 20 * self.picks_per_day)
        elif self.picks_per_day > 30:
            self.volume_score = max(0, 100 - (self.picks_per_day - 30) * 3)
        else:
            optimal_ppd = 10.0
            ppd_deviation = abs(self.picks_per_day - optimal_ppd)
            self.volume_score = max(0, 100 - ppd_deviation * 5)
        
        # Consistency score
        rates = []
        if self.pts_picks >= 10:
            rates.append(self.pts_rate)
        if self.reb_picks >= 10:
            rates.append(self.reb_rate)
        if self.ast_picks >= 10:
            rates.append(self.ast_rate)
        
        if len(rates) >= 2:
            rate_std = statistics.stdev(rates)
            self.consistency_score = max(0, 100 - rate_std * 500)
        else:
            self.consistency_score = 70
        
        # Combined: accuracy 60%, volume 25%, consistency 15%
        self.quality_score = (
            accuracy_score * 0.60 +
            self.volume_score * 0.25 +
            self.consistency_score * 0.15
        )
    
    def _identify_strengths_weaknesses(self) -> None:
        """Identify model strengths and weaknesses."""
        self.strengths = []
        self.weaknesses = []
        
        # Overall accuracy
        if self.hit_rate >= 0.60:
            self.strengths.append(f"Excellent hit rate: {self.hit_rate*100:.1f}%")
        elif self.hit_rate >= 0.55:
            self.strengths.append(f"Solid hit rate: {self.hit_rate*100:.1f}%")
        elif self.hit_rate < 0.50:
            self.weaknesses.append(f"Below breakeven: {self.hit_rate*100:.1f}%")
        elif self.hit_rate < 0.52:
            self.weaknesses.append(f"Near breakeven: {self.hit_rate*100:.1f}%")
        
        # Direction analysis
        if self.over_picks >= 10:
            if self.over_rate >= 0.58:
                self.strengths.append(f"Strong OVER: {self.over_rate*100:.1f}%")
            elif self.over_rate < 0.48:
                self.weaknesses.append(f"Weak OVER: {self.over_rate*100:.1f}%")
        
        if self.under_picks >= 10:
            if self.under_rate >= 0.58:
                self.strengths.append(f"Strong UNDER: {self.under_rate*100:.1f}%")
            elif self.under_rate < 0.48:
                self.weaknesses.append(f"Weak UNDER: {self.under_rate*100:.1f}%")
        
        # Prop type analysis
        if self.pts_picks >= 10:
            if self.pts_rate >= 0.58:
                self.strengths.append(f"Strong PTS: {self.pts_rate*100:.1f}%")
            elif self.pts_rate < 0.48:
                self.weaknesses.append(f"Weak PTS: {self.pts_rate*100:.1f}%")
        
        if self.reb_picks >= 10:
            if self.reb_rate >= 0.58:
                self.strengths.append(f"Strong REB: {self.reb_rate*100:.1f}%")
            elif self.reb_rate < 0.48:
                self.weaknesses.append(f"Weak REB: {self.reb_rate*100:.1f}%")
        
        if self.ast_picks >= 10:
            if self.ast_rate >= 0.58:
                self.strengths.append(f"Strong AST: {self.ast_rate*100:.1f}%")
            elif self.ast_rate < 0.50:
                self.weaknesses.append(f"Volatile AST: {self.ast_rate*100:.1f}%")
        
        # Confidence tier analysis
        if self.premium_picks >= 5:
            if self.premium_rate >= 0.65:
                self.strengths.append(f"Premium tier delivers: {self.premium_rate*100:.1f}%")
            elif self.premium_rate < 0.55:
                self.weaknesses.append(f"Premium tier underperforms: {self.premium_rate*100:.1f}%")
        
        if self.high_picks >= 10:
            if self.high_rate >= 0.60:
                self.strengths.append(f"High confidence reliable: {self.high_rate*100:.1f}%")
            elif self.high_rate < 0.52:
                self.weaknesses.append(f"High confidence unreliable: {self.high_rate*100:.1f}%")
        
        # Volume analysis
        if self.picks_per_day < 2 and self.days_tested >= 10:
            self.weaknesses.append(f"Very low volume: {self.picks_per_day:.1f}/day")
        elif self.picks_per_day > 30:
            self.weaknesses.append(f"Excessive picks: {self.picks_per_day:.1f}/day")
        elif 5 <= self.picks_per_day <= 15:
            self.strengths.append(f"Good volume: {self.picks_per_day:.1f}/day")


# ============================================================================
# BACKTEST RESULT EXTRACTOR
# ============================================================================

def extract_result_metrics(raw_result: Any, result: ModelResult) -> None:
    """
    Extract metrics from a model's raw backtest result.
    Different models have different result structures, so we try multiple approaches.
    """
    if raw_result is None:
        return
    
    # Try to extract total picks/hits from various attribute names
    for attr in ['total_picks', 'picks_count', 'num_picks', 'total']:
        if hasattr(raw_result, attr):
            result.total_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['total_hits', 'hits', 'correct', 'wins']:
        if hasattr(raw_result, attr):
            result.total_hits = getattr(raw_result, attr) or 0
            break
    
    # If no hits found, try to calculate from hit_rate
    if result.total_hits == 0 and result.total_picks > 0:
        for attr in ['hit_rate', 'accuracy', 'win_rate']:
            if hasattr(raw_result, attr):
                rate = getattr(raw_result, attr) or 0
                # Handle both 0.XX and XX% formats
                if rate > 1:
                    rate = rate / 100
                result.total_hits = int(result.total_picks * rate)
                break
    
    # Direction breakdown
    for attr in ['over_picks', 'over_count', 'total_over', 'num_over']:
        if hasattr(raw_result, attr):
            result.over_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['over_hits', 'over_wins', 'over_correct']:
        if hasattr(raw_result, attr):
            result.over_hits = getattr(raw_result, attr) or 0
            break
    
    for attr in ['under_picks', 'under_count', 'total_under', 'num_under']:
        if hasattr(raw_result, attr):
            result.under_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['under_hits', 'under_wins', 'under_correct']:
        if hasattr(raw_result, attr):
            result.under_hits = getattr(raw_result, attr) or 0
            break
    
    # Prop type breakdown
    for attr in ['pts_picks', 'pts_count', 'pts_total']:
        if hasattr(raw_result, attr):
            result.pts_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['pts_hits', 'pts_wins', 'pts_correct']:
        if hasattr(raw_result, attr):
            result.pts_hits = getattr(raw_result, attr) or 0
            break
    
    for attr in ['reb_picks', 'reb_count', 'reb_total']:
        if hasattr(raw_result, attr):
            result.reb_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['reb_hits', 'reb_wins', 'reb_correct']:
        if hasattr(raw_result, attr):
            result.reb_hits = getattr(raw_result, attr) or 0
            break
    
    for attr in ['ast_picks', 'ast_count', 'ast_total']:
        if hasattr(raw_result, attr):
            result.ast_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['ast_hits', 'ast_wins', 'ast_correct']:
        if hasattr(raw_result, attr):
            result.ast_hits = getattr(raw_result, attr) or 0
            break
    
    # Confidence tiers
    for attr in ['premium_picks', 'premium_count', 'premium_total']:
        if hasattr(raw_result, attr):
            result.premium_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['premium_hits', 'premium_wins', 'premium_correct']:
        if hasattr(raw_result, attr):
            result.premium_hits = getattr(raw_result, attr) or 0
            break
    
    for attr in ['high_picks', 'high_count', 'high_total', 'high_confidence_total', 'high_conf_total']:
        if hasattr(raw_result, attr):
            result.high_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['high_hits', 'high_wins', 'high_correct', 'high_confidence_hits', 'high_conf_hits']:
        if hasattr(raw_result, attr):
            result.high_hits = getattr(raw_result, attr) or 0
            break
    
    # Line source tracking
    for attr in ['sportsbook_picks', 'sportsbook_count', 'sb_picks']:
        if hasattr(raw_result, attr):
            result.sportsbook_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['sportsbook_hits', 'sportsbook_wins', 'sb_hits']:
        if hasattr(raw_result, attr):
            result.sportsbook_hits = getattr(raw_result, attr) or 0
            break
    
    for attr in ['derived_picks', 'derived_count']:
        if hasattr(raw_result, attr):
            result.derived_picks = getattr(raw_result, attr) or 0
            break
    
    for attr in ['derived_hits', 'derived_wins']:
        if hasattr(raw_result, attr):
            result.derived_hits = getattr(raw_result, attr) or 0
            break
    
    # Try to get from by_direction dict
    if hasattr(raw_result, 'by_direction'):
        bd = raw_result.by_direction
        if isinstance(bd, dict):
            if 'OVER' in bd:
                o = bd['OVER']
                result.over_picks = o.get('total', o.get('picks', 0))
                result.over_hits = o.get('hits', o.get('wins', 0))
            if 'UNDER' in bd:
                u = bd['UNDER']
                result.under_picks = u.get('total', u.get('picks', 0))
                result.under_hits = u.get('hits', u.get('wins', 0))
    
    # Try to get from by_prop_type dict
    if hasattr(raw_result, 'by_prop_type'):
        bp = raw_result.by_prop_type
        if isinstance(bp, dict):
            for pt, data in bp.items():
                if isinstance(data, dict):
                    total = data.get('total', data.get('picks', 0))
                    hits = data.get('hits', data.get('wins', 0))
                    if pt.upper() == 'PTS':
                        result.pts_picks = total
                        result.pts_hits = hits
                    elif pt.upper() == 'REB':
                        result.reb_picks = total
                        result.reb_hits = hits
                    elif pt.upper() == 'AST':
                        result.ast_picks = total
                        result.ast_hits = hits


# ============================================================================
# MODEL TESTING
# ============================================================================

def test_model(
    model_def: ModelDefinition,
    start_date: str,
    end_date: str,
    verbose: bool = False
) -> ModelResult:
    """
    Test a single model and return results.
    
    Args:
        model_def: Model definition
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        verbose: Show verbose output
    
    Returns:
        ModelResult with test results
    """
    result = ModelResult(
        model_id=model_def.model_id,
        model_name=model_def.display_name,
        version=model_def.version,
        start_date=start_date,
        end_date=end_date,
    )
    
    # Calculate days tested
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    result.days_tested = (end_dt - start_dt).days + 1
    
    start_time = time.time()
    
    try:
        # Import the module
        module = importlib.import_module(model_def.module_path)
        
        # Get the backtest function
        backtest_func = getattr(module, model_def.backtest_func)
        
        # Get config if available
        config = None
        if model_def.config_class:
            try:
                config_class = getattr(module, model_def.config_class)
                config = config_class()
            except:
                pass
        
        # Run backtest - handle different function signatures
        if config is not None:
            # Try with config parameter
            try:
                raw_result = backtest_func(start_date, end_date, config=config, verbose=verbose)
            except TypeError:
                # Try without config
                try:
                    raw_result = backtest_func(start_date, end_date, verbose=verbose)
                except TypeError:
                    raw_result = backtest_func(start_date, end_date)
        else:
            try:
                raw_result = backtest_func(start_date, end_date, verbose=verbose)
            except TypeError:
                raw_result = backtest_func(start_date, end_date)
        
        result.raw_result = raw_result
        result.success = True
        
        # Extract metrics
        extract_result_metrics(raw_result, result)
        
    except Exception as e:
        result.success = False
        result.error_message = str(e)
        if verbose:
            traceback.print_exc()
    
    result.runtime_seconds = time.time() - start_time
    
    # Calculate derived metrics
    if result.success:
        result.calculate_metrics()
    
    return result


def run_all_tests(
    start_date: str,
    end_date: str,
    models: Optional[List[ModelDefinition]] = None,
    verbose: bool = False
) -> List[ModelResult]:
    """
    Run backtests on all models.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        models: List of models to test (default: ALL_MODELS)
        verbose: Show verbose output
    
    Returns:
        List of ModelResult objects
    """
    if models is None:
        models = ALL_MODELS
    
    results = []
    total_models = len(models)
    
    print_header(f"NBA Props Model Analysis Suite - Testing {total_models} Models")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for i, model_def in enumerate(models, 1):
        # Update overall progress
        print_model_progress(i, total_models, model_def.display_name)
        
        # Test the model
        result = test_model(model_def, start_date, end_date, verbose=verbose)
        results.append(result)
        
        # Brief status update
        if result.success:
            status = f"{Colors.GREEN}✓{Colors.ENDC} {result.hit_rate*100:.1f}% ({result.total_picks} picks)"
        else:
            status = f"{Colors.RED}✗ Error{Colors.ENDC}"
        
        # Clear line and print status
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        print(f"  [{i:2}/{total_models}] {model_def.display_name:<25} {status}")
    
    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_ranking_table(results: List[ModelResult]) -> str:
    """Generate a ranking table sorted by quality score."""
    # Filter successful results
    valid = [r for r in results if r.success and r.total_picks > 0]
    
    # Sort by quality score descending
    valid.sort(key=lambda x: x.quality_score, reverse=True)
    
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}MODEL RANKINGS BY QUALITY SCORE{Colors.ENDC}")
    lines.append("═" * 100)
    lines.append(f"{'Rank':<5} {'Model':<28} {'Hit Rate':>10} {'Picks':>7} {'PPD':>7} {'Quality':>10} {'Grade':>8}")
    lines.append("─" * 100)
    
    for i, r in enumerate(valid, 1):
        # Grade based on quality score
        if r.quality_score >= 80:
            grade = f"{Colors.GREEN}A+{Colors.ENDC}"
        elif r.quality_score >= 65:
            grade = f"{Colors.GREEN}A{Colors.ENDC}"
        elif r.quality_score >= 50:
            grade = f"{Colors.CYAN}B{Colors.ENDC}"
        elif r.quality_score >= 35:
            grade = f"{Colors.YELLOW}C{Colors.ENDC}"
        elif r.quality_score >= 20:
            grade = f"{Colors.YELLOW}D{Colors.ENDC}"
        else:
            grade = f"{Colors.RED}F{Colors.ENDC}"
        
        # Color code hit rate
        if r.hit_rate >= 0.58:
            hr_color = Colors.GREEN
        elif r.hit_rate >= 0.52:
            hr_color = Colors.CYAN
        elif r.hit_rate >= 0.48:
            hr_color = Colors.YELLOW
        else:
            hr_color = Colors.RED
        
        lines.append(
            f"{i:<5} {r.model_name:<28} {hr_color}{r.hit_rate*100:>9.1f}%{Colors.ENDC} "
            f"{r.total_picks:>7} {r.picks_per_day:>6.1f} {r.quality_score:>9.1f} {grade:>8}"
        )
    
    lines.append("═" * 100)
    
    return "\n".join(lines)


def generate_direction_analysis(results: List[ModelResult]) -> str:
    """Generate direction (OVER/UNDER) analysis."""
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}DIRECTION ANALYSIS (OVER vs UNDER){Colors.ENDC}")
    lines.append("═" * 90)
    lines.append(f"{'Model':<28} {'OVER Rate':>12} {'OVER Picks':>12} {'UNDER Rate':>12} {'UNDER Picks':>12}")
    lines.append("─" * 90)
    
    for r in results:
        if not r.success or r.total_picks == 0:
            continue
        
        over_rate_str = f"{r.over_rate*100:.1f}%" if r.over_picks > 0 else "-"
        under_rate_str = f"{r.under_rate*100:.1f}%" if r.under_picks > 0 else "-"
        
        lines.append(
            f"{r.model_name:<28} {over_rate_str:>12} {r.over_picks:>12} {under_rate_str:>12} {r.under_picks:>12}"
        )
    
    lines.append("═" * 90)
    
    return "\n".join(lines)


def generate_prop_type_analysis(results: List[ModelResult]) -> str:
    """Generate prop type (PTS/REB/AST) analysis."""
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}PROP TYPE ANALYSIS (PTS/REB/AST){Colors.ENDC}")
    lines.append("═" * 100)
    lines.append(f"{'Model':<28} {'PTS Rate':>10} {'PTS #':>7} {'REB Rate':>10} {'REB #':>7} {'AST Rate':>10} {'AST #':>7}")
    lines.append("─" * 100)
    
    for r in results:
        if not r.success or r.total_picks == 0:
            continue
        
        pts_rate = f"{r.pts_rate*100:.1f}%" if r.pts_picks > 0 else "-"
        reb_rate = f"{r.reb_rate*100:.1f}%" if r.reb_picks > 0 else "-"
        ast_rate = f"{r.ast_rate*100:.1f}%" if r.ast_picks > 0 else "-"
        
        lines.append(
            f"{r.model_name:<28} {pts_rate:>10} {r.pts_picks:>7} "
            f"{reb_rate:>10} {r.reb_picks:>7} {ast_rate:>10} {r.ast_picks:>7}"
        )
    
    lines.append("═" * 100)
    
    return "\n".join(lines)


def generate_strengths_weaknesses(results: List[ModelResult]) -> str:
    """Generate detailed strength/weakness analysis for each model."""
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}DETAILED STRENGTH/WEAKNESS ANALYSIS{Colors.ENDC}")
    lines.append("═" * 80)
    
    for r in results:
        if not r.success:
            lines.append(f"\n{Colors.BOLD}{r.model_name}{Colors.ENDC} - {Colors.RED}ERROR: {r.error_message}{Colors.ENDC}")
            continue
        
        if r.total_picks == 0:
            lines.append(f"\n{Colors.BOLD}{r.model_name}{Colors.ENDC} - {Colors.YELLOW}No picks generated{Colors.ENDC}")
            continue
        
        lines.append(f"\n{Colors.BOLD}{r.model_name}{Colors.ENDC} (v{r.version})")
        lines.append(f"  Overall: {r.hit_rate*100:.1f}% ({r.total_hits}/{r.total_picks}) | Quality: {r.quality_score:.1f}")
        
        if r.strengths:
            lines.append(f"  {Colors.GREEN}Strengths:{Colors.ENDC}")
            for s in r.strengths:
                lines.append(f"    ✓ {s}")
        
        if r.weaknesses:
            lines.append(f"  {Colors.RED}Weaknesses:{Colors.ENDC}")
            for w in r.weaknesses:
                lines.append(f"    ✗ {w}")
        
        if not r.strengths and not r.weaknesses:
            lines.append(f"  {Colors.YELLOW}(No significant strengths or weaknesses identified){Colors.ENDC}")
    
    lines.append("\n" + "═" * 80)
    
    return "\n".join(lines)


def generate_summary_statistics(results: List[ModelResult]) -> str:
    """Generate overall summary statistics."""
    valid = [r for r in results if r.success and r.total_picks > 0]
    failed = [r for r in results if not r.success]
    no_picks = [r for r in results if r.success and r.total_picks == 0]
    
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}SUMMARY STATISTICS{Colors.ENDC}")
    lines.append("═" * 60)
    
    lines.append(f"  Models Tested:      {len(results)}")
    lines.append(f"  Successful:         {len(valid)}")
    lines.append(f"  Failed:             {len(failed)}")
    lines.append(f"  No Picks Generated: {len(no_picks)}")
    
    if valid:
        hit_rates = [r.hit_rate for r in valid]
        quality_scores = [r.quality_score for r in valid]
        ppd_values = [r.picks_per_day for r in valid]
        
        lines.append("")
        lines.append(f"  {Colors.CYAN}Hit Rate Statistics:{Colors.ENDC}")
        lines.append(f"    Best:    {max(hit_rates)*100:.1f}%")
        lines.append(f"    Worst:   {min(hit_rates)*100:.1f}%")
        lines.append(f"    Average: {statistics.mean(hit_rates)*100:.1f}%")
        lines.append(f"    Median:  {statistics.median(hit_rates)*100:.1f}%")
        
        lines.append("")
        lines.append(f"  {Colors.CYAN}Quality Score Statistics:{Colors.ENDC}")
        lines.append(f"    Best:    {max(quality_scores):.1f}")
        lines.append(f"    Worst:   {min(quality_scores):.1f}")
        lines.append(f"    Average: {statistics.mean(quality_scores):.1f}")
        
        lines.append("")
        lines.append(f"  {Colors.CYAN}Picks Per Day Statistics:{Colors.ENDC}")
        lines.append(f"    Highest: {max(ppd_values):.1f}")
        lines.append(f"    Lowest:  {min(ppd_values):.1f}")
        lines.append(f"    Average: {statistics.mean(ppd_values):.1f}")
    
    # Identify top 3 models
    top3 = sorted(valid, key=lambda x: x.quality_score, reverse=True)[:3]
    if top3:
        lines.append("")
        lines.append(f"  {Colors.GREEN}TOP 3 MODELS:{Colors.ENDC}")
        for i, r in enumerate(top3, 1):
            lines.append(f"    {i}. {r.model_name}: {r.hit_rate*100:.1f}% ({r.total_picks} picks, QS: {r.quality_score:.1f})")
    
    # Identify bottom 3 models  
    bottom3 = sorted(valid, key=lambda x: x.quality_score)[:3]
    if bottom3:
        lines.append("")
        lines.append(f"  {Colors.RED}BOTTOM 3 MODELS:{Colors.ENDC}")
        for i, r in enumerate(bottom3, 1):
            lines.append(f"    {i}. {r.model_name}: {r.hit_rate*100:.1f}% ({r.total_picks} picks, QS: {r.quality_score:.1f})")
    
    lines.append("═" * 60)
    
    return "\n".join(lines)


def generate_recommendations(results: List[ModelResult]) -> str:
    """Generate recommendations based on analysis."""
    valid = [r for r in results if r.success and r.total_picks > 0]
    
    lines = []
    lines.append("")
    lines.append(f"{Colors.BOLD}RECOMMENDATIONS{Colors.ENDC}")
    lines.append("═" * 70)
    
    # Find best OVER model
    over_models = [r for r in valid if r.over_picks >= 20 and r.over_rate > 0]
    if over_models:
        best_over = max(over_models, key=lambda x: x.over_rate)
        lines.append(f"\n  {Colors.CYAN}Best for OVER picks:{Colors.ENDC}")
        lines.append(f"    {best_over.model_name}: {best_over.over_rate*100:.1f}% ({best_over.over_picks} picks)")
    
    # Find best UNDER model
    under_models = [r for r in valid if r.under_picks >= 20 and r.under_rate > 0]
    if under_models:
        best_under = max(under_models, key=lambda x: x.under_rate)
        lines.append(f"\n  {Colors.CYAN}Best for UNDER picks:{Colors.ENDC}")
        lines.append(f"    {best_under.model_name}: {best_under.under_rate*100:.1f}% ({best_under.under_picks} picks)")
    
    # Find best PTS model
    pts_models = [r for r in valid if r.pts_picks >= 20 and r.pts_rate > 0]
    if pts_models:
        best_pts = max(pts_models, key=lambda x: x.pts_rate)
        lines.append(f"\n  {Colors.CYAN}Best for PTS props:{Colors.ENDC}")
        lines.append(f"    {best_pts.model_name}: {best_pts.pts_rate*100:.1f}% ({best_pts.pts_picks} picks)")
    
    # Find best REB model
    reb_models = [r for r in valid if r.reb_picks >= 20 and r.reb_rate > 0]
    if reb_models:
        best_reb = max(reb_models, key=lambda x: x.reb_rate)
        lines.append(f"\n  {Colors.CYAN}Best for REB props:{Colors.ENDC}")
        lines.append(f"    {best_reb.model_name}: {best_reb.reb_rate*100:.1f}% ({best_reb.reb_picks} picks)")
    
    # Find best balanced model (quality score)
    if valid:
        best_quality = max(valid, key=lambda x: x.quality_score)
        lines.append(f"\n  {Colors.GREEN}Best Overall (Quality Score):{Colors.ENDC}")
        lines.append(f"    {best_quality.model_name}: {best_quality.hit_rate*100:.1f}% | QS: {best_quality.quality_score:.1f}")
    
    # Find best high-volume model (>10 ppd with >55% hit rate)
    high_volume = [r for r in valid if r.picks_per_day >= 10 and r.hit_rate >= 0.55]
    if high_volume:
        best_hv = max(high_volume, key=lambda x: x.hit_rate)
        lines.append(f"\n  {Colors.CYAN}Best High-Volume Model:{Colors.ENDC}")
        lines.append(f"    {best_hv.model_name}: {best_hv.hit_rate*100:.1f}% @ {best_hv.picks_per_day:.1f} picks/day")
    
    # Models to avoid
    avoid = [r for r in valid if r.hit_rate < 0.50]
    if avoid:
        lines.append(f"\n  {Colors.RED}Models to Avoid (below 50%):{Colors.ENDC}")
        for r in avoid:
            lines.append(f"    • {r.model_name}: {r.hit_rate*100:.1f}%")
    
    lines.append("\n" + "═" * 70)
    
    return "\n".join(lines)


def generate_full_report(results: List[ModelResult], start_date: str, end_date: str) -> str:
    """Generate the complete analysis report."""
    report_parts = []
    
    # Header
    report_parts.append("\n")
    report_parts.append(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.ENDC}")
    report_parts.append(f"{Colors.BOLD}{Colors.CYAN}  NBA PROPS MODEL COMPREHENSIVE ANALYSIS REPORT{Colors.ENDC}")
    report_parts.append(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.ENDC}")
    report_parts.append(f"\n  Analysis Period: {start_date} to {end_date}")
    report_parts.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Add each section
    report_parts.append(generate_ranking_table(results))
    report_parts.append(generate_summary_statistics(results))
    report_parts.append(generate_direction_analysis(results))
    report_parts.append(generate_prop_type_analysis(results))
    report_parts.append(generate_strengths_weaknesses(results))
    report_parts.append(generate_recommendations(results))
    
    # Footer
    report_parts.append("\n")
    report_parts.append(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.ENDC}")
    report_parts.append(f"{Colors.BOLD}{Colors.CYAN}  END OF REPORT{Colors.ENDC}")
    report_parts.append(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.ENDC}")
    
    return "\n".join(report_parts)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive NBA Props Model Analysis Suite"
    )
    parser.add_argument(
        "--start", "-s",
        type=str,
        default="2025-10-22",
        help="Start date (YYYY-MM-DD). Default: 2025-10-22 (season start)"
    )
    parser.add_argument(
        "--end", "-e",
        type=str,
        default="2026-02-02",
        help="End date (YYYY-MM-DD). Default: 2026-02-02"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        help="Specific model IDs to test (e.g., v18_general v19_under)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output during testing"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick test (last 14 days only)"
    )
    
    args = parser.parse_args()
    
    # Adjust dates for quick mode
    start_date = args.start
    end_date = args.end
    
    if args.quick:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=14)
        start_date = start_dt.strftime("%Y-%m-%d")
        print(f"  Quick mode: Testing {start_date} to {end_date}")
    
    # Filter models if specified
    models_to_test = ALL_MODELS
    if args.models:
        models_to_test = [m for m in ALL_MODELS if m.model_id in args.models]
        if not models_to_test:
            print(f"Error: No valid models found matching: {args.models}")
            print(f"Available models: {[m.model_id for m in ALL_MODELS]}")
            sys.exit(1)
    
    # Run all tests
    print_header("Starting Comprehensive Model Analysis")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Models: {len(models_to_test)}")
    print()
    
    start_time = time.time()
    results = run_all_tests(start_date, end_date, models_to_test, verbose=args.verbose)
    total_time = time.time() - start_time
    
    # Generate and print report
    report = generate_full_report(results, start_date, end_date)
    print(report)
    
    # Print timing info
    print(f"\n  Total analysis time: {total_time/60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
