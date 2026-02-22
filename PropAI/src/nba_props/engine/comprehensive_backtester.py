"""
Comprehensive Backtester - Unified backtesting for all NBA Props models.
=========================================================================

This module provides a unified interface for backtesting all models,
handling the diverse architectures and outputting standardized results.

Features:
---------
1. Universal model adapter for heterogeneous backtest result formats
2. Progress bar support for both terminal and GUI
3. Comprehensive quality metrics and analysis
4. Batch testing of multiple models with ranking
5. Detailed strength/weakness analysis

Author: PropAI Team
Created: February 2026
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple
from pathlib import Path
import json

from .model_registry import (
    MODEL_REGISTRY,
    ModelInfo,
    ModelCategory,
    ModelCapability,
    UnifiedBacktestResult,
    get_all_models,
    get_active_models,
    get_model_by_id,
    load_model_module,
    get_backtest_function,
    get_config_class,
    print_progress_bar,
)

# V19.3: Trade auto-detection and tank cache management for walk-forward backtesting
from .trade_tracker import (
    auto_detect_trades_from_boxscores,
    auto_update_team_roster_status,
    init_trade_tables,
)
from .tank_detector import clear_tank_detection_cache


# ============================================================================
# PROGRESS CALLBACK SYSTEM
# ============================================================================

@dataclass
class BacktestProgress:
    """Represents progress of a backtest operation."""
    model_id: str
    model_name: str
    current_model: int = 0
    total_models: int = 0
    current_date: str = ""
    current_date_num: int = 0
    total_dates: int = 0
    status: str = "pending"  # pending, running, completed, error
    error_message: str = ""
    start_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    
    @property
    def model_progress_pct(self) -> float:
        """Progress percentage for current model."""
        if self.total_dates <= 0:
            return 0.0
        return (self.current_date_num / self.total_dates) * 100
    
    @property
    def overall_progress_pct(self) -> float:
        """Overall progress percentage across all models."""
        if self.total_models <= 0:
            return 0.0
        model_pct = ((self.current_model - 1) / self.total_models) * 100
        within_model = (self.model_progress_pct / 100) * (100 / self.total_models)
        return model_pct + within_model
    
    def to_dict(self) -> Dict:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "current_model": self.current_model,
            "total_models": self.total_models,
            "current_date": self.current_date,
            "current_date_num": self.current_date_num,
            "total_dates": self.total_dates,
            "status": self.status,
            "error_message": self.error_message,
            "model_progress_pct": round(self.model_progress_pct, 1),
            "overall_progress_pct": round(self.overall_progress_pct, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


# Type for progress callback
ProgressCallback = Callable[[BacktestProgress], None]


# ============================================================================
# RESULT ADAPTER
# ============================================================================

def adapt_backtest_result(
    raw_result: Any,
    model_info: ModelInfo,
    start_date: str,
    end_date: str,
) -> UnifiedBacktestResult:
    """
    Adapt a model-specific backtest result to the unified format.
    
    This handles the diverse BacktestResult classes from different models:
    - BacktestResultV18, BacktestResultV16General, BacktestResult, etc.
    
    Args:
        raw_result: The result object from the model's backtest function
        model_info: ModelInfo for the model
        start_date: Start date of backtest
        end_date: End date of backtest
    
    Returns:
        UnifiedBacktestResult with normalized data
    """
    result = UnifiedBacktestResult(
        model_id=model_info.model_id,
        model_name=model_info.display_name,
        model_version=model_info.version,
        start_date=start_date,
        end_date=end_date,
        raw_result=raw_result,
    )
    
    # Calculate days tested
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    result.days_tested = (end_dt - start_dt).days + 1
    
    if raw_result is None:
        return result
    
    # Try to extract common fields using various attribute names
    # Different models use different naming conventions
    
    # Total picks/hits
    result.total_picks = _safe_get(raw_result, 
        ["total_picks", "picks_count", "num_picks", "total"], 0)
    result.total_hits = _safe_get(raw_result,
        ["total_hits", "hits", "wins", "correct"], 0)
    result.total_misses = result.total_picks - result.total_hits
    
    # Direction breakdown
    result.over_picks = _safe_get(raw_result,
        ["over_picks", "overs", "over_total"], 0)
    result.over_hits = _safe_get(raw_result,
        ["over_hits", "over_wins", "overs_hit"], 0)
    
    result.under_picks = _safe_get(raw_result,
        ["under_picks", "unders", "under_total"], 0)
    result.under_hits = _safe_get(raw_result,
        ["under_hits", "under_wins", "unders_hit"], 0)
    
    # Prop type breakdown
    result.pts_picks = _safe_get(raw_result,
        ["pts_picks", "pts_total", "points_picks"], 0)
    result.pts_hits = _safe_get(raw_result,
        ["pts_hits", "pts_wins", "points_hits"], 0)
    
    result.reb_picks = _safe_get(raw_result,
        ["reb_picks", "reb_total", "rebounds_picks"], 0)
    result.reb_hits = _safe_get(raw_result,
        ["reb_hits", "reb_wins", "rebounds_hits"], 0)
    
    result.ast_picks = _safe_get(raw_result,
        ["ast_picks", "ast_total", "assists_picks"], 0)
    result.ast_hits = _safe_get(raw_result,
        ["ast_hits", "ast_wins", "assists_hits"], 0)
    
    # Confidence tiers
    result.premium_picks = _safe_get(raw_result,
        ["premium_picks", "premium_total"], 0)
    result.premium_hits = _safe_get(raw_result,
        ["premium_hits", "premium_wins"], 0)
    
    result.high_picks = _safe_get(raw_result,
        ["high_picks", "high_conf_picks", "high_total"], 0)
    result.high_hits = _safe_get(raw_result,
        ["high_hits", "high_conf_hits", "high_wins"], 0)
    
    result.standard_picks = _safe_get(raw_result,
        ["standard_picks", "standard_total", "med_conf_picks", "medium_picks"], 0)
    result.standard_hits = _safe_get(raw_result,
        ["standard_hits", "standard_wins", "med_conf_hits", "medium_hits"], 0)
    
    # Line source
    result.sportsbook_picks = _safe_get(raw_result,
        ["sportsbook_picks", "sb_picks"], 0)
    result.sportsbook_hits = _safe_get(raw_result,
        ["sportsbook_hits", "sb_hits"], 0)
    
    result.derived_picks = _safe_get(raw_result,
        ["derived_picks", "derived_total"], 0)
    result.derived_hits = _safe_get(raw_result,
        ["derived_hits", "derived_wins"], 0)
    
    # MAE metrics
    result.mae_pts = _safe_get(raw_result, ["mae_pts"], 0.0)
    result.mae_reb = _safe_get(raw_result, ["mae_reb"], 0.0)
    result.mae_ast = _safe_get(raw_result, ["mae_ast"], 0.0)
    
    # Calculate all derived metrics
    result.calculate_derived_metrics()
    
    return result


def _safe_get(obj: Any, attr_names: List[str], default: Any) -> Any:
    """
    Safely get an attribute from an object, trying multiple names.
    
    Args:
        obj: Object to get attribute from
        attr_names: List of attribute names to try
        default: Default value if none found
    
    Returns:
        Attribute value or default
    """
    for name in attr_names:
        # Try as attribute
        if hasattr(obj, name):
            value = getattr(obj, name, None)
            if value is not None:
                return value
        
        # Try as dict key
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    
    return default


# ============================================================================
# SINGLE MODEL BACKTESTER
# ============================================================================

def run_single_model_backtest(
    model_id: str,
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> Optional[UnifiedBacktestResult]:
    """
    Run a backtest for a single model.
    
    Args:
        model_id: ID of the model to test (e.g., "v18_general")
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        db_path: Optional path to database
        verbose: Print detailed output
        show_progress: Show terminal progress bar
        progress_callback: Optional callback for GUI progress updates
    
    Returns:
        UnifiedBacktestResult or None if model not found/error
    """
    model_info = get_model_by_id(model_id)
    if not model_info:
        print(f"Model '{model_id}' not found in registry")
        return None
    
    if not model_info.is_active:
        print(f"Model '{model_id}' is not active")
        return None
    
    # Get backtest function
    backtest_fn = get_backtest_function(model_info)
    if not backtest_fn:
        print(f"No backtest function found for model '{model_id}'")
        return None
    
    # V19.3: Clear tank detection cache for fresh per-date evaluation
    clear_tank_detection_cache()
    
    # Initialize progress
    progress = BacktestProgress(
        model_id=model_id,
        model_name=model_info.display_name,
        current_model=1,
        total_models=1,
        status="running",
        start_time=time.time(),
    )
    
    # Calculate total dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    progress.total_dates = (end_dt - start_dt).days + 1
    
    if progress_callback:
        progress_callback(progress)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Running backtest: {model_info.display_name}")
        print(f"  Period: {start_date} to {end_date}")
        print(f"{'='*60}")
    
    try:
        # Determine arguments to pass to backtest function
        kwargs = {}
        
        # Try to get function signature hints
        import inspect
        try:
            sig = inspect.signature(backtest_fn)
            params = sig.parameters
            
            if "db_path" in params and db_path:
                kwargs["db_path"] = db_path
            if "verbose" in params:
                kwargs["verbose"] = verbose
            if "show_progress" in params:
                kwargs["show_progress"] = show_progress
        except Exception:
            pass
        
        # Run the backtest
        raw_result = backtest_fn(start_date, end_date, **kwargs)
        
        # Adapt to unified format
        result = adapt_backtest_result(raw_result, model_info, start_date, end_date)
        
        # Update progress
        progress.status = "completed"
        progress.elapsed_seconds = time.time() - progress.start_time
        if progress_callback:
            progress_callback(progress)
        
        if verbose:
            print(f"\n{result.summary()}")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"\nError running backtest for {model_id}: {error_msg}")
            traceback.print_exc()
        
        progress.status = "error"
        progress.error_message = error_msg
        progress.elapsed_seconds = time.time() - progress.start_time
        if progress_callback:
            progress_callback(progress)
        
        return None


# ============================================================================
# MULTI-MODEL BACKTESTER
# ============================================================================

@dataclass
class ComprehensiveBacktestResults:
    """Results from running backtests on multiple models."""
    start_date: str
    end_date: str
    models_tested: int = 0
    models_succeeded: int = 0
    models_failed: int = 0
    total_elapsed_seconds: float = 0.0
    
    results: List[UnifiedBacktestResult] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    
    # Rankings
    ranked_by_hit_rate: List[str] = field(default_factory=list)
    ranked_by_quality: List[str] = field(default_factory=list)
    ranked_by_volume: List[str] = field(default_factory=list)
    
    def calculate_rankings(self) -> None:
        """Calculate model rankings by different criteria."""
        valid_results = [r for r in self.results if r.total_picks > 0]
        
        # Rank by hit rate
        by_hit_rate = sorted(valid_results, key=lambda r: r.hit_rate, reverse=True)
        self.ranked_by_hit_rate = [r.model_id for r in by_hit_rate]
        
        # Rank by quality score
        by_quality = sorted(valid_results, key=lambda r: r.quality_score, reverse=True)
        self.ranked_by_quality = [r.model_id for r in by_quality]
        
        # Rank by volume (picks per day)
        by_volume = sorted(valid_results, key=lambda r: r.picks_per_day, reverse=True)
        self.ranked_by_volume = [r.model_id for r in by_volume]
    
    def get_best_model(self) -> Optional[UnifiedBacktestResult]:
        """Get the best model by quality score."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.quality_score)
    
    def get_result_by_model(self, model_id: str) -> Optional[UnifiedBacktestResult]:
        """Get result for a specific model."""
        for r in self.results:
            if r.model_id == model_id:
                return r
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "models_tested": self.models_tested,
            "models_succeeded": self.models_succeeded,
            "models_failed": self.models_failed,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 1),
            "results": [r.to_dict() for r in self.results],
            "errors": self.errors,
            "ranked_by_hit_rate": self.ranked_by_hit_rate,
            "ranked_by_quality": self.ranked_by_quality,
            "ranked_by_volume": self.ranked_by_volume,
        }
    
    def summary(self) -> str:
        """Generate comprehensive summary report."""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║              COMPREHENSIVE MODEL BACKTEST RESULTS                        ║",
            "╠══════════════════════════════════════════════════════════════════════════╣",
            f"║  Test Period: {self.start_date} to {self.end_date}                                  ║",
            f"║  Models Tested: {self.models_tested} | Succeeded: {self.models_succeeded} | Failed: {self.models_failed}                         ║",
            f"║  Total Runtime: {self.total_elapsed_seconds:.1f}s                                                    ║",
            "╚══════════════════════════════════════════════════════════════════════════╝",
            "",
        ]
        
        if not self.results:
            lines.append("  No results available.")
            return "\n".join(lines)
        
        # Top models by quality
        lines.append("  📊 TOP MODELS BY QUALITY SCORE")
        lines.append("  ─────────────────────────────────────────────────────────────────────────")
        
        valid_results = sorted(
            [r for r in self.results if r.total_picks > 0],
            key=lambda r: r.quality_score,
            reverse=True
        )
        
        for i, r in enumerate(valid_results[:10]):
            rank = f"#{i+1:2d}"
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            lines.append(
                f"  {medal} {rank} {r.model_name:25s} | "
                f"Hit: {r.hit_rate*100:5.1f}% | "
                f"Quality: {r.quality_score:5.1f} | "
                f"Picks: {r.total_picks:5d} | "
                f"{r.picks_per_day:4.1f}/day"
            )
        
        # Best for each criterion
        lines.extend([
            "",
            "  🏆 BEST BY CATEGORY",
            "  ─────────────────────────────────────────────────────────────────────────",
        ])
        
        # Best hit rate
        best_hit = max(valid_results, key=lambda r: r.hit_rate) if valid_results else None
        if best_hit:
            lines.append(f"  Highest Hit Rate:    {best_hit.model_name} ({best_hit.hit_rate*100:.1f}%)")
        
        # Best for UNDER
        under_results = [r for r in valid_results if r.under_picks >= 10]
        if under_results:
            best_under = max(under_results, key=lambda r: r.under_rate)
            lines.append(f"  Best UNDER:          {best_under.model_name} ({best_under.under_rate*100:.1f}%)")
        
        # Best for OVER
        over_results = [r for r in valid_results if r.over_picks >= 10]
        if over_results:
            best_over = max(over_results, key=lambda r: r.over_rate)
            lines.append(f"  Best OVER:           {best_over.model_name} ({best_over.over_rate*100:.1f}%)")
        
        # Best for PTS
        pts_results = [r for r in valid_results if r.pts_picks >= 10]
        if pts_results:
            best_pts = max(pts_results, key=lambda r: r.pts_rate)
            lines.append(f"  Best PTS:            {best_pts.model_name} ({best_pts.pts_rate*100:.1f}%)")
        
        # Best for REB
        reb_results = [r for r in valid_results if r.reb_picks >= 10]
        if reb_results:
            best_reb = max(reb_results, key=lambda r: r.reb_rate)
            lines.append(f"  Best REB:            {best_reb.model_name} ({best_reb.reb_rate*100:.1f}%)")
        
        # Errors
        if self.errors:
            lines.extend([
                "",
                "  ⚠️  FAILED MODELS",
                "  ─────────────────────────────────────────────────────────────────────────",
            ])
            for model_id, error in self.errors.items():
                lines.append(f"  • {model_id}: {error[:60]}...")
        
        lines.append("")
        
        return "\n".join(lines)


def run_comprehensive_backtest(
    start_date: str,
    end_date: str,
    model_ids: Optional[List[str]] = None,
    db_path: Optional[str] = None,
    verbose: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> ComprehensiveBacktestResults:
    """
    Run backtests on multiple models and compile results.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        model_ids: List of model IDs to test (None = all active models)
        db_path: Optional path to database
        verbose: Print detailed output
        show_progress: Show terminal progress bar
        progress_callback: Optional callback for GUI progress updates
    
    Returns:
        ComprehensiveBacktestResults with all model results and rankings
    """
    results = ComprehensiveBacktestResults(
        start_date=start_date,
        end_date=end_date,
    )
    
    start_time = time.time()
    
    # V19.3: Auto-detect trades from boxscore data before running backtests.
    # This ensures the walk-forward backtest has trade context available
    # for all models, without requiring manual CLI trade entry.
    _run_trade_auto_detection(db_path=db_path, verbose=verbose)
    
    # Clear tank detection cache to start fresh
    clear_tank_detection_cache()
    
    # Get models to test
    if model_ids:
        models_to_test = [get_model_by_id(mid) for mid in model_ids]
        models_to_test = [m for m in models_to_test if m is not None]
    else:
        models_to_test = get_active_models()
    
    results.models_tested = len(models_to_test)
    
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  COMPREHENSIVE MODEL BACKTEST")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Models: {len(models_to_test)}")
        print(f"{'═'*70}")
    
    # Initialize overall progress
    overall_progress = BacktestProgress(
        model_id="",
        model_name="",
        total_models=len(models_to_test),
        status="running",
        start_time=start_time,
    )
    
    # Test each model
    for idx, model_info in enumerate(models_to_test):
        overall_progress.current_model = idx + 1
        overall_progress.model_id = model_info.model_id
        overall_progress.model_name = model_info.display_name
        overall_progress.current_date_num = 0
        
        if show_progress:
            print_progress_bar(
                idx + 1, len(models_to_test),
                prefix=f"Testing Models",
                suffix=f"{model_info.model_id:20s}",
                length=40
            )
        
        if progress_callback:
            overall_progress.elapsed_seconds = time.time() - start_time
            progress_callback(overall_progress)
        
        try:
            result = run_single_model_backtest(
                model_info.model_id,
                start_date,
                end_date,
                db_path=db_path,
                verbose=False,  # Don't be verbose for each model in batch
                show_progress=False,  # Don't show individual progress bars
                progress_callback=None,  # Handle progress at batch level
            )
            
            if result:
                results.results.append(result)
                results.models_succeeded += 1
            else:
                results.errors[model_info.model_id] = "Backtest returned None"
                results.models_failed += 1
                
        except Exception as e:
            results.errors[model_info.model_id] = str(e)
            results.models_failed += 1
            if verbose:
                print(f"\n  ⚠️  Error testing {model_info.model_id}: {e}")
    
    # Calculate rankings
    results.calculate_rankings()
    
    # Final timing
    results.total_elapsed_seconds = time.time() - start_time
    
    # Final progress update
    overall_progress.status = "completed"
    overall_progress.elapsed_seconds = results.total_elapsed_seconds
    if progress_callback:
        progress_callback(overall_progress)
    
    if verbose:
        print(results.summary())
    
    return results


# ============================================================================
# QUICK COMPARISON FUNCTIONS
# ============================================================================

def compare_latest_models(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    verbose: bool = True,
) -> ComprehensiveBacktestResults:
    """
    Compare the latest multi-file models (v16-v19).
    
    This is a quick way to compare the most recent model versions.
    """
    latest_models = [
        "v19_general", "v19_under",
        "v18_general", "v18_under",
        "v17_general", "v17_under",
        "v16_general", "v16_under",
    ]
    
    return run_comprehensive_backtest(
        start_date=start_date,
        end_date=end_date,
        model_ids=latest_models,
        db_path=db_path,
        verbose=verbose,
        show_progress=True,
    )


def compare_single_file_models(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    verbose: bool = True,
) -> ComprehensiveBacktestResults:
    """
    Compare the single-file models (v2-v10).
    """
    single_models = ["v10", "v9", "v8", "v5", "v4", "v3", "v2"]
    
    return run_comprehensive_backtest(
        start_date=start_date,
        end_date=end_date,
        model_ids=single_models,
        db_path=db_path,
        verbose=verbose,
        show_progress=True,
    )


def compare_under_models(
    start_date: str,
    end_date: str,
    db_path: Optional[str] = None,
    verbose: bool = True,
) -> ComprehensiveBacktestResults:
    """
    Compare all specialized UNDER models.
    """
    under_models = [
        "v19_under", "v18_under", "v17_under", "v16_under",
        "under_v2"
    ]
    
    return run_comprehensive_backtest(
        start_date=start_date,
        end_date=end_date,
        model_ids=under_models,
        db_path=db_path,
        verbose=verbose,
        show_progress=True,
    )


def compare_general_models(
    start_date: str,
    end_date: str,
    verbose: bool = True,
) -> ComprehensiveBacktestResults:
    """
    Compare all general models.
    """
    general_models = [
        "v19_general", "v18_general", "v17_general", "v16_general",
        "v15_general", "v14_general", "v13_general", "v12_general",
        "v10", "v9", "v8", "production", "final"
    ]
    
    return run_comprehensive_backtest(
        start_date=start_date,
        end_date=end_date,
        model_ids=general_models,
        verbose=verbose,
        show_progress=True,
    )


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def generate_model_comparison_report(
    results: ComprehensiveBacktestResults,
    output_format: str = "text",
) -> str:
    """
    Generate a detailed comparison report from backtest results.
    
    Args:
        results: ComprehensiveBacktestResults
        output_format: "text", "markdown", or "json"
    
    Returns:
        Formatted report string
    """
    if output_format == "json":
        return json.dumps(results.to_dict(), indent=2)
    
    elif output_format == "markdown":
        lines = [
            "# Model Comparison Report",
            "",
            f"**Test Period:** {results.start_date} to {results.end_date}",
            f"**Models Tested:** {results.models_tested}",
            f"**Total Runtime:** {results.total_elapsed_seconds:.1f}s",
            "",
            "## Rankings by Quality Score",
            "",
            "| Rank | Model | Hit Rate | Quality | Picks | Picks/Day |",
            "|------|-------|----------|---------|-------|-----------|",
        ]
        
        valid_results = sorted(
            [r for r in results.results if r.total_picks > 0],
            key=lambda r: r.quality_score,
            reverse=True
        )
        
        for i, r in enumerate(valid_results):
            lines.append(
                f"| {i+1} | {r.model_name} | {r.hit_rate*100:.1f}% | "
                f"{r.quality_score:.1f} | {r.total_picks} | {r.picks_per_day:.1f} |"
            )
        
        # Detailed breakdown
        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])
        
        for r in valid_results[:5]:  # Top 5 only
            lines.extend([
                f"### {r.model_name}",
                "",
                f"- **Overall:** {r.hit_rate*100:.1f}% ({r.total_hits}/{r.total_picks})",
                f"- **OVER:** {r.over_rate*100:.1f}% ({r.over_hits}/{r.over_picks})",
                f"- **UNDER:** {r.under_rate*100:.1f}% ({r.under_hits}/{r.under_picks})",
                f"- **PTS:** {r.pts_rate*100:.1f}% | **REB:** {r.reb_rate*100:.1f}% | **AST:** {r.ast_rate*100:.1f}%",
                "",
                "**Strengths:**",
            ])
            for s in r.strengths:
                lines.append(f"- ✓ {s}")
            lines.append("")
            lines.append("**Weaknesses:**")
            for w in r.weaknesses:
                lines.append(f"- ✗ {w}")
            lines.append("")
        
        return "\n".join(lines)
    
    else:  # text format
        return results.summary()


def get_model_recommendations(
    results: ComprehensiveBacktestResults,
) -> Dict[str, Any]:
    """
    Generate recommendations based on backtest results.
    
    Returns dict with:
    - best_overall: Best model by quality score
    - best_under: Best model for UNDER picks
    - best_over: Best model for OVER picks
    - best_pts: Best model for PTS predictions
    - best_reb: Best model for REB predictions
    - avoid: Models to avoid (below breakeven)
    """
    valid = [r for r in results.results if r.total_picks > 0]
    
    recommendations = {
        "best_overall": None,
        "best_under": None,
        "best_over": None,
        "best_pts": None,
        "best_reb": None,
        "best_volume": None,
        "avoid": [],
    }
    
    if not valid:
        return recommendations
    
    # Best overall
    best = max(valid, key=lambda r: r.quality_score)
    recommendations["best_overall"] = {
        "model_id": best.model_id,
        "model_name": best.model_name,
        "hit_rate": round(best.hit_rate * 100, 1),
        "quality_score": round(best.quality_score, 1),
        "reason": best.strengths[0] if best.strengths else "Highest quality score",
    }
    
    # Best for UNDER
    under_valid = [r for r in valid if r.under_picks >= 10]
    if under_valid:
        best_under = max(under_valid, key=lambda r: r.under_rate)
        recommendations["best_under"] = {
            "model_id": best_under.model_id,
            "model_name": best_under.model_name,
            "rate": round(best_under.under_rate * 100, 1),
            "picks": best_under.under_picks,
        }
    
    # Best for OVER
    over_valid = [r for r in valid if r.over_picks >= 10]
    if over_valid:
        best_over = max(over_valid, key=lambda r: r.over_rate)
        recommendations["best_over"] = {
            "model_id": best_over.model_id,
            "model_name": best_over.model_name,
            "rate": round(best_over.over_rate * 100, 1),
            "picks": best_over.over_picks,
        }
    
    # Best for PTS
    pts_valid = [r for r in valid if r.pts_picks >= 10]
    if pts_valid:
        best_pts = max(pts_valid, key=lambda r: r.pts_rate)
        recommendations["best_pts"] = {
            "model_id": best_pts.model_id,
            "model_name": best_pts.model_name,
            "rate": round(best_pts.pts_rate * 100, 1),
            "picks": best_pts.pts_picks,
        }
    
    # Best for REB
    reb_valid = [r for r in valid if r.reb_picks >= 10]
    if reb_valid:
        best_reb = max(reb_valid, key=lambda r: r.reb_rate)
        recommendations["best_reb"] = {
            "model_id": best_reb.model_id,
            "model_name": best_reb.model_name,
            "rate": round(best_reb.reb_rate * 100, 1),
            "picks": best_reb.reb_picks,
        }
    
    # Best volume
    best_vol = max(valid, key=lambda r: r.picks_per_day)
    recommendations["best_volume"] = {
        "model_id": best_vol.model_id,
        "model_name": best_vol.model_name,
        "picks_per_day": round(best_vol.picks_per_day, 1),
    }
    
    # Models to avoid (below breakeven)
    for r in valid:
        if r.hit_rate < 0.50 and r.total_picks >= 20:
            recommendations["avoid"].append({
                "model_id": r.model_id,
                "model_name": r.model_name,
                "hit_rate": round(r.hit_rate * 100, 1),
                "reason": r.weaknesses[0] if r.weaknesses else "Below breakeven",
            })
    
    return recommendations


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def _run_trade_auto_detection(
    db_path: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    V19.3: Run trade auto-detection from boxscore data.
    
    This ensures that walk-forward backtests have trade context available
    without requiring manual CLI trade entry. It scans boxscore data for
    team changes and auto-generates team roster status from detected trades.
    
    Safe to call multiple times — already-recorded trades are skipped.
    """
    try:
        from ..db import Db
        from ..paths import get_paths
        
        paths = get_paths()
        actual_db_path = db_path or str(paths.db_path)
        
        db = Db(actual_db_path)
        with db.connect() as conn:
            # Initialize trade tables if needed
            init_trade_tables(conn)
            
            # Auto-detect trades from boxscore team changes
            trades = auto_detect_trades_from_boxscores(
                conn, verbose=verbose
            )
            
            # Auto-update team roster status from detected trades
            if trades:
                auto_update_team_roster_status(conn, verbose=verbose)
            
            if verbose and trades:
                print(f"  ✅ Trade auto-detection: found {len(trades)} trades")
                
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Trade auto-detection failed (non-fatal): {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NBA Props Model Backtester")
    parser.add_argument("--start", default="2025-10-22", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-02", help="End date (YYYY-MM-DD)")
    parser.add_argument("--models", nargs="+", help="Specific model IDs to test")
    parser.add_argument("--latest", action="store_true", help="Test only latest models (v16-v19)")
    parser.add_argument("--under", action="store_true", help="Test only UNDER models")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    
    args = parser.parse_args()
    
    if args.latest:
        results = compare_latest_models(args.start, args.end, verbose=args.verbose)
    elif args.under:
        results = compare_under_models(args.start, args.end, verbose=args.verbose)
    elif args.models:
        results = run_comprehensive_backtest(
            args.start, args.end,
            model_ids=args.models,
            verbose=args.verbose,
            show_progress=True,
        )
    else:
        results = run_comprehensive_backtest(
            args.start, args.end,
            verbose=args.verbose,
            show_progress=True,
        )
    
    # Output report
    report = generate_model_comparison_report(results, args.format)
    print(report)
