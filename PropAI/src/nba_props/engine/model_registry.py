"""
Model Registry - Comprehensive catalog of all NBA Props prediction models.
==========================================================================

This module provides a unified registry for all models, enabling:
1. Automated discovery and cataloging of all model versions
2. Standardized backtest execution across heterogeneous model architectures
3. Comprehensive performance comparison and analysis
4. Quality metrics including picks-to-accuracy ratios

Model Categories:
-----------------
1. SINGLE FILE MODELS (v2-v10):
   - Self-contained models with all logic in one file
   - Interface: run_backtest_v{N}() or run_backtest()
   
2. MULTI-FILE MODELS (v12-v19):
   - Split architecture: _general.py, _under.py, _shared.py
   - May have 2 or 3 files depending on version
   - Interface: run_backtest_v{N}_general(), run_backtest_v{N}_under()
   
3. FOLDER-BASED MODELS (v6, v7):
   - Models organized in subdirectories
   - Multiple specialized modules (backtester, projector, etc.)
   
4. SPECIALIZED MODELS:
   - model_production.py - Production deployment model
   - model_final.py - Final consolidated model
   - under_model_v2.py - Specialized UNDER model

Author: PropAI Team
Created: February 2026
"""
from __future__ import annotations

import sys
import importlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from enum import Enum
from pathlib import Path


class ModelCategory(Enum):
    """Categories of models based on architecture."""
    SINGLE_FILE = "single_file"      # v2-v10: all logic in one file
    MULTI_FILE = "multi_file"        # v12-v19: general/under/shared
    FOLDER_BASED = "folder"          # v6, v7: subdirectory structure
    SPECIALIZED = "specialized"      # production, final, under_model_v2


class ModelCapability(Enum):
    """Capabilities a model may support."""
    OVER_PICKS = "over"              # Can generate OVER predictions
    UNDER_PICKS = "under"            # Can generate UNDER predictions
    PTS_PROPS = "pts"                # Supports points predictions
    REB_PROPS = "reb"                # Supports rebounds predictions
    AST_PROPS = "ast"                # Supports assists predictions
    SPORTSBOOK_LINES = "sportsbook"  # Uses actual sportsbook lines
    DERIVED_LINES = "derived"        # Can use derived lines
    PROGRESS_BAR = "progress"        # Has progress bar support
    VERBOSE_OUTPUT = "verbose"       # Supports verbose output


@dataclass
class ModelInfo:
    """Information about a registered model."""
    model_id: str                    # Unique identifier (e.g., "v18_general")
    display_name: str                # Human-readable name
    version: str                     # Version number (e.g., "18.0")
    category: ModelCategory          # Model architecture category
    capabilities: List[ModelCapability]  # Supported capabilities
    
    # File information
    primary_file: str                # Main model file
    secondary_files: List[str] = field(default_factory=list)  # Related files
    
    # Function references
    backtest_function: Optional[str] = None  # Function name for backtesting
    picks_function: Optional[str] = None     # Function name for getting picks
    config_class: Optional[str] = None       # Configuration class name
    
    # Metadata
    description: str = ""            # Model description
    created_date: Optional[str] = None
    is_active: bool = True           # Whether model should be tested
    
    # Performance hints
    expected_runtime_mins: float = 5.0  # Estimated backtest time for 30 days


@dataclass 
class UnifiedBacktestResult:
    """
    Unified backtest result that normalizes outputs from different models.
    
    This standardizes the diverse BacktestResult classes from each model
    into a common format for comparison and analysis.
    """
    # Model identification
    model_id: str
    model_name: str
    model_version: str
    
    # Test period
    start_date: str
    end_date: str
    days_tested: int = 0
    
    # Core metrics
    total_picks: int = 0
    total_hits: int = 0
    total_misses: int = 0
    
    # Accuracy metrics
    hit_rate: float = 0.0            # total_hits / total_picks
    
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
    
    # Confidence tiers
    premium_picks: int = 0
    premium_hits: int = 0
    premium_rate: float = 0.0
    
    high_picks: int = 0
    high_hits: int = 0
    high_rate: float = 0.0
    
    standard_picks: int = 0
    standard_hits: int = 0
    standard_rate: float = 0.0
    
    # Line source (if tracked)
    sportsbook_picks: int = 0
    sportsbook_hits: int = 0
    sportsbook_rate: float = 0.0
    
    derived_picks: int = 0
    derived_hits: int = 0
    derived_rate: float = 0.0
    
    # Quality Metrics (NEW - for picks-to-accuracy analysis)
    picks_per_day: float = 0.0       # Average picks per day
    quality_score: float = 0.0       # Combined metric balancing picks and accuracy
    volume_score: float = 0.0        # Score based on pick volume (not too few, not too many)
    consistency_score: float = 0.0   # How consistent is performance across days
    
    # Error metrics (if available)
    mae_pts: float = 0.0
    mae_reb: float = 0.0
    mae_ast: float = 0.0
    
    # Raw result object for additional analysis
    raw_result: Any = None
    
    # Detailed pick results
    pick_details: List[Dict] = field(default_factory=list)
    
    # Strengths and weaknesses
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def calculate_derived_metrics(self) -> None:
        """Calculate all derived metrics from raw counts."""
        # Basic hit rate
        self.hit_rate = self.total_hits / self.total_picks if self.total_picks > 0 else 0.0
        
        # Direction rates
        self.over_rate = self.over_hits / self.over_picks if self.over_picks > 0 else 0.0
        self.under_rate = self.under_hits / self.under_picks if self.under_picks > 0 else 0.0
        
        # Prop type rates
        self.pts_rate = self.pts_hits / self.pts_picks if self.pts_picks > 0 else 0.0
        self.reb_rate = self.reb_hits / self.reb_picks if self.reb_picks > 0 else 0.0
        self.ast_rate = self.ast_hits / self.ast_picks if self.ast_picks > 0 else 0.0
        
        # Tier rates
        self.premium_rate = self.premium_hits / self.premium_picks if self.premium_picks > 0 else 0.0
        self.high_rate = self.high_hits / self.high_picks if self.high_picks > 0 else 0.0
        self.standard_rate = self.standard_hits / self.standard_picks if self.standard_picks > 0 else 0.0
        
        # Line source rates
        self.sportsbook_rate = self.sportsbook_hits / self.sportsbook_picks if self.sportsbook_picks > 0 else 0.0
        self.derived_rate = self.derived_hits / self.derived_picks if self.derived_picks > 0 else 0.0
        
        # Picks per day
        self.picks_per_day = self.total_picks / self.days_tested if self.days_tested > 0 else 0.0
        
        # Calculate quality scores
        self._calculate_quality_score()
        self._identify_strengths_weaknesses()
    
    def _calculate_quality_score(self) -> None:
        """
        Calculate comprehensive quality score that balances picks and accuracy.
        
        A model that makes 1000 picks at 52% is not as valuable as one making
        50 picks at 65%. This score rewards:
        - Higher accuracy (primary factor)
        - Reasonable pick volume (too few = unreliable, too many = noise)
        - Consistency across prop types
        """
        if self.total_picks == 0:
            self.quality_score = 0.0
            return
        
        # Base score from hit rate (0-100 scale, 50% = 0, 65% = 100)
        # A 55% model gets 33.3, a 60% gets 66.7
        accuracy_score = max(0, (self.hit_rate - 0.50) / 0.15 * 100)
        
        # Volume score using bell curve around optimal picks/day
        # Optimal: 5-15 picks per day. Too few or too many reduces score.
        optimal_ppd = 10.0
        ppd_deviation = abs(self.picks_per_day - optimal_ppd)
        if self.picks_per_day < 2:
            # Very few picks = unreliable sample
            self.volume_score = max(0, 20 * self.picks_per_day)
        elif self.picks_per_day > 30:
            # Too many picks = likely low quality
            self.volume_score = max(0, 100 - (self.picks_per_day - 30) * 3)
        else:
            # Normal range: slight penalty for deviation from optimal
            self.volume_score = max(0, 100 - ppd_deviation * 5)
        
        # Consistency score: variance between prop type hit rates
        rates = []
        if self.pts_picks >= 10:
            rates.append(self.pts_rate)
        if self.reb_picks >= 10:
            rates.append(self.reb_rate)
        if self.ast_picks >= 10:
            rates.append(self.ast_rate)
        
        if len(rates) >= 2:
            import statistics
            rate_std = statistics.stdev(rates)
            # Lower variance = more consistent = higher score
            self.consistency_score = max(0, 100 - rate_std * 500)
        else:
            self.consistency_score = 70  # Default for limited prop types
        
        # Combined quality score
        # Accuracy is 60%, Volume is 25%, Consistency is 15%
        self.quality_score = (
            accuracy_score * 0.60 +
            self.volume_score * 0.25 +
            self.consistency_score * 0.15
        )
    
    def _identify_strengths_weaknesses(self) -> None:
        """Identify model strengths and weaknesses based on metrics."""
        self.strengths = []
        self.weaknesses = []
        
        # Overall accuracy
        if self.hit_rate >= 0.60:
            self.strengths.append(f"Strong overall hit rate: {self.hit_rate*100:.1f}%")
        elif self.hit_rate >= 0.55:
            self.strengths.append(f"Solid overall hit rate: {self.hit_rate*100:.1f}%")
        elif self.hit_rate < 0.50:
            self.weaknesses.append(f"Below breakeven hit rate: {self.hit_rate*100:.1f}%")
        
        # Direction analysis
        if self.over_picks >= 10:
            if self.over_rate >= 0.58:
                self.strengths.append(f"Excellent OVER predictions: {self.over_rate*100:.1f}%")
            elif self.over_rate < 0.48:
                self.weaknesses.append(f"Poor OVER predictions: {self.over_rate*100:.1f}%")
        
        if self.under_picks >= 10:
            if self.under_rate >= 0.58:
                self.strengths.append(f"Excellent UNDER predictions: {self.under_rate*100:.1f}%")
            elif self.under_rate < 0.48:
                self.weaknesses.append(f"Poor UNDER predictions: {self.under_rate*100:.1f}%")
        
        # Prop type analysis
        if self.pts_picks >= 10:
            if self.pts_rate >= 0.58:
                self.strengths.append(f"Strong PTS predictions: {self.pts_rate*100:.1f}%")
            elif self.pts_rate < 0.48:
                self.weaknesses.append(f"Weak PTS predictions: {self.pts_rate*100:.1f}%")
        
        if self.reb_picks >= 10:
            if self.reb_rate >= 0.58:
                self.strengths.append(f"Strong REB predictions: {self.reb_rate*100:.1f}%")
            elif self.reb_rate < 0.48:
                self.weaknesses.append(f"Weak REB predictions: {self.reb_rate*100:.1f}%")
        
        if self.ast_picks >= 10:
            if self.ast_rate >= 0.58:
                self.strengths.append(f"Strong AST predictions: {self.ast_rate*100:.1f}%")
            elif self.ast_rate < 0.50:
                self.weaknesses.append(f"Volatile AST predictions: {self.ast_rate*100:.1f}%")
        
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
                self.weaknesses.append(f"High confidence inconsistent: {self.high_rate*100:.1f}%")
        
        # Volume analysis
        if self.picks_per_day < 2 and self.days_tested >= 10:
            self.weaknesses.append(f"Very low pick volume: {self.picks_per_day:.1f}/day")
        elif self.picks_per_day > 30:
            self.weaknesses.append(f"Excessive picks (may be noise): {self.picks_per_day:.1f}/day")
        elif 5 <= self.picks_per_day <= 15:
            self.strengths.append(f"Optimal pick volume: {self.picks_per_day:.1f}/day")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "days_tested": self.days_tested,
            
            "total_picks": self.total_picks,
            "total_hits": self.total_hits,
            "hit_rate": round(self.hit_rate * 100, 2),
            
            "over_picks": self.over_picks,
            "over_hits": self.over_hits,
            "over_rate": round(self.over_rate * 100, 2),
            
            "under_picks": self.under_picks,
            "under_hits": self.under_hits,
            "under_rate": round(self.under_rate * 100, 2),
            
            "pts_picks": self.pts_picks,
            "pts_hits": self.pts_hits,
            "pts_rate": round(self.pts_rate * 100, 2),
            
            "reb_picks": self.reb_picks,
            "reb_hits": self.reb_hits,
            "reb_rate": round(self.reb_rate * 100, 2),
            
            "ast_picks": self.ast_picks,
            "ast_hits": self.ast_hits,
            "ast_rate": round(self.ast_rate * 100, 2),
            
            "premium_picks": self.premium_picks,
            "premium_hits": self.premium_hits,
            "premium_rate": round(self.premium_rate * 100, 2),
            
            "high_picks": self.high_picks,
            "high_hits": self.high_hits,
            "high_rate": round(self.high_rate * 100, 2),
            
            "standard_picks": self.standard_picks,
            "standard_hits": self.standard_hits,
            "standard_rate": round(self.standard_rate * 100, 2),
            
            "sportsbook_picks": self.sportsbook_picks,
            "sportsbook_hits": self.sportsbook_hits,
            "sportsbook_rate": round(self.sportsbook_rate * 100, 2),
            
            "derived_picks": self.derived_picks,
            "derived_hits": self.derived_hits,
            "derived_rate": round(self.derived_rate * 100, 2),
            
            "picks_per_day": round(self.picks_per_day, 2),
            "quality_score": round(self.quality_score, 2),
            "volume_score": round(self.volume_score, 2),
            "consistency_score": round(self.consistency_score, 2),
            
            "mae_pts": round(self.mae_pts, 2) if self.mae_pts else None,
            "mae_reb": round(self.mae_reb, 2) if self.mae_reb else None,
            "mae_ast": round(self.mae_ast, 2) if self.mae_ast else None,
            
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"═══════════════════════════════════════════════════════════════",
            f"  {self.model_name} (v{self.model_version})",
            f"═══════════════════════════════════════════════════════════════",
            f"  Test Period: {self.start_date} to {self.end_date} ({self.days_tested} days)",
            f"",
            f"  OVERALL PERFORMANCE",
            f"  ─────────────────────────────────────────────────────────────",
            f"  Hit Rate:      {self.hit_rate*100:.1f}% ({self.total_hits}/{self.total_picks})",
            f"  Quality Score: {self.quality_score:.1f}/100",
            f"  Picks/Day:     {self.picks_per_day:.1f}",
            f"",
            f"  BY DIRECTION",
            f"  ─────────────────────────────────────────────────────────────",
            f"  OVER:          {self.over_rate*100:.1f}% ({self.over_hits}/{self.over_picks})",
            f"  UNDER:         {self.under_rate*100:.1f}% ({self.under_hits}/{self.under_picks})",
            f"",
            f"  BY PROP TYPE",
            f"  ─────────────────────────────────────────────────────────────",
            f"  PTS:           {self.pts_rate*100:.1f}% ({self.pts_hits}/{self.pts_picks})",
            f"  REB:           {self.reb_rate*100:.1f}% ({self.reb_hits}/{self.reb_picks})",
            f"  AST:           {self.ast_rate*100:.1f}% ({self.ast_hits}/{self.ast_picks})",
        ]
        
        if self.premium_picks > 0 or self.high_picks > 0:
            lines.extend([
                f"",
                f"  BY CONFIDENCE",
                f"  ─────────────────────────────────────────────────────────────",
            ])
            if self.premium_picks > 0:
                lines.append(f"  Premium:       {self.premium_rate*100:.1f}% ({self.premium_hits}/{self.premium_picks})")
            if self.high_picks > 0:
                lines.append(f"  High:          {self.high_rate*100:.1f}% ({self.high_hits}/{self.high_picks})")
            if self.standard_picks > 0:
                lines.append(f"  Standard:      {self.standard_rate*100:.1f}% ({self.standard_hits}/{self.standard_picks})")
        
        if self.sportsbook_picks > 0 or self.derived_picks > 0:
            lines.extend([
                f"",
                f"  BY LINE SOURCE",
                f"  ─────────────────────────────────────────────────────────────",
            ])
            if self.sportsbook_picks > 0:
                lines.append(f"  Sportsbook:    {self.sportsbook_rate*100:.1f}% ({self.sportsbook_hits}/{self.sportsbook_picks})")
            if self.derived_picks > 0:
                lines.append(f"  Derived:       {self.derived_rate*100:.1f}% ({self.derived_hits}/{self.derived_picks})")
        
        if self.strengths or self.weaknesses:
            lines.extend([
                f"",
                f"  ANALYSIS",
                f"  ─────────────────────────────────────────────────────────────",
            ])
            for strength in self.strengths:
                lines.append(f"  ✓ {strength}")
            for weakness in self.weaknesses:
                lines.append(f"  ✗ {weakness}")
        
        lines.append("═══════════════════════════════════════════════════════════════")
        
        return "\n".join(lines)


# ============================================================================
# MODEL REGISTRY
# ============================================================================

# Complete registry of all models with their configurations
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # -------------------------------------------------------------------------
    # MULTI-FILE MODELS (v12-v19) - Most Recent
    # -------------------------------------------------------------------------
    "v19_general": ModelInfo(
        model_id="v19_general",
        display_name="Model V19 General",
        version="19.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.PROGRESS_BAR,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v19_general.py",
        secondary_files=["model_v19_shared.py", "model_v19_under.py"],
        backtest_function="run_backtest_v19_general",
        picks_function="get_daily_picks_v19_general",
        config_class="ModelConfigV19General",
        description="Multi-factor model with strict alignment requirements and comprehensive box score analysis",
        is_active=True,
    ),
    
    "v19_under": ModelInfo(
        model_id="v19_under",
        display_name="Model V19 Under",
        version="19.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.PROGRESS_BAR,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v19_under.py",
        secondary_files=["model_v19_shared.py"],
        backtest_function="run_backtest_v19_under",
        picks_function="get_daily_picks_v19_under",
        config_class="ModelConfigV19Under",
        description="Specialized UNDER model with multi-factor requirements",
        is_active=True,
    ),
    
    "v18_general": ModelInfo(
        model_id="v18_general",
        display_name="Model V18 General",
        version="18.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.PROGRESS_BAR,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v18_general.py",
        secondary_files=["model_v18_shared.py", "model_v18_under.py"],
        backtest_function="run_backtest_v18_general",
        picks_function="get_daily_picks_v18_general",
        config_class="ModelConfigV18General",
        description="Holistic multi-factor model with comprehensive box score analysis",
        is_active=True,
    ),
    
    "v18_under": ModelInfo(
        model_id="v18_under",
        display_name="Model V18 Under",
        version="18.5",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.PROGRESS_BAR,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v18_under.py",
        secondary_files=["model_v18_shared.py"],
        backtest_function="run_backtest_v18_under",
        picks_function="get_daily_picks_v18_under",
        config_class="ModelConfigV18Under",
        description="Specialized UNDER model with validated factor weights",
        is_active=True,
    ),
    
    "v17_general": ModelInfo(
        model_id="v17_general",
        display_name="Model V17 General",
        version="17.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.PROGRESS_BAR,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v17_general.py",
        secondary_files=["model_v17_shared.py", "model_v17_under.py"],
        backtest_function="run_backtest_v17_general",
        picks_function="get_daily_picks_v17_general",
        config_class="ModelConfigV17General",
        description="Holistic multi-factor approach with strategic direction selection",
        is_active=True,
    ),
    
    "v17_under": ModelInfo(
        model_id="v17_under",
        display_name="Model V17 Under",
        version="17.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v17_under.py",
        secondary_files=["model_v17_shared.py"],
        backtest_function="run_backtest_v17_under",
        picks_function="get_daily_picks_v17_under",
        config_class="ModelConfigV17Under",
        description="Specialized UNDER model for V17",
        is_active=True,
    ),
    
    "v16_general": ModelInfo(
        model_id="v16_general",
        display_name="Model V16 General",
        version="16.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v16_general.py",
        secondary_files=["model_v16_shared.py", "model_v16_under.py"],
        backtest_function="run_backtest_v16_general",
        picks_function="get_daily_picks_v16_general",
        config_class="ModelConfigV16General",
        description="Pattern-based picks with hybrid line approach",
        is_active=True,
    ),
    
    "v16_under": ModelInfo(
        model_id="v16_under",
        display_name="Model V16 Under",
        version="16.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v16_under.py",
        secondary_files=["model_v16_shared.py"],
        backtest_function="run_backtest_v16_under",
        picks_function="get_daily_picks_v16_under",
        config_class="ModelConfigV16Under",
        description="Specialized UNDER model with defense integration",
        is_active=True,
    ),
    
    "v15_general": ModelInfo(
        model_id="v15_general",
        display_name="Model V15 General",
        version="15.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v15_general.py",
        secondary_files=["model_v15_shared.py", "model_v15_under.py"],
        backtest_function="run_backtest_v15_general",
        picks_function="get_daily_picks_v15_general",
        config_class="ModelConfigV15General",
        description="Derived line fallacy fix with pattern-based picks",
        is_active=True,
    ),
    
    "v14_general": ModelInfo(
        model_id="v14_general",
        display_name="Model V14 General",
        version="14.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v14_general.py",
        secondary_files=["model_v14_shared.py", "model_v14_under.py"],
        backtest_function="run_backtest_v14_general",
        picks_function="get_daily_picks_v14_general",
        config_class="ModelConfigV14General",
        description="Market-aware model with hybrid line handling",
        is_active=True,
    ),
    
    "v13_general": ModelInfo(
        model_id="v13_general",
        display_name="Model V13 General",
        version="13.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v13_general.py",
        secondary_files=["model_v13_under.py"],
        backtest_function="run_backtest_v13_general",
        picks_function="get_daily_picks_v13_general",
        config_class="ModelConfigV13General",
        description="General prediction model with direction preferences",
        is_active=True,
    ),
    
    "v12_general": ModelInfo(
        model_id="v12_general",
        display_name="Model V12 General",
        version="12.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v12_general.py",
        secondary_files=["model_v12_shared.py", "model_v12_under.py"],
        backtest_function="run_backtest_general",
        picks_function="get_daily_picks_general",
        config_class="GeneralModelConfig",
        description="Pattern-based predictions with sportsbook integration",
        is_active=True,
    ),
    
    "v12_combined": ModelInfo(
        model_id="v12_combined",
        display_name="Model V12 Combined",
        version="12.0",
        category=ModelCategory.MULTI_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
        ],
        primary_file="model_v12_combined.py",
        secondary_files=["model_v12_shared.py"],
        backtest_function="run_backtest_v12",
        picks_function="get_combined_daily_picks_v12",
        description="Combined general + under model",
        is_active=True,
    ),
    
    # -------------------------------------------------------------------------
    # SINGLE FILE MODELS (v2-v10)
    # -------------------------------------------------------------------------
    "v10": ModelInfo(
        model_id="v10",
        display_name="Model V10",
        version="10.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v10.py",
        backtest_function="run_backtest_v10",
        picks_function="get_daily_picks_v10",
        config_class="ModelConfigV10",
        description="Market-aware model requiring actual sportsbook lines",
        is_active=True,
    ),
    
    "v9": ModelInfo(
        model_id="v9",
        display_name="Model V9",
        version="9.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v9.py",
        backtest_function="run_backtest_v9",
        picks_function="get_daily_picks_v9",
        config_class="ModelConfigV9",
        description="Enhanced filtering and direction preferences",
        is_active=True,
    ),
    
    "v8": ModelInfo(
        model_id="v8",
        display_name="Model V8",
        version="8.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.AST_PROPS,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_v8.py",
        backtest_function="run_backtest_v8",
        picks_function="get_daily_picks",
        config_class="ModelV8Config",
        description="Baseline production model with pattern detection",
        is_active=True,
    ),
    
    "v5": ModelInfo(
        model_id="v5",
        display_name="Model V5",
        version="5.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.AST_PROPS,
            ModelCapability.DERIVED_LINES,
        ],
        primary_file="model_v5.py",
        backtest_function="run_backtest_v5",
        picks_function="get_daily_picks",
        config_class="ModelV5Config",
        description="Mid-generation model with basic patterns",
        is_active=True,
    ),
    
    "v4": ModelInfo(
        model_id="v4",
        display_name="Model V4",
        version="4.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.AST_PROPS,
            ModelCapability.DERIVED_LINES,
        ],
        primary_file="model_v4.py",
        backtest_function="run_backtest_v4",
        picks_function="get_daily_picks",
        config_class="ModelV4Config",
        description="Early model with basic projections",
        is_active=True,
    ),
    
    "v3": ModelInfo(
        model_id="v3",
        display_name="Model V3",
        version="3.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.AST_PROPS,
            ModelCapability.DERIVED_LINES,
        ],
        primary_file="model_v3.py",
        backtest_function="run_backtest_v3",
        picks_function="get_daily_picks_v3",
        description="Early generation model",
        is_active=True,
    ),
    
    "v2": ModelInfo(
        model_id="v2",
        display_name="Model V2",
        version="2.0",
        category=ModelCategory.SINGLE_FILE,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.AST_PROPS,
            ModelCapability.DERIVED_LINES,
        ],
        primary_file="model_v2.py",
        backtest_function="run_backtest",
        picks_function="get_daily_picks",
        description="First generation prediction model",
        is_active=True,
    ),
    
    # -------------------------------------------------------------------------
    # SPECIALIZED MODELS
    # -------------------------------------------------------------------------
    "production": ModelInfo(
        model_id="production",
        display_name="Model Production",
        version="1.0",
        category=ModelCategory.SPECIALIZED,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.AST_PROPS,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="model_production.py",
        backtest_function="run_backtest_production",
        picks_function="get_daily_picks",
        config_class="ModelConfig",
        description="Production deployment model with validated patterns",
        is_active=True,
    ),
    
    "final": ModelInfo(
        model_id="final",
        display_name="Model Final",
        version="1.0",
        category=ModelCategory.SPECIALIZED,
        capabilities=[
            ModelCapability.OVER_PICKS,
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.DERIVED_LINES,
        ],
        primary_file="model_final.py",
        backtest_function="run_backtest_final",
        picks_function="get_daily_picks",
        config_class="FinalModelConfig",
        description="Final consolidated model",
        is_active=True,
    ),
    
    "under_v2": ModelInfo(
        model_id="under_v2",
        display_name="Under Model V2",
        version="2.0",
        category=ModelCategory.SPECIALIZED,
        capabilities=[
            ModelCapability.UNDER_PICKS,
            ModelCapability.PTS_PROPS,
            ModelCapability.REB_PROPS,
            ModelCapability.SPORTSBOOK_LINES,
            ModelCapability.DERIVED_LINES,
            ModelCapability.VERBOSE_OUTPUT,
        ],
        primary_file="under_model_v2.py",
        backtest_function="run_backtest_under_v2",
        picks_function="get_under_picks_v2",
        description="Specialized UNDER prediction model",
        is_active=True,
    ),
}


def get_all_models() -> List[ModelInfo]:
    """Get list of all registered models."""
    return list(MODEL_REGISTRY.values())


def get_active_models() -> List[ModelInfo]:
    """Get list of all active models."""
    return [m for m in MODEL_REGISTRY.values() if m.is_active]


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get a specific model by its ID."""
    return MODEL_REGISTRY.get(model_id)


def get_models_by_category(category: ModelCategory) -> List[ModelInfo]:
    """Get all models of a specific category."""
    return [m for m in MODEL_REGISTRY.values() if m.category == category]


def get_models_with_capability(capability: ModelCapability) -> List[ModelInfo]:
    """Get all models that have a specific capability."""
    return [m for m in MODEL_REGISTRY.values() if capability in m.capabilities]


def get_model_count_summary() -> Dict[str, int]:
    """Get summary of model counts by category."""
    summary = {
        "total": len(MODEL_REGISTRY),
        "active": len(get_active_models()),
        "single_file": len(get_models_by_category(ModelCategory.SINGLE_FILE)),
        "multi_file": len(get_models_by_category(ModelCategory.MULTI_FILE)),
        "folder": len(get_models_by_category(ModelCategory.FOLDER_BASED)),
        "specialized": len(get_models_by_category(ModelCategory.SPECIALIZED)),
    }
    return summary


# ============================================================================
# PROGRESS BAR UTILITY
# ============================================================================

def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "█",
    print_end: str = "\r"
) -> None:
    """
    Print a progress bar to the terminal.
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        decimals: Decimal places in percent complete
        length: Character length of bar
        fill: Fill character
        print_end: End character
    """
    if total <= 0:
        return
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        print()


# ============================================================================
# DYNAMIC MODEL LOADER
# ============================================================================

def load_model_module(model_info: ModelInfo):
    """
    Dynamically load a model module.
    
    Args:
        model_info: ModelInfo with file and function information
    
    Returns:
        Loaded module
    """
    # Build module path
    module_name = model_info.primary_file.replace(".py", "")
    full_module = f"src.nba_props.engine.{module_name}"
    
    try:
        return importlib.import_module(full_module)
    except ImportError as e:
        # Try without src prefix
        try:
            full_module = f"nba_props.engine.{module_name}"
            return importlib.import_module(full_module)
        except ImportError:
            raise ImportError(f"Could not load model module {module_name}: {e}")


def get_backtest_function(model_info: ModelInfo) -> Optional[Callable]:
    """
    Get the backtest function for a model.
    
    Args:
        model_info: ModelInfo with function information
    
    Returns:
        Backtest function or None
    """
    if not model_info.backtest_function:
        return None
    
    try:
        module = load_model_module(model_info)
        return getattr(module, model_info.backtest_function, None)
    except Exception as e:
        print(f"Error loading backtest function for {model_info.model_id}: {e}")
        return None


def get_picks_function(model_info: ModelInfo) -> Optional[Callable]:
    """
    Get the daily picks function for a model.
    
    Args:
        model_info: ModelInfo with function information
    
    Returns:
        Picks function or None
    """
    if not model_info.picks_function:
        return None
    
    try:
        module = load_model_module(model_info)
        return getattr(module, model_info.picks_function, None)
    except Exception as e:
        print(f"Error loading picks function for {model_info.model_id}: {e}")
        return None


def get_config_class(model_info: ModelInfo):
    """
    Get the configuration class for a model.
    
    Args:
        model_info: ModelInfo with config information
    
    Returns:
        Config class or None
    """
    if not model_info.config_class:
        return None
    
    try:
        module = load_model_module(model_info)
        return getattr(module, model_info.config_class, None)
    except Exception:
        return None
