"""
Experiment data models for the retention decision studio.

Pydantic/dataclass schemas for experiment configuration, assignments,
exposure and outcome events, and analysis results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class ArmType(str, Enum):
    """Experiment arm type."""
    CONTROL = "control"
    TREATMENT = "treatment"


class MetricType(str, Enum):
    """Metric type for analysis."""
    BINARY = "binary"  # e.g., churn, conversion
    CONTINUOUS = "continuous"  # e.g., revenue, tenure


@dataclass
class MetricDefinition:
    """Definition of a metric used in experiment analysis."""
    name: str
    metric_type: MetricType
    primary: bool = False
    higher_is_better: bool = False  # for churn, lower is better
    description: str = ""


@dataclass
class ExperimentConfig:
    """Configuration for an A/B experiment."""
    experiment_id: str
    name: str
    description: str = ""
    allocation: float = 0.5  # treatment fraction (0.5 = 50/50)
    primary_metric: str = "churn_rate"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    stratification_cols: List[str] = field(default_factory=list)
    min_sample_size: int = 1000


@dataclass
class Assignment:
    """Experiment assignment for a single entity."""
    experiment_id: str
    entity_id: str  # e.g., customerID
    arm: ArmType
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    stratum: Optional[str] = None


@dataclass
class ExposureEvent:
    """Event recording when entity was exposed to experiment arm."""
    experiment_id: str
    entity_id: str
    arm: ArmType
    exposed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeEvent:
    """Event recording experiment outcome for an entity."""
    experiment_id: str
    entity_id: str
    arm: ArmType
    outcome_name: str
    outcome_value: float  # binary: 0/1, continuous: raw value
    observed_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmStats:
    """Statistics for a single experiment arm."""
    arm: ArmType
    n: int
    mean: float
    std: float
    ci_low: float
    ci_high: float


@dataclass
class SegmentResult:
    """Per-segment analysis result."""
    segment_name: str
    segment_value: str
    control_n: int
    treatment_n: int
    control_mean: float
    treatment_mean: float
    lift: float
    lift_pct: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool


@dataclass
class AnalysisResult:
    """Complete experiment analysis result."""
    experiment_id: str
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    primary_metric: str = "churn_rate"
    
    # SRM
    srm_passed: bool = True
    srm_p_value: Optional[float] = None
    expected_allocation: float = 0.5
    actual_control_frac: Optional[float] = None
    actual_treatment_frac: Optional[float] = None
    
    # Main result
    control_stats: Optional[ArmStats] = None
    treatment_stats: Optional[ArmStats] = None
    lift: Optional[float] = None
    lift_pct: Optional[float] = None
    p_value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    effect_size: Optional[float] = None
    
    # CUPED
    cuped_applied: bool = False
    cuped_lift: Optional[float] = None
    cuped_p_value: Optional[float] = None
    variance_reduction_pct: Optional[float] = None
    
    # Segments
    segments: List[SegmentResult] = field(default_factory=list)
    
    # Recommendation
    recommendation: str = "iterate"  # ship, hold, iterate
    recommendation_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            "experiment_id": self.experiment_id,
            "analysis_timestamp": self.analysis_timestamp.isoformat(),
            "primary_metric": self.primary_metric,
            "srm_passed": self.srm_passed,
            "srm_p_value": self.srm_p_value,
            "expected_allocation": self.expected_allocation,
            "actual_control_frac": self.actual_control_frac,
            "actual_treatment_frac": self.actual_treatment_frac,
            "lift": self.lift,
            "lift_pct": self.lift_pct,
            "p_value": self.p_value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "effect_size": self.effect_size,
            "cuped_applied": self.cuped_applied,
            "cuped_lift": self.cuped_lift,
            "cuped_p_value": self.cuped_p_value,
            "variance_reduction_pct": self.variance_reduction_pct,
            "recommendation": self.recommendation,
            "recommendation_reason": self.recommendation_reason,
        }
        if self.control_stats:
            d["control_stats"] = {
                "arm": self.control_stats.arm.value,
                "n": self.control_stats.n,
                "mean": self.control_stats.mean,
                "std": self.control_stats.std,
                "ci_low": self.control_stats.ci_low,
                "ci_high": self.control_stats.ci_high,
            }
        if self.treatment_stats:
            d["treatment_stats"] = {
                "arm": self.treatment_stats.arm.value,
                "n": self.treatment_stats.n,
                "mean": self.treatment_stats.mean,
                "std": self.treatment_stats.std,
                "ci_low": self.treatment_stats.ci_low,
                "ci_high": self.treatment_stats.ci_high,
            }
        d["segments"] = [
            {
                "segment_name": s.segment_name,
                "segment_value": s.segment_value,
                "control_n": s.control_n,
                "treatment_n": s.treatment_n,
                "control_mean": s.control_mean,
                "treatment_mean": s.treatment_mean,
                "lift": s.lift,
                "lift_pct": s.lift_pct,
                "p_value": s.p_value,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
                "significant": s.significant,
            }
            for s in self.segments
        ]
        return d
