"""Experimentation module for A/B testing and retention decision studio."""

from .schema import (
    ExperimentConfig,
    Assignment,
    ExposureEvent,
    OutcomeEvent,
    AnalysisResult,
    ArmType,
)
from .assignment import assign_entity, assign_entities
from .event_store import append_exposures, append_outcomes, read_exposures, read_outcomes
from .analyze import run_analysis, load_experiment_data
from .report import render_exec_summary

__all__ = [
    "ExperimentConfig",
    "Assignment",
    "ExposureEvent",
    "OutcomeEvent",
    "AnalysisResult",
    "ArmType",
    "assign_entity",
    "assign_entities",
    "append_exposures",
    "append_outcomes",
    "read_exposures",
    "read_outcomes",
    "run_analysis",
    "load_experiment_data",
    "render_exec_summary",
]
