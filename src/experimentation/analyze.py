"""
Experiment analysis entrypoint.

Input: experiment_id, start/end, metric (primary = churn_rate or retention_rate), optional CUPED.
Output: AnalysisResult JSON + plots saved to artifacts/experiments/<experiment_id>/
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .schema import (
    AnalysisResult,
    ArmStats,
    ArmType,
    ExperimentConfig,
)
from .event_store import read_exposures, read_outcomes, read_participants
from .stats import (
    check_srm,
    proportions_z_test,
    continuous_t_test,
    build_arm_stats,
    segment_analysis,
    cuped_analysis,
)

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = "artifacts/experiments"
DEFAULT_DATA_DIR = "data/experiments"


def load_experiment_data(
    experiment_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    primary_metric: str = "churn_rate",
    data_dir: str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """
    Load and merge exposures with outcomes for analysis.
    
    Returns:
        DataFrame with columns: entity_id, arm, outcome_value, and segment cols if present
    """
    participants = read_participants(experiment_id, base_dir=data_dir)
    if not participants.empty and "outcome_value" in participants.columns:
        return participants
    
    exposures = read_exposures(experiment_id, start_date, end_date, base_dir=data_dir)
    outcomes = read_outcomes(
        experiment_id,
        outcome_name=primary_metric.replace("_rate", "").replace("churn", "churned"),
        start_date=start_date,
        end_date=end_date,
        base_dir=data_dir,
    )
    
    # Outcome name might be stored differently
    out_names = ["churned", "churn", "churn_rate", "retention", "retained"]
    if outcomes.empty and not exposures.empty:
        outcomes = read_outcomes(experiment_id, start_date=start_date, end_date=end_date, base_dir=data_dir)
    
    if exposures.empty:
        raise ValueError(f"No exposure data for experiment {experiment_id}")
    
    if outcomes.empty:
        return exposures
    
    # Keep one row per entity (last if multiple outcomes)
    out_sub = outcomes.drop_duplicates(subset=["entity_id"], keep="last")
    out_cols = ["entity_id", "outcome_value"]
    if "arm" in out_sub.columns:
        out_cols.append("arm")
    out_sub = out_sub[[c for c in out_cols if c in out_sub.columns]]
    
    merged = exposures.merge(out_sub, on="entity_id", how="inner", suffixes=("", "_y"))
    if "arm_y" in merged.columns:
        merged = merged.drop(columns=["arm_y"])
    if "outcome_value" not in merged.columns and "outcome_value_y" in merged.columns:
        merged["outcome_value"] = merged["outcome_value_y"]
    return merged


def run_analysis(
    experiment_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    primary_metric: str = "churn_rate",
    metric_type: str = "binary",
    use_cuped: bool = False,
    covariate_col: Optional[str] = None,
    segment_cols: Optional[list] = None,
    data_dir: str = DEFAULT_DATA_DIR,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
) -> AnalysisResult:
    """
    Run full experiment analysis.
    
    Args:
        experiment_id: Experiment ID
        start_date: Start of analysis window
        end_date: End of analysis window
        primary_metric: Primary metric name
        metric_type: 'binary' (proportions) or 'continuous'
        use_cuped: Whether to apply CUPED
        covariate_col: Column for CUPED (e.g. tenure, churn_prob)
        segment_cols: Columns for segment breakdown
        data_dir: Base data directory
        artifacts_dir: Base artifacts directory
        
    Returns:
        AnalysisResult
    """
    df = load_experiment_data(
        experiment_id, start_date, end_date, primary_metric, data_dir
    )
    
    arm_col = "arm"
    outcome_col = "outcome_value"
    
    if outcome_col not in df.columns:
        raise ValueError(
            f"Outcome column '{outcome_col}' not found. "
            "Ensure outcomes are logged with outcome_value."
        )
    
    ctrl = df[df[arm_col] == ArmType.CONTROL.value]
    treat = df[df[arm_col] == ArmType.TREATMENT.value]
    
    n_c = len(ctrl)
    n_t = len(treat)
    expected_frac = 0.5
    
    srm_passed, _, srm_p = check_srm(n_c, n_t, expected_frac)
    actual_treat_frac = n_t / (n_c + n_t) if (n_c + n_t) > 0 else 0
    actual_ctrl_frac = n_c / (n_c + n_t) if (n_c + n_t) > 0 else 0
    
    control_stats = build_arm_stats(ArmType.CONTROL, ctrl[outcome_col].values)
    treatment_stats = build_arm_stats(ArmType.TREATMENT, treat[outcome_col].values)
    
    cuped_lift = None
    cuped_p = None
    var_reduction = None
    
    if metric_type == "binary":
        x_c = int(ctrl[outcome_col].sum())
        x_t = int(treat[outcome_col].sum())
        lift, lift_pct, p_val, ci_lo, ci_hi = proportions_z_test(n_c, x_c, n_t, x_t)
        effect_size = (treat[outcome_col].mean() - ctrl[outcome_col].mean()) / (
            np.sqrt(
                (ctrl[outcome_col].var() + treat[outcome_col].var()) / 2
            )
        ) if (ctrl[outcome_col].var() + treat[outcome_col].var()) > 0 else 0
    else:
        lift, lift_pct, p_val, ci_lo, ci_hi, effect_size = continuous_t_test(
            ctrl[outcome_col].values, treat[outcome_col].values
        )
    
    if use_cuped and covariate_col and covariate_col in df.columns:
        x_c_arr = ctrl[covariate_col].fillna(ctrl[covariate_col].median()).values
        x_t_arr = treat[covariate_col].fillna(treat[covariate_col].median()).values
        cuped_lift, cuped_p, var_reduction, _ = cuped_analysis(
            ctrl[outcome_col].values,
            treat[outcome_col].values,
            x_c_arr,
            x_t_arr,
        )
    
    segments = []
    if segment_cols:
        for seg_col in segment_cols:
            if seg_col in df.columns:
                segments.extend(
                    segment_analysis(
                        df, seg_col, arm_col, outcome_col, metric_type
                    )
                )
    
    # Recommendation logic
    if not srm_passed:
        recommendation = "hold"
        reason = "SRM detected: allocation deviates from expected. Do not interpret results."
    elif p_val < 0.05 and lift < 0 and primary_metric == "churn_rate":
        recommendation = "ship"
        reason = "Statistically significant churn reduction. Recommend rollout."
    elif p_val < 0.05 and lift > 0 and primary_metric == "churn_rate":
        recommendation = "hold"
        reason = "Statistically significant churn increase. Do not rollout."
    elif p_val >= 0.05 and abs(lift_pct or 0) < 5:
        recommendation = "iterate"
        reason = "No significant effect. Consider larger sample or different intervention."
    else:
        recommendation = "iterate"
        reason = "Inconclusive. Gather more data or refine hypothesis."
    
    result = AnalysisResult(
        experiment_id=experiment_id,
        start_date=start_date,
        end_date=end_date,
        primary_metric=primary_metric,
        srm_passed=srm_passed,
        srm_p_value=srm_p,
        expected_allocation=expected_frac,
        actual_control_frac=actual_ctrl_frac,
        actual_treatment_frac=actual_treat_frac,
        control_stats=control_stats,
        treatment_stats=treatment_stats,
        lift=lift,
        lift_pct=lift_pct,
        p_value=p_val,
        ci_low=ci_lo,
        ci_high=ci_hi,
        effect_size=effect_size,
        cuped_applied=use_cuped and covariate_col is not None,
        cuped_lift=cuped_lift,
        cuped_p_value=cuped_p,
        variance_reduction_pct=var_reduction,
        segments=segments,
        recommendation=recommendation,
        recommendation_reason=reason,
    )
    
    out_dir = Path(artifacts_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "analysis.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    if segments:
        seg_df = pd.DataFrame([
            {k: getattr(s, k) for k in ["segment_name", "segment_value", "control_n", "treatment_n",
                                         "control_mean", "treatment_mean", "lift", "lift_pct", "p_value"]}
            for s in segments
        ])
        seg_df.to_csv(out_dir / "segments.csv", index=False)
    
    _save_plots(result, segments, out_dir)
    logger.info(f"Analysis saved to {out_dir}")
    return result


def _save_plots(result: AnalysisResult, segments: list, out_dir: Path) -> None:
    """Generate and save analysis plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    out_dir = Path(out_dir)
    
    if result.control_stats and result.treatment_stats:
        fig, ax = plt.subplots(figsize=(6, 4))
        arms = ["Control", "Treatment"]
        means = [result.control_stats.mean, result.treatment_stats.mean]
        errs = [
            (result.control_stats.mean - result.control_stats.ci_low, result.control_stats.ci_high - result.control_stats.mean),
            (result.treatment_stats.mean - result.treatment_stats.ci_low, result.treatment_stats.ci_high - result.treatment_stats.mean),
        ]
        ax.bar(arms, means, color=["#3498db", "#2ecc71"], yerr=[[e[0] for e in errs], [e[1] for e in errs]], capsize=5)
        ax.set_ylabel(result.primary_metric)
        ax.set_title("Experiment: Control vs Treatment")
        plt.tight_layout()
        plt.savefig(out_dir / "lift_chart.png", dpi=100)
        plt.close()
    
    if result.ci_low is not None and result.ci_high is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axhline(0, color="gray", linestyle="--")
        ax.barh(["Lift"], [result.lift or 0], xerr=[[(result.lift or 0) - (result.ci_low or 0)], [(result.ci_high or 0) - (result.lift or 0)]], color="#9b59b6", capsize=5)
        ax.set_xlabel("Lift with 95% CI")
        ax.set_title("Primary Metric Lift")
        plt.tight_layout()
        plt.savefig(out_dir / "lift_ci.png", dpi=100)
        plt.close()
